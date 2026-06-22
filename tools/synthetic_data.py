#!/usr/bin/env python3
"""Augment a DistillForge dataset with local synthetic training examples."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SYNTHETIC_COLUMNS = {
    "is_synthetic": False,
    "synthetic_parent_request_id": None,
    "synthetic_method": None,
    "synthetic_confidence": None,
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, help="Source dataset directory")
    parser.add_argument(
        "--out",
        help="Dataset root directory. Default: same root as source dataset",
    )
    parser.add_argument("--dataset-id", help="Output dataset id. Default: generated id")
    parser.add_argument(
        "--multiplier",
        type=int,
        default=2,
        help="Maximum synthetic examples per real training example. Default: 2",
    )
    parser.add_argument(
        "--max-synthetic-ratio",
        type=float,
        default=0.8,
        help="Maximum synthetic share of the output dataset. Default: 0.8",
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=1.0,
        help="Minimum synthetic confidence to keep. Local templates score 1.0. Default: 1.0",
    )
    parser.add_argument(
        "--require-validation",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Only synthesize rows with validated_output. Default: true",
    )
    args = parser.parse_args()

    try:
        import pandas as pd
    except ImportError:
        print(
            "pandas and pyarrow are required. Install with: python3 -m pip install -r requirements-training.txt",
            file=sys.stderr,
        )
        return 2

    if args.multiplier < 0:
        print("--multiplier must be >= 0", file=sys.stderr)
        return 1
    if not 0 <= args.max_synthetic_ratio < 1:
        print("--max-synthetic-ratio must be >= 0 and < 1", file=sys.stderr)
        return 1

    source_dir = Path(args.dataset)
    manifest_path = source_dir / "manifest.json"
    if not manifest_path.exists():
        print(f"missing dataset manifest: {manifest_path}", file=sys.stderr)
        return 1

    manifest = json.loads(manifest_path.read_text())
    task_id = manifest["task_id"]
    source_dataset_id = manifest["dataset_id"]
    output_dataset_id = args.dataset_id or default_dataset_id(task_id)
    output_root = Path(args.out) if args.out else source_dir.parent.parent
    output_dir = output_root / task_id / output_dataset_id
    output_dir.mkdir(parents=True, exist_ok=True)

    frames = read_splits(pd, source_dir)
    for split, frame in frames.items():
        frames[split] = ensure_synthetic_columns(frame)

    train_df = frames["train"]
    real_train = train_df[~train_df["is_synthetic"].fillna(False)].copy()
    if args.require_validation:
        real_train = real_train[real_train["validated_output"].notna()]
        real_train = real_train[real_train["validated_output"].astype(str).str.len() > 0]

    synthetic_df = build_synthetic_rows(
        pd=pd,
        real_train=real_train,
        multiplier=args.multiplier,
        max_synthetic_ratio=args.max_synthetic_ratio,
        min_confidence=args.min_confidence,
    )
    frames["train"] = pd.concat([train_df, synthetic_df], ignore_index=True)

    for split, frame in frames.items():
        frame.to_parquet(output_dir / f"{split}.parquet", index=False)

    augmented_manifest = build_manifest(
        source_manifest=manifest,
        output_dataset_id=output_dataset_id,
        source_dataset_id=source_dataset_id,
        frames=frames,
        synthetic_samples=len(synthetic_df),
        args=args,
    )
    (output_dir / "manifest.json").write_text(
        json.dumps(augmented_manifest, indent=2, sort_keys=True) + "\n"
    )

    print(f"Wrote synthetic dataset {output_dataset_id} to {output_dir}")
    print(json.dumps(augmented_manifest, indent=2, sort_keys=True))
    return 0


def read_splits(pd: Any, source_dir: Path) -> dict[str, Any]:
    frames = {}
    for split in ("train", "validation", "test"):
        path = source_dir / f"{split}.parquet"
        frames[split] = pd.read_parquet(path) if path.exists() else pd.DataFrame()
    return frames


def ensure_synthetic_columns(frame: Any) -> Any:
    frame = frame.copy()
    for column, default in SYNTHETIC_COLUMNS.items():
        if column not in frame.columns:
            frame[column] = default
    return frame


def build_synthetic_rows(
    pd: Any,
    real_train: Any,
    multiplier: int,
    max_synthetic_ratio: float,
    min_confidence: float,
) -> Any:
    if real_train.empty or multiplier == 0 or max_synthetic_ratio == 0:
        return pd.DataFrame(columns=real_train.columns)
    if min_confidence > 1.0:
        return pd.DataFrame(columns=real_train.columns)

    max_by_multiplier = len(real_train) * multiplier
    max_by_ratio = math.floor((len(real_train) * max_synthetic_ratio) / (1 - max_synthetic_ratio))
    target = max(0, min(max_by_multiplier, max_by_ratio))
    existing_inputs = set(real_train["input"].astype(str))
    rows = []
    created_at = datetime.now(timezone.utc).isoformat()

    for variant_index in range(multiplier):
        for _, row in real_train.iterrows():
            if len(rows) >= target:
                break
            synthetic_input = synthesize_input(str(row["input"]), variant_index)
            if not synthetic_input or synthetic_input in existing_inputs:
                continue
            existing_inputs.add(synthetic_input)

            synthetic = row.copy()
            parent_request_id = row.get("request_id")
            parent_ref = str(parent_request_id or row.get("input_hash") or len(rows))
            synthetic["input"] = synthetic_input
            synthetic["teacher_output"] = row["validated_output"]
            synthetic["validated_output"] = row["validated_output"]
            synthetic["request_id"] = f"{parent_ref}:synthetic:{variant_index}"
            synthetic["input_hash"] = sha256_text(synthetic_input)
            synthetic["timestamp"] = created_at
            synthetic["routing_decision"] = "synthetic"
            synthetic["routing_reason"] = "local_template_augmentation"
            synthetic["estimated_cost_usd"] = 0.0
            synthetic["estimated_teacher_cost_usd"] = 0.0
            synthetic["estimated_savings_usd"] = 0.0
            synthetic["is_synthetic"] = True
            synthetic["synthetic_parent_request_id"] = parent_request_id
            synthetic["synthetic_method"] = f"local_template_{variant_index % len(TEMPLATES)}"
            synthetic["synthetic_confidence"] = 1.0
            rows.append(synthetic)
        if len(rows) >= target:
            break

    return pd.DataFrame(rows, columns=real_train.columns)


TEMPLATES = (
    "Message utilisateur:\n{text}",
    "Demande reformulee sans changer l'intention:\n{text}",
    "{text}\n\nRepondre pour la meme intention.",
    "Contexte: requete client.\n{text}",
)


def synthesize_input(text: str, variant_index: int) -> str:
    normalized = " ".join(text.split())
    if not normalized:
        return ""
    return TEMPLATES[variant_index % len(TEMPLATES)].format(text=normalized)


def build_manifest(
    source_manifest: dict[str, Any],
    output_dataset_id: str,
    source_dataset_id: str,
    frames: dict[str, Any],
    synthetic_samples: int,
    args: argparse.Namespace,
) -> dict[str, Any]:
    samples = sum(len(frame) for frame in frames.values())
    real_samples = samples - synthetic_samples
    return {
        **source_manifest,
        "dataset_id": output_dataset_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_dataset_id": source_dataset_id,
        "samples": int(samples),
        "splits": {split: int(len(frame)) for split, frame in frames.items()},
        "synthetic_data": {
            "enabled": True,
            "method": "local_templates",
            "multiplier": args.multiplier,
            "max_synthetic_ratio": args.max_synthetic_ratio,
            "require_validation": args.require_validation,
            "min_confidence": args.min_confidence,
            "real_samples": int(real_samples),
            "synthetic_samples": int(synthetic_samples),
            "synthetic_ratio": float(synthetic_samples / samples) if samples else 0.0,
            "notes": [
                "labels are copied from validated_output",
                "validation and test splits remain real-only",
                "templates do not introduce names, emails, secrets, or external facts",
            ],
        },
    }


def default_dataset_id(task_id: str) -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    safe_task_id = "".join(char if char.isalnum() or char in "-_" else "_" for char in task_id)
    return f"ds_{safe_task_id}_synthetic_{timestamp}"


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
