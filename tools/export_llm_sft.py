#!/usr/bin/env python3
"""Export a DistillForge classification dataset to MLX-LM SFT JSONL."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from train_student import prepare_inputs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, help="DistillForge dataset directory")
    parser.add_argument("--out", required=True, help="Output directory for train/valid/test.jsonl")
    parser.add_argument(
        "--input-format",
        choices=("raw", "openai_user_content"),
        default="raw",
        help="How to convert dataset input into prompt text.",
    )
    parser.add_argument(
        "--max-input-chars",
        type=int,
        default=0,
        help="Truncate prepared input text to this many characters. 0 disables truncation.",
    )
    parser.add_argument(
        "--task-description",
        default="Classify the text into exactly one allowed label.",
        help="Instruction included in every prompt.",
    )
    parser.add_argument(
        "--format",
        choices=("prompt_completion", "messages"),
        default="prompt_completion",
        help="Output JSONL schema. Default: prompt_completion.",
    )
    args = parser.parse_args()

    try:
        import pandas as pd
    except ImportError as exc:
        print(
            f"missing dependency: {exc}. Install with: python3 -m pip install -r requirements-training.txt",
            file=sys.stderr,
        )
        return 2

    dataset_dir = Path(args.dataset)
    output_dir = Path(args.out)
    output_dir.mkdir(parents=True, exist_ok=True)

    frames: dict[str, Any] = {}
    labels = set()
    for split in ("train", "validation", "test"):
        path = dataset_dir / f"{split}.parquet"
        if not path.exists():
            continue
        frame = pd.read_parquet(path)
        frames[split] = frame
        labels.update(str(value) for value in frame["validated_output"].dropna().unique())

    if not frames:
        print(f"no parquet splits found in {dataset_dir}", file=sys.stderr)
        return 1

    sorted_labels = sorted(labels)
    split_names = {"train": "train", "validation": "valid", "test": "test"}
    source_manifest = load_source_manifest(dataset_dir)
    manifest: dict[str, Any] = {
        "source_dataset": str(dataset_dir),
        "input_format": args.input_format,
        "max_input_chars": args.max_input_chars,
        "task_description": args.task_description,
        "labels": sorted_labels,
        "splits": {},
        "format": f"mlx_lm_{args.format}_jsonl",
    }
    if source_manifest:
        manifest["source_logs"] = source_manifest.get("source_logs")
        manifest["source_task_id"] = source_manifest.get("task_id")
        manifest["source_target_field"] = source_manifest.get("filters", {}).get("target_field")
        manifest["source_filters"] = source_manifest.get("filters")

    for source_split, target_split in split_names.items():
        frame = frames.get(source_split)
        if frame is None:
            continue
        path = output_dir / f"{target_split}.jsonl"
        texts = prepare_inputs(
            frame["input"].astype(str),
            args.input_format,
            args.max_input_chars,
        )
        labels_for_rows = frame["validated_output"].astype(str).tolist()
        with path.open("w", encoding="utf-8") as handle:
            for text, label in zip(texts, labels_for_rows):
                prompt = build_prompt(args.task_description, sorted_labels, text)
                record = build_record(prompt, label, args.format)
                handle.write(json.dumps(record, sort_keys=True) + "\n")
        manifest["splits"][target_split] = int(len(frame))

    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


def build_prompt(task_description: str, labels: list[str], text: str) -> str:
    allowed = "\n".join(f"- {label}" for label in labels)
    return (
        f"{task_description.strip()}\n"
        "Return exactly one label from the allowed list. Do not explain.\n\n"
        f"Allowed labels:\n{allowed}\n\n"
        f"Text:\n{text}\n\n"
        "Label:"
    )


def build_record(prompt: str, label: str, output_format: str) -> dict[str, Any]:
    if output_format == "messages":
        return {
            "messages": [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": label},
            ]
        }
    return {"prompt": prompt, "completion": label}


def load_source_manifest(dataset_dir: Path) -> dict[str, Any]:
    path = dataset_dir / "manifest.json"
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


if __name__ == "__main__":
    raise SystemExit(main())
