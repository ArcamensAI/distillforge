#!/usr/bin/env python3
"""Train a simple DistillForge student model from a versioned dataset."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, help="Dataset directory")
    parser.add_argument(
        "--out",
        default="models",
        help="Model output root directory. Default: models",
    )
    parser.add_argument("--model-id", help="Model id. Default: generated id")
    parser.add_argument(
        "--min-train-samples",
        type=int,
        default=1,
        help="Fail if fewer training samples are available. Default: 1",
    )
    args = parser.parse_args()

    try:
        import joblib
        import pandas as pd
        from sklearn.dummy import DummyClassifier
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import accuracy_score, f1_score
        from sklearn.pipeline import Pipeline
    except ImportError as exc:
        print(
            f"missing training dependency: {exc}. Install with: python3 -m pip install -r requirements-training.txt",
            file=sys.stderr,
        )
        return 2

    dataset_dir = Path(args.dataset)
    manifest_path = dataset_dir / "manifest.json"
    if not manifest_path.exists():
        print(f"missing dataset manifest: {manifest_path}", file=sys.stderr)
        return 1

    manifest = json.loads(manifest_path.read_text())
    task_id = manifest["task_id"]
    dataset_id = manifest["dataset_id"]
    model_id = args.model_id or default_model_id(task_id)
    model_dir = Path(args.out) / task_id / model_id
    model_dir.mkdir(parents=True, exist_ok=True)

    frames = {}
    for split in ("train", "validation", "test"):
        path = dataset_dir / f"{split}.parquet"
        frames[split] = pd.read_parquet(path) if path.exists() else pd.DataFrame()

    train_df = frames["train"]
    if train_df.empty:
        train_df = pd.concat([frame for frame in frames.values() if not frame.empty], ignore_index=True)
    if len(train_df) < args.min_train_samples:
        print(
            f"not enough training samples: {len(train_df)} < {args.min_train_samples}",
            file=sys.stderr,
        )
        return 1

    eval_df = first_non_empty(frames["validation"], frames["test"], train_df)
    x_train = train_df["input"].astype(str)
    y_train = train_df["validated_output"].astype(str)
    x_eval = eval_df["input"].astype(str)
    y_eval = eval_df["validated_output"].astype(str)

    unique_labels = sorted(y_train.unique())
    if len(unique_labels) < 2 or len(train_df) < 3:
        estimator = DummyClassifier(strategy="most_frequent")
        model_kind = "dummy_most_frequent"
    else:
        estimator = LogisticRegression(max_iter=1000, class_weight="balanced")
        model_kind = "tfidf_logistic_regression"

    model = Pipeline(
        [
            ("tfidf", TfidfVectorizer(max_features=50_000, ngram_range=(1, 2))),
            ("classifier", estimator),
        ]
    )
    model.fit(x_train, y_train)
    predictions = model.predict(x_eval)

    accuracy = float(accuracy_score(y_eval, predictions))
    macro_f1 = float(f1_score(y_eval, predictions, average="macro", zero_division=0))

    model_path = model_dir / "model.joblib"
    joblib.dump(model, model_path)

    eval_report = {
        "model_id": model_id,
        "task_id": task_id,
        "dataset_id": dataset_id,
        "model_kind": model_kind,
        "train_samples": int(len(train_df)),
        "eval_samples": int(len(eval_df)),
        "unique_labels": int(len(unique_labels)),
        "metrics": {
            "accuracy": accuracy,
            "macro_f1": macro_f1,
        },
    }
    (model_dir / "eval_report.json").write_text(
        json.dumps(eval_report, indent=2, sort_keys=True) + "\n"
    )

    model_manifest = {
        "model_id": model_id,
        "task_id": task_id,
        "base_model": model_kind,
        "dataset_id": dataset_id,
        "status": "offline_eval",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "artifact": "model.joblib",
        "runtime": "python_sklearn",
        "metrics": eval_report["metrics"],
        "training": {
            "train_samples": int(len(train_df)),
            "eval_samples": int(len(eval_df)),
            "unique_labels": int(len(unique_labels)),
        },
    }
    (model_dir / "manifest.json").write_text(
        json.dumps(model_manifest, indent=2, sort_keys=True) + "\n"
    )

    print(f"Wrote model {model_id} to {model_dir}")
    print(json.dumps(eval_report, indent=2, sort_keys=True))
    return 0


def first_non_empty(*frames: Any) -> Any:
    for frame in frames:
        if not frame.empty:
            return frame
    return frames[-1]


def default_model_id(task_id: str) -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    safe_task_id = "".join(char if char.isalnum() or char in "-_" else "_" for char in task_id)
    return f"student_{safe_task_id}_{timestamp}"


if __name__ == "__main__":
    raise SystemExit(main())
