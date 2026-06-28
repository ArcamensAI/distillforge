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
    parser.add_argument(
        "--student-kind",
        choices=(
            "tfidf",
            "tfidf_mlp",
            "embedding_logistic",
            "embedding_mlp",
            "hybrid_embedding_tfidf",
            "hybrid_embedding_tfidf_mlp",
        ),
        default="tfidf",
        help="Student model family. Default: tfidf",
    )
    parser.add_argument(
        "--embedding-model",
        default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        help="SentenceTransformer model for embedding_logistic students.",
    )
    parser.add_argument(
        "--input-format",
        choices=("raw", "openai_user_content"),
        default="raw",
        help="How to convert dataset input into training text. Default: raw",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Embedding batch size for embedding_logistic. Default: 128",
    )
    parser.add_argument(
        "--max-input-chars",
        type=int,
        default=0,
        help="Truncate prepared input text to this many characters. 0 disables truncation.",
    )
    parser.add_argument(
        "--embedding-max-seq-length",
        type=int,
        default=0,
        help="Set SentenceTransformer max_seq_length. 0 keeps the model default.",
    )
    args = parser.parse_args()

    try:
        import joblib
        import pandas as pd
        from sklearn.dummy import DummyClassifier
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import accuracy_score, f1_score
        from sklearn.neural_network import MLPClassifier
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
    x_train_text = prepare_inputs(x_train, args.input_format, args.max_input_chars)
    x_eval_text = prepare_inputs(x_eval, args.input_format, args.max_input_chars)

    unique_labels = sorted(y_train.unique())
    if args.student_kind in {
        "embedding_logistic",
        "embedding_mlp",
        "hybrid_embedding_tfidf",
        "hybrid_embedding_tfidf_mlp",
    } and len(unique_labels) >= 2 and len(train_df) >= 3:
        model, predictions, model_kind, artifact, runtime, base_model = train_embedding_student(
            x_train_text,
            y_train,
            x_eval_text,
            args.embedding_model,
            args.batch_size,
            args.embedding_max_seq_length,
            args.student_kind,
        )
    elif args.student_kind == "tfidf_mlp" and len(unique_labels) >= 2 and len(train_df) >= 3:
        model = Pipeline(
            [
                ("tfidf", TfidfVectorizer(max_features=20_000, ngram_range=(1, 2), min_df=2)),
                (
                    "classifier",
                    MLPClassifier(
                        hidden_layer_sizes=(256, 128),
                        activation="relu",
                        solver="adam",
                        alpha=1e-4,
                        batch_size=64,
                        learning_rate_init=1e-3,
                        max_iter=80,
                        early_stopping=False,
                        n_iter_no_change=8,
                        random_state=42,
                    ),
                ),
            ]
        )
        model.fit(x_train_text, y_train)
        predictions = model.predict(x_eval_text)
        model_kind = "tfidf_mlp_classifier"
        artifact = "model.joblib"
        runtime = "python_sklearn"
        base_model = model_kind
    else:
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
        model.fit(x_train_text, y_train)
        predictions = model.predict(x_eval_text)
        artifact = "model.joblib"
        runtime = "python_sklearn"
        base_model = model_kind

    accuracy = float(accuracy_score(y_eval, predictions))
    macro_f1 = float(f1_score(y_eval, predictions, average="macro", zero_division=0))

    model_path = model_dir / artifact
    joblib.dump(model, model_path)

    eval_report = {
        "model_id": model_id,
        "task_id": task_id,
        "dataset_id": dataset_id,
        "model_kind": model_kind,
        "input_format": args.input_format,
        "max_input_chars": int(args.max_input_chars),
        "embedding_max_seq_length": int(args.embedding_max_seq_length),
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
        "base_model": base_model,
        "dataset_id": dataset_id,
        "status": "offline_eval",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "artifact": artifact,
        "runtime": runtime,
        "metrics": eval_report["metrics"],
        "input_format": args.input_format,
        "max_input_chars": int(args.max_input_chars),
        "embedding_max_seq_length": int(args.embedding_max_seq_length),
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


def prepare_inputs(values: Any, input_format: str, max_input_chars: int = 0) -> list[str]:
    if input_format == "raw":
        texts = [str(value) for value in values]
    else:
        texts = [extract_openai_user_content(str(value)) for value in values]
    if max_input_chars <= 0:
        return texts
    return [text[:max_input_chars] for text in texts]


def extract_openai_user_content(value: str) -> str:
    try:
        payload = json.loads(value)
    except json.JSONDecodeError:
        return value
    if not isinstance(payload, dict):
        return value
    messages = payload.get("messages")
    if not isinstance(messages, list):
        return value
    parts = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        if message.get("role") != "user":
            continue
        content = message.get("content")
        if isinstance(content, str):
            parts.append(content)
        elif content is not None:
            parts.append(json.dumps(content, sort_keys=True))
    return "\n".join(parts) if parts else value


def train_embedding_student(
    x_train: list[str],
    y_train: Any,
    x_eval: list[str],
    embedding_model: str,
    batch_size: int,
    embedding_max_seq_length: int,
    student_kind: str,
) -> tuple[Any, Any, str, str, str, str]:
    try:
        from sentence_transformers import SentenceTransformer
        from scipy.sparse import csr_matrix, hstack
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.neural_network import MLPClassifier
    except ImportError as exc:
        raise SystemExit(
            f"missing embedding training dependency: {exc}. "
            "Install with: python3 -m pip install -r requirements-training.txt"
        ) from exc

    encoder = SentenceTransformer(embedding_model)
    if embedding_max_seq_length > 0:
        encoder.max_seq_length = embedding_max_seq_length
    train_embeddings = encoder.encode(
        x_train,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    eval_embeddings = encoder.encode(
        x_eval,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    model_kind = "embedding_logistic_regression"
    artifact = "classifier.joblib"
    runtime = "python_sentence_transformers"
    train_features = train_embeddings
    eval_features = eval_embeddings
    vectorizer = None
    if student_kind in {"hybrid_embedding_tfidf", "hybrid_embedding_tfidf_mlp"}:
        vectorizer = TfidfVectorizer(max_features=50_000, ngram_range=(1, 2))
        train_tfidf = vectorizer.fit_transform(x_train)
        eval_tfidf = vectorizer.transform(x_eval)
        train_features = hstack([csr_matrix(train_embeddings), train_tfidf], format="csr")
        eval_features = hstack([csr_matrix(eval_embeddings), eval_tfidf], format="csr")
        model_kind = "hybrid_embedding_tfidf_logistic_regression"
        artifact = "hybrid.joblib"
        runtime = "python_sentence_transformers_hybrid"

    if student_kind in {"embedding_mlp", "hybrid_embedding_tfidf_mlp"}:
        classifier = MLPClassifier(
            hidden_layer_sizes=(256,),
            activation="relu",
            solver="adam",
            alpha=1e-4,
            batch_size=64,
            learning_rate_init=1e-3,
            max_iter=120,
            early_stopping=False,
            n_iter_no_change=10,
            random_state=42,
        )
        model_kind = "embedding_mlp_classifier"
        if vectorizer is not None:
            model_kind = "hybrid_embedding_tfidf_mlp_classifier"
            artifact = "hybrid_mlp.joblib"
            runtime = "python_sentence_transformers_hybrid"
    else:
        classifier = LogisticRegression(max_iter=2000, class_weight="balanced")
    classifier.fit(train_features, y_train)
    predictions = classifier.predict(eval_features)
    model = classifier if vectorizer is None else {"classifier": classifier, "vectorizer": vectorizer}
    return (
        model,
        predictions,
        model_kind,
        artifact,
        runtime,
        embedding_model,
    )


def default_model_id(task_id: str) -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    safe_task_id = "".join(char if char.isalnum() or char in "-_" else "_" for char in task_id)
    return f"student_{safe_task_id}_{timestamp}"


if __name__ == "__main__":
    raise SystemExit(main())
