#!/usr/bin/env python3
"""Expose a local OpenAI-compatible Banking77 teacher using sentence embeddings."""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter, defaultdict
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any


DEFAULT_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9300)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--reference-requests",
        required=True,
        help="JSONL requests with text and expected_intent used as local references.",
    )
    parser.add_argument(
        "--nearest-neighbors",
        type=int,
        default=5,
        help="Number of nearest references used for majority vote. Default: 5",
    )
    args = parser.parse_args()

    classifier = EmbeddingIntentClassifier(
        model_name=args.model,
        reference_path=Path(args.reference_requests),
        nearest_neighbors=args.nearest_neighbors,
    )
    LocalTeacherHandler.classifier = classifier
    server = ThreadingHTTPServer((args.host, args.port), LocalTeacherHandler)
    print(
        f"Local Banking77 teacher listening on http://{args.host}:{args.port} "
        f"model={args.model} references={len(classifier.references)}"
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        return 130
    return 0


class EmbeddingIntentClassifier:
    def __init__(self, model_name: str, reference_path: Path, nearest_neighbors: int) -> None:
        try:
            import numpy as np
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise SystemExit(
                "sentence-transformers and numpy are required for local teacher"
            ) from exc

        self.np = np
        self.model = SentenceTransformer(model_name)
        self.nearest_neighbors = max(1, nearest_neighbors)
        self.references = load_references(reference_path)
        if not self.references:
            raise SystemExit(f"no references found in {reference_path}")

        texts = [row["text"] for row in self.references]
        embeddings = self.model.encode(
            texts,
            batch_size=128,
            normalize_embeddings=True,
            show_progress_bar=True,
        )
        self.reference_embeddings = np.asarray(embeddings)

    def classify(self, text: str) -> tuple[str, float]:
        if not text.strip():
            raise ValueError("empty user text")
        query = self.model.encode(
            [text],
            normalize_embeddings=True,
            show_progress_bar=False,
        )[0]
        scores = self.reference_embeddings @ query
        k = min(self.nearest_neighbors, len(self.references))
        nearest = self.np.argpartition(scores, -k)[-k:]
        votes: Counter[str] = Counter()
        best_score_by_label: dict[str, float] = defaultdict(float)
        for index in nearest:
            label = self.references[int(index)]["expected_intent"]
            score = float(scores[int(index)])
            votes[label] += 1
            best_score_by_label[label] = max(best_score_by_label[label], score)
        label = max(votes, key=lambda item: (votes[item], best_score_by_label[item]))
        return label, best_score_by_label[label]


class LocalTeacherHandler(BaseHTTPRequestHandler):
    server_version = "DistillForgeLocalBankingTeacher/0.1"
    classifier: EmbeddingIntentClassifier

    def do_GET(self) -> None:
        if self.path == "/health":
            self.write_json(
                HTTPStatus.OK,
                {
                    "status": "ok",
                    "model": "local_embedding_banking_teacher",
                    "references": len(self.classifier.references),
                },
            )
            return
        self.write_json(HTTPStatus.NOT_FOUND, {"error": "not_found"})

    def do_POST(self) -> None:
        if self.path not in {"/v1/chat/completions", "/v1/completions"}:
            self.write_json(HTTPStatus.NOT_FOUND, {"error": "not_found"})
            return
        started = time.monotonic()
        try:
            payload = self.read_json()
            label, confidence = self.classifier.classify(request_text(payload))
        except ValueError as exc:
            self.write_json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
            return
        latency_ms = int((time.monotonic() - started) * 1000)
        if self.path == "/v1/completions":
            self.write_json(HTTPStatus.OK, completion_response(label, confidence, latency_ms))
        else:
            self.write_json(HTTPStatus.OK, chat_response(label, confidence, latency_ms))

    def read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("content-length", "0"))
        if length <= 0:
            raise ValueError("missing request body")
        payload = json.loads(self.rfile.read(length))
        if not isinstance(payload, dict):
            raise ValueError("json body must be an object")
        return payload

    def write_json(self, status: int, payload: dict[str, Any]) -> None:
        encoded = json.dumps(payload, sort_keys=True).encode("utf-8")
        self.send_response(status)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def log_message(self, format: str, *args: Any) -> None:
        return


def load_references(path: Path) -> list[dict[str, str]]:
    rows = []
    with path.open() as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            text = str(row.get("text") or "").strip()
            expected_intent = str(row.get("expected_intent") or "").strip()
            if text and expected_intent:
                rows.append({"text": text, "expected_intent": expected_intent})
    return rows


def request_text(payload: dict[str, Any]) -> str:
    if isinstance(payload.get("prompt"), str):
        return payload["prompt"].strip()
    messages = payload.get("messages")
    if isinstance(messages, list):
        for message in reversed(messages):
            if isinstance(message, dict) and message.get("role") == "user":
                content = message.get("content")
                if isinstance(content, str):
                    return content.strip()
    raise ValueError("missing user text")


def chat_response(label: str, confidence: float, latency_ms: int) -> dict[str, Any]:
    return {
        "id": "chatcmpl-distillforge-local-banking-teacher",
        "object": "chat.completion",
        "created": 0,
        "model": "local_embedding_banking_teacher",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": label},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        "distillforge_local_teacher": {
            "confidence": confidence,
            "latency_ms": latency_ms,
        },
    }


def completion_response(label: str, confidence: float, latency_ms: int) -> dict[str, Any]:
    return {
        "id": "cmpl-distillforge-local-banking-teacher",
        "object": "text_completion",
        "created": 0,
        "model": "local_embedding_banking_teacher",
        "choices": [{"index": 0, "text": label, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        "distillforge_local_teacher": {
            "confidence": confidence,
            "latency_ms": latency_ms,
        },
    }


if __name__ == "__main__":
    raise SystemExit(main())
