#!/usr/bin/env python3
"""Expose a local MLX LLM as an OpenAI-compatible label classifier."""

from __future__ import annotations

import argparse
import json
import re
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any


DEFAULT_MODEL = "mlx-community/Qwen3-8B-4bit"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9600)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--model-id", default="local_llm_label_teacher")
    parser.add_argument("--labels", required=True, help="JSON list or mapping of allowed labels")
    parser.add_argument(
        "--task-description",
        default="Classify the user text.",
        help="Short instruction describing the classification task.",
    )
    parser.add_argument("--max-tokens", type=int, default=96)
    args = parser.parse_args()

    classifier = MlxLabelClassifier(
        model_name=args.model,
        model_id=args.model_id,
        labels=load_labels(Path(args.labels)),
        task_description=args.task_description,
        max_tokens=args.max_tokens,
    )
    LocalLlmHandler.classifier = classifier
    server = ThreadingHTTPServer((args.host, args.port), LocalLlmHandler)
    print(
        f"Local MLX LLM label teacher listening on http://{args.host}:{args.port} "
        f"model={args.model} model_id={args.model_id} labels={len(classifier.labels)}"
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        return 130
    return 0


class MlxLabelClassifier:
    def __init__(
        self,
        model_name: str,
        model_id: str,
        labels: list[str],
        task_description: str,
        max_tokens: int,
    ) -> None:
        try:
            from mlx_lm import generate, load
        except ImportError as exc:
            raise SystemExit("mlx-lm is required for local LLM teacher") from exc

        if not labels:
            raise SystemExit("label list is empty")
        self.generate = generate
        self.model, self.tokenizer = load(model_name)
        self.model_name = model_name
        self.model_id = model_id
        self.labels = labels
        self.label_set = set(labels)
        self.task_description = task_description.strip()
        self.max_tokens = max_tokens
        self.lock = threading.Lock()

    def classify(self, text: str) -> tuple[str, str, int]:
        if not text.strip():
            raise ValueError("empty user text")
        started = time.monotonic()
        with self.lock:
            output = self.generate(
                self.model,
                self.tokenizer,
                prompt=self.prompt(text),
                max_tokens=self.max_tokens,
                verbose=False,
            )
        latency_ms = int((time.monotonic() - started) * 1000)
        return normalize_label(output, self.labels), output, latency_ms

    def prompt(self, text: str) -> str:
        labels = "\n".join(f"- {label}" for label in self.labels)
        return (
            f"{self.task_description}\n"
            "Return exactly one label from the allowed list.\n"
            "Do not explain. Do not return JSON.\n"
            "If you think internally, still put the final label plainly.\n"
            "/no_think\n\n"
            f"Allowed labels:\n{labels}\n\n"
            f"Text:\n{text}\n\n"
            "Label:"
        )


class LocalLlmHandler(BaseHTTPRequestHandler):
    server_version = "DistillForgeLocalLlmLabelTeacher/0.1"
    classifier: MlxLabelClassifier

    def do_GET(self) -> None:
        if self.path == "/health":
            self.write_json(
                HTTPStatus.OK,
                {
                    "status": "ok",
                    "model": self.classifier.model_name,
                    "model_id": self.classifier.model_id,
                    "labels": len(self.classifier.labels),
                },
            )
            return
        self.write_json(HTTPStatus.NOT_FOUND, {"error": "not_found"})

    def do_POST(self) -> None:
        if self.path not in {"/v1/chat/completions", "/v1/completions"}:
            self.write_json(HTTPStatus.NOT_FOUND, {"error": "not_found"})
            return
        try:
            payload = self.read_json()
            label, raw_output, latency_ms = self.classifier.classify(request_text(payload))
        except ValueError as exc:
            self.write_json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
            return
        except Exception as exc:
            self.write_json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": str(exc)})
            return

        if self.path == "/v1/completions":
            self.write_json(
                HTTPStatus.OK,
                completion_response(self.classifier, label, raw_output, latency_ms),
            )
        else:
            self.write_json(
                HTTPStatus.OK,
                chat_response(self.classifier, label, raw_output, latency_ms),
            )

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


def load_labels(path: Path) -> list[str]:
    payload = json.loads(path.read_text())
    values = payload.keys() if isinstance(payload, dict) else payload
    return sorted({str(value).strip() for value in values if str(value).strip()})


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


def normalize_label(value: str, labels: list[str]) -> str:
    label_set = set(labels)
    cleaned = re.sub(r"<think>.*?</think>", " ", value.strip(), flags=re.DOTALL)
    candidates = [cleaned, *re.findall(r"[A-Za-z][A-Za-z0-9_/-]+", cleaned)]
    normalized_lookup = {canonicalize(label): label for label in labels}
    for candidate in candidates:
        candidate = candidate.strip().strip("`'\".,:; ")
        if candidate in label_set:
            return candidate
        normalized = canonicalize(candidate)
        if normalized in normalized_lookup:
            return normalized_lookup[normalized]
    normalized_cleaned = canonicalize(cleaned)
    for normalized, label in normalized_lookup.items():
        if normalized and normalized in normalized_cleaned:
            return label
    raise ValueError(f"local LLM returned an unknown label: {value!r}")


def canonicalize(value: str) -> str:
    value = value.strip().lower()
    value = re.sub(r"[^a-z0-9]+", "_", value)
    return re.sub(r"_+", "_", value).strip("_")


def chat_response(
    classifier: MlxLabelClassifier,
    label: str,
    raw_output: str,
    latency_ms: int,
) -> dict[str, Any]:
    return {
        "id": "chatcmpl-distillforge-local-llm-label",
        "object": "chat.completion",
        "created": 0,
        "model": classifier.model_id,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": label},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        "distillforge_local_llm": {
            "raw_output": raw_output,
            "latency_ms": latency_ms,
        },
    }


def completion_response(
    classifier: MlxLabelClassifier,
    label: str,
    raw_output: str,
    latency_ms: int,
) -> dict[str, Any]:
    return {
        "id": "cmpl-distillforge-local-llm-label",
        "object": "text_completion",
        "created": 0,
        "model": classifier.model_id,
        "choices": [{"index": 0, "text": label, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        "distillforge_local_llm": {
            "raw_output": raw_output,
            "latency_ms": latency_ms,
        },
    }


if __name__ == "__main__":
    raise SystemExit(main())
