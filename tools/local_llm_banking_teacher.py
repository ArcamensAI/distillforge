#!/usr/bin/env python3
"""Expose a local MLX LLM as an OpenAI-compatible Banking77 classifier."""

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


DEFAULT_MASTER_MODEL = "mlx-community/Qwen3-8B-4bit"
DEFAULT_STUDENT_MODEL = "mlx-community/Qwen2.5-1.5B-Instruct-4bit"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9400)
    parser.add_argument("--model", default=DEFAULT_MASTER_MODEL)
    parser.add_argument("--model-id", default="local_llm_banking_teacher")
    parser.add_argument("--intents", required=True, help="JSON file with Banking77 intents")
    parser.add_argument("--max-tokens", type=int, default=96)
    args = parser.parse_args()

    classifier = MlxIntentClassifier(
        model_name=args.model,
        model_id=args.model_id,
        intents=load_intents(Path(args.intents)),
        max_tokens=args.max_tokens,
    )
    LocalLlmHandler.classifier = classifier
    server = ThreadingHTTPServer((args.host, args.port), LocalLlmHandler)
    print(
        f"Local MLX LLM teacher listening on http://{args.host}:{args.port} "
        f"model={args.model} model_id={args.model_id} intents={len(classifier.intents)}"
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        return 130
    return 0


class MlxIntentClassifier:
    def __init__(
        self,
        model_name: str,
        model_id: str,
        intents: list[str],
        max_tokens: int,
    ) -> None:
        try:
            from mlx_lm import generate, load
        except ImportError as exc:
            raise SystemExit("mlx-lm is required for local LLM teacher") from exc

        if not intents:
            raise SystemExit("intent list is empty")
        self.generate = generate
        self.model, self.tokenizer = load(model_name)
        self.model_name = model_name
        self.model_id = model_id
        self.intents = intents
        self.intent_set = set(intents)
        self.max_tokens = max_tokens
        self.lock = threading.Lock()

    def classify(self, text: str) -> tuple[str, str, int]:
        if not text.strip():
            raise ValueError("empty user text")
        prompt = self.prompt(text)
        started = time.monotonic()
        with self.lock:
            output = self.generate(
                self.model,
                self.tokenizer,
                prompt=prompt,
                max_tokens=self.max_tokens,
                verbose=False,
            )
        latency_ms = int((time.monotonic() - started) * 1000)
        return normalize_label(output, self.intents), output, latency_ms

    def prompt(self, text: str) -> str:
        labels = "\n".join(f"- {intent}" for intent in self.intents)
        return (
            "You classify online banking support queries.\n"
            "Return exactly one label from the allowed list.\n"
            "Do not explain.\n"
            "If you think internally, still put the final label plainly.\n"
            "/no_think\n\n"
            f"Allowed labels:\n{labels}\n\n"
            f"Query:\n{text}\n\n"
            "Label:"
        )


class LocalLlmHandler(BaseHTTPRequestHandler):
    server_version = "DistillForgeLocalLlmBankingTeacher/0.1"
    classifier: MlxIntentClassifier

    def do_GET(self) -> None:
        if self.path == "/health":
            self.write_json(
                HTTPStatus.OK,
                {
                    "status": "ok",
                    "model": self.classifier.model_name,
                    "model_id": self.classifier.model_id,
                    "intents": len(self.classifier.intents),
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
            self.write_json(HTTPStatus.OK, completion_response(self.classifier, label, raw_output, latency_ms))
        else:
            self.write_json(HTTPStatus.OK, chat_response(self.classifier, label, raw_output, latency_ms))

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


def load_intents(path: Path) -> list[str]:
    payload = json.loads(path.read_text())
    values = payload.values() if isinstance(payload, dict) else payload
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


def normalize_label(value: str, intents: list[str]) -> str:
    aliases = {
        "get_virtual_card": "getting_virtual_card",
        "top_up_by_card": "topping_up_by_card",
    }
    intent_set = set(intents)
    cleaned = value.strip().strip("`'\" ")
    cleaned = re.sub(r"<think>.*?</think>", " ", cleaned, flags=re.DOTALL)
    candidates = [cleaned, *re.findall(r"[A-Za-z][A-Za-z0-9_?]+", cleaned)]
    for candidate in candidates:
        candidate = candidate.strip().strip("`'\".,:; ")
        if candidate in aliases and aliases[candidate] in intent_set:
            return aliases[candidate]
        if candidate in intent_set:
            return candidate
    raise ValueError(f"local LLM returned an unknown intent: {value!r}")


def chat_response(
    classifier: MlxIntentClassifier,
    label: str,
    raw_output: str,
    latency_ms: int,
) -> dict[str, Any]:
    return {
        "id": "chatcmpl-distillforge-local-llm-banking",
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
    classifier: MlxIntentClassifier,
    label: str,
    raw_output: str,
    latency_ms: int,
) -> dict[str, Any]:
    return {
        "id": "cmpl-distillforge-local-llm-banking",
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
