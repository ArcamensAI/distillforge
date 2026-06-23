#!/usr/bin/env python3
"""Expose a local OpenAI-compatible teacher backed by Groq."""

from __future__ import annotations

import argparse
import json
import os
import re
import time
import urllib.error
import urllib.request
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any


DEFAULT_GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9200)
    parser.add_argument(
        "--model",
        default=os.environ.get("GROQ_MODEL", "openai/gpt-oss-20b"),
        help="Groq model id. Default: openai/gpt-oss-20b",
    )
    parser.add_argument(
        "--intents",
        required=True,
        help="JSON file containing Banking77 intent names.",
    )
    parser.add_argument("--timeout", type=float, default=30.0)
    args = parser.parse_args()

    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise SystemExit("GROQ_API_KEY is required")

    TeacherHandler.api_key = api_key
    TeacherHandler.model = args.model
    TeacherHandler.intents = load_intents(Path(args.intents))
    TeacherHandler.timeout = args.timeout

    server = ThreadingHTTPServer((args.host, args.port), TeacherHandler)
    print(
        f"Groq teacher listening on http://{args.host}:{args.port} "
        f"model={args.model} intents={len(TeacherHandler.intents)}"
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        return 130
    return 0


class TeacherHandler(BaseHTTPRequestHandler):
    server_version = "DistillForgeGroqTeacher/0.1"
    api_key = ""
    model = "openai/gpt-oss-20b"
    intents: list[str] = []
    timeout = 30.0

    def do_GET(self) -> None:
        if self.path == "/health":
            self.write_json(HTTPStatus.OK, {"status": "ok", "model": self.model})
            return
        self.write_json(HTTPStatus.NOT_FOUND, {"error": "not_found"})

    def do_POST(self) -> None:
        if self.path not in {"/v1/chat/completions", "/v1/completions"}:
            self.write_json(HTTPStatus.NOT_FOUND, {"error": "not_found"})
            return

        try:
            payload = self.read_json()
            text = request_text(payload)
            label = classify_with_groq(
                api_key=self.api_key,
                model=self.model,
                intents=self.intents,
                text=text,
                timeout=self.timeout,
            )
        except ValueError as exc:
            self.write_json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
            return
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            self.write_json(exc.code, {"error": "groq_http_error", "body": body})
            return
        except Exception as exc:
            self.write_json(
                HTTPStatus.BAD_GATEWAY,
                {"error": "groq_teacher_failed", "detail": str(exc)},
            )
            return

        if self.path == "/v1/completions":
            self.write_json(HTTPStatus.OK, completion_response(label))
        else:
            self.write_json(HTTPStatus.OK, chat_response(label))

    def read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("content-length", "0"))
        if length <= 0:
            raise ValueError("missing request body")
        body = self.rfile.read(length)
        try:
            payload = json.loads(body)
        except json.JSONDecodeError as exc:
            raise ValueError("invalid json") from exc
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
    if isinstance(payload, dict):
        values = payload.values()
    elif isinstance(payload, list):
        values = payload
    else:
        raise SystemExit(f"unsupported intents file: {path}")
    intents = sorted({str(value).strip() for value in values if str(value).strip()})
    if not intents:
        raise SystemExit(f"no intents found in {path}")
    return intents


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


def classify_with_groq(
    api_key: str,
    model: str,
    intents: list[str],
    text: str,
    timeout: float,
) -> str:
    if not text:
        raise ValueError("empty user text")

    payload = {
        "model": model,
        "temperature": 0,
        "max_completion_tokens": 512,
        "reasoning_format": "hidden",
        "reasoning_effort": "low",
        "messages": [
            {
                "role": "system",
                "content": (
                    "You classify online banking support queries. "
                    "Return exactly one intent label from the allowed list. "
                    "Do not explain your answer."
                ),
            },
            {
                "role": "user",
                "content": (
                    "Allowed intents:\n"
                    + "\n".join(f"- {intent}" for intent in intents)
                    + "\n\nQuery:\n"
                    + text
                ),
            },
        ],
    }
    request = urllib.request.Request(
        DEFAULT_GROQ_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "authorization": f"Bearer {api_key}",
            "content-type": "application/json",
            "accept": "application/json",
            "user-agent": "distillforge-groq-teacher/0.1",
        },
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        data = json.loads(response.read())
    raw_label = data["choices"][0]["message"]["content"]
    return normalize_label(raw_label, intents)


def normalize_label(value: str, intents: list[str]) -> str:
    stripped = value.strip().strip("`'\" ")
    aliases = {
        "get_virtual_card": "getting_virtual_card",
        "top_up_by_card": "topping_up_by_card",
    }
    intent_set = set(intents)
    if stripped in aliases and aliases[stripped] in intent_set:
        return aliases[stripped]
    if stripped in intent_set:
        return stripped

    candidates = re.findall(r"[a-zA-Z][a-zA-Z0-9_?]+", stripped)
    for candidate in candidates:
        if candidate in aliases and aliases[candidate] in intent_set:
            return aliases[candidate]
        if candidate in intent_set:
            return candidate
    raise ValueError(f"Groq returned an unknown intent: {value!r}")


def chat_response(label: str) -> dict[str, Any]:
    return {
        "id": "chatcmpl-distillforge-groq-teacher",
        "object": "chat.completion",
        "created": 0,
        "model": "distillforge-groq-teacher",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": label},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }


def completion_response(label: str) -> dict[str, Any]:
    return {
        "id": "cmpl-distillforge-groq-teacher",
        "object": "text_completion",
        "created": 0,
        "model": "distillforge-groq-teacher",
        "choices": [{"index": 0, "text": label, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }


if __name__ == "__main__":
    raise SystemExit(main())
