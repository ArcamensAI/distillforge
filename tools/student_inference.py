#!/usr/bin/env python3
"""Serve a trained DistillForge sklearn student model over HTTP."""

from __future__ import annotations

import argparse
import json
import sys
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True, help="Directory containing model.joblib")
    parser.add_argument("--listen", default="127.0.0.1:9100", help="Listen address")
    args = parser.parse_args()

    try:
        import joblib
    except ImportError as exc:
        print(
            f"missing inference dependency: {exc}. Install with: python3 -m pip install -r requirements-training.txt",
            file=sys.stderr,
        )
        return 2

    model_dir = Path(args.model_dir)
    model_path = model_dir / "model.joblib"
    manifest_path = model_dir / "manifest.json"
    if not model_path.exists():
        print(f"missing model artifact: {model_path}", file=sys.stderr)
        return 1

    model = joblib.load(model_path)
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    model_id = manifest.get("model_id", model_dir.name)

    host, port = parse_listen(args.listen)
    handler = make_handler(model, model_id)
    server = ThreadingHTTPServer((host, port), handler)
    print(f"DistillForge student inference listening on {host}:{port} model={model_id}")
    server.serve_forever()
    return 0


def make_handler(model: Any, model_id: str) -> type[BaseHTTPRequestHandler]:
    class StudentInferenceHandler(BaseHTTPRequestHandler):
        server_version = "DistillForgeStudent/0.1"

        def do_GET(self) -> None:
            if self.path == "/health":
                self.write_json({"status": "ok", "model_id": model_id})
            else:
                self.write_text(HTTPStatus.NOT_FOUND, "not found\n")

        def do_POST(self) -> None:
            started = time.perf_counter()
            body = self.read_json_body()
            if body is None:
                return

            if self.path == "/infer":
                input_text = extract_infer_input(body)
                output, confidence = predict(model, input_text)
                self.write_json(
                    {
                        "model_id": model_id,
                        "output": output,
                        "confidence": confidence,
                        "latency_ms": int((time.perf_counter() - started) * 1000),
                    }
                )
            elif self.path == "/v1/chat/completions":
                input_text = extract_chat_input(body)
                output, _confidence = predict(model, input_text)
                self.write_json(
                    {
                        "id": f"chatcmpl_student_{int(time.time() * 1000)}",
                        "object": "chat.completion",
                        "model": model_id,
                        "choices": [
                            {
                                "index": 0,
                                "message": {"role": "assistant", "content": output},
                                "finish_reason": "stop",
                            }
                        ],
                    }
                )
            elif self.path == "/v1/completions":
                input_text = extract_completion_input(body)
                output, _confidence = predict(model, input_text)
                self.write_json(
                    {
                        "id": f"cmpl_student_{int(time.time() * 1000)}",
                        "object": "text_completion",
                        "model": model_id,
                        "choices": [
                            {
                                "index": 0,
                                "text": output,
                                "finish_reason": "stop",
                            }
                        ],
                    }
                )
            else:
                self.write_text(HTTPStatus.NOT_FOUND, "not found\n")

        def read_json_body(self) -> Any | None:
            length = int(self.headers.get("content-length", "0"))
            raw = self.rfile.read(length)
            try:
                return json.loads(raw or b"{}")
            except json.JSONDecodeError:
                self.write_text(HTTPStatus.BAD_REQUEST, "invalid json\n")
                return None

        def write_json(self, payload: dict[str, Any]) -> None:
            body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
            self.send_response(HTTPStatus.OK)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def write_text(self, status: HTTPStatus, text: str) -> None:
            body = text.encode("utf-8")
            self.send_response(status)
            self.send_header("content-type", "text/plain; charset=utf-8")
            self.send_header("content-length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, _format: str, *args: Any) -> None:
            return

    return StudentInferenceHandler


def predict(model: Any, input_text: str) -> tuple[str, float]:
    prediction = str(model.predict([input_text])[0])
    confidence = 1.0
    if hasattr(model, "predict_proba"):
        try:
            probabilities = model.predict_proba([input_text])[0]
            confidence = float(max(probabilities))
        except Exception:
            confidence = 1.0
    return prediction, confidence


def extract_infer_input(body: Any) -> str:
    if isinstance(body, dict):
        value = body.get("input") or body.get("prompt") or body.get("text") or body
        return value if isinstance(value, str) else json.dumps(value, sort_keys=True)
    return str(body)


def extract_chat_input(body: Any) -> str:
    if not isinstance(body, dict):
        return str(body)
    messages = body.get("messages")
    if not isinstance(messages, list):
        return json.dumps(body, sort_keys=True)
    parts = []
    for message in messages:
        if isinstance(message, dict):
            content = message.get("content", "")
            if isinstance(content, str):
                parts.append(content)
            else:
                parts.append(json.dumps(content, sort_keys=True))
    return "\n".join(parts)


def extract_completion_input(body: Any) -> str:
    if isinstance(body, dict):
        prompt = body.get("prompt", "")
        if isinstance(prompt, str):
            return prompt
        return json.dumps(prompt, sort_keys=True)
    return str(body)


def parse_listen(value: str) -> tuple[str, int]:
    if ":" not in value:
        raise ValueError("--listen must be formatted as host:port")
    host, port = value.rsplit(":", 1)
    return host, int(port)


if __name__ == "__main__":
    raise SystemExit(main())
