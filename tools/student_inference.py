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

    model_dir = Path(args.model_dir)
    manifest_path = model_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    artifact = manifest.get("artifact", "model.joblib")
    model_path = model_dir / artifact
    if not model_path.exists():
        print(f"missing model artifact: {model_path}", file=sys.stderr)
        return 1

    try:
        model = load_model(model_path, manifest)
    except ImportError as exc:
        print(
            f"missing inference dependency: {exc}. Install with: python3 -m pip install -r requirements-training.txt",
            file=sys.stderr,
        )
        return 2
    model_id = manifest.get("model_id", model_dir.name)
    input_format = manifest.get("input_format", "raw")

    host, port = parse_listen(args.listen)
    handler = make_handler(model, model_id, input_format)
    server = ThreadingHTTPServer((host, port), handler)
    print(f"DistillForge student inference listening on {host}:{port} model={model_id}")
    server.serve_forever()
    return 0


class SentenceTransformerStudent:
    def __init__(self, encoder: Any, classifier: Any, vectorizer: Any | None = None) -> None:
        self.encoder = encoder
        self.classifier = classifier
        self.vectorizer = vectorizer

    def features(self, values: list[str]) -> Any:
        embeddings = self.encoder.encode(
            values,
            batch_size=32,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        if self.vectorizer is None:
            return embeddings
        from scipy.sparse import csr_matrix, hstack

        tfidf = self.vectorizer.transform(values)
        return hstack([csr_matrix(embeddings), tfidf], format="csr")

    def predict(self, values: list[str]) -> Any:
        return self.classifier.predict(self.features(values))

    def predict_proba(self, values: list[str]) -> Any:
        return self.classifier.predict_proba(self.features(values))

    def predict_with_confidence(self, input_text: str) -> tuple[str, float]:
        features = self.features([input_text])
        prediction = str(self.classifier.predict(features)[0])
        confidence = 1.0
        if hasattr(self.classifier, "predict_proba"):
            probabilities = self.classifier.predict_proba(features)[0]
            confidence = float(max(probabilities))
        return prediction, confidence


def load_model(model_path: Path, manifest: dict[str, Any]) -> Any:
    import joblib

    runtime = manifest.get("runtime", "python_sklearn")
    if runtime in {"python_sentence_transformers", "python_sentence_transformers_hybrid"}:
        from sentence_transformers import SentenceTransformer

        encoder = SentenceTransformer(manifest["base_model"])
        artifact = joblib.load(model_path)
        if runtime == "python_sentence_transformers_hybrid":
            return SentenceTransformerStudent(
                encoder,
                artifact["classifier"],
                artifact["vectorizer"],
            )
        return SentenceTransformerStudent(encoder, artifact)
    return joblib.load(model_path)


def make_handler(model: Any, model_id: str, input_format: str = "raw") -> type[BaseHTTPRequestHandler]:
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
                input_text = extract_chat_input(body, input_format)
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
    if hasattr(model, "predict_with_confidence"):
        return model.predict_with_confidence(input_text)
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


def extract_chat_input(body: Any, input_format: str = "raw") -> str:
    if not isinstance(body, dict):
        return str(body)
    messages = body.get("messages")
    if not isinstance(messages, list):
        return json.dumps(body, sort_keys=True)
    parts = []
    for message in messages:
        if isinstance(message, dict):
            if input_format == "openai_user_content" and message.get("role") != "user":
                continue
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
