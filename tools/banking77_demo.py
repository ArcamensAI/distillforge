#!/usr/bin/env python3
"""Prepare and exercise the Groq + Banking77 DistillForge demo."""

from __future__ import annotations

import argparse
import csv
import io
import json
import sys
import time
import urllib.error
import urllib.request
from collections import defaultdict
from pathlib import Path
from typing import Any


TASK_ID = "banking_intent_v1"
DEFAULT_ROOT = Path("examples/groq_banking77/data")
TRAIN_URL = "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/master/banking_data/train.csv"
TEST_URL = "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/master/banking_data/test.csv"
LABELS_URL = "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/master/banking_data/categories.json"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare", help="Download and sample Banking77")
    prepare.add_argument("--out", default=str(DEFAULT_ROOT))
    prepare.add_argument("--train-url", default=TRAIN_URL)
    prepare.add_argument("--test-url", default=TEST_URL)
    prepare.add_argument("--labels-url", default=LABELS_URL)
    prepare.add_argument("--train-limit", type=int, default=240)
    prepare.add_argument("--eval-limit", type=int, default=80)

    run_proxy = subparsers.add_parser("run-proxy", help="Send sampled requests through DistillForge")
    run_proxy.add_argument(
        "--requests",
        default=str(DEFAULT_ROOT / "requests" / "train_requests.jsonl"),
    )
    run_proxy.add_argument("--proxy-url", default="http://127.0.0.1:6188")
    run_proxy.add_argument("--out", default=str(DEFAULT_ROOT / "teacher_calls.jsonl"))
    run_proxy.add_argument("--limit", type=int, default=100)
    run_proxy.add_argument("--sleep-ms", type=int, default=250)

    evaluate = subparsers.add_parser("evaluate-calls", help="Compare teacher calls to Banking77 labels")
    evaluate.add_argument("--calls", default=str(DEFAULT_ROOT / "teacher_calls.jsonl"))

    budget = subparsers.add_parser("estimate-budget", help="Estimate prompt budget for a request file")
    budget.add_argument(
        "--requests",
        default=str(DEFAULT_ROOT / "requests" / "train_requests.jsonl"),
    )
    budget.add_argument("--limit", type=int, default=100)
    budget.add_argument("--intents", default=str(DEFAULT_ROOT / "intents.json"))
    budget.add_argument(
        "--token-budget",
        type=int,
        default=200_000,
        help="Token budget used to estimate max request count. Default: 200000",
    )

    args = parser.parse_args()
    if args.command == "prepare":
        return prepare_dataset(args)
    if args.command == "run-proxy":
        return run_proxy_requests(args)
    if args.command == "evaluate-calls":
        return evaluate_calls(args)
    if args.command == "estimate-budget":
        return estimate_budget(args)
    return 2


def prepare_dataset(args: argparse.Namespace) -> int:
    root = Path(args.out)
    raw_dir = root / "raw"
    requests_dir = root / "requests"
    raw_dir.mkdir(parents=True, exist_ok=True)
    requests_dir.mkdir(parents=True, exist_ok=True)

    train_csv = read_resource(args.train_url)
    test_csv = read_resource(args.test_url)
    labels_text = read_resource(args.labels_url)
    (raw_dir / "train.csv").write_text(train_csv)
    (raw_dir / "test.csv").write_text(test_csv)
    (raw_dir / "categories.json").write_text(labels_text)

    train_rows = parse_rows(train_csv, "train")
    test_rows = parse_rows(test_csv, "test")
    intents = load_intents(labels_text, train_rows + test_rows)

    train_sample = balanced_sample(train_rows, args.train_limit)
    eval_sample = balanced_sample(test_rows, args.eval_limit)
    write_requests(requests_dir / "train_requests.jsonl", train_sample)
    write_requests(requests_dir / "eval_requests.jsonl", eval_sample)
    (root / "intents.json").write_text(json.dumps(intents, indent=2, sort_keys=True) + "\n")

    manifest = {
        "task_id": TASK_ID,
        "dataset": "BANKING77",
        "source": "PolyAI-LDN/task-specific-datasets",
        "source_urls": {
            "train": TRAIN_URL,
            "test": TEST_URL,
            "labels": LABELS_URL,
        },
        "license": "CC-BY-4.0",
        "train_rows": len(train_rows),
        "test_rows": len(test_rows),
        "intents": len(intents),
        "sampled_train_requests": len(train_sample),
        "sampled_eval_requests": len(eval_sample),
    }
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


def run_proxy_requests(args: argparse.Namespace) -> int:
    requests = list(read_jsonl(Path(args.requests)))
    if args.limit > 0:
        requests = requests[: args.limit]
    output_path = Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    success = 0
    with output_path.open("w") as handle:
        for index, row in enumerate(requests, start=1):
            result = call_proxy(args.proxy_url, row)
            handle.write(json.dumps(result, sort_keys=True) + "\n")
            if result["http_status"] < 400:
                success += 1
            if args.sleep_ms > 0 and index < len(requests):
                time.sleep(args.sleep_ms / 1000)
    print(json.dumps({"requests": len(requests), "success": success, "out": str(output_path)}, indent=2))
    return 0 if success == len(requests) else 1


def evaluate_calls(args: argparse.Namespace) -> int:
    total = 0
    correct = 0
    invalid = 0
    for row in read_jsonl(Path(args.calls)):
        total += 1
        expected = row.get("expected_intent")
        predicted = extract_label(row.get("response_body"))
        if not predicted:
            invalid += 1
        elif predicted == expected:
            correct += 1
    accuracy = correct / total if total else 0.0
    print(
        json.dumps(
            {"calls": total, "correct": correct, "invalid": invalid, "accuracy": accuracy},
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def estimate_budget(args: argparse.Namespace) -> int:
    requests = list(read_jsonl(Path(args.requests)))
    if args.limit > 0:
        requests = requests[: args.limit]
    intents = json.loads(Path(args.intents).read_text())
    prompt_chars = sum(len(row["text"]) for row in requests)
    intent_chars = len("\n".join(f"- {intent}" for intent in intents)) * len(requests)
    estimated_tokens = (prompt_chars + intent_chars) // 4
    estimated_tokens_per_request = estimated_tokens / len(requests) if requests else 0
    estimated_requests_for_budget = (
        int(args.token_budget / estimated_tokens_per_request)
        if estimated_tokens_per_request > 0
        else 0
    )
    print(
        json.dumps(
            {
                "requests": len(requests),
                "intents": len(intents),
                "estimated_prompt_tokens": estimated_tokens,
                "estimated_prompt_tokens_per_request": round(
                    estimated_tokens_per_request, 2
                ),
                "token_budget": args.token_budget,
                "estimated_requests_for_budget": estimated_requests_for_budget,
                "note": "rough chars/4 estimate; Groq account limits remain authoritative",
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def read_resource(value: str) -> str:
    path = Path(value)
    if path.exists():
        return path.read_text()
    with urllib.request.urlopen(value, timeout=60) as response:
        return response.read().decode("utf-8")


def parse_rows(csv_text: str, split: str) -> list[dict[str, str]]:
    reader = csv.DictReader(io.StringIO(csv_text))
    rows = []
    for index, row in enumerate(reader):
        text = (row.get("text") or "").strip()
        category = (row.get("category") or "").strip()
        if text and category:
            rows.append(
                {
                    "request_id": f"banking77_{split}_{index}",
                    "split": split,
                    "text": text,
                    "expected_intent": category,
                }
            )
    return rows


def load_intents(labels_text: str, rows: list[dict[str, str]]) -> list[str]:
    try:
        payload = json.loads(labels_text)
    except json.JSONDecodeError:
        payload = None
    if isinstance(payload, dict):
        values = payload.values()
    elif isinstance(payload, list):
        values = payload
    else:
        values = [row["expected_intent"] for row in rows]
    return sorted({str(value).strip() for value in values if str(value).strip()})


def balanced_sample(rows: list[dict[str, str]], limit: int) -> list[dict[str, str]]:
    if limit <= 0 or limit >= len(rows):
        return rows
    by_intent: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_intent[row["expected_intent"]].append(row)
    sample = []
    intents = sorted(by_intent)
    while len(sample) < limit:
        progressed = False
        for intent in intents:
            if by_intent[intent]:
                sample.append(by_intent[intent].pop(0))
                progressed = True
                if len(sample) >= limit:
                    break
        if not progressed:
            break
    return sample


def write_requests(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open() as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def call_proxy(proxy_url: str, row: dict[str, Any]) -> dict[str, Any]:
    payload = {
        "model": "teacher",
        "temperature": 0,
        "messages": [{"role": "user", "content": row["text"]}],
    }
    request = urllib.request.Request(
        proxy_url.rstrip("/") + "/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "content-type": "application/json",
            "x-client-id": "banking_demo",
            "x-task-id": TASK_ID,
            "x-request-id": row["request_id"],
            "x-cost-center": "demo",
        },
        method="POST",
    )
    started = time.monotonic()
    try:
        with urllib.request.urlopen(request, timeout=90) as response:
            status = response.status
            body = response.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        status = exc.code
        body = exc.read().decode("utf-8", errors="replace")
    return {
        **row,
        "http_status": status,
        "latency_ms": int((time.monotonic() - started) * 1000),
        "response_body": body,
        "predicted_intent": extract_label(body),
    }


def extract_label(body: Any) -> str | None:
    if not isinstance(body, str) or not body.strip():
        return None
    try:
        payload = json.loads(body)
    except json.JSONDecodeError:
        return None
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        return None
    first = choices[0]
    if not isinstance(first, dict):
        return None
    message = first.get("message")
    if isinstance(message, dict) and isinstance(message.get("content"), str):
        return message["content"].strip()
    if isinstance(first.get("text"), str):
        return first["text"].strip()
    return None


if __name__ == "__main__":
    raise SystemExit(main())
