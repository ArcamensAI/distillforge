#!/usr/bin/env python3
"""Prepare and exercise the CFPB complaints DistillForge demo."""

from __future__ import annotations

import argparse
import csv
import io
import json
import re
import sys
import time
import urllib.error
import urllib.request
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


TASK_ID = "cfpb_product_triage_v1"
CLIENT_ID = "cfpb_demo"
DEFAULT_ROOT = Path("examples/cfpb_complaints/data_llm")
CFPB_ZIP_URL = "https://files.consumerfinance.gov/ccdb/complaints.csv.zip"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare", help="Prepare CFPB complaint requests")
    prepare.add_argument("--out", default=str(DEFAULT_ROOT))
    prepare.add_argument(
        "--source",
        help="Local complaints CSV or complaints.csv.zip. If omitted, downloads the CFPB ZIP.",
    )
    prepare.add_argument("--download-url", default=CFPB_ZIP_URL)
    prepare.add_argument("--train-limit", type=int, default=500)
    prepare.add_argument("--eval-limit", type=int, default=100)
    prepare.add_argument("--max-source-rows", type=int, default=200_000)
    prepare.add_argument("--min-narrative-chars", type=int, default=80)
    prepare.add_argument(
        "--allowed-labels",
        help="Optional JSON label list or manifest containing labels. Rows outside this taxonomy are skipped.",
    )

    run_proxy = subparsers.add_parser("run-proxy", help="Send prepared requests through DistillForge")
    run_proxy.add_argument(
        "--requests",
        default=str(DEFAULT_ROOT / "requests" / "train_requests.jsonl"),
    )
    run_proxy.add_argument("--proxy-url", default="http://127.0.0.1:6188")
    run_proxy.add_argument("--out", default=str(DEFAULT_ROOT / "teacher_calls.jsonl"))
    run_proxy.add_argument("--limit", type=int, default=100)
    run_proxy.add_argument("--sleep-ms", type=int, default=0)
    run_proxy.add_argument(
        "--skip-existing-log",
        help="Skip request ids already present as successful calls in this DistillForge proxy log.",
    )

    evaluate = subparsers.add_parser("evaluate-calls", help="Compare calls to CFPB product labels")
    evaluate.add_argument("--calls", default=str(DEFAULT_ROOT / "teacher_calls.jsonl"))

    args = parser.parse_args()
    if args.command == "prepare":
        return prepare_dataset(args)
    if args.command == "run-proxy":
        return run_proxy_requests(args)
    if args.command == "evaluate-calls":
        return evaluate_calls(args)
    return 2


def prepare_dataset(args: argparse.Namespace) -> int:
    root = Path(args.out)
    raw_dir = root / "raw"
    requests_dir = root / "requests"
    raw_dir.mkdir(parents=True, exist_ok=True)
    requests_dir.mkdir(parents=True, exist_ok=True)

    source = Path(args.source) if args.source else download_source(raw_dir, args.download_url)
    rows = list(
        read_complaints(
            source,
            max_rows=args.max_source_rows,
            min_narrative_chars=args.min_narrative_chars,
        )
    )
    if not rows:
        print(f"no eligible CFPB complaints found in {source}", file=sys.stderr)
        return 1

    allowed_labels = load_allowed_labels(Path(args.allowed_labels)) if args.allowed_labels else None
    if allowed_labels:
        rows = [row for row in rows if slugify(row["product"]) in allowed_labels]
        if not rows:
            print(f"no rows match allowed labels from {args.allowed_labels}", file=sys.stderr)
            return 1

    label_map = build_label_map(rows)
    for row in rows:
        row["expected_intent"] = label_map[row["product"]]
        row["text"] = build_prompt_text(row)

    train_sample = balanced_sample(rows, args.train_limit)
    for row in train_sample:
        row["split"] = "train"
    eval_pool_ids = {row["request_id"] for row in train_sample}
    eval_rows = [row for row in rows if row["request_id"] not in eval_pool_ids]
    eval_sample = balanced_sample(eval_rows, args.eval_limit)
    for row in eval_sample:
        row["split"] = "test"

    write_requests(requests_dir / "train_requests.jsonl", train_sample)
    write_requests(requests_dir / "eval_requests.jsonl", eval_sample)
    labels = sorted(set(label_map.values()))
    (root / "labels.json").write_text(json.dumps(labels, indent=2, sort_keys=True) + "\n")
    (root / "product_labels.json").write_text(
        json.dumps({value: key for key, value in sorted(label_map.items())}, indent=2, sort_keys=True)
        + "\n"
    )

    manifest = {
        "task_id": TASK_ID,
        "dataset": "CFPB Consumer Complaint Database",
        "source": str(source),
        "source_url": args.download_url,
        "source_rows_scan_limit": args.max_source_rows,
        "eligible_rows": len(rows),
        "labels": len(labels),
        "allowed_labels": sorted(allowed_labels) if allowed_labels else None,
        "sampled_train_requests": len(train_sample),
        "sampled_eval_requests": len(eval_sample),
        "target": "CFPB product label slug",
    }
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


def download_source(raw_dir: Path, url: str) -> Path:
    output = raw_dir / "complaints.csv.zip"
    if output.exists():
        return output
    print(f"Downloading CFPB complaints from {url} to {output}", file=sys.stderr)
    with urllib.request.urlopen(url, timeout=120) as response:
        with output.open("wb") as handle:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                handle.write(chunk)
    return output


def read_complaints(path: Path, max_rows: int, min_narrative_chars: int) -> Iterable[dict[str, str]]:
    with complaint_csv_text(path) as text:
        reader = csv.DictReader(text)
        for index, row in enumerate(reader):
            if max_rows > 0 and index >= max_rows:
                break
            narrative = (row.get("Consumer complaint narrative") or "").strip()
            product = (row.get("Product") or "").strip()
            if len(narrative) < min_narrative_chars or not product:
                continue
            complaint_id = (row.get("Complaint ID") or "").strip() or str(index)
            yield {
                "request_id": f"cfpb_{complaint_id}",
                "split": "train",
                "complaint_id": complaint_id,
                "product": product,
                "issue": (row.get("Issue") or "").strip(),
                "sub_product": (row.get("Sub-product") or "").strip(),
                "date_received": (row.get("Date received") or "").strip(),
                "company_response": (row.get("Company response to consumer") or "").strip(),
                "narrative": narrative,
            }


class complaint_csv_text:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.handle: Any | None = None
        self.wrapper: io.TextIOWrapper | None = None
        self.zip_file: zipfile.ZipFile | None = None

    def __enter__(self) -> io.TextIOBase:
        if self.path.suffix == ".zip":
            self.zip_file = zipfile.ZipFile(self.path)
            csv_name = next(name for name in self.zip_file.namelist() if name.endswith(".csv"))
            self.handle = self.zip_file.open(csv_name)
            self.wrapper = io.TextIOWrapper(self.handle, encoding="utf-8", errors="replace", newline="")
            return self.wrapper
        self.handle = self.path.open("r", encoding="utf-8", errors="replace", newline="")
        return self.handle

    def __exit__(self, *_exc: Any) -> None:
        if self.wrapper is not None:
            self.wrapper.close()
        elif self.handle is not None:
            self.handle.close()
        if self.zip_file is not None:
            self.zip_file.close()


def build_label_map(rows: list[dict[str, str]]) -> dict[str, str]:
    products = sorted({row["product"] for row in rows})
    slugs = {}
    used = set()
    for product in products:
        slug = slugify(product)
        base = slug
        suffix = 2
        while slug in used:
            slug = f"{base}_{suffix}"
            suffix += 1
        slugs[product] = slug
        used.add(slug)
    return slugs


def build_prompt_text(row: dict[str, str]) -> str:
    parts = [
        "Consumer complaint narrative:",
        row["narrative"],
    ]
    if row.get("issue"):
        parts.extend(["", f"Reported issue: {row['issue']}"])
    if row.get("sub_product"):
        parts.append(f"Reported sub-product: {row['sub_product']}")
    return "\n".join(parts)


def balanced_sample(rows: list[dict[str, str]], limit: int) -> list[dict[str, str]]:
    if limit <= 0 or limit >= len(rows):
        return rows
    by_label: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_label[row["expected_intent"]].append(row)
    sample = []
    labels = sorted(by_label)
    while len(sample) < limit:
        progressed = False
        for label in labels:
            if by_label[label]:
                row = dict(by_label[label].pop(0))
                sample.append(row)
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


def load_allowed_labels(path: Path) -> set[str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        values = payload.get("labels")
        if not isinstance(values, list):
            raise ValueError(f"{path} does not contain a labels list")
    elif isinstance(payload, list):
        values = payload
    else:
        raise ValueError(f"{path} must contain a JSON list or object")
    labels = {str(value).strip() for value in values if str(value).strip()}
    if not labels:
        raise ValueError(f"{path} contains no labels")
    return labels


def run_proxy_requests(args: argparse.Namespace) -> int:
    requests = list(read_jsonl(Path(args.requests)))
    if args.skip_existing_log:
        existing = successful_request_ids(Path(args.skip_existing_log))
        requests = [row for row in requests if row.get("request_id") not in existing]
    if args.limit > 0:
        requests = requests[: args.limit]
    output_path = Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    success = 0
    with output_path.open("w") as handle:
        for index, row in enumerate(requests, start=1):
            result = call_proxy(args.proxy_url, row)
            handle.write(json.dumps(result, sort_keys=True) + "\n")
            handle.flush()
            if result["http_status"] < 400:
                success += 1
            if args.sleep_ms > 0 and index < len(requests):
                time.sleep(args.sleep_ms / 1000)
    print(json.dumps({"requests": len(requests), "success": success, "out": str(output_path)}, indent=2))
    return 0 if success == len(requests) else 1


def successful_request_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    request_ids = set()
    for row in read_jsonl(path):
        proxy_success = row.get("status") == "success"
        call_success = isinstance(row.get("http_status"), int) and row["http_status"] < 400
        if (proxy_success or call_success) and row.get("request_id"):
            request_ids.add(str(row["request_id"]))
    return request_ids


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
            "x-client-id": CLIENT_ID,
            "x-task-id": TASK_ID,
            "x-request-id": row["request_id"],
            "x-cost-center": "demo",
        },
        method="POST",
    )
    started = time.monotonic()
    try:
        with urllib.request.urlopen(request, timeout=180) as response:
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


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open() as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


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


def slugify(value: str) -> str:
    value = value.lower().replace("&", " and ")
    value = re.sub(r"[^a-z0-9]+", "_", value)
    return re.sub(r"_+", "_", value).strip("_")


if __name__ == "__main__":
    raise SystemExit(main())
