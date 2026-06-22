#!/usr/bin/env python3
"""Detect simple student drift and optionally roll routes back to teacher_only."""

from __future__ import annotations

import argparse
import glob
import json
import os
import statistics
import sys
import tempfile
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any


ACTIVE_MODES = {"shadow", "canary", "student_only", "bandit"}
STUDENT_MODES = {"canary", "student_only", "bandit"}
NEGATIVE_RATINGS = {"bad", "incorrect", "wrong", "negative", "thumbs_down", "0", "1"}


@dataclass
class TaskStats:
    task_id: str
    route_mode: str
    route: dict[str, Any]
    student_requests: int = 0
    student_errors: int = 0
    student_latencies_ms: list[int] = field(default_factory=list)
    feedback_total: int = 0
    feedback_negative: int = 0
    reasons: list[str] = field(default_factory=list)

    def error_rate(self) -> float:
        if self.student_requests == 0:
            return 0.0
        return self.student_errors / self.student_requests

    def negative_feedback_rate(self) -> float:
        if self.feedback_total == 0:
            return 0.0
        return self.feedback_negative / self.feedback_total

    def p95_latency_ms(self) -> int | None:
        return percentile(self.student_latencies_ms, 95)

    def as_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "route_mode": self.route_mode,
            "student_model": self.route.get("student_model"),
            "student_requests": self.student_requests,
            "student_errors": self.student_errors,
            "student_error_rate": round(self.error_rate(), 6),
            "student_p95_latency_ms": self.p95_latency_ms(),
            "feedback_total": self.feedback_total,
            "feedback_negative": self.feedback_negative,
            "negative_feedback_rate": round(self.negative_feedback_rate(), 6),
            "rollback_reasons": self.reasons,
        }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--logs",
        default="data/logs/*.jsonl",
        help="Proxy JSONL log path or glob. Default: data/logs/*.jsonl",
    )
    parser.add_argument(
        "--feedback",
        default="data/logs/feedback.jsonl",
        help="Feedback JSONL path or glob. Default: data/logs/feedback.jsonl",
    )
    parser.add_argument(
        "--snapshot",
        default="config/routing_snapshot.json",
        help="Routing snapshot path. Default: config/routing_snapshot.json",
    )
    parser.add_argument(
        "--registry",
        default="registry/events.jsonl",
        help="Registry events JSONL path. Default: registry/events.jsonl",
    )
    parser.add_argument("--task-id", help="Limit analysis to one task")
    parser.add_argument("--window-hours", type=int, default=24)
    parser.add_argument("--min-requests", type=int, default=20)
    parser.add_argument("--max-error-rate", type=float, default=0.05)
    parser.add_argument(
        "--max-p95-latency-ms",
        type=int,
        default=0,
        help="Disable latency guard with 0. Default: 0",
    )
    parser.add_argument("--min-feedback", type=int, default=5)
    parser.add_argument("--max-negative-feedback-rate", type=float, default=0.20)
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Write teacher_only rollback routes for tasks breaching thresholds.",
    )
    args = parser.parse_args()

    snapshot_path = Path(args.snapshot)
    snapshot = load_snapshot(snapshot_path)
    active_tasks = active_routes(snapshot, args.task_id)
    if not active_tasks:
        print(json.dumps({"status": "ok", "message": "no active student routes"}, indent=2))
        return 0

    since = datetime.now(timezone.utc) - timedelta(hours=args.window_hours)
    stats = {
        task_id: TaskStats(task_id=task_id, route_mode=route.get("mode", ""), route=route)
        for task_id, route in active_tasks.items()
    }

    request_index = read_proxy_logs(args.logs, since, stats)
    read_feedback(args.feedback, since, stats, request_index)
    evaluate(stats, args)

    rollback_tasks = [task_id for task_id, task_stats in stats.items() if task_stats.reasons]
    if args.apply and rollback_tasks:
        snapshot = rollback_snapshot(snapshot, rollback_tasks)
        write_json_atomic(snapshot_path, snapshot)
        append_event(Path(args.registry), snapshot_path, snapshot, stats, rollback_tasks, args)

    report = {
        "status": "rollback_required" if rollback_tasks else "ok",
        "applied": bool(args.apply and rollback_tasks),
        "window_hours": args.window_hours,
        "thresholds": {
            "min_requests": args.min_requests,
            "max_error_rate": args.max_error_rate,
            "max_p95_latency_ms": args.max_p95_latency_ms,
            "min_feedback": args.min_feedback,
            "max_negative_feedback_rate": args.max_negative_feedback_rate,
        },
        "rollback_tasks": rollback_tasks,
        "tasks": [task_stats.as_dict() for task_stats in stats.values()],
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 1 if rollback_tasks and not args.apply else 0


def load_snapshot(path: Path) -> dict[str, Any]:
    if not path.exists():
        print(f"missing routing snapshot: {path}", file=sys.stderr)
        raise SystemExit(2)
    return json.loads(path.read_text())


def active_routes(snapshot: dict[str, Any], task_id: str | None) -> dict[str, dict[str, Any]]:
    tasks = snapshot.get("tasks", {})
    active: dict[str, dict[str, Any]] = {}
    for current_task_id, route in tasks.items():
        if task_id and current_task_id != task_id:
            continue
        if route.get("mode") in ACTIVE_MODES:
            active[current_task_id] = dict(route)
    return active


def read_proxy_logs(
    pattern: str,
    since: datetime,
    stats: dict[str, TaskStats],
) -> dict[str, tuple[str, str]]:
    request_index: dict[str, tuple[str, str]] = {}
    for path in expand_paths(pattern):
        for entry in read_jsonl(path):
            timestamp = parse_timestamp(entry.get("timestamp"))
            if timestamp is None or timestamp < since:
                continue

            task_id = entry.get("task_id")
            task_stats = stats.get(task_id)
            if task_stats is None:
                continue

            routing_decision = str(entry.get("routing_decision") or "")
            request_id = entry.get("request_id")
            if isinstance(request_id, str) and request_id:
                request_index[request_id] = (task_id, routing_decision)

            if task_stats.route_mode not in STUDENT_MODES or routing_decision != "student":
                continue

            task_stats.student_requests += 1
            http_status = as_int(entry.get("http_status")) or 0
            if entry.get("status") == "error" or http_status >= 400:
                task_stats.student_errors += 1

            latency_ms = as_int(entry.get("latency_ms"))
            if latency_ms is not None:
                task_stats.student_latencies_ms.append(latency_ms)

    return request_index


def read_feedback(
    pattern: str,
    since: datetime,
    stats: dict[str, TaskStats],
    request_index: dict[str, tuple[str, str]],
) -> None:
    for path in expand_paths(pattern):
        for entry in read_jsonl(path):
            timestamp = parse_timestamp(entry.get("timestamp"))
            if timestamp is None or timestamp < since:
                continue

            task_id = None
            request_id = entry.get("request_id")
            if isinstance(request_id, str) and request_id in request_index:
                indexed_task_id, routing_decision = request_index[request_id]
                if routing_decision == "student":
                    task_id = indexed_task_id
            elif isinstance(entry.get("task_id"), str):
                task_id = entry["task_id"]

            task_stats = stats.get(task_id)
            if task_stats is None or task_stats.route_mode not in STUDENT_MODES:
                continue

            task_stats.feedback_total += 1
            rating = str(entry.get("rating") or "").strip().lower()
            if rating in NEGATIVE_RATINGS:
                task_stats.feedback_negative += 1


def evaluate(stats: dict[str, TaskStats], args: argparse.Namespace) -> None:
    for task_stats in stats.values():
        if task_stats.route_mode == "shadow":
            continue

        if task_stats.student_requests >= args.min_requests:
            error_rate = task_stats.error_rate()
            if error_rate > args.max_error_rate:
                task_stats.reasons.append(
                    f"student_error_rate {error_rate:.4f} > {args.max_error_rate:.4f}"
                )

            p95_latency = task_stats.p95_latency_ms()
            if (
                args.max_p95_latency_ms > 0
                and p95_latency is not None
                and p95_latency > args.max_p95_latency_ms
            ):
                task_stats.reasons.append(
                    f"student_p95_latency_ms {p95_latency} > {args.max_p95_latency_ms}"
                )

        if task_stats.feedback_total >= args.min_feedback:
            negative_rate = task_stats.negative_feedback_rate()
            if negative_rate > args.max_negative_feedback_rate:
                task_stats.reasons.append(
                    "negative_feedback_rate "
                    f"{negative_rate:.4f} > {args.max_negative_feedback_rate:.4f}"
                )


def rollback_snapshot(
    snapshot: dict[str, Any],
    rollback_tasks: list[str],
) -> dict[str, Any]:
    updated = dict(snapshot)
    updated["version"] = int(updated.get("version", 0)) + 1
    tasks = dict(updated.get("tasks", {}))
    for task_id in rollback_tasks:
        tasks[task_id] = {"mode": "teacher_only"}
    updated["tasks"] = tasks
    return updated


def append_event(
    path: Path,
    snapshot_path: Path,
    snapshot: dict[str, Any],
    stats: dict[str, TaskStats],
    rollback_tasks: list[str],
    args: argparse.Namespace,
) -> None:
    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "event": "drift_guard_rollback",
        "snapshot": str(snapshot_path),
        "snapshot_version": snapshot["version"],
        "window_hours": args.window_hours,
        "tasks": [stats[task_id].as_dict() for task_id in rollback_tasks],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False) as tmp:
        json.dump(payload, tmp, indent=2, sort_keys=True)
        tmp.write("\n")
        tmp_path = Path(tmp.name)
    os.replace(tmp_path, path)


def expand_paths(pattern: str) -> list[Path]:
    matches = [Path(path) for path in sorted(glob.glob(pattern))]
    if matches:
        return matches
    path = Path(pattern)
    return [path] if path.exists() else []


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    with path.open() as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                print(f"skipping invalid JSON in {path}:{line_number}: {exc}", file=sys.stderr)
                continue
            if isinstance(value, dict):
                entries.append(value)
    return entries


def parse_timestamp(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    normalized = value.replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def as_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def percentile(values: list[int], percent: int) -> int | None:
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    quantiles = statistics.quantiles(values, n=100, method="inclusive")
    return int(round(quantiles[percent - 1]))


if __name__ == "__main__":
    raise SystemExit(main())
