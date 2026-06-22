#!/usr/bin/env python3
"""Promote or roll back a DistillForge student model in a routing snapshot."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SUPPORTED_MODES = {"teacher_only", "shadow", "canary", "student_only", "bandit"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", help="Model directory containing manifest.json")
    parser.add_argument("--task-id", help="Task id. Required for teacher_only rollback without model")
    parser.add_argument(
        "--snapshot",
        default="config/routing_snapshot.json",
        help="Routing snapshot path",
    )
    parser.add_argument(
        "--registry",
        default="registry/events.jsonl",
        help="Registry events JSONL path",
    )
    parser.add_argument("--mode", required=True, choices=sorted(SUPPORTED_MODES))
    parser.add_argument(
        "--student-traffic-percentage",
        type=int,
        default=1,
        help="Canary percentage. Default: 1",
    )
    parser.add_argument(
        "--teacher-probe-percentage",
        type=int,
        default=2,
        help="Bandit teacher probe percentage. Default: 2",
    )
    parser.add_argument("--min-accuracy", type=float, default=0.0)
    parser.add_argument("--min-macro-f1", type=float, default=0.0)
    args = parser.parse_args()

    if args.mode != "teacher_only" and not args.model_dir:
        print("--model-dir is required unless --mode teacher_only", file=sys.stderr)
        return 2

    model_manifest = load_model_manifest(Path(args.model_dir)) if args.model_dir else None
    task_id = args.task_id or (model_manifest or {}).get("task_id")
    if not task_id:
        print("--task-id is required when no model manifest provides task_id", file=sys.stderr)
        return 2

    if args.mode == "canary" and not 0 <= args.student_traffic_percentage <= 100:
        print("--student-traffic-percentage must be between 0 and 100", file=sys.stderr)
        return 2
    if args.mode == "bandit" and not 0 <= args.teacher_probe_percentage <= 100:
        print("--teacher-probe-percentage must be between 0 and 100", file=sys.stderr)
        return 2

    if model_manifest is not None:
        validation_error = validate_model(model_manifest, args.min_accuracy, args.min_macro_f1)
        if validation_error:
            print(validation_error, file=sys.stderr)
            return 1

    snapshot_path = Path(args.snapshot)
    snapshot = load_snapshot(snapshot_path)
    new_snapshot = update_snapshot(
        snapshot,
        task_id,
        args.mode,
        model_manifest,
        args.student_traffic_percentage,
        args.teacher_probe_percentage,
    )
    write_json_atomic(snapshot_path, new_snapshot)

    event = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "event": "routing_updated",
        "task_id": task_id,
        "mode": args.mode,
        "snapshot": str(snapshot_path),
        "snapshot_version": new_snapshot["version"],
        "model": model_manifest,
        "thresholds": {
            "min_accuracy": args.min_accuracy,
            "min_macro_f1": args.min_macro_f1,
        },
    }
    append_jsonl(Path(args.registry), event)

    print(
        json.dumps(
            {
                "task_id": task_id,
                "mode": args.mode,
                "snapshot_version": new_snapshot["version"],
                "snapshot": str(snapshot_path),
                "registry": args.registry,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def load_model_manifest(model_dir: Path) -> dict[str, Any]:
    manifest_path = model_dir / "manifest.json"
    if not manifest_path.exists():
        raise SystemExit(f"missing model manifest: {manifest_path}")
    return json.loads(manifest_path.read_text())


def validate_model(manifest: dict[str, Any], min_accuracy: float, min_macro_f1: float) -> str | None:
    metrics = manifest.get("metrics", {})
    accuracy = float(metrics.get("accuracy", 0.0))
    macro_f1 = float(metrics.get("macro_f1", 0.0))
    status = manifest.get("status")
    if status not in {"offline_eval", "shadow", "canary", "production"}:
        return f"model status {status!r} is not promotable"
    if accuracy < min_accuracy:
        return f"model accuracy {accuracy} is below threshold {min_accuracy}"
    if macro_f1 < min_macro_f1:
        return f"model macro_f1 {macro_f1} is below threshold {min_macro_f1}"
    return None


def load_snapshot(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"version": 0, "default_mode": "teacher_only", "tasks": {}}
    return json.loads(path.read_text())


def update_snapshot(
    snapshot: dict[str, Any],
    task_id: str,
    mode: str,
    model_manifest: dict[str, Any] | None,
    canary_percentage: int,
    teacher_probe_percentage: int,
) -> dict[str, Any]:
    updated = dict(snapshot)
    updated["version"] = int(updated.get("version", 0)) + 1
    updated.setdefault("default_mode", "teacher_only")
    tasks = dict(updated.get("tasks", {}))

    route: dict[str, Any] = {"mode": mode}
    if mode != "teacher_only":
        assert model_manifest is not None
        route["student_model"] = model_manifest["model_id"]
        if mode == "canary":
            route["student_traffic_percentage"] = canary_percentage
        if mode == "bandit":
            route["teacher_probe_percentage"] = teacher_probe_percentage

    tasks[task_id] = route
    updated["tasks"] = tasks
    return updated


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False) as tmp:
        json.dump(payload, tmp, indent=2, sort_keys=True)
        tmp.write("\n")
        tmp_path = Path(tmp.name)
    os.replace(tmp_path, path)


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
