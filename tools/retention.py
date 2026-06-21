#!/usr/bin/env python3
"""Apply DistillForge local retention policies with dry-run by default."""

from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--logs", default="data/logs/proxy*.jsonl")
    parser.add_argument("--feedback", default="data/logs/feedback.jsonl")
    parser.add_argument("--shadow", default="data/logs/shadow.jsonl")
    parser.add_argument("--registry", default="registry/events.jsonl")
    parser.add_argument("--datasets-root", default="data/datasets")
    parser.add_argument("--logs-days", type=int, default=90)
    parser.add_argument("--feedback-days", type=int, default=365)
    parser.add_argument("--shadow-days", type=int, default=90)
    parser.add_argument("--registry-days", type=int, default=1095)
    parser.add_argument("--datasets-days", type=int, default=365)
    parser.add_argument("--apply", action="store_true", help="Modify files. Default is dry-run.")
    args = parser.parse_args()

    now = datetime.now(timezone.utc)
    report = {
        "mode": "apply" if args.apply else "dry_run",
        "generated_at": now.isoformat(),
        "jsonl": [
            prune_jsonl_paths(args.logs, args.logs_days, now, args.apply, "proxy_logs"),
            prune_jsonl_paths(args.feedback, args.feedback_days, now, args.apply, "feedback"),
            prune_jsonl_paths(args.shadow, args.shadow_days, now, args.apply, "shadow"),
            prune_jsonl_paths(args.registry, args.registry_days, now, args.apply, "registry"),
        ],
        "datasets": prune_datasets(Path(args.datasets_root), args.datasets_days, now, args.apply),
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


def prune_jsonl_paths(
    pattern: str,
    retention_days: int,
    now: datetime,
    apply: bool,
    label: str,
) -> dict[str, Any]:
    cutoff = now - timedelta(days=retention_days)
    files = expand_paths(pattern)
    return {
        "label": label,
        "pattern": pattern,
        "retention_days": retention_days,
        "cutoff": cutoff.isoformat(),
        "files": [prune_jsonl(path, cutoff, apply) for path in files],
    }


def prune_jsonl(path: Path, cutoff: datetime, apply: bool) -> dict[str, Any]:
    kept: list[str] = []
    removed = 0
    total = 0
    invalid = 0
    old_timestamp_missing = 0

    with path.open() as handle:
        for line in handle:
            total += 1
            stripped = line.strip()
            if not stripped:
                kept.append(line)
                continue
            try:
                payload = json.loads(stripped)
            except json.JSONDecodeError:
                invalid += 1
                kept.append(line)
                continue

            timestamp = parse_timestamp(payload.get("timestamp") if isinstance(payload, dict) else None)
            if timestamp is None:
                old_timestamp_missing += 1
                kept.append(line)
            elif timestamp < cutoff:
                removed += 1
            else:
                kept.append(line)

    if apply and removed > 0:
        write_lines_atomic(path, kept)

    return {
        "path": str(path),
        "total_lines": total,
        "removed_lines": removed,
        "kept_lines": total - removed,
        "invalid_lines_kept": invalid,
        "missing_timestamp_kept": old_timestamp_missing,
    }


def prune_datasets(root: Path, retention_days: int, now: datetime, apply: bool) -> dict[str, Any]:
    cutoff = now - timedelta(days=retention_days)
    expired: list[dict[str, Any]] = []
    kept = 0

    if not root.exists():
        return {
            "root": str(root),
            "retention_days": retention_days,
            "cutoff": cutoff.isoformat(),
            "kept": 0,
            "expired": [],
        }

    for manifest_path in sorted(root.glob("*/*/manifest.json")):
        dataset_dir = manifest_path.parent
        created_at = dataset_created_at(manifest_path)
        if created_at is not None and created_at < cutoff:
            expired.append({"path": str(dataset_dir), "created_at": created_at.isoformat()})
            if apply:
                shutil.rmtree(dataset_dir)
        else:
            kept += 1

    return {
        "root": str(root),
        "retention_days": retention_days,
        "cutoff": cutoff.isoformat(),
        "kept": kept,
        "expired": expired,
    }


def dataset_created_at(manifest_path: Path) -> Optional[datetime]:
    try:
        manifest = json.loads(manifest_path.read_text())
    except json.JSONDecodeError:
        return None
    timestamp = parse_timestamp(manifest.get("created_at"))
    if timestamp is not None:
        return timestamp
    return datetime.fromtimestamp(manifest_path.stat().st_mtime, tz=timezone.utc)


def write_lines_atomic(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False) as tmp:
        tmp.writelines(lines)
        tmp_path = Path(tmp.name)
    os.replace(tmp_path, path)


def expand_paths(pattern: str) -> list[Path]:
    matches = [Path(path) for path in sorted(glob.glob(pattern))]
    if matches:
        return matches
    path = Path(pattern)
    return [path] if path.exists() else []


def parse_timestamp(value: Any) -> Optional[datetime]:
    if not isinstance(value, str) or not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


if __name__ == "__main__":
    raise SystemExit(main())
