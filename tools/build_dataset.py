#!/usr/bin/env python3
"""Build a versioned DistillForge dataset from redacted JSONL logs."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--logs",
        default="data/logs/*.jsonl",
        help="JSONL log path or glob. Default: data/logs/*.jsonl",
    )
    parser.add_argument("--task-id", required=True, help="Task id to export")
    parser.add_argument(
        "--out",
        default="data/datasets",
        help="Dataset root directory. Default: data/datasets",
    )
    parser.add_argument(
        "--dataset-id",
        help="Dataset id. Default: ds_{task_id}_{UTC timestamp}",
    )
    parser.add_argument(
        "--min-samples",
        type=int,
        default=1,
        help="Fail if fewer eligible examples are found. Default: 1",
    )
    args = parser.parse_args()

    try:
        import duckdb
    except ImportError:
        print(
            "duckdb is required. Install it with: python3 -m pip install -r requirements-analytics.txt",
            file=sys.stderr,
        )
        return 2

    dataset_id = args.dataset_id or default_dataset_id(args.task_id)
    dataset_dir = Path(args.out) / args.task_id / dataset_id
    dataset_dir.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect(":memory:")
    logs_literal = sql_string_literal(args.logs)
    task_literal = sql_string_literal(args.task_id)

    try:
        con.execute(
            f"""
            CREATE TEMP VIEW logs AS
            SELECT *
            FROM read_json_auto({logs_literal}, format='newline_delimited', union_by_name=true)
            """
        )
        con.execute(
            f"""
            CREATE TEMP TABLE eligible AS
            WITH ranked AS (
              SELECT
                task_id,
                request_id,
                client_id,
                cost_center,
                timestamp,
                teacher_model,
                selected_model,
                routing_decision,
                routing_reason,
                prompt_redacted AS input,
                response_redacted AS teacher_output,
                response_redacted AS validated_output,
                input_hash,
                input_tokens,
                output_tokens,
                estimated_cost_usd,
                estimated_teacher_cost_usd,
                estimated_savings_usd,
                row_number() OVER (
                  PARTITION BY coalesce(CAST(input_hash AS VARCHAR), CAST(request_id AS VARCHAR))
                  ORDER BY timestamp DESC
                ) AS duplicate_rank
              FROM logs
              WHERE task_id = {task_literal}
                AND status = 'success'
                AND coalesce(training_eligible, false)
                AND prompt_redacted IS NOT NULL
                AND response_redacted IS NOT NULL
            )
            SELECT
              *,
              CASE
                WHEN hash(coalesce(CAST(request_id AS VARCHAR), CAST(input_hash AS VARCHAR))) % 100 < 70 THEN 'train'
                WHEN hash(coalesce(CAST(request_id AS VARCHAR), CAST(input_hash AS VARCHAR))) % 100 < 85 THEN 'validation'
                ELSE 'test'
              END AS split
            FROM ranked
            WHERE duplicate_rank = 1
            """
        )
    except Exception as exc:
        print(f"failed to build eligible dataset from {args.logs}: {exc}", file=sys.stderr)
        return 1

    total = con.execute("SELECT count(*) FROM eligible").fetchone()[0]
    if total < args.min_samples:
        print(
            f"not enough eligible samples for {args.task_id}: {total} < {args.min_samples}",
            file=sys.stderr,
        )
        return 1

    for split in ("train", "validation", "test"):
        split_literal = sql_string_literal(split)
        path_literal = sql_string_literal(str(dataset_dir / f"{split}.parquet"))
        con.execute(
            f"""
            COPY (
              SELECT
                task_id,
                input,
                teacher_output,
                validated_output,
                split,
                request_id,
                client_id,
                cost_center,
                timestamp,
                teacher_model,
                selected_model,
                routing_decision,
                routing_reason,
                input_hash,
                input_tokens,
                output_tokens,
                estimated_cost_usd,
                estimated_teacher_cost_usd,
                estimated_savings_usd
              FROM eligible
              WHERE split = {split_literal}
              ORDER BY timestamp, request_id
            )
            TO {path_literal} (FORMAT PARQUET)
            """
        )

    manifest = build_manifest(con, args.logs, args.task_id, dataset_id)
    manifest_path = dataset_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    print(f"Wrote dataset {dataset_id} to {dataset_dir}")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


def build_manifest(con: Any, logs: str, task_id: str, dataset_id: str) -> dict[str, Any]:
    counts = dict(
        con.execute(
            """
            SELECT split, count(*) AS samples
            FROM eligible
            GROUP BY split
            """
        ).fetchall()
    )
    summary = con.execute(
        """
        SELECT
          count(*) AS samples,
          min(timestamp) AS min_timestamp,
          max(timestamp) AS max_timestamp,
          sum(coalesce(estimated_teacher_cost_usd, 0)) AS teacher_cost_usd,
          sum(coalesce(estimated_savings_usd, 0)) AS estimated_savings_usd
        FROM eligible
        """
    ).fetchone()

    return {
        "dataset_id": dataset_id,
        "task_id": task_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_logs": logs,
        "format": "parquet",
        "samples": int(summary[0] or 0),
        "splits": {
            "train": int(counts.get("train", 0)),
            "validation": int(counts.get("validation", 0)),
            "test": int(counts.get("test", 0)),
        },
        "time_range": {
            "min_timestamp": str(summary[1]) if summary[1] is not None else None,
            "max_timestamp": str(summary[2]) if summary[2] is not None else None,
        },
        "teacher_cost_usd": float(summary[3] or 0.0),
        "estimated_savings_usd": float(summary[4] or 0.0),
        "filters": {
            "status": "success",
            "training_eligible": True,
            "required_fields": ["prompt_redacted", "response_redacted"],
            "dedupe_key": "coalesce(input_hash, request_id)",
            "split_strategy": "stable hash(request_id/input_hash): 70/15/15",
        },
    }


def default_dataset_id(task_id: str) -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    safe_task_id = "".join(char if char.isalnum() or char in "-_" else "_" for char in task_id)
    return f"ds_{safe_task_id}_{timestamp}"


def sql_string_literal(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


if __name__ == "__main__":
    raise SystemExit(main())
