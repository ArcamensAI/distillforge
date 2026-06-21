#!/usr/bin/env python3
"""Generate a FinOps report from DistillForge JSONL logs using DuckDB."""

from __future__ import annotations

import argparse
import sys
from typing import Any


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--logs",
        default="data/logs/*.jsonl",
        help="JSONL log path or glob. Default: data/logs/*.jsonl",
    )
    parser.add_argument(
        "--candidate-min-requests",
        type=int,
        default=20,
        help="Minimum requests for distillation candidates. Default: 20",
    )
    args = parser.parse_args()

    try:
        import duckdb
    except ImportError:
        print(
            "duckdb is required for this report. Install it with: python3 -m pip install duckdb",
            file=sys.stderr,
        )
        return 2

    con = duckdb.connect(":memory:")
    logs_literal = sql_string_literal(args.logs)
    try:
        con.execute(
            f"""
            CREATE TEMP VIEW logs AS
            SELECT *
            FROM read_json_auto({logs_literal}, format='newline_delimited', union_by_name=true)
            """
        )
    except Exception as exc:  # DuckDB reports file/glob errors with its own types.
        print(f"failed to read logs from {args.logs}: {exc}", file=sys.stderr)
        return 1

    print("# DistillForge FinOps Report\n")
    print(f"Logs: `{args.logs}`\n")

    print("## Summary\n")
    print_table(
        *query(
            con,
            """
            SELECT
              count(*) AS requests,
              sum(CASE WHEN routing_decision = 'teacher' THEN 1 ELSE 0 END) AS teacher_requests,
              sum(CASE WHEN routing_decision = 'student' THEN 1 ELSE 0 END) AS student_requests,
              sum(CASE WHEN status = 'error' THEN 1 ELSE 0 END) AS errors,
              round(sum(coalesce(estimated_cost_usd, 0)), 8) AS estimated_cost_usd,
              round(sum(coalesce(estimated_teacher_cost_usd, 0)), 8) AS estimated_teacher_cost_usd,
              round(sum(coalesce(estimated_savings_usd, 0)), 8) AS estimated_savings_usd,
              round(avg(latency_ms), 2) AS avg_latency_ms,
              round(quantile_cont(latency_ms, 0.95), 2) AS p95_latency_ms
            FROM logs
            """,
        )
    )

    print("\n## By Task\n")
    print_table(
        *query(
            con,
            """
            SELECT
              task_id,
              count(*) AS requests,
              sum(CASE WHEN routing_decision = 'student' THEN 1 ELSE 0 END) AS student_requests,
              round(sum(coalesce(estimated_cost_usd, 0)), 8) AS estimated_cost_usd,
              round(sum(coalesce(estimated_savings_usd, 0)), 8) AS estimated_savings_usd,
              round(avg(latency_ms), 2) AS avg_latency_ms,
              round(quantile_cont(latency_ms, 0.95), 2) AS p95_latency_ms
            FROM logs
            GROUP BY task_id
            ORDER BY estimated_savings_usd DESC, requests DESC
            LIMIT 20
            """,
        )
    )

    print("\n## By Model\n")
    print_table(
        *query(
            con,
            """
            SELECT
              selected_model,
              routing_decision,
              count(*) AS requests,
              round(sum(coalesce(estimated_cost_usd, 0)), 8) AS estimated_cost_usd,
              round(sum(coalesce(estimated_savings_usd, 0)), 8) AS estimated_savings_usd,
              round(avg(latency_ms), 2) AS avg_latency_ms
            FROM logs
            GROUP BY selected_model, routing_decision
            ORDER BY requests DESC
            """,
        )
    )

    print("\n## Errors\n")
    print_table(
        *query(
            con,
            """
            SELECT
              coalesce(error_code, 'none') AS error_code,
              count(*) AS requests
            FROM logs
            WHERE status = 'error'
            GROUP BY error_code
            ORDER BY requests DESC
            """,
        )
    )

    print("\n## Distillation Candidates\n")
    print_table(
        *query(
            con,
            f"""
            SELECT
              task_id,
              count(*) AS requests,
              sum(CASE WHEN routing_decision = 'teacher' THEN 1 ELSE 0 END) AS teacher_requests,
              round(sum(coalesce(estimated_teacher_cost_usd, 0)), 8) AS teacher_cost_usd,
              round(sum(coalesce(estimated_teacher_cost_usd, 0)) - sum(coalesce(estimated_cost_usd, 0)), 8) AS current_savings_usd,
              round(avg(latency_ms), 2) AS avg_latency_ms,
              round(sum(CASE WHEN status = 'error' THEN 1 ELSE 0 END)::DOUBLE / count(*), 6) AS error_rate,
              round(
                sum(coalesce(estimated_teacher_cost_usd, 0))
                * CASE
                    WHEN sum(CASE WHEN routing_decision = 'student' THEN 1 ELSE 0 END) = 0 THEN 0.70
                    ELSE 0.25
                  END,
                8
              ) AS estimated_remaining_savings_usd
            FROM logs
            GROUP BY task_id
            HAVING count(*) >= {int(args.candidate_min_requests)}
            ORDER BY estimated_remaining_savings_usd DESC, teacher_cost_usd DESC
            LIMIT 20
            """,
        )
    )

    return 0


def query(con: Any, sql: str) -> tuple[list[str], list[tuple[Any, ...]]]:
    result = con.execute(sql)
    headers = [column[0] for column in result.description]
    return headers, result.fetchall()


def sql_string_literal(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def print_table(headers: list[str], rows: list[tuple[Any, ...]]) -> None:
    if not rows:
        print("_No rows._")
        return

    rendered_rows = [[format_cell(value) for value in row] for row in rows]
    widths = [
        max(len(str(header)), *(len(row[index]) for row in rendered_rows))
        for index, header in enumerate(headers)
    ]
    print("| " + " | ".join(pad(header, widths[index]) for index, header in enumerate(headers)) + " |")
    print("| " + " | ".join("-" * width for width in widths) + " |")
    for row in rendered_rows:
        print("| " + " | ".join(pad(value, widths[index]) for index, value in enumerate(row)) + " |")


def format_cell(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.8f}".rstrip("0").rstrip(".")
    return str(value)


def pad(value: Any, width: int) -> str:
    return str(value).ljust(width)


if __name__ == "__main__":
    raise SystemExit(main())
