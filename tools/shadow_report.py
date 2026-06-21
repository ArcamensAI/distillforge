#!/usr/bin/env python3
"""Generate a divergence report from DistillForge shadow JSONL logs."""

from __future__ import annotations

import argparse
import sys
from typing import Any


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--logs",
        default="data/logs/shadow.jsonl",
        help="Shadow JSONL log path or glob. Default: data/logs/shadow.jsonl",
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

    con = duckdb.connect(":memory:")
    try:
        con.execute(
            f"""
            CREATE TEMP VIEW shadow AS
            SELECT *
            FROM read_json_auto({sql_string_literal(args.logs)}, format='newline_delimited', union_by_name=true)
            """
        )
    except Exception as exc:
        print(f"failed to read shadow logs from {args.logs}: {exc}", file=sys.stderr)
        return 1

    print("# DistillForge Shadow Report\n")
    print(f"Logs: `{args.logs}`\n")

    print("## Summary\n")
    print_table(
        *query(
            con,
            """
            SELECT
              count(*) AS probes,
              sum(CASE WHEN status = 'error' THEN 1 ELSE 0 END) AS errors,
              round(sum(CASE WHEN status = 'error' THEN 1 ELSE 0 END)::DOUBLE / count(*), 6) AS error_rate,
              sum(CASE WHEN response_exact_match = false THEN 1 ELSE 0 END) AS divergences,
              round(sum(CASE WHEN response_exact_match = false THEN 1 ELSE 0 END)::DOUBLE / count(*), 6) AS divergence_rate,
              round(avg(latency_ms), 2) AS avg_latency_ms,
              round(quantile_cont(latency_ms, 0.95), 2) AS p95_latency_ms
            FROM shadow
            """,
        )
    )

    print("\n## By Task And Student\n")
    print_table(
        *query(
            con,
            """
            SELECT
              task_id,
              student_model,
              count(*) AS probes,
              sum(CASE WHEN status = 'error' THEN 1 ELSE 0 END) AS errors,
              round(sum(CASE WHEN status = 'error' THEN 1 ELSE 0 END)::DOUBLE / count(*), 6) AS error_rate,
              sum(CASE WHEN response_exact_match = false THEN 1 ELSE 0 END) AS divergences,
              round(sum(CASE WHEN response_exact_match = false THEN 1 ELSE 0 END)::DOUBLE / count(*), 6) AS divergence_rate,
              round(avg(latency_ms), 2) AS avg_latency_ms,
              round(quantile_cont(latency_ms, 0.95), 2) AS p95_latency_ms
            FROM shadow
            GROUP BY task_id, student_model
            ORDER BY divergence_rate DESC, probes DESC
            """
        )
    )

    print("\n## Errors\n")
    print_table(
        *query(
            con,
            """
            SELECT
              coalesce(error_code, 'none') AS error_code,
              count(*) AS probes
            FROM shadow
            WHERE status = 'error'
            GROUP BY error_code
            ORDER BY probes DESC
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
