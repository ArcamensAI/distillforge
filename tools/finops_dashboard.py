#!/usr/bin/env python3
"""Generate a self-contained DistillForge FinOps HTML dashboard."""

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
    parser.add_argument(
        "--out",
        default="reports/distillforge_dashboard.html",
        help="Output HTML path. Default: reports/distillforge_dashboard.html",
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
            "duckdb is required. Install it with: python3 -m pip install -r requirements-analytics.txt",
            file=sys.stderr,
        )
        return 2

    con = duckdb.connect(":memory:")
    try:
        create_logs_view(con, args.logs)
    except Exception as exc:
        print(f"failed to read logs from {args.logs}: {exc}", file=sys.stderr)
        return 1

    data = build_dashboard_data(con, args.logs, args.candidate_min_requests)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(render_html(data), encoding="utf-8")
    print(f"Wrote dashboard to {out_path}")
    return 0


def create_logs_view(con: Any, logs: str) -> None:
    logs_literal = sql_string_literal(logs)
    con.execute(
        f"""
        CREATE TEMP VIEW raw_logs AS
        SELECT *
        FROM read_json_auto({logs_literal}, format='newline_delimited', union_by_name=true)
        """
    )
    con.execute(
        """
        CREATE TEMP VIEW logs AS
        SELECT
          CAST(coalesce(timestamp, '') AS VARCHAR) AS timestamp,
          CAST(coalesce(request_id, '') AS VARCHAR) AS request_id,
          CAST(coalesce(task_id, 'unknown_task') AS VARCHAR) AS task_id,
          CAST(coalesce(client_id, 'unknown_client') AS VARCHAR) AS client_id,
          CAST(coalesce(cost_center, 'unassigned') AS VARCHAR) AS cost_center,
          CAST(coalesce(selected_model, 'unknown_model') AS VARCHAR) AS selected_model,
          CAST(coalesce(routing_decision, 'unknown') AS VARCHAR) AS routing_decision,
          CAST(coalesce(routing_reason, '') AS VARCHAR) AS routing_reason,
          CAST(coalesce(status, 'unknown') AS VARCHAR) AS status,
          CAST(coalesce(error_code, '') AS VARCHAR) AS error_code,
          CAST(coalesce(latency_ms, 0) AS DOUBLE) AS latency_ms,
          CAST(coalesce(input_tokens, 0) AS DOUBLE) AS input_tokens,
          CAST(coalesce(output_tokens, 0) AS DOUBLE) AS output_tokens,
          CAST(coalesce(estimated_cost_usd, 0) AS DOUBLE) AS estimated_cost_usd,
          CAST(coalesce(estimated_teacher_cost_usd, 0) AS DOUBLE) AS estimated_teacher_cost_usd,
          CAST(coalesce(estimated_savings_usd, 0) AS DOUBLE) AS estimated_savings_usd
        FROM raw_logs
        """
    )


def build_dashboard_data(con: Any, logs: str, candidate_min_requests: int) -> dict[str, Any]:
    summary = one(
        con,
        """
        SELECT
          count(*) AS requests,
          sum(CASE WHEN routing_decision = 'teacher' THEN 1 ELSE 0 END) AS teacher_requests,
          sum(CASE WHEN routing_decision = 'student' THEN 1 ELSE 0 END) AS student_requests,
          sum(CASE WHEN routing_reason LIKE '%fallback%' THEN 1 ELSE 0 END) AS fallback_requests,
          sum(CASE WHEN status = 'error' THEN 1 ELSE 0 END) AS errors,
          round(sum(estimated_cost_usd), 8) AS estimated_cost_usd,
          round(sum(estimated_teacher_cost_usd), 8) AS estimated_teacher_cost_usd,
          round(sum(estimated_savings_usd), 8) AS estimated_savings_usd,
          round(avg(latency_ms), 2) AS avg_latency_ms,
          round(quantile_cont(latency_ms, 0.95), 2) AS p95_latency_ms
        FROM logs
        """,
    )
    requests = float(summary.get("requests") or 0)
    errors = float(summary.get("errors") or 0)
    summary["error_rate"] = round(errors / requests, 6) if requests else 0.0

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_logs": logs,
        "summary": summary,
        "by_task": rows(
            con,
            """
            SELECT
              task_id,
              count(*) AS requests,
              sum(CASE WHEN routing_decision = 'teacher' THEN 1 ELSE 0 END) AS teacher_requests,
              sum(CASE WHEN routing_decision = 'student' THEN 1 ELSE 0 END) AS student_requests,
              sum(CASE WHEN routing_reason LIKE '%fallback%' THEN 1 ELSE 0 END) AS fallback_requests,
              sum(CASE WHEN status = 'error' THEN 1 ELSE 0 END) AS errors,
              round(sum(estimated_cost_usd), 8) AS estimated_cost_usd,
              round(sum(estimated_teacher_cost_usd), 8) AS estimated_teacher_cost_usd,
              round(sum(estimated_savings_usd), 8) AS estimated_savings_usd,
              round(avg(latency_ms), 2) AS avg_latency_ms,
              round(quantile_cont(latency_ms, 0.95), 2) AS p95_latency_ms
            FROM logs
            GROUP BY task_id
            ORDER BY estimated_teacher_cost_usd DESC, requests DESC
            LIMIT 50
            """,
        ),
        "by_client": rows(
            con,
            """
            SELECT
              client_id,
              count(*) AS requests,
              round(sum(estimated_cost_usd), 8) AS estimated_cost_usd,
              round(sum(estimated_teacher_cost_usd), 8) AS estimated_teacher_cost_usd,
              round(sum(estimated_savings_usd), 8) AS estimated_savings_usd,
              round(avg(latency_ms), 2) AS avg_latency_ms
            FROM logs
            GROUP BY client_id
            ORDER BY estimated_cost_usd DESC, requests DESC
            LIMIT 25
            """,
        ),
        "by_cost_center": rows(
            con,
            """
            SELECT
              cost_center,
              count(*) AS requests,
              round(sum(estimated_cost_usd), 8) AS estimated_cost_usd,
              round(sum(estimated_savings_usd), 8) AS estimated_savings_usd
            FROM logs
            GROUP BY cost_center
            ORDER BY estimated_cost_usd DESC, requests DESC
            LIMIT 25
            """,
        ),
        "routing_mix": rows(
            con,
            """
            SELECT
              routing_decision,
              count(*) AS requests,
              round(sum(estimated_cost_usd), 8) AS estimated_cost_usd
            FROM logs
            GROUP BY routing_decision
            ORDER BY requests DESC
            """,
        ),
        "daily_trend": rows(
            con,
            """
            SELECT
              substr(timestamp, 1, 10) AS day,
              count(*) AS requests,
              round(sum(estimated_cost_usd), 8) AS estimated_cost_usd,
              round(sum(estimated_savings_usd), 8) AS estimated_savings_usd,
              round(avg(latency_ms), 2) AS avg_latency_ms
            FROM logs
            WHERE length(timestamp) >= 10
            GROUP BY day
            ORDER BY day
            LIMIT 120
            """,
        ),
        "candidates": candidate_rows(con, candidate_min_requests),
    }


def candidate_rows(con: Any, min_requests: int) -> list[dict[str, Any]]:
    min_requests_literal = int(min_requests)
    return rows(
        con,
        f"""
        SELECT
          task_id,
          count(*) AS requests,
          sum(CASE WHEN routing_decision = 'teacher' THEN 1 ELSE 0 END) AS teacher_requests,
          round(sum(estimated_teacher_cost_usd), 8) AS teacher_cost_usd,
          round(sum(estimated_teacher_cost_usd) - sum(estimated_cost_usd), 8) AS current_savings_usd,
          round(avg(latency_ms), 2) AS avg_latency_ms,
          round(sum(CASE WHEN status = 'error' THEN 1 ELSE 0 END)::DOUBLE / count(*), 6) AS error_rate,
          round(
            sum(estimated_teacher_cost_usd)
            * CASE
                WHEN sum(CASE WHEN routing_decision = 'student' THEN 1 ELSE 0 END) = 0 THEN 0.70
                ELSE 0.25
              END,
            8
          ) AS estimated_remaining_savings_usd
        FROM logs
        GROUP BY task_id
        HAVING count(*) >= {min_requests_literal}
        ORDER BY estimated_remaining_savings_usd DESC, teacher_cost_usd DESC
        LIMIT 25
        """,
    )


def rows(con: Any, sql: str) -> list[dict[str, Any]]:
    result = con.execute(sql)
    headers = [column[0] for column in result.description]
    return [normalize_row(dict(zip(headers, row))) for row in result.fetchall()]


def one(con: Any, sql: str) -> dict[str, Any]:
    result = rows(con, sql)
    return result[0] if result else {}


def normalize_row(row: dict[str, Any]) -> dict[str, Any]:
    normalized: dict[str, Any] = {}
    for key, value in row.items():
        if value is None:
            normalized[key] = None
        elif isinstance(value, (bool, int, float, str)):
            normalized[key] = value
        else:
            normalized[key] = str(value)
    return normalized


def render_html(data: dict[str, Any]) -> str:
    embedded = json.dumps(data, ensure_ascii=True, sort_keys=True)
    return f"""<!doctype html>
<html lang="fr">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>DistillForge FinOps Dashboard</title>
  <style>
    :root {{
      --bg: #f5f7fa;
      --panel: #ffffff;
      --text: #17202a;
      --muted: #667085;
      --line: #d9e1ea;
      --header: #16324f;
      --blue: #2f6fbb;
      --green: #27865c;
      --red: #bf3d3d;
      --amber: #b7791f;
      --violet: #7157a8;
      --radius: 8px;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      line-height: 1.45;
    }}
    .wrap {{ max-width: 1440px; margin: 0 auto; padding: 16px; }}
    header {{
      background: var(--header);
      color: white;
      border-radius: var(--radius);
      padding: 18px 20px;
      display: flex;
      justify-content: space-between;
      gap: 16px;
      align-items: flex-end;
      flex-wrap: wrap;
    }}
    h1 {{ font-size: 22px; margin: 0 0 4px; font-weight: 700; letter-spacing: 0; }}
    h2 {{ font-size: 15px; margin: 0 0 14px; font-weight: 700; letter-spacing: 0; }}
    .sub {{ color: rgba(255,255,255,.75); font-size: 13px; }}
    .filters {{ display: flex; gap: 10px; flex-wrap: wrap; align-items: center; }}
    .filters label {{ font-size: 12px; color: rgba(255,255,255,.74); }}
    select {{
      border: 1px solid rgba(255,255,255,.3);
      background: rgba(255,255,255,.12);
      color: white;
      border-radius: 6px;
      padding: 7px 9px;
      min-width: 170px;
    }}
    option {{ color: var(--text); background: white; }}
    .kpis {{
      display: grid;
      grid-template-columns: repeat(6, minmax(150px, 1fr));
      gap: 12px;
      margin: 14px 0;
    }}
    .panel, .kpi {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: var(--radius);
      box-shadow: 0 1px 2px rgba(16, 24, 40, .04);
    }}
    .kpi {{ padding: 14px; min-height: 92px; }}
    .kpi .label {{ color: var(--muted); font-size: 12px; margin-bottom: 6px; }}
    .kpi .value {{ font-size: 25px; font-weight: 750; white-space: nowrap; }}
    .kpi .note {{ color: var(--muted); font-size: 12px; margin-top: 4px; }}
    .grid {{ display: grid; grid-template-columns: 1.1fr .9fr; gap: 14px; margin-bottom: 14px; }}
    .panel {{ padding: 16px; min-width: 0; }}
    .bars {{ display: grid; gap: 9px; }}
    .bar-row {{ display: grid; grid-template-columns: minmax(130px, 210px) 1fr auto; gap: 10px; align-items: center; font-size: 13px; }}
    .bar-label {{ overflow: hidden; text-overflow: ellipsis; white-space: nowrap; color: var(--text); }}
    .track {{ height: 13px; background: #edf2f7; border-radius: 999px; overflow: hidden; }}
    .fill {{ height: 100%; border-radius: 999px; background: var(--blue); min-width: 2px; }}
    .fill.green {{ background: var(--green); }}
    .fill.red {{ background: var(--red); }}
    .fill.amber {{ background: var(--amber); }}
    .fill.violet {{ background: var(--violet); }}
    .bar-value {{ color: var(--muted); text-align: right; font-variant-numeric: tabular-nums; }}
    .trend {{ display: flex; align-items: end; gap: 4px; height: 180px; border-bottom: 1px solid var(--line); padding-top: 8px; }}
    .trend-col {{ flex: 1; display: flex; gap: 2px; align-items: end; min-width: 3px; }}
    .trend-requests {{ width: 100%; background: var(--blue); border-radius: 4px 4px 0 0; min-height: 1px; opacity: .8; }}
    .trend-savings {{ width: 100%; background: var(--green); border-radius: 4px 4px 0 0; min-height: 1px; opacity: .8; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
    th, td {{ padding: 9px 10px; border-bottom: 1px solid #eef2f6; text-align: left; vertical-align: top; }}
    th {{ color: var(--muted); font-size: 12px; font-weight: 700; cursor: pointer; white-space: nowrap; }}
    td.num {{ text-align: right; font-variant-numeric: tabular-nums; }}
    tr:hover td {{ background: #f8fafc; }}
    .badge {{ display: inline-block; border-radius: 999px; padding: 2px 8px; background: #edf2f7; color: var(--muted); font-size: 12px; }}
    .footer {{ color: var(--muted); font-size: 12px; padding: 10px 2px; }}
    @media (max-width: 1100px) {{ .kpis {{ grid-template-columns: repeat(3, 1fr); }} .grid {{ grid-template-columns: 1fr; }} }}
    @media (max-width: 680px) {{ .wrap {{ padding: 10px; }} .kpis {{ grid-template-columns: repeat(2, 1fr); }} .bar-row {{ grid-template-columns: 1fr; gap: 4px; }} .bar-value {{ text-align: left; }} }}
  </style>
</head>
<body>
  <div class="wrap">
    <header>
      <div>
        <h1>DistillForge FinOps</h1>
        <div class="sub">Logs: <span id="sourceLogs"></span> · generation: <span id="generatedAt"></span></div>
      </div>
      <div class="filters">
        <label for="taskFilter">Tache</label>
        <select id="taskFilter"></select>
      </div>
    </header>

    <section class="kpis">
      <div class="kpi"><div class="label">Requetes</div><div class="value" id="kpiRequests">0</div><div class="note">volume filtre</div></div>
      <div class="kpi"><div class="label">Cout estime</div><div class="value" id="kpiCost">$0</div><div class="note">cout reel/proxy</div></div>
      <div class="kpi"><div class="label">Cout teacher</div><div class="value" id="kpiTeacherCost">$0</div><div class="note">baseline</div></div>
      <div class="kpi"><div class="label">Economies</div><div class="value" id="kpiSavings">$0</div><div class="note">cumulees</div></div>
      <div class="kpi"><div class="label">p95 latence</div><div class="value" id="kpiP95">0ms</div><div class="note">proxy observe</div></div>
      <div class="kpi"><div class="label">Erreur</div><div class="value" id="kpiErrors">0%</div><div class="note">taux global</div></div>
    </section>

    <section class="grid">
      <div class="panel"><h2>Cout teacher par tache</h2><div class="bars" id="taskCostBars"></div></div>
      <div class="panel"><h2>Mix de routage</h2><div class="bars" id="routingBars"></div></div>
    </section>

    <section class="grid">
      <div class="panel"><h2>Tendance quotidienne</h2><div class="trend" id="dailyTrend"></div></div>
      <div class="panel"><h2>Cout par centre</h2><div class="bars" id="costCenterBars"></div></div>
    </section>

    <section class="panel" style="margin-bottom:14px">
      <h2>Taches candidates a distillation</h2>
      <div id="candidatesTable"></div>
    </section>

    <section class="panel">
      <h2>Detail par tache</h2>
      <div id="taskTable"></div>
    </section>

    <div class="footer">Dashboard autonome genere depuis les logs JSONL DistillForge.</div>
  </div>
  <script>
    const DATA = {embedded};
    let currentTask = "all";
    let sortState = {{}};

    const $ = (id) => document.getElementById(id);

    function init() {{
      $("sourceLogs").textContent = DATA.source_logs;
      $("generatedAt").textContent = DATA.generated_at;
      const select = $("taskFilter");
      select.innerHTML = '<option value="all">Toutes les taches</option>' +
        DATA.by_task.map(row => `<option value="${{escapeAttr(row.task_id)}}">${{escapeHtml(row.task_id)}}</option>`).join("");
      select.addEventListener("change", () => {{ currentTask = select.value; render(); }});
      render();
    }}

    function filteredTaskRows() {{
      return currentTask === "all" ? DATA.by_task : DATA.by_task.filter(row => row.task_id === currentTask);
    }}

    function filteredData() {{
      const taskRows = filteredTaskRows();
      if (currentTask === "all") return {{
        summary: DATA.summary,
        by_task: taskRows,
        candidates: DATA.candidates,
      }};
      const row = taskRows[0] || {{}};
      const requests = Number(row.requests || 0);
      const errors = Number(row.errors || 0);
      return {{
        summary: {{
          requests,
          teacher_requests: row.teacher_requests || 0,
          student_requests: row.student_requests || 0,
          fallback_requests: row.fallback_requests || 0,
          errors,
          estimated_cost_usd: row.estimated_cost_usd || 0,
          estimated_teacher_cost_usd: row.estimated_teacher_cost_usd || 0,
          estimated_savings_usd: row.estimated_savings_usd || 0,
          p95_latency_ms: row.p95_latency_ms || 0,
          error_rate: requests ? errors / requests : 0,
        }},
        by_task: taskRows,
        candidates: DATA.candidates.filter(row => row.task_id === currentTask),
      }};
    }}

    function render() {{
      const view = filteredData();
      renderKpis(view.summary);
      renderBars("taskCostBars", view.by_task.slice(0, 12), "task_id", "estimated_teacher_cost_usd", "currency", "blue");
      renderBars("routingBars", DATA.routing_mix, "routing_decision", "requests", "number", "violet");
      renderBars("costCenterBars", DATA.by_cost_center, "cost_center", "estimated_cost_usd", "currency", "amber");
      renderTrend(DATA.daily_trend);
      renderTable("candidatesTable", view.candidates, [
        ["task_id", "Tache"],
        ["requests", "Req.", "number"],
        ["teacher_requests", "Teacher", "number"],
        ["teacher_cost_usd", "Cout teacher", "currency"],
        ["estimated_remaining_savings_usd", "Potentiel", "currency"],
        ["error_rate", "Erreur", "percent"],
        ["avg_latency_ms", "Lat. moy.", "ms"],
      ], "estimated_remaining_savings_usd");
      renderTable("taskTable", view.by_task, [
        ["task_id", "Tache"],
        ["requests", "Req.", "number"],
        ["teacher_requests", "Teacher", "number"],
        ["student_requests", "Student", "number"],
        ["fallback_requests", "Fallback", "number"],
        ["estimated_cost_usd", "Cout", "currency"],
        ["estimated_savings_usd", "Economies", "currency"],
        ["p95_latency_ms", "p95", "ms"],
        ["errors", "Erreurs", "number"],
      ], "estimated_teacher_cost_usd");
    }}

    function renderKpis(summary) {{
      $("kpiRequests").textContent = formatNumber(summary.requests || 0);
      $("kpiCost").textContent = formatCurrency(summary.estimated_cost_usd || 0);
      $("kpiTeacherCost").textContent = formatCurrency(summary.estimated_teacher_cost_usd || 0);
      $("kpiSavings").textContent = formatCurrency(summary.estimated_savings_usd || 0);
      $("kpiP95").textContent = `${{formatNumber(summary.p95_latency_ms || 0)}}ms`;
      $("kpiErrors").textContent = formatPercent(summary.error_rate || 0);
    }}

    function renderBars(id, rows, labelField, valueField, format, color) {{
      const max = Math.max(...rows.map(row => Number(row[valueField] || 0)), 0);
      const html = rows.length ? rows.map(row => {{
        const value = Number(row[valueField] || 0);
        const width = max > 0 ? Math.max(2, value / max * 100) : 0;
        return `<div class="bar-row"><div class="bar-label" title="${{escapeAttr(row[labelField])}}">${{escapeHtml(row[labelField])}}</div><div class="track"><div class="fill ${{color}}" style="width:${{width}}%"></div></div><div class="bar-value">${{formatValue(value, format)}}</div></div>`;
      }}).join("") : '<span class="badge">Aucune donnee</span>';
      $(id).innerHTML = html;
    }}

    function renderTrend(rows) {{
      const maxReq = Math.max(...rows.map(row => Number(row.requests || 0)), 0);
      const maxSavings = Math.max(...rows.map(row => Number(row.estimated_savings_usd || 0)), 0);
      $("dailyTrend").innerHTML = rows.length ? rows.map(row => {{
        const reqH = maxReq ? Math.max(1, Number(row.requests || 0) / maxReq * 170) : 1;
        const savH = maxSavings ? Math.max(1, Number(row.estimated_savings_usd || 0) / maxSavings * 170) : 1;
        return `<div class="trend-col" title="${{escapeAttr(row.day)}} · ${{formatNumber(row.requests)}} req · ${{formatCurrency(row.estimated_savings_usd || 0)}} saved"><div class="trend-requests" style="height:${{reqH}}px"></div><div class="trend-savings" style="height:${{savH}}px"></div></div>`;
      }}).join("") : '<span class="badge">Aucune donnee horodatee</span>';
    }}

    function renderTable(id, rows, columns, defaultSort) {{
      const state = sortState[id] || {{ field: defaultSort, dir: "desc" }};
      sortState[id] = state;
      const sorted = [...rows].sort((a, b) => compare(a[state.field], b[state.field]) * (state.dir === "asc" ? 1 : -1));
      const head = columns.map(col => `<th data-field="${{escapeAttr(col[0])}}">${{escapeHtml(col[1])}}${{state.field === col[0] ? (state.dir === "asc" ? " ▲" : " ▼") : ""}}</th>`).join("");
      const body = sorted.slice(0, 50).map(row => `<tr>${{columns.map(col => `<td class="${{col[2] ? "num" : ""}}">${{formatValue(row[col[0]], col[2])}}</td>`).join("")}}</tr>`).join("");
      $(id).innerHTML = rows.length ? `<table><thead><tr>${{head}}</tr></thead><tbody>${{body}}</tbody></table>` : '<span class="badge">Aucune donnee</span>';
      $(id).querySelectorAll("th").forEach(th => th.addEventListener("click", () => {{
        const field = th.dataset.field;
        if (state.field === field) state.dir = state.dir === "asc" ? "desc" : "asc";
        else {{ state.field = field; state.dir = "desc"; }}
        render();
      }}));
    }}

    function compare(a, b) {{
      const an = Number(a), bn = Number(b);
      if (!Number.isNaN(an) && !Number.isNaN(bn)) return an - bn;
      return String(a || "").localeCompare(String(b || ""));
    }}

    function formatValue(value, format) {{
      if (format === "currency") return formatCurrency(Number(value || 0));
      if (format === "percent") return formatPercent(Number(value || 0));
      if (format === "number") return formatNumber(Number(value || 0));
      if (format === "ms") return `${{formatNumber(Number(value || 0))}}ms`;
      return escapeHtml(value ?? "");
    }}
    function formatCurrency(value) {{
      if (Math.abs(value) >= 1000) return "$" + value.toLocaleString(undefined, {{ maximumFractionDigits: 0 }});
      if (Math.abs(value) >= 1) return "$" + value.toLocaleString(undefined, {{ maximumFractionDigits: 2 }});
      return "$" + value.toFixed(6).replace(/0+$/, "").replace(/\\.$/, "");
    }}
    function formatPercent(value) {{ return (value * 100).toFixed(1) + "%"; }}
    function formatNumber(value) {{ return Number(value || 0).toLocaleString(undefined, {{ maximumFractionDigits: 1 }}); }}
    function escapeHtml(value) {{
      return String(value).replace(/[&<>"']/g, char => ({{"&":"&amp;","<":"&lt;",">":"&gt;","\\"":"&quot;","'":"&#39;"}}[char]));
    }}
    function escapeAttr(value) {{ return escapeHtml(value).replace(/`/g, "&#96;"); }}
    init();
  </script>
</body>
</html>
"""


def sql_string_literal(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


if __name__ == "__main__":
    raise SystemExit(main())
