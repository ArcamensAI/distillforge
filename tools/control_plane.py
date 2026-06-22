#!/usr/bin/env python3
"""Run a minimal local DistillForge control plane HTTP API."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Optional
from urllib.parse import parse_qs, urlparse


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
SUPPORTED_PROMOTION_MODES = {"teacher_only", "shadow", "canary", "student_only", "bandit"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8090)
    args = parser.parse_args()

    server = ThreadingHTTPServer((args.host, args.port), ControlPlaneHandler)
    print(f"DistillForge control plane listening on http://{args.host}:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        return 130
    return 0


class ControlPlaneHandler(BaseHTTPRequestHandler):
    server_version = "DistillForgeControlPlane/0.1"

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        segments = path_segments(parsed.path)
        query = parse_query(parsed.query)

        if parsed.path == "/health":
            self.write_json(HTTPStatus.OK, {"status": "ok"})
            return

        if segments == ["admin", "models"]:
            self.write_json(
                HTTPStatus.OK,
                {"models": list_models(path_arg(query, "models_root", default_models_root()))},
            )
            return

        if len(segments) == 3 and segments[:2] == ["admin", "models"]:
            model = find_model(segments[2], path_arg(query, "models_root", default_models_root()))
            if model is None:
                self.write_json(HTTPStatus.NOT_FOUND, {"error": "model_not_found"})
                return
            self.write_json(HTTPStatus.OK, model)
            return

        if len(segments) == 4 and segments[:2] == ["admin", "tasks"] and segments[3] == "status":
            self.write_json(
                HTTPStatus.OK,
                task_status(
                    segments[2],
                    path_arg(query, "snapshot", default_snapshot_path()),
                    path_arg(query, "datasets_root", default_datasets_root()),
                    path_arg(query, "models_root", default_models_root()),
                    query.get("logs", default_logs_path()),
                ),
            )
            return

        self.write_json(HTTPStatus.NOT_FOUND, {"error": "not_found"})

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        segments = path_segments(parsed.path)
        query = parse_query(parsed.query)
        payload = self.read_json_body()
        if payload is None:
            return

        if len(segments) != 4 or segments[:2] != ["admin", "tasks"]:
            self.write_json(HTTPStatus.NOT_FOUND, {"error": "not_found"})
            return

        task_id = segments[2]
        action = segments[3]
        try:
            if action == "train":
                self.write_json(HTTPStatus.ACCEPTED, train_task(task_id, payload, query))
                return
            if action == "evaluate":
                self.write_json(HTTPStatus.OK, evaluate_model(task_id, payload, query))
                return
            if action == "promote":
                self.write_json(HTTPStatus.OK, promote_task(task_id, payload, query))
                return
            if action == "rollback":
                self.write_json(HTTPStatus.OK, rollback_task(task_id, payload, query))
                return
        except CommandError as exc:
            self.write_json(
                HTTPStatus.BAD_REQUEST,
                {
                    "error": "command_failed",
                    "command": exc.command,
                    "returncode": exc.returncode,
                    "stdout": exc.stdout,
                    "stderr": exc.stderr,
                },
            )
            return
        except ValueError as exc:
            self.write_json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
            return

        self.write_json(HTTPStatus.NOT_FOUND, {"error": "not_found"})

    def read_json_body(self) -> Optional[dict[str, Any]]:
        length = int(self.headers.get("Content-Length", "0") or "0")
        if length == 0:
            return {}
        try:
            body = self.rfile.read(length)
            value = json.loads(body)
        except json.JSONDecodeError:
            self.write_json(HTTPStatus.BAD_REQUEST, {"error": "invalid_json"})
            return None
        if not isinstance(value, dict):
            self.write_json(HTTPStatus.BAD_REQUEST, {"error": "json_body_must_be_object"})
            return None
        return value

    def write_json(self, status: HTTPStatus, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, indent=2, sort_keys=True).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt: str, *args: Any) -> None:
        sys.stderr.write("%s - %s\n" % (self.log_date_time_string(), fmt % args))


def train_task(task_id: str, payload: dict[str, Any], query: dict[str, str]) -> dict[str, Any]:
    datasets_root = path_arg(query | payload, "datasets_root", default_datasets_root())
    models_root = path_arg(query | payload, "models_root", default_models_root())
    logs = str_arg(query | payload, "logs", default_logs_path())
    dataset_id = str_arg(payload, "dataset_id", "")
    min_samples = int_arg(payload, "min_samples", 1)
    min_train_samples = int_arg(payload, "min_train_samples", 1)
    model_id = str_arg(payload, "model_id", "")

    build_args = [
        sys.executable,
        str(TOOLS / "build_dataset.py"),
        "--task-id",
        task_id,
        "--logs",
        logs,
        "--out",
        str(datasets_root),
        "--min-samples",
        str(min_samples),
    ]
    if dataset_id:
        build_args.extend(["--dataset-id", dataset_id])
    build_result = run_command(build_args)

    dataset_dir = latest_child_dir(datasets_root / task_id)
    train_args = [
        sys.executable,
        str(TOOLS / "train_student.py"),
        "--dataset",
        str(dataset_dir),
        "--out",
        str(models_root),
        "--min-train-samples",
        str(min_train_samples),
    ]
    if model_id:
        train_args.extend(["--model-id", model_id])
    train_result = run_command(train_args)
    model_dir = latest_child_dir(models_root / task_id)

    append_control_event(
        path_arg(query | payload, "registry", default_registry_path()),
        {
            "event": "control_plane_train",
            "task_id": task_id,
            "dataset_dir": str(dataset_dir),
            "model_dir": str(model_dir),
        },
    )

    return {
        "status": "completed",
        "task_id": task_id,
        "dataset_dir": str(dataset_dir),
        "model_dir": str(model_dir),
        "build_dataset": build_result,
        "train_student": train_result,
        "model": read_json(model_dir / "manifest.json"),
        "eval_report": read_json(model_dir / "eval_report.json"),
    }


def evaluate_model(task_id: str, payload: dict[str, Any], query: dict[str, str]) -> dict[str, Any]:
    model_dir = model_dir_arg(task_id, payload, query)
    manifest = read_json(model_dir / "manifest.json")
    eval_report = read_json(model_dir / "eval_report.json")
    min_accuracy = float_arg(payload, "min_accuracy", 0.0)
    min_macro_f1 = float_arg(payload, "min_macro_f1", 0.0)
    metrics = manifest.get("metrics", {})
    accuracy = float(metrics.get("accuracy", 0.0))
    macro_f1 = float(metrics.get("macro_f1", 0.0))
    passed = accuracy >= min_accuracy and macro_f1 >= min_macro_f1
    return {
        "status": "passed" if passed else "failed",
        "task_id": task_id,
        "model_dir": str(model_dir),
        "thresholds": {"min_accuracy": min_accuracy, "min_macro_f1": min_macro_f1},
        "metrics": {"accuracy": accuracy, "macro_f1": macro_f1},
        "model": manifest,
        "eval_report": eval_report,
    }


def promote_task(task_id: str, payload: dict[str, Any], query: dict[str, str]) -> dict[str, Any]:
    mode = str_arg(payload, "mode", "shadow")
    if mode not in SUPPORTED_PROMOTION_MODES - {"teacher_only"}:
        raise ValueError(f"unsupported promotion mode: {mode}")
    model_dir = model_dir_arg(task_id, payload, query)
    command = [
        sys.executable,
        str(TOOLS / "promote_model.py"),
        "--task-id",
        task_id,
        "--model-dir",
        str(model_dir),
        "--mode",
        mode,
        "--snapshot",
        str(path_arg(query | payload, "snapshot", default_snapshot_path())),
        "--registry",
        str(path_arg(query | payload, "registry", default_registry_path())),
        "--student-traffic-percentage",
        str(int_arg(payload, "student_traffic_percentage", 1)),
        "--teacher-probe-percentage",
        str(int_arg(payload, "teacher_probe_percentage", 2)),
        "--min-accuracy",
        str(float_arg(payload, "min_accuracy", 0.0)),
        "--min-macro-f1",
        str(float_arg(payload, "min_macro_f1", 0.0)),
    ]
    result = run_command(command)
    return {"status": "completed", "task_id": task_id, "mode": mode, "promote_model": result}


def rollback_task(task_id: str, payload: dict[str, Any], query: dict[str, str]) -> dict[str, Any]:
    command = [
        sys.executable,
        str(TOOLS / "promote_model.py"),
        "--task-id",
        task_id,
        "--mode",
        "teacher_only",
        "--snapshot",
        str(path_arg(query | payload, "snapshot", default_snapshot_path())),
        "--registry",
        str(path_arg(query | payload, "registry", default_registry_path())),
    ]
    result = run_command(command)
    return {"status": "completed", "task_id": task_id, "mode": "teacher_only", "rollback": result}


def task_status(
    task_id: str,
    snapshot_path: Path,
    datasets_root: Path,
    models_root: Path,
    logs: str,
) -> dict[str, Any]:
    snapshot = read_json(snapshot_path) if snapshot_path.exists() else {}
    route = snapshot.get("tasks", {}).get(task_id, {"mode": snapshot.get("default_mode", "teacher_only")})
    return {
        "task_id": task_id,
        "route": route,
        "snapshot_version": snapshot.get("version"),
        "datasets": list_dataset_manifests(datasets_root / task_id),
        "models": [model for model in list_models(models_root) if model.get("task_id") == task_id],
        "recent_logs": log_summary(task_id, logs),
    }


def list_dataset_manifests(task_root: Path) -> list[dict[str, Any]]:
    manifests = []
    if not task_root.exists():
        return manifests
    for manifest_path in sorted(task_root.glob("*/manifest.json")):
        manifest = read_json(manifest_path)
        manifest["path"] = str(manifest_path.parent)
        manifests.append(manifest)
    return manifests


def list_models(models_root: Path) -> list[dict[str, Any]]:
    models = []
    if not models_root.exists():
        return models
    for manifest_path in sorted(models_root.glob("*/*/manifest.json")):
        manifest = read_json(manifest_path)
        manifest["path"] = str(manifest_path.parent)
        models.append(manifest)
    return models


def find_model(model_id: str, models_root: Path) -> Optional[dict[str, Any]]:
    for model in list_models(models_root):
        if model.get("model_id") == model_id:
            return model
    return None


def log_summary(task_id: str, logs: str) -> dict[str, Any]:
    try:
        import duckdb
    except ImportError:
        return {"available": False, "reason": "duckdb_not_installed"}

    con = duckdb.connect(":memory:")
    try:
        con.execute(
            f"""
            SELECT
              count(*) AS requests,
              sum(CASE WHEN status = 'error' THEN 1 ELSE 0 END) AS errors,
              sum(coalesce(estimated_teacher_cost_usd, 0)) AS teacher_cost_usd,
              sum(coalesce(estimated_savings_usd, 0)) AS estimated_savings_usd,
              max(timestamp) AS last_seen
            FROM read_json_auto({sql_string_literal(logs)}, format='newline_delimited', union_by_name=true)
            WHERE task_id = {sql_string_literal(task_id)}
            """
        )
        row = con.fetchone()
    except Exception as exc:
        return {"available": False, "reason": str(exc)}
    return {
        "available": True,
        "requests": int(row[0] or 0),
        "errors": int(row[1] or 0),
        "teacher_cost_usd": float(row[2] or 0.0),
        "estimated_savings_usd": float(row[3] or 0.0),
        "last_seen": str(row[4]) if row[4] is not None else None,
    }


class CommandError(Exception):
    def __init__(self, command: list[str], returncode: int, stdout: str, stderr: str) -> None:
        super().__init__("command failed")
        self.command = command
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


def run_command(command: list[str]) -> dict[str, Any]:
    completed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, check=False)
    result = {
        "command": command,
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }
    if completed.returncode != 0:
        raise CommandError(command, completed.returncode, completed.stdout, completed.stderr)
    return result


def append_control_event(path: Path, payload: dict[str, Any]) -> None:
    event = {"timestamp": datetime.now(timezone.utc).isoformat(), **payload}
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(json.dumps(event, sort_keys=True) + "\n")


def model_dir_arg(task_id: str, payload: dict[str, Any], query: dict[str, str]) -> Path:
    if "model_dir" in payload:
        return Path(payload["model_dir"])
    if "model_dir" in query:
        return Path(query["model_dir"])
    model_id = str_arg(query | payload, "model_id", "")
    models_root = path_arg(query | payload, "models_root", default_models_root())
    if model_id:
        return models_root / task_id / model_id
    return latest_child_dir(models_root / task_id)


def latest_child_dir(path: Path) -> Path:
    if not path.exists():
        raise ValueError(f"missing directory: {path}")
    children = [child for child in path.iterdir() if child.is_dir()]
    if not children:
        raise ValueError(f"no child directories found in {path}")
    return max(children, key=lambda child: child.stat().st_mtime)


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise ValueError(f"missing file: {path}")
    return json.loads(path.read_text())


def path_segments(path: str) -> list[str]:
    return [segment for segment in path.split("/") if segment]


def parse_query(query: str) -> dict[str, str]:
    parsed = parse_qs(query, keep_blank_values=False)
    return {key: values[-1] for key, values in parsed.items() if values}


def path_arg(values: dict[str, Any], name: str, default: str) -> Path:
    return Path(str_arg(values, name, default))


def str_arg(values: dict[str, Any], name: str, default: str) -> str:
    value = values.get(name, default)
    if value is None:
        return default
    return str(value)


def int_arg(values: dict[str, Any], name: str, default: int) -> int:
    return int(values.get(name, default))


def float_arg(values: dict[str, Any], name: str, default: float) -> float:
    return float(values.get(name, default))


def default_logs_path() -> str:
    return os.environ.get("DISTILLFORGE_LOGS", "data/logs/*.jsonl")


def default_datasets_root() -> str:
    return os.environ.get("DISTILLFORGE_DATASETS_ROOT", "data/datasets")


def default_models_root() -> str:
    return os.environ.get("DISTILLFORGE_MODELS_ROOT", "models")


def default_snapshot_path() -> str:
    return os.environ.get("DISTILLFORGE_ROUTING_SNAPSHOT", "config/routing_snapshot.json")


def default_registry_path() -> str:
    return os.environ.get("DISTILLFORGE_REGISTRY", "registry/events.jsonl")


def sql_string_literal(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


if __name__ == "__main__":
    raise SystemExit(main())
