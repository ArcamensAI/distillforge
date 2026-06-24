# DistillForge

*Read this in other languages: [Français](README.md)*

FinOps proxy for LLMs.

## Documentation

- [Functional specifications](specs.md)
- [Technical architecture](ARCHITECTURE.md)
- [Groq + BANKING77 demo](examples/groq_banking77/README.md)

## V1 proxy

The first DistillForge implementation targets the `teacher_only` mode:

- Rust proxy based on Pingora;
- proxied endpoints: `POST /v1/chat/completions` and `POST /v1/completions`;
- validation of the `X-Client-ID` and `X-Task-ID` headers according to the configuration;
- JSONL logs;
- internal counters prepared for `/metrics`.

Run locally:

```sh
cargo run
```

By default the proxy listens on `127.0.0.1:6188` and forwards to the teacher
configured in `config/example.yaml`.

## Local routing

Routing is driven by `config/routing_snapshot.json`. The snapshot is loaded
at startup and can be reloaded without restarting the proxy:

```sh
curl -X POST http://127.0.0.1:6188/admin/reload-routing
```

Modes supported by the dataplane:

- `teacher_only`
- `shadow`
- `student_only`
- `canary`
- `bandit`

If a snapshot points to a student model absent from `config/example.yaml`,
DistillForge falls back to the teacher and logs
`student_backend_missing_teacher_fallback`.

If a `student` route is selected but the connection to the student backend
fails, DistillForge automatically retries the request on the teacher and logs
`student_connect_error_teacher_fallback`. The Prometheus counter
`distillforge_fallback_requests_total` exposes these fallbacks.

Upstream timeouts are configurable in `config/example.yaml`:

```yaml
timeouts:
  upstream_connection_timeout_ms: 2000
  teacher_inference_timeout_ms: 30000
  student_inference_timeout_ms: 2000
  upstream_write_timeout_ms: 30000
  shadow_student_timeout_ms: 5000
```

## Local rate limiting

DistillForge can apply in-memory rate limiting, without a third-party service,
before calling the LLM backends. The default limit applies per
`X-Client-ID`; specific limits can be set per client or per task:

```yaml
rate_limits:
  enabled: true
  window_ms: 60000
  default_requests_per_window: 120
  clients:
    crm_backend: 600
  tasks:
    email_classification_v1: 300
```

A request above the threshold is rejected with `429` and the reason
`rate_limited_client` or `rate_limited_task` in the logs. The Prometheus
counter `distillforge_rate_limited_requests_total` exposes these rejections.

## Minimal student worker

A test HTTP worker makes it possible to validate student routing without an ML
model:

```sh
cargo run --bin student_worker
```

By default it listens on `127.0.0.1:9100`, exposes `/health`, `/infer`,
`/v1/chat/completions` and `/v1/completions`, and returns a deterministic
response configurable via environment variables:

```sh
DISTILLFORGE_STUDENT_RESPONSE="student ok" cargo run --bin student_worker
```

## FinOps report

JSONL logs can be aggregated with DuckDB:

```sh
python3 -m pip install -r requirements-analytics.txt
python3 tools/finops_report.py --logs 'data/logs/*.jsonl'
```

The report shows volumes, routing decisions, latencies, estimated costs,
estimated savings and tasks that are candidates for distillation.

A standalone HTML dashboard can also be generated without a server:

```sh
python3 tools/finops_dashboard.py --logs 'data/logs/*.jsonl'
```

The `reports/distillforge_dashboard.html` file shows the FinOps KPIs,
the routing mix, the daily trend, costs per center and tasks that are
candidates for distillation.

## Dataset builder

Eligible redacted logs can be converted into a versioned dataset:

```sh
python3 tools/build_dataset.py --task-id test_task --logs 'data/logs/*.jsonl'
```

The builder filters `success` requests with `training_eligible=true`,
deduplicates by `input_hash`, and produces `train.parquet`, `validation.parquet`,
`test.parquet` and a `manifest.json`.

For an OpenAI-compatible teacher that returns a chat completion JSON, the
label can be extracted from `choices[0].message.content`:

```sh
python3 tools/build_dataset.py \
  --task-id test_task \
  --logs 'data/logs/*.jsonl' \
  --target-field openai_message_content
```

A dataset can then be augmented with local synthetic examples,
without calling a third-party platform:

```sh
python3 tools/synthetic_data.py \
  --dataset data/datasets/test_task/ds_example \
  --multiplier 2 \
  --max-synthetic-ratio 0.8
```

The V1 generator uses deterministic templates, modifies only the `train`
split, copies the labels from `validated_output` and annotates the manifest
with the resulting synthetic ratio.

## Minimal offline training

A classic student model can be trained from a versioned dataset:

```sh
python3 -m pip install -r requirements-training.txt
python3 tools/train_student.py --dataset data/datasets/test_task/ds_example
```

The trainer produces `model.joblib`, `eval_report.json` and `manifest.json`
under `models/{task_id}/{model_id}`. It uses TF-IDF + LogisticRegression when
the dataset contains several classes, otherwise a `DummyClassifier` baseline.

The trained model can be served over HTTP:

```sh
python3 tools/student_inference.py --model-dir models/test_task/student_example
```

This worker exposes `/health`, `/infer`, `/v1/chat/completions` and
`/v1/completions`, which makes it possible to declare it as a `students`
backend in `config/example.yaml`.

## Local control plane

A minimal admin HTTP server orchestrates the V1 scripts without any external
platform dependency:

```sh
python3 tools/control_plane.py --host 127.0.0.1 --port 8090
```

Main endpoints:

- `GET /health`
- `GET /admin/tasks/{task_id}/status`
- `GET /admin/models`
- `GET /admin/models/{model_id}`
- `POST /admin/tasks/{task_id}/train`
- `POST /admin/tasks/{task_id}/evaluate`
- `POST /admin/tasks/{task_id}/promote`
- `POST /admin/tasks/{task_id}/rollback`

Example of a shadow promotion via the control plane:

```sh
curl -X POST http://127.0.0.1:8090/admin/tasks/test_task/promote \
  -H 'Content-Type: application/json' \
  -d '{"model_dir":"models/test_task/student_example","mode":"shadow","min_accuracy":0.95}'
```

## Promotion and rollback

A validated model can update the routing snapshot:

```sh
python3 tools/promote_model.py \
  --model-dir models/test_task/student_example \
  --mode canary \
  --student-traffic-percentage 10 \
  --min-accuracy 0.95
curl -X POST http://127.0.0.1:6188/admin/reload-routing
```

The `shadow` mode serves the client with the teacher and sends a bounded copy
of the request to the student in the background:

```sh
python3 tools/promote_model.py \
  --model-dir models/test_task/student_example \
  --mode shadow \
  --min-accuracy 0.95
```

Shadow probes are logged in `logging.shadow_path` with latency,
HTTP status, redacted student response and a `response_exact_match` indicator.
A divergence report can be produced with:

```sh
python3 tools/shadow_report.py --logs data/logs/shadow.jsonl
```

The `bandit` mode serves the student by default while keeping a deterministic
percentage of teacher probes to monitor quality:

```sh
python3 tools/promote_model.py \
  --model-dir models/test_task/student_example \
  --mode bandit \
  --teacher-probe-percentage 2 \
  --min-accuracy 0.95
```

Teacher rollback for a task:

```sh
python3 tools/promote_model.py --task-id test_task --mode teacher_only
curl -X POST http://127.0.0.1:6188/admin/reload-routing
```

Each promotion appends a JSONL event to `registry/events.jsonl`.

## Drift guard

A local safeguard can analyze recent logs and detect simple drift
on the `canary` and `student_only` routes: student error rate, p95 latency
and negative human feedback.

Audit without modification:

```sh
python3 tools/drift_guard.py \
  --logs 'data/logs/*.jsonl' \
  --feedback data/logs/feedback.jsonl \
  --window-hours 24
```

Automatic rollback of drifting tasks to `teacher_only`:

```sh
python3 tools/drift_guard.py \
  --logs 'data/logs/*.jsonl' \
  --feedback data/logs/feedback.jsonl \
  --max-error-rate 0.05 \
  --max-negative-feedback-rate 0.20 \
  --apply
curl -X POST http://127.0.0.1:6188/admin/reload-routing
```

Each rollback writes a `drift_guard_rollback` event to
`registry/events.jsonl`.

## Redacted logs

In `redacted` mode, DistillForge captures at most
`logging.max_capture_bytes` bytes of the request and response bodies, then
writes `prompt_redacted` and `response_redacted` to the JSONL. Emails,
bearer tokens, `sk-*` keys and common JSON fields such as `password`, `token`,
`secret`, `api_key` and `authorization` are masked.

In `metadata_only` mode, no content or input hash is stored.

## Local retention

V1 retention is applied by a local tool, in dry-run by default:

```sh
python3 tools/retention.py
```

Default thresholds:

- proxy logs: 90 days;
- shadow logs: 90 days;
- feedback: 365 days;
- datasets: 365 days;
- registry/audit: 1095 days.

Effective application:

```sh
python3 tools/retention.py --apply
```

## Deployment

Container images:

```sh
docker build -f Dockerfile.proxy -t distillforge/proxy:0.1.0 .
docker build -f Dockerfile.control-plane -t distillforge/control-plane:0.1.0 .
```

Static Kubernetes manifests:

```sh
kubectl apply -f deploy/kubernetes/distillforge.yaml
```

The manifest creates a `distillforge` namespace, a PVC for the logs/datasets/
models/registry, two proxy replicas, a local control plane and the associated
HTTP services. The routing snapshot is initialized from the ConfigMap then
stored in `/data/config/routing_snapshot.json` to allow promotions
and rollbacks.

## Human feedback

Clients can send a correction tied to a request:

```sh
curl -X POST http://127.0.0.1:6188/v1/feedback \
  -H 'Content-Type: application/json' \
  -H 'X-Client-ID: crm_backend' \
  -H 'X-Task-ID: email_classification_v1' \
  -d '{"request_id":"req_123","rating":"bad","correct_output":"billing_issue","comment":"Wrong class"}'
```

Feedback is written to `logging.feedback_path` in JSONL format, with
redaction of sensitive text fields.
