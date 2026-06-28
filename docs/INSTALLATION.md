# DistillForge Installation

This guide installs DistillForge for local development and offline demos.
DistillForge does not require a managed platform: the proxy is Rust, the
control/training tooling is Python, logs are JSONL, and analytics use DuckDB.

## Prerequisites

- macOS or Linux.
- Rust stable with Cargo.
- Python 3.9+.
- Docker, optional, for container builds.
- `curl`, useful for smoke tests.

Recommended on macOS:

```sh
xcode-select --install
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

Check the toolchain:

```sh
rustc --version
cargo --version
python3 --version
```

## Clone

```sh
git clone git@github.com:ArcamensAI/distillforge.git
cd distillforge
```

If you are using another remote, the rest of the commands are unchanged.

## Python Environment

Create an isolated virtual environment:

```sh
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
```

Install the minimal analytics dependency:

```sh
python3 -m pip install -r requirements-analytics.txt
```

Install training and local inference dependencies when you need student
training, BGE embeddings, or dataset generation:

```sh
python3 -m pip install -r requirements-training.txt
```

For Apple Silicon local LLM teachers based on MLX, install `mlx-lm` as an
optional dependency:

```sh
python3 -m pip install mlx-lm
```

## Rust Build

Build and test the proxy:

```sh
cargo build
cargo test
```

Run the default proxy:

```sh
cargo run --bin distillforge
```

By default the proxy reads `config/example.yaml`, listens on
`127.0.0.1:6188`, and exposes:

- `POST /v1/chat/completions`
- `POST /v1/completions`
- `POST /admin/reload-routing`
- `GET /metrics`

Use a specific configuration with:

```sh
DISTILLFORGE_CONFIG=examples/cfpb_complaints/config.local_llm.yaml \
cargo run --bin distillforge
```

## Smoke Test

In another terminal, start the deterministic student worker:

```sh
DISTILLFORGE_STUDENT_RESPONSE="student ok" cargo run --bin student_worker
```

Then call the proxy:

```sh
curl -sS http://127.0.0.1:6188/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -H 'X-Client-ID: demo_client' \
  -H 'X-Task-ID: test_task' \
  -d '{"model":"teacher","messages":[{"role":"user","content":"hello"}]}'
```

The exact response depends on the active routing snapshot and configured
teacher/student backends.

## Docker

Build the proxy image:

```sh
docker build -f Dockerfile.proxy -t distillforge-proxy .
```

Run it with the bundled example configuration:

```sh
docker run --rm -p 6188:6188 distillforge-proxy
```

Build the control-plane image:

```sh
docker build -f Dockerfile.control-plane -t distillforge-control-plane .
```

## Analytics

Generate a FinOps report from JSONL logs:

```sh
python3 tools/finops_report.py --logs 'data/logs/*.jsonl'
```

Generate a standalone HTML dashboard:

```sh
python3 tools/finops_dashboard.py --logs 'data/logs/*.jsonl'
```

## Student Training

Build a dataset from eligible logs:

```sh
python3 tools/build_dataset.py \
  --task-id test_task \
  --logs 'data/logs/*.jsonl' \
  --target-field openai_message_content
```

Train the default TF-IDF student:

```sh
python3 tools/train_student.py \
  --dataset data/datasets/test_task/ds_example \
  --student-kind tfidf
```

Train a BGE-based local neural student:

```sh
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
python3 tools/train_student.py \
  --dataset examples/cfpb_complaints/data_llm/datasets/cfpb_product_triage_v1/ds_cfpb_product_llm_2000 \
  --out examples/cfpb_complaints/models \
  --model-id cfpb_product_student_bge_m3_hybrid_mlp_2000 \
  --min-train-samples 1000 \
  --student-kind hybrid_embedding_tfidf_mlp \
  --embedding-model BAAI/bge-m3 \
  --input-format openai_user_content \
  --batch-size 4 \
  --max-input-chars 1800 \
  --embedding-max-seq-length 256
```

The `HF_HUB_OFFLINE` and `TRANSFORMERS_OFFLINE` flags require the model to be
already present in the local Hugging Face cache. Remove them only when you
intend to download a model.

Serve a trained student:

```sh
python3 tools/student_inference.py \
  --model-dir examples/cfpb_complaints/models/cfpb_product_triage_v1/cfpb_product_student_bge_m3_hybrid_mlp_2000 \
  --listen 127.0.0.1:9102
```

## Configuration

The active proxy configuration is selected with `DISTILLFORGE_CONFIG`.
Important files:

- `config/example.yaml`: default local configuration.
- `config/routing_snapshot.json`: default routing snapshot.
- `examples/groq_banking77/config.yaml`: Groq + BANKING77 demo.
- `examples/cfpb_complaints/config.local_llm.yaml`: local LLM teacher demo.

Secrets should go in `.env` and must not be committed. Example:

```sh
GROQ_API_KEY=...
GROQ_MODEL=llama-3.1-8b-instant
```

## Kubernetes

Static manifests are available in:

```sh
deploy/kubernetes/distillforge.yaml
```

Apply them after building and publishing an image accessible by your cluster:

```sh
kubectl apply -f deploy/kubernetes/distillforge.yaml
```

## Troubleshooting

- `PermissionError` during `py_compile`: set `PYTHONPYCACHEPREFIX=/tmp/distillforge_pycache`.
- Slow BGE-M3 training on long documents: use `--max-input-chars` and
  `--embedding-max-seq-length`.
- Missing local Hugging Face model in offline mode: run once without
  `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1` to populate the cache.
- Port already used: change the listen address in the config or with the
  relevant `--host`, `--port`, or `--listen` flag.
