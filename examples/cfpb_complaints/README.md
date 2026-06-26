# CFPB Complaints Local LLM Teacher Demo

This scenario demonstrates DistillForge on a realistic enterprise workflow:
triaging financial customer complaints from the public CFPB Consumer Complaint
Database.

The teacher is a local large LLM, typically `mlx-community/Qwen3-8B-4bit` on an
Apple Silicon machine. The student is a local classifier trained from
DistillForge logs, usually the hybrid `BAAI/bge-m3` + TF-IDF runtime.

## Task

Input: a consumer complaint narrative.

Output: one CFPB product label slug, for example:

- `credit_card_or_prepaid_card`
- `credit_reporting_or_other_personal_consumer_reports`
- `debt_collection`
- `mortgage`

The original CFPB product label is kept in `product_labels.json`; the LLM uses
short slug labels to reduce invalid generations.

## Files

- `config.local_llm.yaml`: DistillForge config for the local LLM teacher.
- `routing_snapshot.local_llm.json`: teacher-only routing snapshot.
- `tools/cfpb_demo.py`: data preparation, proxy calls and evaluation.
- `tools/local_llm_label_teacher.py`: generic local MLX LLM label classifier.
- `tools/train_student.py`: trains TF-IDF, embedding or hybrid students.
- `tools/student_inference.py`: serves the trained student over HTTP.

Generated data is ignored by Git under `examples/*/data*`,
`examples/*/models` and `examples/*/registry`.

## Data Source

The demo uses the CFPB public complaints export:

`https://files.consumerfinance.gov/ccdb/complaints.csv.zip`

The file is large, around 1.3 GB at the time this scenario was added. For
development, prefer downloading it once and passing the local ZIP with
`--source`.

## Prepare Data

Prepare a first LLM-sized sample:

```sh
python3 tools/cfpb_demo.py prepare \
  --out examples/cfpb_complaints/data_llm \
  --source examples/cfpb_complaints/data_llm/raw/complaints.csv.zip \
  --train-limit 500 \
  --eval-limit 100 \
  --max-source-rows 200000
```

If `--source` is omitted, the script downloads the CFPB ZIP into
`examples/cfpb_complaints/data_llm/raw/complaints.csv.zip`.

## Start The Local 8B Teacher

```sh
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
python3 tools/local_llm_label_teacher.py \
  --host 127.0.0.1 \
  --port 9600 \
  --model mlx-community/Qwen3-8B-4bit \
  --model-id qwen3_8b_cfpb_teacher \
  --labels examples/cfpb_complaints/data_llm/labels.json \
  --task-description "Classify the CFPB consumer complaint into the most likely financial product category." \
  --max-tokens 96
```

## Start DistillForge

```sh
DISTILLFORGE_CONFIG=examples/cfpb_complaints/config.local_llm.yaml \
  cargo run --bin distillforge
```

## Collect Teacher Labels

Start with a small smoke run because local 8B inference is slow:

```sh
python3 tools/cfpb_demo.py run-proxy \
  --requests examples/cfpb_complaints/data_llm/requests/train_requests.jsonl \
  --proxy-url http://127.0.0.1:6188 \
  --out examples/cfpb_complaints/data_llm/teacher_calls_smoke.jsonl \
  --limit 20 \
  --sleep-ms 0
```

Then scale up:

```sh
python3 tools/cfpb_demo.py run-proxy \
  --requests examples/cfpb_complaints/data_llm/requests/train_requests.jsonl \
  --proxy-url http://127.0.0.1:6188 \
  --out examples/cfpb_complaints/data_llm/teacher_calls.jsonl \
  --limit 500 \
  --sleep-ms 0
```

Evaluate teacher agreement with the official CFPB product labels:

```sh
python3 tools/cfpb_demo.py evaluate-calls \
  --calls examples/cfpb_complaints/data_llm/teacher_calls.jsonl
```

The official label is not treated as perfect ground truth for every narrative,
but it gives a useful consistency signal.

## Build Dataset

```sh
python3 tools/build_dataset.py \
  --task-id cfpb_product_triage_v1 \
  --logs examples/cfpb_complaints/data_llm/logs/proxy.jsonl \
  --out examples/cfpb_complaints/data_llm/datasets \
  --dataset-id ds_cfpb_product_llm_500 \
  --target-field openai_message_content \
  --min-samples 100
```

## Train Student

```sh
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
python3 tools/train_student.py \
  --dataset examples/cfpb_complaints/data_llm/datasets/cfpb_product_triage_v1/ds_cfpb_product_llm_500 \
  --out examples/cfpb_complaints/models \
  --model-id cfpb_product_student_hybrid_bge_m3 \
  --min-train-samples 100 \
  --student-kind hybrid_embedding_tfidf \
  --embedding-model BAAI/bge-m3 \
  --input-format openai_user_content \
  --batch-size 64
```

## Serve Student

```sh
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
python3 tools/student_inference.py \
  --model-dir examples/cfpb_complaints/models/cfpb_product_triage_v1/cfpb_product_student_hybrid_bge_m3 \
  --listen 127.0.0.1:9102
```

## Scale Guidance

Qwen3 8B local inference is materially slower than an embedding teacher. Use it
to create higher-quality distillation labels, not to process 10k requests during
an interactive session.

Recommended progression:

1. `20` calls: smoke test prompt and parsing.
2. `500` calls: first useful student.
3. `2,000+` calls: overnight run for stronger coverage.
4. Shadow mode: compare student against fresh teacher traffic before promotion.
