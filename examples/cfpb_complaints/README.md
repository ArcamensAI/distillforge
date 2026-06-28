# CFPB Complaints Local LLM Teacher Demo

This scenario demonstrates DistillForge on a realistic enterprise workflow:
triaging financial customer complaints from the public CFPB Consumer Complaint
Database.

The teacher is a local large LLM, typically `mlx-community/Qwen3-8B-4bit` on an
Apple Silicon machine. The student can be either a local classifier trained from
DistillForge logs or a fine-tuned small LLM. The strongest 2k baseline so far is
TF-IDF logistic regression; the tested LLM student is Qwen2.5 0.5B with LoRA
fine-tuning.

## Task

Input: a consumer complaint narrative.

Output: one CFPB product label slug, for example:

- `credit_card`
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
- `tools/export_llm_sft.py`: exports a DistillForge dataset for MLX-LM SFT.
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

The student dataset is built from DistillForge proxy traces. This is the core
loop of the product: route requests to the teacher, log redacted successful
teacher responses as `training_eligible`, then train the student on those traces.
The CFPB official product label is useful for diagnostics, but it is not the
student target in this distillation run.

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

For a fully local neural-network student without sentence-transformer
embeddings, train the TF-IDF MLP variant:

```sh
python3 tools/train_student.py \
  --dataset examples/cfpb_complaints/data_llm/datasets/cfpb_product_triage_v1/ds_cfpb_product_llm_2000 \
  --out examples/cfpb_complaints/models \
  --model-id cfpb_product_student_tfidf_mlp_2000 \
  --min-train-samples 1000 \
  --student-kind tfidf_mlp \
  --input-format openai_user_content
```

On the 2,000-complaint local Qwen3 teacher run, this MLP student produced
`0.7878` validation accuracy and `0.6907` macro-F1 over 12 labels. The simpler
TF-IDF logistic baseline remained stronger on the same split at `0.8094`
accuracy and `0.7746` macro-F1.

For a BGE-based neural student, keep the BGE encoder local and add a neural
classification head. CFPB narratives can be long, so cap both the extracted
input text and the BGE sequence length:

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

On the same 2,000-complaint run:

- `embedding_mlp` with BGE-M3 only: `0.7050` accuracy, `0.6247` macro-F1.
- `hybrid_embedding_tfidf_mlp`: `0.7950` accuracy, `0.7256` macro-F1.
- `hybrid_embedding_tfidf` logistic head: `0.7734` accuracy, `0.7507` macro-F1.

The BGE hybrid variants are viable local neural students, but this CFPB task
still benefits heavily from lexical features in the reported product and issue
fields.

## Fine-Tune A Small LLM Student

Export the DistillForge teacher-trace dataset to MLX-LM chat fine-tuning
format. The source dataset manifest points back to
`examples/cfpb_complaints/data_llm/logs/proxy.jsonl`, and the target is the
teacher output extracted from `openai_message_content`.

```sh
python3 tools/export_llm_sft.py \
  --dataset examples/cfpb_complaints/data_llm/datasets/cfpb_product_triage_v1/ds_cfpb_product_llm_2000 \
  --out examples/cfpb_complaints/data_llm/sft_qwen_cfpb_2000_messages \
  --input-format openai_user_content \
  --max-input-chars 1800 \
  --task-description "Classify the CFPB consumer complaint into the most likely financial product category." \
  --format messages
```

Fine-tune Qwen2.5 0.5B with LoRA:

```sh
python3 -m mlx_lm.lora \
  --model mlx-community/Qwen2.5-0.5B-Instruct-4bit \
  --train \
  --data examples/cfpb_complaints/data_llm/sft_qwen_cfpb_2000_messages \
  --adapter-path examples/cfpb_complaints/models/cfpb_product_triage_v1/qwen2_5_0_5b_lora_cfpb_2000_adapters_600 \
  --fine-tune-type lora \
  --mask-prompt \
  --batch-size 1 \
  --grad-accumulation-steps 4 \
  --iters 600 \
  --val-batches 60 \
  --steps-per-report 100 \
  --steps-per-eval 200 \
  --learning-rate 1e-5 \
  --num-layers 16 \
  --max-seq-length 768 \
  --save-every 600
```

Serve the fine-tuned student through the same local OpenAI-compatible
classifier server:

```sh
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
python3 tools/local_llm_label_teacher.py \
  --host 127.0.0.1 \
  --port 9703 \
  --model mlx-community/Qwen2.5-0.5B-Instruct-4bit \
  --adapter-path examples/cfpb_complaints/models/cfpb_product_triage_v1/qwen2_5_0_5b_lora_cfpb_2000_adapters_600 \
  --model-id qwen2_5_0_5b_lora600_cfpb_student \
  --labels examples/cfpb_complaints/data_llm/labels.json \
  --task-description "Classify the CFPB consumer complaint into the most likely financial product category." \
  --max-tokens 12 \
  --max-input-chars 1800
```

Measured against the Qwen3-8B teacher labels on the 278-example validation
split:

- Qwen2.5 0.5B prompt-only: `0.5108` accuracy, `0.3475` macro-F1.
- Qwen2.5 0.5B + LoRA, 120 steps: `0.6043` accuracy, `0.4655` macro-F1.
- Qwen2.5 0.5B + LoRA, 600 steps: `0.7230` accuracy, `0.6439` macro-F1.

The 600-step LoRA adapter is `22 MB`, trains `2.933M` parameters, and served at
`273.9 ms` mean latency / `303 ms` p95 on the local validation run. It is a real
fine-tuned LLM student, but it still trails the TF-IDF logistic baseline
(`0.8094` accuracy, `0.7746` macro-F1) on this 2k dataset.

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
