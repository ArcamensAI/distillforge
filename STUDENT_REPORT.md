# DistillForge Student Quality Report

Date: 2026-06-25

## Objective

Improve the Banking77 student model quality after the TF-IDF baseline plateaued
around 0.82 accuracy.

## Dataset

- Scenario: `examples/groq_banking77` local 10k
- Dataset id: `ds_banking77_local_10k`
- Task id: `banking_intent_v1`
- Samples after deduplication: `9,996`
- Split: `7,005` train, `1,560` validation, `1,431` test
- Labels: `77`
- Teacher: local embedding teacher
- Teacher quality on 10k calls: accuracy `0.9412`, invalid labels `0`

## Student Results

| Student | Runtime | Accuracy | Macro F1 |
| --- | --- | ---: | ---: |
| `banking_intent_student_local_10k` | TF-IDF + logistic regression | `0.8192` | `0.8231` |
| `banking_intent_student_tfidf_clean_10k` | TF-IDF on extracted user text | `0.8135` | `0.8165` |
| `banking_intent_student_embedding_miniLM_10k` | MiniLM embeddings + logistic regression | `0.8853` | `0.8852` |
| `banking_intent_student_embedding_bge_m3_10k` | `BAAI/bge-m3` embeddings + logistic regression | `0.8981` | `0.8997` |
| `banking_intent_student_hybrid_bge_m3_10k` | `BAAI/bge-m3` embeddings + TF-IDF + logistic regression | `0.9135` | `0.9177` |

## Decision

The recommended student for the Banking77 demo is:

`banking_intent_student_hybrid_bge_m3_10k`

It improves the previous baseline by:

- `+9.43` accuracy points versus TF-IDF baseline;
- `+9.46` macro F1 points versus TF-IDF baseline;
- no LLM generation at inference time;
- deterministic label space with standard classifier probabilities.

## Implementation

The training pipeline now supports:

- `--student-kind tfidf`
- `--student-kind embedding_logistic`
- `--student-kind hybrid_embedding_tfidf`
- `--embedding-model`
- `--input-format openai_user_content`

The inference server now supports:

- existing sklearn `model.joblib` artifacts;
- SentenceTransformer classifier artifacts;
- hybrid SentenceTransformer + TF-IDF artifacts.

## Reproduction

Train the recommended student:

```sh
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
python3 tools/train_student.py \
  --dataset examples/groq_banking77/data_local_10k/datasets/banking_intent_v1/ds_banking77_local_10k \
  --out examples/groq_banking77/models \
  --model-id banking_intent_student_hybrid_bge_m3_10k \
  --min-train-samples 7000 \
  --student-kind hybrid_embedding_tfidf \
  --embedding-model BAAI/bge-m3 \
  --input-format openai_user_content \
  --batch-size 64
```

Serve it locally:

```sh
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
python3 tools/student_inference.py \
  --model-dir examples/groq_banking77/models/banking_intent_v1/banking_intent_student_hybrid_bge_m3_10k \
  --listen 127.0.0.1:9101
```

## Next Steps

1. Run this hybrid student in `shadow` mode against fresh traffic.
2. Add a threshold policy based on classifier confidence and teacher margin.
3. Promote to canary only if shadow agreement and manual spot checks are stable.
4. Test a true fine-tuned encoder as a later track if the target is above `0.95`.
