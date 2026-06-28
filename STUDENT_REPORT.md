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

## CFPB Local LLM Teacher Scenario

Date: 2026-06-28

Scenario: `examples/cfpb_complaints`

- Task id: `cfpb_product_triage_v1`
- Teacher: `mlx-community/Qwen3-8B-4bit`
- Distilled dataset: `ds_cfpb_product_llm_2000`
- Source traces: `examples/cfpb_complaints/data_llm/logs/proxy.jsonl`
- Training target: teacher response extracted from `openai_message_content`
- Dataset samples after deduplication: `1,952`
- Split: `1,368` train, `278` validation, `306` test
- Labels: `12`

This dataset is built from DistillForge teacher traces, not from the official
CFPB labels. The proxy logs successful `training_eligible` teacher calls, then
`tools/build_dataset.py` extracts the teacher answer from the OpenAI-compatible
response body. The official CFPB product label is used only as an external
diagnostic signal.

Student results measured against the Qwen3-8B teacher labels on the validation
split:

| Student | Runtime | Accuracy | Macro F1 |
| --- | --- | ---: | ---: |
| `cfpb_product_student_tfidf_2000` | TF-IDF + logistic regression | `0.8094` | `0.7746` |
| `cfpb_product_student_tfidf_mlp_2000` | TF-IDF + MLP | `0.7878` | `0.6907` |
| `cfpb_product_student_bge_m3_mlp_2000` | `BAAI/bge-m3` embeddings + MLP | `0.7050` | `0.6247` |
| `cfpb_product_student_bge_m3_hybrid_mlp_2000` | `BAAI/bge-m3` embeddings + TF-IDF + MLP | `0.7950` | `0.7256` |
| `cfpb_product_student_bge_m3_hybrid_logistic_2000` | `BAAI/bge-m3` embeddings + TF-IDF + logistic regression | `0.7734` | `0.7507` |
| `qwen2_5_0_5b_cfpb_prompt` | `mlx-community/Qwen2.5-0.5B-Instruct-4bit` prompt-only classifier, no fine-tuning | `0.5108` | `0.3475` |
| `qwen3_0_6b_cfpb_prompt` | `mlx-community/Qwen3-0.6B-4bit` prompt-only classifier, no fine-tuning | `0.3597` | `0.2418` |
| `qwen2_5_0_5b_lora_cfpb_student` | Qwen2.5 0.5B + LoRA fine-tuning, 120 steps, 8 layers | `0.6043` | `0.4655` |
| `qwen2_5_0_5b_lora600_cfpb_student` | Qwen2.5 0.5B + LoRA fine-tuning, 600 steps, 16 layers | `0.7230` | `0.6439` |

The two `prompt` rows are included only as baselines. They are not fine-tuned
students. The actual LLM student is the LoRA-adapted Qwen2.5 0.5B model.

Fine-tuned LLM student artifacts:

- SFT export: `examples/cfpb_complaints/data_llm/sft_qwen_cfpb_2000_messages`
- SFT source traces: `examples/cfpb_complaints/data_llm/logs/proxy.jsonl`
- 120-step adapter: `examples/cfpb_complaints/models/cfpb_product_triage_v1/qwen2_5_0_5b_lora_cfpb_2000_adapters`
- 600-step adapter: `examples/cfpb_complaints/models/cfpb_product_triage_v1/qwen2_5_0_5b_lora_cfpb_2000_adapters_600`
- Base model cache size: `282 MB`
- 120-step adapter size: `11 MB`
- 600-step adapter size: `22 MB`

The 600-step LoRA student was trained with `2.933M` trainable parameters out of
`494.033M` total parameters (`0.594%`). Validation latency through the local
OpenAI-compatible HTTP server was:

- Mean latency: `273.9 ms`
- Median latency: `274 ms`
- P95 latency: `303 ms`

Additional direct evaluation of `qwen2_5_0_5b_cfpb_prompt` against the
official CFPB product labels on `400` held-out prepared requests:

- Accuracy: `0.5975`
- Invalid labels: `0`
- Mean latency: `273.4 ms`
- Median latency: `272 ms`
- P95 latency: `305 ms`
- Local model cache size: `282 MB`

The Qwen3 0.6B prompt classifier was also tested. It was slower than Qwen2.5
0.5B and lower quality on this label set:

- Distilled validation accuracy: `0.3597`
- Distilled validation macro F1: `0.2418`
- Mean latency: `358.0 ms`
- Median latency: `353 ms`
- P95 latency: `418 ms`
- Local model cache size: `335 MB`

Conclusion: LoRA fine-tuning turns the small Qwen2.5 0.5B model into a real
student and improves it materially over prompt-only inference:

- `+21.22` accuracy points versus Qwen2.5 0.5B prompt-only;
- `+29.64` macro F1 points versus Qwen2.5 0.5B prompt-only.

It still does not beat the TF-IDF logistic baseline on the 2k CFPB distilled
dataset. The current best overall CFPB student remains
`cfpb_product_student_tfidf_2000`. The best neural/non-LLM variant remains
`cfpb_product_student_bge_m3_hybrid_mlp_2000`.
