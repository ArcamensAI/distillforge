# Demo Groq + BANKING77

Cette demo montre DistillForge sur un cas realiste : classification
d'intentions de support bancaire avec Groq comme teacher et un student local
entraine ensuite depuis les logs.

Sources utilisees :

- Groq API : https://console.groq.com/docs/quickstart
- Groq rate limits : https://console.groq.com/docs/rate-limits
- BANKING77 : https://huggingface.co/datasets/PolyAI/banking77
- Fichiers source : https://github.com/PolyAI-LDN/task-specific-datasets/tree/master/banking_data

Au 2026-06-22, la documentation Groq indique des limites gratuites par modele.
Les modeles `openai/gpt-oss-20b` et `openai/gpt-oss-120b` apparaissent avec
`200K` tokens/jour. La page Limits de votre organisation Groq reste la source
autoritative.

## Scenario

1. Télécharger un echantillon BANKING77.
2. Lancer un teacher local qui appelle Groq.
3. Lancer DistillForge avec ce teacher.
4. Envoyer des requetes client via le proxy.
5. Construire un dataset depuis les logs DistillForge.
6. Entrainer un student local.
7. Promouvoir le student en `shadow`, puis en `canary` ou `bandit`.

## Preparation

```sh
python3 tools/banking77_demo.py prepare \
  --out examples/groq_banking77/data \
  --train-limit 240 \
  --eval-limit 80
```

Le script ecrit :

- `examples/groq_banking77/data/intents.json`
- `examples/groq_banking77/data/requests/train_requests.jsonl`
- `examples/groq_banking77/data/requests/eval_requests.jsonl`
- `examples/groq_banking77/data/manifest.json`

Budget approximatif pour 100 appels teacher :

```sh
python3 tools/banking77_demo.py estimate-budget \
  --requests examples/groq_banking77/data/requests/train_requests.jsonl \
  --intents examples/groq_banking77/data/intents.json \
  --limit 100
```

## Teacher Groq

Configurer la cle API :

```sh
export GROQ_API_KEY="<your-api-key>"
export GROQ_MODEL="openai/gpt-oss-20b"
```

Lancer l'adaptateur teacher :

```sh
python3 tools/groq_teacher.py \
  --host 127.0.0.1 \
  --port 9200 \
  --intents examples/groq_banking77/data/intents.json \
  --model "$GROQ_MODEL"
```

L'adaptateur expose `/v1/chat/completions`, appelle Groq, force la sortie sur
une des 77 intentions, puis renvoie une reponse OpenAI-compatible stable.
Pour les modeles GPT-OSS, il utilise `reasoning_format: hidden` afin que les
tokens de raisonnement ne remplacent pas le label final.

## Proxy DistillForge

Dans un autre terminal :

```sh
DISTILLFORGE_CONFIG=examples/groq_banking77/config.yaml cargo run --bin distillforge
```

Envoyer 100 requetes BANKING77 via le proxy :

```sh
python3 tools/banking77_demo.py run-proxy \
  --requests examples/groq_banking77/data/requests/train_requests.jsonl \
  --proxy-url http://127.0.0.1:6188 \
  --out examples/groq_banking77/data/teacher_calls.jsonl \
  --limit 100 \
  --sleep-ms 250
```

Comparer les reponses teacher aux labels BANKING77 :

```sh
python3 tools/banking77_demo.py evaluate-calls \
  --calls examples/groq_banking77/data/teacher_calls.jsonl
```

Les logs proxy sont ecrits dans :

```text
examples/groq_banking77/data/logs/proxy.jsonl
```

## Dataset et student

Construire le dataset depuis les logs DistillForge en extrayant le contenu
assistant OpenAI-compatible :

```sh
python3 tools/build_dataset.py \
  --task-id banking_intent_v1 \
  --logs examples/groq_banking77/data/logs/proxy.jsonl \
  --out examples/groq_banking77/data/datasets \
  --dataset-id ds_banking77_groq \
  --target-field openai_message_content
```

Entrainer le student :

```sh
python3 tools/train_student.py \
  --dataset examples/groq_banking77/data/datasets/banking_intent_v1/ds_banking77_groq \
  --out examples/groq_banking77/models \
  --model-id banking_intent_student
```

Servir le student :

```sh
python3 tools/student_inference.py \
  --model-dir examples/groq_banking77/models/banking_intent_v1/banking_intent_student \
  --host 127.0.0.1 \
  --port 9100
```

## Promotion

Promotion shadow :

```sh
python3 tools/promote_model.py \
  --model-dir examples/groq_banking77/models/banking_intent_v1/banking_intent_student \
  --snapshot examples/groq_banking77/routing_snapshot.json \
  --registry examples/groq_banking77/registry/events.jsonl \
  --mode shadow \
  --min-accuracy 0.50

curl -X POST http://127.0.0.1:6188/admin/reload-routing
```

Promotion bandit avec 2 % de probes teacher :

```sh
python3 tools/promote_model.py \
  --model-dir examples/groq_banking77/models/banking_intent_v1/banking_intent_student \
  --snapshot examples/groq_banking77/routing_snapshot.json \
  --registry examples/groq_banking77/registry/events.jsonl \
  --mode bandit \
  --teacher-probe-percentage 2 \
  --min-accuracy 0.50

curl -X POST http://127.0.0.1:6188/admin/reload-routing
```

Rollback :

```sh
python3 tools/promote_model.py \
  --task-id banking_intent_v1 \
  --snapshot examples/groq_banking77/routing_snapshot.json \
  --registry examples/groq_banking77/registry/events.jsonl \
  --mode teacher_only

curl -X POST http://127.0.0.1:6188/admin/reload-routing
```
