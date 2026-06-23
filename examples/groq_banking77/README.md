# Demo Groq + BANKING77

Cette demo montre DistillForge sur un cas realiste : classifier des intentions
de support bancaire avec Groq comme teacher, capturer les traces dans le proxy,
construire un dataset, entrainer un student local, puis promouvoir ce student
en `shadow`, `canary` ou `bandit`.

Le scenario est volontairement proche d'un usage produit : une application
support appelle un endpoint OpenAI-compatible, ajoute `X-Client-ID` et
`X-Task-ID`, puis DistillForge decide si la requete part vers le teacher Groq
ou vers un student specialise.

## Sources

- Groq API : https://console.groq.com/docs/quickstart
- Groq rate limits : https://console.groq.com/docs/rate-limits
- Groq reasoning : https://console.groq.com/docs/reasoning
- BANKING77 : https://huggingface.co/datasets/PolyAI/banking77
- Fichiers source : https://github.com/PolyAI-LDN/task-specific-datasets/tree/master/banking_data

Au 2026-06-23, les limites gratuites Groq restent a verifier dans la page
Limits de votre organisation. Cette demo est calibree pour rester sous un
budget indicatif de `200K` tokens teacher avec le modele
`openai/gpt-oss-20b`, mais l'interface Groq reste la source autoritative.

## Architecture de la demo

```text
client demo
  -> DistillForge proxy :6188
    -> teacher local :9200
      -> Groq /openai/v1/chat/completions

logs DistillForge
  -> build_dataset.py
  -> train_student.py
  -> student_inference.py :9100
  -> promote_model.py
```

Fichiers principaux :

- `config.yaml` : scenario smoke.
- `routing_snapshot.json` : routage smoke.
- `config.volume.yaml` : scenario volume controle.
- `routing_snapshot.volume.json` : routage volume.
- `tools/groq_teacher.py` : adaptateur Groq local.
- `tools/banking77_demo.py` : preparation, budget, appels proxy, evaluation.

Les donnees generees sont ignorees par Git :

- `examples/groq_banking77/data/`
- `examples/groq_banking77/data_volume/`
- `examples/groq_banking77/models/`
- `examples/groq_banking77/registry/`

## Prerequis

Installer les dependances Python :

```sh
python3 -m pip install -r requirements-analytics.txt
python3 -m pip install -r requirements-training.txt
```

Configurer Groq dans `.env` :

```sh
GROQ_API_KEY="<your-api-key>"
GROQ_MODEL="openai/gpt-oss-20b"
```

Charger l'environnement dans chaque terminal qui appelle Groq :

```sh
set -a
source .env
set +a
```

Note pour GPT-OSS : l'adaptateur utilise `reasoning_format: hidden`,
`reasoning_effort: low` et `max_completion_tokens: 64`. Sans cela, le modele
peut consommer les premiers tokens en raisonnement et renvoyer un `content`
vide.

## Scenario 1 : smoke test

Objectif : verifier le chemin complet avec tres peu d'appels Groq.

Preparer un petit echantillon :

```sh
python3 tools/banking77_demo.py prepare \
  --out examples/groq_banking77/data \
  --train-limit 5 \
  --eval-limit 2
```

Estimer le budget :

```sh
python3 tools/banking77_demo.py estimate-budget \
  --requests examples/groq_banking77/data/requests/train_requests.jsonl \
  --intents examples/groq_banking77/data/intents.json \
  --limit 3 \
  --token-budget 200000
```

Lancer le teacher Groq :

```sh
set -a
source .env
set +a

python3 tools/groq_teacher.py \
  --host 127.0.0.1 \
  --port 9200 \
  --intents examples/groq_banking77/data/intents.json \
  --model "$GROQ_MODEL"
```

Dans un autre terminal, lancer DistillForge :

```sh
DISTILLFORGE_CONFIG=examples/groq_banking77/config.yaml \
  cargo run --bin distillforge
```

Envoyer trois requetes via le proxy :

```sh
python3 tools/banking77_demo.py run-proxy \
  --requests examples/groq_banking77/data/requests/train_requests.jsonl \
  --proxy-url http://127.0.0.1:6188 \
  --out examples/groq_banking77/data/teacher_calls.jsonl \
  --limit 3 \
  --sleep-ms 250
```

Evaluer le teacher par rapport aux labels BANKING77 :

```sh
python3 tools/banking77_demo.py evaluate-calls \
  --calls examples/groq_banking77/data/teacher_calls.jsonl
```

Resultat observe pendant le smoke test :

```json
{
  "accuracy": 1.0,
  "calls": 3,
  "correct": 3,
  "invalid": 0
}
```

Construire le dataset depuis les logs proxy :

```sh
python3 tools/build_dataset.py \
  --task-id banking_intent_v1 \
  --logs examples/groq_banking77/data/logs/proxy.jsonl \
  --out examples/groq_banking77/data/datasets \
  --dataset-id ds_banking77_groq_smoke \
  --target-field openai_message_content \
  --min-samples 1
```

Entrainer un student smoke :

```sh
python3 tools/train_student.py \
  --dataset examples/groq_banking77/data/datasets/banking_intent_v1/ds_banking77_groq_smoke \
  --out examples/groq_banking77/models \
  --model-id banking_intent_student_smoke \
  --min-train-samples 1
```

Servir le student :

```sh
python3 tools/student_inference.py \
  --model-dir examples/groq_banking77/models/banking_intent_v1/banking_intent_student_smoke \
  --listen 127.0.0.1:9100
```

Le smoke test valide la plomberie. Il ne valide pas la qualite du student :
trois exemples ne suffisent pas pour couvrir 77 intentions.

## Scenario 2 : volume controle

Objectif : demarrer une demo avec plus de volumetrie tout en restant prudents
sur le budget gratuit Groq.

Parametres recommandes :

- `400` exemples prepares pour le train.
- `100` exemples prepares pour l'evaluation.
- `300` appels teacher maximum dans une premiere passe.
- `sleep-ms 3000`, soit environ 20 requetes/minute.
- `config.volume.yaml`, avec logs dedies dans `data_volume/`.

Preparer le scenario volume :

```sh
python3 tools/banking77_demo.py prepare \
  --out examples/groq_banking77/data_volume \
  --train-limit 400 \
  --eval-limit 100
```

Estimer le budget avant tout appel Groq :

```sh
python3 tools/banking77_demo.py estimate-budget \
  --requests examples/groq_banking77/data_volume/requests/train_requests.jsonl \
  --intents examples/groq_banking77/data_volume/intents.json \
  --limit 300 \
  --token-budget 200000
```

Si `estimated_prompt_tokens` reste sous le budget accepte, lancer le teacher :

```sh
set -a
source .env
set +a

python3 tools/groq_teacher.py \
  --host 127.0.0.1 \
  --port 9200 \
  --intents examples/groq_banking77/data_volume/intents.json \
  --model "$GROQ_MODEL"
```

Lancer le proxy volume :

```sh
DISTILLFORGE_CONFIG=examples/groq_banking77/config.volume.yaml \
  cargo run --bin distillforge
```

Envoyer 300 appels teacher :

```sh
python3 tools/banking77_demo.py run-proxy \
  --requests examples/groq_banking77/data_volume/requests/train_requests.jsonl \
  --proxy-url http://127.0.0.1:6188 \
  --out examples/groq_banking77/data_volume/teacher_calls.jsonl \
  --limit 300 \
  --sleep-ms 3000
```

Evaluer le teacher :

```sh
python3 tools/banking77_demo.py evaluate-calls \
  --calls examples/groq_banking77/data_volume/teacher_calls.jsonl
```

Construire le dataset volume :

```sh
python3 tools/build_dataset.py \
  --task-id banking_intent_v1 \
  --logs examples/groq_banking77/data_volume/logs/proxy.jsonl \
  --out examples/groq_banking77/data_volume/datasets \
  --dataset-id ds_banking77_groq_volume \
  --target-field openai_message_content \
  --min-samples 100
```

Optionnel : augmenter localement le dataset sans nouvel appel Groq :

```sh
python3 tools/synthetic_data.py \
  --dataset examples/groq_banking77/data_volume/datasets/banking_intent_v1/ds_banking77_groq_volume \
  --out examples/groq_banking77/data_volume/datasets \
  --dataset-id ds_banking77_groq_volume_synth \
  --multiplier 1 \
  --max-synthetic-ratio 0.5
```

Entrainer le student volume :

```sh
python3 tools/train_student.py \
  --dataset examples/groq_banking77/data_volume/datasets/banking_intent_v1/ds_banking77_groq_volume \
  --out examples/groq_banking77/models \
  --model-id banking_intent_student_volume \
  --min-train-samples 100
```

Servir le student volume :

```sh
python3 tools/student_inference.py \
  --model-dir examples/groq_banking77/models/banking_intent_v1/banking_intent_student_volume \
  --listen 127.0.0.1:9100
```

Promouvoir en shadow :

```sh
python3 tools/promote_model.py \
  --model-dir examples/groq_banking77/models/banking_intent_v1/banking_intent_student_volume \
  --snapshot examples/groq_banking77/routing_snapshot.volume.json \
  --registry examples/groq_banking77/registry/events.jsonl \
  --mode shadow \
  --min-accuracy 0.50

curl -X POST http://127.0.0.1:6188/admin/reload-routing
```

Passer en bandit avec 2 % de probes teacher :

```sh
python3 tools/promote_model.py \
  --model-dir examples/groq_banking77/models/banking_intent_v1/banking_intent_student_volume \
  --snapshot examples/groq_banking77/routing_snapshot.volume.json \
  --registry examples/groq_banking77/registry/events.jsonl \
  --mode bandit \
  --teacher-probe-percentage 2 \
  --min-accuracy 0.50

curl -X POST http://127.0.0.1:6188/admin/reload-routing
```

## Observabilite

Metrics proxy :

```sh
curl -sS http://127.0.0.1:6188/metrics
```

Rapport FinOps :

```sh
python3 tools/finops_report.py \
  --logs 'examples/groq_banking77/data_volume/logs/*.jsonl'
```

Dashboard :

```sh
python3 tools/finops_dashboard.py \
  --logs 'examples/groq_banking77/data_volume/logs/*.jsonl' \
  --out examples/groq_banking77/data_volume/reports/dashboard.html
```

Rapport shadow :

```sh
python3 tools/shadow_report.py \
  --logs examples/groq_banking77/data_volume/logs/shadow.jsonl
```

Drift guard :

```sh
python3 tools/drift_guard.py \
  --logs 'examples/groq_banking77/data_volume/logs/*.jsonl' \
  --feedback examples/groq_banking77/data_volume/logs/feedback.jsonl \
  --snapshot examples/groq_banking77/routing_snapshot.volume.json
```

## Rollback

```sh
python3 tools/promote_model.py \
  --task-id banking_intent_v1 \
  --snapshot examples/groq_banking77/routing_snapshot.volume.json \
  --registry examples/groq_banking77/registry/events.jsonl \
  --mode teacher_only

curl -X POST http://127.0.0.1:6188/admin/reload-routing
```

## Nettoyage local

Les donnees generees sont ignorees par Git. Pour repartir de zero, supprimer
les dossiers locaux de la demo :

```sh
rm -rf examples/groq_banking77/data examples/groq_banking77/data_volume
rm -rf examples/groq_banking77/models examples/groq_banking77/registry
```

Ne supprimez pas `.env` si vous voulez conserver la cle Groq locale.

## Depannage

- `403 error code: 1010` : verifier que l'adaptateur utilise bien un
  `User-Agent` explicite. C'est le cas dans `tools/groq_teacher.py`.
- `content=""` avec GPT-OSS : verifier `reasoning_format: hidden`.
- `429` Groq : augmenter `--sleep-ms`, reduire `--limit`, ou attendre le reset
  de quota.
- Dataset vide : verifier que les logs proxy contiennent des entrees
  `status=success` et que `--target-field openai_message_content` est utilise.
- Student faible : augmenter la volumetrie reelle avant d'utiliser les donnees
  synthetiques.
