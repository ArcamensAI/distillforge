# DistillForge

Proxy FinOps pour LLM.

## Documentation

- [Specifications fonctionnelles](specs.md)
- [Architecture technique](ARCHITECTURE.md)

## V1 proxy

La premiere implementation de DistillForge vise le mode `teacher_only` :

- proxy Rust base sur Pingora ;
- endpoints proxifies : `POST /v1/chat/completions` et `POST /v1/completions` ;
- validation des headers `X-Client-ID` et `X-Task-ID` selon la configuration ;
- logs JSONL ;
- compteurs internes prepares pour `/metrics`.

Lancer en local :

```sh
cargo run
```

Par defaut le proxy ecoute sur `127.0.0.1:6188` et transmet au teacher
configure dans `config/example.yaml`.

## Routage local

Le routage est pilote par `config/routing_snapshot.json`. Le snapshot est charge
au demarrage et peut etre recharge sans redemarrer le proxy :

```sh
curl -X POST http://127.0.0.1:6188/admin/reload-routing
```

Modes supportes par le dataplane :

- `teacher_only`
- `shadow`
- `student_only`
- `canary`

Si un snapshot pointe vers un modele etudiant absent de `config/example.yaml`,
DistillForge retombe vers le teacher et journalise
`student_backend_missing_teacher_fallback`.

Si une route `student` est choisie mais que la connexion au backend etudiant
echoue, DistillForge retente automatiquement la requete sur le teacher et
journalise `student_connect_error_teacher_fallback`. Le compteur Prometheus
`distillforge_fallback_requests_total` expose ces bascules.

Les timeouts upstream sont configurables dans `config/example.yaml` :

```yaml
timeouts:
  upstream_connection_timeout_ms: 2000
  teacher_inference_timeout_ms: 30000
  student_inference_timeout_ms: 2000
  upstream_write_timeout_ms: 30000
  shadow_student_timeout_ms: 5000
```

## Worker etudiant minimal

Un worker HTTP de test permet de valider le routage student sans modele ML :

```sh
cargo run --bin student_worker
```

Par defaut il ecoute sur `127.0.0.1:9100`, expose `/health`, `/infer`,
`/v1/chat/completions` et `/v1/completions`, et retourne une reponse
deterministe configurable par variables d'environnement :

```sh
DISTILLFORGE_STUDENT_RESPONSE="student ok" cargo run --bin student_worker
```

## Rapport FinOps

Les logs JSONL peuvent etre agreges avec DuckDB :

```sh
python3 -m pip install -r requirements-analytics.txt
python3 tools/finops_report.py --logs 'data/logs/*.jsonl'
```

Le rapport affiche les volumes, decisions de routage, latences, couts estimes,
economies estimees et taches candidates a distillation.

Un dashboard HTML autonome peut aussi etre genere sans serveur :

```sh
python3 tools/finops_dashboard.py --logs 'data/logs/*.jsonl'
```

Le fichier `reports/distillforge_dashboard.html` affiche les KPIs FinOps,
le mix de routage, la tendance quotidienne, les couts par centre et les taches
candidates a distillation.

## Dataset builder

Les logs redacted eligibles peuvent etre convertis en dataset versionne :

```sh
python3 tools/build_dataset.py --task-id test_task --logs 'data/logs/*.jsonl'
```

Le builder filtre les requetes `success` avec `training_eligible=true`,
deduplique par `input_hash`, produit `train.parquet`, `validation.parquet`,
`test.parquet` et un `manifest.json`.

## Training offline minimal

Un modele etudiant classique peut etre entraine depuis un dataset versionne :

```sh
python3 -m pip install -r requirements-training.txt
python3 tools/train_student.py --dataset data/datasets/test_task/ds_example
```

Le trainer produit `model.joblib`, `eval_report.json` et `manifest.json` sous
`models/{task_id}/{model_id}`. Il utilise TF-IDF + LogisticRegression lorsque
le dataset contient plusieurs classes, sinon un baseline `DummyClassifier`.

Le modele entraine peut etre servi en HTTP :

```sh
python3 tools/student_inference.py --model-dir models/test_task/student_example
```

Ce worker expose `/health`, `/infer`, `/v1/chat/completions` et
`/v1/completions`, ce qui permet de le declarer comme backend `students` dans
`config/example.yaml`.

## Control plane local

Un serveur HTTP admin minimal orchestre les scripts V1 sans dependance de
plateforme externe :

```sh
python3 tools/control_plane.py --host 127.0.0.1 --port 8090
```

Endpoints principaux :

- `GET /health`
- `GET /admin/tasks/{task_id}/status`
- `GET /admin/models`
- `GET /admin/models/{model_id}`
- `POST /admin/tasks/{task_id}/train`
- `POST /admin/tasks/{task_id}/evaluate`
- `POST /admin/tasks/{task_id}/promote`
- `POST /admin/tasks/{task_id}/rollback`

Exemple de promotion shadow via control plane :

```sh
curl -X POST http://127.0.0.1:8090/admin/tasks/test_task/promote \
  -H 'Content-Type: application/json' \
  -d '{"model_dir":"models/test_task/student_example","mode":"shadow","min_accuracy":0.95}'
```

## Promotion et rollback

Un modele valide peut mettre a jour le snapshot de routage :

```sh
python3 tools/promote_model.py \
  --model-dir models/test_task/student_example \
  --mode canary \
  --student-traffic-percentage 10 \
  --min-accuracy 0.95
curl -X POST http://127.0.0.1:6188/admin/reload-routing
```

Le mode `shadow` sert le client avec le teacher et envoie une copie bornee de
la requete au student en arriere-plan :

```sh
python3 tools/promote_model.py \
  --model-dir models/test_task/student_example \
  --mode shadow \
  --min-accuracy 0.95
```

Les probes shadow sont journalises dans `logging.shadow_path` avec latence,
statut HTTP, reponse student redigee et indicateur `response_exact_match`.
Un rapport de divergence peut etre produit avec :

```sh
python3 tools/shadow_report.py --logs data/logs/shadow.jsonl
```

Rollback teacher pour une tache :

```sh
python3 tools/promote_model.py --task-id test_task --mode teacher_only
curl -X POST http://127.0.0.1:6188/admin/reload-routing
```

Chaque promotion ajoute un evenement JSONL dans `registry/events.jsonl`.

## Drift guard

Un garde-fou local peut analyser les logs recents et detecter une derive simple
sur les routes `canary` et `student_only` : taux d'erreur student, latence p95
et feedback humain negatif.

Audit sans modification :

```sh
python3 tools/drift_guard.py \
  --logs 'data/logs/*.jsonl' \
  --feedback data/logs/feedback.jsonl \
  --window-hours 24
```

Rollback automatique des taches en derive vers `teacher_only` :

```sh
python3 tools/drift_guard.py \
  --logs 'data/logs/*.jsonl' \
  --feedback data/logs/feedback.jsonl \
  --max-error-rate 0.05 \
  --max-negative-feedback-rate 0.20 \
  --apply
curl -X POST http://127.0.0.1:6188/admin/reload-routing
```

Chaque rollback ecrit un evenement `drift_guard_rollback` dans
`registry/events.jsonl`.

## Logs redacted

En mode `redacted`, DistillForge capture au maximum
`logging.max_capture_bytes` octets des corps de requete et de reponse, puis
ecrit `prompt_redacted` et `response_redacted` dans le JSONL. Les emails,
tokens bearer, cles `sk-*` et champs JSON courants comme `password`, `token`,
`secret`, `api_key` et `authorization` sont masques.

En mode `metadata_only`, aucun contenu ni hash d'entree n'est stocke.

## Feedback humain

Les clients peuvent envoyer une correction liee a une requete :

```sh
curl -X POST http://127.0.0.1:6188/v1/feedback \
  -H 'Content-Type: application/json' \
  -H 'X-Client-ID: crm_backend' \
  -H 'X-Task-ID: email_classification_v1' \
  -d '{"request_id":"req_123","rating":"bad","correct_output":"billing_issue","comment":"Mauvaise classe"}'
```

Les feedbacks sont rediges dans `logging.feedback_path` au format JSONL, avec
redaction des champs texte sensibles.
