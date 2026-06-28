# DistillForge

*Lire dans une autre langue : [English](README.md)*

Proxy FinOps pour LLM.

## Documentation

- [Installation](docs/INSTALLATION.fr.md)
- [Specifications fonctionnelles](specs.md)
- [Architecture technique](ARCHITECTURE.md)
- [Demo Groq + BANKING77](examples/groq_banking77/README.md)
- [Demo CFPB avec teacher LLM local](examples/cfpb_complaints/README.md)

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
- `bandit`

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

## Limitation de debit locale

DistillForge peut appliquer une limitation de debit en memoire, sans service
tiers, avant d'appeler les backends LLM. La limite par defaut s'applique par
`X-Client-ID`; des limites specifiques peuvent etre posees par client ou par
tache :

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

Une requete au-dessus du seuil est refusee en `429` avec le motif
`rate_limited_client` ou `rate_limited_task` dans les logs. Le compteur
Prometheus `distillforge_rate_limited_requests_total` expose ces refus.

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

Pour un teacher OpenAI-compatible qui renvoie un JSON de chat completion, le
label peut etre extrait de `choices[0].message.content` :

```sh
python3 tools/build_dataset.py \
  --task-id test_task \
  --logs 'data/logs/*.jsonl' \
  --target-field openai_message_content
```

Un dataset peut ensuite etre augmente avec des exemples synthetiques locaux,
sans appel a une plateforme tierce :

```sh
python3 tools/synthetic_data.py \
  --dataset data/datasets/test_task/ds_example \
  --multiplier 2 \
  --max-synthetic-ratio 0.8
```

Le generateur V1 utilise des templates deterministes, ne modifie que le split
`train`, copie les labels depuis `validated_output` et annote le manifest avec
le ratio synthetique obtenu.

## Training offline minimal

Un modele etudiant classique peut etre entraine depuis un dataset versionne :

```sh
python3 -m pip install -r requirements-training.txt
python3 tools/train_student.py --dataset data/datasets/test_task/ds_example
```

Le trainer produit `model.joblib`, `eval_report.json` et `manifest.json` sous
`models/{task_id}/{model_id}`. Il supporte les students `tfidf`,
`tfidf_mlp`, `embedding_logistic`, `embedding_mlp`,
`hybrid_embedding_tfidf` et `hybrid_embedding_tfidf_mlp`. Le mode par defaut
`tfidf` utilise TF-IDF + LogisticRegression lorsque le dataset contient
plusieurs classes, sinon un baseline `DummyClassifier`.

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

Le mode `bandit` sert le student par defaut tout en conservant un pourcentage
deterministe de probes teacher pour surveiller la qualite :

```sh
python3 tools/promote_model.py \
  --model-dir models/test_task/student_example \
  --mode bandit \
  --teacher-probe-percentage 2 \
  --min-accuracy 0.95
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

## Retention locale

La retention V1 est appliquee par un outil local en dry-run par defaut :

```sh
python3 tools/retention.py
```

Seuils par defaut :

- logs proxy : 90 jours ;
- logs shadow : 90 jours ;
- feedback : 365 jours ;
- datasets : 365 jours ;
- registre/audit : 1095 jours.

Application effective :

```sh
python3 tools/retention.py --apply
```

## Deploiement

Images conteneur :

```sh
docker build -f Dockerfile.proxy -t distillforge/proxy:0.1.0 .
docker build -f Dockerfile.control-plane -t distillforge/control-plane:0.1.0 .
```

Manifests Kubernetes statiques :

```sh
kubectl apply -f deploy/kubernetes/distillforge.yaml
```

Le manifeste cree un namespace `distillforge`, un PVC pour les logs/datasets/
modeles/registre, deux replicas du proxy, un control plane local et les services
HTTP associes. Le snapshot de routage est initialise depuis le ConfigMap puis
stocke dans `/data/config/routing_snapshot.json` pour permettre les promotions
et rollbacks.

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
