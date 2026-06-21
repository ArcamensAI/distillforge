# Architecture technique DistillForge

Ce document propose une architecture d'implementation pour les specifications de
`specs.md`. Il ne remplace pas les specifications fonctionnelles : il precise les
choix techniques, les frontieres entre composants et une trajectoire pragmatique
pour construire le systeme progressivement.

## Objectifs

DistillForge est un proxy FinOps pour LLM. Il doit :

- rester compatible avec des APIs HTTP de type OpenAI ;
- mesurer les couts, latences, erreurs et decisions de routage ;
- router prudemment vers des modeles eleves moins couteux ;
- permettre shadow mode, canary, fallback et rollback immediat ;
- construire des datasets d'entrainement a partir des usages reels ;
- evaluer automatiquement les modeles avant promotion ;
- eviter les integrations de plateformes tierces non necessaires en V1.

## Principes d'architecture

1. Le chemin critique est en Rust.
   Le proxy doit garder un overhead p95 inferieur a 50 ms. Les decisions de
   routage simples doivent donc se prendre localement dans le proxy, sans appel
   synchrone obligatoire vers un service Python.

2. Python pilote le control plane.
   L'orchestration, l'analytics, les datasets, l'entrainement et l'evaluation
   beneficient de l'ecosysteme Python.

3. HTTP est le protocole prioritaire.
   Les integrations internes et externes doivent exposer des APIs HTTP simples.
   Les protocoles plus specialises peuvent etre ajoutes plus tard si le besoin
   est mesure.

4. DuckDB sert l'analytics, pas l'etat operationnel.
   DuckDB est excellent pour requeter des logs et fichiers Parquet. Il ne doit
   pas etre la source de verite transactionnelle des jobs, regles, permissions
   ou statuts de modeles.

5. Les dependances doivent rester populaires et remplacables.
   Les bibliotheques open source largement adoptees sont acceptables. Les
   integrations fortes avec des plateformes externes sont evitees en V1.

## Vue d'ensemble

```text
Clients
  |
  | OpenAI-compatible HTTP
  v
Rust Pingora Proxy
  |-- validation headers
  |-- routage local
  |-- appels teacher / student
  |-- fallback / shadow / canary
  |-- emission logs et metrics
  |
  +--> Teacher providers HTTP
  |
  +--> Student inference workers HTTP
  |
  +--> JSONL logs append-only
          |
          v
      Parquet / DuckDB analytics
          |
          v
Python Control Plane
  |-- admin API
  |-- orchestration jobs
  |-- dataset builder
  |-- training / evaluation
  |-- model registry
  |-- publication routing snapshots
```

## Choix technologiques

| Besoin | Choix recommande | Raison |
| --- | --- | --- |
| Proxy HTTP | Rust + Pingora | Performance, failover, load balancing, reload, controle fin du dataplane |
| Routage hot path | Rust, en memoire | Evite un aller-retour Python sur chaque requete |
| Configuration routage | Snapshot JSON genere depuis YAML/TOML | Simple a auditer, versionner et recharger atomiquement |
| APIs admin | Python + FastAPI | Productivite, schemas types, OpenAPI |
| Etat operationnel | PostgreSQL | Transactions, concurrence, audit, registry, jobs |
| Analytics FinOps | DuckDB | OLAP local sur JSONL/Parquet, excellent pour agregations |
| Logs bruts | JSONL compresse | Append-only, robuste, facile a inspecter |
| Logs analytiques | Parquet | Format colonne efficace pour DuckDB |
| Jobs async V1 | Table `jobs` PostgreSQL | Suffisant pour demarrer sans broker |
| Jobs async plus tard | Redis/RQ, Dramatiq ou RabbitMQ | Si concurrence et retries deviennent plus complexes |
| Datasets | Parquet + `manifest.json` | Portable, versionnable, lisible par Python/DuckDB |
| Model registry V1 | PostgreSQL + fichiers modele | Evite MLflow au debut, conserve audit et versioning |
| Training | PyTorch, Transformers, scikit-learn, XGBoost/LightGBM | Ecosysteme mature |
| Inference CPU | ONNX Runtime | Bon runtime CPU pour modeles exportables |
| Petits LLM quantifies | GGUF / llama.cpp ou ONNX selon modele | Optimisation CPU, INT4/INT8 |
| Monitoring | Metrics Prometheus-compatible | Standard ouvert, pas de plateforme imposee |
| Dashboard V1 | FastAPI HTML simple ou Streamlit local | Rapide pour piloter FinOps sans produit frontend complet |

## Composants

### 1. Proxy Rust

Responsabilites :

- exposer `POST /v1/chat/completions`, `POST /v1/completions`, `GET /health`,
  `GET /metrics` et `POST /admin/reload-routing` ;
- valider les headers requis, notamment `X-Task-ID` et `X-Client-ID` ;
- generer ou propager `request_id` ;
- charger un snapshot de routage en memoire ;
- appliquer les modes `teacher_only`, `shadow`, `canary`, `student_only` et
  `fallback` ;
- gerer timeouts, retries limites, circuit breakers et backpressure ;
- journaliser chaque interaction en JSONL ;
- exposer les metriques techniques.

Le proxy ne doit pas contenir de logique MLOps complexe. Il applique des
decisions precompilees dans le snapshot de routage.

### 2. Routing engine local

Le routage doit etre une bibliotheque Rust interne au proxy.

Entrees principales :

- `task_id` ;
- `client_id` ;
- `quality_mode` ;
- `pii_level` ;
- disponibilite des modeles ;
- taux d'erreur recent ;
- etat de derive publie par le control plane ;
- pourcentage canary ;
- seuils de confiance et timeout.

Sortie :

```json
{
  "selected_model": "qwen2.5-1.5b-email-classifier-v3",
  "routing_mode": "student",
  "fallback_model": "gpt-4.1",
  "reason": "validated_student_available",
  "requires_shadow_teacher": false
}
```

### 3. Control plane Python

Responsabilites :

- exposer les APIs admin ;
- gerer les tasks, modeles, datasets, evaluations et promotions ;
- stocker audit trail et permissions ;
- publier les snapshots de routage ;
- piloter les jobs d'analyse, dataset, training et evaluation ;
- declencher rollback ou desactivation selon les seuils.

Endpoints cibles :

```http
POST /admin/tasks/{task_id}/train
POST /admin/tasks/{task_id}/evaluate
POST /admin/tasks/{task_id}/promote
POST /admin/tasks/{task_id}/rollback
GET /admin/tasks/{task_id}/status
GET /admin/models
GET /admin/models/{model_id}
```

### 4. Analytics FinOps

DuckDB lit les fichiers JSONL ou Parquet produits par le proxy.

Analyses V1 :

- cout par `task_id`, `client_id`, `cost_center` ;
- latence p50/p95/p99 par modele ;
- taux d'erreur et de fallback ;
- estimation du cout evite ;
- classement des taches candidates a distillation ;
- detection simple de derive sur fenetre glissante.

Le pipeline recommande est :

```text
JSONL brut -> redaction/normalisation -> Parquet partitionne -> DuckDB
```

### 5. Dataset builder

Responsabilites :

- selectionner les logs eligibles ;
- dedupliquer ;
- filtrer erreurs, sorties invalides et donnees non autorisees ;
- anonymiser ou pseudonymiser ;
- separer train / validation / test par groupe logique ;
- produire un dataset versionne.

Structure proposee :

```text
data/datasets/{task_id}/{dataset_id}/
  train.parquet
  validation.parquet
  test.parquet
  manifest.json
```

### 6. Training workers

Le systeme ne doit pas entrainer un petit LLM par defaut. Le choix du modele
doit dependre de la tache.

| Type de tache | Premier choix |
| --- | --- |
| Classification courte | scikit-learn, MiniLM, DistilBERT |
| Extraction JSON simple | T5-small/base ou petit Qwen |
| Scoring tabulaire | XGBoost ou LightGBM |
| Normalisation / mapping | Regles, cache, modele classique |
| Resume court | T5-base ou Qwen 1.5B |
| Conversation ouverte | Teacher par defaut |

Les sorties d'entrainement doivent inclure :

- artefact modele ;
- tokenizer ou preprocessors ;
- rapport d'evaluation ;
- metadonnees de training ;
- cout estime de training ;
- contraintes runtime.

### 7. Student inference workers

Les workers d'inference doivent rester simples et interchangeables.

Responsabilites :

- exposer une API HTTP interne ;
- charger un modele unique ou un petit pool de modeles compatibles ;
- faire le warm-up au demarrage ;
- appliquer timeout, max input length et batch controle ;
- retourner sortie, score de confiance, model_id et latence.

Exemple de reponse interne :

```json
{
  "model_id": "qwen2.5-1.5b-email-classifier-v3",
  "output": "billing_issue",
  "confidence": 0.991,
  "latency_ms": 142
}
```

## Stockage

### PostgreSQL

Tables principales :

- `tasks` ;
- `models` ;
- `datasets` ;
- `evaluations` ;
- `routing_rules` ;
- `routing_snapshots` ;
- `jobs` ;
- `audit_events` ;
- `feedback` ;
- `retention_policies`.

### Fichiers

```text
data/
  logs/
    raw-jsonl/YYYY/MM/DD/*.jsonl.zst
    redacted-parquet/YYYY/MM/DD/*.parquet
  datasets/
    {task_id}/{dataset_id}/
  models/
    {task_id}/{model_id}/
```

Un object storage S3-compatible peut remplacer le filesystem plus tard, sans
changer les formats.

## Snapshot de routage

Le control plane publie un snapshot atomique lu par le proxy.

```json
{
  "version": 42,
  "created_at": "2026-06-21T10:15:30Z",
  "default_mode": "teacher_only",
  "tasks": {
    "email_classification_v1": {
      "mode": "canary",
      "teacher_model": "gpt-4.1",
      "student_model": "qwen2.5-1.5b-email-classifier-v3",
      "student_traffic_percentage": 10,
      "quality_modes": {
        "strict": "teacher_only",
        "balanced": "canary",
        "economy": "student_preferred"
      },
      "fallback": {
        "on_timeout": true,
        "on_error": true,
        "on_low_confidence": true,
        "min_confidence": 0.95
      },
      "timeouts": {
        "student_ms": 2000,
        "teacher_ms": 30000
      }
    }
  }
}
```

Le reload doit etre atomique : si le nouveau snapshot est invalide, le proxy
conserve l'ancien.

## Securite et donnees sensibles

Modes de logs a supporter :

- `metadata_only` ;
- `redacted` par defaut ;
- `full_encrypted` ;
- `disabled`.

La V1 doit au minimum fournir :

- redaction des secrets evidents ;
- hashing des inputs bruts ;
- politique `X-No-Train` respectee ;
- exclusion des `pii_level=high` de l'entrainement par defaut ;
- retention configuree ;
- audit trail des actions admin.

Le chiffrement applicatif complet peut etre ajoute apres validation du pipeline
de base, mais les formats doivent le prevoir.

## Rollout

### Phase 1 : proxy et observabilite

- proxy Pingora en `teacher_only` ;
- compatibilite OpenAI minimale ;
- logs JSONL redacted ;
- cout estime par requete ;
- `/metrics` ;
- requetes DuckDB de base.

### Phase 2 : analytics FinOps

- conversion JSONL vers Parquet ;
- classement des taches candidates ;
- dashboard minimal ;
- schemas PostgreSQL pour tasks, jobs, models et audit.

### Phase 3 : entrainement offline

- dataset builder ;
- premiers modeles classiques et ONNX ;
- evaluation offline ;
- registry maison.

### Phase 4 : shadow mode

- appel student parallele ;
- comparaison teacher/student ;
- rapport de divergence ;
- aucune reponse student au client.

### Phase 5 : canary

- routage 1 %, 10 %, 25 % ;
- fallback automatique ;
- rollback admin ;
- mesure d'economies reelles.

### Phase 6 : production controlee

- bandit simple ;
- teacher probes ;
- detection de derive ;
- reentrainement planifie.

## Decisions a revoir plus tard

- Remplacer les jobs PostgreSQL par Redis, RabbitMQ ou Kafka si le volume le
  justifie.
- Ajouter MLflow si le registry maison devient insuffisant.
- Ajouter OpenSearch si la recherche textuelle de prompts devient un besoin
  produit.
- Ajouter Kubernetes et Helm uniquement quand l'exploitation multi-replicas est
  prioritaire.
- Externaliser les fichiers vers un object storage S3-compatible quand le volume
  depasse le disque local ou qu'un deploiement multi-noeud l'exige.

## Questions ouvertes pour l'implementation

1. Quels endpoints OpenAI-compatible sont strictement necessaires en premier :
   chat uniquement, ou chat + completions ?
2. La V1 doit-elle tourner en mono-machine locale ou viser directement un
   deploiement conteneurise multi-service ?
3. Quel provider teacher doit etre implemente en premier ?
4. Quel niveau de conservation des prompts est acceptable par defaut :
   `metadata_only` ou `redacted` ?
5. Faut-il commencer avec PostgreSQL des la V1, ou accepter SQLite pour un
   prototype local ?
6. Quelle premiere tache metier servira de cas de validation end-to-end ?
