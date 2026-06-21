# Spécifications détaillées — Proxy FinOps pour LLM

## 1. Objectif du projet

Le projet consiste à développer un proxy intelligent placé entre les applications clientes et un ou plusieurs fournisseurs de modèles LLM. Son objectif principal est de réduire les coûts d’inférence en identifiant automatiquement les tâches répétitives, simples ou fortement structurées, puis en les redirigeant vers des modèles plus légers, exécutables sur CPU ou sur infrastructure à coût réduit.

Le système doit permettre de conserver une qualité de réponse équivalente ou quasi équivalente à celle d’un modèle enseignant plus coûteux, tout en diminuant progressivement la dépendance aux grands modèles généralistes.

Le proxy doit être transparent pour les applications clientes : les appels doivent rester compatibles avec les APIs LLM existantes, notamment les interfaces de type OpenAI-compatible API, lorsque cela est pertinent.

## 2. Principes directeurs

Le système doit respecter les principes suivants :

1. **Simplicité d’intégration** : les applications existantes doivent pouvoir utiliser le proxy avec un minimum de modification.
2. **Réduction mesurable des coûts** : chaque décision de routage doit être traçable et associée à une estimation de coût évité.
3. **Qualité contrôlée** : aucun modèle léger ne doit être promu en production sans validation automatique et seuil qualité explicite.
4. **Observabilité complète** : requêtes, réponses, latences, coûts, erreurs, décisions de routage et performances des modèles doivent être journalisés.
5. **Réversibilité immédiate** : le système doit pouvoir revenir au modèle enseignant en cas de dérive, d’erreur ou de baisse de qualité.
6. **Sécurité des données** : les logs, prompts, réponses et datasets doivent être filtrés, anonymisés ou chiffrés selon les exigences de l’organisation.
7. **Extensibilité** : le proxy doit pouvoir supporter plusieurs fournisseurs LLM, plusieurs familles de modèles élèves et plusieurs stratégies de routage.

---

# 3. Périmètre fonctionnel

## 3.1 Fonctionnalités principales

Le système doit fournir les fonctionnalités suivantes :

* Proxy HTTP compatible avec les appels LLM standards.
* Journalisation exhaustive des requêtes et réponses.
* Classification des requêtes par tâche métier.
* Constitution automatique de datasets d’entraînement.
* Génération optionnelle de données synthétiques.
* Entraînement de modèles légers spécialisés.
* Évaluation automatique des modèles élèves.
* Promotion progressive des modèles validés.
* Routage dynamique entre modèle enseignant et modèle élève.
* Détection de dérive de distribution et de qualité.
* Mesure des économies réalisées.
* Tableau de bord d’observabilité et de pilotage.
* Mécanismes de rollback automatique.

## 3.2 Hors périmètre initial

Pour une première version, les éléments suivants peuvent être exclus :

* Fine-tuning de très grands modèles propriétaires.
* Distillation multi-modale.
* Support complet des images, audio et vidéo.
* Apprentissage en ligne temps réel.
* Réentraînement à chaque requête.
* Optimisation GPU avancée.
* Marketplace de modèles.

---

# 4. Architecture générale

## 4.1 Vue d’ensemble

L’architecture est composée des modules suivants :

1. **Proxy HTTP**
2. **Service de routage**
3. **Orchestrateur FinOps / MLOps**
4. **Stockage des logs et datasets**
5. **Nœuds d’inférence**
6. **Nœuds d’entraînement**
7. **Service d’évaluation**
8. **Service de monitoring**
9. **Console d’administration**
10. **Registre des modèles**

## 4.2 Flux nominal

1. Une application cliente envoie une requête au proxy.
2. Le proxy extrait les métadonnées, notamment l’identifiant de tâche.
3. Le service de routage détermine le modèle cible :

   * modèle enseignant coûteux ;
   * modèle élève spécialisé ;
   * fallback vers modèle enseignant ;
   * mode shadow pour comparaison silencieuse.
4. La requête est transmise au modèle sélectionné.
5. La réponse est retournée au client.
6. Le flux complet est journalisé.
7. Les données sont utilisées pour alimenter l’évaluation, l’entraînement et l’analyse FinOps.

---

# 5. Proxy HTTP

## 5.1 Responsabilités

Le proxy HTTP est le point d’entrée unique des requêtes LLM. Il doit :

* accepter les requêtes HTTP entrantes ;
* valider les en-têtes obligatoires ;
* transmettre les requêtes vers le modèle approprié ;
* capturer les réponses ;
* mesurer les latences ;
* journaliser les flux ;
* gérer les erreurs et timeouts ;
* appliquer les règles de sécurité ;
* exposer des métriques techniques.

## 5.2 Compatibilité API

Le proxy doit idéalement supporter une API compatible avec les endpoints suivants :

* `POST /v1/chat/completions`
* `POST /v1/completions`
* `POST /v1/embeddings`, optionnel en V1
* `GET /health`
* `GET /metrics`
* `POST /admin/reload-routing`, réservé à l’administration

## 5.3 En-têtes HTTP requis

Les en-têtes suivants doivent être supportés :

| En-tête          | Obligatoire | Description                                                     |
| ---------------- | ----------: | --------------------------------------------------------------- |
| `X-Task-ID`      |         Oui | Identifiant logique de la tâche métier                          |
| `X-Client-ID`    |         Oui | Identifiant de l’application cliente                            |
| `X-Request-ID`   |         Non | Identifiant unique fourni par le client                         |
| `X-User-ID`      |         Non | Identifiant utilisateur pseudonymisé                            |
| `X-Cost-Center`  |         Non | Centre de coût FinOps                                           |
| `X-Quality-Mode` |         Non | Mode qualité demandé : `strict`, `balanced`, `economy`          |
| `X-No-Train`     |         Non | Si `true`, exclut la requête des datasets d’entraînement        |
| `X-PII-Level`    |         Non | Niveau de sensibilité déclaré : `none`, `low`, `medium`, `high` |

Si `X-Task-ID` est absent, le comportement doit être configurable :

* rejet de la requête avec `400 Bad Request` ;
* routage vers modèle enseignant ;
* classification automatique expérimentale ;
* journalisation dans une catégorie `unknown_task`.

Par défaut, en production, le système doit router vers le modèle enseignant et signaler l’anomalie.

## 5.4 Format de log JSONL

Chaque requête doit produire une entrée JSONL. Une ligne correspond à une interaction complète.

Exemple :

```json
{
  "timestamp": "2026-06-21T10:15:30.123Z",
  "request_id": "req_123456",
  "client_id": "crm_backend",
  "task_id": "email_classification_v1",
  "cost_center": "support",
  "quality_mode": "balanced",
  "input_hash": "sha256:...",
  "prompt_redacted": "Classifie cet email...",
  "response_redacted": "billing_issue",
  "teacher_model": "gpt-4.1",
  "selected_model": "qwen2.5-1.5b-email-classifier-v3",
  "routing_decision": "student",
  "routing_reason": "student_validated_accuracy_98_7",
  "latency_ms": 142,
  "teacher_estimated_latency_ms": 1200,
  "input_tokens": 320,
  "output_tokens": 12,
  "estimated_cost_usd": 0.00008,
  "estimated_teacher_cost_usd": 0.0042,
  "estimated_savings_usd": 0.00412,
  "status": "success",
  "error_code": null,
  "pii_detected": false,
  "training_eligible": true,
  "created_dataset_record": true
}
```

## 5.5 Données sensibles

Le proxy ne doit pas stocker aveuglément les prompts et réponses bruts. Il doit supporter plusieurs modes de journalisation :

| Mode             | Description                                                |
| ---------------- | ---------------------------------------------------------- |
| `metadata_only`  | Stocke uniquement les métadonnées, tokens, coûts, latences |
| `redacted`       | Stocke les prompts/réponses après anonymisation            |
| `full_encrypted` | Stocke le contenu complet chiffré                          |
| `disabled`       | Ne stocke aucun contenu exploitable pour l’entraînement    |

Le mode recommandé par défaut est `redacted`.

---

# 6. Service de routage

## 6.1 Objectif

Le service de routage décide, pour chaque requête, quel modèle doit traiter la demande.

## 6.2 Modes de routage

Le système doit supporter les modes suivants :

| Mode           | Description                                                                      |
| -------------- | -------------------------------------------------------------------------------- |
| `teacher_only` | Toutes les requêtes vont au modèle enseignant                                    |
| `student_only` | Toutes les requêtes vont au modèle élève validé                                  |
| `shadow`       | Le modèle enseignant répond au client, le modèle élève est testé en arrière-plan |
| `canary`       | Une fraction du trafic est envoyée au modèle élève                               |
| `bandit`       | Le routage est adapté dynamiquement selon les performances                       |
| `fallback`     | Retour automatique au modèle enseignant en cas d’échec                           |

## 6.3 Politique de décision

Le routage doit tenir compte des critères suivants :

* `task_id`
* disponibilité d’un modèle élève validé ;
* seuil qualité atteint ;
* latence observée ;
* coût estimé ;
* taux d’erreur récent ;
* mode qualité demandé ;
* niveau de sensibilité des données ;
* règles métier explicites ;
* état de dérive ;
* niveau de confiance du modèle élève.

Exemple de règle :

```yaml
task_id: email_classification_v1
default_model: teacher
validated_students:
  - model_id: qwen2.5-1.5b-email-classifier-v3
    min_accuracy: 0.98
    max_p95_latency_ms: 300
    max_error_rate: 0.01
routing:
  mode: canary
  student_traffic_percentage: 25
fallback:
  on_timeout: true
  on_low_confidence: true
  on_error: true
```

## 6.4 Fallback

Le fallback vers le modèle enseignant doit être immédiat dans les cas suivants :

* timeout du modèle élève ;
* erreur d’inférence ;
* score de confiance insuffisant ;
* tâche non reconnue ;
* modèle élève désactivé ;
* dérive détectée ;
* demande explicite en mode `strict`.

---

# 7. Orchestrateur FinOps / MLOps

## 7.1 Responsabilités

L’orchestrateur est chargé de :

* analyser les logs ;
* détecter les tâches répétitives ;
* estimer le potentiel d’économie ;
* constituer les datasets ;
* lancer la génération synthétique ;
* déclencher les entraînements ;
* piloter les évaluations ;
* promouvoir ou rejeter les modèles élèves ;
* mettre à jour les règles de routage ;
* détecter la dérive ;
* planifier les réentraînements.

## 7.2 Critères de déclenchement d’un entraînement

Un entraînement peut être déclenché si les conditions suivantes sont réunies :

* volume minimal de données atteint ;
* tâche suffisamment stable ;
* coût cumulé significatif ;
* faible diversité excessive des prompts ;
* taux de réussite du modèle enseignant suffisant ;
* données conformes aux règles de sécurité ;
* absence de blocage manuel.

Exemple de seuils configurables :

```yaml
training_trigger:
  min_samples: 1000
  min_unique_users: 10
  min_total_teacher_cost_usd: 50
  min_task_age_days: 7
  max_error_rate_teacher: 0.02
  min_label_balance_ratio: 0.05
```

## 7.3 Sélection automatique du modèle élève

L’orchestrateur doit choisir le type de modèle selon la nature de la tâche :

| Type de tâche                           | Modèle recommandé                                        |
| --------------------------------------- | -------------------------------------------------------- |
| Classification courte                   | BERT-like, DistilBERT, MiniLM, T5-small                  |
| Classification avec consignes complexes | Qwen 0.5B / 1.5B                                         |
| Extraction structurée                   | T5-small, T5-base, Qwen 1.5B                             |
| Résumé court                            | T5-base, Qwen 1.5B                                       |
| Réécriture simple                       | T5-base, Qwen 1.5B                                       |
| Réponse conversationnelle ouverte       | conserver modèle enseignant ou Qwen plus large           |
| Scoring / régression                    | modèle classique, XGBoost, LightGBM ou petit transformer |
| Embeddings / similarité                 | modèle spécialisé d’embeddings                           |

Le système ne doit pas se limiter à Qwen ou T5. Pour des tâches simples, un modèle classique peut être plus performant, moins coûteux et plus facile à maintenir.

---

# 8. Gestion des données

## 8.1 Dataset de base

Chaque exemple d’entraînement doit contenir :

```json
{
  "task_id": "email_classification_v1",
  "input": "...",
  "teacher_output": "...",
  "validated_output": "...",
  "metadata": {
    "client_id": "crm_backend",
    "timestamp": "2026-06-21T10:15:30Z",
    "teacher_model": "gpt-4.1",
    "latency_ms": 1200,
    "cost_usd": 0.0042
  }
}
```

## 8.2 Nettoyage des données

Avant entraînement, le pipeline doit appliquer :

* suppression des doublons ;
* anonymisation ou pseudonymisation ;
* suppression des données sensibles non autorisées ;
* filtrage des requêtes en erreur ;
* filtrage des réponses incohérentes ;
* équilibrage des classes ;
* séparation train / validation / test ;
* détection des prompts adversariaux ;
* normalisation du format de sortie.

## 8.3 Split des données

Par défaut :

* 70 % entraînement ;
* 15 % validation ;
* 15 % test.

Pour éviter les fuites de données, le split doit être effectué par regroupement logique lorsque c’est possible :

* par utilisateur ;
* par client ;
* par conversation ;
* par période temporelle.

## 8.4 Données synthétiques

Le système doit permettre de générer des données synthétiques afin d’augmenter le volume et la diversité du dataset.

Paramètre global :

```yaml
synthetic_data:
  enabled: true
  multiplier: 5
  max_synthetic_ratio: 0.8
  teacher_model: "gpt-4.1"
  require_validation: true
```

Le paramètre `multiplier` indique le nombre maximal d’exemples synthétiques générés par exemple réel.

Exemple : avec `multiplier = 5`, 1 000 exemples réels peuvent produire jusqu’à 5 000 exemples synthétiques.

Cependant, les données synthétiques ne doivent jamais remplacer totalement les données réelles. Le ratio synthétique maximal doit être configurable.

## 8.5 Validation des données synthétiques

Les exemples synthétiques doivent être filtrés selon :

* conformité au schéma de sortie ;
* absence de données personnelles inventées ;
* diversité suffisante ;
* cohérence avec la tâche ;
* validation par le modèle enseignant ;
* score de confiance minimal ;
* absence de contradiction avec les exemples réels.

---

# 9. Entraînement des modèles élèves

## 9.1 Objectif

L’entraînement vise à produire un modèle spécialisé capable d’exécuter une tâche définie avec un coût et une latence inférieurs au modèle enseignant.

## 9.2 Types d’entraînement

Le système doit supporter :

* fine-tuning supervisé ;
* distillation par imitation des sorties du modèle enseignant ;
* entraînement sur labels métier ;
* entraînement hybride labels humains + sorties enseignant ;
* entraînement de modèles classiques pour tâches simples ;
* quantization post-training ;
* export optimisé ONNX, GGUF ou équivalent.

## 9.3 Modèles cibles initiaux

Les familles suivantes doivent être supportées en priorité :

* Qwen 2.5 0.5B / 1.5B / 3B ;
* T5-small / T5-base ;
* DistilBERT / MiniLM pour classification ;
* XGBoost / LightGBM pour scoring tabulaire ;
* modèles d’embeddings spécialisés si nécessaire.

## 9.4 Configuration d’entraînement

Exemple :

```yaml
training_job:
  task_id: email_classification_v1
  base_model: qwen2.5-1.5b
  dataset_version: ds_email_classification_2026_06_21
  epochs: 3
  batch_size: 16
  learning_rate: 0.00002
  max_sequence_length: 2048
  early_stopping: true
  quantization: int8
  target_runtime: cpu
```

## 9.5 Optimisation CPU

Pour que le modèle soit réellement intéressant FinOps, il doit être optimisé pour l’inférence CPU lorsque c’est l’objectif de déploiement.

Optimisations attendues :

* quantization INT8 ou INT4 selon tolérance qualité ;
* batching contrôlé ;
* cache de tokenizer ;
* compilation ou export ONNX si applicable ;
* limitation de la longueur maximale ;
* warm-up au démarrage ;
* pool de workers ;
* contrôle strict des timeouts.

---

# 10. Évaluation des modèles

## 10.1 Objectif

Un modèle élève ne peut être promu que s’il démontre une qualité suffisante par rapport au modèle enseignant ou à une vérité terrain validée.

## 10.2 Métriques par type de tâche

| Type de tâche         | Métriques                                                    |
| --------------------- | ------------------------------------------------------------ |
| Classification        | accuracy, precision, recall, F1, matrice de confusion        |
| Extraction JSON       | exact match, valid JSON rate, field-level accuracy           |
| Résumé                | similarité sémantique, factualité, longueur, taux d’omission |
| Réécriture            | similarité sémantique, préservation du sens, style           |
| Régression            | MAE, RMSE, R²                                                |
| Génération contrainte | conformité au schéma, taux de rejet                          |
| Conversation          | évaluation LLM-as-judge + échantillonnage humain             |

## 10.3 Seuils de validation

Les seuils doivent être configurables par tâche.

Exemple :

```yaml
validation:
  min_accuracy_vs_teacher: 0.98
  min_f1: 0.97
  max_json_invalid_rate: 0.005
  max_latency_p95_ms: 300
  max_cost_ratio_vs_teacher: 0.2
  min_eval_samples: 500
```

## 10.4 Comparaison au modèle enseignant

Deux approches sont possibles :

1. **Comparaison à la sortie du modèle enseignant** : utile pour la distillation.
2. **Comparaison à une vérité terrain validée** : préférable lorsque des labels métier fiables existent.

Lorsque des labels humains ou métier existent, ils doivent primer sur les sorties du modèle enseignant.

## 10.5 Promotion du modèle

Un modèle peut être promu uniquement si :

* les seuils de qualité sont atteints ;
* les seuils de latence sont atteints ;
* l’économie estimée est significative ;
* le modèle est stable sur plusieurs runs ;
* le taux d’erreur technique est acceptable ;
* aucun test de sécurité n’échoue ;
* le modèle produit des sorties conformes au format attendu.

La promotion doit suivre une séquence progressive :

1. `offline_eval`
2. `shadow`
3. `canary_1_percent`
4. `canary_10_percent`
5. `canary_25_percent`
6. `production`
7. `full_student`, si autorisé

---

# 11. Algorithme de bandit manchot et contrôle de dérive

## 11.1 Objectif

Après promotion d’un modèle élève, le système doit continuer à comparer périodiquement ses performances avec celles du modèle enseignant afin de détecter toute dérive de qualité ou de distribution.

## 11.2 Principe

Une fraction du trafic est envoyée au modèle enseignant en parallèle ou à la place du modèle élève. Les résultats sont comparés pour vérifier que le modèle élève reste fiable.

## 11.3 Paramètres configurables

```yaml
bandit:
  enabled: true
  teacher_probe_rate: 0.02
  min_daily_teacher_samples: 100
  max_daily_teacher_samples: 5000
  divergence_threshold: 0.03
  rolling_window_hours: 24
  action_on_drift: retrain
```

## 11.4 Modes de comparaison

| Mode               | Description                                                              |
| ------------------ | ------------------------------------------------------------------------ |
| `parallel_shadow`  | Le client reçoit la réponse élève, le teacher est appelé en arrière-plan |
| `teacher_probe`    | Une petite fraction du trafic est servie directement par le teacher      |
| `confidence_based` | Le teacher est appelé lorsque le modèle élève est peu confiant           |
| `random_sampling`  | Échantillonnage aléatoire contrôlé                                       |

## 11.5 Déclencheurs de dérive

Une dérive doit être signalée si :

* divergence excessive avec le modèle enseignant ;
* chute de score sur les données récentes ;
* hausse des erreurs de format ;
* hausse des fallbacks ;
* changement significatif de distribution des prompts ;
* augmentation de la latence ;
* baisse du score de confiance ;
* hausse des réclamations ou corrections humaines.

## 11.6 Actions en cas de dérive

Selon la gravité :

* alerte uniquement ;
* augmentation temporaire du trafic teacher ;
* passage en mode shadow ;
* désactivation du modèle élève ;
* déclenchement d’un réentraînement ;
* retour complet au modèle enseignant.

---

# 12. Registre des modèles

## 12.1 Objectif

Le registre des modèles conserve l’historique complet des modèles entraînés, évalués et déployés.

## 12.2 Métadonnées requises

Chaque modèle doit avoir :

```json
{
  "model_id": "qwen2.5-1.5b-email-classifier-v3",
  "task_id": "email_classification_v1",
  "base_model": "qwen2.5-1.5b",
  "dataset_version": "ds_email_classification_2026_06_21",
  "status": "production",
  "created_at": "2026-06-21T12:00:00Z",
  "metrics": {
    "accuracy": 0.987,
    "f1": 0.982,
    "p95_latency_ms": 210,
    "cost_ratio_vs_teacher": 0.08
  },
  "deployment": {
    "runtime": "cpu",
    "quantization": "int8",
    "replicas": 4
  }
}
```

## 12.3 Statuts possibles

* `training`
* `failed_training`
* `offline_eval`
* `rejected`
* `shadow`
* `canary`
* `production`
* `deprecated`
* `disabled`

---

# 13. Observabilité et FinOps

## 13.1 Métriques techniques

Le système doit exposer au minimum :

* nombre de requêtes par tâche ;
* latence moyenne, p50, p95, p99 ;
* taux d’erreur ;
* taux de fallback ;
* débit par modèle ;
* saturation CPU / mémoire ;
* file d’attente d’inférence ;
* temps d’entraînement ;
* disponibilité des services.

## 13.2 Métriques FinOps

Le système doit mesurer :

* coût réel ou estimé par requête ;
* coût par tâche ;
* coût par client ;
* coût par centre de coût ;
* coût évité ;
* économie cumulée ;
* coût d’entraînement ;
* délai de retour sur investissement ;
* ratio coût teacher / student ;
* économies par modèle élève.

## 13.3 Exemple de calcul d’économie

```text
coût évité = coût estimé teacher - coût réel student - coût additionnel de monitoring
```

Le coût d’entraînement doit être amorti sur la durée d’utilisation du modèle.

```text
ROI = économies cumulées / coût total d'entraînement et d'exploitation
```

## 13.4 Dashboards recommandés

La console doit présenter :

* top tâches les plus coûteuses ;
* top tâches avec potentiel de distillation ;
* modèles élèves actifs ;
* économies réalisées ;
* qualité par modèle ;
* alertes de dérive ;
* historique des promotions ;
* taux de fallback ;
* taux de requêtes servies par CPU.

---

# 14. Sécurité, conformité et gouvernance

## 14.1 Sécurité des données

Le système doit intégrer :

* chiffrement en transit ;
* chiffrement au repos ;
* anonymisation des prompts ;
* détection de PII ;
* masquage des secrets ;
* contrôle d’accès par rôle ;
* audit trail ;
* politiques de rétention ;
* suppression sélective des données ;
* exclusion de certaines tâches de l’entraînement.

## 14.2 Contrôle d’accès

Rôles recommandés :

| Rôle             | Permissions                          |
| ---------------- | ------------------------------------ |
| `viewer`         | Consultation des métriques           |
| `finops_admin`   | Analyse des coûts, règles de routage |
| `ml_engineer`    | Lancement entraînements, évaluations |
| `security_admin` | Politiques de données et rétention   |
| `platform_admin` | Administration complète              |

## 14.3 Rétention

Exemple de politique :

```yaml
retention:
  raw_logs_days: 7
  redacted_logs_days: 90
  datasets_days: 365
  metrics_days: 730
  audit_logs_days: 1095
```

## 14.4 Audit

Chaque action administrative doit être journalisée :

* modification de règle de routage ;
* promotion d’un modèle ;
* désactivation d’un modèle ;
* lancement d’un entraînement ;
* suppression de données ;
* modification de seuil qualité ;
* changement de politique de sécurité.

---

# 15. APIs internes

## 15.1 API de routage

Endpoint :

```http
POST /internal/route
```

Entrée :

```json
{
  "task_id": "email_classification_v1",
  "client_id": "crm_backend",
  "quality_mode": "balanced",
  "input_tokens": 320
}
```

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

## 15.2 API d’orchestration

Endpoints recommandés :

```http
POST /admin/tasks/{task_id}/train
POST /admin/tasks/{task_id}/evaluate
POST /admin/tasks/{task_id}/promote
POST /admin/tasks/{task_id}/rollback
GET /admin/tasks/{task_id}/status
GET /admin/models
GET /admin/models/{model_id}
```

## 15.3 API de feedback humain

Pour améliorer les modèles, le système doit permettre aux applications clientes d’envoyer un feedback.

```http
POST /v1/feedback
```

Exemple :

```json
{
  "request_id": "req_123456",
  "rating": "bad",
  "correct_output": "billing_issue",
  "comment": "La réponse aurait dû classer l'email en facturation."
}
```

---

# 16. Stockage

## 16.1 Types de stockage

Le système doit utiliser plusieurs stockages spécialisés :

| Besoin               | Technologie possible           |
| -------------------- | ------------------------------ |
| Logs JSONL bruts     | Object storage S3-compatible   |
| Métadonnées requêtes | PostgreSQL                     |
| Métriques temps réel | Prometheus / VictoriaMetrics   |
| Datasets versionnés  | Object storage + DVC ou LakeFS |
| Registre modèles     | MLflow ou équivalent           |
| Files de jobs        | Redis, RabbitMQ, Kafka         |
| Recherche de prompts | OpenSearch, optionnel          |

## 16.2 Versioning des datasets

Chaque dataset doit être versionné avec :

* identifiant unique ;
* période couverte ;
* nombre d’exemples ;
* ratio réel / synthétique ;
* stratégie de filtrage ;
* hash de contenu ;
* tâche associée ;
* modèle enseignant utilisé.

---

# 17. Performance et scalabilité

## 17.1 Objectifs de performance

Objectifs recommandés pour la V1 :

| Métrique                 |                     Objectif |
| ------------------------ | ---------------------------: |
| Overhead proxy p95       |                      < 50 ms |
| Disponibilité proxy      |                       99,9 % |
| Latence student p95      | < 300 ms pour tâches simples |
| Taux d’erreur proxy      |                      < 0,1 % |
| Taux fallback injustifié |                        < 2 % |
| Perte qualité maximale   |       configurable par tâche |

## 17.2 Scalabilité

Le système doit supporter :

* scaling horizontal du proxy ;
* scaling séparé des workers d’inférence ;
* queues pour entraînement asynchrone ;
* limitation de débit par client ;
* circuit breakers ;
* backpressure ;
* timeouts configurables.

## 17.3 Timeouts recommandés

```yaml
timeouts:
  proxy_total_timeout_ms: 30000
  student_inference_timeout_ms: 2000
  teacher_inference_timeout_ms: 30000
  shadow_teacher_timeout_ms: 60000
```

---

# 18. Configuration

## 18.1 Exemple de configuration globale

```yaml
proxy:
  require_task_id: true
  default_missing_task_behavior: teacher_fallback
  log_mode: redacted

routing:
  default_mode: teacher_only
  enable_student_routing: true
  enable_shadow: true
  enable_canary: true

synthetic_data:
  enabled: true
  multiplier: 5
  max_synthetic_ratio: 0.8

training:
  min_samples: 1000
  default_epochs: 3
  enable_cpu_optimized_models: true

validation:
  default_min_accuracy_vs_teacher: 0.98
  default_max_latency_p95_ms: 300

bandit:
  enabled: true
  teacher_probe_rate: 0.02
  divergence_threshold: 0.03

security:
  pii_detection_enabled: true
  encrypt_logs: true
  raw_log_retention_days: 7
```

---

# 19. Déploiement

## 19.1 Environnements

Le système doit prévoir :

* `dev`
* `staging`
* `production`

## 19.2 Déploiement recommandé

Les composants doivent être conteneurisés.

Architecture possible :

* Kubernetes pour l’orchestration ;
* Helm charts pour le déploiement ;
* autoscaling horizontal ;
* node pools CPU pour inférence légère ;
* node pools GPU optionnels pour entraînement ;
* object storage pour logs et datasets ;
* Prometheus + Grafana pour monitoring.

## 19.3 Haute disponibilité

Le proxy doit être déployé avec au moins deux réplicas en production.

Les composants critiques doivent être redondants :

* proxy ;
* service de routage ;
* base de métadonnées ;
* stockage des règles ;
* registre modèles.

---

# 20. Stratégie de rollout

## 20.1 Phase 1 — Proxy et observabilité

Objectif : mesurer sans modifier le comportement.

Fonctionnalités :

* proxy HTTP ;
* logs JSONL ;
* extraction `X-Task-ID` ;
* estimation des coûts ;
* dashboard de base ;
* routage `teacher_only`.

## 20.2 Phase 2 — Analyse FinOps

Objectif : identifier les tâches candidates.

Fonctionnalités :

* agrégation par tâche ;
* classement par coût ;
* détection des tâches répétitives ;
* estimation du potentiel d’économie ;
* sélection manuelle des premières tâches.

## 20.3 Phase 3 — Entraînement offline

Objectif : entraîner les premiers modèles élèves.

Fonctionnalités :

* génération de datasets ;
* nettoyage ;
* entraînement ;
* évaluation offline ;
* registre modèles.

## 20.4 Phase 4 — Shadow mode

Objectif : tester sans risque client.

Fonctionnalités :

* inférence parallèle ;
* comparaison teacher/student ;
* monitoring qualité ;
* rapport de divergence.

## 20.5 Phase 5 — Canary

Objectif : envoyer une faible part du trafic vers le modèle élève.

Fonctionnalités :

* routage 1 %, 10 %, 25 % ;
* fallback automatique ;
* alertes ;
* mesure des économies réelles.

## 20.6 Phase 6 — Production contrôlée

Objectif : routage dynamique optimisé.

Fonctionnalités :

* bandit manchot ;
* détection de dérive ;
* réentraînement automatique ;
* politique de rollback ;
* dashboards FinOps avancés.

---

# 21. Critères d’acceptation

## 21.1 Critères proxy

Le proxy est accepté si :

* il accepte les requêtes compatibles API LLM ;
* il extrait correctement `X-Task-ID` ;
* il journalise toutes les requêtes en JSONL ;
* il mesure les latences ;
* il gère les erreurs et timeouts ;
* il ajoute moins de 50 ms de latence p95 ;
* il supporte le fallback enseignant.

## 21.2 Critères orchestration

L’orchestrateur est accepté si :

* il identifie les tâches candidates ;
* il construit un dataset versionné ;
* il déclenche un entraînement ;
* il lance une évaluation automatique ;
* il enregistre le modèle dans le registre ;
* il met à jour les règles de routage après validation.

## 21.3 Critères modèle élève

Un modèle élève est accepté si :

* il atteint les seuils qualité configurés ;
* il réduit le coût par requête ;
* il réduit ou maintient la latence ;
* il respecte le format de sortie attendu ;
* il supporte la charge cible ;
* il peut être désactivé immédiatement.

## 21.4 Critères FinOps

Le module FinOps est accepté si :

* il calcule le coût par tâche ;
* il estime le coût évité ;
* il agrège les économies par client ;
* il affiche les économies cumulées ;
* il prend en compte le coût d’entraînement ;
* il produit un rapport exportable.

---

# 22. Risques et mesures de mitigation

| Risque                               | Impact | Mitigation                                   |
| ------------------------------------ | ------ | -------------------------------------------- |
| Baisse de qualité du modèle élève    | Fort   | Shadow, canary, fallback, seuils stricts     |
| Dérive des données                   | Fort   | Bandit, probes teacher, réentraînement       |
| Logs contenant des données sensibles | Fort   | Anonymisation, chiffrement, rétention courte |
| Surcoût lié aux appels shadow        | Moyen  | Échantillonnage contrôlé                     |
| Mauvaise classification des tâches   | Moyen  | `X-Task-ID` obligatoire, validation métier   |
| Données synthétiques biaisées        | Moyen  | Ratio maximal, validation teacher, filtrage  |
| Modèle CPU trop lent                 | Moyen  | Quantization, ONNX, choix modèle plus simple |
| Complexité MLOps excessive           | Moyen  | Déploiement progressif, V1 limitée           |

---

# 23. Recommandations de conception

## 23.1 Ne pas utiliser un LLM léger pour tout

Certaines tâches seront mieux servies par :

* règles déterministes ;
* regex ;
* classifieur classique ;
* modèle d’embeddings ;
* XGBoost ;
* moteur de recherche lexical ;
* cache de réponses.

L’orchestrateur doit donc recommander la solution la plus économique, pas uniquement un petit LLM.

## 23.2 Commencer par les tâches à ROI rapide

Les meilleures tâches candidates sont généralement :

* classification d’emails ;
* extraction de champs ;
* catégorisation de tickets ;
* reformulation courte ;
* normalisation de texte ;
* génération de JSON contraint ;
* scoring simple ;
* routage d’intention ;
* résumé court et structuré.

## 23.3 Préférer la vérité terrain au modèle enseignant

Le modèle enseignant peut se tromper. Lorsque des labels métier ou corrections humaines existent, ils doivent être utilisés comme référence prioritaire.

## 23.4 Prévoir une console de contrôle humain

Même avec automatisation, les actions suivantes doivent pouvoir être validées ou bloquées manuellement :

* promotion en production ;
* changement de seuil qualité ;
* suppression de dataset ;
* activation du bandit ;
* passage en `student_only`.

---

# 24. Exemple de parcours complet

1. Le service support appelle le proxy avec `X-Task-ID: email_classification_v1`.
2. Pendant 7 jours, toutes les requêtes sont servies par le modèle enseignant.
3. Le proxy collecte 15 000 exemples.
4. L’orchestrateur détecte un coût cumulé de 800 USD.
5. Le système génère un dataset nettoyé et versionné.
6. Des données synthétiques sont générées avec un multiplicateur de 3.
7. Un modèle Qwen 1.5B quantifié INT8 est entraîné.
8. L’évaluation offline atteint 98,7 % d’accuracy.
9. Le modèle passe en shadow mode pendant 48 heures.
10. La divergence reste inférieure à 2 %.
11. Le modèle passe en canary à 10 %.
12. Aucun incident n’est détecté.
13. Le routage passe progressivement à 75 % student.
14. Le système mesure 82 % d’économie sur cette tâche.
15. Le bandit conserve 2 % d’appels teacher pour surveiller la dérive.

---

# 25. Livrables attendus

## 25.1 Livrables techniques

* Code du proxy HTTP.
* Service de routage.
* Orchestrateur.
* Pipeline dataset.
* Pipeline entraînement.
* Pipeline évaluation.
* Registre modèles.
* Dashboards.
* Documentation API.
* Helm charts ou manifests de déploiement.
* Tests unitaires et d’intégration.

## 25.2 Livrables documentation

* Guide d’intégration client.
* Guide d’exploitation.
* Guide sécurité.
* Guide MLOps.
* Guide FinOps.
* Procédure de rollback.
* Procédure d’ajout d’une nouvelle tâche.
* Procédure de validation d’un modèle.

---

# 26. Résumé exécutif

Le Proxy FinOps pour LLM doit devenir une couche d’optimisation intelligente entre les applications et les modèles coûteux. Il observe d’abord les usages, identifie les tâches répétitives, entraîne des modèles spécialisés moins chers, les valide rigoureusement, puis route progressivement le trafic vers ces modèles tout en maintenant un contrôle qualité permanent.

La réussite du projet dépend de trois points clés :

1. **Une instrumentation exhaustive des coûts et de la qualité.**
2. **Une promotion prudente des modèles élèves.**
3. **Un fallback immédiat vers le modèle enseignant en cas de doute.**

Le système doit être conçu comme une plateforme progressive : d’abord observabilité, puis entraînement offline, puis shadow mode, puis canary, puis optimisation dynamique.




