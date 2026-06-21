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
- `student_only`
- `canary`

Si un snapshot pointe vers un modele etudiant absent de `config/example.yaml`,
DistillForge retombe vers le teacher et journalise
`student_backend_missing_teacher_fallback`.

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

Le rapport affiche les volumes, decisions de routage, latences, couts estimes
et economies estimees par tache et par modele.

## Logs redacted

En mode `redacted`, DistillForge capture au maximum
`logging.max_capture_bytes` octets des corps de requete et de reponse, puis
ecrit `prompt_redacted` et `response_redacted` dans le JSONL. Les emails,
tokens bearer, cles `sk-*` et champs JSON courants comme `password`, `token`,
`secret`, `api_key` et `authorization` sont masques.

En mode `metadata_only`, aucun contenu ni hash d'entree n'est stocke.
