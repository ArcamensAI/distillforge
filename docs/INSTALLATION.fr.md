# Installation de DistillForge

Ce guide installe DistillForge pour le developpement local et les demos
offline. DistillForge ne depend pas d'une plateforme managee : le proxy est en
Rust, l'outillage control/training est en Python, les logs sont en JSONL et
l'analytics utilise DuckDB.

## Prerequis

- macOS ou Linux.
- Rust stable avec Cargo.
- Python 3.9+.
- Docker, optionnel, pour les builds conteneur.
- `curl`, utile pour les smoke tests.

Recommande sur macOS :

```sh
xcode-select --install
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

Verifier la toolchain :

```sh
rustc --version
cargo --version
python3 --version
```

## Clone

```sh
git clone git@github.com:ArcamensAI/distillforge.git
cd distillforge
```

Si vous utilisez un autre remote, les commandes suivantes restent identiques.

## Environnement Python

Creer un environnement virtuel isole :

```sh
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
```

Installer la dependance minimale d'analytics :

```sh
python3 -m pip install -r requirements-analytics.txt
```

Installer les dependances de training et d'inference locale quand vous voulez
entrainer des students, utiliser BGE ou generer des datasets :

```sh
python3 -m pip install -r requirements-training.txt
```

Pour les teachers LLM locaux Apple Silicon bases sur MLX, installer `mlx-lm`
comme dependance optionnelle :

```sh
python3 -m pip install mlx-lm
```

## Build Rust

Compiler et tester le proxy :

```sh
cargo build
cargo test
```

Lancer le proxy par defaut :

```sh
cargo run --bin distillforge
```

Par defaut, le proxy lit `config/example.yaml`, ecoute sur `127.0.0.1:6188` et
expose :

- `POST /v1/chat/completions`
- `POST /v1/completions`
- `POST /admin/reload-routing`
- `GET /metrics`

Utiliser une configuration specifique :

```sh
DISTILLFORGE_CONFIG=examples/cfpb_complaints/config.local_llm.yaml \
cargo run --bin distillforge
```

## Smoke Test

Dans un autre terminal, lancer le worker student deterministe :

```sh
DISTILLFORGE_STUDENT_RESPONSE="student ok" cargo run --bin student_worker
```

Puis appeler le proxy :

```sh
curl -sS http://127.0.0.1:6188/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -H 'X-Client-ID: demo_client' \
  -H 'X-Task-ID: test_task' \
  -d '{"model":"teacher","messages":[{"role":"user","content":"hello"}]}'
```

La reponse exacte depend du snapshot de routage actif et des backends
teacher/student configures.

## Docker

Construire l'image proxy :

```sh
docker build -f Dockerfile.proxy -t distillforge-proxy .
```

La lancer avec la configuration d'exemple :

```sh
docker run --rm -p 6188:6188 distillforge-proxy
```

Construire l'image control-plane :

```sh
docker build -f Dockerfile.control-plane -t distillforge-control-plane .
```

## Analytics

Generer un rapport FinOps depuis les logs JSONL :

```sh
python3 tools/finops_report.py --logs 'data/logs/*.jsonl'
```

Generer un dashboard HTML autonome :

```sh
python3 tools/finops_dashboard.py --logs 'data/logs/*.jsonl'
```

## Training Student

Construire un dataset depuis les logs eligibles :

```sh
python3 tools/build_dataset.py \
  --task-id test_task \
  --logs 'data/logs/*.jsonl' \
  --target-field openai_message_content
```

Entrainer le student TF-IDF par defaut :

```sh
python3 tools/train_student.py \
  --dataset data/datasets/test_task/ds_example \
  --student-kind tfidf
```

Entrainer un student neural local base sur BGE :

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

Les flags `HF_HUB_OFFLINE` et `TRANSFORMERS_OFFLINE` supposent que le modele
est deja present dans le cache Hugging Face local. Retirez-les uniquement si
vous voulez telecharger un modele.

Servir un student entraine :

```sh
python3 tools/student_inference.py \
  --model-dir examples/cfpb_complaints/models/cfpb_product_triage_v1/cfpb_product_student_bge_m3_hybrid_mlp_2000 \
  --listen 127.0.0.1:9102
```

## Configuration

La configuration active du proxy est choisie avec `DISTILLFORGE_CONFIG`.
Fichiers importants :

- `config/example.yaml` : configuration locale par defaut.
- `config/routing_snapshot.json` : snapshot de routage par defaut.
- `examples/groq_banking77/config.yaml` : demo Groq + BANKING77.
- `examples/cfpb_complaints/config.local_llm.yaml` : demo teacher LLM local.

Les secrets doivent etre places dans `.env` et ne doivent pas etre commites.
Exemple :

```sh
GROQ_API_KEY=...
GROQ_MODEL=llama-3.1-8b-instant
```

## Kubernetes

Les manifestes statiques sont disponibles ici :

```sh
deploy/kubernetes/distillforge.yaml
```

Les appliquer apres avoir construit et publie une image accessible par le
cluster :

```sh
kubectl apply -f deploy/kubernetes/distillforge.yaml
```

## Depannage

- `PermissionError` pendant `py_compile` : definir
  `PYTHONPYCACHEPREFIX=/tmp/distillforge_pycache`.
- Training BGE-M3 lent sur documents longs : utiliser `--max-input-chars` et
  `--embedding-max-seq-length`.
- Mode Hugging Face offline sans modele local : lancer une fois sans
  `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1` pour remplir le cache.
- Port deja utilise : changer l'adresse dans la config ou via le flag
  `--host`, `--port` ou `--listen` concerne.
