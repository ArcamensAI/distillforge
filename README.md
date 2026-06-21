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
