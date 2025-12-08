# Project Structure (for agents)

> Quick map of where things live and how to extend the YAML-driven RAG variants.

- `main.py` — Streamlit entrypoint; injects `src/` into `PYTHONPATH` then calls `app.ui.app.run_app`.
- `config/`
  - `pipelines/` — YAML pipeline variants (drives simple if/else dispatch).
  - `schemas/` — reserved for future config validation.
- `data/`
  - `country_indicators.json` — canonical source data.
  - `processed/` — derived artifacts (keywords, chunks, caches).
- `src/app/`
  - `config/paths.py` — project directories.
  - `config/settings.py` — env-backed settings (OpenAI, Qdrant).
  - `config/loader.py` — YAML → `PipelineConfig` normalization.
  - `pipeline/models.py` — `PipelineConfig` and result structs.
  - `pipeline/pipeline.py` — placeholder pipeline (wire in RAG steps here).
  - `pipeline/pipeline_registry.py` — loads all YAML configs.
  - `steps/` — pre-ingest, embedding, ingestion, retrieval implementations (future).
  - `adapters/` — Qdrant + LLM/embedding clients and I/O helpers (future).
  - `ui/app.py` — Streamlit UI rendering logic.
  - `utils/` — shared helpers (add as needed).
- `tests/`
  - `unit/` — step-level tests.
  - `integration/` — end-to-end runs using sample configs and data slices.
- `static/` — images/assets used by the Streamlit app.
- `docs/` — diagrams and reference docs.

How to extend:
- Add a pipeline: drop a YAML file into `config/pipelines/`; include `name`, `description`, `vector_store.collection_name`, `embedding.model`, `reranker.model`, and `retrieval` strategy keys where applicable.
- Implement logic: plug ingestion/retrieval/rerank steps into `src/app/pipeline/pipeline.py` and supporting modules under `steps/` and `adapters/`.
- Data changes: keep the authoritative JSON in `data/`; write derived outputs to `data/processed/`.


