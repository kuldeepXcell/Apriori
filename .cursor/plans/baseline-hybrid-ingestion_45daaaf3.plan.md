---
name: baseline-hybrid-ingestion
overview: Implement ingestion for the baseline_hybrid pipeline to load country indicators, generate dense/sparse vectors, and upsert into Qdrant using named vectors.
todos:
  - id: adapters
    content: Add OpenAI + Qdrant adapter with fastembed sparse
    status: completed
  - id: ingest-step
    content: Implement baseline_hybrid ingestion from country_indicators.json
    status: completed
  - id: runner-docs
    content: Add ingestion runner/CLI and update docs/deps
    status: completed
---

# Baseline Hybrid Ingestion Plan

- Implement embedding + Qdrant adapters: add OpenAI embedding helper for `text-embedding-3-small` and a Qdrant client helper configured via `settings` (URL/API key), including fastembed BM25 sparse encoder for keyword strings. Files: `src/app/adapters/embedding.py`, `src/app/adapters/qdrant.py`.
- Build baseline ingestion step: load `data/country_indicators.json`, join the 15-keyword array into one string, generate three dense vectors (definition, question, application_context) plus a sparse BM25 vector, assemble payload metadata, and create/upsert named-vector collection (`definition`, `question`, `application`, `keywords`) with the dimensions from `config/pipelines/baseline_hybrid.yaml`. File: `src/app/steps/ingest/baseline_hybrid.py`.
- Provide runnable entrypoint + docs: add a small CLI/runner (e.g., `src/app/ingest.py` or script) to trigger the baseline ingestion, and update `application_flow.md` to mark ingestion implemented and describe the run command. Update `pyproject.toml` deps to include fastembed extra for qdrant-client.
- Add concise one-line comments for each function/class/logic block to clarify intent.