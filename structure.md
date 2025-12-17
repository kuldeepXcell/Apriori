# Project structure (rebuilt)

config/
  pipelines/                # YAML per pipeline (models, weights, collections, step knobs)
    baseline_hybrid.yaml
    multivector_weighted_v1.yaml
  schemas/                  # JSON/Pydantic schemas to validate pipeline YAML
  evals/                    # Query sets and expectations for regression checks

data/                       # Single bucket (raw + derived)
  country_indicators.json   # Canonical source
  processed/                # Optional: chunks, caches, expansions
  indexes/                  # Optional: exported index snapshots
temporary script/           # One-off helpers for data extraction or migration (non-production)

src/app/
  config/
    settings.py             # Env-backed settings (OpenAI key, Qdrant URL/key, timeouts)
    loader.py               # Load & validate pipeline YAML → PipelineConfig
  core/
    types.py                # Shared dataclasses (Query, Doc, Hit, Context, Answer)
    errors.py               # Custom exceptions
  adapters/                 # Shared infra (pipeline-agnostic)
    llm.py                  # OpenAI chat/completions wrapper
    embeddings.py           # OpenAI embeddings wrapper
    vector_store.py         # Qdrant abstraction (collections, upsert, search dense/sparse/hybrid)
    reranker.py             # Thin rerank helper (can call llm or other backend)
  pipeline/
    executor.py             # Orchestrates steps per pipeline name
    registry.py             # Lookup PipelineConfig by name
    models.py               # PipelineConfig, StepConfig, RuntimeContext
    hooks.py                # Optional tracing/logging/metrics hooks
  steps/
    preprocess/
      clean_query.py
      expand_abbrev.py
      keywords.py
      splitters.py
    ingest/
      embed_and_upsert.py
      rebuild_collection.py
    retrieval/             # Pipeline-specific retrieval modules
      baseline_hybrid.py    # Dense+sparse fusion for baseline_hybrid
      baseline_hybrid_retrieval.py  # Weighted named vectors retrieval
      hyde.py               # HyDE-specific retrieval
    rerank/                # Pipeline-specific rerank modules
      baseline_hybrid.py
      multivector_weighted_v1.py
      hyde.py
    augment/
      context_builder.py    # Build context windows, dedup, select spans
      template.py           # Prompt assembly
    postprocess/
      deduplicate.py
      threshold.py
      safety.py
    generate/
      answer.py             # Final LLM call for answer generation
  ui/
    app.py                  # Streamlit entry; calls pipeline.executor
    components/             # Reusable UI pieces (results tables, debug panes)
  utils/
    timing.py, logging.py, batching.py  # Shared utilities

tests/
  unit/steps/               # Step-level unit tests
  integration/pipelines/    # End-to-end pipeline runs on sample data

docs/
  diagrams/, guides/        # Architecture notes, flows, references

# Example minimal pipeline config

name: multivector_weighted_v1
collection_name: baseline_hybrid
embedding:
  model: text-embedding-3-small
  dimensions: 1536
retrieval:
  top_k_vector: 15
llm_rerank:
  model: gpt-5-nano
  temperature: 0.2
  prompt: |
    You are a reranker. Given a query and a list of retrieved items,
    return the top items ordered by relevance. Consider semantic closeness
    and factual alignment.
