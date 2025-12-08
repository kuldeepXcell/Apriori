# Application Flow

> **Purpose**: This document tracks the current state of the application, including active models, technical decisions, and data flow. Agents must read this before making changes and update it after implementation.

## Current Pipeline Configurations

### Active Pipelines
1. **baseline_hybrid** (`config/pipelines/baseline_hybrid.yaml`)
   - Embedding: `text-embedding-3-small` (1536 dimensions)
   - LLM (reranker): `gpt-5-mini`
   - Collection: `baseline_hybrid`
   - Purpose: Multivector hybrid baseline (dense + sparse)

## Data Flow

### One-Time Processes
1. **Initial Ingestion** (TBD)
   - Reads raw data from `data/`
   - Generates embeddings per pipeline config
   - Creates Qdrant collections
   - Upserts indicators with embeddings
   - **Status**: Not implemented yet (pipeline code is a stub)

### Runtime Processes
1. **Query Execution** (Streamlit UI)
   - User enters natural language query
   - Pipelines execute in parallel (ThreadPoolExecutor)
   - Each pipeline (future):
     - Applies preprocessing (if configured)
     - Generates query embedding
     - Searches Qdrant collection
     - Re-ranks results using LLM
     - Applies postprocessing (if configured)
   - Results displayed side-by-side for comparison

## Technical Decisions

### Why Multiple Collections?
Different embedding models have different dimensions. Qdrant requires fixed dimensions per collection, so each embedding model needs its own collection.

### Why Parallel Execution?
To enable real-time A/B testing and model comparison. Users can see performance and quality differences immediately.

### Current Preprocessing/Postprocessing
- **Preprocessing**: Available but not enabled by default (clean_query, lowercase_query, expand_abbreviations)
- **Postprocessing**: Available but not enabled by default (deduplicate_results, filter_by_threshold)

## Data Sources

### Current Status
- Authoritative source: `data/country_indicators.json`
- Derived artifacts should live in `data/processed/`

### Expected Format
*To be updated when real data is added*

## Recent Changes
- **2025-12-08**: Restructured project layout to `config/`, `data/`, `src/app/`; added YAML loader + registry stubs and documented structure.

---
*Last Updated: 2025-12-08*
