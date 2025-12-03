# Application Flow

> **Purpose**: This document tracks the current state of the application, including active models, technical decisions, and data flow. Agents must read this before making changes and update it after implementation.

## Current Pipeline Configurations

### Active Pipelines
1. **baseline**
   - Embedding: `text-embedding-3-small` (1536 dimensions)
   - LLM: `gpt-4o`
   - Collection: `financial_indicators_baseline`
   - Purpose: High-quality baseline for comparison

2. **fast**
   - Embedding: `text-embedding-ada-002` (1536 dimensions)
   - LLM: `gpt-3.5-turbo`
   - Collection: `financial_indicators_fast`
   - Purpose: Cost-effective alternative for testing

## Data Flow

### One-Time Processes
1. **Initial Ingestion** (`scripts/run_ingestion.py`)
   - Loads canonical indicator data from `country_indicators.json`
   - Generates three dense embeddings per indicator (question, definition, application) plus a BM25 sparse vector derived from keywords
   - Creates multi-vector Qdrant collections per embedding model (named dense vectors + `keywords_sparse_vector`)
   - Upserts indicators with full payload metadata; supports `--range 5-10` (1-indexed inclusive) to ingest only a slice for storage analysis
   - **Status**: Ready for partial or full runs once OpenAI & Qdrant credentials are configured
2. **Keyword Generation** (`scripts/run_keyword_generation_agent.py`)
   - Reads canonical indicator metadata from `country_indicators.json`
   - Invokes the OpenAI Agents SDK-based keyword agent **once per indicator** to create 15 hybrid-search-ready keywords
   - Updates the source JSON in-place (or an optional `--output` path) with a `keywords` array on every indicator plus run metadata; progress is flushed after each indicator and already-processed indicators are skipped on subsequent runs (unless `--force` is provided)
   - **Status**: Ready to run once OpenAI credentials are configured

### Runtime Processes
1. **Query Execution** (Triggered by user in Streamlit UI)
   - User enters natural language query
   - All pipelines execute in parallel (ThreadPoolExecutor)
   - Each pipeline:
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
- **Country Indicators**: 96 financial indicators extracted from `Dataset_directory.xlsx` (Country sheet)
- Data includes: indicator names, normalized names, definitions, use case questions, and application contexts
- JSON format available at `country_indicators.json`
- Indicators cover macroeconomic metrics: GDP, GNI, sectoral breakdowns, etc.

### Expected Format
*Country indicators follow this structure:*
- `subsection`: Category grouping (e.g., "Production-level", "Labor market")
- `subsubsection`: Sub-category (e.g., "Aggregate Output", "Productivity/Efficiency")
- `indicator_name`: Original indicator name
- `normalized_indicator_name`: Standardized name for processing
- `definition`: Detailed explanation of the indicator
- `question`: Use case question for analysis
- `application_context`: How to analyze the indicator

## Recent Changes
- **2025-12-04**: Wired ingestion to multi-vector + sparse hybrid flow
  - Added FastEmbed BM25 encoder and named dense vectors (`question_vector`, `definition_vector`, `application_vector`)
  - Script now loads `country_indicators.json` directly, writing normalized metadata to payloads
  - New `--range` flag enables partial upserts (e.g., ingest indicators 5-10) to inspect Qdrant storage impact before full runs
- **2025-12-02**: Initial multi-pipeline architecture implemented
  - Created pipeline configuration system
  - Implemented parallel execution
  - Built comparison UI
- **2025-12-03**: Added OpenAI Agents SDK keyword generation workflow
  - Introduced `KeywordGenerationAgent` service with strict JSON output parsing
  - Added `scripts/run_keyword_generation_agent.py` CLI to batch process all indicators and store keywords for hybrid search


*Last Updated: 2025-12-04*
