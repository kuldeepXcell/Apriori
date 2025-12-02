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
   - Reads raw data from `data/` directory
   - Generates embeddings for each active pipeline config
   - Creates Qdrant collections (one per embedding model)
   - Upserts indicators with embeddings to respective collections
   - **Status**: Not yet run (waiting for real data)

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
- Using dummy data (2 placeholder financial indicators)
- Awaiting real financial indicator files in `data/` directory

### Expected Format
*To be updated when real data is added*

## Recent Changes
- **2025-12-02**: Initial multi-pipeline architecture implemented
  - Created pipeline configuration system
  - Implemented parallel execution
  - Built comparison UI

---
*Last Updated: 2025-12-02*
