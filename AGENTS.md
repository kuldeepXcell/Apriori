# Project Context: Financial Indicator RAG

## 1. Mission
Build a robust Retrieval-Augmented Generation (RAG) system for financial indicators. The system ingests raw indicator definitions, stores them in a Qdrant vector database (using a hybrid Dense + Sparse approach), and provides a Streamlit-based interface for users to query and retrieve relevant indicators with high precision.

**Experimentation Focus**: This project is designed to support experimentation with different embedding models, LLM models, preprocessing logic, and postprocessing strategies. The architecture enables easy swapping of models and parallel execution to compare results side-by-side in the UI.

## 2. Architecture Overview
The application follows a **pipeline-based experimentation architecture**, allowing multiple retrieval configurations to run in parallel.

- **Frontend**: Streamlit (`main.py`) - Displays results from multiple pipelines side-by-side
- **Pipeline Layer**: `app/pipelines/`
    - Each pipeline represents a unique combination of models and logic
    - Pipelines can run independently or in parallel
- **Service Layer**: `app/services/`
    - `ingestion.py`: Handles data parsing, embedding generation, and Qdrant upsert (supports multiple embedding models).
    - `retrieval.py`: Base retrieval logic (can be extended per pipeline).
    - `llm_service.py`: Model-agnostic wrapper for OpenAI (or other providers).
    - `qdrant_service.py`: Manages multiple collections (one per embedding model).
- **Configuration**: `app/configs/`
    - Pipeline configurations define: embedding model, LLM model, preprocessing, postprocessing.
- **Data Storage**: Qdrant (Vector DB) - Separate collections per embedding model.

## 3. Tech Stack
- **Language**: Python 3.11 (Strict requirement)
- **Package Manager**: `uv` (manage dependencies and virtual environments with `uv venv`)
- **UI Framework**: Streamlit
- **Vector DB**: Qdrant (via `qdrant-client`)
- **LLM/Embeddings**: OpenAI (`text-embedding-3-small` for dense, `gpt-5-mini` or similar for re-ranking/generation)
- **Data Validation**: Pydantic
- **Environment**: `python-dotenv` for secrets.

## 4. Directory Structure
```
/home/ubuntu/Desktop/APriori/Apriori/
├── app/
│   ├── core/           # Config (Env vars)
│   ├── configs/        # Pipeline configurations (JSON/Python)
│   ├── pipelines/      # Pipeline definitions (different model combinations)
│   ├── services/       # Business logic (Ingestion, Retrieval, LLM)
│   └── models/         # Pydantic schemas
├── data/               # Raw input files
├── scripts/            # One-off scripts (e.g., run_ingestion.py)
├── tests/              # Pytest suite
└── main.py             # Streamlit entry point
```

## 5. Operational Rules & Standards
- **Modularity**: Keep UI logic separate from business logic. UI should only call functions in `app/services`.
- **Configuration**: NEVER hardcode API keys. Use `.env` and `app/core/config.py`.
- **Type Safety**: Use Python type hints for all function signatures.
- **Documentation**: Add docstrings to all major functions and classes.
- **Error Handling**: Fail gracefully in the UI. Show user-friendly errors, log technical details.
- **Pathing**: Use absolute paths or robust relative path resolution (e.g., `pathlib`).
- **Structure Evolution**: The defined structure is a baseline. Agents are empowered to adapt, refactor, or expand the architecture and directory structure as project requirements evolve, using their best judgment to maintain code quality and scalability.
- **Experimentation Friendly**: Support easy addition of new models, pipelines, and processing steps. Configuration should drive behavior, not hardcoded logic. Enable parallel pipeline execution for A/B testing.

## 6. Application Flow Documentation
**CRITICAL**: Always maintain `application_flow.md` in the project root. This file documents:
- Current pipeline configurations and active models
- Technical decisions and their rationale
- One-time vs. recurring processes
- Data flow through the system
- Any important implementation details

**Agent Responsibility**: 
- Read `application_flow.md` before making changes to understand current state
- Update it immediately after implementing new features or changing architecture
- Keep it concise but complete - future agents depend on this as source of truth

## 7. Key Workflows
- **Ingestion**: Run `python scripts/run_ingestion.py` to parse `data/` and populate Qdrant collections (one per embedding model). This is a one-time or scheduled process.
- **Querying**: User enters text in Streamlit -> All configured pipelines execute in parallel -> Each pipeline performs: Embedding -> Qdrant Search -> LLM Re-rank -> Results displayed side-by-side for comparison.
- **Adding New Pipeline**: Create a new config in `app/configs/`, optionally create custom pipeline class in `app/pipelines/`, restart the app.
