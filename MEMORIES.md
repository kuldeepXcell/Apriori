## Memories

- Logging setup with colored output via `rich`: use `setup_logging(settings.log_level)` (LOG_LEVEL env, default INFO) and `get_logger`. `ModuleName` enum values: pipeline, retrieval, rerank, ingestion, preprocess, ui, adapter, steps. Include `extra={"module_name": ModuleName.<X>}` when logging. Helper lives in `src/app/core/logging.py`; `rich` dependency added in `pyproject.toml`.
- Keyword generator preprocessing step is in `src/app/steps/preprocess/keyword_generator.py`; default keyword model is defined in the file (`DEFAULT_KEYWORD_MODEL = "gpt-4o-mini"`), not env-driven. CLI supports `--model` override; still uses shared logging with ModuleName.PREPROCESS.
- Always refactor code properly after any change: update function names, docstrings, call sites, variable names, and remove unused code to keep the codebase clean and consistent.
- Model migration: Updated all gpt-4o-mini references to gpt-5-mini across settings (`sql_agent_model`) and pipeline configs (`sample_baseline_hybrid.yaml` rerank model). The multivector pipeline uses gpt-5-nano separately.

