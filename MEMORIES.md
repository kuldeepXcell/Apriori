## Memories

- Logging setup with colored output via `rich`: use `setup_logging(settings.log_level)` (LOG_LEVEL env, default INFO) and `get_logger`. `ModuleName` enum values: pipeline, retrieval, rerank, ingestion, preprocess, ui, adapter, steps. Include `extra={"module_name": ModuleName.<X>}` when logging. Helper lives in `src/app/core/logging.py`; `rich` dependency added in `pyproject.toml`.
- Keyword generator preprocessing step is in `src/app/steps/preprocess/keyword_generator.py`; default keyword model is defined in the file (`DEFAULT_KEYWORD_MODEL = "gpt-4o-mini"`), not env-driven. CLI supports `--model` override; still uses shared logging with ModuleName.PREPROCESS.

