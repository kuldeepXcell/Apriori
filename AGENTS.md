
I want to optimize and properly define the structure this time with a lot of thoughts and proper planning.


I am not good with OOPS and class based things so explain me why for those when i ask questions.


we will use uv, venv and python3 always with this project and venv is already present.

Always use context7 and web search to look at latest docs for smooth integration of Openai, qdrant or any other framework,library,service etc.

no need to rush things, take it slow and explain me what you are doing and why you are doing it So that we can make a combined decision.
Take one decision at a time.


structure is in structure.md.

Always refer to this structure befor coding and place things in respective folders/files only.

If "remember" is ever mentioned by user then add that thing to a file called MEMORIES.md.( you can read from this also.)

When introducing shared utilities or moving logic between modules (e.g., pipeline loader, registry, executor), confirm the plan with the developer (explain why/alternatives) before relocating code. Keep YAML-driven pipelines aligned with `structure.md`, ensure reusable helpers live under the designated utils/core folders, and record any new standing rules back in this file.

When serializing sheet_signature metadata for LLM prompts, convert the JSON into the TOON format using python-toon so the context stays compact.

Pipeline plan (for any new agent):
- Goal: pipeline behavior is fully driven by YAML files in `config/pipelines/`.
- `src/app/pipeline/models.py`: define `PipelineConfig` and nested sections (ingestion, retrieval, rerank, eval) so YAML gets parsed into typed objects.
- `src/app/config/loader.py`: implement `load_pipeline_config(name)` that reads YAML, validates via Pydantic/dataclasses, and returns `PipelineConfig`.
- `src/app/pipeline/registry.py`: cache loaded configs and provide helpers like `get_pipeline_config(name)` and `list_pipelines()`.
- `src/app/pipeline/executor.py`: orchestrate ingestion/retrieval/rerank flows by selecting strategies based on the config instead of direct imports. UI/CLI call `executor.run_retrieval(pipeline_name, overrides)` etc.
- Reusable adapters/utilities stay under `src/app/adapters`, `src/app/utils`, or `src/app/core`; pipeline-specific strategy code remains in `src/app/steps`.
- Any decision to move logic into shared modules should be discussed with the developer first, explaining why/alternatives.
