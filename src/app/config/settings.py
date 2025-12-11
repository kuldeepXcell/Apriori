from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Env-driven settings for external services."""

    openai_api_key: str
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str | None = None
    openai_timeout: float = 30.0
    log_level: str = "INFO"
    sparse_model_name: str = "Qdrant/bm25"
    llm_rerank_model: str = "gpt-5-mini"
    llm_rerank_temperature: float = 0.1
    embedding_model: str = "text-embedding-3-small"
    embedding_dimensions: int = 1536
    qdrant_collection: str = "baseline_hybrid_auto_keyword"
    definition_vector_name: str = "definition_dense"
    question_vector_name: str = "question_dense"
    context_vector_name: str = "context_dense"
    sparse_vector_name: str = "keywords_sparse"
    langsmith_tracing: bool = False
    langsmith_api_key: str | None = None
    langsmith_endpoint: str | None = None
    langsmith_project: str | None = None
    langsmith_workspace_id: str | None = None

    model_config = {"env_file": ".env", "extra": "ignore"}


settings = Settings()

