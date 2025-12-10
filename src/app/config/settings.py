from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Env-driven settings for external services."""

    openai_api_key: str
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str | None = None
    openai_timeout: float = 30.0
    log_level: str = "INFO"
    sparse_model_name: str = "Qdrant/bm25"

    model_config = {"env_file": ".env", "extra": "ignore"}


settings = Settings()

