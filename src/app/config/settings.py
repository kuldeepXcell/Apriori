"""Environment-backed runtime settings."""

from pydantic import BaseSettings


class Settings(BaseSettings):
    OPENAI_API_KEY: str | None = None
    QDRANT_URL: str = "http://localhost:6333"
    QDRANT_API_KEY: str | None = None

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


settings = Settings()


