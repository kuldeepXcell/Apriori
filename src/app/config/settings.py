"""Environment-backed runtime settings."""

from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict

from app.config import paths

# Load environment variables from the project root .env.
load_dotenv(paths.BASE_DIR / ".env")


class Settings(BaseSettings):
    OPENAI_API_KEY: str | None = None
    KEYWORD_AGENT_MODEL: str = "gpt-5-nano"
    QDRANT_URL: str = "http://localhost:6333"
    QDRANT_API_KEY: str | None = None

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )


settings = Settings()


