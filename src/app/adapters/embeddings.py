from typing import Any, Iterable

from openai import OpenAI

from app.config.settings import settings

client = OpenAI(api_key=settings.openai_api_key, timeout=settings.openai_timeout)


def embed_texts(texts: Iterable[str], model: str, dimensions: int | None = None) -> list[list[float]]:
    """Embed a batch of texts and return raw vectors."""
    response = client.embeddings.create(model=model, input=list(texts), dimensions=dimensions, timeout=settings.openai_timeout)
    return [item.embedding for item in response.data]

