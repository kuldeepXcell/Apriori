from collections.abc import Iterable
from typing import Any

from langchain_openai import OpenAIEmbeddings

from app.config.settings import settings

_default_embedder = OpenAIEmbeddings(
    model=settings.embedding_model,
    api_key=settings.openai_api_key,
    dimensions=settings.embedding_dimensions,
    timeout=settings.openai_timeout,
    max_retries=settings.openai_max_retries,
)


def _build_embedder(model: str | None = None, dimensions: int | None = None) -> OpenAIEmbeddings:
    if not model and not dimensions:
        return _default_embedder
    return OpenAIEmbeddings(
        model=model or settings.embedding_model,
        api_key=settings.openai_api_key,
        dimensions=dimensions or settings.embedding_dimensions,
        timeout=settings.openai_timeout,
        max_retries=settings.openai_max_retries,
    )


def embed_texts(texts: Iterable[str], model: str | None = None, dimensions: int | None = None) -> list[list[float]]:
    """Embed a batch of texts and return raw vectors."""
    embedder = _build_embedder(model=model, dimensions=dimensions)
    return embedder.embed_documents(list(texts))


def embed_query(text: str, model: str | None = None, dimensions: int | None = None) -> list[float]:
    """Embed a single query text."""
    embedder = _build_embedder(model=model, dimensions=dimensions)
    return embedder.embed_query(text)
