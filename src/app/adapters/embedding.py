"""OpenAI embedding adapter."""

from __future__ import annotations

from typing import List, Sequence

from fastembed import SparseTextEmbedding
from openai import OpenAI

from app.config.settings import settings


DEFAULT_SPARSE_MODEL = "Qdrant/bm25"


class OpenAIEmbeddingAdapter:
    """Small helper around the OpenAI embeddings API."""

    def __init__(self, default_model: str | None = None):
        if not settings.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY is not set.")
        self.client = OpenAI(api_key=settings.OPENAI_API_KEY)
        self.default_model = default_model or "text-embedding-3-small"

    def embed(self, texts: Sequence[str], model: str | None = None) -> List[List[float]]:
        """Embed a batch of texts and return vectors in order."""
        if not texts:
            return []
        model_name = model or self.default_model
        cleaned_inputs = [text.replace("\n", " ") for text in texts]
        response = self.client.embeddings.create(input=cleaned_inputs, model=model_name)
        return [item.embedding for item in response.data]

    def embed_one(self, text: str, model: str | None = None) -> List[float]:
        """Embed a single text and return one vector."""
        vectors = self.embed([text], model=model)
        return vectors[0] if vectors else []


class SparseEmbeddingAdapter:
    """Wrapper for fastembed sparse text embeddings."""

    def __init__(self, model_name: str = DEFAULT_SPARSE_MODEL):
        self.model_name = model_name
        self.encoder = SparseTextEmbedding(model_name=model_name)

    def encode(self, texts: Sequence[str]):
        """Yield sparse embeddings for each text."""
        return self.encoder.embed(texts)

    def encode_keywords(self, keywords: Sequence[str]):
        """Encode a keyword list by joining to a single string."""
        text = " ".join(keywords)
        return self.encoder.embed([text])


openai_embedding_adapter = OpenAIEmbeddingAdapter()
sparse_embedding_adapter = SparseEmbeddingAdapter()

