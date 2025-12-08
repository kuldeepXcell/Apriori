"""External service adapters (Qdrant, LLMs, file I/O)."""

from app.adapters.embedding import (
    OpenAIEmbeddingAdapter,
    SparseEmbeddingAdapter,
    openai_embedding_adapter,
    sparse_embedding_adapter,
)
from app.adapters.qdrant import QdrantAdapter, qdrant_adapter

__all__ = [
    "OpenAIEmbeddingAdapter",
    "SparseEmbeddingAdapter",
    "openai_embedding_adapter",
    "sparse_embedding_adapter",
    "QdrantAdapter",
    "qdrant_adapter",
]


