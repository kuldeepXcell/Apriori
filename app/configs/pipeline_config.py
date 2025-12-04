from __future__ import annotations

# from pathlib import Path
from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field, field_validator


class NamedVectorConfig(BaseModel):
    """Configuration for an individual named vector within a collection."""

    type: Literal["dense", "sparse", "multivector"]
    dim: Optional[int] = Field(
        default=None, description="Dimension for dense or multivectors."
    )

    @field_validator("dim")
    @classmethod
    def validate_dim_for_dense(cls, value: Optional[int], info) -> Optional[int]:
        vector_type = info.data.get("type", "dense")
        if vector_type in {"dense", "multivector"} and value is None:
            raise ValueError(f"Named vector of type '{vector_type}' requires 'dim'")
        return value


class VectorStoreConfig(BaseModel):
    collection_name: str
    multivector: bool
    named_vectors: Dict[str, NamedVectorConfig]


class EmbeddingConfig(BaseModel):
    model: str
    dimensions: int


class HybridConfig(BaseModel):
    mode: Literal["prefetch"]
    dense_weight: float
    sparse_weight: float


class RetrievalConfig(BaseModel):
    search_limit: int
    hybrid: HybridConfig


class RerankerConfig(BaseModel):
    model: str
    temperature: float


class PipelineConfig(BaseModel):
    """Configuration for a retrieval pipeline loaded from YAML."""

    name: str
    description: str
    vector_store: VectorStoreConfig
    embedding: EmbeddingConfig
    retrieval: RetrievalConfig
    reranker: RerankerConfig
    preprocessing: List[str]
    postprocessing: List[str]

    class Config:
        from_attributes = True
