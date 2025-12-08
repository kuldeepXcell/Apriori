"""Qdrant helper for named dense + sparse vectors."""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional

from qdrant_client import QdrantClient
from qdrant_client.http import models

from app.config.settings import settings


class QdrantAdapter:
    """Encapsulates Qdrant client setup and common collection ops."""

    def __init__(self) -> None:
        if not settings.QDRANT_URL:
            raise ValueError("QDRANT_URL is not set.")
        self.client = QdrantClient(
            url=settings.QDRANT_URL,
            api_key=settings.QDRANT_API_KEY,
            timeout=90.0,
        )

    def ensure_named_collection(
        self,
        collection_name: str,
        vector_store_cfg: Dict,
        force_recreate: bool = False,
    ) -> None:
        """
        Create (or recreate) a collection with named dense + sparse vectors.

        vector_store_cfg expects a structure like:
        {
          "multivector": True,
          "named_vectors": {
            "definition": {"type": "dense", "dim": 1536},
            "question": {"type": "dense", "dim": 1536},
            "application": {"type": "dense", "dim": 1536},
            "keywords": {"type": "sparse"}
          }
        }
        """
        dense_vectors, sparse_vectors = self._build_vectors_config(vector_store_cfg)
        if not dense_vectors and not sparse_vectors:
            raise ValueError("vector_store_cfg must define named_vectors for multivector collections.")

        exists = self.client.collection_exists(collection_name)
        if exists and not force_recreate:
            return

        create_fn = self.client.recreate_collection if exists and force_recreate else self.client.create_collection
        create_fn(
            collection_name=collection_name,
            vectors_config=dense_vectors if dense_vectors else None,
            sparse_vectors_config=sparse_vectors if sparse_vectors else None,
        )

    def upsert_points(
        self,
        collection_name: str,
        points: Iterable[models.PointStruct],
        wait: bool = True,
    ) -> None:
        """Upsert a batch of points into Qdrant."""
        self.client.upsert(collection_name=collection_name, points=list(points), wait=wait)

    @staticmethod
    def build_point(
        point_id: str | int,
        dense_vectors: Dict[str, List[float]],
        sparse_vector: Optional[models.SparseVector],
        payload: Dict,
    ) -> models.PointStruct:
        """Construct a PointStruct with named dense + optional sparse vectors."""
        vector_payload: Dict[str, object] = {**dense_vectors}
        if sparse_vector is not None:
            vector_payload["keywords"] = sparse_vector
        return models.PointStruct(id=point_id, vector=vector_payload, payload=payload)

    @staticmethod
    def make_sparse_vector(indices: List[int], values: List[float]) -> models.SparseVector:
        """Create a SparseVector from indices/values lists."""
        return models.SparseVector(indices=indices, values=values)

    @staticmethod
    def _build_vectors_config(vector_store_cfg: Dict) -> tuple[Dict[str, object], Dict[str, models.SparseVectorParams]]:
        named_vectors = vector_store_cfg.get("named_vectors") or {}
        dense_config: Dict[str, object] = {}
        sparse_config: Dict[str, models.SparseVectorParams] = {}
        for name, cfg in named_vectors.items():
            vector_type = (cfg or {}).get("type", "dense")
            if vector_type == "sparse":
                sparse_config[name] = models.SparseVectorParams()
            else:
                dim = cfg.get("dim")
                if dim is None:
                    raise ValueError(f"Missing dim for dense vector '{name}'.")
                dense_config[name] = models.VectorParams(size=int(dim), distance=models.Distance.COSINE)
        return dense_config, sparse_config


qdrant_adapter = QdrantAdapter()

