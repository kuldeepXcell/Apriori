from collections.abc import Iterable
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.http import models as qm

from app.config.settings import settings

# Reuse one client; can be swapped for grpc via settings if needed later.
client = QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key)


def upsert(collection: str, points: Iterable[qm.PointStruct], wait: bool = True) -> Any:
    """Insert or update points in a collection."""
    return client.upsert(collection_name=collection, points=list(points), wait=wait)


def search_dense(
    collection: str,
    vector_name: str,
    vector: list[float],
    limit: int = 10,
    with_payload: bool = True,
    with_vectors: bool = False,
    query_filter: qm.Filter | None = None,
):
    """Search against a named dense vector."""
    return client.search(
        collection_name=collection,
        query_vector=(vector_name, vector),
        limit=limit,
        with_payload=with_payload,
        with_vectors=with_vectors,
        query_filter=query_filter,
    )


def search_sparse(
    collection: str,
    sparse: qm.SparseVector,
    limit: int = 10,
    with_payload: bool = True,
    with_vectors: bool = False,
    query_filter: qm.Filter | None = None,
):
    """Search using a sparse vector representation."""
    return client.search(
        collection_name=collection,
        query_vector=sparse,
        limit=limit,
        with_payload=with_payload,
        with_vectors=with_vectors,
        query_filter=query_filter,
    )


def search_hybrid(
    collection: str,
    dense: tuple[str, list[float]],
    sparse: qm.SparseVector,
    alpha: float,
    limit: int = 10,
    with_payload: bool = True,
    with_vectors: bool = False,
    query_filter: qm.Filter | None = None,
):
    """
    Hybrid search using named dense vector + sparse vector.
    Alpha controls weight toward dense (1.0) vs sparse (0.0).
    """
    return client.search(
        collection_name=collection,
        query_vector=[qm.NamedVector(name=dense[0], vector=dense[1]), sparse],
        limit=limit,
        with_payload=with_payload,
        with_vectors=with_vectors,
        query_filter=query_filter,
        search_params=qm.SearchParams(
            hnsw_ef=128,
            exact=False,
            quantization=False,
            fusion_params=qm.FusionParams(fusion=qm.FusionStrategy.RELATIVE_SCORE, alpha=alpha),
        ),
    )

