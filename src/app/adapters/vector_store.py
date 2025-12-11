from collections.abc import Iterable
from typing import Any, Optional

from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import FastEmbedSparse, QdrantVectorStore, RetrievalMode
from qdrant_client import QdrantClient
from qdrant_client.http import models as qm

from app.config.settings import settings

# Shared clients/embedders reused across ingest/retrieval.
client = QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key)
embeddings = OpenAIEmbeddings(
    model=settings.embedding_model,
    api_key=settings.openai_api_key,
    dimensions=settings.embedding_dimensions,
)
sparse_embeddings = FastEmbedSparse(model_name=settings.sparse_model_name)


def ensure_collection(recreate: bool = False) -> None:
    """Create the multi-vector + sparse collection if missing."""
    exists = client.collection_exists(settings.qdrant_collection)
    if exists and recreate:
        client.delete_collection(collection_name=settings.qdrant_collection)
        exists = False
    if not exists:
        client.create_collection(
            collection_name=settings.qdrant_collection,
            vectors_config={
                settings.definition_vector_name: qm.VectorParams(
                    size=settings.embedding_dimensions, distance=qm.Distance.COSINE
                ),
                settings.question_vector_name: qm.VectorParams(
                    size=settings.embedding_dimensions, distance=qm.Distance.COSINE
                ),
                settings.context_vector_name: qm.VectorParams(
                    size=settings.embedding_dimensions, distance=qm.Distance.COSINE
                ),
            },
            sparse_vectors_config={
                settings.sparse_vector_name: qm.SparseVectorParams(
                    # Enable IDF modifier so BM25-style scoring can be applied server-side.
                    modifier=qm.Modifier.IDF,
                    index=qm.SparseIndexParams(on_disk=False),
                )
            },
        )


def upsert_points(points: Iterable[qm.PointStruct], wait: bool = True) -> Any:
    """Insert or update points in the configured collection."""
    return client.upsert(collection_name=settings.qdrant_collection, points=list(points), wait=wait)


def get_vector_store(
    *,
    vector_name: Optional[str] = None,
    retrieval_mode: RetrievalMode = RetrievalMode.HYBRID,
) -> QdrantVectorStore:
    """
    Build a LangChain QdrantVectorStore bound to the configured collection.

    vector_name lets us target a specific dense field (definition/question/context).
    """
    return QdrantVectorStore(
        client=client,
        collection_name=settings.qdrant_collection,
        embedding=embeddings,
        sparse_embedding=sparse_embeddings,
        vector_name=vector_name,
        sparse_vector_name=settings.sparse_vector_name,
        retrieval_mode=retrieval_mode,
        content_payload_key="normalized_indicator_name",
    )


def hybrid_search(
    query: str,
    *,
    k: int = 10,
    vector_name: Optional[str] = None,
    retrieval_mode: RetrievalMode = RetrievalMode.HYBRID,
) -> list[tuple[Any, float]]:
    """
    Perform HYBRID search (dense + sparse) using LangChain vectorstore.

    Returns list of (Document, score) tuples.
    """
    store = get_vector_store(vector_name=vector_name, retrieval_mode=retrieval_mode)
    return store.similarity_search_with_score(query, k=k)

