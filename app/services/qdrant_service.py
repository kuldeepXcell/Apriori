from qdrant_client import QdrantClient
from qdrant_client.http import models
from app.core.config import settings
from app.models.schemas import FinancialIndicator
from typing import List, Dict, Sequence, Optional

class QdrantService:
    def __init__(self):
        self.client = QdrantClient(
            url=settings.QDRANT_URL,
            api_key=settings.QDRANT_API_KEY,
            timeout=90
        )
        self.default_vector_name = "definition_vector"

    def create_collection(
        self,
        collection_name: str,
        vector_size: int,
        vector_names: Optional[Sequence[str]] = None,
        sparse_vector_name: Optional[str] = None
    ):
        """
        Creates a Qdrant collection if it doesn't exist.
        Configures named dense vectors plus an optional sparse vector slot.
        
        Args:
            collection_name: Name of the collection
            vector_size: Dimension of the embedding vectors
            vector_names: Names of dense vectors to configure on the collection
            sparse_vector_name: Name for the sparse vector slot (keywords BM25)
        """
        dense_vector_names = list(vector_names) if vector_names else [self.default_vector_name]
        vectors_config = {
            name: models.VectorParams(
                size=vector_size,
                distance=models.Distance.COSINE
            )
            for name in dense_vector_names
        }
        sparse_config = None
        if sparse_vector_name:
            sparse_config = {
                sparse_vector_name: models.SparseVectorParams(
                    modifier=models.Modifier.IDF
                )
            }

        if not self.client.collection_exists(collection_name):
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=vectors_config,
                sparse_vectors_config=sparse_config
            )
            
            # Create payload index for keywords to enable efficient text search
            self.client.create_payload_index(
                collection_name=collection_name,
                field_name="keywords",
                field_schema=models.TextIndexParams(
                    type="text",
                    tokenizer=models.TokenizerType.WORD,
                    min_token_len=2,
                    max_token_len=20,
                    lowercase=True
                )
            )
            print(f"Collection '{collection_name}' created with vector size {vector_size}.")
        else:
            print(f"Collection '{collection_name}' already exists.")

    def upsert_indicators(
        self,
        collection_name: str,
        indicators: List[FinancialIndicator],
        dense_vectors: List[Dict[str, List[float]]],
        sparse_vectors: Optional[List[models.SparseVector]] = None,
        sparse_vector_name: Optional[str] = None
    ):
        """
        Upserts indicators into Qdrant with named dense vectors and optional sparse vectors.
        
        Args:
            collection_name: Target collection
            indicators: List of financial indicators
            dense_vectors: Named dense vectors per indicator
            sparse_vectors: Optional sparse vectors aligned with indicators
            sparse_vector_name: Name of the sparse vector slot
        """
        if len(indicators) != len(dense_vectors):
            raise ValueError("Dense vectors must align with indicator count.")

        if sparse_vectors is not None and len(sparse_vectors) != len(indicators):
            raise ValueError("Sparse vectors must align with indicator count.")

        if sparse_vectors is not None and not sparse_vector_name:
            raise ValueError("Sparse vector name is required when sparse vectors are provided.")

        points = []
        for idx, indicator in enumerate(indicators):
            named_vectors: Dict[str, List[float] | models.SparseVector] = {
                name: vector
                for name, vector in dense_vectors[idx].items()
            }
            if sparse_vectors is not None and sparse_vector_name:
                named_vectors[sparse_vector_name] = sparse_vectors[idx]

            points.append(models.PointStruct(
                id=indicator.id,
                vector=named_vectors,
                payload=indicator.model_dump()
            ))
        
        self.client.upsert(
            collection_name=collection_name,
            points=points
        )
        print(f"Upserted {len(points)} points to '{collection_name}'.")

    def search(
        self,
        collection_name: str,
        query_vector: List[float],
        limit: int = 20,
        vector_name: Optional[str] = None
    ) -> List[models.ScoredPoint]:
        """
        Performs a dense vector search.
        
        Args:
            collection_name: Collection to search
            query_vector: Query embedding vector
            limit: Number of results to return
            vector_name: Named vector slot to use for search
        """
        search_vector_name = vector_name or self.default_vector_name
        named_vector = models.NamedVector(
            name=search_vector_name,
            vector=query_vector
        )
        return self.client.search(
            collection_name=collection_name,
            query_vector=named_vector,
            limit=limit
        )

qdrant_service = QdrantService()
