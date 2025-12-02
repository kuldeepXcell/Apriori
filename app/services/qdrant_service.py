from qdrant_client import QdrantClient
from qdrant_client.http import models
from app.core.config import settings
from app.models.schemas import FinancialIndicator
from typing import List, Dict, Any

class QdrantService:
    def __init__(self):
        self.client = QdrantClient(
            url=settings.QDRANT_URL,
            api_key=settings.QDRANT_API_KEY
        )

    def create_collection(self, collection_name: str, vector_size: int):
        """
        Creates a Qdrant collection if it doesn't exist.
        Configures Dense Vector and Payload Index for keywords.
        
        Args:
            collection_name: Name of the collection
            vector_size: Dimension of the embedding vectors
        """
        if not self.client.collection_exists(collection_name):
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=models.VectorParams(
                    size=vector_size,
                    distance=models.Distance.COSINE
                )
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

    def upsert_indicators(self, collection_name: str, indicators: List[FinancialIndicator], embeddings: List[List[float]]):
        """
        Upserts indicators into Qdrant.
        
        Args:
            collection_name: Target collection
            indicators: List of financial indicators
            embeddings: Corresponding embedding vectors
        """
        points = []
        for indicator, embedding in zip(indicators, embeddings):
            points.append(models.PointStruct(
                id=indicator.id,
                vector=embedding,
                payload=indicator.model_dump()
            ))
        
        self.client.upsert(
            collection_name=collection_name,
            points=points
        )
        print(f"Upserted {len(points)} points to '{collection_name}'.")

    def search(self, collection_name: str, query_vector: List[float], limit: int = 20) -> List[models.ScoredPoint]:
        """
        Performs a dense vector search.
        
        Args:
            collection_name: Collection to search
            query_vector: Query embedding vector
            limit: Number of results to return
        """
        return self.client.search(
            collection_name=collection_name,
            query_vector=query_vector,
            limit=limit
        )

qdrant_service = QdrantService()
