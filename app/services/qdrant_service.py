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
        self.collection_name = settings.QDRANT_COLLECTION_NAME

    def create_collection(self):
        """
        Creates the Qdrant collection if it doesn't exist.
        Configures Dense Vector and Payload Index for keywords.
        """
        if not self.client.collection_exists(self.collection_name):
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=settings.EMBEDDING_DIM,
                    distance=models.Distance.COSINE
                )
            )
            
            # Create payload index for keywords to enable efficient text search
            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name="keywords",
                field_schema=models.TextIndexParams(
                    type="text",
                    tokenizer=models.TokenizerType.WORD,
                    min_token_len=2,
                    max_token_len=20,
                    lowercase=True
                )
            )
            print(f"Collection '{self.collection_name}' created.")
        else:
            print(f"Collection '{self.collection_name}' already exists.")

    def upsert_indicators(self, indicators: List[FinancialIndicator], embeddings: List[List[float]]):
        """
        Upserts indicators into Qdrant.
        """
        points = []
        for indicator, embedding in zip(indicators, embeddings):
            points.append(models.PointStruct(
                id=indicator.id,
                vector=embedding,
                payload=indicator.model_dump()
            ))
        
        self.client.upsert(
            collection_name=self.collection_name,
            points=points
        )
        print(f"Upserted {len(points)} points.")

    def search(self, query_vector: List[float], limit: int = 20) -> List[models.ScoredPoint]:
        """
        Performs a dense vector search. 
        TODO: Add hybrid search logic (combining with keyword filter).
        """
        return self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            limit=limit
        )

qdrant_service = QdrantService()
