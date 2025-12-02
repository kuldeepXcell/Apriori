from typing import List
from app.models.schemas import FinancialIndicator
from app.services.llm_service import llm_service
from app.services.qdrant_service import qdrant_service
import uuid

class IngestionService:
    def load_data(self) -> List[FinancialIndicator]:
        """
        Loads raw data and converts it to FinancialIndicator objects.
        TODO: Implement actual file reading logic (CSV/Excel/JSON).
        For now, returns dummy data.
        """
        # Placeholder data
        dummy_data = [
            {
                "name": "GDP Growth Rate",
                "definition": "The annual percentage growth rate of GDP at market prices based on constant local currency.",
                "keywords": ["gdp", "growth", "economy", "annual"]
            },
            {
                "name": "Inflation, consumer prices",
                "definition": "Inflation as measured by the consumer price index reflects the annual percentage change in the cost to the average consumer of acquiring a basket of goods and services.",
                "keywords": ["inflation", "cpi", "prices", "consumer"]
            }
        ]
        
        indicators = []
        for item in dummy_data:
            indicators.append(FinancialIndicator(
                id=str(uuid.uuid4()),
                name=item["name"],
                definition=item["definition"],
                keywords=item["keywords"],
                metadata={"source": "dummy"}
            ))
        return indicators

    def run_ingestion(self):
        """
        Orchestrates the ingestion process: Load -> Embed -> Upsert.
        """
        print("Starting ingestion...")
        
        # 1. Load Data
        indicators = self.load_data()
        print(f"Loaded {len(indicators)} indicators.")
        
        # 2. Generate Embeddings
        embeddings = []
        for ind in indicators:
            # Create a rich text representation for embedding
            text_to_embed = f"{ind.name}: {ind.definition}"
            emb = llm_service.get_embedding(text_to_embed)
            embeddings.append(emb)
        print("Embeddings generated.")
        
        # 3. Ensure Collection Exists
        qdrant_service.create_collection()
        
        # 4. Upsert to Qdrant
        qdrant_service.upsert_indicators(indicators, embeddings)
        print("Ingestion complete.")

ingestion_service = IngestionService()
