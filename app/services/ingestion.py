from typing import List
from app.models.schemas import FinancialIndicator
from app.services.llm_service import llm_service
from app.services.qdrant_service import qdrant_service
from app.configs.pipeline_config import PipelineConfig
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

    def run_ingestion(self, pipeline_configs: List[PipelineConfig]):
        """
        Orchestrates the ingestion process for multiple pipeline configurations.
        Each pipeline with a unique embedding model gets its own collection.
        
        Args:
            pipeline_configs: List of pipeline configurations to ingest for
        """
        print("Starting multi-pipeline ingestion...")
        
        # 1. Load Data (once)
        indicators = self.load_data()
        print(f"Loaded {len(indicators)} indicators.")
        
        # 2. Group configs by embedding model (to avoid duplicate work)
        embedding_groups = {}
        for config in pipeline_configs:
            key = (config.embedding_model, config.embedding_dim)
            if key not in embedding_groups:
                embedding_groups[key] = []
            embedding_groups[key].append(config)
        
        # 3. For each unique embedding model, generate embeddings and ingest
        for (embedding_model, embedding_dim), configs in embedding_groups.items():
            print(f"\nProcessing embedding model: {embedding_model}")
            
            # Generate Embeddings
            embeddings = []
            for ind in indicators:
                text_to_embed = f"{ind.name}: {ind.definition}"
                emb = llm_service.get_embedding(text_to_embed, model=embedding_model)
                embeddings.append(emb)
            print(f"Generated {len(embeddings)} embeddings with {embedding_model}.")
            
            # Upsert to each collection that uses this embedding model
            for config in configs:
                print(f"  Upserting to collection: {config.collection_name}")
                qdrant_service.create_collection(config.collection_name, embedding_dim)
                qdrant_service.upsert_indicators(config.collection_name, indicators, embeddings)
        
        print("\nIngestion complete for all pipelines.")

ingestion_service = IngestionService()
