import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
import uuid

from fastembed import SparseTextEmbedding
from qdrant_client.http import models

from app.configs.pipeline_config import PipelineConfig
from app.models.schemas import FinancialIndicator
from app.services.llm_service import llm_service
from app.services.qdrant_service import qdrant_service


class IngestionService:
    """
    Ingests financial indicators into Qdrant with multi-vector (question/definition/application) support
    plus a sparse BM25 vector derived from indicator keywords.
    """

    DATASET_PATH = Path(__file__).parent.parent.parent / "country_indicators.json"
    VECTOR_NAMES: Sequence[str] = ("question_vector", "definition_vector", "application_vector")
    SPARSE_VECTOR_NAME = "keywords_sparse_vector"
    UPSERT_BATCH_SIZE = 10

    def __init__(self) -> None:
        self._bm25_encoder = SparseTextEmbedding(model_name="Qdrant/bm25")

    def load_data(self, index_range: Optional[Tuple[int, int]] = None) -> List[FinancialIndicator]:
        """
        Loads raw indicator data from the canonical JSON file.

        Args:
            index_range: Optional tuple of (start, end) indices (0-based, end-exclusive)
                used to limit how many indicators are ingested.
        """
        if not self.DATASET_PATH.exists():
            raise FileNotFoundError(f"Indicator file not found at {self.DATASET_PATH}")

        with self.DATASET_PATH.open("r", encoding="utf-8") as file:
            dataset = json.load(file)

        raw_indicators = dataset.get("indicators", [])
        total = len(raw_indicators)

        start_idx = 0
        end_idx = total
        if index_range:
            start_idx = max(0, index_range[0])
            end_idx = min(total, index_range[1])

        selected = raw_indicators[start_idx:end_idx]
        indicators: List[FinancialIndicator] = []
        for offset, item in enumerate(selected, start=start_idx + 1):
            keywords = item.get("keywords") or []
            normalized_name = item.get("normalized_indicator_name") or item.get("indicator_name", "").strip().lower()
            indicator_id = str(uuid.uuid5(uuid.NAMESPACE_URL, normalized_name))

            indicators.append(
                FinancialIndicator(
                    id=indicator_id,
                    normalized_name=normalized_name,
                    name=item["indicator_name"],
                    definition=item["definition"],
                    question=item["question"],
                    application_context=item["application_context"],
                    subsection=item.get("subsection"),
                    subsubsection=item.get("subsubsection"),
                    keywords=keywords,
                    metadata={
                        "source_file": str(self.DATASET_PATH),
                        "source_sheet": dataset.get("source_sheet"),
                        "sequence_number": offset,
                    },
                )
            )

        human_readable_range = f"{start_idx + 1}-{start_idx + len(selected)}" if selected else "none"
        print(
            f"Loaded {len(indicators)} of {total} indicators "
            f"(slice {human_readable_range})."
        )
        return indicators

    def _generate_dense_vectors(
        self,
        indicators: List[FinancialIndicator],
        embedding_model: str
    ) -> List[Dict[str, List[float]]]:
        dense_vectors: List[Dict[str, List[float]]] = []
        for indicator in indicators:
            dense_vectors.append(
                {
                    "question_vector": llm_service.get_embedding(indicator.question, model=embedding_model),
                    "definition_vector": llm_service.get_embedding(indicator.definition, model=embedding_model),
                    "application_vector": llm_service.get_embedding(
                        indicator.application_context,
                        model=embedding_model,
                    ),
                }
            )
        return dense_vectors

    def _generate_sparse_vectors(
        self,
        indicators: List[FinancialIndicator]
    ) -> List[models.SparseVector]:
        keyword_texts = [
            " ".join(ind.keywords) if ind.keywords else ind.name
            for ind in indicators
        ]
        sparse_embeddings = list(self._bm25_encoder.embed(keyword_texts))
        sparse_vectors: List[models.SparseVector] = []
        for embedding in sparse_embeddings:
            sparse_vectors.append(
                models.SparseVector(
                    indices=embedding.indices.tolist(),
                    values=embedding.values.tolist(),
                )
            )
        return sparse_vectors

    def run_ingestion(
        self,
        pipeline_configs: List[PipelineConfig],
        index_range: Optional[Tuple[int, int]] = None
    ):
        """
        Orchestrates the ingestion process for multiple pipeline configurations.
        Each pipeline with a unique embedding model gets its own collection.
        
        Args:
            pipeline_configs: List of pipeline configurations to ingest for
            index_range: Optional tuple specifying the slice of indicators to ingest
        """
        print("Starting multi-pipeline ingestion...")
        
        # 1. Load Data once (with optional slicing for storage analysis)
        indicators = self.load_data(index_range=index_range)
        if not indicators:
            print("No indicators selected for ingestion.")
            return
        sparse_vectors = self._generate_sparse_vectors(indicators)
        print("Generated sparse BM25 vectors for keywords.")
        
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
            dense_vectors = self._generate_dense_vectors(indicators, embedding_model)
            print(
                f"Generated {len(dense_vectors)} multi-vector embeddings "
                f"({len(self.VECTOR_NAMES)} vectors/indicator) with {embedding_model}."
            )
            
            # Upsert to each collection that uses this embedding model
            for config in configs:
                print(f"  Upserting to collection: {config.collection_name}")
                qdrant_service.create_collection(
                    collection_name=config.collection_name,
                    vector_size=embedding_dim,
                    vector_names=self.VECTOR_NAMES,
                    sparse_vector_name=self.SPARSE_VECTOR_NAME
                )
                for start_idx in range(0, len(indicators), self.UPSERT_BATCH_SIZE):
                    end_idx = min(start_idx + self.UPSERT_BATCH_SIZE, len(indicators))
                    print(
                        f"    - Batch {start_idx + 1}-{end_idx} "
                        f"({end_idx - start_idx} indicators)"
                    )
                    qdrant_service.upsert_indicators(
                        collection_name=config.collection_name,
                        indicators=indicators[start_idx:end_idx],
                        dense_vectors=dense_vectors[start_idx:end_idx],
                        sparse_vectors=sparse_vectors[start_idx:end_idx],
                        sparse_vector_name=self.SPARSE_VECTOR_NAME
                    )
        
        print("\nIngestion complete for all pipelines.")

ingestion_service = IngestionService()
