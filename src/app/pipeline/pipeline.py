"""Stub pipeline that will later host the full RAG logic."""

from __future__ import annotations

import time
from typing import Optional

from app.pipeline.models import (
    MultivectorWeights,
    PipelineConfig,
    PipelineSearchBundle,
)
from app.steps.retrieval.hybrid import HybridRetriever


class Pipeline:
    """A single pipeline instance backed by a config."""

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.execution_time: float = 0.0
        self.retriever: HybridRetriever | None = None

    def search(
        self, query: str, weights: Optional[MultivectorWeights] = None
    ) -> PipelineSearchBundle:
        """Run both multivector and HyDe searches."""
        start = time.perf_counter()
        mv_weights = weights or MultivectorWeights()
        if self.retriever is None:
            self.retriever = HybridRetriever(
                collection_name=self.config.collection_name or "baseline_hybrid",
                embedding_model=self.config.embedding_model,
                reranker_model=self.config.llm_model,
            )
        multivector = self.retriever.multivector_search(query, mv_weights)
        hyde = self.retriever.hyde_search(query)
        self.execution_time = time.perf_counter() - start
        return PipelineSearchBundle(multivector=multivector, hyde=hyde)


