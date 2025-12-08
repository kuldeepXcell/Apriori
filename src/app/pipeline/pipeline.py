"""Stub pipeline that will later host the full RAG logic."""

from __future__ import annotations

import time
from typing import List

from app.pipeline.models import PipelineConfig


class Pipeline:
    """A single pipeline instance backed by a config."""

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.execution_time: float = 0.0

    def search(self, query: str) -> List:
        """Placeholder search that returns no results for now."""
        start = time.perf_counter()
        # TODO: Plug in ingestion + retrieval + rerank based on self.config
        self.execution_time = time.perf_counter() - start
        return []


