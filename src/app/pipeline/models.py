"""Typed representations for pipeline configs and results."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class PipelineConfig:
    name: str
    description: str = ""
    collection_name: Optional[str] = None
    embedding_model: Optional[str] = None
    llm_model: Optional[str] = None
    retrieval_strategy: Optional[str] = None
    multivector: bool = False
    raw: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PipelineResult:
    results: List[Any]
    execution_time: float
    error: Optional[str] = None


@dataclass
class MultivectorWeights:
    """User-provided weights for query fields."""

    application: float = 0.33
    context: float = 0.33
    question: float = 0.33

    def normalized(self) -> "MultivectorWeights":
        total = max(self.application + self.context + self.question, 1e-6)
        return MultivectorWeights(
            application=self.application / total,
            context=self.context / total,
            question=self.question / total,
        )


@dataclass
class Indicator:
    name: str
    normalized_name: Optional[str] = None
    subsection: Optional[str] = None
    subsubsection: Optional[str] = None
    definition: Optional[str] = None
    question: Optional[str] = None
    application_context: Optional[str] = None
    keywords: Optional[List[str]] = None
    source: Optional[str] = None
    sheet_name: Optional[str] = None


@dataclass
class SearchHit:
    indicator: Indicator
    score: float
    relevance_reason: Optional[str] = None


@dataclass
class SearchResponse:
    """Structured response for one approach (multivector or HyDe)."""

    hits: List[SearchHit] = field(default_factory=list)
    reranked: List[SearchHit] = field(default_factory=list)
    generated_definition: Optional[str] = None
    error: Optional[str] = None


@dataclass
class PipelineSearchBundle:
    """Aggregated results for both multivector and HyDe flows."""

    multivector: SearchResponse
    hyde: SearchResponse


