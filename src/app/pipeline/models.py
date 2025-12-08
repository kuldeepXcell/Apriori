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


