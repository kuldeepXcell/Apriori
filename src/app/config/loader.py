"""Load and normalize pipeline YAML configs into typed objects."""

from pathlib import Path
from typing import List

import yaml

from app.pipeline.models import PipelineConfig


def load_pipeline_configs(pipelines_dir: Path) -> List[PipelineConfig]:
    """Load all YAML configs from the pipelines directory."""
    configs: List[PipelineConfig] = []
    for path in sorted(pipelines_dir.glob("*.yaml")):
        with path.open("r", encoding="utf-8") as file:
            raw = yaml.safe_load(file) or {}
        configs.append(_build_config(raw, path.name))
    return configs


def _build_config(raw: dict, filename: str) -> PipelineConfig:
    """Convert raw YAML dict into a PipelineConfig with sensible defaults."""
    name = raw.get("name") or filename.replace(".yaml", "")
    description = (raw.get("description") or "").strip()
    vector_store = raw.get("vector_store", {}) or {}
    retrieval = raw.get("retrieval", {}) or {}
    embedding = raw.get("embedding", {}) or {}
    reranker = raw.get("reranker", {}) or {}

    return PipelineConfig(
        name=name,
        description=description,
        collection_name=vector_store.get("collection_name"),
        embedding_model=embedding.get("model"),
        llm_model=reranker.get("model"),
        retrieval_strategy="hybrid" if "hybrid" in retrieval else retrieval.get("strategy"),
        multivector=bool(vector_store.get("multivector", False)),
        raw=raw,
    )


