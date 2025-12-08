"""CLI entrypoint to ingest pipelines into Qdrant."""

from __future__ import annotations

import argparse
import sys

from app.config import paths
from app.config.loader import load_pipeline_configs
from app.steps.ingest.baseline_hybrid import ingest_baseline_hybrid


def run(pipeline_name: str) -> int:
    configs = load_pipeline_configs(paths.PIPELINES_DIR)
    cfg = next((c for c in configs if c.name == pipeline_name), None)
    if not cfg:
        raise SystemExit(f"Pipeline '{pipeline_name}' not found in {paths.PIPELINES_DIR}")

    raw = cfg.raw
    vector_store_cfg = raw.get("vector_store", {})
    embedding_model = (raw.get("embedding") or {}).get("model")
    collection_name = vector_store_cfg.get("collection_name") or cfg.collection_name

    if not collection_name:
        raise SystemExit("Collection name is required in pipeline config.")
    if not embedding_model:
        raise SystemExit("Embedding model is required in pipeline config.")

    data_path = paths.DATA_DIR / "country_indicators.json"
    if not data_path.exists():
        raise SystemExit(f"Data file not found: {data_path}")

    total = ingest_baseline_hybrid(
        collection_name=collection_name,
        vector_store_cfg=vector_store_cfg,
        embedding_model=embedding_model,
        data_path=data_path,
    )
    print(f"Ingested {total} points into '{collection_name}'.")
    return total


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest pipelines into Qdrant.")
    parser.add_argument("--pipeline", default="baseline_hybrid", help="Pipeline name to ingest.")
    args = parser.parse_args()
    run(args.pipeline)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # pragma: no cover - CLI convenience
        print(f"Ingestion failed: {exc}", file=sys.stderr)
        raise

