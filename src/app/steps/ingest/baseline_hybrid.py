"""Ingestion for baseline_hybrid pipeline (dense + sparse named vectors)."""

from __future__ import annotations

import json

import time
from pathlib import Path
import uuid
from typing import Dict, Iterable, List, Sequence

from app.adapters import openai_embedding_adapter, qdrant_adapter, sparse_embedding_adapter

BATCH_SIZE = 10


def _load_indicators(data_path: Path) -> List[Dict]:
    """Load and flatten indicators from the country_indicators.json structure."""
    raw = json.loads(data_path.read_text())
    sheets = raw.get("sheets", [])
    indicators: List[Dict] = []
    for sheet in sheets:
        for ind in sheet.get("indicators", []):
            indicators.append(
                {
                    "sheet_name": sheet.get("sheet_name"),
                    "subsection": ind.get("subsection"),
                    "subsubsection": ind.get("subsubsection"),
                    "indicator_name": ind.get("indicator_name"),
                    "normalized_indicator_name": ind.get("normalized_indicator_name"),
                    "definition": ind.get("definition") or "",
                    "question": ind.get("question") or "",
                    "application_context": ind.get("application_context") or "",
                    "keywords": ind.get("keywords") or [],
                }
            )
    return indicators


def _batch(iterable: Sequence, size: int) -> Iterable[Sequence]:
    for i in range(0, len(iterable), size):
        yield iterable[i : i + size]


def ingest_baseline_hybrid(
    collection_name: str,
    vector_store_cfg: Dict,
    embedding_model: str,
    data_path: Path,
) -> int:
    """
    Ingest all indicators into Qdrant using named vectors.

    Returns number of upserted points.
    """
    indicators = _load_indicators(data_path)
    if not indicators:
        return 0

    total_indicators = len(indicators)
    print(
        f"Starting ingestion into collection '{collection_name}' with {total_indicators} indicators (batch size={BATCH_SIZE})"
    )

    qdrant_adapter.ensure_named_collection(collection_name, vector_store_cfg, force_recreate=False)

    total = 0
    for batch_idx, batch in enumerate(_batch(indicators, BATCH_SIZE), start=1):
        batch_start = time.perf_counter()
        points = []

        # Dense embeddings: embed each field separately to match named vectors.
        definitions = [item["definition"] for item in batch]
        questions = [item["question"] for item in batch]
        applications = [item["application_context"] for item in batch]

        definition_vecs = openai_embedding_adapter.embed(definitions, model=embedding_model)
        question_vecs = openai_embedding_adapter.embed(questions, model=embedding_model)
        application_vecs = openai_embedding_adapter.embed(applications, model=embedding_model)

        # Sparse embeddings from keywords.
        keyword_texts = [" ".join(item["keywords"]) for item in batch]
        sparse_vecs = list(sparse_embedding_adapter.encode(keyword_texts))

        for idx, item in enumerate(batch):
            dense_vectors = {
                "definition": definition_vecs[idx],
                "question": question_vecs[idx],
                "application": application_vecs[idx],
            }
            sparse = sparse_vecs[idx]
            sparse_vec = qdrant_adapter.make_sparse_vector(sparse.indices, sparse.values)

            payload = {
                "indicator_name": item["indicator_name"],
                "normalized_indicator_name": item["normalized_indicator_name"],
                "subsection": item["subsection"],
                "subsubsection": item["subsubsection"],
                "definition": item["definition"],
                "question": item["question"],
                "application_context": item["application_context"],
                "keywords": item["keywords"],
                "source": "country_indicators.json",
                "sheet_name": item["sheet_name"],
            }

            raw_id = item["normalized_indicator_name"] or item["indicator_name"] or f"idx-{total+idx}"
            # Qdrant Cloud requires point IDs to be UUID or unsigned int; use stable UUID5.
            point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, raw_id))
            points.append(
                qdrant_adapter.build_point(
                    point_id=point_id,
                    dense_vectors=dense_vectors,
                    sparse_vector=sparse_vec,
                    payload=payload,
                )
            )

        print(
                f"Batch {batch_idx}: dense shapes def={len(definition_vecs[0]) if definition_vecs else 0} q={len(question_vecs[0]) if question_vecs else 0} app={len(application_vecs[0]) if application_vecs else 0} | sparse count={len(sparse_vecs)}"
            )
        print(
            f"Batch {batch_idx}: point IDs={[p.id for p in points]}"
        )

        try:
            qdrant_adapter.upsert_points(collection_name, points)
        except Exception as exc:  # pragma: no cover - operational logging
            print(
                f"Upsert failed for collection '{collection_name}', batch {batch_idx} (size={len(points)}): {exc}"
            )
            raise

        total += len(points)
        batch_ms = (time.perf_counter() - batch_start) * 1000
        print(
            f"Batch {batch_idx} upserted {len(points)} points (cumulative={total}) in {batch_ms:.1f} ms"
        )

    print(
        f"Ingestion complete for collection '{collection_name}': {total} points upserted from {total_indicators} indicators."
    )
    return total

