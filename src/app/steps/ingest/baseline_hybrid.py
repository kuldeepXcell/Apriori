"""Ingest indicators into Qdrant for the multivector_weighted pipeline."""

from __future__ import annotations

import uuid
import json
from pathlib import Path
import sys
from typing import Any, Iterable, List, Tuple
from qdrant_client.http import models as qm

ROOT_DIR = Path(__file__).resolve().parents[4]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from app.adapters import embeddings, vector_store
from app.config.settings import settings
from app.core.logging import ModuleName, get_logger, setup_logging

# ---- Defaults (adjust in-file) ------------------------------------------------
INPUT_PATH = ROOT_DIR / "data" / "country_indicators.json"
COLLECTION = settings.qdrant_collection
DEF_VECTOR_NAME = settings.definition_vector_name
QUESTION_VECTOR_NAME = settings.question_vector_name
CONTEXT_VECTOR_NAME = settings.context_vector_name
SPARSE_VECTOR_NAME = settings.sparse_vector_name
EMBED_MODEL = settings.embedding_model
EMBED_DIM = settings.embedding_dimensions
BATCH_SIZE = 10
DRY_RUN = False  # set True to skip upsert
RECREATE = False
LIMIT: int | None = 1  # temporary: process only 1 indicator

logger = get_logger(__name__)


# ---- Helpers -----------------------------------------------------------------
def stable_id(normalized_indicator_name: str) -> uuid.UUID:
    """Derive a stable, Qdrant-valid UUID from normalized name"""
    return uuid.uuid5(uuid.NAMESPACE_URL, normalized_indicator_name)


def resolve_point_id(rec: dict[str, Any]) -> uuid.UUID:
    """Always derive a stable UUID from the normalized name (ignore provided ids)."""
    return stable_id(rec["normalized_indicator_name"])


def load_indicators(path: Path) -> list[dict[str, Any]]:
    """Load indicators and attach sheet_name to each entry."""
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict) or "sheets" not in data:
        raise ValueError("Expected object with 'sheets' array in JSON file.")
    records: list[dict[str, Any]] = []
    for sheet in data.get("sheets", []):
        sheet_name = sheet.get("sheet_name")
        for rec in sheet.get("indicators", []):
            rec = dict(rec)
            rec["sheet_name"] = sheet_name
            records.append(rec)
    return records


def validate_record(rec: dict[str, Any]) -> None:
    required = [
        "indicator_name",
        "normalized_indicator_name",
        "definition",
        "question",
        "application_context",
        "keywords",
    ]
    missing = [k for k in required if k not in rec or rec[k] in (None, "", [])]
    if missing:
        raise ValueError(f"Missing required fields {missing} for record {rec}")
    if not isinstance(rec["keywords"], list):
        raise ValueError("keywords must be a list")


def to_payload(rec: dict[str, Any]) -> dict[str, Any]:
    payload = {
        "indicator_name": rec.get("indicator_name"),
        "normalized_indicator_name": rec.get("normalized_indicator_name"),
        "definition": rec.get("definition"),
        "question": rec.get("question"),
        "application_context": rec.get("application_context"),
        "keywords": rec.get("keywords", []),
        "sheet_name": rec.get("sheet_name"),
        "subsection": rec.get("subsection"),
        "subsubsection": rec.get("subsubsection"),
    }
    return payload


def ensure_collection(recreate: bool = False) -> None:
    logger.info(
        "Ensuring collection %s exists (recreate=%s)", COLLECTION, recreate, extra={"module_name": ModuleName.INGESTION}
    )
    vector_store.ensure_collection(recreate=recreate)


def embed_batch(texts: Iterable[str]) -> list[list[float]]:
    return embeddings.embed_texts(texts, model=EMBED_MODEL, dimensions=EMBED_DIM)


def sparse_from_keywords(keywords: list[str]) -> tuple[qm.SparseVector, str]:
    if not keywords:
        return qm.SparseVector(indices=[], values=[]), ""
    seen: set[str] = set()
    tokens: list[str] = []
    for kw in keywords:
        for tok in kw.lower().split():
            if tok and tok not in seen:
                seen.add(tok)
                tokens.append(tok)
    if not tokens:
        return qm.SparseVector(indices=[], values=[]), ""
    text = " ".join(tokens)
    emb = vector_store.sparse_embeddings.embed_documents([text])[0]
    # FastEmbedSparse returns SparseEmbedding with numpy arrays; convert to lists for Qdrant model
    return qm.SparseVector(indices=emb.indices.tolist(), values=emb.values.tolist()), text


def points_for_batch(
    recs: list[dict[str, Any]],
    def_vecs: list[list[float]],
    q_vecs: list[list[float]],
    ctx_vecs: list[list[float]],
    sparse_vecs: list[qm.SparseVector],
) -> list[qm.PointStruct]:
    if not (len(recs) == len(def_vecs) == len(q_vecs) == len(ctx_vecs) == len(sparse_vecs)):
        raise ValueError("Record/vector count mismatch")
    points: list[qm.PointStruct] = []
    for rec, dvec, qvec, cvec, svec in zip(recs, def_vecs, q_vecs, ctx_vecs, sparse_vecs):
        pid = resolve_point_id(rec)
        payload = to_payload(rec)
        payload["id"] = str(pid)
        points.append(
            qm.PointStruct(
                id=pid,
                vector={
                    DEF_VECTOR_NAME: dvec,
                    QUESTION_VECTOR_NAME: qvec,
                    CONTEXT_VECTOR_NAME: cvec,
                    SPARSE_VECTOR_NAME: svec,
                },
                payload=payload,
            )
        )
    return points


# ---- Main ingestion ----------------------------------------------------------
def ingest() -> None:
    setup_logging(settings.log_level)
    logger.info(
        "Starting ingestion: collection=%s model=%s recreate=%s dry_run=%s",
        COLLECTION,
        EMBED_MODEL,
        RECREATE,
        DRY_RUN,
        extra={"module_name": ModuleName.INGESTION},
    )

    ensure_collection(recreate=RECREATE)

    records = load_indicators(INPUT_PATH)
    if LIMIT:
        records = records[:LIMIT]
    logger.info("Loaded %s records from %s", len(records), INPUT_PATH, extra={"module_name": ModuleName.INGESTION})

    # Batch process
    for start in range(0, len(records), BATCH_SIZE):
        batch = records[start : start + BATCH_SIZE]
        # validate
        for rec in batch:
            validate_record(rec)

        def_texts = [r.get("definition", "") for r in batch]
        q_texts = [r.get("question", "") for r in batch]
        ctx_texts = [r.get("application_context", "") for r in batch]

        logger.debug(
            "Embedding batch %s-%s (size=%s)",
            start,
            start + len(batch) - 1,
            len(batch),
            extra={"module_name": ModuleName.INGESTION},
        )
        def_vecs = embed_batch(def_texts)
        q_vecs = embed_batch(q_texts)
        ctx_vecs = embed_batch(ctx_texts)
        sparse_results = [sparse_from_keywords(r.get("keywords", [])) for r in batch]
        sparse_vecs = [sr[0] for sr in sparse_results]
        sparse_texts = [sr[1] for sr in sparse_results]
        logger.debug(
            "Embedding complete for batch %s-%s",
            start,
            start + len(batch) - 1,
            extra={"module_name": ModuleName.INGESTION},
        )

        pts = points_for_batch(batch, def_vecs, q_vecs, ctx_vecs, sparse_vecs)

        logger.info(
            "Upserting batch %s-%s (size=%s)",
            start,
            start + len(batch) - 1,
            len(batch),
            extra={"module_name": ModuleName.INGESTION},
        )
        if logger.isEnabledFor(10):  # DEBUG
            logger.debug("Sample payload: %s", pts[0].payload, extra={"module_name": ModuleName.INGESTION})
            for rec, dtext, qtext, ctext, kwtext in list(
                zip(batch, def_texts, q_texts, ctx_texts, sparse_texts)
            )[:10]:
                pid = rec.get("id") or stable_id(rec["normalized_indicator_name"])
                logger.debug(
                    "Record %s keywords_dedup=%r",
                    pid,
                    kwtext,
                    extra={"module_name": ModuleName.INGESTION},
                )

        if DRY_RUN:
            continue
        res = vector_store.upsert_points(pts, wait=True)
        logger.info(
            "Upserted batch %s-%s (size=%s) result=%s",
            start,
            start + len(batch) - 1,
            len(batch),
            res,
            extra={"module_name": ModuleName.INGESTION},
        )

    logger.info("Ingestion complete", extra={"module_name": ModuleName.INGESTION})


def main() -> None:
    ingest()


if __name__ == "__main__":
    main()

