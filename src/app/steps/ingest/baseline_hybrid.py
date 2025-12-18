"""Ingest indicators into Qdrant for the multivector_weighted pipeline."""

from __future__ import annotations

import uuid
import json
from pathlib import Path
import sys
from typing import Any, Iterable, List
from qdrant_client.http import models as qm

# Add src directory to path for imports
import sys
sys.path.insert(0, "/home/ubuntu/Desktop/APriori/Apriori/src")

from app.adapters import embeddings, vector_store
from app.config.settings import settings
from app.core.logging import ModuleName, get_logger, setup_logging
from app.core.langsmith import configure_langsmith

# ---- Defaults (adjust in-file) ------------------------------------------------
INPUT_PATH = Path("/home/ubuntu/Desktop/APriori/Apriori/data/all_indicators_final.json")
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
# Process a sub-range of records (inclusive start, exclusive end); None means default.
RANGE_START: int | None = None
RANGE_END: int | None = None

logger = get_logger(__name__)


# ---- Helpers -----------------------------------------------------------------
def resolve_point_id(rec: dict[str, Any]) -> uuid.UUID:
    """Always derive a stable UUID from the normalized name (ignore provided ids)."""
    return uuid.uuid5(uuid.NAMESPACE_URL, rec["normalized_indicator_name"])


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
    ]
    missing = [k for k in required if k not in rec or rec[k] in (None, "", [])]
    if missing:
        raise ValueError(f"Missing required fields {missing} for record {rec}")


def to_payload(rec: dict[str, Any]) -> dict[str, Any]:
    """
    Build payload for Qdrant.

    LangChain QdrantVectorStore only exposes `payload[metadata_payload_key]`
    (default "metadata") via Document.metadata, and uses `content_payload_key`
    for page_content. Per request, we keep the normalized name as page_content
    and put everything else in metadata.
    """
    meta = {
        "indicator_name": rec.get("indicator_name"),
        "normalized_indicator_name": rec.get("normalized_indicator_name"),
        "question": rec.get("question"),
        "definition": rec.get("definition"),
        "application_context": rec.get("application_context"),
        "sheet_name": rec.get("sheet_name"),
        "subsection": rec.get("subsection"),
        "subsubsection": rec.get("subsubsection"),
        "keywords": rec.get("keywords", []),
        "sparse_text": rec.get("sparse_text"),
    }
    payload = {
        # page_content will come from this key
        "normalized_indicator_name": rec.get("normalized_indicator_name"),
        # metadata consumed by LangChain
        "metadata": meta,
    }
    return payload


def build_sparse_source_text(rec: dict[str, Any]) -> str:
    """
    Combine indicator fields into a single BM25-ready string.

    This keeps ingestion free of LLM-based keyword generation by letting the
    Qdrant BM25 sparse model tokenize the combined text.
    """
    parts: list[str] = []
    for key in [
        "indicator_name",
        "normalized_indicator_name",
        "question",
        "definition",
        "application_context",
        "subsection",
        "subsubsection",
    ]:
        val = rec.get(key)
        if isinstance(val, str):
            cleaned = val.strip()
            if cleaned:
                parts.append(cleaned)
    if not parts:
        raise ValueError("Sparse source text is empty; ensure indicator fields are populated.")
    return " | ".join(parts)


def ensure_collection(recreate: bool = False) -> None:
    logger.info(
        "Ensuring collection %s exists (recreate=%s)", COLLECTION, recreate, extra={"module_name": ModuleName.INGESTION}
    )
    vector_store.ensure_collection(recreate=recreate)


def embed_batch(texts: Iterable[str]) -> list[list[float]]:
    return embeddings.embed_texts(texts, model=EMBED_MODEL, dimensions=EMBED_DIM)


def sparse_from_text(text: str) -> tuple[qm.SparseVector, str]:
    """
    Generate a sparse vector directly from combined indicator text using BM25.
    """
    cleaned = " ".join(text.split())
    if not cleaned:
        return qm.SparseVector(indices=[], values=[]), ""
    emb = vector_store.sparse_embeddings.embed_documents([cleaned])[0]
    # FastEmbedSparse may return numpy arrays or plain lists depending on backend.
    indices = emb.indices.tolist() if hasattr(emb.indices, "tolist") else list(emb.indices)
    values = emb.values.tolist() if hasattr(emb.values, "tolist") else list(emb.values)
    return qm.SparseVector(indices=indices, values=values), cleaned


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
    configure_langsmith(settings)
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

    # Apply optional range slicing (inclusive start, exclusive end).
    start_idx = RANGE_START or 0
    end_idx = RANGE_END if RANGE_END is not None else None
    records = records[start_idx:end_idx]

    logger.info(
        "Loaded %s records from %s (range %s:%s)",
        len(records),
        INPUT_PATH,
        start_idx,
        "" if end_idx is None else end_idx,
        extra={"module_name": ModuleName.INGESTION},
    )

    # Batch process
    for start in range(0, len(records), BATCH_SIZE):
        batch = records[start : start + BATCH_SIZE]
        # validate
        for rec in batch:
            validate_record(rec)

        def_texts = [r.get("definition", "") for r in batch]
        q_texts = [r.get("question", "") for r in batch]
        ctx_texts = [r.get("application_context", "") for r in batch]
        sparse_source_texts = [build_sparse_source_text(r) for r in batch]

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
        sparse_results = [sparse_from_text(text) for text in sparse_source_texts]
        sparse_vecs = [sr[0] for sr in sparse_results]
        sparse_texts = [sr[1] for sr in sparse_results]
        for rec, sparse_text in zip(batch, sparse_texts):
            rec["sparse_text"] = sparse_text
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
            for rec, dtext, qtext, ctext, sparse_text in list(
                zip(batch, def_texts, q_texts, ctx_texts, sparse_texts)
            )[:10]:
                pid = rec.get("id") or uuid.uuid5(uuid.NAMESPACE_URL, rec["normalized_indicator_name"])
                logger.debug(
                    "Record %s sparse_text=%r",
                    pid,
                    sparse_text,
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
