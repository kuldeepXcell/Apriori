"""Ingest indicators into Qdrant for the multivector_weighted pipeline."""

from __future__ import annotations

import ast
import json
import os
import re
import sys
import uuid
from io import BytesIO
from pathlib import Path
from typing import Any, Iterable, Optional

import openpyxl
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from supabase import create_client
from qdrant_client.http import models as qm

# Add src directory to path for imports
sys.path.insert(0, "/home/ubuntu/Desktop/APriori/apriori/src")

from app.adapters import embeddings, vector_store
from app.config.settings import settings
from app.core.logging import ModuleName, get_logger, setup_logging
from app.core.langsmith import configure_langsmith

# ---- Defaults (adjust in-file) ------------------------------------------------
# Dataset_directory workbook location.
DATASET_PATH = Path("/home/ubuntu/Desktop/APriori/apriori/data/Dataset_directory.xlsx")
# Root folder that contains indicator data folders.
CLEANED_SHEETS_DIR = Path("/home/ubuntu/Desktop/APriori/apriori/data/cleaned_excel_sheets")
# Supabase Storage config (public bucket).
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_SERVICE_ROLE_KEY") or os.environ.get("SUPABASE_KEY")
SUPABASE_BUCKET = os.environ.get("SUPABASE_BUCKET")
# Dataset sheets to ingest.
DATASET_SHEETS = ["Country", "Global Consumer"]
COLLECTION = settings.qdrant_collection
DEF_VECTOR_NAME = settings.definition_vector_name
QUESTION_VECTOR_NAME = settings.question_vector_name
CONTEXT_VECTOR_NAME = settings.context_vector_name
SPARSE_VECTOR_NAME = settings.sparse_vector_name
EMBED_MODEL = settings.embedding_model
EMBED_DIM = settings.embedding_dimensions
BATCH_SIZE = 10
DRY_RUN = True  # set True to skip upsert
RECREATE = False
# Process a sub-range of records (inclusive start, exclusive end); None means default.
RANGE_START: int | None = None
RANGE_END: int | None = None

logger = get_logger(__name__)

PLACEHOLDER_VALUES = {
    "",
    "-",
    "--",
    "n/a",
    "na",
    "none",
    "null",
    "tbd",
    "pending",
}

COLUMN_ALIASES = {
    "cleaning_status": ["cleaning status"],
    "section": ["section", "unnamed: 0"],
    "subsection": ["subsection"],
    "subsubsection": ["subsubsection"],
    "indicator_name": ["indicators", "indicator"],
    "normalized_indicator_name": [
        "normalized indicators name",
        "normalized indicator names",
        "normalised indicator name",
    ],
    "definition": ["definition"],
    "question": ["use case question"],
    "application_context": [
        "application",
        "application (context to how the indicator will have to be analysed)",
    ],
}


# ---- Helpers -----------------------------------------------------------------
def clean_cell(value: Any) -> Optional[str]:
    """Normalize cell text while filtering placeholder tokens."""
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    lowered = text.lower()
    if lowered in PLACEHOLDER_VALUES:
        return None
    return text


def normalize_header(value: Any) -> str:
    """Normalize header strings for matching."""
    if value is None:
        return ""
    return str(value).strip().lower()


def resolve_column_indices(headers: list[Any], sheet_name: str) -> dict[str, int]:
    """Map canonical field names to column indices."""
    normalized_headers = [normalize_header(h) for h in headers]
    indices: dict[str, int] = {}
    for field, aliases in COLUMN_ALIASES.items():
        for alias in aliases:
            if alias in normalized_headers:
                indices[field] = normalized_headers.index(alias)
                break
    # Fallback for Country sheet where the section header is blank.
    if "section" not in indices and sheet_name.strip().lower() == "country":
        if headers and (headers[0] is None or normalize_header(headers[0]) == ""):
            indices["section"] = 0
    missing = sorted(set(COLUMN_ALIASES.keys()) - set(indices.keys()))
    if missing:
        raise ValueError(f"{sheet_name} is missing required columns: {missing}")
    return indices


def resolve_sheet_name(target: str, sheet_names: list[str]) -> str:
    """Find a sheet name that matches the target after trimming whitespace."""
    target_key = target.strip().lower()
    for name in sheet_names:
        if name.strip().lower() == target_key:
            return name
    raise ValueError(f"Dataset sheet missing: {target}")


def build_indicator_index(root: Path) -> dict[str, list[Path]]:
    """Map normalized indicator names to one or more folder paths."""
    index: dict[str, list[Path]] = {}
    for json_path in root.rglob("*.json"):
        if json_path.stem != json_path.parent.name:
            continue
        key = json_path.stem.strip().lower()
        index.setdefault(key, []).append(json_path.parent)
    if not index:
        raise FileNotFoundError(f"No indicator metadata JSON files found under {root}")
    return index


def resolve_indicator_dir(
    index: dict[str, list[Path]],
    normalized: str,
    sheet_name: str,
) -> Path:
    """Choose a folder when the same normalized name appears in multiple domains."""
    candidates = index.get(normalized.strip().lower())
    if not candidates:
        raise FileNotFoundError(f"Missing indicator folder for {normalized}")
    if len(candidates) == 1:
        return candidates[0]
    sheet_key = sheet_name.strip().lower()
    if "consumer" in sheet_key:
        preferred = [p for p in candidates if "consumer" in {part.lower() for part in p.parts}]
    elif "country" in sheet_key:
        preferred = [p for p in candidates if "country" in {part.lower() for part in p.parts}]
    else:
        preferred = []
    if len(preferred) == 1:
        return preferred[0]
    raise ValueError(f"Duplicate indicator folder for {normalized}: {candidates}")


def iter_sheet_rows(ws: openpyxl.worksheet.worksheet.Worksheet, indices: dict[str, int]) -> Iterable[dict[str, Any]]:
    """Yield row dicts with merged section/subsection/subsubsection filled down."""
    current_section: Optional[str] = None
    current_subsection: Optional[str] = None
    current_subsubsection: Optional[str] = None
    for row in ws.iter_rows(min_row=2, values_only=True):
        # Forward-fill merged section values.
        section = clean_cell(row[indices["section"]]) if row else None
        subsection = clean_cell(row[indices["subsection"]]) if row else None
        subsubsection = clean_cell(row[indices["subsubsection"]]) if row else None
        if section is not None:
            current_section = section
        if subsection is not None:
            current_subsection = subsection
        if subsubsection is not None:
            current_subsubsection = subsubsection

        yield {
            "cleaning_status": clean_cell(row[indices["cleaning_status"]]) if row else None,
            "section": current_section,
            "subsection": current_subsection,
            "subsubsection": current_subsubsection,
            "indicator_name": clean_cell(row[indices["indicator_name"]]) if row else None,
            "normalized_indicator_name": clean_cell(row[indices["normalized_indicator_name"]]) if row else None,
            "definition": clean_cell(row[indices["definition"]]) if row else None,
            "question": clean_cell(row[indices["question"]]) if row else None,
            "application_context": clean_cell(row[indices["application_context"]]) if row else None,
        }


def is_completed(status: Optional[str]) -> bool:
    """Check if a cleaning status is marked as completed."""
    return status is not None and status.strip().lower() == "completed"


def require_value(value: Optional[str], field_name: str, sheet_name: str, row_number: int) -> str:
    """Raise a clear error if a required field is missing."""
    if value is None:
        raise ValueError(f"{sheet_name} row {row_number}: missing required '{field_name}'")
    return value


def validate_completed_row(row_data: dict[str, Any], sheet_name: str, row_number: int) -> None:
    """Enforce required fields for completed indicators."""
    require_value(row_data.get("indicator_name"), "indicator_name", sheet_name, row_number)
    require_value(row_data.get("normalized_indicator_name"), "normalized_indicator_name", sheet_name, row_number)
    require_value(row_data.get("definition"), "definition", sheet_name, row_number)
    require_value(row_data.get("question"), "question", sheet_name, row_number)
    require_value(row_data.get("application_context"), "application_context", sheet_name, row_number)
    require_value(row_data.get("section"), "section", sheet_name, row_number)
    require_value(row_data.get("subsection"), "subsection", sheet_name, row_number)
    require_value(row_data.get("subsubsection"), "subsubsection", sheet_name, row_number)


def count_valid_indicators(wb: openpyxl.Workbook, sheet_name: str) -> int:
    """Count valid completed indicators for a sheet."""
    ws = wb[sheet_name]
    headers = [cell.value for cell in next(ws.iter_rows(min_row=1, max_row=1))]
    indices = resolve_column_indices(headers, sheet_name)
    count = 0
    for offset, row_data in enumerate(iter_sheet_rows(ws, indices), start=2):
        if not is_completed(row_data.get("cleaning_status")):
            continue
        validate_completed_row(row_data, sheet_name, offset)
        count += 1
    return count


def select_single_file(paths: list[Path], label: str) -> Path:
    """Return a single file path or raise with a helpful error."""
    if not paths:
        raise FileNotFoundError(f"Missing {label} file")
    if len(paths) > 1:
        raise ValueError(f"Expected a single {label} file, found: {paths}")
    return paths[0]


def sanitize_sheet_name(name: str) -> str:
    """Normalize sheet names for file keys."""
    return re.sub(r"[^a-z0-9]+", "_", name.strip().lower()).strip("_")


def get_supabase_client():
    """Build a Supabase client for Storage uploads."""
    if not SUPABASE_URL or not SUPABASE_KEY or not SUPABASE_BUCKET:
        raise ValueError("Missing SUPABASE_URL, SUPABASE_KEY, or SUPABASE_BUCKET.")
    return create_client(SUPABASE_URL, SUPABASE_KEY)


def upload_parquet_files(indicator_dir: Path, workbook_path: Path) -> list[dict[str, str]]:
    """Convert each workbook sheet to parquet and upload to Supabase Storage."""
    client = get_supabase_client()
    parquet_files: list[dict[str, str]] = []
    excel = pd.ExcelFile(workbook_path, engine="openpyxl")
    relative_dir = indicator_dir.relative_to(CLEANED_SHEETS_DIR).as_posix()
    bucket = client.storage.from_(SUPABASE_BUCKET)
    for sheet_name in excel.sheet_names:
        df = excel.parse(sheet_name)
        table = pa.Table.from_pandas(df)
        buffer = BytesIO()
        pq.write_table(table, buffer)
        buffer.seek(0)
        object_path = f"{relative_dir}/{sanitize_sheet_name(sheet_name)}.parquet"
        bucket.upload(
            object_path,
            buffer,
            {"content-type": "application/x-parquet", "upsert": "true"},
        )
        public_url = bucket.get_public_url(object_path)
        if isinstance(public_url, dict):
            public_url = public_url.get("publicUrl") or public_url.get("public_url")
        parquet_files.append(
            {
                "sheet_name": sheet_name,
                "object_path": object_path,
                "public_url": public_url or "",
            }
        )
    return parquet_files


def extract_file_info(indicator_dir: Path) -> tuple[str, str, dict[str, Any]]:
    """Load data_source, last_updated_date, and sheet dimensions from indicator files."""
    # Find and read the metadata JSON file.
    json_candidates = sorted(indicator_dir.glob("*.json"))
    json_path = select_single_file(json_candidates, "metadata JSON")
    raw_metadata = json_path.read_text(encoding="utf-8")
    try:
        metadata = json.loads(raw_metadata)
    except json.JSONDecodeError:
        metadata = ast.literal_eval(raw_metadata)
    if not isinstance(metadata, dict):
        raise ValueError(f"Metadata file did not parse to a dict: {json_path}")
    data_source = metadata.get("data_source")
    last_updated_date = metadata.get("last_updated_date")
    if "data_source" not in metadata or "last_updated_date" not in metadata:
        raise ValueError(f"Missing data_source/last_updated_date keys in {json_path}")

    # Find and read the indicator workbook for sheet dimensions.
    workbook_candidates = [p for p in indicator_dir.glob("*.xlsx") if not p.name.startswith("~$")]
    workbook_path = select_single_file(workbook_candidates, "workbook")
    parquet_files = upload_parquet_files(indicator_dir, workbook_path)
    wb = openpyxl.load_workbook(workbook_path, read_only=True, data_only=True)
    sheet_dimensions = []
    for sheet in wb.sheetnames:
        ws = wb[sheet]
        # Count only rows/columns that contain at least one non-empty cell.
        max_row = 0
        max_col = 0
        for row_idx, row in enumerate(ws.iter_rows(values_only=True), start=1):
            row_has_data = False
            for col_idx, value in enumerate(row, start=1):
                if value is None:
                    continue
                if isinstance(value, str) and not value.strip():
                    continue
                row_has_data = True
                if col_idx > max_col:
                    max_col = col_idx
            if row_has_data:
                max_row = row_idx
        sheet_dimensions.append(
            {
                "sheet_name": sheet,
                "rows": max_row,
                "columns": max_col,
            }
        )

    file_info = {
        "file_type": "excel_workbooks_converted_to_postgres_tables",
        "data_source": data_source,
        "last_updated_date": last_updated_date,
        "number_of_sheets": len(wb.sheetnames),
        "sheet_dimensions": sheet_dimensions,
        "parquet_files": parquet_files,
    }
    return str(json_path), str(workbook_path), file_info


def load_indicators_from_dataset() -> list[dict[str, Any]]:
    """Extract completed indicators from Dataset_directory.xlsx and enrich metadata."""
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Dataset workbook not found at {DATASET_PATH}")
    if not CLEANED_SHEETS_DIR.exists():
        raise FileNotFoundError(f"Cleaned sheets folder not found at {CLEANED_SHEETS_DIR}")

    # Load the dataset directory workbook once.
    wb = openpyxl.load_workbook(DATASET_PATH, read_only=True, data_only=True)
    resolved_sheets = {name: resolve_sheet_name(name, wb.sheetnames) for name in DATASET_SHEETS}
    indicator_index = build_indicator_index(CLEANED_SHEETS_DIR)
    records: list[dict[str, Any]] = []

    # Count valid indicators per sheet before processing.
    for sheet_name in DATASET_SHEETS:
        resolved_name = resolved_sheets[sheet_name]
        valid_count = count_valid_indicators(wb, resolved_name)
        logger.info(
            "Valid indicators in %s: %s",
            sheet_name,
            valid_count,
            extra={"module_name": ModuleName.INGESTION},
        )

    # Build records with enrichment data.
    for sheet_name in DATASET_SHEETS:
        resolved_name = resolved_sheets[sheet_name]
        ws = wb[resolved_name]
        headers = [cell.value for cell in next(ws.iter_rows(min_row=1, max_row=1))]
        indices = resolve_column_indices(headers, resolved_name)
        for offset, row_data in enumerate(iter_sheet_rows(ws, indices), start=2):
            if not is_completed(row_data.get("cleaning_status")):
                continue
            validate_completed_row(row_data, sheet_name, offset)

            normalized = row_data["normalized_indicator_name"]
            indicator_dir = resolve_indicator_dir(indicator_index, normalized, sheet_name)

            logger.info(
                "Processing indicator %s (%s row %s)",
                normalized,
                sheet_name,
                offset,
                extra={"module_name": ModuleName.INGESTION},
            )

            path_in_drive, _workbook_path, file_info = extract_file_info(indicator_dir)

            records.append(
                {
                    "indicator_name": row_data["indicator_name"],
                    "normalized_indicator_name": normalized,
                    "question": row_data["question"],
                    "definition": row_data["definition"],
                    "application_context": row_data["application_context"],
                    "sheet_name": sheet_name,
                    "section": row_data["section"],
                    "subsection": row_data["subsection"],
                    "subsubsection": row_data["subsubsection"],
                    "path_in_drive": path_in_drive,
                    "file_info": file_info,
                    "keywords": None,
                }
            )

    return records


def resolve_point_id(rec: dict[str, Any]) -> uuid.UUID:
    """Always derive a stable UUID from the normalized name (ignore provided ids)."""
    return uuid.uuid5(uuid.NAMESPACE_URL, rec["normalized_indicator_name"])


def validate_record(rec: dict[str, Any]) -> None:
    """Validate the fully built ingestion record."""
    required = [
        "indicator_name",
        "normalized_indicator_name",
        "definition",
        "question",
        "application_context",
        "sheet_name",
        "section",
        "subsection",
        "subsubsection",
        "path_in_drive",
        "file_info",
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
    # Build the metadata payload used by LangChain.
    meta = {
        "indicator_name": rec.get("indicator_name"),
        "normalized_indicator_name": rec.get("normalized_indicator_name"),
        "question": rec.get("question"),
        "definition": rec.get("definition"),
        "application_context": rec.get("application_context"),
        "sheet_name": rec.get("sheet_name"),
        "section": rec.get("section"),
        "subsection": rec.get("subsection"),
        "subsubsection": rec.get("subsubsection"),
        "sparse_text": rec.get("sparse_text"),
        "path_in_drive": rec.get("path_in_drive"),
        "file_info": rec.get("file_info"),
        "keywords": rec.get("keywords"),
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
        "section",
        "subsection",
        "subsubsection",
        "sheet_name",
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
    """Ensure the target Qdrant collection exists."""
    logger.info(
        "Ensuring collection %s exists (recreate=%s)", COLLECTION, recreate, extra={"module_name": ModuleName.INGESTION}
    )
    vector_store.ensure_collection(recreate=recreate)


def embed_batch(texts: Iterable[str]) -> list[list[float]]:
    """Embed a batch of texts using the configured model."""
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
    """Assemble Qdrant points for the batch."""
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

    # Ensure collection exists before embedding.
    ensure_collection(recreate=RECREATE)

    # Load and enrich indicator records from the Dataset_directory workbook.
    records = load_indicators_from_dataset()

    # Apply optional range slicing (inclusive start, exclusive end).
    start_idx = RANGE_START or 0
    end_idx = RANGE_END if RANGE_END is not None else None
    records = records[start_idx:end_idx]

    logger.info(
        "Loaded %s records from Dataset_directory.xlsx (range %s:%s)",
        len(records),
        start_idx,
        "" if end_idx is None else end_idx,
        extra={"module_name": ModuleName.INGESTION},
    )

    # Batch process
    for start in range(0, len(records), BATCH_SIZE):
        batch = records[start : start + BATCH_SIZE]
        # Validate record fields before embedding.
        for rec in batch:
            validate_record(rec)

        # Extract text fields for embeddings.
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
        # Attach sparse text to each record for payload metadata.
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
