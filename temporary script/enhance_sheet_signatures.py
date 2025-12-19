"""Enhance Qdrant sheet_signature metadata for the first four indicators.

This script inspects Postgres tables, extracts schema + sample rows, and then
updates the `metadata.sheet_signature` payload so downstream agents receive the
rich Format 1 structure described in the SQL Agent upgrade plan.

Usage:
    uv run python "temporary script/enhance_sheet_signatures.py"

Optional flags:
    --dry-run   Preview signatures without pushing updates to Qdrant.
"""

from __future__ import annotations

import argparse
import logging
import sys
import uuid
from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

from psycopg import sql
from psycopg.rows import dict_row
from qdrant_client import QdrantClient

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from app.adapters.postgres import get_connection  # noqa: E402
from app.config.settings import settings  # noqa: E402


logger = logging.getLogger("sheet_signature_enhancer")


@dataclass(frozen=True)
class IndicatorSpec:
    normalized_indicator_name: str
    table_name: str | None = None

    @property
    def table(self) -> str:
        return self.table_name or self.normalized_indicator_name


INDICATORS: tuple[IndicatorSpec, ...] = (
    IndicatorSpec("television_viewership"),
    IndicatorSpec("consumer_confidence_index_score"),
    IndicatorSpec("house_price_to_income_ratio"),
    IndicatorSpec("air_quality_index_major_indian_cities"),
)


def configure_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def point_id(normalized_indicator_name: str) -> str:
    """Mirror ingestion logic (UUID5 of normalized indicator name)."""

    return str(uuid.uuid5(uuid.NAMESPACE_URL, normalized_indicator_name))


def _serialize_value(value: Any) -> Any:
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    return value


def _fetch_table_signature(spec: IndicatorSpec) -> dict[str, Any]:
    query_columns = sql.SQL(
        """
        SELECT column_name, data_type
        FROM information_schema.columns
        WHERE table_schema = 'public' AND table_name = %s
        ORDER BY ordinal_position
        """
    )

    table_identifier = sql.Identifier(spec.table)
    count_query = sql.SQL("SELECT COUNT(*) AS count FROM {}" ).format(table_identifier)
    sample_query = sql.SQL("SELECT * FROM {} ORDER BY RANDOM() LIMIT %s" ).format(
        table_identifier
    )

    with get_connection() as conn, conn.cursor(row_factory=dict_row) as cur:
        cur.execute(query_columns, (spec.table,))
        columns = cur.fetchall()
        if not columns:
            raise ValueError(f"Table {spec.table} not found or has no columns")

        cur.execute(count_query)
        row_count_record = cur.fetchone()
        row_count = int(row_count_record["count"]) if row_count_record else 0

        sample_rows: list[dict[str, Any]] = []
        if row_count:
            limit = min(3, row_count)
            cur.execute(sample_query, (limit,))
            sample_rows = [
                {key: _serialize_value(value) for key, value in row.items()}
                for row in cur.fetchall()
            ]

    return {
        "table_name": spec.table,
        "row_count": row_count,
        "column_count": len(columns),
        "columns": [
            {"name": row["column_name"], "sql_type": row["data_type"].upper()}
            for row in columns
        ],
        "sample_rows": sample_rows,
    }


def _update_payload(
    client: QdrantClient, spec: IndicatorSpec, signature: dict[str, Any], *, dry_run: bool
) -> None:
    pid = point_id(spec.normalized_indicator_name)
    if dry_run:
        logger.info("[DRY RUN] %s => %s", spec.normalized_indicator_name, signature)
        return

    points = client.retrieve(
        collection_name=settings.qdrant_collection,
        ids=[pid],
        with_payload=True,
        with_vectors=False,
    )
    if not points:
        logger.error("Point not found for indicator %s (id %s)", spec.normalized_indicator_name, pid)
        return

    payload = points[0].payload or {}
    metadata = payload.get("metadata") or {}
    metadata["sheet_signature"] = signature
    payload["metadata"] = metadata

    client.set_payload(
        collection_name=settings.qdrant_collection,
        payload={"metadata": metadata},
        points=[pid],
    )
    logger.info(
        "Updated sheet_signature for %s | rows=%s cols=%s",
        spec.normalized_indicator_name,
        signature.get("row_count"),
        signature.get("column_count"),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="Preview without writing to Qdrant")
    args = parser.parse_args()

    configure_logging(settings.log_level)
    client = QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key)
    logger.info("Connected to Qdrant at %s", settings.qdrant_url)

    for spec in INDICATORS:
        logger.info("Processing %s (table=%s)", spec.normalized_indicator_name, spec.table)
        signature = _fetch_table_signature(spec)
        _update_payload(client, spec, signature, dry_run=args.dry_run)

    logger.info("Sheet signature enhancement complete")


if __name__ == "__main__":
    main()
