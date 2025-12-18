"""Populate sample table metadata for selected indicators in Qdrant.

This script updates the `metadata.sheet_signature.sample_table` field for
the first batch of indicators so the UI can display a concise schema preview.

Usage:
    uv run python temporary\ script/update_sample_tables.py
"""

from __future__ import annotations

import logging
import sys
import uuid
from pathlib import Path

from qdrant_client import QdrantClient

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from app.config.settings import settings  # noqa: E402


logger = logging.getLogger("sample_table_updater")


INDICATOR_CONFIGS: list[dict[str, object]] = [
    {
        "normalized_indicator_name": "television_viewership",
        "sample_table": [
            ["year", "2018", "2019", "2020", "2021", "2022", "2023", "2024"],
            [
                "Television viewership across India from 2018 to 2024 (in billion AMAs)",
                "1604",
                "1614",
                "1731",
                "1591",
                "1474",
                "1508",
                "1530",
            ],
        ],
    },
    {
        "normalized_indicator_name": "consumer_confidence_index_score",
        "sample_table": [
            ["country_name", "consumer_confidence_index"],
            ["Global - All 32", "48.5"],
        ],
    },
    {
        "normalized_indicator_name": "house_price_to_income_ratio",
        "sample_table": [
            ["ref_area", "ref_area_label", "2021-q4"],
            ["ARE", "United Arab Emirates", "76.541"],
        ],
    },
    {
        "normalized_indicator_name": "air_quality_index_major_indian_cities",
        "sample_table": [
            [
                "state",
                "city",
                "2022 (No. of Days) - Good",
                "2022_no_of_days_satisfactory",
                "2022_no_of_days_moderate",
                "2022_no_of_days_poor",
                "2022_no_of_days_very_poor",
                "2022_no_of_days_severe",
                "2023_no_of_days_good",
                "…",
                "2024_no_of_days_severe",
            ],
            ["Maharashtra", "Mumbai", "21", "152", "148", "42", "1", "0", "5", "…", "0"],
        ],
    },
]


def configure_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def point_id(normalized_indicator_name: str) -> str:
    """Mirror ingestion logic (UUID5 of normalized indicator name)."""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, normalized_indicator_name))


def update_indicator(client: QdrantClient, config: dict[str, object]) -> None:
    norm_name = config["normalized_indicator_name"]
    sample_table = config["sample_table"]
    pid = point_id(norm_name)
    logger.info("Updating %s (point id %s)", norm_name, pid)

    points = client.retrieve(
        collection_name=settings.qdrant_collection,
        ids=[pid],
        with_payload=True,
        with_vectors=False,
    )
    if not points:
        logger.error("Point not found for indicator %s (id %s)", norm_name, pid)
        return

    payload = points[0].payload or {}
    metadata = payload.get("metadata") or {}
    sheet_signature = metadata.get("sheet_signature") or {}
    sheet_signature["sample_table"] = sample_table
    metadata["sheet_signature"] = sheet_signature
    payload["metadata"] = metadata

    client.set_payload(
        collection_name=settings.qdrant_collection,
        payload={"metadata": metadata},
        points=[pid],
    )
    logger.info("Sample table stored for %s", norm_name)


def main() -> None:
    configure_logging(settings.log_level)
    client = QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key)
    logger.info("Connected to Qdrant at %s", settings.qdrant_url)

    for config in INDICATOR_CONFIGS:
        update_indicator(client, config)

    logger.info("Completed sample table updates.")


if __name__ == "__main__":
    main()
