"""
Check which indicators in Qdrant have corresponding .json files in the Cleaned folder.

Usage:
    uv run python "temporary script/check_cleaned_indicators.py"

This script fetches all indicators from Qdrant and checks if each has a corresponding
.json file in the Cleaned folder with the normalized_indicator_name.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any

from qdrant_client import QdrantClient

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from app.config.settings import settings  # noqa: E402

logger = logging.getLogger("check_cleaned_indicators")


def configure_logging() -> None:
    """Set up logging configuration."""
    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler()],
    )


def find_json_file(cleaned_root: Path, normalized_name: str) -> Path | None:
    """
    Search for a .json file with the given normalized name in the Cleaned folder.

    Args:
        cleaned_root: Path to the Cleaned folder
        normalized_name: The normalized indicator name to search for

    Returns:
        Path to the .json file if found, None otherwise
    """
    json_filename = f"{normalized_name}.json"

    # Use glob to find the file recursively
    pattern = f"**/{json_filename}"
    matches = list(cleaned_root.glob(pattern))

    if matches:
        return matches[0]  # Return the first match
    return None


def fetch_all_indicators(client: QdrantClient) -> list[dict[str, Any]]:
    """
    Fetch all indicators from Qdrant collection using scroll API.

    Args:
        client: Qdrant client instance

    Returns:
        List of indicator payloads from Qdrant
    """
    indicators = []
    offset = None

    while True:
        response = client.scroll(
            collection_name=settings.qdrant_collection,
            limit=100,  # Fetch in batches
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )

        points, next_offset = response

        for point in points:
            if point.payload:
                indicators.append(point.payload)

        if next_offset is None:
            break

        offset = next_offset

    return indicators


def main() -> None:
    configure_logging()
    logger.info("Starting check for cleaned indicators")

    # Paths
    cleaned_root = PROJECT_ROOT / "Cleaned"

    if not cleaned_root.exists():
        logger.error("Cleaned folder not found at %s", cleaned_root)
        return

    # Connect to Qdrant
    client = QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key)
    logger.info("Connected to Qdrant at %s", settings.qdrant_url)

    # Fetch all indicators
    logger.info("Fetching all indicators from Qdrant...")
    indicators = fetch_all_indicators(client)
    logger.info("Found %d indicators in Qdrant", len(indicators))

    # Check each indicator (only Country and Consumer sheets)
    found_count = 0
    missing_count = 0
    filtered_count = 0
    missing_indicators = []

    for indicator in indicators:
        metadata = indicator.get("metadata", {})
        sheet_name = metadata.get("sheet_name")

        # Only process Country and Consumer sheets
        if sheet_name not in ["Country", "Consumer"]:
            filtered_count += 1
            continue

        normalized_name = metadata.get("normalized_indicator_name")

        if not normalized_name:
            logger.warning("Indicator missing normalized_indicator_name: %s", indicator.get("id"))
            continue

        json_path = find_json_file(cleaned_root, normalized_name)

        if json_path:
            found_count += 1
            logger.debug("Found .json file for %s: %s", normalized_name, json_path)
        else:
            missing_count += 1
            missing_indicators.append({
                "normalized_name": normalized_name,
                "indicator_name": metadata.get("indicator_name", "Unknown"),
                "sheet_name": sheet_name,
                "id": indicator.get("id")
            })
            logger.info("Missing .json file for %s", normalized_name)

    # Report results
    logger.info("Check completed:")
    logger.info("  Total indicators in Qdrant: %d", len(indicators))
    logger.info("  Filtered out (not Country/Consumer): %d", filtered_count)
    logger.info("  Processed Country/Consumer indicators: %d", found_count + missing_count)
    logger.info("  Found .json files: %d", found_count)
    logger.info("  Missing .json files: %d", missing_count)

    if missing_indicators:
        logger.info("Missing indicators:")
        for missing in missing_indicators:
            logger.info("  - %s (%s) [%s] [ID: %s]",
                       missing["normalized_name"],
                       missing["indicator_name"],
                       missing["sheet_name"],
                       missing["id"])

        # Save missing indicators to file for reference
        missing_file = PROJECT_ROOT / "missing_cleaned_indicators.json"
        with open(missing_file, 'w', encoding='utf-8') as f:
            json.dump(missing_indicators, f, indent=2, ensure_ascii=False)
        logger.info("Saved list of missing indicators to %s", missing_file)


if __name__ == "__main__":
    main()
