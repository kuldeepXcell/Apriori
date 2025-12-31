"""Update path_in_drive in Qdrant metadata payloads."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from qdrant_client import QdrantClient

from app.config.settings import settings

# ---- Defaults (adjust in-file) ------------------------------------------------
CLEANED_SHEETS_DIR = Path("/home/ubuntu/Desktop/APriori/apriori/data/cleaned_excel_sheets")
COLLECTION = settings.qdrant_collection
BATCH_SIZE = 100
FAIL_ON_MISSING = True


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
    sheet_name: str | None,
) -> Path:
    """Choose a folder when the same normalized name appears in multiple domains."""
    candidates = index.get(normalized.strip().lower())
    if not candidates:
        raise FileNotFoundError(f"Missing indicator folder for {normalized}")
    if len(candidates) == 1:
        return candidates[0]
    sheet_key = (sheet_name or "").strip().lower()
    if "consumer" in sheet_key:
        preferred = [p for p in candidates if "consumer" in {part.lower() for part in p.parts}]
    elif "country" in sheet_key:
        preferred = [p for p in candidates if "country" in {part.lower() for part in p.parts}]
    else:
        preferred = []
    if len(preferred) == 1:
        return preferred[0]
    raise ValueError(f"Duplicate indicator folder for {normalized}: {candidates}")


def main() -> None:
    if not CLEANED_SHEETS_DIR.exists():
        raise FileNotFoundError(f"Cleaned sheets folder not found at {CLEANED_SHEETS_DIR}")

    client = QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key)
    indicator_index = build_indicator_index(CLEANED_SHEETS_DIR)

    updated = 0
    missing = 0
    offset = None

    while True:
        points, offset = client.scroll(
            collection_name=COLLECTION,
            limit=BATCH_SIZE,
            with_payload=True,
            with_vectors=False,
            offset=offset,
        )
        if not points:
            break

        for point in points:
            payload: dict[str, Any] = point.payload or {}
            meta: dict[str, Any] = payload.get("metadata") or {}
            normalized = meta.get("normalized_indicator_name") or payload.get("normalized_indicator_name")
            if not normalized:
                continue
            sheet_name = meta.get("sheet_name")
            try:
                indicator_dir = resolve_indicator_dir(indicator_index, str(normalized), sheet_name)
            except FileNotFoundError:
                missing += 1
                if FAIL_ON_MISSING:
                    raise
                continue

            path_in_drive = indicator_dir.relative_to(CLEANED_SHEETS_DIR).as_posix()
            if meta.get("path_in_drive") == path_in_drive:
                continue

            meta["path_in_drive"] = path_in_drive
            client.set_payload(
                collection_name=COLLECTION,
                payload={"metadata": meta},
                points=[point.id],
            )
            updated += 1

        if offset is None:
            break

    print(json.dumps({"updated": updated, "missing": missing}, indent=2))


if __name__ == "__main__":
    main()
