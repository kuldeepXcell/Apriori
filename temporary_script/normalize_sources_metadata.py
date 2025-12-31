"""Normalize cleaned_excel_sheets metadata JSONs to use a sources array."""

from __future__ import annotations

import ast
import json
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
CLEANED_SHEETS_DIR = ROOT_DIR / "data" / "cleaned_excel_sheets"


def load_metadata(path: Path) -> dict:
    """Load metadata file that may contain JSON or Python literals."""
    raw = path.read_text(encoding="utf-8")
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        data = ast.literal_eval(raw)
    if not isinstance(data, dict):
        raise ValueError(f"Metadata file did not parse to a dict: {path}")
    return data


def normalize_sources(data: dict) -> dict:
    """Ensure metadata includes a sources array."""
    sources = data.get("sources")
    if isinstance(sources, list) and sources:
        data.pop("data_source", None)
        data.pop("description", None)
        return data

    data_source = data.get("data_source")
    description = data.get("description")
    if "data_source" in data:
        data["sources"] = [
            {
                "data_source": data_source,
                "description": description,
            }
        ]
        data.pop("data_source", None)
        data.pop("description", None)
        return data

    raise ValueError("Missing sources/data_source")


def main() -> None:
    if not CLEANED_SHEETS_DIR.exists():
        raise FileNotFoundError(f"Cleaned sheets folder not found at {CLEANED_SHEETS_DIR}")

    updated = 0
    for json_path in CLEANED_SHEETS_DIR.rglob("*.json"):
        if json_path.stem != json_path.parent.name:
            continue
        data = load_metadata(json_path)
        try:
            normalized = normalize_sources(data)
        except ValueError:
            raise ValueError(f"Missing sources/data_source in {json_path}") from None
        json_path.write_text(json.dumps(normalized, indent=2, ensure_ascii=True), encoding="utf-8")
        updated += 1

    print(f"Normalized sources for {updated} metadata files.")


if __name__ == "__main__":
    main()
