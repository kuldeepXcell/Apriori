"""
Temporary utility to extract indicator metadata from Dataset_directory.xlsx.
Outputs all_indicators.json mirroring combined_indicators.json schema and
prints duplicate indicators detected by normalized name.
"""

from __future__ import annotations

import json
import re
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]
DATASET_PATH = BASE_DIR / "Dataset_directory.xlsx"
OUTPUT_PATH = BASE_DIR / "data" / "all_indicators.json"

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

SHEET_CONFIGS = {
    "Country": {
        "section_col": "Unnamed: 0",
        "subsection_col": "Subsection",
        "subsubsection_col": "SubSubSection",
        "indicator_col": "Indicators",
        "normalized_col": "Normalized Indicators name",
        "definition_col": "Definition",
        "question_col": "Use Case Question",
        "application_col": "Application (context to how the indicator will have to be analysed)",
    },
    "Consumer": {
        "section_col": "Section",
        "subsection_col": "Subsection",
        "subsubsection_col": "SubSubSection",
        "indicator_col": "Indicators",
        "normalized_col": "Normalised Indicator name",
        "definition_col": "Definition",
        "question_col": "Use Case Question",
        "application_col": "Application (context to how the indicator will have to be analysed)",
    },
    "Company": {
        "section_col": "Section",
        "subsection_col": "Subsection",
        "subsubsection_col": "SubSubSection",
        "indicator_col": "Indicators",
        "normalized_col": "Normalized Indicator Names",
        "definition_col": "Definition",
        "question_col": "Use Case Question",
        "application_col": "Application",
    },
    "Category": {
        "section_col": "Section",
        "subsection_col": "Subsection",
        "subsubsection_col": "SubSubSection",
        "indicator_col": "Indicators",
        "normalized_col": "Normalized Indicator Names",
        "definition_col": "Definition",
        "question_col": "Use Case Question",
        "application_col": "Application",
    },
}


def clean_text(value: Any) -> Optional[str]:
    """Convert cell values to stripped strings while filtering placeholder tokens."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    if isinstance(value, str):
        text = value.strip()
    else:
        if pd.isna(value):
            return None
        text = str(value).strip()
    lowered = text.lower()
    if lowered in PLACEHOLDER_VALUES:
        return None
    return text if text else None


def slugify_indicator(name: Optional[str]) -> Optional[str]:
    """Create a query-friendly identifier from an indicator name."""
    if not name:
        return None
    normalized = unicodedata.normalize("NFKD", name)
    normalized = "".join(ch for ch in normalized if not unicodedata.combining(ch))
    normalized = normalized.lower()
    normalized = re.sub(r"[^a-z0-9]+", "_", normalized)
    normalized = re.sub(r"_+", "_", normalized).strip("_")
    return normalized or None


def process_sheet(
    sheet_name: str,
    config: Dict[str, str],
    duplicate_tracker: Dict[str, List[Dict[str, str]]],
) -> List[Dict[str, Any]]:
    """Parse a single sheet and return normalized indicator entries."""
    df = pd.read_excel(DATASET_PATH, sheet_name=sheet_name)
    df = df.rename(columns=lambda c: c.strip() if isinstance(c, str) else c)

    required_columns = {
        config["section_col"],
        config["subsection_col"],
        config["subsubsection_col"],
        config["indicator_col"],
        config["normalized_col"],
        config["definition_col"],
        config["question_col"],
        config["application_col"],
    }
    missing = sorted(required_columns - set(df.columns))
    if missing:
        raise ValueError(f"{sheet_name} sheet is missing columns: {missing}")

    df[[config["section_col"], config["subsection_col"], config["subsubsection_col"]]] = (
        df[[config["section_col"], config["subsection_col"], config["subsubsection_col"]]].ffill()
    )

    indicators: List[Dict[str, Any]] = []

    for _, row in df.iterrows():
        section = clean_text(row[config["section_col"]])
        subsection = clean_text(row[config["subsection_col"]])
        subsubsection = clean_text(row[config["subsubsection_col"]])
        indicator_name = clean_text(row[config["indicator_col"]])
        definition = clean_text(row[config["definition_col"]])
        question = clean_text(row[config["question_col"]])
        application = clean_text(row[config["application_col"]])
        normalized_candidate = clean_text(row[config["normalized_col"]])

        if not indicator_name or not definition or not question or not application:
            continue

        normalized_name = normalized_candidate or slugify_indicator(indicator_name)

        entry = {
            "section": section,
            "subsection": subsection,
            "subsubsection": subsubsection,
            "indicator_name": indicator_name,
            "normalized_indicator_name": normalized_name,
            "definition": definition,
            "question": question,
            "application_context": application,
            "keywords": None,
        }
        indicators.append(entry)

        duplicate_key = normalized_name or indicator_name
        if duplicate_key:
            duplicate_tracker.setdefault(duplicate_key, []).append(
                {"sheet": sheet_name, "indicator_name": indicator_name}
            )

    return indicators


def main() -> None:
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Workbook not found at {DATASET_PATH}")

    extraction_timestamp = datetime.now(timezone.utc).isoformat()
    duplicate_tracker: Dict[str, List[Dict[str, str]]] = {}
    sheets_payload: List[Dict[str, Any]] = []

    for sheet_name, config in SHEET_CONFIGS.items():
        indicators = process_sheet(sheet_name, config, duplicate_tracker)
        sheets_payload.append(
            {
                "sheet_name": sheet_name,
                "total_indicators": len(indicators),
                "extraction_date": extraction_timestamp,
                "indicators": indicators,
            }
        )
        print(f"{sheet_name}: captured {len(indicators)} indicators")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps({"sheets": sheets_payload}, indent=2))
    print(f"Wrote {OUTPUT_PATH.relative_to(BASE_DIR)}")

    duplicates = {k: v for k, v in duplicate_tracker.items() if len(v) > 1}
    if duplicates:
        print("\nPotential duplicates detected (by normalized name):")
        for norm_name, entries in duplicates.items():
            print(f"  {norm_name} -> {entries}")
    else:
        print("\nNo duplicates detected.")


if __name__ == "__main__":
    main()
