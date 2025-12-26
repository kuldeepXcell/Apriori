"""
Temporary utility to build all_indicators_v2.json from Dataset_directory.xlsx.
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]
DATASET_PATH = BASE_DIR / "data" / "Dataset_directory.xlsx"
OUTPUT_PATH = BASE_DIR / "data" / "all_indicators_v2.json"

SKIP_INDICATOR_NAMES: list[str] = [
    "Marital Status Distribution (% Never Married, Married, Divorced, Widowed)",   #this all 
    "Median Household Income",
    "Mean Household Income",
    "Primary Education Completion Rate (% of relevant age group)",
    "Educational Attainment (% of population with Secondary, Tertiary education)",
    "Labor Force Participation Rate (Total)",
    "Population Distribution by State/Region",
    "Household Final Consumption Expenditure (HFCE) (% of GDP annual growth)",
    "Average Monthly Household Expenditure (MPCE)",
    "Mean Annual Household Expenditure by Category (Food, Housing, Transport, etc)",
    "Share of Wallet (% of total spend on Essentials vs. Discretionary)",
    "Monthly Per Capita Consumption Expenditure (MPCE) by Income Quintile",
    "Rural vs. Urban Expenditure Differential by Category",
    "Alcohol Consumption (Litres of pure alcohol per capita per year)",
    "Tobacco Use Prevalence (% of adults)",

    "World Trade Volume Growth",  # question and application is missing
    "Groups by Caste (Pg 17 >)",  # normalized indicator name is missing
    "Personal Disposable Income (India)",  # red
    "Household Disposable Income (Global)",  # red
    "New Consumer Classification System (NCCS) Distribution (A1 to E3)",  # red
    "Gross Household Savings Rate (% of GDP)",  # red
    "Dietary Pattern (% Vegetarian, % Non-Vegetarian)",  # red
    "Public vs Private Healthcare usage"  # red
    ]

SHEETS = ["Country", "Consumer", "Company", "Category"]

COLUMN_CONFIG = {
    "section_col": "section",
    "subsection_col": "Subsection",
    "subsubsection_col": "SubSubSection",
    "indicator_col": "Indicators",
    "normalized_col": "Normalized Indicators name",
    "definition_col": "Definition",
    "question_col": "Use Case Question",
    "application_col": "Application",
}


def clean_text(value: Any) -> Optional[str]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    if isinstance(value, str):
        text = value.strip()
    else:
        if pd.isna(value):
            return None
        text = str(value).strip()
    return text if text else None


def require_value(
    value: Optional[str],
    field_name: str,
    sheet_name: str,
    row_number: int,
) -> str:
    if value is None:
        raise ValueError(
            f"{sheet_name} row {row_number}: missing required '{field_name}'"
        )
    return value


def build_sparse_text(values: List[Optional[str]]) -> str:
    return " ".join(value for value in values if value)


def process_sheet(sheet_name: str, config: Dict[str, str]) -> List[Dict[str, Any]]:
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
    skipped: List[str] = []
    sheet_name_lower = sheet_name.lower()

    for row_index, row in df.iterrows():
        row_number = row_index + 2

        indicator_name = clean_text(row[config["indicator_col"]])
        normalized_name = clean_text(row[config["normalized_col"]])
        definition = clean_text(row[config["definition_col"]])
        question = clean_text(row[config["question_col"]])
        application = clean_text(row[config["application_col"]])

        if indicator_name is None:
            if any([normalized_name, definition, question, application]):
                raise ValueError(
                    f"{sheet_name} row {row_number}: indicator name is empty"
                )
            continue

        if indicator_name in SKIP_INDICATOR_NAMES:
            skipped.append(f"row {row_number}: {indicator_name}")
            continue
        normalized_name = require_value(
            normalized_name, "normalized_indicator_name", sheet_name, row_number
        )
        definition = require_value(definition, "definition", sheet_name, row_number)
        question = require_value(question, "question", sheet_name, row_number)
        application = require_value(
            application, "application_context", sheet_name, row_number
        )

        section = require_value(
            clean_text(row[config["section_col"]]), "section", sheet_name, row_number
        )
        subsection = require_value(
            clean_text(row[config["subsection_col"]]),
            "subsection",
            sheet_name,
            row_number,
        )
        subsubsection = require_value(
            clean_text(row[config["subsubsection_col"]]),
            "subsubsection",
            sheet_name,
            row_number,
        )

        path = f"{sheet_name_lower}/{section}/{subsection}/{subsubsection}"
        sparse_text = build_sparse_text(
            [
                indicator_name,
                normalized_name,
                question,
                definition,
                application,
                section,
                subsection,
                subsubsection,
                path,
                sheet_name_lower,
            ]
        )

        indicators.append(
            {
                "id": str(uuid.uuid4()),
                "normalized_indicator_name": normalized_name,
                "metadata": {
                    "indicator_name": indicator_name,
                    "normalized_indicator_name": normalized_name,
                    "question": question,
                    "definition": definition,
                    "application_context": application,
                    "sheet_name": sheet_name_lower,
                    "section": section,
                    "subsection": subsection,
                    "subsubsection": subsubsection,
                    "path": path,
                    "sparse_text": sparse_text,
                    "file_info": [],
                    # Example future shape (varies by file type):
                    # "file_info": [
                    #   {
                    #     "file_type": "excel_workbook_converted_to_tables_in_db",
                    #     "number_of_sheets_inside": 5,
                    #   }
                    # ]
                    # or
                    # "file_info": [
                    #   {
                    #     "file_type": "text",
                    #     "number_of_chunks_200_words": 8,
                    #   }
                    # ]
                },
            }
        )

    if skipped:
        print(f"{sheet_name}: skipped {len(skipped)} indicators:")
        for entry in skipped:
            print(f"  - {entry}")

    return indicators


def main() -> None:
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Workbook not found at {DATASET_PATH}")

    sheets_payload: List[Dict[str, Any]] = []
    for sheet_name in SHEETS:
        indicators = process_sheet(sheet_name, COLUMN_CONFIG)
        sheets_payload.append(
            {
                "sheet_name": sheet_name.lower(),
                "total_indicators": len(indicators),
                "indicators": indicators,
            }
        )
        print(f"{sheet_name}: captured {len(indicators)} indicators")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {"note": "", "sheets": sheets_payload}
    OUTPUT_PATH.write_text(json.dumps(payload, indent=2))
    print(f"Wrote {OUTPUT_PATH.relative_to(BASE_DIR)}")


if __name__ == "__main__":
    main()
