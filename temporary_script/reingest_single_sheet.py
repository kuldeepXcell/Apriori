"""Re-ingest a single sheet into Postgres using the same ingest helpers."""

from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

# Add src directory to path for imports.
sys.path.insert(0, "/home/ubuntu/Desktop/APriori/apriori/src")

from app.adapters.postgres import get_connection
from app.steps.ingest.postgres_ingest import (
    MAPPING_TABLE,
    SCHEMA,
    coerce_text_columns,
    create_table,
    ensure_mapping_table,
    ensure_unique_columns,
    infer_sql_type,
    insert_rows,
    normalize_identifier,
    normalize_null_tokens,
)

# ---- Inputs (adjust in-file) -------------------------------------------------
WORKBOOK_PATH = Path(
    "/home/ubuntu/Desktop/APriori/apriori/data/cleaned_excel_sheets/global_consumer/"
    "socioeconomic_profile/social_classification/global_classification_of_countries_by_income_levels/"
    "global_classification_of_countries_by_income_levels.xlsx"
)
INDICATOR_NAME = "global_classification_of_countries_by_income_levels"
SHEET_NAME = "notes"


def main() -> None:
    if not WORKBOOK_PATH.exists():
        raise FileNotFoundError(f"Workbook not found at {WORKBOOK_PATH}")

    excel = pd.ExcelFile(WORKBOOK_PATH, engine="openpyxl")
    if SHEET_NAME not in excel.sheet_names:
        raise ValueError(f"Sheet '{SHEET_NAME}' not found in {WORKBOOK_PATH}")

    # Load and normalize the sheet data.
    df = excel.parse(SHEET_NAME)
    df = normalize_null_tokens(df)
    df = coerce_text_columns(df, INDICATOR_NAME, SHEET_NAME)

    # Normalize and infer columns for table creation.
    df.columns = ensure_unique_columns([str(col) if col is not None else "column" for col in df.columns])
    columns = [(col, infer_sql_type(df[col])) for col in df.columns]

    table_name = normalize_identifier(SHEET_NAME)

    # Insert rows and refresh mapping entry.
    with get_connection(autocommit=False) as conn:
        with conn.cursor() as cursor:
            if SCHEMA != "public":
                cursor.execute(f'CREATE SCHEMA IF NOT EXISTS "{SCHEMA}"')
            ensure_mapping_table(cursor)
            create_table(cursor, table_name, columns)
            insert_rows(cursor, table_name, df)
            mapping = f"\"{SCHEMA}\".\"{MAPPING_TABLE}\""
            cursor.execute(
                f"DELETE FROM {mapping} WHERE indicator_name=%s AND sheet_name=%s AND table_name=%s",
                (INDICATOR_NAME, SHEET_NAME, table_name),
            )
            cursor.execute(
                f"INSERT INTO {mapping} (indicator_name, sheet_name, table_name) VALUES (%s, %s, %s)",
                (INDICATOR_NAME, SHEET_NAME, table_name),
            )
        conn.commit()

    print(f"Re-ingested '{SHEET_NAME}' into {SCHEMA}.{table_name}")


if __name__ == "__main__":
    main()
