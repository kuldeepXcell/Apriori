"""
One-off helper to ingest a handful of Excel indicators into the local Postgres DB.

Usage:
    UV_CACHE_DIR=.uv-cache uv run python "temporary script/ingest_sample_indicators.py"
"""

from __future__ import annotations

import io
from pathlib import Path

import pandas as pd
from psycopg import sql

from app.adapters.postgres import get_connection

DATA_ROOT = Path("data/cleaned_excel_sheets")

INDICATOR_FILES: dict[str, Path] = {
    "television_viewership": DATA_ROOT
    / "consumer"
    / "media_consumption_habits"
    / "traditional_media"
    / "television_viewership"
    / "television_viewership.xlsx",
    "consumer_confidence_index_score": DATA_ROOT
    / "consumer"
    / "lifestyle_values_health"
    / "values_and_attitudes"
    / "consumer_confidence_index_score"
    / "consumer_confidence_index_score.xlsx",
    "house_price_to_income_ratio": DATA_ROOT
    / "country"
    / "prices_and_inflation"
    / "asset_and_commodity_prices"
    / "house_price_to_income_ratio"
    / "house_price_to_income_ratio.xlsx",
    "air_quality_index_major_indian_cities": DATA_ROOT
    / "country"
    / "ecological_climate"
    / "pollution_and_resources"
    / "air_quality_index_major_indian_cities"
    / "air_quality_index_major_indian_cities.xlsx",
}


def snake_case(name: str) -> str:
    cleaned = (
        name.strip()
        .lower()
        .replace(" ", "_")
        .replace("-", "_")
        .replace("%", "pct")
        .replace("/", "_")
    )
    return "".join(ch for ch in cleaned if ch.isalnum() or ch == "_")


def infer_sql_type(series: pd.Series) -> str:
    if pd.api.types.is_integer_dtype(series.dropna()):
        return "BIGINT"
    if pd.api.types.is_float_dtype(series.dropna()):
        return "DOUBLE PRECISION"
    return "TEXT"


def load_dataframe(path: Path) -> pd.DataFrame:
    xl = pd.ExcelFile(path)
    sheet_name = xl.sheet_names[0]
    df = xl.parse(sheet_name)
    df = df.rename(columns=lambda col: snake_case(str(col)))
    return df


def create_table(table_name: str, df: pd.DataFrame) -> None:
    with get_connection() as conn, conn.cursor() as cur:
        cur.execute(sql.SQL("DROP TABLE IF EXISTS {}").format(sql.Identifier(table_name)))
        column_defs = [
            sql.SQL("{} {}").format(sql.Identifier(column), sql.SQL(infer_sql_type(df[column])))
            for column in df.columns
        ]
        create_stmt = sql.SQL("CREATE TABLE {} ({})").format(
            sql.Identifier(table_name), sql.SQL(", ").join(column_defs)
        )
        cur.execute(create_stmt)

        buffer = io.StringIO()
        df.to_csv(buffer, index=False, header=False)
        buffer.seek(0)

        copy_stmt = sql.SQL("COPY {} ({}) FROM STDIN WITH (FORMAT CSV)").format(
            sql.Identifier(table_name), sql.SQL(", ").join(map(sql.Identifier, df.columns))
        )
        with cur.copy(copy_stmt) as copy:
            copy.write(buffer.getvalue())


def ingest_indicator(table_name: str, path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(path)
    df = load_dataframe(path)
    create_table(table_name, df)
    print(f"Ingested {table_name} ({len(df)} rows, {len(df.columns)} columns)")


def main() -> None:
    for table_name, path in INDICATOR_FILES.items():
        ingest_indicator(table_name, path)


if __name__ == "__main__":
    main()
