"""Ingest cleaned Excel workbooks into Postgres tables."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable
import hashlib
import sys
import pandas as pd
from pandas.api.types import is_bool_dtype, is_datetime64_any_dtype, is_float_dtype, is_integer_dtype

# Add src directory to path for imports
sys.path.insert(0, "/home/ubuntu/Desktop/APriori/apriori/src")

from app.adapters.postgres import get_connection
from app.config.settings import settings
from app.core.logging import ModuleName, get_logger, setup_logging




# ---- Defaults (adjust in-file) ------------------------------------------------
CLEANED_SHEETS_DIR = Path("/home/ubuntu/Desktop/APriori/apriori/data/cleaned_excel_sheets")
SCHEMA = "public"
DRY_RUN = False
RECREATE_TABLES = False
BATCH_SIZE = 500
MAPPING_TABLE = "indicator_sheet_map"
FAIL_FAST = True
NULL_TOKENS = {"-", "—", "--", "n/a", "na", "null", "none", ""}
MAX_SHEET_NAME_LEN = 31
MAX_TABLE_NAME_LEN = 63
ALLOW_MAX_SHEET_NAME_LEN = True
ENFORCE_NAME_LENGTH_LIMITS = True
TEXT_ONLY_COLUMNS = {"data_note"}
TEXT_COLUMN_OVERRIDES: dict[tuple[str, str], set[str]] = {
    ("active_mobile_broadband_subscriptions_per_100_people", "active_mobile_broadband_subscri"): {"data_note"},
}
# Folder names (indicator directories) to skip.
# SKIP_INDICATORS = {
#     "foreign_funding_for_health_expense_by_country",
#     "circular_economy_score_material_footprint",
#     "informal_employment_pct_of_non_agricultural_employment_male_and_female",
#     "smartphone_penetration_rate_pct_population",
#     "business_confidence_index_score",
#     "consumer_confidence_index_score"
# }
SKIP_INDICATORS = {}

logger = get_logger(__name__)


# ---- Helpers -----------------------------------------------------------------
def normalize_identifier(name: str, max_len: int = 63) -> str:
    """Normalize a string for use as a SQL identifier."""
    cleaned = "".join(ch.lower() if ch.isalnum() else "_" for ch in name.strip())
    cleaned = "_".join(filter(None, cleaned.split("_")))
    if not cleaned:
        cleaned = "col"
    if len(cleaned) <= max_len:
        return cleaned
    digest = hashlib.sha1(cleaned.encode("utf-8")).hexdigest()[:8]
    prefix_len = max_len - 9
    return f"{cleaned[:prefix_len]}_{digest}"


def ensure_unique_columns(columns: Iterable[str]) -> list[str]:
    """Deduplicate column names by appending a suffix when needed."""
    seen: dict[str, int] = {}
    result: list[str] = []
    for name in columns:
        base = normalize_identifier(name or "column")
        count = seen.get(base, 0)
        if count:
            new_name = f"{base}_{count + 1}"
            seen[base] = count + 1
            result.append(new_name)
        else:
            seen[base] = 1
            result.append(base)
    return result


def detect_mixed_types(series: pd.Series) -> bool:
    """Return True if multiple non-null Python types are present."""
    types = {type(value) for value in series.dropna()}
    return len(types) > 1


def _text_override_set(indicator_name: str, sheet_name: str) -> set[str]:
    key = (normalize_identifier(indicator_name), normalize_identifier(sheet_name))
    return TEXT_COLUMN_OVERRIDES.get(key, set())


def validate_sheet(df: pd.DataFrame, indicator_name: str, sheet_name: str) -> None:
    """Fail fast on issues that would break ingestion semantics."""
    if ENFORCE_NAME_LENGTH_LIMITS:
        if len(sheet_name) > MAX_SHEET_NAME_LEN:
            raise ValueError(
                f"{indicator_name}/{sheet_name}: sheet name exceeds {MAX_SHEET_NAME_LEN} chars"
            )
        if len(sheet_name) == MAX_SHEET_NAME_LEN and not ALLOW_MAX_SHEET_NAME_LEN:
            raise ValueError(
                f"{indicator_name}/{sheet_name}: sheet name is at {MAX_SHEET_NAME_LEN} chars; likely truncated"
            )
    if df.empty:
        raise ValueError(f"{indicator_name}/{sheet_name}: sheet has no rows")
    if not df.columns.size:
        raise ValueError(f"{indicator_name}/{sheet_name}: sheet has no columns")
    if all(str(col).strip() == "" or str(col).lower().startswith("unnamed") for col in df.columns):
        raise ValueError(f"{indicator_name}/{sheet_name}: all columns are unnamed")
    if df.columns.duplicated().any():
        raise ValueError(f"{indicator_name}/{sheet_name}: duplicate column names detected before normalization")
    text_overrides = _text_override_set(indicator_name, sheet_name)
    for col in df.columns:
        norm_col = normalize_identifier(str(col))
        if norm_col in TEXT_ONLY_COLUMNS or norm_col in text_overrides:
            continue
        series = df[col]
        if detect_mixed_types(series):
            raise ValueError(f"{indicator_name}/{sheet_name}: mixed types in column '{col}'")


def normalize_null_tokens(df: pd.DataFrame) -> pd.DataFrame:
    """Replace common placeholder tokens with nulls."""
    def _clean(value: Any) -> Any:
        if value is None:
            return None
        if isinstance(value, str) and value.strip().lower() in NULL_TOKENS:
            return None
        return value

    return df.map(_clean)


def coerce_text_columns(df: pd.DataFrame, indicator_name: str, sheet_name: str) -> pd.DataFrame:
    """Coerce selected columns to string values (preserve nulls)."""
    text_overrides = _text_override_set(indicator_name, sheet_name)
    for col in df.columns:
        norm_col = normalize_identifier(str(col))
        if norm_col in TEXT_ONLY_COLUMNS or norm_col in text_overrides:
            df[col] = df[col].apply(lambda v: None if v is None else str(v))
    return df


def infer_sql_type(series: pd.Series) -> str:
    """Infer a Postgres type for a pandas Series."""
    if is_integer_dtype(series):
        return "bigint"
    if is_float_dtype(series):
        return "double precision"
    if is_bool_dtype(series):
        return "boolean"
    if is_datetime64_any_dtype(series):
        return "timestamp"
    return "text"


def get_example_row(df: pd.DataFrame) -> dict[str, Any]:
    """Return one example row with non-null values when available."""
    if df.empty:
        return {}
    for _, row in df.iterrows():
        if row.notna().any():
            return row.where(pd.notna(row), None).to_dict()
    return df.iloc[0].where(pd.notna(df.iloc[0]), None).to_dict()


def iter_indicator_dirs(root: Path) -> list[Path]:
    """Collect indicator directories based on matching metadata file names."""
    indicator_dirs: list[Path] = []
    for json_path in root.rglob("*.json"):
        if json_path.stem != json_path.parent.name:
            continue
        indicator_dirs.append(json_path.parent)
    return indicator_dirs


def build_table_name(
    base: str,
    sheet_name: str | None,
    multi_sheet: bool,
    used: set[str],
    *,
    indicator_name: str,
) -> str:
    """Generate a table name (sheet-only for multi-sheet workbooks)."""
    if multi_sheet and sheet_name:
        table_name = normalize_identifier(sheet_name)
    else:
        table_name = normalize_identifier(base)
    if table_name in used:
        raise ValueError(
            f"Table name collision detected: {table_name} (indicator={indicator_name}, sheet={sheet_name})"
        )
    used.add(table_name)
    return table_name


def print_dry_run(
    indicator_name: str,
    sheet_names: list[str],
    table_info: list[tuple[str, list[tuple[str, str]], dict[str, Any]]],
) -> None:
    """Print a dry-run preview for one indicator workbook."""
    print(f"\nIndicator: {indicator_name}")
    if len(sheet_names) > 1:
        print("Sheets:")
        for name in sheet_names:
            print(f"  - {name}")
    for table_name, columns, example in table_info:
        print(f"\nTable: {table_name}")
        print("Columns:")
        for col_name, col_type in columns:
            print(f"  - {col_name}: {col_type}")
        print("Example row:")
        print(example)


def create_table(cursor, table_name: str, columns: list[tuple[str, str]]) -> None:
    """Create the target table, optionally recreating it."""
    column_sql = ", ".join(f"\"{name}\" {dtype}" for name, dtype in columns)
    full_name = f"\"{SCHEMA}\".\"{table_name}\""
    if RECREATE_TABLES:
        cursor.execute(f"DROP TABLE IF EXISTS {full_name}")
    cursor.execute(f"CREATE TABLE IF NOT EXISTS {full_name} ({column_sql})")


def ensure_mapping_table(cursor) -> None:
    """Create the indicator-to-sheet mapping table if missing."""
    full_name = f"\"{SCHEMA}\".\"{MAPPING_TABLE}\""
    cursor.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {full_name} (
            indicator_name TEXT NOT NULL,
            sheet_name TEXT NOT NULL,
            table_name TEXT NOT NULL
        )
        """
    )


def insert_mapping_rows(cursor, rows: list[tuple[str, str, str]]) -> None:
    """Insert mapping rows for multi-sheet indicators."""
    if not rows:
        return
    full_name = f"\"{SCHEMA}\".\"{MAPPING_TABLE}\""
    cursor.executemany(
        f"INSERT INTO {full_name} (indicator_name, sheet_name, table_name) VALUES (%s, %s, %s)",
        rows,
    )


def insert_rows(cursor, table_name: str, df: pd.DataFrame) -> int:
    """Insert rows into the target table in batches."""
    if df.empty:
        return 0
    full_name = f"\"{SCHEMA}\".\"{table_name}\""
    columns = [f"\"{col}\"" for col in df.columns]
    placeholders = ", ".join(["%s"] * len(df.columns))
    sql = f"INSERT INTO {full_name} ({', '.join(columns)}) VALUES ({placeholders})"

    df = df.where(pd.notna(df), None)
    rows = [tuple(row) for row in df.itertuples(index=False, name=None)]

    for start in range(0, len(rows), BATCH_SIZE):
        batch = rows[start : start + BATCH_SIZE]
        cursor.executemany(sql, batch)
    return len(rows)


# ---- Main ingestion ----------------------------------------------------------
def ingest() -> None:
    setup_logging(settings.log_level)
    if not CLEANED_SHEETS_DIR.exists():
        raise FileNotFoundError(f"Cleaned sheets folder not found at {CLEANED_SHEETS_DIR}")

    indicator_dirs = iter_indicator_dirs(CLEANED_SHEETS_DIR)
    if not indicator_dirs:
        raise FileNotFoundError("No indicator metadata files found under cleaned_excel_sheets")

    errors: list[str] = []
    used_tables: set[str] = set()
    for indicator_dir in sorted(indicator_dirs):
        indicator_name = indicator_dir.name
        if indicator_name in SKIP_INDICATORS:
            logger.info(
                "Skipping indicator %s (in skip list)",
                indicator_name,
                extra={"module_name": ModuleName.INGESTION},
            )
            continue
        if ENFORCE_NAME_LENGTH_LIMITS and len(indicator_name) > MAX_TABLE_NAME_LEN:
            raise ValueError(
                f"{indicator_name}: indicator name exceeds {MAX_TABLE_NAME_LEN} chars"
            )
        workbook_path = indicator_dir / f"{indicator_name}.xlsx"
        if not workbook_path.exists():
            raise FileNotFoundError(f"Workbook not found at {workbook_path}")

        excel = pd.ExcelFile(workbook_path, engine="openpyxl")
        sheet_names = excel.sheet_names
        multi_sheet = len(sheet_names) > 1
        table_info: list[tuple[str, list[tuple[str, str]], dict[str, Any]]] = []
        sheet_tables: list[tuple[str, pd.DataFrame]] = []
        mapping_rows: list[tuple[str, str, str]] = []

        for sheet_name in sheet_names:
            df = excel.parse(sheet_name)
            df = normalize_null_tokens(df)
            df = coerce_text_columns(df, indicator_name, sheet_name)
            try:
                validate_sheet(df, indicator_name, sheet_name)
            except ValueError as exc:
                if DRY_RUN and not FAIL_FAST:
                    errors.append(str(exc))
                    continue
                raise
            df.columns = ensure_unique_columns([str(col) if col is not None else "column" for col in df.columns])
            columns = [(col, infer_sql_type(df[col])) for col in df.columns]
            table_name = build_table_name(
                indicator_name,
                sheet_name,
                multi_sheet,
                used_tables,
                indicator_name=indicator_name,
            )
            example = get_example_row(df)
            table_info.append((table_name, columns, example))
            sheet_tables.append((table_name, df))
            mapping_rows.append((indicator_name, sheet_name, table_name))

        if DRY_RUN:
            print_dry_run(indicator_name, sheet_names, table_info)
            continue

        with get_connection(autocommit=False) as conn:
            with conn.cursor() as cursor:
                if SCHEMA != "public":
                    cursor.execute(f"CREATE SCHEMA IF NOT EXISTS \"{SCHEMA}\"")
                ensure_mapping_table(cursor)
                for table_name, df in sheet_tables:
                    columns = [(col, infer_sql_type(df[col])) for col in df.columns]
                    create_table(cursor, table_name, columns)
                    inserted = insert_rows(cursor, table_name, df)
                    logger.info(
                        "Inserted %s rows into %s",
                        inserted,
                        table_name,
                        extra={"module_name": ModuleName.INGESTION},
                    )
                insert_mapping_rows(cursor, mapping_rows)
            conn.commit()

    if DRY_RUN:
        if errors:
            print("\nDry-run errors:")
            for message in errors:
                print(f"  - {message}")
        logger.info("Dry run complete", extra={"module_name": ModuleName.INGESTION})


def main() -> None:
    ingest()


if __name__ == "__main__":
    main()
