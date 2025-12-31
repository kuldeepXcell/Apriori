"""Report indicator sheets that fail Postgres ingestion validations (dry-run only)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable
import hashlib
import json
import sys

import pandas as pd
from pandas.api.types import is_bool_dtype, is_datetime64_any_dtype, is_float_dtype, is_integer_dtype

# Add src directory to path for imports
sys.path.insert(0, "/home/ubuntu/Desktop/APriori/apriori/src")

from app.config.settings import settings
from app.core.logging import ModuleName, get_logger, setup_logging

# ---- Defaults (adjust in-file) ------------------------------------------------
ROOT_DIR = Path("/home/ubuntu/Desktop/APriori/apriori")
CLEANED_SHEETS_DIR = ROOT_DIR / "data" / "cleaned_excel_sheets"
REPORT_PATH = ROOT_DIR / "data" / "processed" / "postgres_ingest_failures.json"
MAX_SHEET_NAME_LEN = 31
MAX_TABLE_NAME_LEN = 63
NULL_TOKENS = {"-", "—", "--", "n/a", "na", "null", "none", ""}
TEXT_ONLY_COLUMNS = {"data_note"}
TEXT_COLUMN_OVERRIDES: dict[tuple[str, str], set[str]] = {
    ("active_mobile_broadband_subscriptions_per_100_people", "active_mobile_broadband_subscri"): {"data_note"},
}

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


def detect_mixed_types(series: pd.Series) -> bool:
    """Return True if multiple non-null Python types are present."""
    types = {type(value) for value in series.dropna()}
    return len(types) > 1


def _text_override_set(indicator_name: str, sheet_name: str) -> set[str]:
    key = (normalize_identifier(indicator_name), normalize_identifier(sheet_name))
    return TEXT_COLUMN_OVERRIDES.get(key, set())


def validate_sheet(df: pd.DataFrame, indicator_name: str, sheet_name: str) -> None:
    """Fail fast on issues that would break ingestion semantics."""
    if len(sheet_name) > MAX_SHEET_NAME_LEN:
        raise ValueError(
            f"{indicator_name}/{sheet_name}: sheet name exceeds {MAX_SHEET_NAME_LEN} chars"
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


def iter_indicator_dirs(root: Path) -> list[Path]:
    """Collect indicator directories based on matching metadata file names."""
    indicator_dirs: list[Path] = []
    for json_path in root.rglob("*.json"):
        if json_path.stem != json_path.parent.name:
            continue
        indicator_dirs.append(json_path.parent)
    return indicator_dirs


def build_table_name(base: str, sheet_name: str | None, multi_sheet: bool, used: set[str], indicator_name: str) -> str:
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


# ---- Main report -------------------------------------------------------------
def report() -> None:
    setup_logging(settings.log_level)
    if not CLEANED_SHEETS_DIR.exists():
        raise FileNotFoundError(f"Cleaned sheets folder not found at {CLEANED_SHEETS_DIR}")

    indicator_dirs = iter_indicator_dirs(CLEANED_SHEETS_DIR)
    if not indicator_dirs:
        raise FileNotFoundError("No indicator metadata files found under cleaned_excel_sheets")

    errors: list[dict[str, str]] = []
    used_tables: set[str] = set()

    for indicator_dir in sorted(indicator_dirs):
        indicator_name = indicator_dir.name
        domain = indicator_dir.relative_to(CLEANED_SHEETS_DIR).parts[0]
        if len(indicator_name) > MAX_TABLE_NAME_LEN:
            errors.append(
                {
                    "domain": domain,
                    "indicator": indicator_name,
                    "sheet": "(indicator)",
                    "error": f"indicator name exceeds {MAX_TABLE_NAME_LEN} chars",
                }
            )
            continue
        workbook_path = indicator_dir / f"{indicator_name}.xlsx"
        if not workbook_path.exists():
            errors.append(
                {
                    "domain": domain,
                    "indicator": indicator_name,
                    "sheet": "(workbook)",
                    "error": f"workbook not found at {workbook_path}",
                }
            )
            continue

        excel = pd.ExcelFile(workbook_path, engine="openpyxl")
        sheet_names = excel.sheet_names
        multi_sheet = len(sheet_names) > 1

        for sheet_name in sheet_names:
            try:
                df = excel.parse(sheet_name)
                df = normalize_null_tokens(df)
                df = coerce_text_columns(df, indicator_name, sheet_name)
                validate_sheet(df, indicator_name, sheet_name)
                build_table_name(indicator_name, sheet_name, multi_sheet, used_tables, indicator_name)
            except ValueError as exc:
                errors.append(
                    {
                        "domain": domain,
                        "indicator": indicator_name,
                        "sheet": sheet_name,
                        "error": str(exc),
                    }
                )

    by_domain: dict[str, list[dict[str, str]]] = {}
    for entry in errors:
        by_domain.setdefault(entry["domain"].lower(), []).append(entry)

    report = {"by_domain": by_domain, "total": len(errors)}
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2, ensure_ascii=True), encoding="utf-8")
    print(f"Report written to: {REPORT_PATH}")


if __name__ == "__main__":
    report()
