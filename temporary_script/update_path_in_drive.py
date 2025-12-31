"""Update path_in_drive in indicator metadata JSON files under cleaned_excel_sheets."""

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


def update_path_in_drive(path: Path) -> bool:
    """Set path_in_drive to a sheet-relative folder path (from cleaned_excel_sheets)."""
    data = load_metadata(path)
    rel_path = path.parent.relative_to(CLEANED_SHEETS_DIR).as_posix()
    if data.get("path_in_drive") == rel_path:
        return False
    data["path_in_drive"] = rel_path
    path.write_text(json.dumps(data, indent=2, ensure_ascii=True), encoding="utf-8")
    return True


def main() -> None:
    if not CLEANED_SHEETS_DIR.exists():
        raise FileNotFoundError(f"Cleaned sheets folder not found at {CLEANED_SHEETS_DIR}")
    updated = 0
    for json_path in CLEANED_SHEETS_DIR.rglob("*.json"):
        # Only touch metadata files that match their folder name.
        if json_path.stem != json_path.parent.name:
            continue
        if update_path_in_drive(json_path):
            updated += 1
    print(f"Updated path_in_drive for {updated} metadata files.")


if __name__ == "__main__":
    main()
