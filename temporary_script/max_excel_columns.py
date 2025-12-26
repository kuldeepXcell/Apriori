import os
import sys
from pathlib import Path

import pandas as pd

ROOT = Path("downloaded_data")

EXCEL_EXTS = {".xlsx", ".xlsm", ".xls", ".xlsb"}


def iter_workbooks(root: Path):
    for dirpath, _, filenames in os.walk(root):
        for name in filenames:
            path = Path(dirpath) / name
            if path.suffix.lower() in EXCEL_EXTS:
                yield path


def main() -> int:
    max_cols = 0
    max_info = None
    skipped = []

    for wb_path in iter_workbooks(ROOT):
        try:
            # Use pandas to list sheet names first
            xls = pd.ExcelFile(wb_path)
            for sheet_name in xls.sheet_names:
                try:
                    df = xls.parse(sheet_name=sheet_name, nrows=0)
                    cols = len(df.columns)
                    if cols > max_cols:
                        max_cols = cols
                        max_info = (wb_path, sheet_name)
                except Exception as exc:  # noqa: BLE001 - we want to continue scanning
                    skipped.append((wb_path, sheet_name, str(exc)))
        except Exception as exc:  # noqa: BLE001
            skipped.append((wb_path, None, str(exc)))

    print(f"Max columns: {max_cols}")
    if max_info:
        print(f"Workbook: {max_info[0]}")
        print(f"Sheet: {max_info[1]}")
    else:
        print("No sheets found.")

    if skipped:
        print("\nSkipped sheets/workbooks:")
        for wb_path, sheet_name, reason in skipped:
            if sheet_name:
                print(f"- {wb_path} :: {sheet_name} :: {reason}")
            else:
                print(f"- {wb_path} :: {reason}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
