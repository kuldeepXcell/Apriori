#!/usr/bin/env python3
"""
Script to analyze the Dataset_directory.xlsx file and provide statistics
about indicators in different sheets.
"""

import pandas as pd
from pathlib import Path

def analyze_sheet(excel_file: str, sheet_name: str) -> dict:
    """Analyze a specific sheet and return indicator statistics."""
    df = pd.read_excel(excel_file, sheet_name=sheet_name)

    # Total indicators (non-null values in Indicators column)
    total_indicators = df['Indicators'].notna().sum()

    # Indicators with definitions (both Indicators and Definition are non-null)
    indicators_with_definitions = df[['Indicators', 'Definition ']].notna().all(axis=1).sum()

    return {
        'sheet_name': sheet_name,
        'total_indicators': total_indicators,
        'indicators_with_definitions': indicators_with_definitions,
        'percentage_with_definitions': round(indicators_with_definitions / total_indicators * 100, 1) if total_indicators > 0 else 0
    }

def main():
    excel_file = Path(__file__).parent.parent / "Dataset_directory.xlsx"

    if not excel_file.exists():
        print(f"Error: {excel_file} not found!")
        return

    # Get all sheet names
    xl = pd.ExcelFile(excel_file)
    sheets = xl.sheet_names

    print("Dataset Directory Analysis")
    print("=" * 50)

    for sheet in sheets:
        try:
            stats = analyze_sheet(excel_file, sheet)
            print(f"\nSheet: {stats['sheet_name']}")
            print(f"  Total indicators: {stats['total_indicators']}")
            print(f"  Indicators with definitions: {stats['indicators_with_definitions']}")
            print(f"  Percentage with definitions: {stats['percentage_with_definitions']}%")
        except KeyError as e:
            print(f"\nSheet: {sheet}")
            print(f"  Error: Required columns not found - {e}")
        except Exception as e:
            print(f"\nSheet: {sheet}")
            print(f"  Error: {e}")

if __name__ == "__main__":
    main()
