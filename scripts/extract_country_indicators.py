#!/usr/bin/env python3
"""
Script to extract country indicators from Dataset_directory.xlsx
and save them to JSON format.
"""

import pandas as pd
import json
from pathlib import Path

def extract_country_indicators(excel_file: str, output_file: str = "country_indicators.json"):
    """Extract indicators from Country sheet and save to JSON."""

    # Read the Country sheet
    df = pd.read_excel(excel_file, sheet_name='Country')

    # Forward fill Subsection and SubSubSection to handle merged cells
    df['Subsection'] = df['Subsection'].ffill()
    df['SubSubSection'] = df['SubSubSection'].ffill()

    # Filter for rows where both Indicators and Definition are not null
    filtered_df = df[df[['Indicators', 'Definition ']].notna().all(axis=1)].copy()

    # Select the required columns
    result_df = filtered_df[[
        'Subsection',
        'SubSubSection',
        'Indicators',
        'Normalized Indicators name',
        'Definition ',
        'Use Case Question ',
        'Application (context to how the indicator will have to be analysed)'
    ]].copy()

    # Rename columns for cleaner JSON
    result_df.columns = [
        'subsection',
        'subsubsection',
        'indicator_name',
        'normalized_indicator_name',
        'definition',
        'question',
        'application_context'
    ]

    # Convert to list of dictionaries
    indicators_list = result_df.to_dict('records')

    # Create the final JSON structure
    json_data = {
        'total_indicators': len(indicators_list),
        'source_sheet': 'Country',
        'extraction_date': pd.Timestamp.now().isoformat(),
        'indicators': indicators_list
    }

    # Save to JSON file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)

    print(f"Extracted {len(indicators_list)} indicators to {output_file}")
    return json_data

def main():
    excel_file = Path(__file__).parent.parent / "Dataset_directory.xlsx"
    output_file = Path(__file__).parent.parent / "country_indicators.json"

    if not excel_file.exists():
        print(f"Error: {excel_file} not found!")
        return

    extract_country_indicators(str(excel_file), str(output_file))

if __name__ == "__main__":
    main()
