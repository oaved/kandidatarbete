import pandas as pd
import os
from pathlib import Path
from typing import List, Dict
import warnings
warnings.filterwarnings('ignore')

def create_excel_from_results():
    """
    Create Excel files for each results folder, including:
    - rss_variables and rss_scores
    - ff_dropped_companies (with ff_dropped_companies_by_metric)
    - GRS results (with grs_results)
    """
    # Define paths
    ROOT = Path(__file__).resolve().parent.parent
    results_dir = ROOT / "data" / "results"
    excel_dir = ROOT / "excel"

    # Create excel directory if it doesn't exist
    excel_dir.mkdir(exist_ok=True)

    # Get all subdirectories in results
    result_folders = [f for f in results_dir.iterdir() if f.is_dir()]

    print(f"Found {len(result_folders)} result folders: {[f.name for f in result_folders]}")
    
    # Track which folders are being processed
    folders_to_process = {
        'ff_dropped_companies': 'Dropped Companies Analysis',
        'GRS': 'GRS Test Results',
        'rss_scores': 'RSS Scores',
        'rss_tertiles': 'RSS Tertiles',
        'ff_regression': 'Fama-French Regression',
        'portfolio_returns': 'Portfolio Returns'
    }

    # Load rss_variables once
    rss_variables_path = results_dir / "rss_variables.csv"
    if rss_variables_path.exists():
        rss_variables_df = pd.read_csv(rss_variables_path)
        print("Loaded rss_variables.csv")
    else:
        rss_variables_df = None
        print("Warning: rss_variables.csv not found")

    def clean_sheet_name(name: str) -> str:
        # Remove common suffixes and truncate to Excel's 31-character limit
        base_name = name.replace("_portfolio_returns", "")
        base_name = base_name.replace("_rss", "")
        base_name = base_name.replace("_tertiles", "")
        base_name = base_name.replace("_results", "")
        base_name = base_name.replace("-", "_")
        base_name = base_name[:31]
        return base_name

    # Process each result folder
    for folder in result_folders:
        folder_name = folder.name
        description = folders_to_process.get(folder_name, folder_name)
        print(f"\nProcessing folder: {folder_name} ({description})")

        # Get all CSV files in this folder
        csv_files = list(folder.glob("*.csv"))

        if not csv_files:
            print(f"  No CSV files found in {folder_name}, skipping...")
            continue

        # Create Excel writer
        excel_path = excel_dir / f"{folder_name}_results.xlsx"
        print(f"  Creating Excel file: {excel_path}")

        with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
            # Add rss_variables sheet only to rss_scores workbook
            if folder_name == 'rss_scores' and rss_variables_df is not None:
                rss_variables_df.to_excel(writer, sheet_name='rss_variables', index=False)
                print("    Added rss_variables sheet")

            # Special handling for key result files
            key_files = {
                'ff_dropped_companies_by_metric.csv': 'Dropped Companies by Metric',
                'grs_results.csv': 'GRS Test Results'
            }
            
            # Track which sheets have been added
            added_sheets = set()
            
            # Add key files first (if they exist in this folder)
            for key_file_name, key_description in key_files.items():
                key_file_path = folder / key_file_name
                if key_file_path.exists():
                    try:
                        df = pd.read_csv(key_file_path)
                        sheet_name = clean_sheet_name(key_file_name.replace('.csv', ''))
                        df.to_excel(writer, sheet_name=sheet_name, index=False)
                        added_sheets.add(key_file_name)
                        print(f"    Added key sheet: {sheet_name} ({key_description})")
                    except Exception as e:
                        print(f"    Error reading {key_file_name}: {e}")

            # Add remaining sheets for this folder's CSV files
            for csv_file in csv_files:
                if csv_file.name not in added_sheets:
                    sheet_name = clean_sheet_name(csv_file.stem)

                    try:
                        df = pd.read_csv(csv_file)
                        df.to_excel(writer, sheet_name=sheet_name, index=False)
                        print(f"    Added sheet: {sheet_name}")
                    except Exception as e:
                        print(f"    Error reading {csv_file.name}: {e}")

        print(f"  Successfully created {excel_path}")

    print("\nDone! Excel files created in excel/ directory")

def save_GRS_to_excel():
    ROOT = Path(__file__).resolve().parent.parent
    GRS_DIR = ROOT / "data" / "results" / "GRS"
    excel_dir = ROOT / "excel"
    excel_dir.mkdir(exist_ok=True)

    grs_path = GRS_DIR / "grs_results.csv"
    if not grs_path.exists():
        print(f"No GRS results found at {grs_path}")
        return

    df = pd.read_csv(grs_path)
    excel_path = excel_dir / "grs_results.xlsx"

    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name='GRS Test Results', index=False)
        print(f"Added sheet: GRS Test Results")

    print(f"Successfully created {excel_path}")
    
if __name__ == "__main__":
    save_GRS_to_excel()