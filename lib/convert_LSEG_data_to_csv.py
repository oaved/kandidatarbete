import pandas as pd
from pathlib import Path
from fetch_excel_sheet_names import fetch_excel_sheet_names

def convert_excel_sheets_to_csv(file_path):
    """
    Converts all sheets in an Excel file to CSV format with the following rules:
    - Sheets 1-2 (IndustryClassification, Companies): saved to LSEG_data/metadata/, fetched in full
    - Sheets 3-4 (TotalReturn, MarketCap): saved to LSEG_data/metrics/, fetched in full
    - Sheets 5+: saved to LSEG_data/metrics/, fetched only up to column L
    """
    sheet_names = fetch_excel_sheet_names(file_path)
    if not sheet_names:
        print("No sheets found in the Excel file.")
        return

    separate_sheets = sheet_names[:2]   # IndustryClassification, Companies
    full_sheets     = sheet_names[2:4]  # TotalReturn, MarketCap
    trimmed_sheets  = sheet_names[4:]   # Everything else → columns A:L only

    lseg_dir    = Path(__file__).parent.parent / "data" / "LSEG_data"
    metadata_dir = lseg_dir / "metadata"
    metrics_dir  = lseg_dir / "metrics"

    metadata_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    # --- Sheets 1 & 2: metadata ---
    for sheet in separate_sheets:
        try:
            df = pd.read_excel(file_path, sheet_name=sheet)
            df.to_csv(metadata_dir / f"{sheet}.csv", index=False)
            print(f"[metadata] Converted '{sheet}' in full.")
        except Exception as e:
            print(f"Error processing '{sheet}': {e}")

    # --- Sheets 3 & 4: full data ---
    for sheet in full_sheets:
        try:
            df = pd.read_excel(file_path, sheet_name=sheet)
            
            if df.empty:
                print(f"[skipped] '{sheet}' is empty.")
                continue
            
            df.to_csv(metrics_dir / f"{sheet}.csv", index=False)
            print(f"[metrics] Converted '{sheet}' in full.")
        except Exception as e:
            print(f"Error processing '{sheet}': {e}")

    # --- Sheets 5+: columns A to L only ---
    for sheet in trimmed_sheets:
        try:
            df = pd.read_excel(file_path, sheet_name=sheet, usecols="A:L")
            
            if df.empty:
                print(f"[skipped] '{sheet}' is empty.")
                continue

            df.to_csv(metrics_dir / f"{sheet}.csv", index=False)
            print(f"[metrics] Converted '{sheet}' (columns A:L only).")
        except Exception as e:
            print(f"Error processing '{sheet}': {e}")

if __name__ == "__main__":
    file_path = Path(__file__).parent.parent /"excel" / "LSEG_data_values.xlsx"
    convert_excel_sheets_to_csv(file_path)