import shutil
from pathlib import Path
from fetch_excel_sheet_names import fetch_excel_sheet_names

def remove_LSEG_data():
    """
    Removes the entire LSEG_data folder and all its contents.
    Riksbanken and other CSVs in data/ are untouched.
    """
    lseg_dir = Path(__file__).parent.parent / "data" / "LSEG_data"

    if lseg_dir.exists():
        shutil.rmtree(lseg_dir)
        print("Removed LSEG_data/ folder and all its contents.")
    else:
        print("No LSEG_data/ folder found, nothing to remove.")

if __name__ == "__main__":
    remove_LSEG_data()