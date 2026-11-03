import pandas as pd
from pathlib import Path

def fetch_excel_sheet_names(file_path):
    """
    Fetches the sheet names from an Excel file.

    Parameters:
    file_path (str): The path to the Excel file.

    Returns:
    list: A list of sheet names in the Excel file.
    """
    try:
        # Load the Excel file
        excel_file = pd.ExcelFile(file_path)
        
        # Get the sheet names
        sheet_names = excel_file.sheet_names
        
        return sheet_names
    except Exception as e:
        print(f"An error occurred while fetching sheet names: {e}")
        return []

if __name__ == "__main__":
    # Example usage
    excel_file_path = Path(__file__).parent.parent / "excel" / "LSEG_data_values.xlsx"
    sheet_names = fetch_excel_sheet_names(excel_file_path)
    print(f"Sheet names in the Excel file: {sheet_names}")
    print(f"Total number of sheets: {len(sheet_names)}")