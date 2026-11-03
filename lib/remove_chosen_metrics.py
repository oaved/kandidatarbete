import shutil

from anyio import Path
from pathlib import Path

def remove_chosen_metrics():
    """
    Removes the chosen_metrics folder and all its contents.
    """
    chosen_dir = Path(__file__).parent.parent / "data" / "LSEG_data" / "chosen_metrics"

    if chosen_dir.exists():
        shutil.rmtree(chosen_dir)
        print("Removed chosen_metrics/ folder and all its contents.")
    else:
        print("No chosen_metrics/ folder found, nothing to remove.")
        
if __name__ == "__main__":
    remove_chosen_metrics()