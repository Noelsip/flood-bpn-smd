"""
Synchronize the project notebooks with the current Isolation Forest workflow.

The canonical implementation is kept in Flood_AutoML.ipynb. This helper copies
that notebook to Flood_AutoML_Tabular.ipynb so the old supervised AutoML notebook
cannot be regenerated accidentally.
"""
from pathlib import Path
import shutil


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "Flood_AutoML.ipynb"
TARGET = ROOT / "Flood_AutoML_Tabular.ipynb"


def main() -> None:
    if not SOURCE.exists():
        raise FileNotFoundError(f"Notebook sumber tidak ditemukan: {SOURCE}")
    shutil.copyfile(SOURCE, TARGET)
    print(f"Synced {TARGET.name} from {SOURCE.name}")


if __name__ == "__main__":
    main()
