"""Execute the current preview and, when present, its default dual-MLP result."""
from pathlib import Path
import sys
import nbformat as nbf
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from build_organized_simulation_notebook import build, NOTEBOOK


def rebuild():
    nb = build()
    NotebookClient(nb, timeout=900, kernel_name="cccphe",
                   resources={"metadata": {"path": str(ROOT)}}).execute()
    nbf.validate(nb)
    assert 1<=len(nb.cells)<=3 and all(cell.cell_type=="code" for cell in nb.cells)
    nbf.write(nb, NOTEBOOK)
    print(f"Complete: {len(nb.cells)} executed cells in {NOTEBOOK}")


if __name__ == "__main__":
    rebuild()
