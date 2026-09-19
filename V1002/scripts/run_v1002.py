"""Run the current two-view V1002 simulation benchmark."""
from pathlib import Path
import os
import sys

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root / "src"))

from phenoniche.v1002.final_experiment import run


if __name__ == "__main__":
    os.chdir(root)
    run()
