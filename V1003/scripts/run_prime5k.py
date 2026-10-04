#!/usr/bin/env python3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from phenoniche.v1003.pipeline import ensure_results


if __name__ == "__main__":
    summary = ensure_results()
    print(f"complete: n={summary['n_cells']:,}, F={summary['retained_ccc_features']}, K={summary['K']}")

