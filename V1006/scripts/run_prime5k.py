#!/usr/bin/env python
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from phenoniche.v1006.pipeline import ensure_results

if __name__ == "__main__":
    result = ensure_results()
    print(result["summary"])
