import argparse
import logging
from pathlib import Path
import sys
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from phenoniche.evaluation.structured_st_first_benchmark import run_structured_st_first


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=root / "outputs/structured_st_first")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    torch.set_num_threads(1)
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(message)s")
    run_structured_st_first(args.output, workers=args.workers)


if __name__ == "__main__":
    main()
