import argparse
import logging
from pathlib import Path
import sys
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from phenoniche.evaluation.inferred_benchmark import run_inferred_benchmark


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=root / "outputs/inferred_wb")
    parser.add_argument("--previous-output", type=Path, default=root / "outputs/challenging_synthetic")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    torch.set_num_threads(1)
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(message)s")
    run_inferred_benchmark(args.output, args.previous_output, args.workers, args.device)


if __name__ == "__main__":
    main()
