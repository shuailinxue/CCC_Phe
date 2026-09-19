import argparse
import logging
from pathlib import Path
import sys
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from phenoniche.evaluation.dual_bank_benchmark import run_dual_bank_benchmark


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=root / "outputs/dual_bank")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    torch.set_num_threads(1)
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(message)s")
    run_dual_bank_benchmark(args.output, root, workers=args.workers, device=args.device)


if __name__ == "__main__":
    main()
