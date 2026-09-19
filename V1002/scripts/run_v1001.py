import argparse
from pathlib import Path
import sys
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from phenoniche.v1001.experiment import run_v1001_experiment


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=root / "outputs/v1001")
    args = parser.parse_args()
    torch.set_num_threads(1)
    run_v1001_experiment(args.output)


if __name__ == "__main__":
    main()
