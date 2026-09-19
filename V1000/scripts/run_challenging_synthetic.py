import argparse
import logging
from pathlib import Path
import sys
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from phenoniche.data.challenging_synthetic import ChallengingConfig
from phenoniche.evaluation.challenging_benchmark import BenchmarkConfig, run_benchmark


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", choices=("capacity", "same_composition", "both"), default="both")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parents[1] / "outputs" / "challenging_synthetic_inferred")
    args = parser.parse_args()
    torch.set_num_threads(1)
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(message)s")
    scenarios = ("capacity", "same_composition") if args.scenario == "both" else (args.scenario,)
    for scenario in scenarios:
        result = run_benchmark(args.output / scenario, ChallengingConfig(scenario=scenario), BenchmarkConfig(device=args.device))
        print(scenario, result["comparison"], flush=True)


if __name__ == "__main__":
    main()
