"""Fit or reuse the closed-window, full-slice PhenoMap V1."""
import argparse
import json
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from phenoniche.v1.network_workflow import fit_run
from phenoniche.v1.polish import Options
from phenoniche.v1.polish_fit import fit_full_slice
from phenoniche.v1.probabilistic import Config
from phenoniche.v1.storage import Network


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--network", required=True, type=Path)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device", default="cuda:2")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--reuse-only", action="store_true", help="Require a compatible completed fit")
    mode.add_argument("--sampled", action="store_true", help="Run the historical sampled benchmark")
    parser.add_argument("--real-target", type=Path)
    parser.add_argument("--legacy", type=Path)
    parser.add_argument("--joint-only", action="store_true")
    args = parser.parse_args()
    overrides = {"seed": args.seed} if args.seed is not None else {}
    if args.sampled:
        settings = json.loads(args.config.read_text()) if args.config else {}
        cfg = Config(**(settings | overrides))
        data = Network(args.network)
        real = Network(args.real_target) if args.real_target else data
        result = fit_run(data, real, args.root, cfg, args.device, not args.joint_only, args.legacy)
    else:
        if args.real_target or args.legacy or args.joint_only:
            parser.error("--real-target, --legacy and --joint-only require --sampled")
        config = args.config or PROJECT / "configs/network_v1_full.json"
        settings = json.loads(config.read_text())
        cfg = Config(**(settings["config"] | overrides))
        result = fit_full_slice(args.network, args.root, cfg, Options(**settings["options"]),
                                settings["temperature"], args.device, reuse_only=args.reuse_only)
    print(result, flush=True)


if __name__ == "__main__":
    main()
