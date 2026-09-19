#!/usr/bin/env python
from __future__ import annotations

import argparse

from cccphe.config import ensure_storage, load_config
from cccphe.network_cox import run_network_sparse_cox


def main() -> None:
    parser = argparse.ArgumentParser(description="Find phenotype-associated exact CCCs")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config)
    ensure_storage(cfg)
    run_network_sparse_cox(cfg, args.smoke)


if __name__ == "__main__":
    main()
