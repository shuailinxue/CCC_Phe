#!/usr/bin/env python
from __future__ import annotations

import argparse
import json

from cccphe.merfish_st_niche import run_discovery, run_posthoc_tls_validation


def main() -> None:
    parser = argparse.ArgumentParser(description="GSE327192 single-cell MERFISH functional niches")
    parser.add_argument("--raw-root", required=True)
    parser.add_argument("--label-root", required=True)
    parser.add_argument("--stable-protective-ccc", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--overwrite-profiles", action="store_true")
    parser.add_argument("--discovery-only", action="store_true")
    args = parser.parse_args()
    discovery = run_discovery(
        args.raw_root, args.label_root, args.stable_protective_ccc, args.output,
        overwrite_profiles=args.overwrite_profiles,
    )
    print(json.dumps(discovery, indent=2))
    if not args.discovery_only:
        validation = run_posthoc_tls_validation(args.output)
        print(json.dumps(validation, indent=2))


if __name__ == "__main__":
    main()
