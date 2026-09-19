#!/usr/bin/env python
from __future__ import annotations

import argparse

from cccphe.merfish_anchor import build_merfish_anchor


def main() -> None:
    parser = argparse.ArgumentParser(description="Build TLS-blind GSE327192 MERFISH CCC anchor")
    parser.add_argument("--raw-root", required=True)
    parser.add_argument("--panel-file", required=True)
    parser.add_argument("--reference-h5ad", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    print(
        build_merfish_anchor(
            args.raw_root,
            args.panel_file,
            args.reference_h5ad,
            args.output,
            overwrite=args.overwrite,
        )
    )


if __name__ == "__main__":
    main()
