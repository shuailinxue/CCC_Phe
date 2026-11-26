"""Fit spatial niches to a registered specimen or a collection of specimens."""

import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys
import anndata
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from phenoniche.spatial import Settings, SpatialNiches
from phenoniche.spatial.input import lung_coordinates, table_coordinates


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--lr", type=Path, help="Species-appropriate ligand/receptor TSV"
    )
    parser.add_argument("--settings", type=Path)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--enrichment-only", action="store_true")
    mode.add_argument("--interface", type=int, nargs=2, metavar=("CLASS_A", "CLASS_B"))
    parser.add_argument("--interface-um", type=float)
    parser.add_argument("--delaunay-max-um", type=float)
    parser.add_argument("--enrichment-permutations", type=int)
    parser.add_argument("--representation", choices=("A", "H1", "H2", "H3"))
    parser.add_argument("--expression-preprocessing", choices=("commuspace", "cp10k"))
    parser.add_argument("--expression-quantile", type=float)
    parser.add_argument("--tensor-log1p", action="store_true", default=None)
    parser.add_argument("--rank-mode", choices=("manual", "auto"))
    parser.add_argument("--rank", type=int)
    parser.add_argument("--rank-candidates", type=int, nargs="+")
    parser.add_argument("--composition-graph-cache", type=Path)
    parser.add_argument("--cp-device", help="auto, cpu, or cuda[:index]")
    parser.add_argument("--cp-batch-size", type=int, help="Maximum cells per computational batch")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--checkpoint-dir", type=Path)
    parser.add_argument("--types", default="celltypes")
    parser.add_argument(
        "--layer", default="counts", help="Raw counts layer; use X for raw .X"
    )
    specimens = parser.add_mutually_exclusive_group(required=True)
    specimens.add_argument("--specimen-key")
    specimens.add_argument("--single-specimen", action="store_true")
    coordinates = parser.add_mutually_exclusive_group(required=True)
    coordinates.add_argument("--coordinates-tsv", type=Path)
    coordinates.add_argument(
        "--obsm-um", help="Registered obsm key already in micrometers"
    )
    coordinates.add_argument(
        "--lung-stim-dir", type=Path, help="Directory containing section_*_aligned.csv"
    )
    args = parser.parse_args()
    if not args.enrichment_only and args.lr is None:
        parser.error("--lr is required for coarse-class and interface fits")
    if args.output.exists():
        raise FileExistsError(args.output)
    settings = (
        Settings(**json.loads(args.settings.read_text()))
        if args.settings
        else Settings()
    )
    overrides = {
        name: getattr(args, name) for name in (
            "interface_um", "delaunay_max_um", "enrichment_permutations", "representation",
            "expression_preprocessing", "expression_quantile", "tensor_log1p", "rank_mode",
            "rank", "rank_candidates", "cp_device", "cp_batch_size"
        ) if getattr(args, name) is not None
    }
    settings = replace(settings, **overrides)
    data = anndata.read_h5ad(args.input)
    if args.coordinates_tsv:
        xyz = table_coordinates(data, args.coordinates_tsv)
    elif args.lung_stim_dir:
        xyz = lung_coordinates(data, args.lung_stim_dir)
    else:
        xyz = data.obsm[args.obsm_um]
    sample = (
        data.obs[args.specimen_key]
        if args.specimen_key
        else np.repeat("specimen", data.n_obs)
    )
    if args.enrichment_only:
        preview = SpatialNiches.coarse(
            data.obs_names, xyz, data.obs[args.types], sample, settings,
            composition_graph_cache=args.composition_graph_cache,
        )
        preview.save(args.output)
        print(f"{preview.cells.coarse.nunique()} coarse classes; saved to {args.output}")
        return
    counts = data.X if args.layer == "X" else data.layers[args.layer]
    result = SpatialNiches.fit(
        counts,
        data.var_names,
        data.obs_names,
        xyz,
        data.obs[args.types],
        sample,
        pd.read_csv(args.lr, sep="\t"),
        settings,
        composition_graph_cache=args.composition_graph_cache,
        workers=args.workers,
        checkpoint_dir=args.checkpoint_dir,
        interface=args.interface,
    )
    result.save(args.output)
    print(
        f"{len(result.cells)} cells, {result.cells.coarse.nunique()} coarse classes, "
        f"{result.cells.niche.nunique()} final niches; saved to {args.output}"
    )


if __name__ == "__main__":
    main()
