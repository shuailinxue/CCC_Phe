"""Fit spatial niches to a registered specimen or a collection of specimens."""

import argparse
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
        "--lr", type=Path, required=True, help="Species-appropriate ligand/receptor TSV"
    )
    parser.add_argument("--settings", type=Path)
    parser.add_argument("--composition-graph-cache", type=Path)
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
    if args.output.exists():
        raise FileExistsError(args.output)
    settings = (
        Settings(**json.loads(args.settings.read_text()))
        if args.settings
        else Settings()
    )
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
    )
    result.save(args.output)
    print(
        f"{len(result.cells)} cells, {result.cells.coarse.nunique()} coarse classes, "
        f"{result.cells.niche.nunique()} final niches; saved to {args.output}"
    )


if __name__ == "__main__":
    main()
