"""Section diagnostics and optional conservative expression calibration for the lung study."""

import numpy as np
import pandas as pd
from scipy import sparse


def audit_sections(adata, xyz, tile_um=250, min_cells=25):
    counts = adata.layers["counts"].tocsr()
    totals = np.asarray(counts.sum(axis=1)).ravel()
    detected = np.diff(counts.indptr)
    z = xyz[:, 2]
    types = adata.obs["celltypes"].astype(str).to_numpy()
    obs = pd.DataFrame(dict(z=z, cell_type=types, transcripts=totals, genes=detected))
    qc = obs.groupby("z").agg(
        n=("genes", "size"),
        median_transcripts=("transcripts", "median"),
        median_genes=("genes", "median"),
    )
    per_type = obs.groupby(["z", "cell_type"]).agg(
        n=("genes", "size"),
        median_transcripts=("transcripts", "median"),
        median_genes=("genes", "median"),
    )
    cp10k = counts.multiply((10000 / np.maximum(totals, 1))[:, None]).tocsr()
    tiles = np.floor(xyz[:, :2] / tile_um).astype(int)
    keys = pd.MultiIndex.from_arrays([z, types, tiles[:, 0], tiles[:, 1]])
    codes, groups = pd.factorize(keys, sort=True)
    sizes = np.bincount(codes)
    design = sparse.csr_matrix(
        (1.0 / sizes[codes], (codes, np.arange(len(codes)))),
        shape=(len(sizes), len(codes)),
    )
    profiles = (design @ cp10k).toarray()
    index = {key: i for i, key in enumerate(groups) if sizes[i] >= min_cells}
    offsets = [np.zeros(adata.n_vars)]
    diagnostics = []
    levels = np.unique(z)
    for low, high in zip(levels[:-1], levels[1:]):
        pairs = [
            (i, index[(high, cell_type, x, y)], cell_type, (x + y) % 2)
            for (layer, cell_type, x, y), i in index.items()
            if layer == low and (high, cell_type, x, y) in index
        ]
        left, right, cell_type, fold = map(np.asarray, zip(*pairs, strict=True))
        raw = np.log2(profiles[right] + 1) - np.log2(profiles[left] + 1)
        expressed = (profiles[right] + profiles[left]) / 2 >= 1

        def estimate(selected):
            classes = np.unique(cell_type[selected])
            values = [
                np.median(raw[selected & (cell_type == c)], axis=0)
                for c in classes
                if (selected & (cell_type == c)).sum() >= 4
            ]
            if len(values) < 4:
                return np.zeros(raw.shape[1])
            values = np.array(values)
            med = np.median(values, axis=0)
            agreement = (np.sign(values) == np.sign(med)).mean(axis=0)
            return np.where(
                (agreement >= 0.75) & (abs(med) >= 0.15),
                np.clip(med * 0.5, -0.35, 0.35),
                0,
            )

        heldout_raw, heldout_adjusted = [], []
        for f in (0, 1):
            shift = estimate(fold != f)
            use = (fold == f)[:, None] & expressed
            heldout_raw.extend(abs(raw[use]))
            heldout_adjusted.extend(abs((raw - shift)[use]))
        shift = estimate(np.ones(len(pairs), dtype=bool))
        before, after = np.mean(heldout_raw), np.mean(heldout_adjusted)
        diagnostics.append(
            (
                low,
                high,
                len(pairs),
                len(np.unique(cell_type)),
                int(np.count_nonzero(shift)),
                before,
                after,
            )
        )
        offsets.append(offsets[-1] + shift)
    metrics = pd.DataFrame(
        diagnostics,
        columns=[
            "z_from",
            "z_to",
            "matched_tiles_types",
            "cell_types",
            "genes_flagged",
            "heldout_raw_MAE",
            "heldout_adjusted_MAE",
        ],
    )
    offsets = np.asarray(offsets)
    offsets = np.clip(offsets - np.median(offsets, axis=0), -0.5, 0.5)
    return (
        qc,
        per_type,
        metrics,
        pd.DataFrame(offsets, index=levels, columns=adata.var_names),
    )


def adjust_expression(counts, z, offsets):
    """Adjust relative expression before CP10k with bounded gene-specific factors."""
    corrected = counts.tocsr().astype(np.float32, copy=True)
    layer = np.repeat(offsets.index.get_indexer(z), np.diff(corrected.indptr))
    if np.any(layer < 0):
        raise ValueError("Section absent from expression calibration")
    corrected.data *= np.exp2(-offsets.to_numpy()[layer, corrected.indices]).astype(
        np.float32
    )
    return corrected
