from pathlib import Path
import json
import math
import sys
import numpy as np
import pandas as pd

from .config import PROJECT_ROOT, REPOSITORY_ROOT


def _import_v1002():
    """Expose the retained V1002 package without copying its CCC implementation."""
    source = REPOSITORY_ROOT / "V1002/src"
    package = source / "phenoniche"
    if not source.is_dir():
        raise FileNotFoundError(f"V1002 source directory is missing: {source}")
    if str(source) not in sys.path:
        sys.path.append(str(source))
    import phenoniche
    if str(package) not in phenoniche.__path__:
        phenoniche.__path__.append(str(package))
    from phenoniche.v1001.neighborhoods import build_neighborhoods
    from phenoniche.v1002.lr_atlas import LRAtlas, load_lr_atlas
    from phenoniche.v1002.simulation_cells import _side_expression, aggregate_pairwise_ccc
    return build_neighborhoods, LRAtlas, load_lr_atlas, _side_expression, aggregate_pairwise_ccc


def resolve_device(requested):
    import torch
    if requested == "auto":
        return "cuda:0" if torch.cuda.is_available() else "cpu"
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA was requested but is unavailable: {requested}")
    return requested


def _manifest_signature(config):
    h5ad = Path(config.processed_h5ad)
    atlas = Path(config.lr_atlas)
    if not h5ad.is_file():
        raise FileNotFoundError(f"Processed Prime 5K AnnData is missing: {h5ad}")
    if not atlas.is_file():
        raise FileNotFoundError(f"V1002 LR atlas is missing: {atlas}")
    return {
        "processed_h5ad": str(h5ad.resolve()),
        "processed_h5ad_bytes": h5ad.stat().st_size,
        "processed_h5ad_mtime_ns": h5ad.stat().st_mtime_ns,
        "lr_atlas": str(atlas.resolve()),
        "lr_atlas_bytes": atlas.stat().st_size,
        "n_neighbors": config.n_neighbors,
        "sigma_um": config.sigma_um,
        "expression_coverage_fraction": config.expression_coverage_fraction,
        "opportunity_support_fraction": config.opportunity_support_fraction,
        "opportunity_support_minimum": config.opportunity_support_minimum,
        "feature_order": "sender_type,receiver_type,measurable_lr",
        "ccc_definition": "V1002 physical directed pairwise opportunity-normalized CCC",
    }


def cache_paths(config):
    root = Path(config.output_dir)
    cache = root / "cache"
    return {
        "root": root,
        "cache": cache,
        "ccc": cache / "ccc.npy",
        "magnitude": cache / "magnitude.npy",
        "coordinates": cache / "coordinates.npy",
        "cells": cache / "cells.csv.gz",
        "features": cache / "features.csv",
        "manifest": cache / "input_manifest.json",
    }


def _cache_is_valid(paths, signature):
    required = [paths[key] for key in ("ccc", "magnitude", "coordinates", "cells", "features", "manifest")]
    if not all(path.is_file() for path in required):
        return False
    try:
        manifest = json.loads(paths["manifest"].read_text())
        matrix = np.load(paths["ccc"], mmap_mode="r")
        magnitude = np.load(paths["magnitude"], mmap_mode="r")
        coordinates = np.load(paths["coordinates"], mmap_mode="r")
        return (manifest.get("signature") == signature
                and matrix.shape == (manifest["n_cells"], manifest["n_features"])
                and magnitude.shape == (manifest["n_cells"],)
                and coordinates.shape == (manifest["n_cells"], 2))
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        return False


def prepare_prime5k_ccc(config, force=False):
    """Build or load the exact V1002 Prime 5K directed CCC matrix.

    The matrix is written as an NPY memmap so downstream training can stream
    batches without duplicating the full N x F array in memory.
    """
    paths = cache_paths(config)
    paths["cache"].mkdir(parents=True, exist_ok=True)
    signature = _manifest_signature(config)
    if not force and _cache_is_valid(paths, signature):
        return load_cached_input(config)

    try:
        import anndata as ad
    except ImportError as error:
        raise ImportError("anndata is required to prepare Prime 5K input") from error
    build_neighborhoods, LRAtlas, load_lr_atlas, side_expression, aggregate = _import_v1002()
    adata = ad.read_h5ad(config.processed_h5ad)
    required_obs = {"cell_id", "cell_type_coarse"}
    missing = required_obs.difference(adata.obs.columns)
    if missing:
        raise ValueError(f"Processed AnnData lacks required obs columns: {sorted(missing)}")
    if "spatial" not in adata.obsm:
        raise ValueError("Processed AnnData lacks obsm['spatial']")
    if adata.uns.get("dataset") != "Xenium_Prime_Breast_Cancer_FFPE":
        raise ValueError("Unexpected dataset identity in processed AnnData")
    if not adata.obs["cell_id"].is_unique:
        raise ValueError("cell_id must be unique")

    coordinates = np.asarray(adata.obsm["spatial"], dtype=np.float32)
    if coordinates.shape != (adata.n_obs, 2) or not np.isfinite(coordinates).all():
        raise ValueError("spatial coordinates must be a finite N x 2 array")
    categories = sorted(adata.obs["cell_type_coarse"].astype(str).unique())
    cell_type = pd.Categorical(adata.obs["cell_type_coarse"].astype(str), categories=categories).codes.astype(np.int16)
    if np.any(cell_type < 0):
        raise ValueError("cell_type_coarse contains missing values")
    n_cells, n_types = adata.n_obs, len(categories)

    neighborhoods = build_neighborhoods(coordinates, k=config.n_neighbors, sigma=config.sigma_um)
    members = neighborhoods.indices.reshape(n_cells, config.n_neighbors)
    anchor_weights = neighborhoods.weights.reshape(n_cells, config.n_neighbors).astype(np.float32)
    device = resolve_device(config.device)
    composition, _, opportunity = aggregate(
        coordinates, cell_type,
        np.zeros((n_cells, 1), dtype=np.float32),
        np.zeros((n_cells, 1), dtype=np.float32),
        n_types, sigma=config.sigma_um, members=members, anchor_weights=anchor_weights,
        batch_size=config.ccc_batch_size, lr_batch_size=config.lr_batch_size,
        device=device, compute_signal=False,
    )
    if not np.allclose(composition.sum(axis=1), 1, atol=1e-5):
        raise RuntimeError("V1002 composition audit failed")
    pair_support = (opportunity > 0).sum(axis=0)

    atlas = load_lr_atlas(config.lr_atlas)
    panel = {str(gene).upper() for gene in adata.var_names}
    measurable = tuple(row for row in atlas.interactions
                       if all(gene in panel for gene in row.ligand_components + row.receptor_components))
    if not measurable:
        raise RuntimeError("No fully measurable LR interaction in the Prime 5K panel")
    measurable_atlas = LRAtlas(measurable, {"n_raw_lr": len(atlas), "n_unique_lr": len(measurable)}, atlas.source_path)
    lr_genes = tuple(sorted({gene for row in measurable
                             for gene in row.ligand_components + row.receptor_components}))
    gene_lookup = {str(gene).upper(): str(gene) for gene in adata.var_names}
    selected_genes = [gene_lookup[gene] for gene in lr_genes]
    selected_expression = adata[:, selected_genes].X
    if hasattr(selected_expression, "toarray"):
        selected_expression = selected_expression.toarray()
    selected_expression = np.asarray(selected_expression, dtype=np.float32)
    ligand, receptor = side_expression(selected_expression, lr_genes, measurable_atlas)
    del selected_expression

    coverage_l = np.stack([(ligand[cell_type == index] > 0).mean(axis=0) for index in range(n_types)])
    coverage_r = np.stack([(receptor[cell_type == index] > 0).mean(axis=0) for index in range(n_types)])
    coverage = ((coverage_l[:, None, :] >= config.expression_coverage_fraction)
                & (coverage_r[None, :, :] >= config.expression_coverage_fraction)).reshape(n_types * n_types, -1)
    support_minimum = max(config.opportunity_support_minimum,
                          int(math.ceil(config.opportunity_support_fraction * n_cells)))
    supported = pair_support >= support_minimum
    feature_mask = (coverage & supported[:, None]).reshape(-1)
    feature_indices = np.flatnonzero(feature_mask)
    if len(feature_indices) < config.niches:
        raise RuntimeError(f"Only {len(feature_indices)} CCC features passed the fixed V1002 filter")

    pair_index, lr_index = np.divmod(feature_indices, len(measurable))
    sender_index, receiver_index = np.divmod(pair_index, n_types)
    feature_table = pd.DataFrame({
        "feature_id": np.arange(len(feature_indices), dtype=np.int64),
        "flat_feature_index": feature_indices,
        "sender": np.asarray(categories)[sender_index],
        "receiver": np.asarray(categories)[receiver_index],
        "lr_id": [measurable[index].lr_id for index in lr_index],
        "ligand": [measurable[index].ligand for index in lr_index],
        "receptor": [measurable[index].receptor for index in lr_index],
    })
    feature_table["ccc"] = (feature_table["sender"] + " → " + feature_table["receiver"]
                            + " | " + feature_table["ligand"] + "–" + feature_table["receptor"])

    temporary = paths["ccc"].with_name("ccc.incomplete.npy")
    output = np.lib.format.open_memmap(temporary, mode="w+", dtype=np.float32,
                                       shape=(n_cells, len(feature_indices)))
    _, signal, signal_opportunity = aggregate(
        coordinates, cell_type, ligand, receptor, n_types,
        sigma=config.sigma_um, members=members, anchor_weights=anchor_weights,
        feature_mask=feature_mask, batch_size=config.ccc_batch_size,
        lr_batch_size=config.lr_batch_size, device=device, communication_out=output,
    )
    if not np.allclose(opportunity, signal_opportunity, rtol=1e-5, atol=1e-6):
        raise RuntimeError("CCC and opportunity passes used inconsistent physical pairs")
    if not np.isfinite(signal).all() or np.any(signal < 0):
        raise RuntimeError("CCC matrix must be finite and nonnegative")
    output.flush()
    del signal, output
    temporary.replace(paths["ccc"])

    matrix = np.load(paths["ccc"], mmap_mode="r")
    magnitude = np.empty(n_cells, dtype=np.float32)
    for start in range(0, n_cells, 8192):
        stop = min(start + 8192, n_cells)
        magnitude[start:stop] = np.asarray(matrix[start:stop]).sum(axis=1)
    np.save(paths["magnitude"], magnitude)
    np.save(paths["coordinates"], coordinates)
    pd.DataFrame({
        "cell_id": adata.obs["cell_id"].astype(str).to_numpy(),
        "cell_type": adata.obs["cell_type_coarse"].astype(str).to_numpy(),
    }).to_csv(paths["cells"], index=False, compression="gzip")
    feature_table.to_csv(paths["features"], index=False)
    manifest = {
        "signature": signature,
        "n_cells": int(n_cells),
        "n_genes": int(adata.n_vars),
        "n_cell_types": int(n_types),
        "cell_types": categories,
        "n_atlas_lr": int(len(atlas)),
        "n_measurable_lr": int(len(measurable)),
        "n_features": int(len(feature_indices)),
        "support_minimum": int(support_minimum),
        "device_used_for_ccc": device,
    }
    paths["manifest"].write_text(json.dumps(manifest, indent=2))
    return load_cached_input(config)


def load_cached_input(config):
    paths = cache_paths(config)
    if not paths["manifest"].is_file():
        raise FileNotFoundError("Prime 5K CCC cache is absent; run prepare_prime5k_ccc first")
    manifest = json.loads(paths["manifest"].read_text())
    matrix = np.load(paths["ccc"], mmap_mode="r")
    magnitude = np.load(paths["magnitude"], mmap_mode="r")
    coordinates = np.load(paths["coordinates"], mmap_mode="r")
    cells = pd.read_csv(paths["cells"])
    features = pd.read_csv(paths["features"])
    expected = (manifest["n_cells"], manifest["n_features"])
    if matrix.shape != expected or len(magnitude) != expected[0] or len(cells) != expected[0]:
        raise ValueError("Cached Prime 5K CCC files have inconsistent shapes")
    if coordinates.shape != (expected[0], 2) or len(features) != expected[1]:
        raise ValueError("Cached coordinate or feature metadata has inconsistent shapes")
    return {
        "matrix": matrix,
        "magnitude": magnitude,
        "coordinates": coordinates,
        "cells": cells,
        "features": features,
        "manifest": manifest,
        "paths": paths,
    }


class CCCProportionDataset:
    """Torch-compatible streaming view of p_n = x_n / (m_n + epsilon)."""

    def __init__(self, matrix, magnitude, epsilon=1e-8):
        if matrix.ndim != 2 or magnitude.shape != (matrix.shape[0],):
            raise ValueError("matrix and magnitude shapes are inconsistent")
        self.matrix = matrix
        self.magnitude = magnitude
        self.epsilon = float(epsilon)

    def __len__(self):
        return self.matrix.shape[0]

    def __getitem__(self, index):
        import torch
        row = np.array(self.matrix[index], dtype=np.float32, copy=True)
        magnitude = np.float32(self.magnitude[index])
        proportion = row / (magnitude + self.epsilon)
        return torch.from_numpy(proportion), torch.tensor(magnitude), int(index)

