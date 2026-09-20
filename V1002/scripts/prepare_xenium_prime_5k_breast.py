"""Prepare the official Xenium Prime 5K breast output for the V1002 notebook.

Only cell metadata, the cell-feature matrix and the small official analysis
cluster archive are needed. The annotations below are provisional, outcome-blind
marker labels for the official gene-expression clusters, not 10x cell types.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import anndata as ad
import h5py
import numpy as np
import pandas as pd
from scipy import sparse
import zarr


DATA_ROOT = Path(
    "/data1/xueshuailin/CCC_Phe_Niche/data/xenium_prime_5k_breast"
)
OUTPUT_NAME = "XeniumPrime5K_Breast_V1002_processed.h5ad"

# Broad cell families only. Ambiguous tumor/normal epithelial distinctions are
# deliberately not inferred from this simple marker panel.
MARKERS = {
    "Epithelial": ("EPCAM", "MUC1", "ERBB2", "GATA3"),
    "Myoepithelial": ("TP63", "ITGA6", "CNN1", "MYL9"),
    "T cell": ("CD3D", "CD3E", "TRAC", "IL7R"),
    "B cell": ("MS4A1", "CD79A", "CD79B", "CD19"),
    "Plasma cell": ("MZB1", "XBP1", "PRDM1", "TNFRSF17"),
    "Macrophage": ("CD68", "CSF1R", "C1QC", "CD163"),
    "Dendritic cell": ("FCER1A", "CLEC10A", "CD1C", "ITGAX"),
    "Fibroblast": ("DCN", "LUM", "COL5A1", "FAP"),
    "Endothelial": ("PECAM1", "KDR", "CDH5", "FLT1"),
    "Pericyte": ("RGS5", "PDGFRB", "CSPG4", "MCAM"),
    "Mast cell": ("KIT", "MS4A2", "TPSG1", "HPGDS"),
}


def load_gene_matrix(raw: Path) -> ad.AnnData:
    """Read only Gene Expression features from the 10x CSC HDF5 matrix."""
    with h5py.File(raw / "cell_feature_matrix.h5", "r") as handle:
        matrix = handle["matrix"]
        feature = matrix["features"]
        names = np.char.decode(feature["name"][:].astype("S"), "utf-8")
        gene_ids = np.char.decode(feature["id"][:].astype("S"), "utf-8")
        types = np.char.decode(feature["feature_type"][:].astype("S"), "utf-8")
        keep = np.flatnonzero(types == "Gene Expression")
        barcodes = np.char.decode(matrix["barcodes"][:].astype("S"), "utf-8")
        shape = tuple(int(value) for value in matrix["shape"][:])
        full = sparse.csc_matrix(
            (matrix["data"][:], matrix["indices"][:], matrix["indptr"][:]),
            shape=shape,
        )
    if shape[1] != len(barcodes) or len(set(names[keep])) != len(keep):
        raise ValueError("10x matrix barcodes or gene names are inconsistent")
    genes_by_cells = full[keep, :]
    del full
    result = ad.AnnData(X=genes_by_cells.T.tocsr())
    result.obs_names = pd.Index(barcodes.astype(str), name="cell_id")
    result.var_names = pd.Index(names[keep].astype(str), name="gene_name")
    result.var["gene_id"] = gene_ids[keep].astype(str)
    result.var["feature_type"] = "Gene Expression"
    return result


def official_graph_clusters(archive: Path, n_cells: int) -> np.ndarray:
    """Decode 10x graph-cluster memberships; cells outside analysis stay -1."""
    labels = np.full(n_cells, -1, dtype=np.int16)
    with zarr.storage.ZipStore(str(archive), mode="r") as store:
        root = zarr.open(store, mode="r")
        attrs = root["cell_groups"].attrs
        if attrs["grouping_names"][0] != "gene_expression_graphclust":
            raise ValueError("Expected official gene-expression graph clusters")
        starts = root["cell_groups/0/indptr"][:].astype(np.int64)
        indices = root["cell_groups/0/indices"][:].astype(np.int64)
    stops = np.r_[starts[1:], len(indices)]
    for cluster, (begin, end) in enumerate(zip(starts, stops)):
        members = indices[begin:end]
        if cluster == len(starts) - 1:
            # The final Zarr chunk is zero-padded in this release. Index 0 is
            # already present in an earlier cluster, so these are padding.
            members = members[members != 0]
        if np.any(members >= n_cells) or np.any(labels[members] != -1):
            raise ValueError("Official graph clusters overlap or exceed cell count")
        labels[members] = cluster
    return labels


def annotate_clusters(
    expression: sparse.csr_matrix, gene_names: pd.Index, clusters: np.ndarray
) -> tuple[np.ndarray, pd.DataFrame, dict[str, list[str]]]:
    """Assign fixed broad marker signatures to official clusters."""
    gene_index = {name.upper(): index for index, name in enumerate(gene_names)}
    available = {
        cell_type: [gene for gene in genes if gene in gene_index]
        for cell_type, genes in MARKERS.items()
    }
    if any(len(genes) < 2 for genes in available.values()):
        raise ValueError("Fewer than two panel genes for a marker signature")
    selected = list(dict.fromkeys(gene for genes in available.values() for gene in genes))
    marker_matrix = expression[:, [gene_index[gene] for gene in selected]].copy()
    marker_matrix.data[:] = 1
    valid = clusters >= 0
    n_clusters = int(clusters[valid].max()) + 1
    cluster_rows = sparse.csr_matrix(
        (np.ones(valid.sum(), dtype=np.float32),
         (clusters[valid].astype(np.int32), np.flatnonzero(valid))),
        shape=(n_clusters, expression.shape[0]),
    )
    sizes = np.bincount(clusters[valid], minlength=n_clusters)
    detection = (cluster_rows @ marker_matrix).toarray() / sizes[:, None]
    global_detection = np.asarray(marker_matrix[valid].mean(axis=0)).ravel()
    relative = np.log2((detection + 0.01) / (global_detection[None, :] + 0.01))
    marker_pos = {gene: index for index, gene in enumerate(selected)}
    names = list(available)
    scores = np.column_stack(
        [relative[:, [marker_pos[gene] for gene in available[name]]].mean(axis=1)
         for name in names]
    )
    top = scores.argmax(axis=1)
    best = scores[np.arange(n_clusters), top]
    second = np.partition(scores, -2, axis=1)[:, -2]
    assigned = np.asarray([names[index] for index in top], dtype=object)
    assigned[best <= 0] = "Unassigned"
    cell_labels = np.full(len(clusters), "Unassigned", dtype=object)
    cell_labels[valid] = assigned[clusters[valid]]
    audit = pd.DataFrame({
        "official_graph_cluster": np.arange(n_clusters) + 1,
        "n_cells": sizes,
        "provisional_cell_type": assigned,
        "marker_score": best,
        "marker_score_margin": best - second,
    })
    for index, name in enumerate(names):
        audit[f"score_{name}"] = scores[:, index]
    return cell_labels, audit, available


def prepare(root: Path = DATA_ROOT) -> Path:
    raw, processed = root / "raw", root / "processed"
    processed.mkdir(parents=True, exist_ok=True)
    for name in ("cells.parquet", "cell_feature_matrix.h5", "analysis.zarr.zip"):
        if not (raw / name).is_file():
            raise FileNotFoundError(raw / name)
    adata = load_gene_matrix(raw)
    # Only these metadata columns are used downstream. The official 10x parquet
    # release has unreadable count columns in some clients; no count metadata is
    # needed because the expression matrix is loaded from the HDF5 file.
    cells = pd.read_parquet(
        raw / "cells.parquet", columns=["cell_id", "x_centroid", "y_centroid"]
    )
    ids = cells["cell_id"].astype(str)
    if not ids.is_unique:
        raise ValueError("Cell metadata has duplicate cell IDs")
    cells.index = pd.Index(ids, name="cell_id")
    cells = cells.reindex(adata.obs_names)
    if cells["cell_id"].isna().any():
        raise ValueError("Cell metadata cannot be aligned to expression barcodes")
    coordinates = cells[["x_centroid", "y_centroid"]].to_numpy(dtype=np.float32)
    if not np.isfinite(coordinates).all() or np.any(adata.X.data < 0):
        raise ValueError("Coordinates or gene counts are invalid")
    adata.obs = cells
    adata.obsm["spatial"] = coordinates
    clusters = official_graph_clusters(raw / "analysis.zarr.zip", adata.n_obs)
    labels, audit, marker_coverage = annotate_clusters(adata.X, adata.var_names, clusters)
    adata.obs["official_graph_cluster"] = pd.Categorical(
        [f"Cluster {index + 1}" if index >= 0 else "Not clustered" for index in clusters]
    )
    adata.obs["cell_type_coarse"] = pd.Categorical(labels)
    adata.obs["cell_type"] = adata.obs["cell_type_coarse"]
    adata.uns["dataset"] = "Xenium_Prime_Breast_Cancer_FFPE"
    adata.uns["cell_type_annotation"] = (
        "Provisional fixed-marker labels of official gene-expression graph clusters; "
        "not official 10x cell-type annotations"
    )
    adata.uns["marker_signatures_json"] = json.dumps(marker_coverage, sort_keys=True)
    output = processed / OUTPUT_NAME
    temporary = output.with_name(output.name + ".part")
    adata.write_h5ad(temporary, compression="gzip")
    temporary.replace(output)
    audit.to_csv(processed / "cluster_marker_audit.csv", index=False)
    source = Path(__file__).resolve().parents[1] / "src"
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))
    from phenoniche.v1002.lr_atlas import load_lr_atlas

    atlas = load_lr_atlas(str(source.parent / "data/commuspace_human_lr_atlas.tsv"))
    panel = set(adata.var_names.str.upper())
    measurable = [
        row for row in atlas.interactions
        if all(gene in panel for gene in row.ligand_components + row.receptor_components)
    ]
    pd.DataFrame(
        [{"lr_index": index, "lr_id": row.lr_id, "ligand": row.ligand,
          "receptor": row.receptor}
         for index, row in enumerate(measurable)]
    ).to_csv(processed / "panel_measurable_lr.csv", index=False)
    manifest = {
        "source": "10x Xenium Prime 5K FFPE human breast cancer",
        "n_cells": int(adata.n_obs),
        "n_genes": int(adata.n_vars),
        "matrix_nnz": int(adata.X.nnz),
        "n_official_graph_clusters": int(audit.shape[0]),
        "n_atlas_lr": len(atlas),
        "n_panel_measurable_lr": len(measurable),
        "n_unassigned": int(np.sum(labels == "Unassigned")),
        "annotation": "provisional_fixed_markers_on_official_graph_clusters",
        "marker_coverage": marker_coverage,
    }
    (processed / "preprocess_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DATA_ROOT)
    args = parser.parse_args()
    print(prepare(args.root))
