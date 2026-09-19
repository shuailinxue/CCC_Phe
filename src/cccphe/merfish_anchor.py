from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.neighbors import NearestNeighbors

from .workflow import _cellchat_lr, _entity_genes


SAMPLES = {
    "GSM9651110": "1041_region_0",
    "GSM9651111": "121_region_0",
    "GSM9651112": "1004_region_0",
    "GSM9651113": "1005_region_0",
    "GSM9651114": "1021_region_0",
    "GSM9651115": "1037_region_0",
    "GSM9651116": "1023_region_0",
    "GSM9651117": "106_region_0",
}


def _write_json(value: dict, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _reference_markers(
    reference_h5ad: Path,
    panel_genes: list[str],
    *,
    cell_type_key: str,
    markers_per_type: int,
) -> tuple[list[str], dict[str, list[str]], dict]:
    import anndata as ad

    adata = ad.read_h5ad(reference_h5ad)
    reference_genes = pd.Index(adata.var_names.astype(str).str.upper())
    if reference_genes.duplicated().any():
        raise ValueError("The scRNA reference contains duplicated gene symbols")
    panel = [gene for gene in panel_genes if gene in reference_genes]
    if len(panel) < 100:
        raise ValueError(f"Only {len(panel)} MERFISH genes overlap the scRNA reference")
    counts = adata[:, panel].X
    counts = counts.tocsr().astype(np.float32) if sp.issparse(counts) else sp.csr_matrix(counts)
    total = np.asarray(adata.X.sum(axis=1)).ravel().astype(np.float64)
    labels = adata.obs[cell_type_key].astype(str).to_numpy()
    cell_types = sorted(pd.unique(labels).tolist())
    centroid_cpm = []
    for cell_type in cell_types:
        selected = labels == cell_type
        centroid_cpm.append(
            np.asarray(counts[selected].sum(axis=0)).ravel()
            / max(float(total[selected].sum()), 1.0)
            * 1e6
        )
    centroid_cpm = np.asarray(centroid_cpm)
    log_centroid = np.log1p(centroid_cpm)
    markers: dict[str, list[str]] = {}
    for index, cell_type in enumerate(cell_types):
        contrast = log_centroid[index] - np.max(
            np.delete(log_centroid, index, axis=0), axis=0
        )
        order = np.argsort(contrast)[::-1]
        chosen = [
            panel[column]
            for column in order
            if centroid_cpm[index, column] >= 1.0
        ][:markers_per_type]
        if len(chosen) < 10:
            raise ValueError(f"Too few panel markers for {cell_type}: {len(chosen)}")
        markers[cell_type] = chosen

    # An outcome- and TLS-blind internal check that the panel resolves the six
    # broad labels already used by BayesPrism.
    selected_genes = sorted({gene for values in markers.values() for gene in values})
    panel_index = {gene: index for index, gene in enumerate(panel)}
    normalized = counts[:, [panel_index[g] for g in selected_genes]].multiply(
        (1e4 / np.maximum(total, 1.0))[:, None]
    ).tocsr()
    normalized.data = np.log1p(normalized.data)
    scores = np.zeros((len(labels), len(cell_types)), dtype=np.float32)
    selected_index = {gene: index for index, gene in enumerate(selected_genes)}
    for index, cell_type in enumerate(cell_types):
        columns = [selected_index[gene] for gene in markers[cell_type]]
        scores[:, index] = np.asarray(normalized[:, columns].mean(axis=1)).ravel()
    predicted = np.asarray(cell_types)[scores.argmax(axis=1)]
    audit = {
        "reference_cells": int(len(labels)),
        "reference_panel_overlap": int(len(panel)),
        "cell_types": cell_types,
        "markers_per_type": int(markers_per_type),
        "reference_resubstitution_accuracy": float(np.mean(predicted == labels)),
    }
    return cell_types, markers, audit


def _aggregate_panel_counts(
    transcript_file: Path,
    retained_cell_ids: np.ndarray,
    panel_genes: list[str],
    *,
    chunksize: int,
) -> sp.csr_matrix:
    cell_index = pd.Index(retained_cell_ids)
    gene_index = {gene: index for index, gene in enumerate(panel_genes)}
    matrix = sp.csr_matrix(
        (len(retained_cell_ids), len(panel_genes)), dtype=np.float32
    )
    for chunk_number, chunk in enumerate(
        pd.read_csv(
            transcript_file,
            compression="gzip",
            usecols=["gene", "cell_id"],
            dtype={"gene": "string", "cell_id": "int64"},
            chunksize=chunksize,
        ),
        start=1,
    ):
        row = cell_index.get_indexer(chunk["cell_id"].to_numpy(np.int64))
        column = chunk["gene"].map(gene_index).fillna(-1).to_numpy(np.int32)
        keep = (row >= 0) & (column >= 0)
        if keep.any():
            key = row[keep].astype(np.int64) * len(panel_genes) + column[keep]
            unique, value = np.unique(key, return_counts=True)
            block = sp.coo_matrix(
                (
                    value.astype(np.float32),
                    (unique // len(panel_genes), unique % len(panel_genes)),
                ),
                shape=matrix.shape,
            ).tocsr()
            matrix = matrix + block
        if chunk_number % 10 == 0:
            print(
                f"{transcript_file.name}: parsed {chunk_number * chunksize:,} transcript rows",
                flush=True,
            )
    matrix.eliminate_zeros()
    return matrix


def _label_cells(
    counts: sp.csr_matrix,
    panel_genes: list[str],
    cell_types: list[str],
    markers: dict[str, list[str]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    marker_genes = sorted({gene for values in markers.values() for gene in values})
    panel_index = {gene: index for index, gene in enumerate(panel_genes)}
    marker_index = {gene: index for index, gene in enumerate(marker_genes)}
    total = np.asarray(counts.sum(axis=1)).ravel()
    normalized = counts[:, [panel_index[gene] for gene in marker_genes]].multiply(
        (1e4 / np.maximum(total, 1.0))[:, None]
    ).tocsr()
    normalized.data = np.log1p(normalized.data)
    scores = np.zeros((counts.shape[0], len(cell_types)), dtype=np.float32)
    for index, cell_type in enumerate(cell_types):
        columns = [marker_index[gene] for gene in markers[cell_type]]
        scores[:, index] = np.asarray(normalized[:, columns].mean(axis=1)).ravel()
    best = scores.argmax(axis=1)
    ordered = np.sort(scores, axis=1)
    labels = np.asarray(cell_types, dtype=object)[best]
    uninformative = ordered[:, -1] <= 0
    labels[uninformative] = "Unclassified"
    margin = ordered[:, -1] - ordered[:, -2]
    return labels.astype(str), ordered[:, -1], margin


def _entity_cpm(
    pseudobulk_cpm: np.ndarray,
    panel_index: dict[str, int],
    entity: str,
    epsilon: float = 1e-8,
) -> np.ndarray:
    columns = [panel_index[gene] for gene in _entity_genes(entity)]
    values = np.maximum(pseudobulk_cpm[:, columns], 0.0)
    result = np.exp(np.log(values + epsilon).mean(axis=1))
    # A heteromeric entity is observable only when every required subunit was
    # actually detected. Epsilon is numerical protection, not evidence.
    result[(values <= 0).any(axis=1)] = 0.0
    return result


def _sample_anchor(
    accession: str,
    metadata: pd.DataFrame,
    counts: sp.csr_matrix,
    labels: np.ndarray,
    cell_types: list[str],
    panel_genes: list[str],
    lr: pd.DataFrame,
    *,
    neighbors: int,
    maximum_neighbor_distance: float,
    minimum_cells_per_type: int,
    minimum_pair_edges: int,
) -> tuple[pd.DataFrame, dict]:
    classified = labels != "Unclassified"
    metadata = metadata.loc[classified].reset_index(drop=True)
    counts = counts[classified]
    labels = labels[classified]
    label_index = {name: index for index, name in enumerate(cell_types)}
    codes = np.asarray([label_index[value] for value in labels], dtype=np.int16)
    type_counts = np.bincount(codes, minlength=len(cell_types))

    coordinates = metadata[["center_x", "center_y"]].to_numpy(np.float64)
    model = NearestNeighbors(n_neighbors=neighbors + 1, algorithm="kd_tree", n_jobs=-1)
    distance, index = model.fit(coordinates).kneighbors(coordinates)
    source = np.repeat(np.arange(len(coordinates), dtype=np.int64), neighbors)
    receiver = index[:, 1:].reshape(-1)
    edge_distance = distance[:, 1:].reshape(-1)
    keep_edge = edge_distance <= maximum_neighbor_distance
    source, receiver, edge_distance = (
        source[keep_edge], receiver[keep_edge], edge_distance[keep_edge]
    )
    pair_code = codes[source].astype(np.int64) * len(cell_types) + codes[receiver]
    pair_count = np.bincount(pair_code, minlength=len(cell_types) ** 2).reshape(
        len(cell_types), len(cell_types)
    )
    receiver_fraction = type_counts / max(type_counts.sum(), 1)
    expected = type_counts[:, None] * neighbors * receiver_fraction[None, :]
    contact_enrichment = np.divide(
        pair_count,
        expected,
        out=np.zeros_like(expected, dtype=np.float64),
        where=expected > 0,
    )

    totals_by_type = np.bincount(
        codes,
        weights=np.asarray(counts.sum(axis=1)).ravel(),
        minlength=len(cell_types),
    )
    summed = np.vstack(
        [np.asarray(counts[codes == index].sum(axis=0)).ravel() for index in range(len(cell_types))]
    )
    pseudobulk_cpm = np.divide(
        summed,
        totals_by_type[:, None],
        out=np.zeros_like(summed, dtype=np.float64),
        where=totals_by_type[:, None] > 0,
    ) * 1e6
    panel_index = {gene: index for index, gene in enumerate(panel_genes)}

    rows = []
    for interaction in lr.itertuples(index=False):
        ligand = _entity_cpm(pseudobulk_cpm, panel_index, interaction.ligand)
        receptor = _entity_cpm(pseudobulk_cpm, panel_index, interaction.receptor)
        molecular_support = np.sqrt(
            np.maximum(ligand[:, None], 0) * np.maximum(receptor[None, :], 0)
        )
        w = contact_enrichment * molecular_support
        for sender_index, sender in enumerate(cell_types):
            for receiver_index, receiver_type in enumerate(cell_types):
                confident = bool(
                    type_counts[sender_index] >= minimum_cells_per_type
                    and type_counts[receiver_index] >= minimum_cells_per_type
                    and pair_count[sender_index, receiver_index] >= minimum_pair_edges
                    and molecular_support[sender_index, receiver_index] > 1e-6
                )
                rows.append(
                    {
                        "sample_id": accession,
                        "sender": sender,
                        "receiver": receiver_type,
                        "lr_id": interaction.lr_id,
                        "ligand": interaction.ligand,
                        "receptor": interaction.receptor,
                        "pathway": interaction.pathway,
                        "signaling_type": interaction.signaling_type,
                        "W_st_sample": w[sender_index, receiver_index],
                        "M_st_sample": molecular_support[sender_index, receiver_index],
                        "ligand_cpm": ligand[sender_index],
                        "receptor_cpm": receptor[receiver_index],
                        "contact_enrichment": contact_enrichment[sender_index, receiver_index],
                        "neighbor_edges": int(pair_count[sender_index, receiver_index]),
                        "confident": confident,
                    }
                )
    audit = {
        "accession": accession,
        "retained_cells": int(len(classified)),
        "classified_cells": int(classified.sum()),
        "unclassified_cells": int((~classified).sum()),
        "cell_type_counts": {
            cell_types[index]: int(type_counts[index]) for index in range(len(cell_types))
        },
        "neighbor_edges_retained": int(len(source)),
        "neighbor_distance_median": float(np.median(edge_distance)),
        "neighbor_distance_q95": float(np.quantile(edge_distance, 0.95)),
    }
    return pd.DataFrame(rows), audit


def build_merfish_anchor(
    raw_root: str | Path,
    panel_file: str | Path,
    reference_h5ad: str | Path,
    output: str | Path,
    *,
    cell_type_key: str = "cell_type",
    minimum_transcripts: int = 10,
    markers_per_type: int = 30,
    neighbors: int = 6,
    maximum_neighbor_distance: float = 30.0,
    minimum_cells_per_type: int = 100,
    minimum_pair_edges: int = 100,
    minimum_st_samples: int = 4,
    chunksize: int = 5_000_000,
    overwrite: bool = False,
) -> Path:
    """Build a TLS-blind single-cell MERFISH directional CCC anchor.

    GSE327192 does not publish cell labels. Broad labels are therefore
    transferred from the same liver scRNA reference used by BayesPrism, using
    only genes in the MERFISH panel. Spatial strength is direct kNN contact
    enrichment multiplied by the MERFISH cell-type LR molecular support.
    """
    raw_root, output = Path(raw_root), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    sample_dir = output / "sample_cache"
    label_dir = output / "cell_labels"
    sample_dir.mkdir(exist_ok=True)
    label_dir.mkdir(exist_ok=True)
    panel_genes = [
        value.strip().upper()
        for value in Path(panel_file).read_text(encoding="utf-8").splitlines()
        if value.strip() and not value.lower().startswith("blank-")
    ]
    cell_types, markers, reference_audit = _reference_markers(
        Path(reference_h5ad), panel_genes,
        cell_type_key=cell_type_key, markers_per_type=markers_per_type,
    )
    pd.DataFrame(
        [(cell_type, rank + 1, gene) for cell_type, values in markers.items() for rank, gene in enumerate(values)],
        columns=["cell_type", "rank", "gene"],
    ).to_csv(output / "label_transfer_markers.tsv", sep="\t", index=False)

    lr_all = _cellchat_lr()
    panel_set = set(panel_genes)
    lr = lr_all[
        lr_all.apply(
            lambda row: set(_entity_genes(row.ligand)).issubset(panel_set)
            and set(_entity_genes(row.receptor)).issubset(panel_set),
            axis=1,
        )
    ].reset_index(drop=True)
    if lr.empty:
        raise ValueError("No CellChat LR pair is fully observed by the MERFISH panel")

    audits = []
    sample_frames = []
    for accession, sample_name in SAMPLES.items():
        cache = sample_dir / f"{accession}.parquet"
        audit_cache = sample_dir / f"{accession}.audit.json"
        if cache.exists() and audit_cache.exists() and not overwrite:
            print(f"MERFISH anchor cache reused: {accession}", flush=True)
            sample_frames.append(pd.read_parquet(cache))
            audits.append(json.loads(audit_cache.read_text(encoding="utf-8")))
            continue
        prefix = f"{accession}_{sample_name}"
        metadata_file = raw_root / f"{prefix}_cell_metadata.csv.gz"
        transcript_file = raw_root / f"{prefix}_detected_transcripts.csv.gz"
        metadata = pd.read_csv(
            metadata_file,
            usecols=["EntityID", "fov", "center_x", "center_y", "transcript_count"],
            dtype={"EntityID": "int64", "fov": "int32", "transcript_count": "int32"},
        )
        metadata = metadata.loc[
            metadata.transcript_count.ge(minimum_transcripts)
        ].reset_index(drop=True)
        print(f"MERFISH counts/labels starting: {accession} ({len(metadata):,} cells)", flush=True)
        counts = _aggregate_panel_counts(
            transcript_file,
            metadata.EntityID.to_numpy(np.int64),
            panel_genes,
            chunksize=chunksize,
        )
        labels, score, margin = _label_cells(
            counts, panel_genes, cell_types, markers
        )
        label_table = metadata.copy()
        label_table.insert(0, "sample_id", accession)
        label_table["cell_type"] = labels
        label_table["label_score"] = score
        label_table["label_margin"] = margin
        label_table.to_parquet(label_dir / f"{accession}.parquet", index=False)
        frame, audit = _sample_anchor(
            accession, metadata, counts, labels, cell_types, panel_genes, lr,
            neighbors=neighbors,
            maximum_neighbor_distance=maximum_neighbor_distance,
            minimum_cells_per_type=minimum_cells_per_type,
            minimum_pair_edges=minimum_pair_edges,
        )
        audit.update(
            {
                "input_cells": int(len(metadata)),
                "minimum_transcripts": int(minimum_transcripts),
                "mean_label_score": float(np.mean(score)),
                "median_label_margin": float(np.median(margin)),
            }
        )
        temporary = cache.with_suffix(".parquet.tmp")
        frame.to_parquet(temporary, index=False)
        temporary.replace(cache)
        _write_json(audit, audit_cache)
        sample_frames.append(frame)
        audits.append(audit)
        print(f"MERFISH anchor complete: {accession}", flush=True)

    per_sample = pd.concat(sample_frames, ignore_index=True)
    per_sample.to_parquet(output / "anchor_per_sample.parquet", index=False)
    keys = ["sender", "receiver", "lr_id", "ligand", "receptor", "pathway", "signaling_type"]
    valid = per_sample.loc[per_sample.confident].copy()
    anchor = (
        valid.groupby(keys, observed=True)
        .agg(
            W_st=("W_st_sample", "median"),
            M_st=("M_st_sample", "median"),
            n_st_samples=("sample_id", "nunique"),
            n_confident=("confident", "sum"),
        )
        .reset_index()
    )
    anchor["confident"] = anchor.n_confident.ge(minimum_st_samples)
    anchor.to_parquet(output / "st_anchor.parquet", index=False)
    lr_genes = sorted(
        {gene for entity in pd.concat([anchor.ligand, anchor.receptor]) for gene in _entity_genes(entity)}
    )
    (output / "lr_genes.txt").write_text("\n".join(lr_genes) + "\n", encoding="utf-8")
    audit = {
        "method": "single-cell MERFISH panel + scRNA broad-label transfer + spatial kNN contact enrichment",
        "tls_blind": True,
        "tls_inputs_used": [],
        "samples": len(SAMPLES),
        "panel_genes": len(panel_genes),
        "panel_complete_cellchat_lr": int(len(lr)),
        "cell_types": cell_types,
        "minimum_st_samples": int(minimum_st_samples),
        "confident_anchor_rows": int(anchor.confident.sum()),
        "reference_label_audit": reference_audit,
        "sample_audits": audits,
    }
    _write_json(audit, output / "audit.json")
    return output / "st_anchor.parquet"
