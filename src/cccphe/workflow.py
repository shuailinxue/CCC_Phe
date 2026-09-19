from __future__ import annotations

import gzip
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.io import mmwrite
from sklearn.neighbors import NearestNeighbors, radius_neighbors_graph

from .ccc_tensor import construct_patient_ccc


def _write_lines(values, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(map(str, values)) + "\n", encoding="utf-8")


def _write_mtx_gz(matrix: sp.spmatrix, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wb") as handle:
        mmwrite(handle, matrix)


def _counts(adata, layer: str | None):
    matrix = adata.X if layer is None else adata.layers[layer]
    matrix = matrix.tocsr() if sp.issparse(matrix) else sp.csr_matrix(matrix)
    if matrix.data.size and matrix.data.min() < 0:
        raise ValueError("Count matrices must be non-negative")
    return matrix


def _read_visium_counts_and_coordinates(source: Path):
    """Read Space Ranger counts and spot coordinates without loading images."""
    import scanpy as sc

    source = Path(source)
    adata = sc.read_10x_h5(source / "filtered_feature_bc_matrix.h5")
    spatial_dir = source / "spatial"
    positions_file = spatial_dir / "tissue_positions.csv"
    if not positions_file.exists():
        positions_file = spatial_dir / "tissue_positions_list.csv"
    if not positions_file.exists():
        raise FileNotFoundError(f"Missing Space Ranger tissue positions: {spatial_dir}")

    with positions_file.open(encoding="utf-8") as handle:
        first_cell = handle.readline().split(",", 1)[0].strip()
    positions = pd.read_csv(
        positions_file,
        header=0 if first_cell.lower() == "barcode" else None,
        index_col=0,
    )
    positions.columns = [
        "in_tissue",
        "array_row",
        "array_col",
        "pxl_col_in_fullres",
        "pxl_row_in_fullres",
    ]
    positions.index = positions.index.astype(str)
    if positions.index.duplicated().any():
        raise ValueError(f"Duplicate barcodes in {positions_file}")

    aligned = positions.reindex(adata.obs_names.astype(str))
    coordinate_columns = ["pxl_row_in_fullres", "pxl_col_in_fullres"]
    if aligned[coordinate_columns].isna().to_numpy().any():
        missing = aligned.index[aligned[coordinate_columns].isna().any(axis=1)].tolist()
        raise ValueError(
            f"{len(missing)} matrix barcodes lack spatial coordinates in {positions_file}; "
            f"examples: {missing[:3]}"
        )
    for column in ["in_tissue", "array_row", "array_col"]:
        adata.obs[column] = aligned[column].to_numpy()
    adata.obsm["spatial"] = aligned[coordinate_columns].to_numpy(dtype=float)
    return adata


def prepare_single_cell_reference(
    h5ad: Path,
    output: Path,
    *,
    cell_type_key: str,
    cell_state_key: str | None = None,
    counts_layer: str | None = "counts",
    overwrite: bool = False,
) -> Path:
    """Export one annotated count reference for both RCTD and BayesPrism."""
    import scanpy as sc

    output = Path(output)
    expected = [
        output / "counts_genes_by_cells.mtx.gz",
        output / "genes.tsv",
        output / "cells.tsv",
    ]
    if all(path.exists() for path in expected) and not overwrite:
        return output
    marker = expected[0]
    adata = sc.read_h5ad(h5ad)
    if cell_type_key not in adata.obs:
        raise KeyError(f"Reference obs lacks {cell_type_key!r}")
    genes = pd.Index(adata.var_names.astype(str).str.upper())
    if genes.duplicated().any():
        raise ValueError("Reference var_names must be unique gene symbols")
    if counts_layer is not None and counts_layer not in adata.layers:
        raise KeyError(f"Reference layers lacks requested raw-count layer {counts_layer!r}")
    counts = _counts(adata, counts_layer)
    cell_ids = pd.Index(adata.obs_names.astype(str))
    cell_type = adata.obs[cell_type_key].astype(str).to_numpy()
    if cell_state_key and cell_state_key in adata.obs:
        cell_state = adata.obs[cell_state_key].astype(str).to_numpy()
    else:
        cell_state = cell_type
    output.mkdir(parents=True, exist_ok=True)
    _write_mtx_gz(counts.T.tocsr(), marker)
    _write_lines(genes, output / "genes.tsv")
    pd.DataFrame(
        {
            "cell_id": cell_ids,
            "cell_type": cell_type,
            "cell_state": cell_state,
            "nUMI": np.asarray(counts.sum(axis=1)).ravel(),
        }
    ).to_csv(output / "cells.tsv", sep="\t", index=False)
    return output


def prepare_visium_rctd_inputs(
    st_root: Path,
    samples: list[str],
    output: Path,
    *,
    overwrite: bool = False,
) -> Path:
    """Convert Space Ranger outputs to compact RCTD inputs and cached h5ad files."""
    output = Path(output)
    for sample in samples:
        destination = output / sample
        expected = [
            destination / "counts_genes_by_spots.mtx.gz",
            destination / "genes.tsv",
            destination / "spots.tsv",
            destination / "visium.h5ad",
        ]
        if all(path.exists() for path in expected) and not overwrite:
            continue
        marker = expected[0]
        source = Path(st_root) / sample
        if not (source / "filtered_feature_bc_matrix.h5").exists():
            raise FileNotFoundError(f"Missing Space Ranger matrix: {source}")
        adata = _read_visium_counts_and_coordinates(source)
        adata.var_names = adata.var_names.astype(str).str.upper()
        if adata.var_names.duplicated().any():
            adata.var_names_make_unique()
        counts = _counts(adata, None)
        destination.mkdir(parents=True, exist_ok=True)
        _write_mtx_gz(counts.T.tocsr(), marker)
        _write_lines(adata.var_names, destination / "genes.tsv")
        coordinates = np.asarray(adata.obsm["spatial"], dtype=float)
        pd.DataFrame(
            {
                "spot_id": adata.obs_names.astype(str),
                "x": coordinates[:, 0],
                "y": coordinates[:, 1],
                "nUMI": np.asarray(counts.sum(axis=1)).ravel(),
            }
        ).to_csv(destination / "spots.tsv", sep="\t", index=False)
        adata.write_h5ad(destination / "visium.h5ad", compression="gzip")
    return output


def run_rctd(
    project_root: Path,
    reference_dir: Path,
    spatial_dir: Path,
    output_dir: Path,
    samples: list[str],
    *,
    cores: int,
    seed: int = 20260730,
    rscript: str | Path = "Rscript",
    overwrite: bool = False,
) -> None:
    def result_complete(sample: str) -> bool:
        destination = Path(output_dir) / sample
        expected = [
            destination / "weights.tsv.gz",
            destination / "rctd.rds",
            destination / "audit.tsv",
        ]
        if not all(path.exists() and path.stat().st_size > 0 for path in expected):
            return False
        try:
            weights = pd.read_csv(expected[0], sep="\t")
            values = weights.iloc[:, 1:].to_numpy(float)
            return (
                len(weights) > 0
                and weights.spot_id.is_unique
                and np.isfinite(values).all()
                and np.allclose(values.sum(axis=1), 1.0, atol=1e-6)
            )
        except Exception:
            return False

    for sample in samples:
        if result_complete(sample) and not overwrite:
            print(f"RCTD already complete: {sample}", flush=True)
            continue
        print(f"RCTD starting: {sample}", flush=True)
        command = [
            str(rscript),
            str(Path(project_root) / "scripts" / "run_rctd.R"),
            str(reference_dir),
            str(spatial_dir),
            str(output_dir),
            str(cores),
            str(seed),
            sample,
        ]
        try:
            subprocess.run(command, check=True)
        except subprocess.CalledProcessError:
            # Some spacexr/snow builds can fail while tearing down workers after
            # all validated files have already been atomically published.
            if not result_complete(sample):
                raise
            print(
                f"RCTD worker cleanup returned non-zero after valid output: {sample}",
                flush=True,
            )
        if not result_complete(sample):
            raise RuntimeError(f"RCTD finished without a valid result for {sample}")
        print(f"RCTD complete: {sample}", flush=True)


def _entity_genes(entity: str, delimiter: str = "_") -> list[str]:
    return [value.upper() for value in str(entity).split(delimiter) if value]


def _geometric_expression(matrix: np.ndarray, indices: list[int]) -> np.ndarray:
    values = np.maximum(matrix[:, indices], 0.0)
    if len(indices) == 1:
        return values[:, 0]
    return np.exp(np.log(values + 1e-8).mean(axis=1))


def _cellchat_lr() -> pd.DataFrame:
    import numpy as _np

    if not hasattr(_np, "Inf"):
        _np.Inf = _np.inf  # COMMOT 0.0.3 compatibility with NumPy >= 2
    import commot as ct

    frames = []
    for signaling_type in ("Secreted Signaling", "Cell-Cell Contact", "ECM-Receptor"):
        frame = ct.pp.ligand_receptor_database(
            database="CellChat", species="human", signaling_type=signaling_type
        ).copy()
        frame.columns = ["ligand", "receptor", "pathway", "signaling_type"]
        frames.append(frame)
    lr = pd.concat(frames, ignore_index=True).drop_duplicates(
        ["ligand", "receptor", "pathway"]
    )
    lr["lr_id"] = lr["ligand"] + "--" + lr["receptor"] + "|" + lr["pathway"]
    return lr.reset_index(drop=True)


def build_commot_anchor(
    visium_inputs: Path,
    rctd_output: Path,
    samples: list[str],
    output: Path,
    *,
    minimum_st_samples: int = 4,
    chunk_size: int = 100,
    distance_threshold: float | None = None,
    cot_eps_p: float = 0.1,
    cot_rho: float = 10.0,
    cot_nitermax: int = 10000,
    overwrite: bool = False,
) -> Path:
    """Build W_ST and M_ST from the current study ST only (RCTD + COMMOT)."""
    import anndata as ad
    import numpy as _np

    if not hasattr(_np, "Inf"):
        _np.Inf = _np.inf
    import commot as ct

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if chunk_size < 1:
        raise ValueError("COMMOT chunk_size must be positive")
    missing_inputs = [
        str(path)
        for sample in samples
        for path in (
            Path(visium_inputs) / sample / "visium.h5ad",
            Path(rctd_output) / sample / "weights.tsv.gz",
        )
        if not path.exists()
    ]
    if missing_inputs:
        raise FileNotFoundError(
            "COMMOT preflight found missing inputs: " + ", ".join(missing_inputs)
        )
    lr_all = _cellchat_lr()
    sample_frames = []
    for sample in samples:
        cache = output / f"{sample}.parquet"
        if cache.exists() and not overwrite:
            sample_frames.append(pd.read_parquet(cache))
            continue
        adata = ad.read_h5ad(Path(visium_inputs) / sample / "visium.h5ad")
        weights = pd.read_csv(Path(rctd_output) / sample / "weights.tsv.gz", sep="\t")
        if weights.spot_id.duplicated().any():
            raise ValueError(f"Duplicate RCTD spot IDs for {sample}")
        weights = weights.set_index("spot_id")
        common_spots = adata.obs_names.intersection(weights.index)
        if len(common_spots) < 10:
            raise ValueError(f"Too few shared Visium/RCTD spots for {sample}: {len(common_spots)}")
        adata = adata[common_spots].copy()
        weights = weights.loc[common_spots]
        probabilities = weights.to_numpy(float)
        if not np.isfinite(probabilities).all() or (probabilities < 0).any():
            raise ValueError(f"Invalid RCTD probabilities for {sample}")
        probabilities /= np.maximum(probabilities.sum(axis=1, keepdims=True), 1e-12)
        cell_types = weights.columns.astype(str).tolist()
        genes = pd.Index(adata.var_names.astype(str).str.upper())
        full_gene_index = {gene: index for index, gene in enumerate(genes)}
        keep_lr = lr_all[
            lr_all.apply(
                lambda row: all(g in full_gene_index for g in _entity_genes(row.ligand))
                and all(g in full_gene_index for g in _entity_genes(row.receptor)),
                axis=1,
            )
        ].reset_index(drop=True)
        if keep_lr.empty:
            raise ValueError(f"No CellChat ligand-receptor pairs overlap {sample}")
        required_genes = sorted(
            {
                gene
                for entity in pd.concat([keep_lr.ligand, keep_lr.receptor])
                for gene in _entity_genes(entity)
            }
        )
        required_columns = [full_gene_index[gene] for gene in required_genes]
        raw_counts = _counts(adata, None)
        totals = np.maximum(np.asarray(raw_counts.sum(axis=1)).ravel(), 1.0)
        counts = raw_counts[:, required_columns].toarray().astype(np.float64)
        cpm = counts / totals[:, None] * 1e6
        commot_expression = np.log1p(cpm)
        gene_index = {gene: index for index, gene in enumerate(required_genes)}
        spatial = np.asarray(adata.obsm["spatial"], float)
        threshold = distance_threshold
        if threshold is None:
            nearest = NearestNeighbors(n_neighbors=2).fit(spatial).kneighbors(spatial)[0][:, 1]
            threshold = float(np.median(nearest[nearest > 0]) * 1.25)
        if not np.isfinite(threshold) or threshold <= 0:
            raise ValueError(f"Invalid COMMOT distance threshold for {sample}: {threshold}")
        support = radius_neighbors_graph(
            spatial, radius=float(threshold), mode="connectivity", include_self=False
        ).tocsr()
        opportunity = np.asarray(probabilities.T @ support @ probabilities)
        chunk_dir = output / "chunks" / sample
        chunk_dir.mkdir(parents=True, exist_ok=True)
        chunk_frames = []
        for start in range(0, len(keep_lr), chunk_size):
            lr = keep_lr.iloc[start : start + chunk_size].copy()
            chunk_cache = chunk_dir / f"lr_{start:05d}_{start + len(lr):05d}.parquet"
            if chunk_cache.exists() and not overwrite:
                chunk_frames.append(pd.read_parquet(chunk_cache))
                continue
            print(
                f"COMMOT {sample}: LR {start + 1}-{start + len(lr)}/{len(keep_lr)}",
                flush=True,
            )
            rows = []
            entities = sorted(set(lr.ligand) | set(lr.receptor))
            entity_ids = {name: f"E{i:05d}" for i, name in enumerate(entities)}
            commot_values = np.column_stack(
                [
                    _geometric_expression(
                        commot_expression, [gene_index[g] for g in _entity_genes(name)]
                    )
                    for name in entities
                ]
            )
            cpm_values = np.column_stack(
                [
                    _geometric_expression(cpm, [gene_index[g] for g in _entity_genes(name)])
                    for name in entities
                ]
            )
            entity_adata = ad.AnnData(X=sp.csr_matrix(commot_values))
            entity_adata.obs_names = adata.obs_names.copy()
            entity_adata.var_names = [entity_ids[name] for name in entities]
            entity_adata.obsm["spatial"] = spatial
            commot_lr = pd.DataFrame(
                {
                    "ligand": lr.ligand.map(entity_ids),
                    "receptor": lr.receptor.map(entity_ids),
                    "pathway": lr.lr_id,
                }
            )
            database_name = "v0"
            ct.tl.spatial_communication(
                entity_adata, database_name=database_name, df_ligrec=commot_lr,
                pathway_sum=False, heteromeric=False, dis_thr=float(threshold),
                cot_eps_p=cot_eps_p, cot_rho=cot_rho, cot_nitermax=cot_nitermax,
            )
            denominators = probabilities.sum(axis=0)
            mean_cpm = np.divide(
                probabilities.T @ cpm_values,
                denominators[:, None],
                out=np.zeros((len(cell_types), len(entities))),
                where=denominators[:, None] > 0,
            )
            entity_column = {name: i for i, name in enumerate(entities)}
            for item in lr.itertuples(index=False):
                ligand_id = entity_ids[item.ligand]
                receptor_id = entity_ids[item.receptor]
                gamma = entity_adata.obsp[f"commot-{database_name}-{ligand_id}-{receptor_id}"]
                numerator = np.asarray(probabilities.T @ gamma @ probabilities)
                w = np.divide(
                    numerator, opportunity, out=np.full_like(numerator, np.nan),
                    where=opportunity > 0,
                )
                ligand_mean = mean_cpm[:, entity_column[item.ligand]]
                receptor_mean = mean_cpm[:, entity_column[item.receptor]]
                m = np.sqrt(np.maximum(ligand_mean[:, None], 0) * np.maximum(receptor_mean[None, :], 0))
                for ai, sender in enumerate(cell_types):
                    for bi, receiver in enumerate(cell_types):
                        rows.append(
                            {
                                "sample_id": sample, "sender": sender, "receiver": receiver,
                                "lr_id": item.lr_id, "ligand": item.ligand,
                                "receptor": item.receptor, "pathway": item.pathway,
                                "signaling_type": item.signaling_type,
                                "W_st_sample": w[ai, bi], "M_st_sample": m[ai, bi],
                                "confident": bool(opportunity[ai, bi] > 0),
                            }
                        )
            chunk_frame = pd.DataFrame(rows)
            temporary_chunk = chunk_cache.with_suffix(chunk_cache.suffix + ".tmp")
            chunk_frame.to_parquet(temporary_chunk, index=False)
            temporary_chunk.replace(chunk_cache)
            chunk_frames.append(chunk_frame)
        frame = pd.concat(chunk_frames, ignore_index=True)
        temporary_cache = cache.with_suffix(cache.suffix + ".tmp")
        frame.to_parquet(temporary_cache, index=False)
        temporary_cache.replace(cache)
        sample_frames.append(frame)
    per_sample = pd.concat(sample_frames, ignore_index=True)
    per_sample.to_parquet(output / "anchor_per_sample.parquet", index=False)
    keys = ["sender", "receiver", "lr_id", "ligand", "receptor", "pathway", "signaling_type"]
    anchor = (
        per_sample.groupby(keys, observed=True)
        .agg(
            W_st=("W_st_sample", "median"), M_st=("M_st_sample", "median"),
            n_st_samples=("sample_id", "nunique"), n_confident=("confident", "sum"),
        )
        .reset_index()
    )
    anchor["confident"] = anchor.n_confident.ge(minimum_st_samples)
    anchor_path = output / "st_anchor.parquet"
    temporary_anchor = anchor_path.with_suffix(anchor_path.suffix + ".tmp")
    anchor.to_parquet(temporary_anchor, index=False)
    lr_genes_path = output / "lr_genes.txt"
    temporary_lr_genes = lr_genes_path.with_suffix(lr_genes_path.suffix + ".tmp")
    _write_lines(
        sorted({g for value in pd.concat([anchor.ligand, anchor.receptor]) for g in _entity_genes(value)}),
        temporary_lr_genes,
    )
    temporary_lr_genes.replace(lr_genes_path)
    # The anchor is the completion marker and is published last.
    temporary_anchor.replace(anchor_path)
    return anchor_path


def build_commot_anchor_parallel(
    visium_inputs: Path,
    rctd_output: Path,
    samples: list[str],
    output: Path,
    *,
    n_jobs: int = 1,
    minimum_st_samples: int = 4,
    chunk_size: int = 100,
    distance_threshold: float | None = None,
    cot_eps_p: float = 0.1,
    cot_rho: float = 10.0,
    cot_nitermax: int = 10000,
    overwrite: bool = False,
) -> Path:
    """Run independent ST slices concurrently, then build one shared anchor.

    Each worker writes to its own directory, so an interrupted or concurrent run
    cannot publish a partially combined anchor. Completed per-slice parquet files
    are copied atomically into ``output`` before the serial aggregation pass.
    """
    from joblib import Parallel, delayed

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    n_jobs = min(max(int(n_jobs), 1), len(samples))
    if n_jobs == 1:
        return build_commot_anchor(
            visium_inputs, rctd_output, samples, output,
            minimum_st_samples=minimum_st_samples, chunk_size=chunk_size,
            distance_threshold=distance_threshold, cot_eps_p=cot_eps_p,
            cot_rho=cot_rho, cot_nitermax=cot_nitermax, overwrite=overwrite,
        )

    worker_root = output / "_slice_workers"
    worker_root.mkdir(parents=True, exist_ok=True)
    # Reuse checkpoints made by an earlier serial run.
    for sample in samples:
        source_chunks = output / "chunks" / sample
        target_chunks = worker_root / sample / "chunks" / sample
        if source_chunks.exists() and not target_chunks.exists():
            target_chunks.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(source_chunks, target_chunks)

    def run_one(sample: str) -> str:
        result = build_commot_anchor(
            visium_inputs, rctd_output, [sample], worker_root / sample,
            minimum_st_samples=1, chunk_size=chunk_size,
            distance_threshold=distance_threshold, cot_eps_p=cot_eps_p,
            cot_rho=cot_rho, cot_nitermax=cot_nitermax, overwrite=overwrite,
        )
        return str(result)

    Parallel(n_jobs=n_jobs, backend="loky", verbose=10)(
        delayed(run_one)(sample) for sample in samples
    )
    for sample in samples:
        source = worker_root / sample / f"{sample}.parquet"
        if not source.exists():
            raise FileNotFoundError(f"Missing completed COMMOT slice cache: {source}")
        destination = output / f"{sample}.parquet"
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        shutil.copyfile(source, temporary)
        temporary.replace(destination)

    # All per-slice caches now exist, so this pass only validates/aggregates them.
    return build_commot_anchor(
        visium_inputs, rctd_output, samples, output,
        minimum_st_samples=minimum_st_samples, chunk_size=chunk_size,
        distance_threshold=distance_threshold, cot_eps_p=cot_eps_p,
        cot_rho=cot_rho, cot_nitermax=cot_nitermax, overwrite=False,
    )


def prepare_tcga_bulk_counts(
    raw_root: Path,
    file_map: Path,
    reference_dir: Path,
    output: Path,
    *,
    sample_type: str = "Primary Tumor",
    overwrite: bool = False,
) -> Path:
    """Create a patient-level raw-count mixture matrix for BayesPrism."""
    output = Path(output)
    expected = [
        output / "counts_samples_by_genes.mtx.gz",
        output / "genes.tsv",
        output / "samples.tsv",
    ]
    if all(path.exists() for path in expected) and not overwrite:
        return output
    marker = expected[0]
    reference_genes = set(pd.read_csv(Path(reference_dir) / "genes.tsv", header=None)[0].astype(str))
    mapping = pd.read_csv(file_map, sep="\t")
    mapping = mapping[mapping.sample_type.eq(sample_type)].copy()
    patient_values: dict[str, list[pd.Series]] = {}
    for row in mapping.itertuples(index=False):
        candidates = [Path(raw_root) / f"{row.file_id}.tsv", Path(raw_root) / row.file_name]
        source = next((path for path in candidates if path.exists()), None)
        if source is None:
            raise FileNotFoundError(f"No raw count file for {row.file_id}")
        frame = pd.read_csv(source, sep="\t", skiprows=1, usecols=["gene_name", "unstranded"])
        frame = frame[frame.gene_name.notna() & frame.gene_name.astype(str).isin(reference_genes)]
        values = frame.groupby(frame.gene_name.astype(str), observed=True).unstranded.sum()
        patient_values.setdefault(str(row.patient_id), []).append(values)
    genes = sorted(set.intersection(*(set(v.index) for values in patient_values.values() for v in values)))
    patients = sorted(patient_values)
    matrix = np.vstack(
        [
            np.mean([values.reindex(genes).to_numpy(float) for values in patient_values[patient]], axis=0)
            for patient in patients
        ]
    )
    matrix = np.rint(matrix).astype(np.int64)
    output.mkdir(parents=True, exist_ok=True)
    _write_mtx_gz(sp.csr_matrix(matrix), marker)
    _write_lines(genes, output / "genes.tsv")
    _write_lines(patients, output / "samples.tsv")
    return output


def run_bayesprism(
    project_root: Path,
    reference_dir: Path,
    bulk_dir: Path,
    output_dir: Path,
    lr_genes: Path,
    *,
    malignant_label: str,
    cores: int,
    chain_length: int,
    burn_in: int,
    seed: int,
    rscript: str | Path = "Rscript",
) -> None:
    subprocess.run(
        [
            str(rscript), str(Path(project_root) / "scripts" / "run_bayesprism.R"),
            str(reference_dir), str(bulk_dir), str(output_dir), str(lr_genes),
            malignant_label, str(cores), str(chain_length), str(burn_in), str(seed),
        ],
        check=True,
    )


def build_patient_tensor(
    anchor_path: Path,
    bayesprism_dir: Path,
    clinical_path: Path,
    output: Path,
    *,
    minimum_cell_fraction: float = 0.001,
    epsilon: float = 1e-8,
    overwrite: bool = False,
) -> Path:
    """Build and save sample x sender x receiver x LR exact CCC propensity."""
    import zarr

    output = Path(output)
    success = output / "_SUCCESS"
    expected = [
        success,
        output / "C_propensity.zarr",
        output / "samples.tsv",
        output / "cell_types.tsv",
        output / "lr_pairs.tsv",
        output / "audit.json",
    ]
    if all(path.exists() for path in expected) and not overwrite:
        return output

    anchor = pd.read_parquet(anchor_path)
    anchor = anchor[anchor.confident].copy()
    file_map = pd.read_csv(Path(bayesprism_dir) / "cell_type_files.tsv", sep="\t")
    expression = {
        row.cell_type: pd.read_csv(Path(bayesprism_dir) / row.file, sep="\t").set_index("Sample_ID")
        for row in file_map.itertuples(index=False)
    }
    clinical = pd.read_csv(clinical_path, sep="\t").set_index("Sample_ID")
    fractions = pd.read_csv(Path(bayesprism_dir) / "fractions_final.tsv.gz", sep="\t").set_index("Sample_ID")
    samples = sorted(set(clinical.index) & set(fractions.index) & set.intersection(*(set(x.index) for x in expression.values())))
    if not samples:
        raise ValueError("No patients are shared by clinical, fractions, and expression outputs")
    cell_types = sorted(set(anchor.sender) | set(anchor.receiver))
    missing_types = set(cell_types) - set(expression)
    if missing_types:
        raise ValueError(f"BayesPrism lacks cell types required by ST anchor: {sorted(missing_types)}")
    missing_fraction_types = set(cell_types) - set(fractions.columns)
    if missing_fraction_types:
        raise ValueError(
            f"BayesPrism fractions lack ST cell types: {sorted(missing_fraction_types)}"
        )
    input_anchor_rows = len(anchor)
    available_genes = {name: set(frame.columns) for name, frame in expression.items()}
    supported = anchor.apply(
        lambda row: set(_entity_genes(row.ligand)).issubset(available_genes[row.sender])
        and set(_entity_genes(row.receptor)).issubset(available_genes[row.receiver]),
        axis=1,
    )
    anchor = anchor[supported].copy()
    if anchor.empty:
        raise ValueError("No confident ST CCC has complete BayesPrism ligand/receptor expression")
    if not np.isfinite(anchor[["W_st", "M_st"]].to_numpy(float)).all():
        raise ValueError("The retained ST anchor contains non-finite W_ST or M_ST values")
    interactions = anchor[["lr_id", "ligand", "receptor", "pathway", "signaling_type"]].drop_duplicates("lr_id")
    interactions = interactions.sort_values("lr_id").reset_index(drop=True)
    ct_index = {name: i for i, name in enumerate(cell_types)}
    lr_index = {name: i for i, name in enumerate(interactions.lr_id)}
    shape = (len(samples), len(cell_types), len(cell_types), len(interactions))
    w_st = np.zeros(shape[1:], dtype=np.float64)
    m_st = np.zeros(shape[1:], dtype=np.float64)
    m_rna = np.zeros(shape, dtype=np.float64)
    structural = np.zeros(shape[1:], dtype=bool)
    cache: dict[tuple[str, str], np.ndarray] = {}

    def entity_values(cell_type: str, entity: str) -> np.ndarray:
        key = (cell_type, entity)
        if key not in cache:
            frame = expression[cell_type].loc[samples]
            genes = _entity_genes(entity)
            missing = set(genes) - set(frame.columns)
            if missing:
                raise ValueError(f"Missing BayesPrism genes for {cell_type}/{entity}: {sorted(missing)}")
            values = np.maximum(frame[genes].to_numpy(float), 0)
            cache[key] = np.exp(np.log(values + epsilon).mean(axis=1))
        return cache[key]

    for row in anchor.itertuples(index=False):
        a, b, p = ct_index[row.sender], ct_index[row.receiver], lr_index[row.lr_id]
        w_st[a, b, p] = row.W_st
        m_st[a, b, p] = row.M_st
        m_rna[:, a, b, p] = np.sqrt(
            entity_values(row.sender, row.ligand) * entity_values(row.receiver, row.receptor)
        )
        structural[a, b, p] = True
    fraction_values = fractions.loc[samples].reindex(columns=cell_types).fillna(0).to_numpy(float)
    cell_observed = fraction_values >= minimum_cell_fraction
    mask = structural[None, ...] & cell_observed[:, :, None, None] & cell_observed[:, None, :, None]
    ccc = construct_patient_ccc(w_st, m_st, m_rna, epsilon=epsilon)
    ccc[~mask] = 0
    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)
    root = zarr.open_group(str(output / "C_propensity.zarr"), mode="w")
    chunks = (min(32, len(samples)), 1, 1, min(128, len(interactions)))
    root.create_array("C", data=ccc, chunks=chunks)
    root.create_array("mask", data=mask, chunks=chunks)
    root.create_array("W_st", data=w_st.astype(np.float32))
    root.create_array("M_st", data=m_st.astype(np.float32))
    root.attrs.update(
        axis_order=["sample", "sender", "receiver", "lr_pair"],
        formula="W_ST * (M_RNA + epsilon) / (M_ST + epsilon)",
        st_rule="current study ST is the only spatial anchor",
        epsilon=epsilon,
    )
    pd.DataFrame({"Sample_ID": samples}).to_csv(output / "samples.tsv", sep="\t", index=False)
    pd.DataFrame({"cell_type": cell_types}).to_csv(output / "cell_types.tsv", sep="\t", index=False)
    lr_meta = interactions.rename(
        columns={"ligand": "ligand_expression_complex", "receptor": "receptor_expression_complex"}
    )
    lr_meta["ligand_complex"] = lr_meta.ligand_expression_complex
    lr_meta["receptor_complex"] = lr_meta.receptor_expression_complex
    lr_meta["pair_key"] = lr_meta.ligand_expression_complex + "--" + lr_meta.receptor_expression_complex
    lr_meta.to_csv(output / "lr_pairs.tsv", sep="\t", index=False)
    audit = {
        "shape": list(shape), "samples": len(samples), "cell_types": len(cell_types),
        "lr_pairs": len(interactions), "observed_fraction": float(mask.mean()),
        "confident_anchor_rows_input": input_anchor_rows,
        "anchor_rows_with_bulk_expression": len(anchor),
        "anchor_rows_dropped_missing_bulk_expression": input_anchor_rows - len(anchor),
        "formula": root.attrs["formula"], "st_rule": root.attrs["st_rule"],
    }
    (output / "audit.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    success.write_text("complete\n", encoding="utf-8")
    return output
