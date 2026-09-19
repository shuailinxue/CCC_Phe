"""Post-hoc discovery and TLS validation of protective ST communication niches.

This module is deliberately downstream of the V0 bulk/Cox mainline.  Discovery
uses only protective stable CCCs, Visium expression/coordinates, and RCTD
weights.  TLS labels are accepted only by :func:`validate_tls_niches`, after all
NMF programs and spatial hotspots have been fixed and written to disk.
"""
from __future__ import annotations

import json
import hashlib
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.optimize import linear_sum_assignment
from scipy.sparse.csgraph import connected_components
from scipy.stats import fisher_exact, mannwhitneyu
from sklearn.decomposition import NMF
from sklearn.metrics import roc_auc_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors, radius_neighbors_graph

from .workflow import _counts, _entity_genes, _geometric_expression


CCC_KEYS = ["sender", "receiver", "lr_id"]


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _feature_id(frame: pd.DataFrame) -> pd.Series:
    return frame["sender"].astype(str) + "->" + frame["receiver"].astype(str) + "|" + frame["lr_id"].astype(str)


def load_protective_cccs(
    stable_ccc: str | Path,
    *,
    stability_threshold: float = 0.70,
) -> pd.DataFrame:
    """Select stable, negative-coefficient CCCs without biological pre-filtering."""
    frame = pd.read_csv(stable_ccc, sep="\t")
    required = set(CCC_KEYS + ["ligand", "receptor", "full_coefficient", "sign_selection_probability"])
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"stable_ccc.tsv lacks required columns: {missing}")
    selected = frame.loc[
        frame["sign_selection_probability"].ge(float(stability_threshold))
        & frame["full_coefficient"].lt(0)
    ].copy()
    if selected.empty:
        raise ValueError("No stable protective CCC passes the requested threshold")
    selected["ccc_id"] = _feature_id(selected)
    if selected["ccc_id"].duplicated().any():
        raise ValueError("Protective CCC identifiers are not unique")
    selected = selected.sort_values(
        ["sign_selection_probability", "stability_importance"], ascending=False
    ).reset_index(drop=True)
    return selected


def _distance_and_graph(coordinates: np.ndarray, distance_threshold: float | None):
    if len(coordinates) < 2:
        raise ValueError("At least two spatial locations are required")
    threshold = distance_threshold
    if threshold is None:
        nearest = NearestNeighbors(n_neighbors=2).fit(coordinates).kneighbors(coordinates)[0][:, 1]
        threshold = float(np.median(nearest[nearest > 0]) * 1.25)
    if not np.isfinite(threshold) or threshold <= 0:
        raise ValueError(f"Invalid spatial distance threshold: {threshold}")
    graph = radius_neighbors_graph(
        coordinates, radius=float(threshold), mode="connectivity", include_self=False
    ).tocsr()
    graph = graph.maximum(graph.T).tocsr()
    return float(threshold), graph


def _commot_selected_lr(
    expression: np.ndarray,
    genes: list[str],
    coordinates: np.ndarray,
    lr: pd.DataFrame,
    *,
    distance_threshold: float,
    cot_eps_p: float,
    cot_rho: float,
    cot_nitermax: int,
) -> dict[str, sp.csr_matrix]:
    import anndata as ad

    # COMMOT 0.0.3 still refers to the removed NumPy alias.
    if not hasattr(np, "Inf"):
        np.Inf = np.inf
    import commot as ct

    entities = sorted(set(lr["ligand"].astype(str)) | set(lr["receptor"].astype(str)))
    entity_ids = {name: f"E{i:03d}" for i, name in enumerate(entities)}
    gene_index = {gene: index for index, gene in enumerate(genes)}
    values = np.column_stack(
        [
            _geometric_expression(expression, [gene_index[g] for g in _entity_genes(name)])
            for name in entities
        ]
    )
    adata = ad.AnnData(X=sp.csr_matrix(values))
    adata.var_names = [entity_ids[name] for name in entities]
    adata.obsm["spatial"] = coordinates
    commot_lr = pd.DataFrame(
        {
            "ligand": lr["ligand"].map(entity_ids),
            "receptor": lr["receptor"].map(entity_ids),
            "pathway": lr["lr_id"],
        }
    )
    database_name = "protective"
    ct.tl.spatial_communication(
        adata,
        database_name=database_name,
        df_ligrec=commot_lr,
        pathway_sum=False,
        heteromeric=False,
        dis_thr=float(distance_threshold),
        cot_eps_p=float(cot_eps_p),
        cot_rho=float(cot_rho),
        cot_nitermax=int(cot_nitermax),
    )
    output = {}
    for row in lr.itertuples(index=False):
        key = f"commot-{database_name}-{entity_ids[row.ligand]}-{entity_ids[row.receptor]}"
        output[row.lr_id] = adata.obsp[key].tocsr()
    return output


def compute_spatial_ccc_activity(
    visium_h5ad: str | Path,
    rctd_weights: str | Path,
    protective_cccs: pd.DataFrame,
    *,
    sample_id: str,
    distance_threshold: float | None = None,
    cot_eps_p: float = 0.1,
    cot_rho: float = 10.0,
    cot_nitermax: int = 10000,
) -> tuple[pd.DataFrame, pd.DataFrame, sp.csr_matrix]:
    """Compute exact directed CCC mass at each spot from selected COMMOT flows.

    For route ``a -> b, LR``, each COMMOT edge is weighted by the RCTD sender
    probability at its source and receiver probability at its target.  Half of
    the directed edge mass is assigned to each endpoint, yielding one local CCC
    activity per Visium spot while retaining the route direction.
    """
    import anndata as ad

    adata = ad.read_h5ad(visium_h5ad)
    weights = pd.read_csv(rctd_weights, sep="\t")
    if "spot_id" not in weights:
        raise ValueError(f"RCTD table lacks spot_id: {rctd_weights}")
    weights["spot_id"] = weights["spot_id"].astype(str)
    weights = weights.set_index("spot_id")
    common = adata.obs_names.astype(str).intersection(weights.index)
    if len(common) < 10:
        raise ValueError(f"Only {len(common)} shared Visium/RCTD spots in {sample_id}")
    adata = adata[common].copy()
    weights = weights.loc[common]
    probabilities = weights.to_numpy(float)
    if not np.isfinite(probabilities).all() or (probabilities < 0).any():
        raise ValueError(f"Invalid RCTD probabilities in {sample_id}")
    probabilities /= np.maximum(probabilities.sum(axis=1, keepdims=True), 1e-12)
    cell_index = {name: i for i, name in enumerate(weights.columns.astype(str))}

    missing_cell_types = sorted(
        (set(protective_cccs.sender.astype(str)) | set(protective_cccs.receiver.astype(str)))
        .difference(cell_index)
    )
    if missing_cell_types:
        raise ValueError(f"RCTD lacks CCC cell types in {sample_id}: {missing_cell_types}")

    genes = pd.Index(adata.var_names.astype(str).str.upper())
    if genes.duplicated().any():
        raise ValueError(f"Duplicated uppercase gene symbols in {sample_id}")
    gene_lookup = {gene: index for index, gene in enumerate(genes)}
    required = sorted(
        {
            gene
            for entity in pd.concat([protective_cccs.ligand, protective_cccs.receptor])
            for gene in _entity_genes(entity)
        }
    )
    missing_genes = sorted(set(required).difference(gene_lookup))
    if missing_genes:
        raise ValueError(f"Visium data lack protective LR genes in {sample_id}: {missing_genes}")
    columns = [gene_lookup[g] for g in required]
    raw = _counts(adata, None)
    totals = np.maximum(np.asarray(raw.sum(axis=1)).ravel(), 1.0)
    cpm = raw[:, columns].toarray().astype(np.float64) / totals[:, None] * 1e6
    expression = np.log1p(cpm)
    coordinates = np.asarray(adata.obsm["spatial"], dtype=float)
    threshold, graph = _distance_and_graph(coordinates, distance_threshold)

    lr = protective_cccs[["ligand", "receptor", "lr_id"]].drop_duplicates().reset_index(drop=True)
    gamma = _commot_selected_lr(
        expression,
        required,
        coordinates,
        lr,
        distance_threshold=threshold,
        cot_eps_p=cot_eps_p,
        cot_rho=cot_rho,
        cot_nitermax=cot_nitermax,
    )
    activity = np.zeros((len(common), len(protective_cccs)), dtype=np.float64)
    for column, row in enumerate(protective_cccs.itertuples(index=False)):
        source = probabilities[:, cell_index[str(row.sender)]]
        target = probabilities[:, cell_index[str(row.receiver)]]
        route_edges = gamma[str(row.lr_id)].multiply(source[:, None]).multiply(target[None, :])
        outgoing = np.asarray(route_edges.sum(axis=1)).ravel()
        incoming = np.asarray(route_edges.sum(axis=0)).ravel()
        activity[:, column] = 0.5 * (outgoing + incoming)

    activity_frame = pd.DataFrame(activity, index=common, columns=protective_cccs.ccc_id)
    activity_frame.index.name = "spot_id"
    spot_frame = pd.DataFrame(
        {
            "sample_id": sample_id,
            "spot_id": common,
            "x": coordinates[:, 0],
            "y": coordinates[:, 1],
            "n_umi": totals,
            "distance_threshold": threshold,
        }
    )
    return activity_frame, spot_frame, graph


def robust_nonnegative_standardize(
    raw_activity: pd.DataFrame,
    sample_ids: Sequence[str],
    *,
    upper_clip: float = 1.5,
    minimum_positive_spots: int = 10,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Log/quantile-scale each CCC within slice while preserving non-negativity."""
    values = raw_activity.to_numpy(float)
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("CCC activities must be finite and non-negative")
    samples = np.asarray(sample_ids, dtype=str)
    transformed = np.zeros_like(values)
    audit = []
    for sample in pd.unique(samples):
        rows = np.flatnonzero(samples == sample)
        for j, ccc in enumerate(raw_activity.columns):
            x = values[rows, j]
            positive = x[x > 0]
            if len(positive) < int(minimum_positive_spots):
                audit.append((sample, ccc, len(positive), np.nan, False))
                continue
            median = float(np.median(positive))
            z = np.log1p(x / max(median, np.finfo(float).tiny))
            scale = float(np.quantile(z[z > 0], 0.95))
            transformed[rows, j] = np.clip(z / max(scale, np.finfo(float).tiny), 0, upper_clip)
            audit.append((sample, ccc, len(positive), scale, True))
    usable = np.flatnonzero(np.max(transformed, axis=0) > 0)
    if len(usable) < 2:
        raise ValueError("Fewer than two protective CCCs have usable spatial activity")
    normalized = pd.DataFrame(
        transformed[:, usable], index=raw_activity.index, columns=raw_activity.columns[usable]
    )
    normalized.index.name = raw_activity.index.name
    audit_frame = pd.DataFrame(
        audit, columns=["sample_id", "ccc_id", "n_positive", "log_q95", "used_in_slice"]
    )
    audit_frame["used_globally"] = audit_frame.ccc_id.isin(normalized.columns)
    return normalized, audit_frame


def _fit_nmf_repeats(
    x: np.ndarray,
    k: int,
    seeds: Sequence[int],
    *,
    max_iter: int,
) -> tuple[list[dict], float]:
    fits = []
    for seed in seeds:
        model = NMF(
            # Random starts are intentional: otherwise NNDSVD is deterministic
            # and the requested repeat-initialization stability is always 1.
            n_components=int(k), init="random", solver="cd", beta_loss="frobenius",
            random_state=int(seed), max_iter=int(max_iter), tol=1e-4,
        )
        w = model.fit_transform(x)
        h = model.components_
        fits.append({"seed": int(seed), "W": w, "H": h, "error": float(model.reconstruction_err_)})
    similarities = np.eye(len(fits))
    for i in range(len(fits)):
        for j in range(i + 1, len(fits)):
            similarity = cosine_similarity(fits[i]["H"], fits[j]["H"])
            rows, columns = linear_sum_assignment(-similarity)
            similarities[i, j] = similarities[j, i] = float(similarity[rows, columns].mean())
    stability = float(similarities[np.triu_indices(len(fits), 1)].mean()) if len(fits) > 1 else 1.0
    medoid = int(np.argmax((similarities.sum(axis=1) - 1) / max(len(fits) - 1, 1)))
    return fits, stability, medoid


def discover_nmf_programs(
    normalized_activity: pd.DataFrame,
    *,
    k_values: Iterable[int] = range(2, 9),
    n_restarts: int = 20,
    random_seed: int = 20260730,
    max_iter: int = 2000,
    selected_k: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, int]:
    """Select K from repeat stability and normalized reconstruction error only."""
    x = normalized_activity.to_numpy(float)
    if (x < 0).any() or not np.isfinite(x).all():
        raise ValueError("NMF input must be finite and non-negative")
    ks = sorted({int(k) for k in k_values if 2 <= int(k) <= min(x.shape)})
    if not ks:
        raise ValueError("No valid NMF K for the activity matrix")
    seeds = [int(random_seed + 104729 * i) for i in range(int(n_restarts))]
    all_fits = {}
    rows = []
    norm = max(float(np.linalg.norm(x)), np.finfo(float).tiny)
    for k in ks:
        fits, stability, medoid = _fit_nmf_repeats(x, k, seeds, max_iter=max_iter)
        all_fits[k] = (fits, medoid)
        errors = np.array([fit["error"] / norm for fit in fits])
        rows.append(
            {"k": k, "stability": stability, "relative_reconstruction_error": errors.mean(),
             "reconstruction_error_sd": errors.std(ddof=1) if len(errors) > 1 else 0.0,
             "medoid_seed": fits[medoid]["seed"]}
        )
    diagnostics = pd.DataFrame(rows)
    # Balanced, transparent rank aggregation. Stability wins exact ties; smaller
    # K wins the final tie. No phenotype/TLS information enters this decision.
    diagnostics["stability_rank"] = diagnostics.stability.rank(ascending=False, method="min")
    diagnostics["error_rank"] = diagnostics.relative_reconstruction_error.rank(ascending=True, method="min")
    diagnostics["selection_rank_sum"] = diagnostics.stability_rank + diagnostics.error_rank
    if selected_k is None:
        chosen = int(
            diagnostics.sort_values(
                ["selection_rank_sum", "stability", "k"], ascending=[True, False, True]
            ).iloc[0].k
        )
    else:
        chosen = int(selected_k)
        if chosen not in all_fits:
            raise ValueError(f"selected_k={chosen} was not tested")
    diagnostics["selected"] = diagnostics.k.eq(chosen)
    fits, medoid = all_fits[chosen]
    fit = fits[medoid]
    names = [f"P{i + 1}" for i in range(chosen)]
    loadings = pd.DataFrame(fit["H"].T, index=normalized_activity.columns, columns=names)
    loadings.index.name = "ccc_id"
    # Fix NMF scale indeterminacy so activity is comparable among programs.
    scales = np.maximum(loadings.sum(axis=0).to_numpy(float), np.finfo(float).tiny)
    loadings = loadings / scales
    activities = pd.DataFrame(
        fit["W"] * scales[None, :], index=normalized_activity.index, columns=names
    )
    activities.index.name = "location_id"
    return diagnostics, loadings, activities, chosen


def call_spatial_hotspots(
    program_activity: pd.DataFrame,
    spots: pd.DataFrame,
    graphs: dict[str, sp.csr_matrix],
    *,
    hotspot_quantile: float = 0.90,
    minimum_niche_spots: int = 5,
    smoothing_self_weight: float = 1.0,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Call high-score connected components; unselected tissue remains background."""
    memberships, summaries, score_rows = [], [], []
    for sample, sample_spots in spots.groupby("sample_id", sort=False):
        positions = sample_spots.index.to_numpy()
        graph = graphs[str(sample)].tocsr()
        if graph.shape[0] != len(positions):
            raise ValueError(f"Graph/spot mismatch for {sample}")
        degree = np.asarray(graph.sum(axis=1)).ravel()
        for program in program_activity.columns:
            raw = program_activity.loc[positions, program].to_numpy(float)
            neighbor = np.asarray(graph @ raw).ravel() / np.maximum(degree, 1.0)
            smooth = (float(smoothing_self_weight) * raw + neighbor) / (
                float(smoothing_self_weight) + (degree > 0).astype(float)
            )
            threshold = float(np.quantile(smooth, hotspot_quantile))
            high = smooth > threshold
            if not high.any():
                continue
            subgraph = graph[high][:, high]
            n_components, labels = connected_components(subgraph, directed=False)
            high_positions = np.flatnonzero(high)
            score_rows.extend(
                {
                    "location_id": positions[i], "sample_id": sample, "spot_id": sample_spots.iloc[i].spot_id,
                    "program": program, "raw_activity": raw[i], "smoothed_activity": smooth[i],
                    "hotspot_threshold": threshold, "is_hotspot": bool(high[i]),
                }
                for i in range(len(positions))
            )
            for component in range(n_components):
                local = high_positions[labels == component]
                if len(local) < int(minimum_niche_spots):
                    continue
                niche_id = f"{sample}|{program}|N{component + 1}"
                for i in local:
                    memberships.append(
                        {"niche_id": niche_id, "sample_id": sample, "program": program,
                         "location_id": positions[i], "spot_id": sample_spots.iloc[i].spot_id,
                         "program_activity": raw[i], "smoothed_activity": smooth[i]}
                    )
                summaries.append(
                    {"niche_id": niche_id, "sample_id": sample, "program": program,
                     "n_spots": len(local), "mean_program_activity": float(raw[local].mean()),
                     "mean_smoothed_activity": float(smooth[local].mean()), "hotspot_threshold": threshold}
                )
    membership = pd.DataFrame(
        memberships,
        columns=["niche_id", "sample_id", "program", "location_id", "spot_id",
                 "program_activity", "smoothed_activity"],
    )
    summary = pd.DataFrame(
        summaries,
        columns=["niche_id", "sample_id", "program", "n_spots", "mean_program_activity",
                 "mean_smoothed_activity", "hotspot_threshold"],
    )
    scores = pd.DataFrame(
        score_rows,
        columns=["location_id", "sample_id", "spot_id", "program", "raw_activity",
                 "smoothed_activity", "hotspot_threshold", "is_hotspot"],
    )
    return membership, summary, scores


def _write_frame(frame: pd.DataFrame, path: Path, index: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    if path.suffix == ".parquet":
        frame.to_parquet(temporary, index=index)
    else:
        frame.to_csv(temporary, sep="\t", index=index)
    temporary.replace(path)


def run_st_niche_discovery(
    stable_ccc: str | Path,
    visium_root: str | Path,
    rctd_root: str | Path,
    samples: Sequence[str],
    output_dir: str | Path,
    *,
    stability_threshold: float = 0.70,
    k_values: Iterable[int] = range(2, 9),
    nmf_restarts: int = 20,
    nmf_max_iter: int = 2000,
    selected_k: int | None = None,
    hotspot_quantile: float = 0.90,
    minimum_niche_spots: int = 5,
    distance_threshold: float | None = None,
    cot_eps_p: float = 0.1,
    cot_rho: float = 10.0,
    cot_nitermax: int = 10000,
    random_seed: int = 20260730,
    overwrite: bool = False,
) -> dict:
    """Run label-free program/hotspot discovery and persist an auditable result."""
    output = Path(output_dir)
    completion = output / "discovery_manifest.json"
    k_values = [int(value) for value in k_values]
    parameters = {
        "stable_ccc_sha256": _sha256(stable_ccc),
        "samples": list(map(str, samples)),
        "stability_threshold": float(stability_threshold),
        "k_values": k_values,
        "nmf_restarts": int(nmf_restarts),
        "nmf_max_iter": int(nmf_max_iter),
        "selected_k": None if selected_k is None else int(selected_k),
        "hotspot_quantile": float(hotspot_quantile),
        "minimum_niche_spots": int(minimum_niche_spots),
        "distance_threshold": None if distance_threshold is None else float(distance_threshold),
        "cot_eps_p": float(cot_eps_p),
        "cot_rho": float(cot_rho),
        "cot_nitermax": int(cot_nitermax),
        "random_seed": int(random_seed),
    }
    if completion.exists() and not overwrite:
        cached = json.loads(completion.read_text(encoding="utf-8"))
        if cached.get("discovery_parameters") != parameters:
            raise ValueError(
                "Existing niche discovery was made with different inputs/parameters; "
                "set NICHE_OVERWRITE=True to recompute it"
            )
        return cached
    output.mkdir(parents=True, exist_ok=True)
    protective = load_protective_cccs(stable_ccc, stability_threshold=stability_threshold)
    _write_frame(protective, output / "protective_cccs.tsv")

    raw_frames, spot_frames, graphs = [], [], {}
    slice_cache = output / "spatial_ccc_by_slice"
    slice_cache.mkdir(parents=True, exist_ok=True)
    for sample in samples:
        raw_cache = slice_cache / f"{sample}.parquet"
        spot_cache = slice_cache / f"{sample}.spots.tsv"
        if raw_cache.exists() and spot_cache.exists() and not overwrite:
            print(f"Protective CCC spatial activity cached: {sample}", flush=True)
            raw = pd.read_parquet(raw_cache)
            if "spot_id" in raw.columns:
                raw = raw.set_index("spot_id")
            spot = pd.read_csv(spot_cache, sep="\t")
            spot.index = raw.index
            if raw.columns.tolist() != protective.ccc_id.tolist():
                raise ValueError(f"Stale protective CCC slice cache for {sample}; use overwrite=True")
            _, graph = _distance_and_graph(
                spot[["x", "y"]].to_numpy(float), float(spot.distance_threshold.iloc[0])
            )
        else:
            print(f"Protective CCC spatial activity: {sample}", flush=True)
            raw, spot, graph = compute_spatial_ccc_activity(
                Path(visium_root) / sample / "visium.h5ad",
                Path(rctd_root) / sample / "weights.tsv.gz",
                protective,
                sample_id=str(sample), distance_threshold=distance_threshold,
                cot_eps_p=cot_eps_p, cot_rho=cot_rho, cot_nitermax=cot_nitermax,
            )
            _write_frame(raw, raw_cache, index=True)
            _write_frame(spot, spot_cache)
        location_ids = pd.Index([f"{sample}|{spot_id}" for spot_id in raw.index], name="location_id")
        raw.index = location_ids
        spot.index = location_ids
        raw_frames.append(raw)
        spot_frames.append(spot)
        graphs[str(sample)] = graph
    raw_activity = pd.concat(raw_frames, axis=0)
    spots = pd.concat(spot_frames, axis=0)
    normalized, normalization_audit = robust_nonnegative_standardize(
        raw_activity, spots.sample_id.to_numpy()
    )
    diagnostics, loadings, activities, chosen_k = discover_nmf_programs(
        normalized, k_values=k_values, n_restarts=nmf_restarts,
        random_seed=random_seed, max_iter=nmf_max_iter, selected_k=selected_k,
    )
    membership, niche_summary, scores = call_spatial_hotspots(
        activities, spots, graphs, hotspot_quantile=hotspot_quantile,
        minimum_niche_spots=minimum_niche_spots,
    )
    composition = (
        loadings.reset_index().melt(id_vars="ccc_id", var_name="program", value_name="loading")
        .merge(protective, on="ccc_id", how="left", validate="many_to_one")
        .sort_values(["program", "loading"], ascending=[True, False])
    )
    composition["loading_rank"] = composition.groupby("program").cumcount() + 1

    _write_frame(raw_activity, output / "X_ST_raw.parquet", index=True)
    _write_frame(normalized, output / "X_ST_normalized.parquet", index=True)
    _write_frame(spots.reset_index(), output / "spatial_locations.tsv")
    _write_frame(normalization_audit, output / "normalization_audit.tsv")
    _write_frame(diagnostics, output / "nmf_k_diagnostics.tsv")
    _write_frame(loadings, output / "program_ccc_loadings.tsv", index=True)
    _write_frame(activities, output / "program_activity.tsv", index=True)
    _write_frame(scores, output / "program_spatial_scores.tsv")
    _write_frame(membership, output / "candidate_niche_membership.tsv")
    _write_frame(niche_summary, output / "candidate_niches.tsv")
    _write_frame(composition, output / "program_composition.tsv")
    manifest = {
        "stage": "label_free_discovery_complete",
        "stable_ccc": str(Path(stable_ccc).resolve()),
        "samples": list(map(str, samples)),
        "n_protective_cccs": int(len(protective)),
        "n_locations": int(len(spots)),
        "n_nmf_cccs": int(normalized.shape[1]),
        "selected_k": int(chosen_k),
        "n_candidate_niches": int(len(niche_summary)),
        "tls_labels_used": False,
        "discovery_parameters": parameters,
        "output_dir": str(output.resolve()),
    }
    temporary = completion.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    temporary.replace(completion)
    return manifest


def read_tls_annotations(path: str | Path) -> pd.DataFrame:
    """Read user/author TLS labels: sample_id, spot_id, and tls columns."""
    path = Path(path)
    frame = pd.read_csv(path, sep="\t" if path.suffix in {".tsv", ".gz"} else ",")
    required = {"sample_id", "spot_id", "tls"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"TLS annotation lacks columns: {missing}")
    frame = frame[["sample_id", "spot_id", "tls"]].copy()
    frame["sample_id"] = frame.sample_id.astype(str)
    frame["spot_id"] = frame.spot_id.astype(str)
    if frame.duplicated(["sample_id", "spot_id"]).any():
        raise ValueError("TLS annotation contains duplicate sample/spot rows")
    if frame.tls.dtype != bool:
        true_values = {"1", "true", "t", "yes", "y", "tls", "positive"}
        false_values = {"0", "false", "f", "no", "n", "non-tls", "negative", "background"}
        text = frame.tls.astype(str).str.strip().str.lower()
        unknown = sorted(set(text).difference(true_values | false_values))
        if unknown:
            raise ValueError(f"Unrecognized TLS labels: {unknown[:5]}")
        frame["tls"] = text.isin(true_values)
    return frame


def _effect_size(values: np.ndarray, labels: np.ndarray) -> float:
    positive, negative = values[labels], values[~labels]
    if len(positive) < 2 or len(negative) < 2:
        return np.nan
    pooled = np.sqrt(((len(positive) - 1) * positive.var(ddof=1) + (len(negative) - 1) * negative.var(ddof=1)) /
                     max(len(positive) + len(negative) - 2, 1))
    return float((positive.mean() - negative.mean()) / pooled) if pooled > 0 else np.nan


def _random_connected_overlap(
    graph: sp.csr_matrix,
    region_size: int,
    tls: np.ndarray,
    observed: int,
    *,
    permutations: int,
    rng: np.random.Generator,
) -> tuple[float, float]:
    n_components, component = connected_components(graph, directed=False)
    component_sizes = np.bincount(component, minlength=n_components)
    valid_starts = np.flatnonzero(component_sizes[component] >= region_size)
    if len(valid_starts) == 0:
        return np.nan, np.nan
    overlaps = []
    for _ in range(int(permutations)):
        selected = {int(rng.choice(valid_starts))}
        frontier = set(selected)
        while len(selected) < region_size and frontier:
            node = int(rng.choice(list(frontier)))
            neighbors = graph.indices[graph.indptr[node] : graph.indptr[node + 1]]
            candidates = [int(x) for x in neighbors if int(x) not in selected]
            if candidates:
                chosen = int(rng.choice(candidates))
                selected.add(chosen)
                frontier.add(chosen)
            else:
                frontier.remove(node)
        if len(selected) != region_size:
            raise RuntimeError("Failed to grow a size-matched connected random region")
        overlaps.append(int(tls[np.fromiter(selected, int)].sum()))
    overlaps = np.asarray(overlaps)
    return float(overlaps.mean()), float((1 + np.sum(overlaps >= observed)) / (len(overlaps) + 1))


def _bh_fdr(values: pd.Series) -> pd.Series:
    result = pd.Series(np.nan, index=values.index, dtype=float)
    keep = values.notna()
    p = values.loc[keep].to_numpy(float)
    if len(p) == 0:
        return result
    order = np.argsort(p)
    adjusted = p[order] * len(p) / np.arange(1, len(p) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    restored = np.empty_like(adjusted)
    restored[order] = np.minimum(adjusted, 1.0)
    result.loc[keep] = restored
    return result


def validate_tls_niches(
    discovery_dir: str | Path,
    tls_annotations: str | Path,
    *,
    permutations: int = 1000,
    random_seed: int = 20260730,
) -> dict[str, pd.DataFrame]:
    """Validate already-fixed programs/niches against held-out TLS annotations."""
    root = Path(discovery_dir)
    manifest = json.loads((root / "discovery_manifest.json").read_text(encoding="utf-8"))
    if manifest.get("stage") != "label_free_discovery_complete" or manifest.get("tls_labels_used") is not False:
        raise ValueError("Discovery manifest does not prove label-free completion")
    tls = read_tls_annotations(tls_annotations)
    spots = pd.read_csv(root / "spatial_locations.tsv", sep="\t")
    activity = pd.read_csv(root / "program_activity.tsv", sep="\t")
    niches = pd.read_csv(root / "candidate_niche_membership.tsv", sep="\t")
    scores = spots.merge(tls, on=["sample_id", "spot_id"], how="inner", validate="one_to_one")
    scores = scores.merge(activity, on="location_id", how="inner", validate="one_to_one")
    if scores.empty:
        raise ValueError("No TLS annotation rows match discovered spatial locations")

    program_rows = []
    programs = [column for column in activity.columns if column.startswith("P")]
    for sample_group, group in [("pooled", scores), *list(scores.groupby("sample_id"))]:
        labels = group.tls.to_numpy(bool)
        for program in programs:
            values = group[program].to_numpy(float)
            auc = roc_auc_score(labels, values) if labels.any() and (~labels).any() else np.nan
            p = mannwhitneyu(values[labels], values[~labels], alternative="two-sided").pvalue \
                if labels.any() and (~labels).any() else np.nan
            program_rows.append(
                {"sample_id": sample_group, "program": program, "n_tls": int(labels.sum()),
                 "n_non_tls": int((~labels).sum()), "mean_tls": float(values[labels].mean()) if labels.any() else np.nan,
                 "mean_non_tls": float(values[~labels].mean()) if (~labels).any() else np.nan,
                 "roc_auc": auc, "standardized_mean_difference": _effect_size(values, labels),
                 "mann_whitney_p": p}
            )
    program_results = pd.DataFrame(program_rows)
    program_results["mann_whitney_fdr"] = program_results.groupby("sample_id", group_keys=False)[
        "mann_whitney_p"
    ].apply(_bh_fdr)

    niche_rows = []
    rng = np.random.default_rng(random_seed)
    tls_by_location = scores.set_index("location_id").tls
    for row in niches.groupby("niche_id", sort=False):
        niche_id, members = row
        sample = str(members.sample_id.iloc[0])
        sample_spots = spots.loc[spots.sample_id.astype(str).eq(sample)].copy()
        sample_labels = sample_spots.location_id.map(tls_by_location)
        annotated = sample_labels.notna().to_numpy()
        sample_spots = sample_spots.loc[annotated].reset_index(drop=True)
        labels = sample_labels.loc[annotated].to_numpy(bool)
        member_ids = set(members.location_id)
        predicted = sample_spots.location_id.isin(member_ids).to_numpy()
        if not predicted.any():
            continue
        intersection = int(np.sum(predicted & labels))
        union = int(np.sum(predicted | labels))
        table = [[intersection, int(np.sum(predicted & ~labels))],
                 [int(np.sum(~predicted & labels)), int(np.sum(~predicted & ~labels))]]
        odds, fisher_p = fisher_exact(table, alternative="greater")
        coordinates = sample_spots[["x", "y"]].to_numpy(float)
        _, graph = _distance_and_graph(coordinates, float(sample_spots.distance_threshold.iloc[0]))
        random_mean, random_p = _random_connected_overlap(
            graph, int(predicted.sum()), labels, intersection,
            permutations=permutations, rng=rng,
        )
        niche_rows.append(
            {"niche_id": niche_id, "sample_id": sample, "program": members.program.iloc[0],
             "n_spots": int(predicted.sum()), "tls_spots": int(labels.sum()), "overlap_spots": intersection,
             "tls_fraction_in_niche": intersection / max(int(predicted.sum()), 1),
             "tls_recall": intersection / max(int(labels.sum()), 1), "jaccard": intersection / max(union, 1),
             "fisher_odds_ratio": odds, "fisher_p": fisher_p,
             "random_connected_mean_overlap": random_mean, "random_connected_p": random_p}
        )
    niche_results = pd.DataFrame(niche_rows)
    if not niche_results.empty:
        niche_results["fisher_fdr"] = niche_results.groupby("sample_id", group_keys=False)[
            "fisher_p"
        ].apply(_bh_fdr)
        niche_results["random_connected_fdr"] = niche_results.groupby("sample_id", group_keys=False)[
            "random_connected_p"
        ].apply(_bh_fdr)
    _write_frame(program_results, root / "tls_program_validation.tsv")
    _write_frame(niche_results, root / "tls_niche_validation.tsv")
    audit = {
        "stage": "post_discovery_tls_validation_complete",
        "tls_annotation": str(Path(tls_annotations).resolve()),
        "n_matched_spots": int(len(scores)),
        "n_tls_spots": int(scores.tls.sum()),
        "permutations": int(permutations),
    }
    (root / "tls_validation_manifest.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    return {"program": program_results, "niche": niche_results}


def plot_program_loadings(discovery_dir: str | Path, *, top_n: int = 10):
    import matplotlib.pyplot as plt

    root = Path(discovery_dir)
    composition = pd.read_csv(root / "program_composition.tsv", sep="\t")
    programs = composition.program.drop_duplicates().tolist()
    fig, axes = plt.subplots(1, len(programs), figsize=(4.2 * len(programs), 4.8), squeeze=False)
    for axis, program in zip(axes.ravel(), programs):
        frame = composition.loc[composition.program.eq(program)].nlargest(top_n, "loading").iloc[::-1]
        labels = frame.sender + "→" + frame.receiver + " | " + frame.ligand + "–" + frame.receptor
        axis.barh(labels, frame.loading, color="#287D8E")
        axis.set_title(program)
        axis.set_xlabel("NMF loading")
    fig.tight_layout()
    return fig


def plot_nmf_diagnostics(discovery_dir: str | Path):
    import matplotlib.pyplot as plt

    diagnostics = pd.read_csv(Path(discovery_dir) / "nmf_k_diagnostics.tsv", sep="\t")
    selected = diagnostics.loc[diagnostics.selected].iloc[0]
    fig, first = plt.subplots(figsize=(5.6, 3.8))
    second = first.twinx()
    first.plot(diagnostics.k, diagnostics.relative_reconstruction_error, "o-", color="#3B528B")
    second.plot(diagnostics.k, diagnostics.stability, "s-", color="#D95F02")
    first.axvline(selected.k, color="0.3", linestyle="--", linewidth=1)
    first.set(xlabel="NMF K", ylabel="Relative reconstruction error")
    second.set_ylabel("Repeat-init component stability")
    first.set_title(f"Selected K = {int(selected.k)} (TLS labels not used)")
    fig.tight_layout()
    return fig


def plot_spatial_programs(
    discovery_dir: str | Path,
    *,
    samples: Sequence[str] | None = None,
    image_root: str | Path | None = None,
    point_size: float = 7.0,
):
    import matplotlib.pyplot as plt

    root = Path(discovery_dir)
    spots = pd.read_csv(root / "spatial_locations.tsv", sep="\t")
    activity = pd.read_csv(root / "program_activity.tsv", sep="\t")
    niches = pd.read_csv(root / "candidate_niche_membership.tsv", sep="\t")
    frame = spots.merge(activity, on="location_id", validate="one_to_one")
    sample_values = list(samples) if samples is not None else frame.sample_id.drop_duplicates().tolist()
    programs = [column for column in activity.columns if column.startswith("P")]
    fig, axes = plt.subplots(len(sample_values), len(programs), figsize=(3.2 * len(programs), 3.0 * len(sample_values)), squeeze=False)
    niche_locations = set(niches.location_id) if not niches.empty else set()
    for i, sample in enumerate(sample_values):
        data = frame.loc[frame.sample_id.eq(sample)]
        image = None
        scale = 1.0
        if image_root is not None:
            spatial_dir = Path(image_root) / str(sample) / "spatial"
            image_path = spatial_dir / "tissue_lowres_image.png"
            scale_path = spatial_dir / "scalefactors_json.json"
            if image_path.exists() and scale_path.exists():
                image = plt.imread(image_path)
                scale = float(json.loads(scale_path.read_text())["tissue_lowres_scalef"])
        for j, program in enumerate(programs):
            axis = axes[i, j]
            if image is not None:
                axis.imshow(image)
            # The cached workflow stores Visium [pixel row, pixel column] as
            # [x, y]; plotting uses column horizontally and row vertically.
            horizontal = data.y * scale
            vertical = data.x * scale
            plot = axis.scatter(horizontal, vertical, c=data[program], s=point_size,
                                cmap="magma", linewidths=0, alpha=0.9)
            selected = data.location_id.isin(niche_locations) & data.location_id.isin(
                set(niches.loc[niches.program.eq(program), "location_id"]) if not niches.empty else set()
            )
            if selected.any():
                axis.scatter(horizontal.loc[selected], vertical.loc[selected], s=point_size * 1.8,
                             facecolors="none", edgecolors="#00E5FF", linewidths=0.5)
            if image is None:
                axis.invert_yaxis()
            axis.set_aspect("equal")
            axis.set_axis_off()
            axis.set_title(f"{sample} · {program}")
            fig.colorbar(plot, ax=axis, fraction=0.04, pad=0.01)
    fig.tight_layout()
    return fig


def plot_tls_overlap(
    discovery_dir: str | Path,
    tls_annotations: str | Path,
    *,
    image_root: str | Path | None = None,
    sample_id: str | None = None,
    program: str | None = None,
    point_size: float = 10.0,
):
    """Plot held-out TLS labels beside already-fixed program/hotspot results."""
    import matplotlib.pyplot as plt

    root = Path(discovery_dir)
    spots = pd.read_csv(root / "spatial_locations.tsv", sep="\t")
    activity = pd.read_csv(root / "program_activity.tsv", sep="\t")
    niches = pd.read_csv(root / "candidate_niche_membership.tsv", sep="\t")
    tls = read_tls_annotations(tls_annotations)
    available = spots.merge(tls, on=["sample_id", "spot_id"], how="inner")
    if available.empty:
        raise ValueError("No TLS annotations overlap discovery spots")
    sample = str(sample_id) if sample_id is not None else str(available.sample_id.iloc[0])
    frame = available.loc[available.sample_id.astype(str).eq(sample)].merge(
        activity, on="location_id", validate="one_to_one"
    )
    if program is None:
        validation = pd.read_csv(root / "tls_program_validation.tsv", sep="\t")
        program = str(
            validation.loc[validation.sample_id.astype(str).eq(sample)]
            .sort_values("roc_auc", ascending=False).iloc[0].program
        )
    if program not in frame:
        raise ValueError(f"Unknown program: {program}")

    image = None
    scale = 1.0
    if image_root is not None:
        spatial_dir = Path(image_root) / sample / "spatial"
        image_path = spatial_dir / "tissue_lowres_image.png"
        scale_path = spatial_dir / "scalefactors_json.json"
        if image_path.exists() and scale_path.exists():
            image = plt.imread(image_path)
            scale = float(json.loads(scale_path.read_text())["tissue_lowres_scalef"])
    horizontal, vertical = frame.y * scale, frame.x * scale
    selected = set(niches.loc[
        niches.sample_id.astype(str).eq(sample) & niches.program.eq(program), "location_id"
    ])
    hotspot = frame.location_id.isin(selected)
    tls_mask = frame.tls.to_numpy(bool)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    for axis in axes:
        if image is not None:
            axis.imshow(image)
        axis.set_aspect("equal")
        axis.set_axis_off()
        if image is None:
            axis.invert_yaxis()
    plot = axes[0].scatter(horizontal, vertical, c=frame[program], s=point_size,
                           cmap="magma", linewidths=0, alpha=0.9)
    axes[0].set_title(f"{sample} · {program} activity")
    fig.colorbar(plot, ax=axes[0], fraction=0.04, pad=0.01)
    axes[1].scatter(horizontal, vertical, c="0.75", s=point_size, linewidths=0, alpha=0.35)
    axes[1].scatter(horizontal[hotspot], vertical[hotspot], c="#00A6D6", s=point_size, linewidths=0)
    axes[1].set_title("Fixed candidate hotspots")
    axes[2].scatter(horizontal, vertical, c="0.75", s=point_size, linewidths=0, alpha=0.35)
    axes[2].scatter(horizontal[tls_mask], vertical[tls_mask], c="#D73027", s=point_size, linewidths=0)
    axes[2].set_title("Held-out TLS annotation")
    fig.tight_layout()
    return fig
