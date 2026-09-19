"""Single-cell GSE327192 MERFISH functional-niche discovery and validation.

Discovery is strictly TLS blind. TLS-like regions are reconstructed only after
CCC profiles, NMF programs, K, and spatial hotspots have been frozen.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.optimize import linear_sum_assignment
from scipy.sparse.csgraph import connected_components
from scipy.stats import mannwhitneyu
from sklearn.decomposition import NMF
from sklearn.metrics import roc_auc_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
from statsmodels.stats.multitest import multipletests

from .merfish_anchor import SAMPLES, _aggregate_panel_counts
from .workflow import _entity_genes


META_COLUMNS = [
    "location_id", "sample_id", "EntityID", "fov", "center_x", "center_y",
    "transcript_count", "cell_type", "label_score", "label_margin",
    "local_density",
]


def _atomic_json(value: dict, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _feature_id(frame: pd.DataFrame) -> pd.Series:
    return frame["sender"].astype(str) + "->" + frame["receiver"].astype(str) + "|" + frame["lr_id"].astype(str)


def load_protective_cccs(path: str | Path, threshold: float = 0.70) -> pd.DataFrame:
    frame = pd.read_csv(path, sep="\t")
    required = {
        "sender", "receiver", "lr_id", "ligand", "receptor",
        "full_coefficient", "sign_selection_probability",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Protective CCC file lacks columns: {sorted(missing)}")
    selected = frame.loc[
        frame.sign_selection_probability.ge(threshold)
        & frame.full_coefficient.lt(0)
    ].copy()
    if len(selected) != 11:
        raise ValueError(f"Expected exactly 11 stable protective CCCs, found {len(selected)}")
    selected["ccc_id"] = _feature_id(selected)
    if selected.ccc_id.duplicated().any():
        raise ValueError("Protective CCC identifiers are not unique")
    return selected.reset_index(drop=True)


def _cell_entity_expression(
    normalized_gene_expression: np.ndarray,
    gene_index: dict[str, int],
    entity: str,
) -> np.ndarray:
    columns = [gene_index[gene] for gene in _entity_genes(entity)]
    values = normalized_gene_expression[:, columns]
    observed = (values > 0).all(axis=1)
    result = np.zeros(len(values), dtype=np.float32)
    if observed.any():
        result[observed] = np.exp(np.log(values[observed]).mean(axis=1))
    return result


def _knn(
    coordinates: np.ndarray,
    neighbors: int,
    maximum_distance: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    distance, index = NearestNeighbors(
        n_neighbors=neighbors + 1, algorithm="kd_tree", n_jobs=-1
    ).fit(coordinates).kneighbors(coordinates)
    distance, index = distance[:, 1:], index[:, 1:]
    valid = distance <= maximum_distance
    return index.astype(np.int32), distance.astype(np.float32), valid


def _profiles_for_sample(
    accession: str,
    sample_name: str,
    raw_root: Path,
    label_root: Path,
    protective: pd.DataFrame,
    *,
    neighbors: int,
    maximum_neighbor_distance: float,
    chunksize: int,
) -> tuple[pd.DataFrame, dict]:
    cells = pd.read_parquet(label_root / f"{accession}.parquet")
    cells = cells.loc[cells.cell_type.ne("Unclassified")].reset_index(drop=True)
    required_genes = sorted(
        {
            gene
            for entity in pd.concat([protective.ligand, protective.receptor])
            for gene in _entity_genes(entity)
        }
    )
    transcript_file = raw_root / f"{accession}_{sample_name}_detected_transcripts.csv.gz"
    counts = _aggregate_panel_counts(
        transcript_file,
        cells.EntityID.to_numpy(np.int64),
        required_genes,
        chunksize=chunksize,
    )
    raw = counts.toarray().astype(np.float32)
    raw *= (1e4 / np.maximum(cells.transcript_count.to_numpy(np.float32), 1.0))[:, None]
    raw = np.log1p(raw)
    gene_index = {gene: index for index, gene in enumerate(required_genes)}
    entities = sorted(set(protective.ligand) | set(protective.receptor))
    expression = {
        entity: _cell_entity_expression(raw, gene_index, entity) for entity in entities
    }
    coordinates = cells[["center_x", "center_y"]].to_numpy(np.float64)
    neighbor_index, distance, valid_edge = _knn(
        coordinates, neighbors, maximum_neighbor_distance
    )
    valid_count = np.maximum(valid_edge.sum(axis=1), 1)
    mean_distance = np.divide(
        (distance * valid_edge).sum(axis=1),
        valid_count,
        out=np.full(len(cells), maximum_neighbor_distance, dtype=np.float32),
        where=valid_count > 0,
    )
    local_density = 1.0 / np.maximum(mean_distance, 1e-3) ** 2
    labels = cells.cell_type.astype(str).to_numpy()
    activity = np.zeros((len(cells), len(protective)), dtype=np.float32)
    focal = np.arange(len(cells), dtype=np.int64)
    for column, item in enumerate(protective.itertuples(index=False)):
        receptor_value = expression[item.receptor]
        ligand_value = expression[item.ligand]
        receiver_match = (labels == item.receiver) & (receptor_value > 0)
        if not receiver_match.any():
            continue
        focal_receiver = focal[receiver_match]
        neighbor = neighbor_index[receiver_match]
        edge_valid = valid_edge[receiver_match]
        sender_match = labels[neighbor] == item.sender
        ligand_neighbor = ligand_value[neighbor]
        edge_valid &= sender_match & (ligand_neighbor > 0)
        pair_mass = np.sqrt(
            ligand_neighbor * receptor_value[receiver_match, None]
        )
        pair_mass[~edge_valid] = 0.0
        activity[focal_receiver, column] = (
            pair_mass.sum(axis=1) / valid_count[receiver_match]
        )

    profile = cells.copy()
    if "sample_id" not in profile:
        profile.insert(0, "sample_id", accession)
    else:
        profile["sample_id"] = accession
    profile.insert(0, "location_id", accession + "|" + profile.EntityID.astype(str))
    profile["local_density"] = local_density
    for column, ccc_id in enumerate(protective.ccc_id):
        profile[ccc_id] = activity[:, column]
    audit = {
        "sample_id": accession,
        "cells": int(len(cells)),
        "active_receiver_cells": int((activity.sum(axis=1) > 0).sum()),
        "positive_cells_per_ccc": {
            protective.ccc_id.iloc[column]: int((activity[:, column] > 0).sum())
            for column in range(activity.shape[1])
        },
        "neighbors": int(neighbors),
        "maximum_neighbor_distance": float(maximum_neighbor_distance),
        "neighbor_edges": int(valid_edge.sum()),
    }
    return profile, audit


def compute_single_cell_ccc_profiles(
    raw_root: str | Path,
    label_root: str | Path,
    stable_protective_ccc: str | Path,
    output: str | Path,
    *,
    stability_threshold: float = 0.70,
    neighbors: int = 6,
    maximum_neighbor_distance: float = 30.0,
    chunksize: int = 5_000_000,
    overwrite: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw_root, label_root, output = Path(raw_root), Path(label_root), Path(output)
    profile_root = output / "ccc_profiles_by_sample"
    profile_root.mkdir(parents=True, exist_ok=True)
    protective = load_protective_cccs(stable_protective_ccc, stability_threshold)
    protective.to_csv(output / "protective_cccs.tsv", sep="\t", index=False)
    frames, audits = [], []
    for accession, sample_name in SAMPLES.items():
        cache = profile_root / f"{accession}.parquet"
        audit_cache = profile_root / f"{accession}.audit.json"
        if cache.exists() and audit_cache.exists() and not overwrite:
            print(f"Single-cell CCC profile cache reused: {accession}", flush=True)
            frame = pd.read_parquet(cache)
            audit = json.loads(audit_cache.read_text(encoding="utf-8"))
        else:
            print(f"Single-cell CCC profiles starting: {accession}", flush=True)
            frame, audit = _profiles_for_sample(
                accession, sample_name, raw_root, label_root, protective,
                neighbors=neighbors,
                maximum_neighbor_distance=maximum_neighbor_distance,
                chunksize=chunksize,
            )
            frame.to_parquet(cache, index=False)
            _atomic_json(audit, audit_cache)
            print(f"Single-cell CCC profiles complete: {accession}", flush=True)
        frames.append(frame)
        audits.append(audit)
    profile = pd.concat(frames, ignore_index=True)
    _atomic_json(
        {
            "tls_blind": True,
            "tls_inputs_used": [],
            "stability_threshold": float(stability_threshold),
            "protective_cccs": int(len(protective)),
            "actually_computable_cccs": int(
                sum(any(a["positive_cells_per_ccc"][ccc] > 0 for a in audits) for ccc in protective.ccc_id)
            ),
            "samples": audits,
        },
        output / "ccc_profile_manifest.json",
    )
    return profile, protective


def robust_standardize_profiles(
    profile: pd.DataFrame,
    protective: pd.DataFrame,
    output: str | Path,
    *,
    minimum_positive_cells: int = 100,
) -> tuple[pd.DataFrame, list[str]]:
    output = Path(output)
    columns = protective.ccc_id.tolist()
    normalized = np.zeros((len(profile), len(columns)), dtype=np.float32)
    log_values = np.log1p(profile[columns].to_numpy(np.float32))
    global_scales = {}
    for column, ccc_id in enumerate(columns):
        positive = log_values[:, column][log_values[:, column] > 0]
        global_scales[ccc_id] = (
            float(np.quantile(positive, 0.95)) if len(positive) else np.nan
        )
    audits = []
    for sample_id, indices in profile.groupby("sample_id", sort=False).indices.items():
        rows = np.asarray(indices, dtype=np.int64)
        values = log_values[rows]
        for column, ccc_id in enumerate(columns):
            positive = values[:, column][values[:, column] > 0]
            local_scale_reliable = len(positive) >= minimum_positive_cells
            scale_source = "sample_q95" if local_scale_reliable else "cohort_q95"
            scale = (
                float(np.quantile(positive, 0.95))
                if local_scale_reliable
                else global_scales[ccc_id]
            )
            observed = len(positive) > 0
            if observed and np.isfinite(scale) and scale > 0:
                normalized[rows, column] = np.clip(values[:, column] / scale, 0, 3)
            audits.append(
                (sample_id, ccc_id, len(positive), scale, scale_source, observed)
            )
    audit = pd.DataFrame(
        audits,
        columns=[
            "sample_id", "ccc_id", "positive_cells", "log_q95",
            "scale_source", "observed",
        ],
    )
    # A route remains analysis-usable when it is genuinely observed in at least
    # two independent samples. The >=100 rule above concerns only whether a
    # sample-specific scale can be estimated reliably; it must not erase rare
    # but observable directional cell-cell interactions.
    usable_columns = [
        ccc for ccc in columns
        if audit.loc[audit.ccc_id.eq(ccc), "observed"].sum() >= 2
    ]
    if len(usable_columns) < 2:
        raise ValueError("Fewer than two protective CCCs have usable single-cell activity")
    normalized_frame = pd.DataFrame(
        normalized[:, [columns.index(c) for c in usable_columns]],
        index=profile.location_id,
        columns=usable_columns,
    )
    audit.to_csv(output / "normalization_audit.tsv", sep="\t", index=False)
    return normalized_frame, usable_columns


def _fit_nmf_repeats(x: np.ndarray, k: int, seeds: list[int], max_iter: int):
    fits = []
    for seed in seeds:
        model = NMF(
            n_components=k, init="random", solver="cd", beta_loss="frobenius",
            random_state=seed, max_iter=max_iter, tol=1e-4,
        )
        model.fit(x)
        fits.append({"seed": seed, "H": model.components_.copy(), "error": model.reconstruction_err_})
    similarity = np.eye(len(fits))
    for first in range(len(fits)):
        for second in range(first + 1, len(fits)):
            matrix = cosine_similarity(fits[first]["H"], fits[second]["H"])
            row, column = linear_sum_assignment(-matrix)
            similarity[first, second] = similarity[second, first] = matrix[row, column].mean()
    stability = float(similarity[np.triu_indices(len(fits), 1)].mean())
    medoid = int(np.argmax((similarity.sum(axis=1) - 1) / (len(fits) - 1)))
    return fits, stability, medoid


def discover_programs(
    normalized: pd.DataFrame,
    profile: pd.DataFrame,
    output: str | Path,
    *,
    k_values=range(2, 7),
    n_restarts: int = 10,
    fit_cells_per_sample: int = 20_000,
    random_seed: int = 20260730,
    max_iter: int = 1000,
) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    output = Path(output)
    rng = np.random.default_rng(random_seed)
    active = normalized.sum(axis=1).to_numpy() > 0
    selected_rows = []
    sample_values = profile.sample_id.to_numpy()
    for sample_id in pd.unique(sample_values):
        candidates = np.flatnonzero(active & (sample_values == sample_id))
        if len(candidates) > fit_cells_per_sample:
            candidates = rng.choice(candidates, fit_cells_per_sample, replace=False)
        selected_rows.extend(candidates.tolist())
    selected_rows = np.asarray(sorted(selected_rows), dtype=np.int64)
    x_fit = normalized.iloc[selected_rows].to_numpy(np.float32)
    if len(x_fit) < 1000:
        raise ValueError(f"Too few active cells for NMF: {len(x_fit)}")
    seeds = [random_seed + 104729 * index for index in range(n_restarts)]
    norm = max(float(np.linalg.norm(x_fit)), np.finfo(float).tiny)
    requested_k_values = list(map(int, k_values))
    tested_k_values = sorted(
        {k for k in requested_k_values if 2 <= k <= normalized.shape[1]}
    )
    if not tested_k_values:
        raise ValueError(
            f"No valid K: requested={requested_k_values}, "
            f"CCC features={normalized.shape[1]}"
        )
    fit_store, rows = {}, []
    for k in tested_k_values:
        fits, stability, medoid = _fit_nmf_repeats(x_fit, int(k), seeds, max_iter)
        fit_store[int(k)] = (fits, medoid)
        errors = np.asarray([fit["error"] / norm for fit in fits])
        medoid_similarity = cosine_similarity(fits[medoid]["H"])
        within_solution_max_cosine = float(
            medoid_similarity[np.triu_indices(int(k), 1)].max()
        )
        rows.append(
            {
                "k": int(k), "stability": stability,
                "relative_reconstruction_error": float(errors.mean()),
                "reconstruction_error_sd": float(errors.std(ddof=1)),
                "within_solution_max_cosine": within_solution_max_cosine,
                "medoid_seed": int(fits[medoid]["seed"]),
            }
        )
        print(
            f"NMF K={k}: stability={stability:.4f}, error={errors.mean():.4f}, "
            f"max_component_cosine={within_solution_max_cosine:.4f}",
            flush=True,
        )
    diagnostics = pd.DataFrame(rows)
    diagnostics["stability_rank"] = diagnostics.stability.rank(ascending=False, method="min")
    diagnostics["error_rank"] = diagnostics.relative_reconstruction_error.rank(ascending=True, method="min")
    diagnostics["selection_rank_sum"] = diagnostics.stability_rank + diagnostics.error_rank
    # Require both repeat stability and distinct components. Repeat matching
    # alone can incorrectly call a solution stable when NMF repeatedly splits
    # one dominant route into duplicate components. TLS labels are not
    # consulted anywhere in this decision.
    admissible = diagnostics.loc[
        diagnostics.stability.ge(0.80)
        & diagnostics.within_solution_max_cosine.le(0.95)
    ].copy()
    if admissible.empty:
        admissible = diagnostics.loc[[
            diagnostics.sort_values(
                ["within_solution_max_cosine", "stability", "k"],
                ascending=[True, False, True],
            ).index[0]
        ]].copy()
    chosen = int(
        admissible.sort_values(
            ["relative_reconstruction_error", "stability", "k"],
            ascending=[True, False, True],
        ).iloc[0].k
    )
    selection_rule = (
        "minimum reconstruction error among repeat-stable (>=0.80) and "
        "nonredundant (max component cosine<=0.95) solutions"
    )
    diagnostics["selected"] = diagnostics.k.eq(chosen)
    fits, medoid = fit_store[chosen]
    seed = fits[medoid]["seed"]
    final_model = NMF(
        n_components=chosen, init="random", solver="cd", beta_loss="frobenius",
        random_state=seed, max_iter=max_iter, tol=1e-4,
    ).fit(x_fit)
    names = [f"P{index + 1}" for index in range(chosen)]
    h = final_model.components_.copy()
    scales = np.maximum(h.sum(axis=1), np.finfo(float).tiny)
    loadings = pd.DataFrame(
        (h / scales[:, None]).T,
        index=normalized.columns,
        columns=names,
    )
    loadings.index.name = "ccc_id"
    activity = np.zeros((len(normalized), chosen), dtype=np.float32)
    block = 250_000
    for start in range(0, len(normalized), block):
        stop = min(start + block, len(normalized))
        activity[start:stop] = final_model.transform(
            normalized.iloc[start:stop].to_numpy(np.float32)
        ) * scales[None, :]
    activities = pd.DataFrame(activity, index=normalized.index, columns=names)
    diagnostics.to_csv(output / "nmf_k_diagnostics.tsv", sep="\t", index=False)
    loadings.reset_index().to_csv(output / "program_ccc_loadings.tsv", sep="\t", index=False)
    pd.DataFrame({"location_id": normalized.index, **{p: activities[p].to_numpy() for p in names}}).to_parquet(
        output / "program_activity.parquet", index=False
    )
    _atomic_json(
        {
            "tls_blind": True, "tls_inputs_used": [], "selected_k": chosen,
            "fit_cells": int(len(x_fit)), "active_cells": int(active.sum()),
            "requested_k_values": requested_k_values,
            "tested_k_values": tested_k_values,
            "selection_rule": selection_rule,
            "n_restarts": int(n_restarts),
        }, output / "nmf_manifest.json"
    )
    return loadings, activities, chosen


def summarize_programs(
    loadings: pd.DataFrame,
    protective: pd.DataFrame,
    output: str | Path,
) -> pd.DataFrame:
    rows = []
    meta = protective.set_index("ccc_id")
    for program in loadings.columns:
        for rank, (ccc_id, loading) in enumerate(loadings[program].sort_values(ascending=False).items(), 1):
            item = meta.loc[ccc_id]
            rows.append(
                {
                    "program": program, "rank": rank, "ccc_id": ccc_id,
                    "loading": loading, "sender": item.sender, "receiver": item.receiver,
                    "ligand": item.ligand, "receptor": item.receptor,
                    "stability": item.sign_selection_probability,
                    "cox_coefficient": item.full_coefficient,
                }
            )
    result = pd.DataFrame(rows)
    result.to_csv(Path(output) / "program_composition.tsv", sep="\t", index=False)
    return result


def call_hotspots(
    profile: pd.DataFrame,
    activities: pd.DataFrame,
    output: str | Path,
    *,
    neighbors: int = 6,
    maximum_neighbor_distance: float = 30.0,
    minimum_niche_cells: int = 30,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    output = Path(output)
    activity = activities.reindex(profile.location_id).to_numpy(np.float32)
    memberships, niches, score_frames = [], [], []
    for sample_id, indices in profile.groupby("sample_id", sort=False).indices.items():
        rows = np.asarray(indices, dtype=np.int64)
        cells = profile.iloc[rows]
        coords = cells[["center_x", "center_y"]].to_numpy(float)
        neighbor, _, valid = _knn(coords, neighbors, maximum_neighbor_distance)
        focal = np.repeat(np.arange(len(cells), dtype=np.int64), neighbors)
        target = neighbor.reshape(-1)
        edge_keep = valid.reshape(-1)
        focal, target = focal[edge_keep], target[edge_keep]
        sample_activity = activity[rows]
        degree = np.maximum(valid.sum(axis=1), 1)
        for p, program in enumerate(activities.columns):
            raw = sample_activity[:, p]
            neighbor_sum = np.zeros(len(cells), dtype=np.float32)
            np.add.at(neighbor_sum, focal, raw[target])
            smooth = (raw + neighbor_sum / degree) / 2.0
            positive = smooth[smooth > 0]
            if len(positive) < minimum_niche_cells:
                continue
            q1, q3 = np.quantile(positive, [0.25, 0.75])
            median = np.median(positive)
            mad = np.median(np.abs(positive - median))
            threshold = float(max(q3 + 1.5 * (q3 - q1), median + 3.0 * mad))
            high = smooth > threshold
            if high.sum() < minimum_niche_cells:
                continue
            high_index = np.flatnonzero(high)
            map_high = np.full(len(cells), -1, dtype=np.int64)
            map_high[high_index] = np.arange(len(high_index))
            keep = high[focal] & high[target]
            graph = sp.coo_matrix(
                (np.ones(keep.sum()), (map_high[focal[keep]], map_high[target[keep]])),
                shape=(len(high_index), len(high_index)),
            ).tocsr()
            count, component = connected_components(graph, directed=False)
            score_frames.append(
                pd.DataFrame(
                    {
                        "location_id": cells.location_id,
                        "sample_id": sample_id,
                        "program": program,
                        "raw_activity": raw,
                        "smoothed_activity": smooth,
                        "hotspot_threshold": threshold,
                        "above_threshold": high,
                    }
                )
            )
            for label in range(count):
                local = high_index[component == label]
                if len(local) < minimum_niche_cells:
                    continue
                niche_id = f"{sample_id}|{program}|N{label + 1}"
                niches.append(
                    {
                        "niche_id": niche_id, "sample_id": sample_id,
                        "program": program, "n_cells": len(local),
                        "mean_activity": float(raw[local].mean()),
                        "mean_smoothed_activity": float(smooth[local].mean()),
                        "hotspot_threshold": threshold,
                    }
                )
                memberships.extend(
                    {
                        "niche_id": niche_id, "sample_id": sample_id,
                        "program": program, "location_id": cells.location_id.iloc[index],
                        "EntityID": cells.EntityID.iloc[index],
                        "cell_type": cells.cell_type.iloc[index],
                        "program_activity": raw[index],
                        "smoothed_activity": smooth[index],
                    }
                    for index in local
                )
    membership = pd.DataFrame(memberships)
    niche = pd.DataFrame(niches)
    scores = pd.concat(score_frames, ignore_index=True) if score_frames else pd.DataFrame()
    membership.to_csv(output / "candidate_niche_membership.tsv.gz", sep="\t", index=False, compression="gzip")
    niche.to_csv(output / "candidate_niches.tsv", sep="\t", index=False)
    scores.to_parquet(output / "program_spatial_scores.parquet", index=False)
    _atomic_json(
        {
            "tls_blind": True, "tls_inputs_used": [],
            "threshold_rule": "max(Q3 + 1.5*IQR, median + 3*MAD) among positive smoothed scores",
            "minimum_niche_cells": int(minimum_niche_cells),
            "candidate_niches": int(len(niche)),
        }, output / "hotspot_manifest.json"
    )
    return membership, niche, scores


def _hedges_g(case: np.ndarray, control: np.ndarray) -> float:
    n1, n0 = len(case), len(control)
    if n1 < 2 or n0 < 2:
        return np.nan
    pooled = np.sqrt(
        ((n1 - 1) * case.var(ddof=1) + (n0 - 1) * control.var(ddof=1))
        / max(n1 + n0 - 2, 1)
    )
    if pooled <= 0:
        return 0.0
    d = (case.mean() - control.mean()) / pooled
    correction = 1 - 3 / max(4 * (n1 + n0) - 9, 1)
    return float(d * correction)


def reconstruct_tls_like_regions(
    profile: pd.DataFrame,
    output: str | Path,
    *,
    tls_neighbors: int = 50,
    maximum_distance: float = 75.0,
    minimum_b_cells: int = 10,
    minimum_t_cells: int = 10,
    minimum_lymphocyte_fraction: float = 0.50,
) -> pd.DataFrame:
    """Post-discovery B/T co-aggregation proxy; not an author TLS annotation."""
    output = Path(output)
    frames = []
    for sample_id, indices in profile.groupby("sample_id", sort=False).indices.items():
        rows = np.asarray(indices, dtype=np.int64)
        cells = profile.iloc[rows]
        coords = cells[["center_x", "center_y"]].to_numpy(float)
        neighbor, _, valid = _knn(coords, tls_neighbors, maximum_distance)
        labels = cells.cell_type.astype(str).to_numpy()
        b_count = ((labels[neighbor] == "B cells") & valid).sum(axis=1)
        t_count = ((labels[neighbor] == "T cells") & valid).sum(axis=1)
        n_valid = np.maximum(valid.sum(axis=1), 1)
        fraction = (b_count + t_count) / n_valid
        tls_like = (
            (b_count >= minimum_b_cells)
            & (t_count >= minimum_t_cells)
            & (fraction >= minimum_lymphocyte_fraction)
        )
        frames.append(
            pd.DataFrame(
                {
                    "location_id": cells.location_id,
                    "sample_id": sample_id,
                    "b_neighbors": b_count,
                    "t_neighbors": t_count,
                    "lymphocyte_fraction": fraction,
                    "tls_like": tls_like,
                }
            )
        )
    result = pd.concat(frames, ignore_index=True)
    result.to_parquet(output / "reconstructed_tls_like_annotation.parquet", index=False)
    _atomic_json(
        {
            "annotation_type": "post-discovery reconstructed TLS-like proxy",
            "author_provided": False,
            "independent_of_programs": True,
            "rule": {
                "neighbors": tls_neighbors, "maximum_distance": maximum_distance,
                "minimum_b_cells": minimum_b_cells, "minimum_t_cells": minimum_t_cells,
                "minimum_lymphocyte_fraction": minimum_lymphocyte_fraction,
            },
        }, output / "tls_proxy_manifest.json"
    )
    return result


def validate_tls(
    profile: pd.DataFrame,
    activities: pd.DataFrame,
    membership: pd.DataFrame,
    niches: pd.DataFrame,
    tls: pd.DataFrame,
    output: str | Path,
    *,
    permutations: int = 200,
    random_seed: int = 20260730,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    output = Path(output)
    tls_map = tls.set_index("location_id").tls_like.reindex(profile.location_id).fillna(False).to_numpy(bool)
    program_rows = []
    for program in activities.columns:
        score = activities.reindex(profile.location_id)[program].to_numpy(float)
        if tls_map.any() and (~tls_map).any():
            auc = roc_auc_score(tls_map, score)
            effect = _hedges_g(score[tls_map], score[~tls_map])
            pvalue = mannwhitneyu(score[tls_map], score[~tls_map], alternative="greater").pvalue
        else:
            auc = effect = pvalue = np.nan
        program_rows.append(
            {
                "program": program, "tls_cells": int(tls_map.sum()),
                "non_tls_cells": int((~tls_map).sum()),
                "tls_mean_activity": float(score[tls_map].mean()) if tls_map.any() else np.nan,
                "non_tls_mean_activity": float(score[~tls_map].mean()) if (~tls_map).any() else np.nan,
                "auc": auc, "hedges_g": effect, "pvalue": pvalue,
            }
        )
    program = pd.DataFrame(program_rows)
    program["fdr"] = multipletests(program.pvalue.fillna(1), method="fdr_bh")[1]
    program.to_csv(output / "tls_program_validation.tsv", sep="\t", index=False)

    rng = np.random.default_rng(random_seed)
    info = profile[["location_id", "sample_id", "cell_type", "local_density"]].copy()
    info["tls_like"] = tls_map
    info["density_bin"] = info.groupby("sample_id").local_density.transform(
        lambda x: pd.qcut(x.rank(method="first"), 5, labels=False)
    )
    info = info.set_index("location_id")
    niche_rows = []
    for item in niches.itertuples(index=False):
        locations = membership.loc[membership.niche_id.eq(item.niche_id), "location_id"]
        observed_cells = info.loc[locations]
        observed = float(observed_cells.tls_like.mean())
        pool = info.loc[info.sample_id.eq(item.sample_id)].drop(index=locations, errors="ignore")
        background = []
        strata = observed_cells.groupby(["cell_type", "density_bin"], observed=True).size()
        for _ in range(permutations):
            values = []
            for (cell_type, density_bin), size in strata.items():
                candidates = pool.loc[
                    pool.cell_type.eq(cell_type) & pool.density_bin.eq(density_bin), "tls_like"
                ].to_numpy()
                if len(candidates):
                    values.extend(rng.choice(candidates, int(size), replace=len(candidates) < size))
            background.append(float(np.mean(values)) if values else 0.0)
        background = np.asarray(background)
        niche_rows.append(
            {
                "niche_id": item.niche_id, "sample_id": item.sample_id,
                "program": item.program, "n_cells": item.n_cells,
                "tls_overlap_fraction": observed,
                "matched_background_mean": float(background.mean()),
                "matched_enrichment": float((observed + 1e-6) / (background.mean() + 1e-6)),
                "empirical_pvalue": float((1 + (background >= observed).sum()) / (permutations + 1)),
            }
        )
    niche = pd.DataFrame(niche_rows)
    if len(niche):
        niche["fdr"] = multipletests(niche.empirical_pvalue, method="fdr_bh")[1]
    niche.to_csv(output / "tls_niche_validation.tsv", sep="\t", index=False)
    return program, niche


def plot_results(
    profile: pd.DataFrame,
    loadings: pd.DataFrame,
    activities: pd.DataFrame,
    membership: pd.DataFrame,
    program_validation: pd.DataFrame,
    tls: pd.DataFrame,
    output: str | Path,
    *,
    random_seed: int = 20260730,
) -> list[Path]:
    output = Path(output)
    mpl.rcParams.update(
        {
            "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "pdf.fonttype": 42, "font.size": 7, "axes.spines.right": False,
            "axes.spines.top": False, "axes.linewidth": 0.8, "legend.frameon": False,
        }
    )
    paths = []
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    image = ax.imshow(loadings.to_numpy().T, aspect="auto", cmap="magma", vmin=0)
    ax.set_yticks(range(len(loadings.columns)), loadings.columns)
    labels = [
        f"{x.split('|')[0]}\n{x.split('|')[1].split('|')[0]}"
        for x in loadings.index
    ]
    ax.set_xticks(range(len(loadings.index)), labels, rotation=70, ha="right")
    ax.set_title("Protective CCC programs (TLS-blind NMF)")
    fig.colorbar(image, ax=ax, label="Normalized loading", fraction=0.025)
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        path = output / f"program_ccc_loadings.{suffix}"
        fig.savefig(path, dpi=400 if suffix == "png" else None, bbox_inches="tight")
        paths.append(path)
    plt.close(fig)

    # The spatial figure must show an actually discovered niche. TLS metrics are
    # deliberately not used to choose the displayed discovery program.
    if len(membership):
        top_program = membership.groupby("program").size().sort_values(ascending=False).index[0]
    else:
        top_program = activities.columns[0]
    activity_map = activities[top_program]
    tls_set = set(tls.loc[tls.tls_like, "location_id"])
    niche_set = set(membership.loc[membership.program.eq(top_program), "location_id"])
    rng = np.random.default_rng(random_seed)
    fig, axes = plt.subplots(2, 4, figsize=(11, 5.5), constrained_layout=True)
    all_positive = activity_map.to_numpy()[activity_map.to_numpy() > 0]
    vmax = float(np.quantile(all_positive, 0.995)) if len(all_positive) else 1.0
    plotted = None
    for ax, (sample_id, indices) in zip(axes.ravel(), profile.groupby("sample_id", sort=False).indices.items()):
        rows = np.asarray(indices)
        cells = profile.iloc[rows]
        if len(rows) > 100_000:
            shown = rng.choice(rows, 100_000, replace=False)
            base = profile.iloc[shown]
        else:
            base = cells
        score = activity_map.reindex(base.location_id).fillna(0).to_numpy()
        plotted = ax.scatter(
            base.center_x, base.center_y, c=score, s=0.18, cmap="viridis",
            vmin=0, vmax=vmax, rasterized=True,
        )
        niche_cells = cells.loc[cells.location_id.isin(niche_set)]
        tls_cells = cells.loc[cells.location_id.isin(tls_set)]
        if len(niche_cells):
            ax.scatter(
                niche_cells.center_x, niche_cells.center_y, s=1.5,
                c="#E64B35", linewidths=0, rasterized=True,
            )
        if len(tls_cells):
            ax.scatter(
                tls_cells.center_x, tls_cells.center_y, s=0.30,
                c="#00A6D6", alpha=0.35, linewidths=0, rasterized=True,
            )
        ax.set_title(sample_id)
        ax.set_aspect("equal")
        ax.invert_yaxis()
        ax.set_xticks([]); ax.set_yticks([])
    if plotted is not None:
        fig.colorbar(plotted, ax=axes, label=f"{top_program} activity", shrink=0.65)
    fig.legend(
        handles=[
            Line2D([], [], marker="o", linestyle="none", color="#E64B35", label="candidate niche"),
            Line2D([], [], marker="o", linestyle="none", color="#00A6D6", label="TLS-like proxy (post hoc)"),
        ],
        loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.01),
    )
    fig.suptitle(f"{top_program} spatial activity and TLS-blind candidate niches")
    for suffix in ("png", "pdf"):
        path = output / f"candidate_niche_program_spatial.{suffix}"
        fig.savefig(path, dpi=400 if suffix == "png" else None, bbox_inches="tight")
        paths.append(path)
    plt.close(fig)
    return paths


def run_discovery(
    raw_root: str | Path,
    label_root: str | Path,
    stable_protective_ccc: str | Path,
    output: str | Path,
    *,
    overwrite_profiles: bool = False,
) -> dict:
    """Run only TLS-blind profile, NMF, and hotspot discovery."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    profile, protective = compute_single_cell_ccc_profiles(
        raw_root, label_root, stable_protective_ccc, output,
        overwrite=overwrite_profiles,
    )
    normalized, usable = robust_standardize_profiles(profile, protective, output)
    loadings, activities, chosen = discover_programs(normalized, profile, output)
    composition = summarize_programs(loadings, protective, output)
    membership, niches, _ = call_hotspots(profile, activities, output)
    manifest = {
        "phase": "TLS-blind discovery complete",
        "tls_inputs_used": [],
        "protective_cccs_input": int(len(protective)),
        "protective_cccs_computable": int(len(usable)),
        "selected_k": int(chosen),
        "candidate_niches": int(len(niches)),
        "top_cccs": composition.loc[composition['rank'].le(3)].to_dict("records"),
    }
    _atomic_json(manifest, output / "discovery_manifest.json")
    return manifest


def run_posthoc_tls_validation(output: str | Path) -> dict:
    """Run reconstructed TLS-like validation only after discovery is frozen."""
    output = Path(output)
    discovery = json.loads((output / "discovery_manifest.json").read_text(encoding="utf-8"))
    if discovery.get("phase") != "TLS-blind discovery complete":
        raise RuntimeError("TLS validation cannot run before discovery is frozen")
    protective = pd.read_csv(output / "protective_cccs.tsv", sep="\t")
    profile = pd.concat(
        [pd.read_parquet(path) for path in sorted((output / "ccc_profiles_by_sample").glob("GSM*.parquet"))],
        ignore_index=True,
    )
    loadings = pd.read_csv(output / "program_ccc_loadings.tsv", sep="\t").set_index("ccc_id")
    activity_frame = pd.read_parquet(output / "program_activity.parquet").set_index("location_id")
    membership = pd.read_csv(output / "candidate_niche_membership.tsv.gz", sep="\t")
    niches = pd.read_csv(output / "candidate_niches.tsv", sep="\t")
    tls = reconstruct_tls_like_regions(profile, output)
    program, niche = validate_tls(profile, activity_frame, membership, niches, tls, output)
    figures = plot_results(profile, loadings, activity_frame, membership, program, tls, output)
    best_program = program.sort_values(["auc", "hedges_g"], ascending=False).iloc[0]
    best_niche = niche.sort_values(
        ["tls_overlap_fraction", "matched_enrichment"], ascending=False
    ).iloc[0] if len(niche) else None
    manifest = {
        "phase": "post-hoc reconstructed TLS-like validation complete",
        "author_tls_annotation_available": False,
        "tls_annotation": "reconstructed B/T co-aggregation proxy",
        "best_program": best_program.to_dict(),
        "best_niche": None if best_niche is None else best_niche.to_dict(),
        "figures": [str(path) for path in figures],
    }
    _atomic_json(manifest, output / "tls_validation_manifest.json")
    return manifest
