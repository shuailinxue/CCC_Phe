import json

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import (adjusted_rand_score, normalized_mutual_info_score,
                             silhouette_score)


def cosine_matrix(matrix, epsilon=1e-8):
    matrix = np.asarray(matrix, dtype=np.float64)
    normalized = matrix / np.maximum(np.linalg.norm(matrix, axis=1, keepdims=True), epsilon)
    return np.clip(normalized @ normalized.T, -1, 1)


def hierarchical_program_groups(h32, k, epsilon=1e-8):
    """Cluster H32 rows by average-linkage cosine distance with deterministic labels."""
    distance = np.maximum(1.0 - cosine_matrix(h32, epsilon), 0.0)
    np.fill_diagonal(distance, 0.0)
    try:
        raw = AgglomerativeClustering(n_clusters=k, metric="precomputed", linkage="average").fit_predict(distance)
    except TypeError:  # scikit-learn < 1.2
        raw = AgglomerativeClustering(n_clusters=k, affinity="precomputed", linkage="average").fit_predict(distance)
    ordered = sorted(np.unique(raw), key=lambda label: int(np.flatnonzero(raw == label)[0]))
    remap = {old: new for new, old in enumerate(ordered)}
    return np.asarray([remap[value] for value in raw], dtype=np.int16)


def consolidate_programs(p32, z32, h32, groups, epsilon=1e-8):
    n, d = p32.shape
    groups = np.asarray(groups, dtype=np.int64)
    k = int(groups.max()) + 1
    q = np.zeros((n, k), dtype=np.float32)
    for program in range(d):
        q[:, groups[program]] += p32[:, program]
    q /= np.maximum(q.sum(1, keepdims=True), epsilon)
    exposure_usage = np.asarray(z32).mean(0)
    mixture_usage = np.asarray(p32).mean(0)
    h = np.zeros((k, h32.shape[1]), dtype=np.float64)
    for niche in range(k):
        members = np.flatnonzero(groups == niche)
        weights = exposure_usage[members]
        if weights.sum() <= epsilon:
            weights = np.ones(len(members))
        h[niche] = np.average(h32[members], axis=0, weights=weights)
    h /= np.maximum(h.sum(1, keepdims=True), epsilon)
    mapping = pd.DataFrame({
        "program_id": np.arange(1, d + 1), "final_niche_id": groups + 1,
        "mean_mixture": mixture_usage, "mean_exposure": exposure_usage,
        "exposure_variance": np.asarray(z32).var(0),
        "fraction_active": (np.asarray(z32) > epsilon).mean(0),
    })
    labels = q.argmax(1)
    confidence = q.max(1)
    return q, h.astype(np.float32), labels, confidence.astype(np.float32), mapping


def _scale01(values):
    values = np.asarray(values, dtype=float)
    finite = np.isfinite(values)
    out = np.zeros_like(values)
    if finite.any():
        low, high = values[finite].min(), values[finite].max()
        out[finite] = 1.0 if high <= low else (values[finite] - low) / (high - low)
    return out


def evaluate_k_range(p32, z32, h32, config):
    """Evaluate H32 hierarchical consolidations without changing or retraining the AE."""
    h_distance = np.maximum(1.0 - cosine_matrix(h32, config.epsilon), 0.0)
    np.fill_diagonal(h_distance, 0.0)
    results, artifacts = [], {}
    n = p32.shape[0]
    tiny_cutoff = max(config.tiny_niche_min_cells,
                      int(np.ceil(config.tiny_niche_fraction_threshold * n)))
    for k in range(config.k_min, config.k_max + 1):
        groups = hierarchical_program_groups(h32, k, config.epsilon)
        q, h, labels, confidence, mapping = consolidate_programs(
            p32, z32, h32, groups, config.epsilon)
        counts = np.bincount(labels, minlength=k)
        fractions = counts / n
        h_sep = 1.0 - cosine_matrix(h, config.epsilon)
        off = h_sep[np.triu_indices(k, 1)]
        results.append({
            "K": k,
            "silhouette": float(silhouette_score(h_distance, groups, metric="precomputed")),
            "program_cluster_sizes": json.dumps(np.bincount(groups, minlength=k).tolist()),
            "cell_niche_counts": json.dumps(counts.tolist()),
            "cell_level_niche_usage": json.dumps(np.asarray(q).mean(0).tolist()),
            "max_niche_cell_fraction": float(fractions.max()),
            "tiny_niche_fraction": float(fractions[counts < tiny_cutoff].sum()),
            "mean_assignment_confidence": float(confidence.mean()),
            "median_assignment_confidence": float(np.median(confidence)),
            "mean_H_program_separation": float(off.mean()),
            "minimum_H_program_separation": float(off.min()),
        })
        artifacts[k] = {"groups": groups, "labels": labels.astype(np.int16), "mapping": mapping}

    table = pd.DataFrame(results)
    table["ari_previous_K"] = np.nan
    table["nmi_previous_K"] = np.nan
    for row in range(1, len(table)):
        previous, current = int(table.loc[row - 1, "K"]), int(table.loc[row, "K"])
        left, right = artifacts[previous]["labels"], artifacts[current]["labels"]
        table.loc[row, "ari_previous_K"] = adjusted_rand_score(left, right)
        table.loc[row, "nmi_previous_K"] = normalized_mutual_info_score(left, right)
    table["ari_next_K"] = table.ari_previous_K.shift(-1)
    table["nmi_next_K"] = table.nmi_previous_K.shift(-1)
    table["adjacent_K_stability"] = table[["ari_previous_K", "nmi_previous_K",
                                            "ari_next_K", "nmi_next_K"]].mean(axis=1)

    table["selection_score"] = (
        .30 * _scale01(table.silhouette)
        + .25 * _scale01(table.adjacent_K_stability)
        + .20 * _scale01(1.0 - table.max_niche_cell_fraction)
        + .15 * _scale01(table.mean_H_program_separation)
        + .10 * _scale01(1.0 - table.tiny_niche_fraction)
    )
    table["eligible"] = ((table.max_niche_cell_fraction <= config.max_dominant_cell_fraction)
                         & (table.tiny_niche_fraction <= config.max_tiny_cell_fraction))
    eligible = table.loc[table.eligible]
    fallback = False
    if eligible.empty:
        fallback = True
        limit = table.max_niche_cell_fraction.min() + .05
        eligible = table.loc[table.max_niche_cell_fraction <= limit]
    best_score = eligible.selection_score.max()
    plateau = eligible.loc[eligible.selection_score >= best_score - config.near_optimal_score_tolerance]
    selected_k = int(plateau.K.min())
    table["selected"] = table.K.eq(selected_k)
    reason = ("smallest K on the near-optimal stable plateau after excluding solutions with "
              f">{config.max_dominant_cell_fraction:.0%} dominant-cell fraction or "
              f">{config.max_tiny_cell_fraction:.0%} cells in tiny niches")
    if fallback:
        reason = "fallback to the simplest near-optimal solution among the least-dominant candidates"
    for k in {selected_k, config.final_niches}:
        groups = artifacts[k]["groups"]
        q, h, labels, confidence, mapping = consolidate_programs(
            p32, z32, h32, groups, config.epsilon)
        artifacts[k].update({"Q": q, "H": h, "labels": labels.astype(np.int16),
                             "confidence": confidence, "mapping": mapping,
                             "counts": np.bincount(labels, minlength=k)})
    return table, artifacts, selected_k, reason


def _latent_block(p32, activity, rows, activity_center, activity_scale, activity_weight):
    p = np.asarray(p32[rows], dtype=np.float32)
    a = (np.log1p(np.asarray(activity[rows], dtype=np.float32)) - activity_center) / activity_scale
    return np.column_stack((np.sqrt(np.maximum(p, 0)), activity_weight * a)).astype(np.float32)


def _soft_membership(distances, scale, epsilon=1e-8):
    logits = -np.square(distances) / max(float(scale), epsilon)
    logits -= logits.max(1, keepdims=True)
    weights = np.exp(logits)
    return weights / np.maximum(weights.sum(1, keepdims=True), epsilon)


def evaluate_cell_latent_k_range(p32, activity, h32, coordinates, config, chunk_size=50000):
    """Select K from cell-level program usage; the trained AE/P32/H32 stay unchanged."""
    from scipy.spatial import cKDTree

    n = len(p32)
    rng = np.random.default_rng(config.seed)
    sample = np.sort(rng.choice(n, min(n, config.clustering_sample_size), replace=False))
    log_activity = np.log1p(np.asarray(activity[sample], dtype=np.float32))
    center = float(np.median(log_activity))
    scale = float(np.std(log_activity)) or 1.0
    sample_x = _latent_block(p32, activity, sample, center, scale, config.activity_weight)
    global_p_mean = np.asarray(p32).mean(0)
    global_activity_mean = float(np.asarray(activity).mean())
    spatial_probe = sample[:min(len(sample), config.silhouette_sample_size)]
    tree = cKDTree(np.asarray(coordinates, dtype=np.float32))
    _, nearest = tree.query(np.asarray(coordinates[spatial_probe], dtype=np.float32), k=2)
    nearest = nearest[:, 1]
    results, candidates = [], {}
    tiny_cutoff = max(config.tiny_niche_min_cells,
                      int(np.ceil(config.tiny_niche_fraction_threshold * n)))
    for k in range(config.k_min, config.k_max + 1):
        model = MiniBatchKMeans(n_clusters=k, random_state=config.seed, batch_size=4096,
                                n_init=5, max_iter=100, reassignment_ratio=0.01)
        model.fit(sample_x)
        sample_distance = model.transform(sample_x)
        membership_scale = float(np.median(np.square(sample_distance.min(1)))) or 1.0
        sample_confidence = _soft_membership(sample_distance, membership_scale, config.epsilon).max(1)
        labels = np.empty(n, dtype=np.int16)
        confidence_sum = 0.0
        q_sums = np.zeros(k, dtype=np.float64)
        for start in range(0, n, chunk_size):
            stop = min(start + chunk_size, n)
            rows = slice(start, stop)
            x = _latent_block(p32, activity, rows, center, scale, config.activity_weight)
            distance = model.transform(x)
            q = _soft_membership(distance, membership_scale, config.epsilon)
            block_labels = distance.argmin(1)
            labels[start:stop] = block_labels
            confidence_sum += q.max(1).sum()
            q_sums += q.sum(0)
        # The first 32 centroid coordinates are means in Hellinger space; squaring
        # yields a compact cell-level program profile without another full matrix pass.
        mean_p = np.square(np.maximum(model.cluster_centers_[:, :p32.shape[1]], 0))
        mean_p /= np.maximum(mean_p.sum(1, keepdims=True), config.epsilon)
        h = mean_p @ np.asarray(h32, dtype=np.float64)
        h /= np.maximum(h.sum(1, keepdims=True), config.epsilon)
        h_sep = 1.0 - cosine_matrix(h, config.epsilon)
        off = h_sep[np.triu_indices(k, 1)]
        counts = np.bincount(labels, minlength=k)
        fractions = counts / n
        probe_labels = labels[spatial_probe]
        silhouette = silhouette_score(sample_x, model.labels_, metric="euclidean",
                                      sample_size=min(config.silhouette_sample_size, len(sample_x)),
                                      random_state=config.seed)
        program_niche = mean_p.argmax(0)
        mapping = pd.DataFrame({
            "program_id": np.arange(1, p32.shape[1] + 1),
            "final_niche_id": program_niche + 1,
            "mean_mixture": global_p_mean,
            "mean_exposure": global_activity_mean * global_p_mean,
            "niche_specific_usage": mean_p[program_niche, np.arange(p32.shape[1])],
        })
        results.append({
            "K": k, "silhouette": float(silhouette),
            "program_cluster_sizes": json.dumps(np.bincount(program_niche, minlength=k).tolist()),
            "cell_niche_counts": json.dumps(counts.tolist()),
            "cell_level_niche_usage": json.dumps((q_sums / n).tolist()),
            "max_niche_cell_fraction": float(fractions.max()),
            "tiny_niche_fraction": float(fractions[counts < tiny_cutoff].sum()),
            "mean_assignment_confidence": float(confidence_sum / n),
            "median_assignment_confidence": float(np.median(sample_confidence)),
            "mean_H_program_separation": float(off.mean()),
            "minimum_H_program_separation": float(off.min()),
            "spatial_neighbor_agreement": float(np.mean(probe_labels == labels[nearest])),
        })
        candidates[k] = {"model": model, "labels": labels, "H": h.astype(np.float32),
                         "mapping": mapping, "counts": counts, "membership_scale": membership_scale,
                         "activity_center": center, "activity_scale": scale}

    table = pd.DataFrame(results)
    table["ari_previous_K"] = np.nan; table["nmi_previous_K"] = np.nan
    for row in range(1, len(table)):
        a = candidates[int(table.loc[row - 1, "K"])]["labels"]
        b = candidates[int(table.loc[row, "K"])]["labels"]
        table.loc[row, "ari_previous_K"] = adjusted_rand_score(a, b)
        table.loc[row, "nmi_previous_K"] = normalized_mutual_info_score(a, b)
    table["ari_next_K"] = table.ari_previous_K.shift(-1)
    table["nmi_next_K"] = table.nmi_previous_K.shift(-1)
    table["adjacent_K_stability"] = table[["ari_previous_K", "nmi_previous_K",
                                            "ari_next_K", "nmi_next_K"]].mean(axis=1)
    table["selection_score"] = (
        .25 * _scale01(table.silhouette) + .20 * _scale01(table.adjacent_K_stability)
        + .20 * _scale01(1 - table.max_niche_cell_fraction)
        + .15 * _scale01(table.spatial_neighbor_agreement)
        + .10 * _scale01(table.mean_H_program_separation)
        + .10 * _scale01(1 - table.tiny_niche_fraction))
    table["eligible"] = ((table.max_niche_cell_fraction <= config.max_dominant_cell_fraction)
                         & (table.tiny_niche_fraction <= config.max_tiny_cell_fraction))
    eligible = table.loc[table.eligible]
    if eligible.empty:
        eligible = table.loc[table.max_niche_cell_fraction <= table.max_niche_cell_fraction.min() + .05]
    best = eligible.selection_score.max()
    selected_k = int(eligible.loc[eligible.selection_score >= best - config.near_optimal_score_tolerance, "K"].min())
    table["selected"] = table.K.eq(selected_k)
    reason = "smallest K on the near-optimal cell-latent stability/quality plateau"

    # Materialize soft Q only for the selected solution; all other diagnostics use hard labels.
    chosen = candidates[selected_k]
    q = np.empty((n, selected_k), dtype=np.float32)
    confidence = np.empty(n, dtype=np.float32)
    for start in range(0, n, chunk_size):
        stop = min(start + chunk_size, n); rows = slice(start, stop)
        x = _latent_block(p32, activity, rows, center, scale, config.activity_weight)
        block = _soft_membership(chosen["model"].transform(x), chosen["membership_scale"], config.epsilon)
        q[start:stop] = block; confidence[start:stop] = block.max(1)
    chosen.update({"Q": q, "confidence": confidence})
    return table, candidates, selected_k, reason
