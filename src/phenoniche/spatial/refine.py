"""Context-local resampling, agreement and connected candidate regions."""

import warnings

import numpy as np
from scipy.sparse.csgraph import connected_components

from .programs import fit_programs, match_programs, project, sample_blocks, top_channels


def components(graph, mask, minimum):
    selected = np.flatnonzero(mask)
    if len(selected) < minimum:
        return
    _, assignment = connected_components(graph[selected][:, selected], directed=False)
    for group in np.split(
        np.argsort(assignment, kind="stable"), np.cumsum(np.bincount(assignment))[:-1]
    ):
        if len(group) >= minimum:
            yield selected[group]


def refine_context(
    ids, key, block, graph, tensor, cfg, rng, diagnostics=None, lr_weights=None
):
    stats = {} if diagnostics is None else diagnostics
    stats.update(
        windows=len(ids),
        completed_fits=0,
        recurrent_programs=0,
        voted_centers=0,
        connected_candidates=0,
        audited_candidates=0,
    )
    if len(ids) < 2 * cfg.min_cells:
        return [], None

    def for_cp(fiber):
        if lr_weights is None:
            return fiber
        s, r, values, opportunity = fiber
        return s, r, values * lr_weights, opportunity

    models = []
    reference = None
    for _ in range(cfg.repeats):
        sample = sample_blocks(ids, block, cfg, rng)
        if len(sample) < cfg.rank:
            continue
        fibers = [for_cp(tensor.fibers(i, key)) for i in sample]
        factors = fit_programs(fibers, tensor.n_types, cfg, rng)
        if factors is None:
            continue
        if reference is None:
            reference = factors
        i, j, cos = match_programs(reference, factors)
        keep = cos >= cfg.loading_cosine
        activities = np.array([project(f, factors) for f in fibers if len(f[0])])
        with warnings.catch_warnings():
            warnings.simplefilter(
                "ignore", RuntimeWarning
            )  # Entirely unobserved program.
            baseline = np.nanmedian(activities, axis=0)
            mad = 1.4826 * np.nanmedian(abs(activities - baseline), axis=0)
            scale = np.maximum(
                mad,
                np.maximum(cfg.relative_floor * np.nanmean(activities, axis=0), 1e-8),
            )
        models.append((factors, i[keep], j[keep], baseline, scale))
        del fibers
    stats["completed_fits"] = len(models)
    if len(models) < cfg.agreement * cfg.repeats:
        return [], None
    channels = top_channels(reference)
    # A reference loading alone is insufficient: require the same named channel
    # among the strongest loadings of aligned, independently refitted programs.
    hits = [{row[:3]: 0 for row in rows} for rows in channels]
    for factors, ref_idx, fit_idx, _, _ in models:
        other_channels = top_channels(factors)
        for p, q in zip(ref_idx, fit_idx):
            for row in other_channels[q]:
                if row[:3] in hits[p]:
                    hits[p][row[:3]] += 1
    channels = [
        [row for row in rows if hits[p][row[:3]] >= cfg.agreement * cfg.repeats]
        for p, rows in enumerate(channels)
    ]
    stats["recurrent_programs"] = sum(bool(rows) for rows in channels)
    rank = len(channels)
    votes = np.zeros((len(ids), rank, 2), dtype=np.int32)
    activity = np.full((len(ids), rank), np.nan, dtype=np.float32)
    observed = [
        np.full((len(ids), len(rows)), np.nan, dtype=np.float32) for rows in channels
    ]
    for pos, cell in enumerate(ids):
        f = tensor.fibers(cell, key)
        if not len(f[0]):
            continue
        indices = {(s, r): j for j, (s, r) in enumerate(zip(f[0], f[1]))}
        for p, rows in enumerate(channels):
            for k, (s, r, lr, _) in enumerate(rows):
                if (s, r) in indices:
                    observed[p][pos, k] = f[2][indices[s, r], lr]
        fitted_fiber = for_cp(f)
        for rep, (factors, i, j, baseline, scale) in enumerate(models):
            score = project(fitted_fiber, factors)
            if rep == 0:
                activity[pos] = score
            z = (score[j] - baseline[j]) / scale[j]
            votes[pos, i, 0] += z <= -cfg.effect_mad
            votes[pos, i, 1] += z >= cfg.effect_mad
    stats["voted_centers"] = int(
        (votes >= cfg.agreement * cfg.repeats).any(axis=(1, 2)).sum()
    )
    local = graph[ids][:, ids]
    candidates = []
    for p, rows in enumerate(channels):
        for sign_index, sign in enumerate((-1, 1)):
            support = votes[:, p, sign_index] / cfg.repeats
            for region in components(local, support >= cfg.agreement, cfg.min_cells):
                stats["connected_candidates"] += 1
                audit = RegionAudit(ids, local, rows, observed[p], sign)
                if audit.contrast(ids[region]):
                    candidates.append((ids[region], support[region], p, sign, audit))
    stats["audited_candidates"] = len(candidates)
    return candidates, (ids, reference, activity)


class RegionAudit:
    def __init__(self, ids, graph, channels, values, sign):
        self.ids, self.graph = ids, graph
        self.channels, self.values, self.sign = channels, values, sign

    def contrast(self, cells):
        region = np.searchsorted(self.ids, cells)
        rim = np.setdiff1d(np.unique(self.graph[region].indices), region)
        evidence = []
        for k, (s, r, lr, loading) in enumerate(self.channels):
            inside = self.values[region, k]
            outside = self.values[rim, k]
            inside, outside = inside[np.isfinite(inside)], outside[np.isfinite(outside)]
            if min(len(inside), len(outside)) >= 2:
                delta = inside.mean() - outside.mean()
                if self.sign * delta > 0:
                    evidence.append(
                        (s, r, lr, loading, float(delta), len(inside), len(outside))
                    )
        return evidence


def assign(coarse, graph, candidates, minimum):
    """Conflicting equal-strength proposals defer to the coarse assignment."""
    winner = np.full(len(coarse), -1)
    strength = np.zeros(len(coarse))
    for k, (key, ids, scores, *_) in enumerate(candidates):
        better = scores > strength[ids]
        tied = scores == strength[ids]
        winner[ids[tied]] = -1
        winner[ids[better]] = k
        strength[ids[better]] = scores[better]
    labels = np.array([f"C{x}" for x in coarse], dtype=object)
    region_ids = np.full(len(coarse), "", dtype=object)
    classes = {}
    accepted = []
    for k, (key, ids, _, p, sign, audit) in enumerate(candidates):
        remaining = ids[winner[ids] == k]
        subgraph = graph[remaining][:, remaining]
        for part in components(subgraph, np.ones(len(remaining), dtype=bool), minimum):
            cells = remaining[part]
            if key[0] != key[1] and len(np.unique(coarse[cells])) < 2:
                continue
            evidence = audit.contrast(cells)
            if not evidence:
                continue
            # Evidence follows reference-loading order. Recurrent regions must
            # agree on context, program, sign and their leading observed channel.
            signature = (key, p, sign, evidence[0][:3])
            label = classes.setdefault(signature, f"F{len(classes) + 1}")
            region = f"R{len(accepted) + 1}"
            region_ids[cells] = region
            labels[cells] = label
            accepted.append(
                (label, region, key, p, sign, evidence, len(cells), len(ids))
            )
    return labels, region_ids, accepted, strength
