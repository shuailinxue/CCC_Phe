"""Specimen-local spatial adjacency, label permutations, and manual interfaces."""

from itertools import combinations

import numpy as np
import pandas as pd
from scipy.spatial import Delaunay, cKDTree


def delaunay_edges(xyz, samples, max_um):
    """Return undirected Delaunay edges, filtered in physical coordinates."""
    edges = []
    for specimen in np.unique(samples):
        ids = np.flatnonzero(samples == specimen)
        points, first = np.unique(xyz[ids], axis=0, return_index=True)
        ids = ids[first]
        if len(points) < 2:
            continue
        centered = points - points.mean(axis=0)
        _, singular, axes = np.linalg.svd(centered, full_matrices=False)
        rank = np.count_nonzero(
            singular > singular[0] * max(centered.shape) * np.finfo(float).eps
        )
        if rank == 1:
            order = np.argsort(centered @ axes[0])
            pairs = np.column_stack((order[:-1], order[1:]))
        else:
            projected = centered @ axes[:rank].T
            simplices = (
                np.arange(len(ids))[None, :]
                if len(ids) == rank + 1
                else Delaunay(projected).simplices
            )
            pairs = np.concatenate(
                [simplices[:, [a, b]] for a, b in combinations(range(rank + 1), 2)]
            )
            pairs = np.unique(np.sort(pairs, axis=1), axis=0)
        a, b = ids[pairs].T
        keep = np.linalg.norm(xyz[a] - xyz[b], axis=1) <= max_um
        edges.append(np.sort(np.column_stack((a[keep], b[keep])), axis=1))
    return np.concatenate(edges) if edges else np.empty((0, 2), dtype=int)


def neighborhood_enrichment(edges, coarse, samples, permutations, seed):
    """Count each undirected edge once and shuffle labels within each specimen."""
    classes, labels = np.unique(coarse, return_inverse=True)
    n = len(classes)
    left, right = edges.T
    a, b = np.triu_indices(n)

    def codes(values):
        u, v = values[left], values[right]
        return np.minimum(u, v) * n + np.maximum(u, v)

    def counts(codes):
        return np.bincount(codes, minlength=n * n).reshape(n, n)[a, b]

    observed_codes = codes(labels)
    observed = counts(observed_codes)
    # Use unique (class pair, cell) combinations to count endpoint support.
    cells = np.r_[left, right]
    pair_cell = np.unique(np.r_[observed_codes, observed_codes] * len(coarse) + cells)
    pair, cell = np.divmod(pair_cell, len(coarse))
    low_side = labels[cell] == pair // n
    support_low = counts(pair[low_side])
    support_high = counts(pair[~low_side])
    support_high[a == b] = support_low[a == b]

    rng = np.random.default_rng(seed)
    groups = [np.flatnonzero(samples == specimen) for specimen in np.unique(samples)]
    shuffled = labels.copy()
    mean = np.zeros(len(a))
    m2 = np.zeros(len(a))
    for i in range(1, permutations + 1):
        for ids in groups:
            shuffled[ids] = rng.permutation(labels[ids])
        delta = counts(codes(shuffled)) - mean
        mean += delta / i
        m2 += delta * delta * ((i - 1) / i)
    sd = np.sqrt(m2 / permutations)
    z = np.divide(observed - mean, sd, out=np.full_like(mean, np.nan), where=sd > 0)
    return pd.DataFrame(dict(
        coarse_a=classes[a], coarse_b=classes[b], observed_edges=observed,
        contact_cells_a=support_low, contact_cells_b=support_high,
        null_mean=mean, null_sd=sd, z_score=z,
    ))


def interface_cells(xyz, samples, coarse, edges, pair, interface_um):
    """Select cells on either flank near a cross-class edge midpoint."""
    a, b = sorted(pair)
    if a == b or not np.isin([a, b], coarse).all():
        raise ValueError("Choose two distinct existing coarse class IDs")
    left, right = edges.T
    cross = (
        ((coarse[left] == a) & (coarse[right] == b))
        | ((coarse[left] == b) & (coarse[right] == a))
    )
    result = []
    for specimen in np.unique(samples[left[cross]]):
        local = cross & (samples[left] == specimen)
        anchors = (xyz[left[local]] + xyz[right[local]]) / 2
        ids = np.flatnonzero((samples == specimen) & np.isin(coarse, [a, b]))
        distance, _ = cKDTree(anchors).query(xyz[ids])
        result.extend(ids[distance <= interface_um])
    if not result:
        raise ValueError(f"No interface cells for coarse classes {a} and {b} at this radius")
    return np.sort(result)
