"""Assign cells to composition interiors and supported interfaces."""

import numpy as np

from .geometry import contexts


def partition_contexts(graph, coarse, cfg):
    """Use bilateral contact support and assign a primary interface before fitting."""
    groups = contexts(graph, coarse)
    sizes = np.bincount(coarse)
    degrees = np.diff(graph.indptr)
    best = np.zeros(len(coarse))
    owner = np.full(len(coarse), -1)
    interfaces = []
    summary = []
    for (a, b), ids in groups.items():
        if a == b:
            continue
        left = ids[coarse[ids] == a]
        right = ids[coarse[ids] == b]
        cross = graph[left][:, right].getnnz(axis=1) / degrees[left]
        reverse = graph[right][:, left].getnnz(axis=1) / degrees[right]
        fa, fb = len(left) / sizes[a], len(right) / sizes[b]
        selected = (
            min(len(left), len(right)) >= cfg.min_cells
            and min(fa, fb) >= cfg.interface_fraction
        )
        summary.append((a, b, len(left), len(right), fa, fb, selected))
        if not selected:
            continue
        ids = np.r_[left, right]
        score = np.r_[cross, reverse]
        better = score > best[ids]
        owner[ids[better]] = len(interfaces)
        best[ids[better]] = score[better]
        interfaces.append((a, b))
    # Recheck support after competing interfaces have claimed centers.
    for i, key in enumerate(interfaces):
        cells = np.flatnonzero(owner == i)
        if any(np.count_nonzero(coarse[cells] == side) < cfg.min_cells for side in key):
            owner[cells] = -1
    result = {
        (c, c): np.flatnonzero((coarse == c) & (owner < 0)) for c in np.unique(coarse)
    }
    result.update({key: np.flatnonzero(owner == i) for i, key in enumerate(interfaces)})
    return {key: ids for key, ids in result.items() if len(ids)}, summary
