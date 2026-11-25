"""Partition each composition interior or spatial interface by its CP activity profiles."""

from dataclasses import dataclass

import igraph as ig
import leidenalg
import numpy as np
from scipy import sparse
from scipy.spatial import cKDTree
from sklearn.metrics import adjusted_rand_score, silhouette_score

from .geometry import contexts
from .programs import fit_programs, project, sample_blocks, top_channels


def partition_contexts(graph, coarse, cfg, flank_fraction=0):
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
        keep_left, keep_right = cross >= flank_fraction, reverse >= flank_fraction
        left, right = left[keep_left], right[keep_right]
        fa, fb = len(left) / sizes[a], len(right) / sizes[b]
        selected = (
            min(len(left), len(right)) >= cfg.min_cells
            and min(fa, fb) >= cfg.interface_fraction
        )
        summary.append((a, b, len(left), len(right), fa, fb, selected))
        if not selected:
            continue
        ids = np.r_[left, right]
        score = np.r_[cross[keep_left], reverse[keep_right]]
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


def _activity_graph(activity, neighbors):
    # Keep the amplitude of the entire program and balance component scales.
    scale = np.maximum(np.quantile(activity, 0.9, axis=0), 1e-7)
    scaled = np.log1p(activity / scale)
    points, inverse, sizes = np.unique(
        scaled.round(6), axis=0, return_inverse=True, return_counts=True
    )
    if len(points) < 2:
        return None, inverse, scaled
    k = min(neighbors + 1, max(3, int(np.sqrt(len(points))) + 1), len(points))
    distances, nearby = cKDTree(points).query(points, k=k)
    a = np.repeat(np.arange(len(points)), k)
    b = nearby.ravel()
    keep = a != b
    a, b = a[keep], b[keep]
    local_scale = np.maximum(distances[:, -1], 1e-8)
    weights = np.exp(
        -distances.ravel()[keep] / np.sqrt(local_scale[a] * local_scale[b])
    )
    weights *= np.sqrt(sizes[a] * sizes[b])
    adj = sparse.csr_matrix((weights, (a, b)), shape=(len(points),) * 2)
    edges = sparse.triu(adj.maximum(adj.T), k=1).tocoo()
    graph = ig.Graph(len(points), np.column_stack([edges.row, edges.col]))
    graph.es["weight"] = edges.data
    return graph, inverse, scaled


def _partitions(activity, cfg, seed):
    graph, inverse, scaled = _activity_graph(activity, cfg.communication_neighbors)
    result = {}
    if graph is not None:
        for resolution in (0.03, 0.07, 0.15, 0.3, 0.6, 1.0):
            partition = leidenalg.find_partition(
                graph,
                leidenalg.RBConfigurationVertexPartition,
                weights="weight",
                resolution_parameter=resolution,
                seed=seed,
            )
            labels = np.asarray(partition.membership)[inverse]
            sizes = np.bincount(labels)
            if 2 <= len(sizes) <= cfg.max_subclasses and sizes.min() >= cfg.min_cells:
                result[resolution] = labels
    return result, scaled


@dataclass
class ContextPartition:
    ids: np.ndarray
    labels: np.ndarray
    status: str
    stability: float
    resolution: float | None
    program: tuple | None
    channels: list


def cluster_context(ids, key, block, tensor, cfg, rng):
    fallback = np.zeros(len(ids), dtype=int)
    if len(ids) < 2 * cfg.min_cells:
        return ContextPartition(ids, fallback, "small_context", 0, None, None, [])
    models = []
    weights = None

    def cp_fiber(f):
        if weights is None:
            return f
        s, r, y, o = f
        return s, r, y * weights, o

    for _ in range(cfg.repeats):
        sampled = sample_blocks(ids, block, cfg, rng)
        if len(sampled) >= cfg.rank:
            fibers = [tensor.fibers(i, key) for i in sampled]
            if weights is None and getattr(tensor, "balance_channels", False):
                values = np.concatenate([f[2] for f in fibers])
                if len(values):
                    rms = np.sqrt(np.mean(np.square(values), axis=0))
                    if np.any(rms > 0):
                        typical = np.median(rms[rms > 0])
                        weights = np.sqrt(
                            np.clip(typical / np.maximum(rms, typical / 9), 1 / 9, 9)
                        )
            factors = fit_programs(
                [cp_fiber(f) for f in fibers],
                tensor.n_types,
                cfg,
                rng,
                getattr(tensor, "n_senders", None),
            )
            if factors is not None:
                models.append(factors)
    if not models:
        return ContextPartition(ids, fallback, "no_program", 0, None, None, [])
    top = top_channels(models[0])
    selected = sorted({row[:3] for rows in top for row in rows})
    measured = np.full((len(ids), len(selected)), np.nan, dtype=np.float32)
    scores = [
        np.full((len(ids), h.shape[1]), np.nan, dtype=np.float32) for _, _, h in models
    ]
    for row, cell in enumerate(ids):
        f = tensor.fibers(cell, key)
        if not len(f[0]):
            continue
        scaled_fiber = cp_fiber(f)
        for output, model in zip(scores, models):
            output[row] = project(scaled_fiber, model)
        lookup = {(s, r): i for i, (s, r) in enumerate(zip(*f[:2]))}
        for j, (s, r, lr) in enumerate(selected):
            fiber = lookup.get((s, r))
            if fiber is not None:
                measured[row, j] = f[2][fiber, lr]
    program = (ids, models[0], scores[0])
    # Missing support remains in the parent class; no imputed zero vectors enter Leiden.
    valid = np.logical_and.reduce([np.isfinite(s).all(axis=1) for s in scores])
    if valid.sum() < 2 * cfg.min_cells or len(models) < 2:
        return ContextPartition(ids, fallback, "limited_support", 0, None, program, [])
    graphs = [_partitions(x[valid], cfg, int(rng.integers(2**31 - 1))) for x in scores]
    reference, scaled = graphs[0]
    sample = rng.choice(len(scaled), min(len(scaled), 1200), replace=False)
    best = None
    for resolution, labels in reference.items():
        # Allow an adjacent resolution to reproduce the same partition.
        agreements = [
            max(
                (
                    adjusted_rand_score(labels, other)
                    for other in candidates.values()
                    if len(np.unique(other)) == len(np.unique(labels))
                ),
                default=0.0,
            )
            for candidates, _ in graphs[1:]
        ]
        if len(agreements) < cfg.repeats - 1:
            agreements += [0.0] * (cfg.repeats - 1 - len(agreements))
        stability = float(np.median(agreements))
        if stability < cfg.subclass_stability or len(np.unique(labels[sample])) < 2:
            continue
        separation = silhouette_score(scaled[sample], labels[sample])
        if separation >= cfg.subclass_separation and (
            best is None or separation > best[0]
        ):
            best = separation, stability, resolution, labels
    if best is None:
        return ContextPartition(ids, fallback, "one_class", 0, None, program, [])
    _, stability, resolution, labels = best
    fallback[valid] = labels + 1
    evidence = []
    for label in np.unique(labels) + 1:
        inside = fallback == label
        for p, rows in enumerate(top):
            for s, r, lr, loading in rows:
                vals = measured[:, selected.index((s, r, lr))]
                one = vals[inside & np.isfinite(vals)]
                rest = vals[valid & ~inside & np.isfinite(vals)]
                if min(len(one), len(rest)) >= 2:
                    evidence.append(
                        (
                            label,
                            p,
                            s,
                            r,
                            lr,
                            loading,
                            float(one.mean() - rest.mean()),
                            len(one),
                            len(rest),
                        )
                    )
    return ContextPartition(
        ids, fallback, "split", stability, resolution, program, evidence
    )
