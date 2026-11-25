"""Bidirectional focal-cell CP and neighborhood-level consensus clustering."""

import igraph as ig
import leidenalg
import numpy as np
from scipy import sparse
from scipy.spatial import cKDTree
from sklearn.metrics import adjusted_rand_score

from .programs import fit_programs


class CellTensor:
    """Focal cell × neighbor type × [outgoing LR, incoming LR]."""

    def __init__(self, graph, types, channels, sigma, mean=True, settings=None):
        self.graph, self.types, self.channels, self.sigma = (
            graph,
            types,
            channels,
            sigma,
        )
        self.n_types = int(types.max()) + 1
        self.mean = mean
        labels = getattr(channels, "signaling_types", np.full(len(channels.names), "Unknown"))
        self.groups = []
        for label in ("Contact", "Secreted", "ECM", "Unknown"):
            select = np.isin(labels, ["Unknown", "Ambiguous"]) if label == "Unknown" else labels == label
            columns = np.flatnonzero(np.r_[select, select])
            if not len(columns):
                continue
            radius = None if settings is None else {
                "Contact": settings.contact_radius_um, "Secreted": settings.secreted_radius_um,
                "ECM": settings.ecm_radius_um, "Unknown": settings.contact_um,
            }[label]
            scale = sigma if settings is None else {
                "Contact": settings.contact_sigma_um, "Secreted": settings.secreted_sigma_um,
                "ECM": settings.ecm_sigma_um, "Unknown": settings.sigma_um,
            }[label]
            self.groups.append((columns, radius, scale))

    def fiber(self, cell):
        start, stop = self.graph.indptr[cell : cell + 2]
        neighbors = self.graph.indices[start:stop]
        distances = self.graph.data[start:stop]
        neighbor_types = self.types[neighbors]
        categories = np.unique(neighbor_types)
        lr = self.channels
        result = np.zeros((len(categories), 2 * len(lr.names)), dtype=np.float32)
        support = np.zeros((len(categories), len(self.groups)))
        if len(neighbors):
            expression = lr.expression[np.r_[cell, neighbors]].toarray()
            focal, other = expression[0], expression[1:]
            for group, (columns, radius, sigma) in enumerate(self.groups):
                eligible = np.ones(len(distances), dtype=bool) if radius is None else distances <= radius
                weights = np.exp(np.maximum(-0.5 * (distances / sigma) ** 2, -700))
                for row, category in enumerate(categories):
                    mask = (neighbor_types == category) & eligible
                    support[row, group] = weights[mask].sum()
                    if not mask.any():
                        continue
                    incoming = weights[mask] @ other[mask]
                    if self.mean and support[row, group] > 0:
                        incoming /= support[row, group]
                    values = np.r_[focal[lr.left] * incoming[lr.right], incoming[lr.left] * focal[lr.right]]
                    result[row, columns] = values[columns]
        return categories, result, support


def _type_membership(types):
    return sparse.csr_matrix((np.ones(len(types)), (np.arange(len(types)), types)))


def contact_support(contacts, ids, types, sigma):
    graph = contacts[ids].copy()
    weights = np.exp(np.maximum(-0.5 * (graph.data / sigma) ** 2, -700))
    graph.data.fill(1)
    membership = _type_membership(types)
    counts = (graph @ membership).toarray().astype(np.int64)
    graph.data = weights
    return counts, (graph @ membership).toarray()


def program_graph(ids, allowed, windows, types, sigma):
    graph = windows[ids].copy()
    graph.data = np.exp(-0.5 * (graph.data / sigma) ** 2)
    graph = graph.multiply(allowed).tocsr()
    self_edges = sparse.csr_matrix(
        (np.ones(len(ids)), (np.arange(len(ids)), ids)), shape=graph.shape
    )
    graph = graph + self_edges
    return graph, (graph @ _type_membership(types)).toarray()


def pool_programs(graph, type_weights, types, scores, mean):
    """Pool by focal-cell type, retaining structural absence as separate support."""
    features = []
    total = type_weights.sum(axis=1)
    for category in range(type_weights.shape[1]):
        activity = graph.multiply(types == category) @ scores
        denominator = type_weights[:, category] if mean else total
        features.append(np.divide(
            activity, denominator[:, None], out=np.zeros_like(activity),
            where=denominator[:, None] > 0,
        ))
    return np.concatenate(features, axis=1)


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


def consensus(activity, cfg, seed):
    """Consensus on existing similarity edges, avoiding a quadratic co-assignment matrix."""
    graph, inverse, _ = _activity_graph(activity, cfg.communication_neighbors)
    if graph is None:
        return np.zeros(len(activity), dtype=int), 1.0
    runs = []
    for resolution in (0.1, 0.3, 0.6):
        for repeat in range(cfg.repeats):
            part = leidenalg.find_partition(
                graph,
                leidenalg.RBConfigurationVertexPartition,
                weights="weight",
                resolution_parameter=resolution,
                seed=(seed + repeat) % (2**31 - 1),
            )
            runs.append(np.asarray(part.membership))
    edges = np.asarray(graph.get_edgelist(), dtype=int)
    votes = np.mean(
        [labels[edges[:, 0]] == labels[edges[:, 1]] for labels in runs], axis=0
    )
    weights = np.asarray(graph.es["weight"]) * votes
    good = weights > 0
    consensus_graph = ig.Graph(graph.vcount(), edges[good].tolist())
    consensus_graph.es["weight"] = weights[good].tolist()
    result = leidenalg.find_partition(
        consensus_graph,
        leidenalg.RBConfigurationVertexPartition,
        weights="weight",
        resolution_parameter=0.3,
        seed=seed % (2**31 - 1),
    )
    labels = np.asarray(result.membership)[inverse]
    stability = float(
        np.median([adjusted_rand_score(labels, r[inverse]) for r in runs])
    )
    return labels, stability


def fit_context(ids, windows, contacts, types, channels, cfg, seed):
    allowed = np.zeros(len(types), dtype=bool)
    allowed[ids] = True
    tensor = CellTensor(contacts, types, channels, cfg.sigma_um, cfg.aggregation != "sum-sum", cfg)
    pool_graph, pool_weights = program_graph(ids, allowed, windows, types, cfg.sigma_um)
    rng = np.random.default_rng(seed)
    models, features, cell_scores, training = [], [], [], []
    for repeat in range(cfg.repeats):
        print(f"  CP repeat {repeat + 1}/{cfg.repeats}: full fit", flush=True)
        fit = fit_programs(ids, tensor.fiber, tensor.n_types, 2 * len(channels.names), cfg, rng)
        if fit is None:
            break
        activity, _, _ = fit.factors
        scores = np.zeros((len(types), activity.shape[1]), dtype=activity.dtype)
        scores[ids] = activity
        models.append(fit.factors)
        cell_scores.append(activity)
        training.append(fit)
        features.append(
            pool_programs(pool_graph, pool_weights, types, scores, cfg.aggregation == "mean-mean")
        )
    if not models:
        return dict(
            ids=ids,
            labels=np.zeros(len(ids), dtype=int),
            status="no_program",
            stability=0.0,
            models=[],
            cell_scores=[],
            feature_scores=[],
            training=training,
            pool_weights=pool_weights,
        )
    labels, graph_stability = consensus(features[0], cfg, seed)
    if len(features) > 1:
        other = [consensus(f, cfg, seed + i + 1)[0] for i, f in enumerate(features[1:])]
        stability = min(
            graph_stability, np.median([adjusted_rand_score(labels, x) for x in other])
        )
    else:
        stability = 0.0
    # Retain reproducibility as a reported statistic; provide all cells a label.
    return dict(
        ids=ids,
        labels=labels,
        status="clustered",
        stability=float(stability),
        models=models,
        cell_scores=cell_scores,
        feature_scores=features,
        training=training,
        pool_weights=pool_weights,
    )
