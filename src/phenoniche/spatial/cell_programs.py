"""Bidirectional focal-cell CP and neighborhood-level consensus clustering."""

import igraph as ig
import leidenalg
import numpy as np
from scipy import sparse
from sklearn.metrics import adjusted_rand_score

from .programs import fit_programs, project, sample_blocks
from .subclasses import _activity_graph


class CellTensor:
    """Focal cell × neighbor type × [outgoing LR, incoming LR]."""

    def __init__(self, graph, types, channels, sigma):
        self.graph, self.types, self.channels, self.sigma = (
            graph,
            types,
            channels,
            sigma,
        )
        self.n_types = int(types.max()) + 1

    def fiber(self, cell, allowed):
        start, stop = self.graph.indptr[cell : cell + 2]
        neighbors = self.graph.indices[start:stop]
        distances = self.graph.data[start:stop]
        keep = allowed[neighbors]
        neighbors, distances = neighbors[keep], distances[keep]
        neighbor_types = self.types[neighbors]
        categories = np.unique(neighbor_types)
        lr = self.channels
        result = np.zeros((len(categories), 2 * len(lr.names)), dtype=np.float32)
        if len(neighbors):
            expression = lr.expression[np.r_[cell, neighbors]].toarray()
            focal, other = expression[0], expression[1:]
            weights = np.exp(-0.5 * (distances / self.sigma) ** 2)
            for row, category in enumerate(categories):
                mask = neighbor_types == category
                incoming = weights[mask] @ other[mask]
                result[row, : len(lr.names)] = focal[lr.left] * incoming[lr.right]
                result[row, len(lr.names) :] = incoming[lr.left] * focal[lr.right]
        # The existing CP solver's singleton sender mode lets it fit the 3D tensor.
        return (
            np.zeros(len(categories), dtype=int),
            categories,
            result,
            np.ones(len(categories)),
        )


def pool_programs(ids, allowed, windows, types, scores, sigma):
    """Aggregate each focal cell's activity into its 3D neighborhood by cell type."""
    graph = windows[ids].copy()
    graph.data = np.exp(-0.5 * (graph.data / sigma) ** 2)
    graph = graph.multiply(allowed).tocsr()
    self_edges = sparse.csr_matrix(
        (np.ones(len(ids)), (np.arange(len(ids)), ids)), shape=graph.shape
    )
    graph = graph + self_edges
    features = []
    for category in range(types.max() + 1):
        mask = (types == category) & allowed
        features.append(graph.multiply(mask) @ scores)
    denominator = np.asarray(graph.sum(axis=1)).ravel()
    return np.concatenate(features, axis=1) / np.maximum(denominator[:, None], 1e-12)


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


def fit_context(ids, key, block, windows, contacts, types, channels, cfg, seed):
    allowed = np.zeros(len(types), dtype=bool)
    allowed[ids] = True
    tensor = CellTensor(contacts, types, channels, cfg.sigma_um)
    rng = np.random.default_rng(seed)
    models, features, cell_scores = [], [], []
    for repeat in range(cfg.repeats):
        print(
            f"  CP repeat {repeat + 1}/{cfg.repeats}: fitting and projecting",
            flush=True,
        )
        train = sample_blocks(ids, block, cfg, rng)
        if not len(train):
            continue
        model = fit_programs(
            [tensor.fiber(i, allowed) for i in train],
            tensor.n_types,
            cfg,
            rng,
            n_senders=1,
        )
        if model is None:
            continue
        scores = np.zeros((len(types), model[2].shape[1]), dtype=np.float32)
        for position, i in enumerate(ids):
            if position and position % 10000 == 0:
                print(f"    projected {position:,}/{len(ids):,} cells", flush=True)
            value = project(tensor.fiber(i, allowed), model)
            scores[i] = np.nan_to_num(value)
        models.append(model)
        cell_scores.append(scores[ids])
        features.append(
            pool_programs(ids, allowed, windows, types, scores, cfg.sigma_um)
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
    )
