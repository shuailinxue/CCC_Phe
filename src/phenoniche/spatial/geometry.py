"""Physical edges never cross specimens. Composition neighbors may connect distant sites."""

import igraph as ig
import leidenalg
import numpy as np
from scipy import sparse
from scipy.spatial import cKDTree


def physical_graph(xyz, specimen, radius):
    row, col, distance = [], [], []
    for sample in np.unique(specimen):
        ids = np.flatnonzero(specimen == sample)
        pairs = cKDTree(xyz[ids]).query_pairs(radius, output_type="ndarray")
        left, right = ids[pairs].T
        row.extend([left, right])
        col.extend([right, left])
        d = np.linalg.norm(xyz[left] - xyz[right], axis=1)
        distance.extend([d, d])
    n = len(xyz)
    return sparse.csr_matrix(
        (np.concatenate(distance), (np.concatenate(row), np.concatenate(col))),
        shape=(n, n),
    )


def composition_partition(graph, types, settings, workers=1, cache_path=None):
    n = len(types)
    weights = graph.copy()
    weights.data = np.exp(-0.5 * (weights.data / settings.sigma_um) ** 2)
    weights.setdiag(1)
    one_hot = sparse.csr_matrix((np.ones(n), (np.arange(n), types)))
    counts = (weights @ one_hot).toarray()
    fractions = counts / counts.sum(axis=1, keepdims=True)
    if np.allclose(fractions, fractions[0], atol=1e-12, rtol=0):
        return fractions, np.zeros(n, dtype=int)
    g, inverse = composition_graph(
        fractions, settings.composition_neighbors, workers, cache_path
    )
    partition = leidenalg.find_partition(
        g,
        leidenalg.RBConfigurationVertexPartition,
        weights=g.es["weight"],
        resolution_parameter=settings.resolution,
        seed=settings.seed,
    )
    return fractions, np.asarray(partition.membership)[inverse]


def composition_graph(fractions, neighbors, workers=1, cache_path=None):
    """Build or reuse the composition similarity graph, verifying exact inputs."""
    import hashlib
    import os
    import tempfile
    from pathlib import Path

    digest = hashlib.sha256(np.ascontiguousarray(fractions).view(np.uint8)).hexdigest()
    path = Path(cache_path) if cache_path is not None else None
    if path is not None and path.exists():
        with np.load(path, allow_pickle=False) as saved:
            if (
                str(saved["digest"]) != digest
                or int(saved["neighbors"]) != neighbors
                or tuple(saved["shape"]) != fractions.shape
            ):
                raise ValueError(
                    "Composition graph cache does not match data or neighbors"
                )
            inverse = saved["inverse"]
            graph = ig.Graph(int(saved["vertices"]), saved["edges"])
            graph.es["weight"] = saved["weights"]
            return graph, inverse

    vectors, inverse, sizes = np.unique(
        np.sqrt(fractions), axis=0, return_inverse=True, return_counts=True
    )
    k = min(neighbors + 1, len(vectors))
    distances, nearby = cKDTree(vectors).query(vectors, k=k, workers=workers)
    scale = max(np.median(distances[:, -1]), 1e-8)
    source = np.repeat(np.arange(len(vectors)), k)
    target = nearby.ravel()
    keep = source != target
    a, b = source[keep], target[keep]
    weights = np.exp(-np.minimum(distances.ravel()[keep] / scale, 60)) * np.sqrt(
        sizes[a] * sizes[b]
    )
    adj = sparse.csr_matrix((weights, (a, b)), shape=(len(vectors),) * 2)
    edges = sparse.triu(adj.maximum(adj.T), k=1).tocoo()
    pairs = np.column_stack([edges.row, edges.col])
    graph = ig.Graph(len(vectors), pairs)
    graph.es["weight"] = edges.data
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(
            prefix=path.name + ".", suffix=".partial", dir=path.parent
        )
        try:
            with os.fdopen(fd, "wb") as handle:
                np.savez(
                    handle,
                    digest=digest,
                    neighbors=neighbors,
                    shape=fractions.shape,
                    inverse=inverse,
                    vertices=len(vectors),
                    edges=pairs,
                    weights=edges.data,
                )
            os.replace(tmp, path)
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)
    return graph, inverse
