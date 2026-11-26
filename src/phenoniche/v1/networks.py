"""Integer node masses and sparse sender-level hurdle sufficient statistics."""
import numpy as np
import pandas as pd
from numba import njit
from scipy import sparse
from scipy.spatial import cKDTree

from phenoniche.spatial.channels import Channels


def expression(counts, genes, atlas, reference, scales=None):
    """Use V0's panel/complex definitions and freeze log-CP10K quantiles for transfer."""
    genes = pd.Index(genes)
    channels = Channels(counts[:1], genes, atlas)
    counts = sparse.csr_matrix(counts, dtype=np.float32, copy=True)
    if np.any(counts.data < 0) or not np.isfinite(counts.data).all():
        raise ValueError("Expected finite nonnegative counts")
    scale = 10000 / np.maximum(np.asarray(counts.sum(1)), 1)
    values = counts.multiply(scale).tocsc()
    values.eliminate_zeros()
    values.data = np.log1p(values.data)
    if scales is None:
        train = values[reference].tocsc()
        scales = np.ones(len(genes), np.float32)
        for j in range(len(genes)):
            col = train.data[train.indptr[j]:train.indptr[j + 1]]
            if len(col):
                scales[j] = np.quantile(col, .95)
    if len(scales) != len(genes) or np.any(scales <= 0):
        raise ValueError("Incompatible expression scales")
    values.data = np.minimum(values.data / np.repeat(scales, np.diff(values.indptr)), 1)
    arms = sorted({x for pair in channels.names for x in pair})
    index = {name: i for i, name in enumerate(arms)}
    columns = []
    for name in arms:
        parts = [name] if name in genes else name.split("+")
        col = values[:, genes.get_loc(parts[0])].power(1 / len(parts))
        for part in parts[1:]:
            col = col.multiply(values[:, genes.get_loc(part)].power(1 / len(parts)))
        columns.append(col.sqrt())
    left, right = np.array([(index[l], index[r]) for l, r in channels.names]).T
    return (sparse.hstack(columns, format="csr").toarray(), left, right,
            channels.names, channels.signaling_types, scales)


def split_region(xyz, specimens, radius, fraction):
    """Right-hand slab per specimen with a strictly disjoint training buffer."""
    train = np.ones(len(xyz), bool)
    test = np.zeros(len(xyz), bool)
    if fraction:
        for sample in np.unique(specimens):
            ids = np.flatnonzero(specimens == sample)
            cut = np.quantile(xyz[ids, 0], 1 - fraction)
            test[ids] = xyz[ids, 0] >= cut
            train[ids] = xyz[ids, 0] < cut - 2 * radius
    if not train.any():
        raise ValueError("Holdout and buffer leave no training centers")
    return train, test


def window_index(xyz, specimens, radius):
    """Variable-length windows include the center and coincident distinct cells."""
    rows = [None] * len(xyz)
    for sample in np.unique(specimens):
        ids = np.flatnonzero(specimens == sample)
        nearby = cKDTree(xyz[ids]).query_ball_point(xyz[ids], radius)
        for cell, local in zip(ids, nearby, strict=True):
            rows[cell] = ids[local].astype(np.int32)
    return rows


@njit(nogil=True)
def observe(ids, xyz, types, nt, expr, left, right, radii2, group, expr_ptr, expr_col):
    """Each ordered sender/receiver cell pair occurs once, excluding self."""
    nl, ng = len(left), len(radii2)
    mass = np.zeros(nt, np.int32)
    support = np.zeros((nt, nt, ng), np.int32)
    stats = np.zeros((nt * nt * nl, 4), np.float32)
    for src in ids:
        s = types[src]
        mass[s] += 1
        neighbors = np.zeros((nt, ng), np.int32)
        receptor = np.zeros((nt, ng, expr.shape[1]), np.float32)
        for dst in ids:
            if src == dst:
                continue
            d2 = 0.
            for axis in range(xyz.shape[1]):
                d2 += (xyz[src, axis] - xyz[dst, axis]) ** 2
            t = types[dst]
            for g in range(ng):
                if d2 <= radii2[g]:
                    neighbors[t, g] += 1
                    for z in range(expr_ptr[dst], expr_ptr[dst + 1]):
                        arm = expr_col[z]
                        receptor[t, g, arm] += expr[dst, arm]
        support[s] += neighbors > 0
        for l in range(nl):
            ligand = expr[src, left[l]]
            if not ligand:
                continue
            g = group[l]
            for t in range(nt):
                total = receptor[t, g, right[l]]
                if total > 0:
                    x = ligand * total / neighbors[t, g]
                    logx = np.log(x)
                    row = (s * nt + t) * nl + l
                    stats[row, 0] += 1
                    stats[row, 1] += logx
                    stats[row, 2] += logx * logx
                    stats[row, 3] += x
    edges = np.nonzero(stats[:, 0])[0].astype(np.int32)
    return mass, support, edges, stats[edges]


def joint(mass, activity, support, tau):
    """Return masked edge mass and reference mass, one local network at a time."""
    if not np.isfinite(tau) or tau <= 0:
        raise ValueError("Positive finite reference scale required")
    p = mass / mass.sum()
    active = np.where(support > 0, activity, 0.)
    denom = tau + active.sum(axis=(1, 2))
    edge = p[:, None, None] * active / denom[:, None, None]
    return np.where(support > 0, edge, np.nan), p * tau / denom
