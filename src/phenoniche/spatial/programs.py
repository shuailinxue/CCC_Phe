"""Nonnegative CP on observed focal-cell, neighbor-type and directed LR scores."""

import numpy as np
from scipy import sparse
from scipy.optimize import nnls


def normalize(factors):
    for matrix in factors[1:]:
        norm = np.maximum(np.linalg.norm(matrix, axis=0), 1e-12)
        matrix /= norm
        factors[0] *= norm


def fit_programs(fibers, n_types, settings, rng):
    w = np.repeat(np.arange(len(fibers)), [len(f[0]) for f in fibers])
    s, r, y, _ = (np.concatenate(items) for items in zip(*fibers))
    if not len(w) or not np.any(y):
        return None
    rank = min(settings.rank, len(np.unique(w)), y.shape[1])
    factors = [
        rng.uniform(0.3, 1, (n, rank))
        for n in (len(fibers), 1, n_types, y.shape[1])
    ]
    ids = [w, s, r]
    y = (
        sparse.csr_matrix(y, dtype=np.float64)
        if np.count_nonzero(y) < 0.2 * y.size
        else y.astype(np.float64)
    )
    h = factors[3]
    for _ in range(settings.iterations):
        projected = y @ h
        gram_lr = h.T @ h
        for mode in range(3):
            others = [factors[m][ids[m]] for m in range(3) if m != mode]
            design = others[0] * others[1]
            current = factors[mode]
            num = np.zeros_like(current)
            den = np.zeros_like(current)
            np.add.at(num, ids[mode], projected * design)
            np.add.at(
                den, ids[mode], ((current[ids[mode]] * design) @ gram_lr) * design
            )
            current *= num / np.maximum(den, 1e-12)
        design = factors[0][w] * factors[1][s] * factors[2][r]
        h *= (y.T @ design) / np.maximum(h @ (design.T @ design), 1e-12)
        normalize(factors)
    return tuple(factors[1:])


def project(fiber, factors):
    sender, receiver, y, _ = fiber
    s, r, h = factors
    if not len(y):
        return np.full(h.shape[1], np.nan)
    design = s[sender] * r[receiver]
    gram = (design.T @ design) * (h.T @ h)
    rhs = ((y @ h) * design).sum(axis=0)
    # Equivalent NNLS using the small rank-by-rank Gram matrix.
    chol = np.linalg.cholesky(gram + np.eye(len(rhs)) * 1e-9)
    scores = nnls(chol.T, np.linalg.solve(chol, rhs))[0]
    scores[np.diag(gram) < 1e-10] = np.nan
    return scores


def sample_blocks(ids, block, settings, rng):
    if len(ids) <= settings.fit_windows:
        return rng.permutation(ids)
    keys = np.unique(block[ids])
    selected = rng.choice(keys, max(1, round(0.8 * len(keys))), replace=False)
    # Shuffle within each block, then draw one per block in successive rounds.
    ordered = ids[np.argsort(block[ids], kind="stable")]
    labels = block[ordered]
    starts = np.searchsorted(labels, selected)
    stops = np.searchsorted(labels, selected, side="right")
    queues = [iter(rng.permutation(ordered[a:b])) for a, b in zip(starts, stops)]
    sampled = []
    while queues and len(sampled) < settings.fit_windows:
        remaining = []
        for queue in queues:
            cell = next(queue, None)
            if cell is not None:
                sampled.append(cell)
                remaining.append(queue)
                if len(sampled) == settings.fit_windows:
                    break
        queues = remaining
    return np.asarray(sampled, dtype=int)
