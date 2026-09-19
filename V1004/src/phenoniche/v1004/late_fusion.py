"""Permutation alignment and observed joint states for independent C/I fits."""
from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment


@dataclass(frozen=True)
class Alignment:
    before: np.ndarray
    after: np.ndarray
    i_to_c: np.ndarray
    i_column_for_c: np.ndarray
    mapping: tuple[dict, ...]
    overall_agreement: float
    per_c_agreement: np.ndarray


def _labels(value, name, k):
    labels = np.asarray(value)
    if labels.ndim != 1 or not np.issubdtype(labels.dtype, np.integer):
        raise ValueError(f"{name} must be a one-dimensional integer label array")
    if np.any(labels < 0) or np.any(labels >= k):
        raise ValueError(f"{name} contains a label outside 0..K-1")
    return labels.astype(np.int64, copy=False)


def align_partitions(c_labels, i_labels, k):
    """Find the I-to-C label permutation maximizing hard-assignment overlap."""
    if not isinstance(k, int) or k < 1:
        raise ValueError("k must be a positive integer")
    c = _labels(c_labels, "c_labels", k)
    i = _labels(i_labels, "i_labels", k)
    if len(c) != len(i) or not len(c):
        raise ValueError("Both views must label the same nonempty set of cells")
    before = np.bincount(c * k + i, minlength=k * k).reshape(k, k)
    c_factors, i_factors = linear_sum_assignment(-before)
    i_to_c = np.empty(k, dtype=np.int64)
    i_to_c[i_factors] = c_factors
    i_column_for_c = np.argsort(i_to_c)
    after = before[:, i_column_for_c]
    c_counts = before.sum(axis=1)
    matched = np.diag(after)
    per_c = np.divide(matched, c_counts, out=np.full(k, np.nan), where=c_counts > 0)
    mapping = tuple({"C_factor": int(c_factor),
                     "original_I_factor": int(i_column_for_c[c_factor]),
                     "matched_cell_count": int(matched[c_factor]),
                     "overlap_fraction": float(per_c[c_factor])}
                    for c_factor in range(k))
    return Alignment(before, after, i_to_c, i_column_for_c, mapping,
                     float(matched.sum() / len(c)), per_c)


def apply_alignment(i_labels, w_i, alignment):
    """Rename I labels and columns only; keep every cell's I assignment."""
    k = len(alignment.i_to_c)
    i = _labels(i_labels, "i_labels", k)
    w = np.asarray(w_i)
    if w.ndim != 2 or w.shape != (len(i), k):
        raise ValueError("w_i must have one row per cell and K columns")
    return alignment.i_to_c[i], w[:, alignment.i_column_for_c]


def build_joint_niches(c_labels, i_labels_aligned, k):
    """Number only observed (C, aligned-I) states in deterministic order."""
    c = _labels(c_labels, "c_labels", k)
    i = _labels(i_labels_aligned, "i_labels_aligned", k)
    if len(c) != len(i) or not len(c):
        raise ValueError("Both views must label the same nonempty set of cells")
    observed = {(int(a), int(b)) for a, b in zip(c, i)}
    keys = sorted((a, b) for a, b in observed if a == b)
    keys += sorted((a, b) for a, b in observed if a != b)
    key_to_id = {key: index for index, key in enumerate(keys)}
    final_labels = np.fromiter((key_to_id[(int(a), int(b))] for a, b in zip(c, i)),
                               dtype=np.int64, count=len(c))
    counts = np.bincount(final_labels, minlength=len(keys))
    mapping = tuple({"final_niche_id": f"Final Niche {index + 1}",
                     "C_state": a, "I_state_aligned": b,
                     "n_cells": int(counts[index]),
                     "fraction": float(counts[index] / len(c)),
                     "concordant": bool(a == b)}
                    for index, (a, b) in enumerate(keys))
    return final_labels, mapping
