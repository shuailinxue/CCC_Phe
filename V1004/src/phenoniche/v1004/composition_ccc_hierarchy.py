"""Observed composition contexts and their independently learned CCC substates."""

import numpy as np


def build_hierarchical_states(c_state, i_state, k):
    """Number observed (C, I) pairs in C-then-I order; never align the views."""
    c = np.asarray(c_state)
    i = np.asarray(i_state)
    if not isinstance(k, int) or k < 1:
        raise ValueError("k must be a positive integer")
    if c.ndim != 1 or i.ndim != 1 or len(c) != len(i) or not len(c):
        raise ValueError("C and I must label the same nonempty set of cells")
    if not np.issubdtype(c.dtype, np.integer) or not np.issubdtype(i.dtype, np.integer):
        raise ValueError("C and I states must be integers")
    if np.any(c < 0) or np.any(c >= k) or np.any(i < 0) or np.any(i >= k):
        raise ValueError("A state lies outside 0..K-1")

    packed = c.astype(np.int64) * k + i
    observed, inverse, counts = np.unique(packed, return_inverse=True, return_counts=True)
    mapping = tuple(
        {"joint_id": int(j + 1), "C_state": int(key // k),
         "I_state": int(key % k), "n_cells": int(count)}
        for j, (key, count) in enumerate(zip(observed, counts))
    )
    return inverse.astype(np.int32) + 1, mapping
