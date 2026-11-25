"""Shared device resolution and training-row feature statistics."""
import numpy as np
import torch

def resolve_device(requested="auto"):
    if requested == "auto":
        return "cuda:0" if torch.cuda.is_available() else "cpu"
    if str(requested).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA requested but unavailable: {requested}")
    return str(requested)


def feature_statistics(matrix, rows, chunk_size=8192):
    count = len(rows)
    sums = np.zeros(matrix.shape[1], dtype=np.float64)
    sums2 = np.zeros_like(sums)
    nonzero = np.zeros(matrix.shape[1], dtype=np.int64)
    for start in range(0, count, chunk_size):
        block = np.asarray(matrix[rows[start:start + chunk_size]], dtype=np.float64)
        sums += block.sum(0)
        sums2 += np.square(block).sum(0)
        nonzero += np.count_nonzero(block, axis=0)
    mean = sums / count
    mean_square = sums2 / count
    variance = np.maximum(mean_square - np.square(mean), 0.)
    return mean, mean_square, variance, nonzero / count
