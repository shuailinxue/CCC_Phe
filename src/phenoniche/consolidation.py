"""Numeric distance and membership helpers shared by retained GNN controls."""
import numpy as np

def cosine_matrix(matrix, epsilon=1e-8):
    matrix = np.asarray(matrix, dtype=np.float64)
    normalized = matrix / np.maximum(np.linalg.norm(matrix, axis=1, keepdims=True), epsilon)
    return np.clip(normalized @ normalized.T, -1, 1)


def _scale01(values):
    values = np.asarray(values, dtype=float)
    finite = np.isfinite(values)
    out = np.zeros_like(values)
    if finite.any():
        low, high = values[finite].min(), values[finite].max()
        out[finite] = 1.0 if high <= low else (values[finite] - low) / (high - low)
    return out


def _latent_block(p32, activity, rows, activity_center, activity_scale, activity_weight):
    p = np.asarray(p32[rows], dtype=np.float32)
    a = (np.log1p(np.asarray(activity[rows], dtype=np.float32)) - activity_center) / activity_scale
    return np.column_stack((np.sqrt(np.maximum(p, 0)), activity_weight * a)).astype(np.float32)


def _soft_membership(distances, scale, epsilon=1e-8):
    logits = -np.square(distances) / max(float(scale), epsilon)
    logits -= logits.max(1, keepdims=True)
    weights = np.exp(logits)
    return weights / np.maximum(weights.sum(1, keepdims=True), epsilon)
