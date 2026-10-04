import numpy as np
import pandas as pd


def cosine_matrix(matrix, epsilon=1e-8):
    normalized = matrix / np.maximum(np.linalg.norm(matrix, axis=1, keepdims=True), epsilon)
    return np.clip(normalized @ normalized.T, -1, 1)


def consolidate_programs(p32, z32, h32, groups, epsilon=1e-8):
    n, d = p32.shape; k = int(groups.max()) + 1
    q8 = np.zeros((n, k), dtype=np.float32)
    for program in range(d): q8[:, groups[program]] += p32[:, program]
    q8 /= np.maximum(q8.sum(1, keepdims=True), epsilon)
    exposure_usage = np.asarray(z32).mean(0); mixture_usage = np.asarray(p32).mean(0)
    h8 = np.zeros((k, h32.shape[1]), dtype=np.float64)
    for niche in range(k):
        members = np.flatnonzero(groups == niche); weights = exposure_usage[members]
        if weights.sum() <= epsilon: weights = np.ones(len(members))
        h8[niche] = np.average(h32[members], axis=0, weights=weights)
    h8 /= np.maximum(h8.sum(1, keepdims=True), epsilon)
    mapping = pd.DataFrame({"program_id": np.arange(1, d + 1), "final_niche_id": groups + 1,
                            "mean_mixture": mixture_usage, "mean_exposure": exposure_usage,
                            "exposure_variance": np.asarray(z32).var(0),
                            "fraction_active": (np.asarray(z32) > epsilon).mean(0)})
    labels = q8.argmax(1); confidence = q8.max(1)
    return q8, h8.astype(np.float32), labels, confidence.astype(np.float32), mapping
