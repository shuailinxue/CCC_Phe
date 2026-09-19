import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from phenoniche.evaluation.metrics import cosine_similarity, pearson_correlation


def as_numpy(value):
    return value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else np.asarray(value)


def match_niches(learned, truth):
    matrices = []
    for factors in (learned, truth):
        hc, hi = as_numpy(factors["HC"]), as_numpy(factors["HI"])
        if hc.ndim != 2 or hi.ndim != 2 or hc.shape[0] != hi.shape[0] or min(hc.shape + hi.shape) < 1:
            raise ValueError("HC and HI must be nonempty matrices with matching niche counts")
        if not np.isfinite(hc).all() or not np.isfinite(hi).all():
            raise ValueError("Dictionaries must be finite")
        joined = np.concatenate((hc, hi), axis=1)
        norms = np.linalg.norm(joined, axis=1, keepdims=True)
        if (norms == 0).any():
            raise ValueError("Cannot match empty niche dictionaries")
        matrices.append(joined / norms)
    if matrices[0].shape != matrices[1].shape:
        raise ValueError("Recovery matching requires equal K and aligned feature dimensions")
    learned_rows, true_rows = linear_sum_assignment(-(matrices[0] @ matrices[1].T))
    permutation = np.empty(len(true_rows), dtype=int)
    permutation[true_rows] = learned_rows
    return permutation


def recovery_report(learned, truth):
    permutation = match_niches(learned, truth)
    report = {"matching": permutation.tolist()}
    k = len(permutation)
    for name in ("HC", "HI", "HO", "WB", "WS"):
        predicted, target = as_numpy(learned[name]), as_numpy(truth[name])
        if predicted.shape != target.shape or predicted.ndim != 2:
            raise ValueError(f"Recovery shapes must agree for {name}")
        predicted = predicted[permutation] if name.startswith("H") else predicted[:, permutation].T
        target = target if name.startswith("H") else target.T
        if predicted.shape[0] != k:
            raise ValueError(f"Recovery niche count mismatch for {name}")
        cosines = [cosine_similarity(a, b) for a, b in zip(predicted, target)]
        correlations = [pearson_correlation(a, b) for a, b in zip(predicted, target)]
        report[name] = {"cosine": float(np.mean(cosines)), "pearson": float(np.mean(correlations)),
                        "per_niche_cosine": cosines, "per_niche_pearson": correlations}
    gamma = as_numpy(learned["gamma"])
    if gamma.shape != (k,) or not np.isfinite(gamma).all():
        raise ValueError("gamma must be finite with shape [K]")
    report["matched_gamma"] = gamma[permutation].tolist()
    return report
