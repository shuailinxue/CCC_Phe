import numpy as np


def cosine_similarity(left, right):
    left, right = np.asarray(left, dtype=float), np.asarray(right, dtype=float)
    if left.shape != right.shape or left.size == 0 or not np.isfinite(left).all() or not np.isfinite(right).all():
        raise ValueError("Metric inputs must have matching nonempty finite shapes")
    denominator = np.linalg.norm(left) * np.linalg.norm(right)
    return float(np.sum(left * right) / denominator) if denominator > 0 else float("nan")


def pearson_correlation(left, right):
    left, right = np.asarray(left, dtype=float), np.asarray(right, dtype=float)
    if left.shape != right.shape or left.size == 0:
        raise ValueError("Correlation inputs must have matching nonempty shapes")
    return cosine_similarity(left - left.mean(), right - right.mean())


def concordance_index(time, event, risk):
    time, event, risk = (np.asarray(value) for value in (time, event, risk))
    if time.ndim != 1 or time.size == 0 or event.shape != time.shape or risk.shape != time.shape:
        raise ValueError("Concordance inputs must have matching nonempty [P] shapes")
    if not all(np.isfinite(v).all() for v in (time, event, risk)) or (time <= 0).any() or not np.isin(event, [0, 1]).all():
        raise ValueError("Concordance requires finite risks, positive times and binary events")
    concordant, comparable = 0.0, 0
    for index in np.flatnonzero(event):
        later = (time > time[index]) | ((time == time[index]) & (event == 0))
        comparable += int(later.sum())
        concordant += float((risk[index] > risk[later]).sum()) + 0.5 * float((risk[index] == risk[later]).sum())
    return concordant / comparable if comparable else float("nan")
