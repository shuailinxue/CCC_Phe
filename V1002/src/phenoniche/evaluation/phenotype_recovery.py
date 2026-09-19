import numpy as np
from scipy.optimize import linear_sum_assignment
from phenoniche.evaluation.matching import as_numpy
from phenoniche.evaluation.metrics import cosine_similarity, pearson_correlation


def _pair_similarity(learned, truth):
    matrices = []
    for factors in (learned, truth):
        hc, hi = as_numpy(factors["HC"]), as_numpy(factors["HI"])
        if hc.ndim != 2 or hi.ndim != 2 or hc.shape[0] != hi.shape[0]:
            raise ValueError("HC and HI must be matrices with matching niche counts")
        joined = np.concatenate((hc, hi), axis=1).astype(float)
        norms = np.linalg.norm(joined, axis=1, keepdims=True)
        if not np.isfinite(joined).all() or (norms <= 0).any():
            raise ValueError("Matching requires finite nonempty dictionaries")
        matrices.append(joined / norms)
    if matrices[0].shape[1] != matrices[1].shape[1]:
        raise ValueError("Learned and true dictionary features must agree")
    return matrices[0] @ matrices[1].T


def phenotype_recovery(learned, truth, roles, inferred_bulk, bulk_truth):
    similarity = _pair_similarity(learned, truth)
    if len(roles) != similarity.shape[1]:
        raise ValueError("Each true niche must have a role")
    wb, target_wb = as_numpy(inferred_bulk), as_numpy(bulk_truth)
    ws, target_ws = as_numpy(learned["WS"]), as_numpy(truth["WS"])
    gamma = as_numpy(learned["gamma"])
    kl, kt = similarity.shape
    if wb.ndim != 2 or target_wb.shape != (wb.shape[0], kt) or wb.shape[1] != kl:
        raise ValueError("Bulk recovery requires aligned patients and correct niche counts")
    if ws.ndim != 2 or target_ws.shape != (ws.shape[0], kt) or ws.shape[1] != kl or gamma.shape != (kl,):
        raise ValueError("Spatial factors and gamma have incompatible niche counts")
    rows, columns = linear_sum_assignment(-similarity)
    assigned = {int(column): int(row) for row, column in zip(rows, columns)}
    best = similarity.argmax(axis=0)
    counts = np.bincount(best, minlength=kl)
    records = []
    for true_index, learned_index in enumerate(best):
        record = {"true_niche": true_index, "role": roles[true_index], "learned_niche": int(learned_index),
                  "best_match_collision": bool(counts[learned_index] > 1),
                  "hungarian_match": assigned.get(true_index),
                  "dictionary_cosine": float(similarity[learned_index, true_index]),
                  "gamma": float(gamma[learned_index]),
                  "WB_correlation": pearson_correlation(wb[:, learned_index], target_wb[:, true_index]),
                  "WS_correlation": pearson_correlation(ws[:, learned_index], target_ws[:, true_index])}
        for name in ("HC", "HI", "HO"):
            prediction, target = as_numpy(learned[name]), as_numpy(truth[name])
            if prediction.shape[0] != kl or target.shape[0] != kt or prediction.shape[1:] != target.shape[1:]:
                raise ValueError(f"Invalid {name} recovery dimensions")
            record[f"{name}_cosine"] = cosine_similarity(prediction[learned_index], target[true_index])
        true_gamma = float(as_numpy(truth["gamma"])[true_index])
        record["gamma_sign_correct"] = bool(record["gamma"] * true_gamma > 0) if true_gamma else None
        records.append(record)
    groups = {}
    for role in ("risk", "protective", "neutral", "nuisance"):
        members = [row for row in records if row["role"] == role]
        metrics = ("dictionary_cosine", "HC_cosine", "HI_cosine", "HO_cosine", "WB_correlation", "WS_correlation", "gamma")
        groups[f"{role}_niche_recovery"] = {key: float(np.mean([row[key] for row in members])) for key in metrics} if members else None
    return {"per_true_niche": records, **groups,
            "overall_dictionary_recovery": float(similarity[rows, columns].sum() / kt),
            "matched_dictionary_cosine": float(similarity[rows, columns].mean()),
            "hungarian_coverage": len(rows) / kt,
            "best_match_collisions": int(sum(counts > 1)),
            "unmatched_true_niches": sorted(set(range(kt)) - set(assigned))}
