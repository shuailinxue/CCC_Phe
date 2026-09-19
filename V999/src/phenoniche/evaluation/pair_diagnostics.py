import numpy as np
from phenoniche.evaluation.matching import as_numpy
from phenoniche.evaluation.metrics import cosine_similarity, pearson_correlation


def pair_separation(learned, truth, ccc_only=False, pair=(0, 1)):
    hi = as_numpy(learned["HI"])
    true_hi = as_numpy(truth["HI"])
    if hi.ndim != 2 or true_hi.ndim != 2 or hi.shape[1] != true_hi.shape[1]:
        raise ValueError("Pair evaluation needs aligned HI matrices")
    if len(pair) != 2 or pair[0] == pair[1] or min(pair) < 0 or max(pair) >= true_hi.shape[0]:
        raise ValueError("Pair must contain two distinct valid true niche indices")
    learned_basis = hi if ccc_only else np.concatenate((as_numpy(learned["HC"]), hi), axis=1)
    true_basis = true_hi if ccc_only else np.concatenate((as_numpy(truth["HC"]), true_hi), axis=1)
    scores = np.array([[cosine_similarity(a, b) for b in true_basis[list(pair)]] for a in learned_basis])
    hi_scores = np.array([[cosine_similarity(a, b) for b in true_hi[list(pair)]] for a in hi])
    best = scores.argmax(axis=0)
    result = {"matching_basis": "HI" if ccc_only else "HC+HI", "A_best_factor": int(best[0]), "B_best_factor": int(best[1]),
              "same_factor_boolean": bool(best[0] == best[1]), "collision": bool(best[0] == best[1]),
              "HI_only_collision": bool(hi_scores[:, 0].argmax() == hi_scores[:, 1].argmax()),
              "learned_pair_HI_cosine": cosine_similarity(hi[best[0]], hi[best[1]])}
    for label, index, true_index in zip(("A", "B"), best, pair):
        result[f"{label}_recovery"] = float(scores[index, 0 if label == "A" else 1])
        result[f"{label}_HI_cosine"] = cosine_similarity(hi[index], true_hi[true_index])
        result[f"{label}_HC_cosine"] = None if ccc_only else cosine_similarity(as_numpy(learned["HC"])[index], as_numpy(truth["HC"])[true_index])
        result[f"{label}_WS_correlation"] = pearson_correlation(as_numpy(learned["WS"])[:, index], as_numpy(truth["WS"])[:, true_index])
        result[f"{label}_gamma"] = float(as_numpy(learned["gamma"])[index])
    return result


def truth_activity_correlations(truth, time, event):
    wb, ws, eta = as_numpy(truth["WB"]), as_numpy(truth["WS"]), as_numpy(truth["eta"])
    proxy = -np.log(as_numpy(time))
    observed = as_numpy(event).astype(bool)
    return {"WB_A_B": pearson_correlation(wb[:, 0], wb[:, 1]),
            "WS_A_B": pearson_correlation(ws[:, 0], ws[:, 1]),
            "WB_B_eta": pearson_correlation(wb[:, 1], eta),
            "WB_B_observed_survival_proxy": pearson_correlation(wb[:, 1], proxy),
            "WB_A_eta": pearson_correlation(wb[:, 0], eta),
            "WB_A_observed_survival_proxy": pearson_correlation(wb[:, 0], proxy),
            "WB_B_uncensored_proxy": pearson_correlation(wb[observed, 1], proxy[observed]) if observed.sum() > 1 else None,
            "observed_survival_proxy": "-log(min(failure_time, censoring_time)); descriptive and censoring-confounded, not a survival effect estimate",
            "true_gamma_A": float(as_numpy(truth["gamma"])[0]), "true_gamma_B": float(as_numpy(truth["gamma"])[1])}
