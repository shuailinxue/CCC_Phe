import numpy as np
from phenoniche.evaluation.matching import as_numpy
from phenoniche.evaluation.metrics import cosine_similarity, pearson_correlation


def _normalized_rows(hc, hi):
    hc, hi = as_numpy(hc).astype(float), as_numpy(hi).astype(float)
    if hc.ndim != 2 or hi.ndim != 2 or hc.shape[0] != hi.shape[0]:
        raise ValueError("HC and HI must be matrices with equal niche counts")
    joined = np.concatenate((hc, hi), axis=1)
    norms = np.linalg.norm(joined, axis=1, keepdims=True)
    if not np.isfinite(joined).all() or (norms <= 0).any():
        raise ValueError("Matching dictionaries must be finite and nonempty")
    return joined / norms


def _cosine_matrix(learned, truth):
    learned = as_numpy(learned).astype(float)
    truth = as_numpy(truth).astype(float)
    if learned.ndim != 2 or truth.ndim != 2 or learned.shape[1] != truth.shape[1]:
        raise ValueError("Learned and truth matrices require aligned feature columns")
    learned_norm = np.linalg.norm(learned, axis=1, keepdims=True)
    truth_norm = np.linalg.norm(truth, axis=1, keepdims=True)
    if not np.isfinite(learned).all() or not np.isfinite(truth).all() or (learned_norm <= 0).any() or (truth_norm <= 0).any():
        raise ValueError("Cosine matching requires finite nonempty rows")
    return learned / learned_norm @ (truth / truth_norm).T


def bank_similarity_matrices(factors, truth):
    true_joint = _normalized_rows(truth["HC"], truth["HI"])
    phenotype_joint = _normalized_rows(factors["HCp"], factors["HIp"])
    background_joint = _normalized_rows(factors["HC0"], factors["HI0"])
    if phenotype_joint.shape[1] != true_joint.shape[1] or background_joint.shape[1] != true_joint.shape[1]:
        raise ValueError("Bank and truth dictionary features must agree")
    return {
        "phenotype_joint": phenotype_joint @ true_joint.T,
        "background_joint": background_joint @ true_joint.T,
        "phenotype_HI": _cosine_matrix(factors["HIp"], truth["HI"]),
        "background_HI": _cosine_matrix(factors["HI0"], truth["HI"]),
    }


def bank_matching_report(factors, truth, roles, inferred_background, inferred_phenotype,
                         true_bulk, label_names=None):
    similarities = bank_similarity_matrices(factors, truth)
    true_count = similarities["phenotype_joint"].shape[1]
    if len(roles) != true_count:
        raise ValueError("Each true niche must have a role")
    labels = label_names or [str(index) for index in range(true_count)]
    if len(labels) != true_count:
        raise ValueError("Each true niche must have a label")
    wb0, wbp, target_wb = map(as_numpy, (inferred_background, inferred_phenotype, true_bulk))
    ws0, wsp, target_ws = map(as_numpy, (factors["WS0"], factors["WSp"], truth["WS"]))
    if target_wb.shape[1] != true_count or target_ws.shape[1] != true_count:
        raise ValueError("Truth activities must align with true niche count")
    if wb0.shape[0] != target_wb.shape[0] or wbp.shape[0] != target_wb.shape[0]:
        raise ValueError("Learned and true bulk activities must align by patient")
    if ws0.shape[0] != target_ws.shape[0] or wsp.shape[0] != target_ws.shape[0]:
        raise ValueError("Learned and true spatial activities must align by anchor")
    records = []
    for true_index, (role, label) in enumerate(zip(roles, labels)):
        p_factor = int(np.argmax(similarities["phenotype_joint"][:, true_index]))
        b_factor = int(np.argmax(similarities["background_joint"][:, true_index]))
        p_score = float(similarities["phenotype_joint"][p_factor, true_index])
        b_score = float(similarities["background_joint"][b_factor, true_index])
        best_bank = "phenotype" if p_score >= b_score else "background"
        best_factor = p_factor if best_bank == "phenotype" else b_factor
        hi_p_factor = int(np.argmax(similarities["phenotype_HI"][:, true_index]))
        hi_b_factor = int(np.argmax(similarities["background_HI"][:, true_index]))
        learned = {"HC": factors["HCp"], "HO": factors["HOp"], "HI": factors["HIp"],
                   "WB": wbp, "WS": wsp} if best_bank == "phenotype" else {
                   "HC": factors["HC0"], "HO": factors["HO0"], "HI": factors["HI0"],
                   "WB": wb0, "WS": ws0}
        record = {
            "true_niche": true_index, "label": label, "role": role,
            "best_bank": best_bank, "best_factor": best_factor,
            "bank_margin": p_score - b_score,
            "best_phenotype_factor": p_factor, "best_phenotype_similarity": p_score,
            "best_background_factor": b_factor, "best_background_similarity": b_score,
            "phenotype_leakage_to_background": b_score,
            "background_leakage_to_phenotype": p_score,
            "HI_only_best_phenotype_factor": hi_p_factor,
            "HI_only_best_phenotype_similarity": float(similarities["phenotype_HI"][hi_p_factor, true_index]),
            "HI_only_best_background_factor": hi_b_factor,
            "HI_only_best_background_similarity": float(similarities["background_HI"][hi_b_factor, true_index]),
            "WB_correlation": pearson_correlation(learned["WB"][:, best_factor], target_wb[:, true_index]),
            "WS_correlation": pearson_correlation(learned["WS"][:, best_factor], target_ws[:, true_index]),
            "phenotype_WB_correlation": pearson_correlation(wbp[:, p_factor], target_wb[:, true_index]),
            "phenotype_WS_correlation": pearson_correlation(wsp[:, p_factor], target_ws[:, true_index]),
            "background_WB_correlation": pearson_correlation(wb0[:, b_factor], target_wb[:, true_index]),
            "background_WS_correlation": pearson_correlation(ws0[:, b_factor], target_ws[:, true_index]),
            "phenotype_gamma": float(as_numpy(factors["gamma"])[p_factor]),
        }
        true_gamma = float(as_numpy(truth["gamma"])[true_index])
        record["phenotype_gamma_sign_correct"] = bool(record["phenotype_gamma"] * true_gamma > 0) if true_gamma else None
        for name in ("HC", "HI", "HO"):
            record[f"{name}_cosine"] = cosine_similarity(as_numpy(learned[name])[best_factor], as_numpy(truth[name])[true_index])
            record[f"phenotype_{name}_cosine"] = cosine_similarity(as_numpy(factors[f"{name}p"])[p_factor], as_numpy(truth[name])[true_index])
            record[f"background_{name}_cosine"] = cosine_similarity(as_numpy(factors[f"{name}0"])[b_factor], as_numpy(truth[name])[true_index])
        if best_bank == "phenotype":
            record["gamma"] = float(as_numpy(factors["gamma"])[best_factor])
            record["gamma_sign_correct"] = bool(record["gamma"] * true_gamma > 0) if true_gamma else None
        else:
            record["gamma"] = None
            record["gamma_sign_correct"] = None
        records.append(record)
    by_label = {record["label"]: record for record in records}
    result = {"per_true_niche": records, "by_label": by_label}
    if "A" in by_label and "B" in by_label:
        a, b = by_label["A"], by_label["B"]
        result["A_B_different_bank"] = a["best_bank"] != b["best_bank"]
        result["A_B_different_factor"] = (a["best_bank"], a["best_factor"]) != (b["best_bank"], b["best_factor"])
        result["phenotype_leakage_A_to_background"] = a["best_background_similarity"]
        result["background_leakage_B_to_phenotype"] = b["best_phenotype_similarity"]
    return result
