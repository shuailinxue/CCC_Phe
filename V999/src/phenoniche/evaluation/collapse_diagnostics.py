import numpy as np
import torch
from phenoniche.losses.collapse import pairwise_view_similarities
from phenoniche.evaluation.metrics import cosine_similarity, pearson_correlation
from phenoniche.evaluation.pair_diagnostics import pair_separation


def redundancy_diagnostics(factors, matched_pair=None):
    with torch.no_grad():
        matrices = {name: value.cpu().numpy() for name, value in pairwise_view_similarities(factors["HC"], factors["HO"], factors["HI"]).items()}
    indices = np.triu_indices(matrices["HC"].shape[0], k=1)
    values = {name: matrix[indices] for name, matrix in matrices.items()}
    product = values["HC"] * values["HO"] * values["HI"]
    pairs = [{"first": int(i), "second": int(j), "sC": float(matrices["HC"][i, j]),
              "sO": float(matrices["HO"][i, j]), "sI": float(matrices["HI"][i, j]),
              "joint_collapse_score": float(matrices["HC"][i, j] * matrices["HO"][i, j] * matrices["HI"][i, j])}
             for i, j in zip(*indices)]
    summary = {"mean_pair_collapse": float(product.mean()) if len(product) else 0.0,
               "max_pair_collapse": float(product.max()) if len(product) else 0.0}
    for name, vector in values.items():
        for statistic, function in (("mean", np.mean), ("median", np.median), ("min", np.min), ("max", np.max)):
            summary[f"{statistic}_{name}_cosine"] = float(function(vector)) if len(vector) else None
    result = {"summary": summary, "pairs": pairs}
    if matched_pair is not None:
        a, b = matched_pair
        if not 0 <= a < matrices["HC"].shape[0] or not 0 <= b < matrices["HC"].shape[0]:
            raise ValueError("Matched factor indices are out of bounds")
        matched = {f"matched_A_B_{name}_cosine": float(matrix[a, b]) for name, matrix in matrices.items()}
        matched["matched_A_B_joint_collapse_score"] = float(np.prod(list(matched.values())))
        result["matched_pair"] = matched
    return result


def detailed_pair_recovery(candidate, truth, train_indices):
    factors = candidate.factors
    report = pair_separation(factors, truth)
    for label, true_index in (("A", 0), ("B", 1)):
        learned_index = report[f"{label}_best_factor"]
        report[f"{label}_HO_cosine"] = cosine_similarity(factors["HO"][learned_index].numpy(), truth["HO"][true_index].numpy())
        report[f"{label}_WB_correlation"] = pearson_correlation(candidate.train_inferred[:, learned_index].numpy(), truth["WB"][train_indices, true_index].numpy())
    return report


def select_collapse_lambda(validation_scores, mean_collapse, tolerance=0.02):
    if not validation_scores or set(validation_scores) != set(mean_collapse):
        raise ValueError("Selection requires matching nonempty validation and collapse grids")
    if not np.isfinite(tolerance) or tolerance < 0:
        raise ValueError("Selection tolerance must be finite and nonnegative")
    if not all(np.isfinite(value) for value in (*validation_scores.values(), *mean_collapse.values())):
        raise ValueError("Selection scores must be finite")
    best = max(validation_scores.values())
    eligible = [weight for weight, score in validation_scores.items() if score >= best - tolerance]
    selected = min(eligible, key=lambda weight: (mean_collapse[weight], weight))
    return {"lambda_collapse": float(selected), "best_validation": float(best),
            "minimum_eligible_validation": float(best - tolerance), "eligible_lambdas": sorted(eligible),
            "validation_scores": validation_scores, "mean_pair_collapse": mean_collapse,
            "test_or_truth_used": False, "tie_break": "smaller lambda"}
