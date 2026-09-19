import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from phenoniche.evaluation.metrics import cosine_similarity, pearson_correlation
from phenoniche.training.st_first import infer_spatial_activities, spatial_reconstruction


def _cosine_matrix(left, right):
    left = np.array(left, dtype=float, copy=True)
    right = np.array(right, dtype=float, copy=True)
    left /= np.linalg.norm(left, axis=1, keepdims=True)
    right /= np.linalg.norm(right, axis=1, keepdims=True)
    return left @ right.T


def matching_similarity(learned, target, blocks=("HC", "HI", "HO")):
    matrices = [_cosine_matrix(learned[name], target[name]) for name in blocks]
    return sum(matrices) / len(matrices)


def match_factors(learned, target, blocks=("HC", "HI", "HO")):
    similarity = matching_similarity(learned, target, blocks)
    rows, columns = linear_sum_assignment(-similarity)
    order = np.empty(similarity.shape[1], dtype=int)
    order[columns] = rows
    return order, similarity


def result_factors(result):
    return {name: getattr(result, name).detach().cpu().numpy() for name in ("WS", "HC", "HI", "HO")}


def build_consensus(results, cs, is_, os, spatial_steps=200):
    if len(results) < 2:
        raise ValueError("Consensus requires at least two ST-only solutions")
    factors = [result_factors(result) for result in results]
    reference_index = int(np.argmin([result.history[-1]["weighted_mean_squared_error"] for result in results]))
    reference = factors[reference_index]
    aligned = []
    alignments = []
    for current in factors:
        order, _ = match_factors(current, reference)
        alignments.append(order.tolist())
        aligned.append({name: current[name][order] if name.startswith("H") else current[name][:, order]
                        for name in ("WS", "HC", "HI", "HO")})
    consensus = {name: np.median(np.stack([current[name] for current in aligned]), axis=0)
                 for name in ("HC", "HI", "HO")}
    scale = consensus["HC"].sum(1)
    for name in consensus:
        consensus[name] = consensus[name] / scale[:, None]
    tensors = {name: torch.tensor(value, dtype=cs.dtype, device=cs.device) for name, value in consensus.items()}
    ws = infer_spatial_activities(cs, is_, os, tensors["HC"], tensors["HI"], tensors["HO"], steps=spatial_steps)
    tensors["WS"] = ws
    return tensors, {"reference_seed": results[reference_index].seed, "alignments": alignments,
                     "spatial_reconstruction": spatial_reconstruction(cs, is_, os, ws, tensors["HC"], tensors["HI"], tensors["HO"])}


def _collision(similarity, first=0, second=1):
    best = similarity.argmax(axis=0)
    return bool(best[first] == best[second]), best.tolist()


def edge_recovery(learned_hi, truth_hi, mapping, learned_index, true_index):
    learned = np.asarray(learned_hi[learned_index], dtype=float)
    truth = np.asarray(truth_hi[true_index], dtype=float)
    learned_order = np.argsort(-learned)
    active_count = int(np.sum(truth > 1e-7))
    active = set(np.argsort(-truth)[:active_count].tolist())
    metrics = {}
    for count in (20, 50):
        predicted = set(learned_order[:count].tolist())
        intersection = len(predicted & active)
        metrics[f"top_{count}_precision"] = intersection / count
        metrics[f"top_{count}_recall"] = intersection / active_count
    metrics["learned_top_edges"] = [{**mapping[int(index)], "weight": float(learned[index])} for index in learned_order[:50]]
    truth_order = np.argsort(-truth)[:50]
    metrics["truth_top_edges"] = [{**mapping[int(index)], "weight": float(truth[index])} for index in truth_order]
    return metrics


def recovery_report(factors, truth, mapping):
    learned = {name: value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else np.asarray(value)
               for name, value in factors.items()}
    target = {name: value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else np.asarray(value)
              for name, value in truth.items() if name in ("HC", "HI", "HO", "WS")}
    order, similarity = match_factors(learned, target)
    records = []
    labels = ("A", "B", "C", "D", "E", "F")
    for true_index, learned_index in enumerate(order):
        record = {"niche": labels[true_index], "true_index": true_index, "learned_index": int(learned_index)}
        for name in ("HC", "HI", "HO"):
            record[f"{name}_recovery"] = cosine_similarity(learned[name][learned_index], target[name][true_index])
        record["WS_recovery"] = pearson_correlation(learned["WS"][:, learned_index], target["WS"][:, true_index])
        record["edges"] = edge_recovery(learned["HI"], target["HI"], mapping, learned_index, true_index)
        records.append(record)
    hc_collision, hc_best = _collision(_cosine_matrix(learned["HC"], target["HC"]))
    hi_collision, hi_best = _collision(_cosine_matrix(learned["HI"], target["HI"]))
    joint_collision, joint_best = _collision(matching_similarity(learned, target, ("HC", "HI")))
    return {"per_niche": records, "matching": order.tolist(),
            "mean_HC_recovery": float(np.mean([row["HC_recovery"] for row in records])),
            "mean_HI_recovery": float(np.mean([row["HI_recovery"] for row in records])),
            "mean_HO_recovery": float(np.mean([row["HO_recovery"] for row in records])),
            "mean_WS_recovery": float(np.mean([row["WS_recovery"] for row in records])),
            "HC_only_A_B_collision": hc_collision, "HI_only_A_B_collision": hi_collision,
            "HC_HI_A_B_collision": joint_collision,
            "HC_only_best": hc_best, "HI_only_best": hi_best, "HC_HI_best": joint_best}


def stability_report(results):
    factors = [result_factors(result) for result in results]
    reference_index = int(np.argmin([result.history[-1]["weighted_mean_squared_error"] for result in results]))
    reference = factors[reference_index]
    aligned = []
    for current in factors:
        order, _ = match_factors(current, reference)
        aligned.append({name: current[name][order] if name.startswith("H") else current[name][:, order]
                        for name in ("WS", "HC", "HI", "HO")})
    pairs = []
    for left in range(len(aligned)):
        for right in range(left + 1, len(aligned)):
            pairs.append({name: float(np.mean([cosine_similarity(aligned[left][name][index], aligned[right][name][index])
                                                       for index in range(aligned[left]["HC"].shape[0])]))
                          for name in ("HC", "HI", "HO")})
    return {"reference_seed": results[reference_index].seed,
            "pairwise_mean": {name: float(np.mean([pair[name] for pair in pairs])) for name in ("HC", "HI", "HO")},
            "pairwise_min": {name: float(np.min([pair[name] for pair in pairs])) for name in ("HC", "HI", "HO")}}
