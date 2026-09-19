from dataclasses import dataclass
import math
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from phenoniche.evaluation.metrics import concordance_index
from phenoniche.evaluation.patient_protocol import fit_frozen_cox, split_patients
from phenoniche.utils.seed import set_seed


@dataclass
class MultiViewResult:
    W: torch.Tensor
    dictionaries: dict
    history: list


def fit_multiview(blocks, niches=5, iterations=500, seed=31, device="cuda", audit_callback=None):
    names = tuple(blocks)
    if not names or any(np.asarray(blocks[name]).ndim != 2 for name in names):
        raise ValueError("At least one aligned matrix block is required")
    rows = {np.asarray(blocks[name]).shape[0] for name in names}
    if len(rows) != 1:
        raise ValueError("All blocks must share observations")
    set_seed(seed)
    tensors = [torch.as_tensor(blocks[name], dtype=torch.float32, device=device) for name in names]
    scales = [value.shape[1] ** -0.5 for value in tensors]
    x = torch.cat([value * scale for value, scale in zip(tensors, scales)], dim=1)
    generator = torch.Generator(device=device).manual_seed(seed)
    w = torch.rand((x.shape[0], niches), generator=generator, device=device) + 0.2
    h = torch.rand((niches, x.shape[1]), generator=generator, device=device) + 0.2
    mean = x.mean().clamp_min(1e-8).sqrt()
    w = w / w.mean() * mean
    h = h / h.mean() * mean
    history = []
    def emit_audit(epoch):
        if audit_callback is not None:
            pieces = torch.split(h, [value.shape[1] for value in tensors], dim=1)
            audit_callback(epoch, w, {name: piece / scale for name, piece, scale in zip(names, pieces, scales)})
    emit_audit(0)
    for iteration in range(iterations):
        h *= (w.T @ x) / ((w.T @ w) @ h + 1e-8)
        w *= (x @ h.T) / (w @ (h @ h.T) + 1e-8)
        if iteration + 1 in (10, 50, 100, iterations):
            emit_audit(iteration + 1)
        if iteration == 0 or (iteration + 1) % 25 == 0 or iteration + 1 == iterations:
            history.append({"iteration": iteration + 1, "weighted_MSE": float((x - w @ h).square().mean())})
    widths = [value.shape[1] for value in tensors]
    sections = torch.split(h, widths, dim=1)
    dictionaries = {name: section / scale for name, section, scale in zip(names, sections, scales)}
    scale_source = dictionaries.get("HC", next(iter(dictionaries.values())))
    factor_scale = scale_source.sum(1).clamp_min(1e-8)
    w *= factor_scale
    dictionaries = {name: value / factor_scale[:, None] for name, value in dictionaries.items()}
    return MultiViewResult(w.cpu(), {name: value.cpu() for name, value in dictionaries.items()}, history)


def infer_activity(blocks, dictionaries, steps=50):
    names = tuple(blocks)
    gram = sum(torch.as_tensor(dictionaries[name]) @ torch.as_tensor(dictionaries[name]).T / np.asarray(blocks[name]).shape[1] for name in names)
    target = sum(torch.as_tensor(blocks[name]) @ torch.as_tensor(dictionaries[name]).T / np.asarray(blocks[name]).shape[1] for name in names)
    step = 1 / gram.abs().sum(1).max().clamp_min(torch.finfo(gram.dtype).tiny)
    activity = torch.zeros_like(target)
    extrapolated = activity
    momentum = 1.0
    for _ in range(steps):
        updated = torch.relu(extrapolated - step * (extrapolated @ gram - target))
        next_momentum = (1 + math.sqrt(1 + 4 * momentum ** 2)) / 2
        extrapolated = updated + (momentum - 1) / next_momentum * (updated - activity)
        activity, momentum = updated, next_momentum
    return activity


def _cosine_rows(left, right):
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    left = left / np.linalg.norm(left, axis=1, keepdims=True).clip(min=1e-12)
    right = right / np.linalg.norm(right, axis=1, keepdims=True).clip(min=1e-12)
    return left @ right.T


def match_niches(result, truth):
    matrices = []
    for name in result.dictionaries:
        if name in truth:
            matrices.append(_cosine_rows(result.dictionaries[name], truth[name]))
    similarity = sum(matrices) / len(matrices)
    rows, columns = linear_sum_assignment(-similarity)
    order = np.empty(5, dtype=int)
    order[columns] = rows
    return order


def _correlation(left, right):
    if np.std(left) < 1e-10 or np.std(right) < 1e-10:
        return 0.0
    return float(np.corrcoef(left, right)[0, 1])


def _auc(labels, scores):
    labels = np.asarray(labels, dtype=bool)
    positive, negative = labels.sum(), (~labels).sum()
    if positive == 0 or negative == 0:
        return float("nan")
    order = np.argsort(scores)
    ranks = np.empty(len(scores), dtype=float)
    ranks[order] = np.arange(1, len(scores) + 1)
    return float((ranks[labels].sum() - positive * (positive + 1) / 2) / (positive * negative))


def _auprc(labels, scores):
    labels = np.asarray(labels, dtype=np.int8)
    order = np.argsort(-np.asarray(scores))
    cumulative = np.cumsum(labels[order])
    precision = cumulative / np.arange(1, len(labels) + 1)
    return float((precision * labels[order]).sum() / max(labels.sum(), 1))


def evaluate_spatial(result, order, labels, ws_truth, truth, active_by_niche):
    aligned_w = result.W.numpy()[:, order]
    predicted = aligned_w.argmax(1) + 1
    sensitivity = []
    localization = []
    ws_recovery = []
    hc_recovery = []
    hi_recovery = []
    precision = []
    recall = []
    auprc = []
    for niche in range(5):
        target = labels == niche + 1
        sensitivity.append(float((predicted[target] == niche + 1).mean()))
        localization.append(_auc(target, aligned_w[:, niche]))
        ws_recovery.append(_correlation(aligned_w[:, niche], ws_truth[:, niche]))
        if "HC" in result.dictionaries:
            learned = result.dictionaries["HC"][order[niche]].numpy()
            hc_recovery.append(float(_cosine_rows(learned[None], truth["HC"][niche:niche + 1])[0, 0]))
        if "HI" in result.dictionaries:
            learned = result.dictionaries["HI"][order[niche]].numpy()
            hi_recovery.append(float(_cosine_rows(learned[None], truth["HI"][niche:niche + 1])[0, 0]))
            signal = np.asarray(active_by_niche[niche], dtype=bool)
            count = int(signal.sum())
            selected = np.argsort(-learned)[:count]
            hits = int(signal[selected].sum())
            precision.append(hits / max(count, 1))
            recall.append(hits / max(count, 1))
            auprc.append(_auprc(signal, learned))
    collision_block = "HI" if "HI" in result.dictionaries else "HC"
    similarity = _cosine_rows(result.dictionaries[collision_block], truth[collision_block][:2])
    collision = bool(similarity[:, 0].argmax() == similarity[:, 1].argmax())
    return {
        "niche_sensitivity": float(np.mean(sensitivity)), "localization_AUC": float(np.mean(localization)),
        "WS_recovery": float(np.mean(ws_recovery)), "HC_recovery": float(np.mean(hc_recovery)) if hc_recovery else None,
        "HI_recovery": float(np.mean(hi_recovery)) if hi_recovery else None,
        "CCC_precision": float(np.mean(precision)) if precision else None,
        "CCC_recall": float(np.mean(recall)) if recall else None, "CCC_AUPRC": float(np.mean(auprc)) if auprc else None,
        "per_niche": [{"niche": niche + 1, "sensitivity": sensitivity[niche], "localization_AUC": localization[niche],
                       "WS_recovery": ws_recovery[niche], "HI_recovery": hi_recovery[niche] if hi_recovery else None}
                      for niche in range(5)], "Niche1_Niche2_collision": collision,
        "predicted_labels": predicted.tolist(), "aligned_WS": aligned_w.tolist()
    }


def score_bulk(method, activity, order, pi_true, time, event):
    aligned = activity.numpy()[:, order]
    recovery = [_correlation(aligned[:, niche], pi_true[:, niche]) for niche in range(5)]
    split = split_patients(len(pi_true), seed=90210)
    train = torch.tensor(split.train)
    gamma = fit_frozen_cox(activity[train], torch.tensor(time)[train], torch.tensor(event)[train], ridge=0.01)
    cindex = {}
    for partition in ("validation", "test"):
        indices = torch.tensor(getattr(split, partition))
        cindex[partition] = concordance_index(time[indices], event[indices], (activity[indices] @ gamma).numpy())
    gamma_true_order = [float(gamma[order[niche]]) for niche in range(5)]
    return {"Method": method, "Overall_WB": float(np.mean(recovery)), "Risk_WB": recovery[0],
            "Neutral_WB": recovery[1], "Protective_WB": recovery[2], "Validation_C": cindex["validation"],
            "Test_C": cindex["test"], "gamma_1": gamma_true_order[0], "gamma_2": gamma_true_order[1],
            "gamma_3": gamma_true_order[2], "risk_sign_correct": gamma_true_order[0] > 0,
            "protective_sign_correct": gamma_true_order[2] < 0, "neutral_consistent": abs(gamma_true_order[1]) < 0.5}
