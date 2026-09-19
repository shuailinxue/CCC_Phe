"""Fixed, outcome-blind balancing of the existing nonnegative matrix model."""
from dataclasses import dataclass
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from phenoniche.utils.seed import set_seed
from phenoniche.v1002.simulation_model import _cosine_rows, _correlation, _auc


@dataclass
class BalancedFit:
    W: torch.Tensor
    dictionaries: dict
    audit: dict


def balance_blocks(blocks, eps=1e-12):
    rms = {name: float(np.sqrt(np.mean(np.asarray(value, dtype=np.float64) ** 2) + eps)) for name, value in blocks.items()}
    return {name: np.asarray(value, dtype=np.float32) / rms[name] for name, value in blocks.items()}, rms


def relative_gradient_norm(w, h, x, eps=1e-12):
    return float((2 * ((w @ h - x) @ h.T) / (x.square().sum() + eps)).norm())


def canonicalize_factors(w, dictionaries):
    """Set each factor's combined per-view dictionary RMS to one.

    The reciprocal rescaling of W and every dictionary preserves each W @ H
    exactly up to floating-point roundoff. Views are averaged over their own
    features, so a wide CCC block does not define the factor scale by width.
    """
    scale = torch.sqrt(sum(h.square().mean(1) for h in dictionaries.values()))
    scale = torch.where(scale > 0, scale, torch.ones_like(scale))
    return w * scale[None, :], {name: h / scale[:, None] for name, h in dictionaries.items()}


def fit_balanced(blocks, seed, iterations=500, balanced=True, device="cuda", niches=6):
    if not isinstance(niches, int) or niches < 2:
        raise ValueError("niches must be an integer of at least two")
    if not blocks or set(blocks) - {"HC", "HI"}:
        raise ValueError("blocks must contain only HC (composition) and/or HI (directed CCC)")
    scaled, rms = balance_blocks(blocks)
    names = tuple(scaled)
    x = {name: torch.as_tensor(scaled[name], device=device) for name in names}
    set_seed(seed)
    generator = torch.Generator(device=device).manual_seed(seed)
    w = torch.rand((len(next(iter(x.values()))), niches), generator=generator, device=device) + 0.2
    h = {name: torch.rand((niches, value.shape[1]), generator=generator, device=device) + 0.2 for name, value in x.items()}
    for name in names:
        h[name] *= x[name].mean().sqrt() / h[name].mean()
    w *= torch.stack([value.mean().sqrt() for value in x.values()]).mean() / w.mean()
    initial = {name: relative_gradient_norm(w, h[name], x[name]) for name in names}
    if balanced and len(names) > 1:
        geometric = float(np.exp(np.mean(np.log(np.maximum(list(initial.values()), 1e-30)))))
        alpha = {name: geometric / max(initial[name], 1e-30) for name in names}
        mean_alpha = float(np.mean(list(alpha.values())))
        alpha = {name: value / mean_alpha for name, value in alpha.items()}
    else:
        alpha = {name: 1.0 for name in names}
    energy = {name: x[name].square().sum().clamp_min(1e-12) for name in names}
    if not balanced:
        # Historical objective: block-width normalized raw squared reconstruction.
        weight = {name: 1.0 / x[name].shape[1] for name in names}
    else:
        weight = {name: alpha[name] / energy[name] for name in names}
    for _ in range(iterations):
        gram = w.T @ w
        for name in names:
            h[name] *= (w.T @ x[name]) / (gram @ h[name] + 1e-8)
        numerator = sum(weight[name] * (x[name] @ h[name].T) for name in names)
        denominator = sum(weight[name] * (h[name] @ h[name].T) for name in names)
        w *= numerator / (w @ denominator + 1e-8)
    final = {name: relative_gradient_norm(w, h[name], x[name]) for name in names}
    weighted_initial = {name: initial[name] * alpha[name] for name in names}
    audit = {"RMS": rms, "initial_gradient_norm": initial, "alpha": alpha,
             "weighted_initial_gradient_norm": weighted_initial, "final_gradient_norm": final,
             "loss": {name: float((x[name] - w @ h[name]).square().sum() / energy[name]) for name in names}}
    w, h = canonicalize_factors(w, h)
    return BalancedFit(w.cpu(), {name: (value * rms[name]).cpu() for name, value in h.items()}, audit)


def evaluate(fit, simulation, ws_truth, blocks):
    h = {name: value.numpy() for name, value in fit.dictionaries.items()}
    truth = {}
    if "HC" in h:
        truth["HC"] = np.vstack((np.full(8, 0.125), simulation.hc_truth))
    if "HI" in h:
        truth["HI"] = np.stack([blocks["HI"][simulation.labels == k].mean(0) for k in range(6)])
    similarity = sum(_cosine_rows(h[name], truth[name]) for name in truth) / len(truth)
    rows, columns = linear_sum_assignment(-similarity)
    order = np.empty(6, dtype=int)
    order[columns] = rows
    w = fit.W.numpy()[:, order]
    w = w / np.maximum(w.sum(1, keepdims=True), 1e-12)
    predicted = w.argmax(1)
    per = []
    for k in range(6):
        target = simulation.labels == k
        per.append({"sensitivity": float(np.mean(predicted[target] == k)),
                    "specificity": float(np.mean(predicted[~target] != k)),
                    "auc": _auc(target, w[:, k]),
                    "ws": _correlation(w[:, k], ws_truth[:, k])})
    def collision(a, b, view):
        if view not in h:
            return None
        score = _cosine_rows(h[view], truth[view][[a, b]])
        return bool(score[:, 0].argmax() == score[:, 1].argmax())
    return {"per_factor": per, "overall_ws": float(np.mean([row["ws"] for row in per])),
            "overall_auc": float(np.mean([row["auc"] for row in per])),
            "overall_sensitivity": float(np.mean([row["sensitivity"] for row in per])),
            "collision_12": collision(1, 2, "HI" if "HI" in h else "HC"),
            "collision_34": collision(3, 4, "HC" if "HC" in h else "HI"),
            "order": order.tolist(), "predicted": predicted.tolist(), "activity": w.tolist()}


def primary_pass(reports):
    return all(
        report["per_factor"][0]["sensitivity"] >= .8
        and all(report["per_factor"][k]["sensitivity"] >= .8 for k in (1, 2, 3, 4))
        and report["overall_ws"] >= .75
        and report["collision_12"] is False
        and report["collision_34"] is False
        for report in reports
    )
