from copy import deepcopy
from dataclasses import dataclass
import math
import numpy as np
import torch
from phenoniche.evaluation.metrics import concordance_index, cosine_similarity, pearson_correlation
from phenoniche.inference.bulk import infer_bulk_activities
from phenoniche.losses.survival import cox_breslow_loss
from phenoniche.model.adapters import BoundedDictionaryAdapter
from phenoniche.training.st_first import infer_spatial_activities, spatial_reconstruction


@dataclass(frozen=True)
class IterativeRefinementConfig:
    outer_iterations: int = 5
    epochs_per_iteration: int = 60
    learning_rate: float = 0.02
    delta_max: float = math.log(1.25)
    lambda_st: float = 1.0
    lambda_bulk: float = 1.0
    lambda_ph: float = 0.05
    lambda_adapter: float = 1.0
    hc_gate: float = 0.95
    hi_gate: float = 0.90
    st_degradation_gate: float = 0.05
    gradient_clip: float = 10.0

    def __post_init__(self):
        if self.outer_iterations != 5 or self.epochs_per_iteration < 1:
            raise ValueError("Primary iterative calibration requires five positive outer iterations")
        values = (self.learning_rate, self.delta_max, self.lambda_st, self.lambda_bulk,
                  self.lambda_ph, self.lambda_adapter, self.hc_gate, self.hi_gate,
                  self.st_degradation_gate, self.gradient_clip)
        if not all(math.isfinite(value) and value > 0 for value in values):
            raise ValueError("Iterative refinement constants must be positive and finite")


def _c_index(activity, gamma, time, event):
    return concordance_index(time.detach().cpu().numpy(), event.detach().cpu().numpy(),
                             (activity @ gamma).detach().cpu().numpy())


def _mean_cosine(current, anchor):
    return float(np.mean([cosine_similarity(current[index].detach().cpu().numpy(), anchor[index].detach().cpu().numpy())
                          for index in range(current.shape[0])]))


def _per_niche_cosine(current, anchor):
    return [cosine_similarity(current[index].detach().cpu().numpy(), anchor[index].detach().cpu().numpy())
            for index in range(current.shape[0])]


def refinement_loss(adapter, train_cb, train_ib, train_time, train_event, cs, is_, os, inner_steps):
    hc, hi, ho = adapter.dictionaries()
    wb = infer_bulk_activities(train_cb, train_ib, hc, hi, lambda_bc=1 / train_cb.shape[1],
                               lambda_bi=1 / train_ib.shape[1], steps=inner_steps)
    st_c = (cs - adapter.WS0 @ hc).square().mean()
    st_i = (is_ - adapter.WS0 @ hi).square().mean()
    st_o = (os - adapter.WS0 @ ho).square().mean()
    bulk_c = (train_cb - wb @ hc).square().mean()
    bulk_i = (train_ib - wb @ hi).square().mean()
    cox = cox_breslow_loss(wb @ adapter.gamma, train_time, train_event)
    delta_c, delta_i = adapter.deltas()
    penalty = delta_c.square().mean() + delta_i.square().mean()
    total = adapter.config.lambda_st * (st_c + st_i + st_o) + adapter.config.lambda_bulk * (bulk_c + bulk_i) + adapter.config.lambda_ph * cox + adapter.config.lambda_adapter * penalty
    return {"total_loss": total, "st_anchor": st_c + st_i + st_o,
            "bulk_reconstruction": bulk_c + bulk_i, "cox_loss": cox,
            "adapter_penalty": penalty, "WB": wb}


def rollback_if_rejected(module, previous_state, accepted):
    if not accepted:
        module.load_state_dict(previous_state)
    return accepted


def run_iterative_refinement(anchor, train, validation, spatial, initial_gamma, inner_steps, config=None):
    config = config or IterativeRefinementConfig()
    adapter = BoundedDictionaryAdapter(anchor["HC"], anchor["HI"], anchor["HO"], anchor["WS"], config.delta_max)
    adapter.config = config
    with torch.no_grad():
        adapter.gamma.copy_(initial_gamma)
    cs, is_, os = spatial
    frozen_st = spatial_reconstruction(cs, is_, os, anchor["WS"], anchor["HC"], anchor["HI"], anchor["HO"])["total"]
    train_cb, train_ib, train_time, train_event = train
    val_cb, val_ib, val_time, val_event = validation
    with torch.no_grad():
        initial_val_wb = infer_bulk_activities(val_cb, val_ib, anchor["HC"], anchor["HI"],
                                               lambda_bc=1 / val_cb.shape[1], lambda_bi=1 / val_ib.shape[1],
                                               steps=inner_steps, create_graph=False)
        initial_validation = _c_index(initial_val_wb, adapter.gamma, val_time, val_event)
    history = []
    best_state = deepcopy(adapter.state_dict())
    best_validation = initial_validation
    max_gradient_norm = 0.0
    clip_count = 0
    for outer in range(1, config.outer_iterations + 1):
        previous_state = deepcopy(adapter.state_dict())
        optimizer = torch.optim.Adam(adapter.parameters(), lr=config.learning_rate)
        final_losses = None
        for _ in range(config.epochs_per_iteration):
            optimizer.zero_grad(set_to_none=True)
            losses = refinement_loss(adapter, train_cb, train_ib, train_time, train_event, cs, is_, os, inner_steps)
            if not torch.isfinite(losses["total_loss"]):
                raise FloatingPointError("Nonfinite iterative refinement objective")
            losses["total_loss"].backward()
            if adapter.HO0.grad is not None or adapter.WS0.grad is not None:
                raise RuntimeError("Frozen spatial anchors received gradients")
            norm = torch.nn.utils.clip_grad_norm_(adapter.parameters(), config.gradient_clip, error_if_nonfinite=True)
            max_gradient_norm = max(max_gradient_norm, float(norm))
            clip_count += int(float(norm) > config.gradient_clip)
            optimizer.step()
            final_losses = losses
        with torch.no_grad():
            hc, hi, ho = adapter.dictionaries()
            st = spatial_reconstruction(cs, is_, os, adapter.WS0, hc, hi, ho)
            degradation = (st["total"] - frozen_st) / max(frozen_st, np.finfo(float).tiny)
            hc_mean = _mean_cosine(hc, adapter.HC0)
            hi_mean = _mean_cosine(hi, adapter.HI0)
            val_wb = infer_bulk_activities(val_cb, val_ib, hc, hi, lambda_bc=1 / val_cb.shape[1],
                                           lambda_bi=1 / val_ib.shape[1], steps=inner_steps, create_graph=False)
            validation_c = _c_index(val_wb, adapter.gamma, val_time, val_event)
            delta_c, delta_i = adapter.deltas()
            maximum = float(torch.maximum(delta_c.abs().max(), delta_i.abs().max()))
            accepted = hc_mean >= config.hc_gate and hi_mean >= config.hi_gate and degradation <= config.st_degradation_gate
            record = {"iteration": outer, "accepted": accepted, "validation_c_index": validation_c,
                      "HC_anchor_cosine": hc_mean, "HI_anchor_cosine": hi_mean,
                      "HC_per_niche_anchor_cosine": _per_niche_cosine(hc, adapter.HC0),
                      "HI_per_niche_anchor_cosine": _per_niche_cosine(hi, adapter.HI0),
                      "ST_reconstruction_change": degradation, "adapter_max_abs_delta": maximum,
                      "final_losses": {key: float(value) for key, value in final_losses.items() if key != "WB"}}
        rollback_if_rejected(adapter, previous_state, accepted)
        if accepted and validation_c > best_validation:
            best_validation = validation_c
            best_state = deepcopy(adapter.state_dict())
        history.append(record)
    adapter.load_state_dict(best_state)
    with torch.no_grad():
        hc, hi, ho = adapter.dictionaries()
    return adapter, history, {"selected_validation_c_index": best_validation,
                              "max_gradient_norm": max_gradient_norm, "gradient_clip_count": clip_count,
                              "all_finite": all(torch.isfinite(value).all() for value in adapter.state_dict().values()),
                              "HC_anchor_cosine": _mean_cosine(hc, adapter.HC0),
                              "HI_anchor_cosine": _mean_cosine(hi, adapter.HI0)}


def evaluate_refinement(adapter, anchor, train, validation, test, spatial, truth, matching, inner_steps):
    hc, hi, ho = adapter.dictionaries()
    activities = {}
    for name, values in (("train", train), ("validation", validation), ("test", test)):
        cb, ib, time, event = values
        activities[name] = infer_bulk_activities(cb, ib, hc, hi, lambda_bc=1 / cb.shape[1],
                                                 lambda_bi=1 / ib.shape[1], steps=inner_steps,
                                                 create_graph=False)
    cs, is_, os = spatial
    ws_eval = infer_spatial_activities(cs, is_, os, hc, hi, ho, steps=max(200, inner_steps), create_graph=False)
    per_niche = []
    for true_index, learned_index in enumerate(matching):
        per_niche.append({"true_index": true_index, "learned_index": int(learned_index),
                          "WB_recovery": pearson_correlation(activities["test"][:, learned_index].numpy(), truth["WB_test"][:, true_index].numpy()),
                          "WS_truth_recovery": pearson_correlation(ws_eval[:, learned_index].numpy(), truth["WS"][:, true_index].numpy()),
                          "WS_anchor_correlation": pearson_correlation(ws_eval[:, learned_index].numpy(), anchor["WS"][:, learned_index].numpy()),
                          "gamma": float(adapter.gamma[learned_index].detach()),
                          "HC_anchor_cosine": cosine_similarity(hc[learned_index].detach().numpy(), anchor["HC"][learned_index].numpy()),
                          "HI_anchor_cosine": cosine_similarity(hi[learned_index].detach().numpy(), anchor["HI"][learned_index].numpy())})
    return {"train_c_index": _c_index(activities["train"], adapter.gamma, train[2], train[3]),
            "validation_c_index": _c_index(activities["validation"], adapter.gamma, validation[2], validation[3]),
            "test_c_index": _c_index(activities["test"], adapter.gamma, test[2], test[3]),
            "gamma": adapter.gamma.detach().cpu().tolist(), "per_niche": per_niche,
            "HC_anchor_cosine": _mean_cosine(hc, anchor["HC"]),
            "HI_anchor_cosine": _mean_cosine(hi, anchor["HI"]),
            "WS_anchor_correlation": float(np.mean([row["WS_anchor_correlation"] for row in per_niche])),
            "spatial_reconstruction": spatial_reconstruction(cs, is_, os, ws_eval, hc, hi, ho)}
