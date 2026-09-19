import numpy as np
import torch
from phenoniche.inference.bulk import infer_bulk_activities
from phenoniche.inference.nnls_reference import nnls_reference
from phenoniche.model.niche_factorization import NicheFactorization
from phenoniche.utils.seed import set_seed


def reconstruction(activity, cb, ib, hc, hi, bc, bi):
    return bc * (cb - activity @ hc).square().sum() + bi * (ib - activity @ hi).square().sum()


def scan_inner_inference(data, weights, seed=31):
    set_seed(seed)
    cases = []
    for k in (4, 6, 8):
        model = NicheFactorization(data.CB.shape[0], data.CS.shape[0], data.CB.shape[1], data.OS.shape[1], data.IB.shape[1], k)
        initial = model.niche_dictionaries()
        cases.append((f"initial_K{k}", initial["HC"].detach(), initial["HI"].detach()))
        joined = torch.cat((data.CB, data.CS), dim=0)
        communications = torch.cat((data.IB, data.IS), dim=0)
        positions = torch.linspace(0, joined.shape[0] - 1, k).long()
        cases.append((f"observed_rows_K{k}", joined[positions].clone(), communications[positions].clone()))
    records = []
    for name, hc, hi in cases:
        for mode, bc in (("full", weights.bc), ("ccc_only", 0.0)):
            reference = nnls_reference(data.CB, data.IB, hc, hi, bc, weights.bi)
            optimum = float(reconstruction(reference, data.CB, data.IB, hc, hi, bc, weights.bi))
            initial = float(reconstruction(torch.zeros_like(reference), data.CB, data.IB, hc, hi, bc, weights.bi))
            for steps in (10, 25, 50, 100):
                for rate in (0.5, 1.0):
                    inferred = infer_bulk_activities(data.CB, data.IB, hc, hi, bc, weights.bi, steps, rate, False)
                    final = float(reconstruction(inferred, data.CB, data.IB, hc, hi, bc, weights.bi))
                    records.append({"case": name, "mode": mode, "steps": steps, "learning_rate": rate,
                                    "initial_reconstruction": initial, "final_reconstruction": final,
                                    "relative_improvement": (initial - final) / max(initial, 1e-12),
                                    "nnls_reconstruction": optimum, "relative_excess_over_initial": (final - optimum) / max(initial, 1e-12),
                                    "finite": bool(torch.isfinite(inferred).all())})
    aggregates = []
    for steps in (10, 25, 50, 100):
        for rate in (0.5, 1.0):
            rows = [row for row in records if row["steps"] == steps and row["learning_rate"] == rate]
            aggregates.append({"steps": steps, "learning_rate": rate,
                               "max_relative_excess": max(row["relative_excess_over_initial"] for row in rows),
                               "mean_relative_excess": float(np.mean([row["relative_excess_over_initial"] for row in rows])),
                               "finite": all(row["finite"] for row in rows)})
    eligible = [row for row in aggregates if row["finite"] and row["max_relative_excess"] <= 1e-4]
    selected = min(eligible, key=lambda row: (row["steps"], row["mean_relative_excess"])) if eligible else min(aggregates, key=lambda row: row["max_relative_excess"])
    return {"criterion": "Minimum steps with maximum excess/initial <= 1e-4; otherwise minimum numerical excess. No outcomes used.",
            "selected": selected, "tolerance_met": bool(eligible), "aggregates": aggregates, "runs": records}


def final_inference_audit(candidate, data, weights):
    factors = candidate.factors
    reference = nnls_reference(data.CB, data.IB, factors["HC"], factors["HI"], weights.bc, weights.bi)
    actual = candidate.train_inferred
    initial = float(reconstruction(torch.zeros_like(actual), data.CB, data.IB, factors["HC"], factors["HI"], weights.bc, weights.bi))
    final = float(reconstruction(actual, data.CB, data.IB, factors["HC"], factors["HI"], weights.bc, weights.bi))
    optimum = float(reconstruction(reference, data.CB, data.IB, factors["HC"], factors["HI"], weights.bc, weights.bi))
    return {"initial_reconstruction": initial, "final_reconstruction": final, "nnls_reconstruction": optimum,
            "relative_improvement": (initial - final) / max(initial, 1e-12),
            "relative_excess_over_initial": (final - optimum) / max(initial, 1e-12)}
