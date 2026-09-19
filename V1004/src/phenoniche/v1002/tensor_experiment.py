import json
import math
from pathlib import Path
import resource
import time
import numpy as np
import torch
from phenoniche.evaluation.metrics import concordance_index, pearson_correlation
from phenoniche.evaluation.patient_protocol import fit_frozen_cox, split_patients
from phenoniche.v1001.simulation import generate_neighborhood_structured
from phenoniche.v1002.bulk_bridge import directional_communication_potential
from phenoniche.v1002.large_lr import build_large_spatial_matrix, summarize_pairs
from phenoniche.v1002.lr_atlas import load_lr_atlas, make_lr_subsets
from phenoniche.v1002.stress_test import _active_specifications, _hi_for_atlas, _molecular_states, _recover, _scale_matrix
from phenoniche.v1002.tensor_lr import dense_tensor_sse, exact_tensor_sse_from_statistics, fit_tensor_st, infer_tensor_activity


def _save(path, value):
    def default(item):
        if isinstance(item, torch.Tensor):
            return item.detach().cpu().tolist()
        if isinstance(item, np.ndarray):
            return item.tolist()
        if isinstance(item, np.generic):
            return item.item()
        raise TypeError(f"Cannot encode {type(item)}")
    Path(path).write_text(json.dumps(value, indent=2, default=default) + "\n", encoding="utf-8")


def numerical_audit(seed=4402):
    torch.manual_seed(seed)
    n, pairs, lr_count, niches, rank = 5, 6, 11, 3, 4
    observed = torch.rand(n, pairs, lr_count, dtype=torch.float64)
    w = torch.rand(n, niches, dtype=torch.float64, requires_grad=True)
    g = torch.rand(niches, pairs, rank, dtype=torch.float64, requires_grad=True)
    v = torch.rand(lr_count, rank, dtype=torch.float64, requires_grad=True)
    dense = dense_tensor_sse(observed, w, g, v)
    base = observed[:, :, 0].clone()
    delta = observed - base[:, :, None]
    indices = torch.arange(pairs * lr_count)
    efficient = exact_tensor_sse_from_statistics(observed.square().sum(), base, indices, delta.reshape(n, -1), lr_count, w, g, v)
    dense_grad = torch.autograd.grad(dense, (w, g, v), retain_graph=True)
    efficient_grad = torch.autograd.grad(efficient, (w, g, v))
    gradient_errors = [float((left - right).abs().max()) for left, right in zip(dense_grad, efficient_grad)]
    return {
        "dense_sse": float(dense), "efficient_sse": float(efficient),
        "absolute_error": float(abs(dense - efficient)),
        "gradient_max_abs_error": {"WS": gradient_errors[0], "G": gradient_errors[1], "V": gradient_errors[2]},
        "value_allclose": bool(torch.allclose(dense, efficient, rtol=1e-10, atol=1e-10)),
        "gradient_allclose": bool(all(torch.allclose(left, right, rtol=1e-10, atol=1e-10) for left, right in zip(dense_grad, efficient_grad)))
    }


def _potential_chunks(expression, genes, atlas, chunk=128):
    for start in range(0, len(atlas), chunk):
        yield start, directional_communication_potential(expression, genes, atlas, lr_start=start, lr_stop=min(start + chunk, len(atlas)))


def _evaluate_seed(seed, cs, communication, os, truth, active, bulk_expression, genes, atlas, cb, wb, survival):
    started = time.perf_counter()
    model = fit_tensor_st(cs, communication, os, seed=seed)
    hi = model.raw_lr_dictionary().reshape(6, -1)
    factors = {"WS": model.WS, "HC": model.HC, "HI": hi, "HO": model.HO}
    recovery = _recover(factors, truth, active)
    projection_started = time.perf_counter()
    bulk_w = infer_tensor_activity(cb, _potential_chunks(bulk_expression, genes, atlas), model.HC, model.G, model.V, survival["communication_scale"])
    projection_seconds = time.perf_counter() - projection_started
    matching = recovery["matching"]
    wb_rows = [pearson_correlation(bulk_w[:, matching[k]], wb[:, k]) for k in range(6)]
    split = split_patients(len(wb), seed=90210)
    train = torch.tensor(split.train)
    gamma = fit_frozen_cox(bulk_w[train], survival["time"][train], survival["event"][train], ridge=0.01)
    cindex = {}
    for partition in ("train", "validation", "test"):
        indices = torch.tensor(getattr(split, partition))
        risk = (bulk_w[indices] @ gamma).numpy()
        cindex[partition] = concordance_index(survival["time"][indices].numpy(), survival["event"][indices].numpy(), risk)
    bulk = {
        "overall_WB": float(np.mean(wb_rows)), "per_niche_WB": wb_rows,
        "risk_A_WB": wb_rows[0], "neutral_B_WB": wb_rows[1], "protective_C_WB": wb_rows[2],
        "train_C": cindex["train"], "validation_C": cindex["validation"], "test_C": cindex["test"],
        "gamma_true_order": [float(gamma[matching[k]]) for k in range(6)]
    }
    return {
        "seed": seed, "recovery": recovery, "bulk": bulk,
        "training_seconds": model.training_seconds, "epoch_seconds": model.epoch_seconds,
        "projection_seconds": projection_seconds, "total_seconds": time.perf_counter() - started,
        "peak_RAM_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024), "history": model.history
    }


def _flat_rows(flat):
    full = flat["full_atlas"]
    recovery, bulk = full["recovery"], full["bulk"]
    st = {
        "Method": "Flat-LR", "Overall_HI": recovery["overall_HI"],
        "A_HI": recovery["per_niche"][0]["HI"], "B_HI": recovery["per_niche"][1]["HI"],
        "A_WS": recovery["per_niche"][0]["WS"], "B_WS": recovery["per_niche"][1]["WS"],
        "A_B_collision": recovery["A_B_collision"], "AUPRC": recovery["AUPRC"]
    }
    bulk_row = {
        "Method": "Flat-LR", "Risk_A_WB": bulk["per_niche_WB"][0],
        "Neutral_B_WB": bulk["per_niche_WB"][1], "Protective_C_WB": bulk["per_niche_WB"][2],
        "Validation_C": bulk["validation_C"], "Test_C": bulk["test_C"],
        "gamma_A": bulk["gamma_true_order"][0], "gamma_B": bulk["gamma_true_order"][1], "gamma_C": bulk["gamma_true_order"][2]
    }
    return st, bulk_row


def _tensor_rows(result):
    recovery, bulk = result["recovery"], result["bulk"]
    st = {
        "Method": "Tensor-shared-LR", "Overall_HI": recovery["overall_HI"],
        "A_HI": recovery["per_niche"][0]["HI"], "B_HI": recovery["per_niche"][1]["HI"],
        "A_WS": recovery["per_niche"][0]["WS"], "B_WS": recovery["per_niche"][1]["WS"],
        "A_B_collision": recovery["A_B_collision"], "AUPRC": recovery["AUPRC"]
    }
    bulk_row = {
        "Method": "Tensor-shared-LR", "Risk_A_WB": bulk["risk_A_WB"],
        "Neutral_B_WB": bulk["neutral_B_WB"], "Protective_C_WB": bulk["protective_C_WB"],
        "Validation_C": bulk["validation_C"], "Test_C": bulk["test_C"],
        "gamma_A": bulk["gamma_true_order"][0], "gamma_B": bulk["gamma_true_order"][1], "gamma_C": bulk["gamma_true_order"][2]
    }
    return st, bulk_row


def _mean_sd(results):
    accessors = {
        "Overall_HI": lambda x: x["recovery"]["overall_HI"], "A_HI": lambda x: x["recovery"]["per_niche"][0]["HI"],
        "B_HI": lambda x: x["recovery"]["per_niche"][1]["HI"], "A_WS": lambda x: x["recovery"]["per_niche"][0]["WS"],
        "B_WS": lambda x: x["recovery"]["per_niche"][1]["WS"], "AUPRC": lambda x: x["recovery"]["AUPRC"],
        "Risk_A_WB": lambda x: x["bulk"]["risk_A_WB"], "Neutral_B_WB": lambda x: x["bulk"]["neutral_B_WB"],
        "Protective_C_WB": lambda x: x["bulk"]["protective_C_WB"], "Validation_C": lambda x: x["bulk"]["validation_C"],
        "Test_C": lambda x: x["bulk"]["test_C"], "gamma_A": lambda x: x["bulk"]["gamma_true_order"][0],
        "gamma_B": lambda x: x["bulk"]["gamma_true_order"][1], "gamma_C": lambda x: x["bulk"]["gamma_true_order"][2]
    }
    summary = {}
    for name, accessor in accessors.items():
        values = np.asarray([accessor(result) for result in results])
        summary[name] = {"mean": float(values.mean()), "SD": float(values.std(ddof=1)) if len(values) > 1 else None}
    summary["A_B_collision_count"] = sum(result["recovery"]["A_B_collision"] for result in results)
    return summary


def run_tensor_experiment(output, atlas_path):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    flat_path = output / "flat_large_lr_baseline_summary.json"
    if not flat_path.is_file():
        raise FileNotFoundError("The preserved flat Full-atlas baseline is required")
    flat = json.loads(flat_path.read_text())
    atlas = load_lr_atlas(atlas_path)
    small = make_lr_subsets(atlas)["Small"]
    specs = _active_specifications(small)
    hi_true, active, _ = _hi_for_atlas(atlas, specs)
    core = generate_neighborhood_structured()
    opportunity, local_niche = summarize_pairs(core.neighborhoods, core.cells.cell_type, core.cells.niche_activity)
    spatial = build_large_spatial_matrix(opportunity, local_niche, hi_true, seed=20261002)
    wb = core.structured.truth["WB"].numpy()
    bulk_expression, genes = _molecular_states(wb, hi_true, atlas, 20261002)
    square_sum = sum(float(value.square().sum()) for _, value in _potential_chunks(bulk_expression, genes, atlas))
    bulk_rms = math.sqrt(square_sum / (len(wb) * 64 * len(atlas)))
    cb = core.structured.data.CB
    communication_scale = bulk_rms / float(cb.square().mean().sqrt())
    communication = _scale_matrix(spatial, communication_scale)
    cs = core.structured.data.CS
    topology_scale = float(core.structured.data.OS.square().mean().sqrt() / cs.square().mean().sqrt())
    os = core.structured.data.OS / topology_scale
    truth = {"HC": core.structured.truth["HC"].numpy(), "HI": hi_true.reshape(6, -1) / communication_scale,
             "HO": core.structured.truth["HO"].numpy() / topology_scale, "WS": core.structured.truth["WS"].numpy()}
    survival = {"time": core.structured.data.time, "event": core.structured.data.event, "communication_scale": communication_scale}
    audit = numerical_audit()
    _save(output / "numerical_audit.json", audit)
    seed31 = _evaluate_seed(31, cs, communication, os, truth, active, bulk_expression, genes, atlas, cb, wb, survival)
    _save(output / "tensor_seed31.json", seed31)
    flat_st, flat_bulk = _flat_rows(flat)
    tensor_st, tensor_bulk = _tensor_rows(seed31)
    margin = 0.02
    improved_hi = tensor_st["Overall_HI"] >= flat_st["Overall_HI"] + margin
    improved_risk = tensor_bulk["Risk_A_WB"] >= flat_bulk["Risk_A_WB"] + margin
    continued = improved_hi and improved_risk
    results = [seed31]
    if continued:
        results.extend(_evaluate_seed(seed, cs, communication, os, truth, active, bulk_expression, genes, atlas, cb, wb, survival) for seed in (32, 33))
    three_seed = {"continued_after_seed31": continued, "seeds": [result["seed"] for result in results], "per_seed": results,
                  "mean_SD": _mean_sd(results), "early_stop_rule": {"absolute_improvement_margin": margin,
                  "Overall_HI_improved": improved_hi, "Risk_A_WB_improved": improved_risk}}
    _save(output / "tensor_3seed.json", three_seed)
    flat_parameters = 6 * 8 * 8 * len(atlas)
    tensor_parameters = 6 * 8 * 8 * 64 + len(atlas) * 64
    parameter_count = {"K": 6, "C": 8, "L": len(atlas), "R": 64, "Flat": flat_parameters,
                       "Tensor": tensor_parameters, "compression_fold": flat_parameters / tensor_parameters}
    _save(output / "parameter_count.json", parameter_count)
    comparison = {"ST": [flat_st, tensor_st], "Bulk": [flat_bulk, tensor_bulk], "seed31_gate_passed": continued}
    _save(output / "flat_vs_tensor.json", comparison)
    tensor_mean = three_seed["mean_SD"]
    useful = continued and tensor_mean["Overall_HI"]["mean"] > flat_st["Overall_HI"] + margin and tensor_mean["Risk_A_WB"]["mean"] > flat_bulk["Risk_A_WB"] + margin
    protective_ok = tensor_mean["Protective_C_WB"]["mean"] >= flat_bulk["Protective_C_WB"] - 0.05
    signs_ok = tensor_mean["gamma_A"]["mean"] > 0 and tensor_mean["gamma_C"]["mean"] < 0
    collision_ok = three_seed["mean_SD"]["A_B_collision_count"] == 0
    if useful and protective_ok and signs_ok and collision_ok:
        conclusion = "A"
    elif improved_hi or improved_risk:
        conclusion = "B"
    else:
        conclusion = "C"
    summary = {"pytest": "pending", "parameters": parameter_count, "comparison": comparison, "three_seed": three_seed,
               "success_audit": {"useful_primary_metrics": useful, "protective_C_preserved": protective_ok,
               "gamma_signs_correct": signs_ok, "A_B_no_collision": collision_ok}, "conclusion": conclusion}
    _save(output / "summary.json", summary)
    return summary
