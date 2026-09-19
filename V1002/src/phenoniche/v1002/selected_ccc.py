import json
from pathlib import Path
import numpy as np
import torch
from phenoniche.evaluation.metrics import concordance_index, pearson_correlation
from phenoniche.evaluation.patient_protocol import fit_frozen_cox, split_patients
from phenoniche.inference.bulk import infer_bulk_activities
from phenoniche.v1001.simulation import generate_neighborhood_structured
from phenoniche.v1002.bulk_bridge import directional_communication_potential
from phenoniche.v1002.large_lr import build_large_spatial_matrix, fit_large_st, summarize_pairs
from phenoniche.v1002.lr_atlas import load_lr_atlas, make_lr_subsets
from phenoniche.v1002.stress_test import SEEDS, _active_specifications, _consensus, _hi_for_atlas, _molecular_states, _project_expression, _scale_matrix


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


def select_top_union(hi, top_m=100):
    if hi.ndim != 2 or not isinstance(top_m, int) or top_m < 1 or top_m > hi.shape[1]:
        raise ValueError("HI and Top-M are invalid")
    indices = torch.topk(hi, top_m, dim=1, largest=True, sorted=True).indices
    return torch.unique(indices.reshape(-1), sorted=True)


def selected_directional_potential(expression, genes, atlas, selected, chunk=128):
    selected = torch.as_tensor(selected, dtype=torch.long)
    lr_count = len(atlas)
    pairs = torch.div(selected, lr_count, rounding_mode="floor")
    lr = selected % lr_count
    result = expression.new_empty((expression.shape[0], len(selected)))
    for start in range(0, lr_count, chunk):
        stop = min(start + chunk, lr_count)
        mask = (lr >= start) & (lr < stop)
        if not mask.any():
            continue
        potential = directional_communication_potential(expression, genes, atlas, lr_start=start, lr_stop=stop)
        result[:, mask] = potential[:, pairs[mask], lr[mask] - start]
    return result


def project_selected_niches(composition, potential, hc, hi, steps=50):
    return infer_bulk_activities(composition, potential, hc, hi, lambda_bc=1 / composition.shape[1],
                                 lambda_bi=1 / potential.shape[1], steps=steps, create_graph=False)


def _score(method, activity, matching, wb_true, time, event):
    recovery = [pearson_correlation(activity[:, matching[k]], wb_true[:, k]) for k in range(6)]
    split = split_patients(len(wb_true), seed=90210)
    train = torch.tensor(split.train)
    gamma = fit_frozen_cox(activity[train], time[train], event[train], ridge=0.01)
    cindices = {}
    for partition in ("validation", "test"):
        indices = torch.tensor(getattr(split, partition))
        cindices[partition] = concordance_index(time[indices].numpy(), event[indices].numpy(), (activity[indices] @ gamma).numpy())
    gamma_true_order = [float(gamma[matching[k]]) for k in range(6)]
    return {
        "Method": method, "nCCC": None, "Risk_A_WB": recovery[0], "Neutral_B_WB": recovery[1],
        "Protective_C_WB": recovery[2], "Validation_C": cindices["validation"], "Test_C": cindices["test"],
        "gamma_A": gamma_true_order[0], "gamma_B": gamma_true_order[1], "gamma_C": gamma_true_order[2]
    }


def run_selected_ccc_experiment(output, atlas_path):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    atlas = load_lr_atlas(atlas_path)
    specs = _active_specifications(make_lr_subsets(atlas)["Small"])
    hi_true, active, _ = _hi_for_atlas(atlas, specs)
    core = generate_neighborhood_structured()
    opportunity, local_niche = summarize_pairs(core.neighborhoods, core.cells.cell_type, core.cells.niche_activity)
    spatial = build_large_spatial_matrix(opportunity, local_niche, hi_true, seed=20261002)
    wb_true = core.structured.truth["WB"].numpy()
    bulk_expression, genes = _molecular_states(wb_true, hi_true, atlas, 20261002)
    square_sum = 0.0
    for start in range(0, len(atlas), 128):
        value = directional_communication_potential(bulk_expression, genes, atlas, lr_start=start, lr_stop=start + 128)
        square_sum += float(value.square().sum())
    bulk_rms = np.sqrt(square_sum / (len(wb_true) * 64 * len(atlas)))
    cb = core.structured.data.CB
    communication_scale = bulk_rms / float(cb.square().mean().sqrt())
    communication = _scale_matrix(spatial, communication_scale)
    cs = core.structured.data.CS
    topology_scale = float(core.structured.data.OS.square().mean().sqrt() / cs.square().mean().sqrt())
    os = core.structured.data.OS / topology_scale
    fitted = [fit_large_st(cs, communication, os, seed=seed) for seed in SEEDS]
    anchor, _ = _consensus(fitted, cs, communication, os)
    truth = {"HC": core.structured.truth["HC"].numpy(), "HI": hi_true.reshape(6, -1) / communication_scale,
             "HO": core.structured.truth["HO"].numpy() / topology_scale, "WS": core.structured.truth["WS"].numpy()}
    from phenoniche.v1002.stress_test import _recover
    recovery = _recover(anchor, truth, active)
    matching = recovery["matching"]
    full_w, _ = _project_expression(bulk_expression, genes, atlas, cb, anchor["HC"], anchor["HI"], communication_scale)
    selected = select_top_union(anchor["HI"], top_m=100)
    selected_potential = selected_directional_potential(bulk_expression, genes, atlas, selected) / communication_scale
    selected_w = project_selected_niches(cb, selected_potential, anchor["HC"], anchor["HI"][:, selected])
    full = _score("Full-CCC", full_w, matching, wb_true, core.structured.data.time, core.structured.data.event)
    selected_row = _score("ST-selected-CCC", selected_w, matching, wb_true, core.structured.data.time, core.structured.data.event)
    full["nCCC"] = 64 * len(atlas)
    selected_row["nCCC"] = len(selected)
    signal = {index for _, index in active}
    selected_set = set(selected.tolist())
    audit = {"top_m_per_niche": 100, "selected_CCC": len(selected_set), "true_signal_CCC": len(signal),
             "signal_edge_precision": len(selected_set & signal) / len(selected_set),
             "signal_edge_recall": len(selected_set & signal) / len(signal),
             "selection_inputs": ["ST-derived HI"], "phenotype_free": True}
    risk_gain = selected_row["Risk_A_WB"] - full["Risk_A_WB"]
    protective_drop = full["Protective_C_WB"] - selected_row["Protective_C_WB"]
    gamma_a_fixed = selected_row["gamma_A"] > 0
    gamma_b_improved = abs(selected_row["gamma_B"]) < abs(full["gamma_B"])
    if risk_gain >= 0.10 and gamma_a_fixed and gamma_b_improved and protective_drop <= 0.05:
        conclusion = "A"
    elif risk_gain > 0 or gamma_a_fixed or gamma_b_improved:
        conclusion = "B"
    else:
        conclusion = "C"
    comparison = {"ST_recovery_frozen": recovery, "methods": [full, selected_row]}
    _save(output / "selected_ccc.json", audit)
    _save(output / "full_vs_selected.json", comparison)
    summary = {"pytest": "pending", "selected_CCC": len(selected_set), "signal_edge_precision": audit["signal_edge_precision"],
               "signal_edge_recall": audit["signal_edge_recall"], "methods": [full, selected_row], "conclusion": conclusion}
    _save(output / "summary.json", summary)
    return summary
