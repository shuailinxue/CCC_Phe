import json
from pathlib import Path
import time
import numpy as np
import torch
from phenoniche.evaluation.metrics import concordance_index
from phenoniche.v1002.lr_atlas import load_lr_atlas
from phenoniche.v1002.simulation_cells import CELL_TYPES, NICHE_NAMES, build_simulation_spec, bulk_potential, simulate_bulk, simulate_spatial
from phenoniche.v1002.simulation_model import evaluate_spatial, fit_multiview, infer_activity, match_niches, score_bulk


METHODS = ("C-only", "I-only", "C+I")


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


def _method_features(method, simulation):
    if method in ("C+I", "I-only"):
        return np.flatnonzero(simulation.final_mask.reshape(-1))
    return None


def _truth_active(spec, features):
    lookup = {int(feature): index for index, feature in enumerate(features)}
    active = []
    for niche in range(5):
        labels = np.zeros(len(features), dtype=bool)
        for current_niche, feature, _, _ in spec.active_edges:
            if current_niche == niche and feature in lookup:
                labels[lookup[feature]] = True
        active.append(labels)
    return active


def _fit_method(method, simulation, spec, model_seed):
    features = _method_features(method, simulation)
    cs_rms = np.sqrt(np.mean(simulation.cs ** 2))
    blocks = {}
    truth = {}
    scales = {}
    if method != "I-only":
        blocks["HC"] = simulation.cs
        truth["HC"] = simulation.hc_truth
    if method != "C-only":
        communication = simulation.communication[:, features]
        communication_scale = np.sqrt(np.mean(communication ** 2)) / cs_rms
        blocks["HI"] = communication / communication_scale
        truth["HI"] = spec.hi_truth[:, features] / communication_scale
        scales["communication"] = float(communication_scale)
    result = fit_multiview(blocks, niches=5, iterations=500, seed=model_seed)
    order = match_niches(result, truth)
    active = _truth_active(spec, features) if features is not None else [np.zeros(1, dtype=bool) for _ in range(5)]
    spatial = evaluate_spatial(result, order, simulation.labels, simulation.ws_truth, truth, active)
    return result, order, spatial, scales, features


def _bulk_method(method, result, order, scales, features, simulation, spec, bulk):
    pi, cb, expression, time_values, event, beta, eta = bulk
    blocks = {}
    dictionaries = {}
    if method != "I-only":
        blocks["HC"] = cb
        dictionaries["HC"] = result.dictionaries["HC"]
    if method != "C-only":
        potential = bulk_potential(expression, spec)[:, features] / scales["communication"]
        blocks["HI"] = potential
        dictionaries["HI"] = result.dictionaries["HI"]
    activity = infer_activity(blocks, dictionaries)
    return score_bulk(method, activity, order, pi, time_values, event), activity


def _run_setting(spec, purity, niche_size, noise, replicate, keep_maps=False):
    started = time.perf_counter()
    seed = 31000 + replicate + int(purity * 1000) + niche_size * 17 + int(noise * 10000)
    simulation = simulate_spatial(spec, purity, niche_size, noise, seed)
    bulk = simulate_bulk(spec, simulation.hc_truth, noise, seed + 7000)
    methods = []
    fitted = {}
    for offset, method in enumerate(METHODS):
        result, order, spatial, scales, features = _fit_method(method, simulation, spec, 31 + replicate)
        bulk_result, activity = _bulk_method(method, result, order, scales, features, simulation, spec, bulk)
        top_features = None
        if keep_maps and method == "C+I":
            scores = result.dictionaries["HI"][order[0]].numpy()
            local = np.argsort(-scores)[:15]
            top_features = [{"feature": int(features[index]), "weight": float(scores[index]),
                             "sender": int(features[index] // len(spec.atlas) // 8),
                             "receiver": int((features[index] // len(spec.atlas)) % 8),
                             "lr_id": spec.atlas.interactions[int(features[index] % len(spec.atlas))].lr_id}
                            for index in local]
        if not keep_maps:
            spatial.pop("predicted_labels")
            spatial.pop("aligned_WS")
        methods.append({"Method": method, "spatial": spatial, "bulk": bulk_result,
                        "nCCC": 0 if features is None else len(features), "top_CCC_features": top_features})
        fitted[method] = (result, order, scales, features, activity)
    signal = {feature for _, feature, _, _ in spec.active_edges}
    retained = set(np.flatnonzero(simulation.final_mask.reshape(-1)).tolist())
    filter_audit = {
        "Raw_LR_count": 8408, "Assay_measurable_LR_count": len(spec.atlas),
        "Raw_directed_CCC_count": 64 * 8408,
        "Coverage_filtered_directed_CCC_count": int(simulation.coverage_mask.sum()),
        "Opportunity_filtered_final_CCC_count": int(simulation.final_mask.sum()),
        "retained_LR_per_sender_receiver": simulation.final_mask.sum(1).reshape(8, 8).tolist(),
        "signal_edge_precision": len(retained & signal) / max(len(retained), 1),
        "signal_edge_recall": len(retained & signal) / len(signal)
    }
    true_c = concordance_index(bulk[3], bulk[4], bulk[6])
    record = {"purity": purity, "niche_size": niche_size, "noise": noise, "replicate": replicate,
              "seed": seed, "methods": methods, "filter_audit": filter_audit,
              "true_survival_C_index": true_c, "runtime_seconds": time.perf_counter() - started}
    return record, simulation, bulk, fitted


def _pseudobulk(simulation, spec, fitted):
    xbin = np.clip((simulation.coordinates[:, 0] / 8).astype(int), 0, 4)
    ybin = np.clip((simulation.coordinates[:, 1] / 9).astype(int), 0, 4)
    tiles = ybin * 5 + xbin
    composition = np.zeros((25, 8), dtype=np.float32)
    expression = np.zeros((25, 8, simulation.expression.shape[1]), dtype=np.float32)
    truth = np.zeros((25, 5), dtype=np.float32)
    global_by_type = np.stack([simulation.expression[simulation.cell_types == value].mean(0) for value in range(8)])
    for tile in range(25):
        selected = tiles == tile
        truth[tile] = simulation.ws_truth[selected].mean(0)
        for cell_type in range(8):
            current = selected & (simulation.cell_types == cell_type)
            composition[tile, cell_type] = current.sum()
            expression[tile, cell_type] = simulation.expression[current].mean(0) if current.any() else global_by_type[cell_type]
    composition /= composition.sum(1, keepdims=True)
    output = {}
    for method in ("C-only", "C+I"):
        result, order, scales, features, _ = fitted[method]
        blocks = {"HC": composition}
        dictionaries = {"HC": result.dictionaries["HC"]}
        if method == "C+I":
            blocks["HI"] = bulk_potential(expression, spec)[:, features] / scales["communication"]
            dictionaries["HI"] = result.dictionaries["HI"]
        activity = infer_activity(blocks, dictionaries).numpy()[:, order]
        recovery = [float(np.corrcoef(activity[:, niche], truth[:, niche])[0, 1]) for niche in range(5)]
        output[method] = {"overall": float(np.mean(recovery)), "per_niche": recovery}
    return output


def run_simulation_benchmark(output, atlas_path):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    full_atlas = load_lr_atlas(atlas_path)
    spec = build_simulation_spec(full_atlas)
    settings = [(purity, size, noise) for purity in (0.3, 0.5, 0.7) for size in (50, 100, 150) for noise in (0.0, 0.05, 0.10)]
    grid = []
    primary_records = []
    primary_assets = None
    total_started = time.perf_counter()
    for purity, size, noise in settings:
        for replicate in range(3):
            primary = purity == 0.5 and size == 100 and noise == 0.05
            record, simulation, bulk, fitted = _run_setting(spec, purity, size, noise, replicate, keep_maps=primary and replicate == 0)
            grid.append(record)
            if primary:
                primary_records.append(record)
                if replicate == 0:
                    primary_assets = (simulation, bulk, fitted)
    simulation, bulk, fitted = primary_assets
    pseudobulk = _pseudobulk(simulation, spec, fitted)
    primary_first = next(record for record in grid if record["purity"] == 0.5 and record["niche_size"] == 100 and record["noise"] == 0.05 and record["replicate"] == 0)
    truth = {"cell_types": list(CELL_TYPES), "niche_names": list(NICHE_NAMES), "coordinates": simulation.coordinates,
             "labels": simulation.labels, "HC_true": simulation.hc_truth, "WS_true": simulation.ws_truth,
             "active_edges": [{"niche": niche + 1, "feature": feature, "sender": feature // len(spec.atlas) // 8,
                               "receiver": (feature // len(spec.atlas)) % 8, "lr_index": feature % len(spec.atlas),
                               "lr_id": spec.atlas.interactions[feature % len(spec.atlas)].lr_id, "strength": strength, "category": category}
                              for niche, feature, strength, category in spec.active_edges]}
    _save(output / "filter_audit.json", primary_first["filter_audit"])
    _save(output / "simulation_truth.json", truth)
    _save(output / "primary_results.json", primary_records)
    _save(output / "robustness_grid.json", grid)
    _save(output / "ablation_results.json", {"methods": list(METHODS), "records": grid})
    _save(output / "bulk_transfer.json", [{"purity": row["purity"], "niche_size": row["niche_size"], "noise": row["noise"],
                                            "replicate": row["replicate"], "methods": [method["bulk"] for method in row["methods"]]} for row in grid])
    _save(output / "pseudobulk_bridge.json", pseudobulk)
    _save(output / "cox_results.json", [{"purity": row["purity"], "niche_size": row["niche_size"], "noise": row["noise"],
                                         "replicate": row["replicate"], "methods": [{key: method["bulk"][key] for key in
                                         ("Method", "Validation_C", "Test_C", "gamma_1", "gamma_2", "gamma_3", "risk_sign_correct", "protective_sign_correct", "neutral_consistent")}
                                         for method in row["methods"]]} for row in grid])
    _save(output / "runtime.json", {"total_seconds": time.perf_counter() - total_started,
                                     "settings": 27, "replicates_per_setting": 3, "model_fits": 27 * 3 * len(METHODS)})
    summary = summarize_results(grid, pseudobulk)
    summary["pytest"] = "pending"
    _save(output / "summary.json", summary)
    return summary


def summarize_results(grid, pseudobulk):
    primary = [row for row in grid if row["purity"] == 0.5 and row["niche_size"] == 100 and row["noise"] == 0.05]
    def aggregate(rows, method, section, key):
        values = [next(item for item in row["methods"] if item["Method"] == method)[section][key] for row in rows]
        if any(value is None for value in values):
            return {"mean": None, "SD": None}
        return {"mean": float(np.mean(values)), "SD": float(np.std(values, ddof=1))}
    primary_table = {}
    for method in METHODS:
        primary_table[method] = {key: aggregate(primary, method, "spatial", key) for key in
                                 ("niche_sensitivity", "localization_AUC", "WS_recovery", "HC_recovery", "HI_recovery", "CCC_AUPRC")}
        primary_table[method].update({key: aggregate(primary, method, "bulk", key) for key in
                                      ("Risk_WB", "Neutral_WB", "Protective_WB", "Validation_C", "Test_C", "gamma_1", "gamma_2", "gamma_3")})
    hard_case = {}
    for method in METHODS:
        current = []
        for row in primary:
            spatial = next(item for item in row["methods"] if item["Method"] == method)["spatial"]
            current.append({"Niche1": spatial["per_niche"][0], "Niche2": spatial["per_niche"][1],
                            "collision": spatial["Niche1_Niche2_collision"]})
        hard_case[method] = current
    robustness = []
    for noise in (0.0, 0.05, 0.10):
        rows = [row for row in grid if row["noise"] == noise]
        robustness.append({"noise": noise, "C+I_sensitivity": aggregate(rows, "C+I", "spatial", "niche_sensitivity"),
                           "C+I_localization_AUC": aggregate(rows, "C+I", "spatial", "localization_AUC"),
                           "C+I_CCC_AUPRC": aggregate(rows, "C+I", "spatial", "CCC_AUPRC"),
                           "C+I_Risk_WB": aggregate(rows, "C+I", "bulk", "Risk_WB"),
                           "C+I_Test_C": aggregate(rows, "C+I", "bulk", "Test_C")})
    full = primary_table["C+I"]
    composition = primary_table["C-only"]
    discovery_ok = full["niche_sensitivity"]["mean"] >= 0.6 and full["localization_AUC"]["mean"] >= 0.75
    transfer_ok = full["Risk_WB"]["mean"] >= 0.5 and full["Protective_WB"]["mean"] >= 0.5
    phenotype_ok = full["gamma_1"]["mean"] > 0 and full["gamma_3"]["mean"] < 0
    if discovery_ok and transfer_ok and phenotype_ok:
        conclusion = "A"
    elif discovery_ok:
        conclusion = "B"
    else:
        conclusion = "C"
    return {"filtering": primary[0]["filter_audit"], "primary": primary_table, "hard_case": hard_case,
            "pseudobulk": pseudobulk, "robustness": robustness,
            "judgement": {"CCC_adds_over_composition": full["Risk_WB"]["mean"] > composition["Risk_WB"]["mean"],
                          "discovery_ok": discovery_ok, "transfer_ok": transfer_ok, "phenotype_ok": phenotype_ok},
            "conclusion": conclusion}
