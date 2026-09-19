from dataclasses import asdict
import json
from pathlib import Path
import time
import tracemalloc
import numpy as np
import torch
from phenoniche.evaluation.metrics import cosine_similarity, pearson_correlation
from phenoniche.evaluation.niche_stability import build_consensus, match_factors, recovery_report
from phenoniche.evaluation.patient_protocol import split_patients
from phenoniche.evaluation.structured_st_first_benchmark import _inner_audit, _partition, _prepare, _score_frozen, _truth_for_partitions
from phenoniche.simulation.structured_ccc import StructuredCCCConfig, generate_structured_ccc
from phenoniche.training.st_first import STFirstConfig, StructuredScaling, fit_st_only
from phenoniche.v1001.neighborhoods import NeighborhoodConfig
from phenoniche.v1001.simulation import generate_neighborhood_structured


SEEDS = (31, 32, 33, 34, 35)


def save_json(path, value):
    def default(item):
        if isinstance(item, torch.Tensor):
            return item.detach().cpu().tolist()
        if isinstance(item, np.ndarray):
            return item.tolist()
        if isinstance(item, np.generic):
            return item.item()
        raise TypeError(f"Cannot encode {type(item)}")
    Path(path).write_text(json.dumps(value, indent=2, default=default), encoding="utf-8")


def prepare_v1001(programs=12):
    generated = generate_neighborhood_structured(StructuredCCCConfig(programs=programs), NeighborhoodConfig())
    structured = generated.structured
    split = split_patients(structured.data.CB.shape[0], seed=90210)
    scaling = StructuredScaling.fit(structured.data.CB[split.train], structured.data.IB[split.train],
                                    structured.data.CS, structured.data.OS)
    bulk = scaling.bulk(structured.data.CB, structured.data.IB)
    spatial = scaling.spatial(structured.data.CS, structured.data.IS, structured.data.OS)
    truth = {name: value.clone() for name, value in structured.truth.items()}
    truth["HI"] /= scaling.communication
    truth["HO"] /= scaling.topology
    prepared = (structured, split, scaling, bulk, spatial, truth)
    return prepared, generated


def _compact_recovery(report):
    records = []
    for row in report["per_niche"]:
        edge = {key: row["edges"][key] for key in ("top_20_precision", "top_20_recall", "top_50_precision", "top_50_recall")}
        records.append({key: value for key, value in row.items() if key != "edges"} | edge)
    return {key: value for key, value in report.items() if key != "per_niche"} | {"per_niche": records}


def _fit_ensemble(spatial, seeds):
    return [fit_st_only(*spatial, STFirstConfig(seed=seed, iterations=500)) for seed in seeds]


def _aligned_stability(results):
    factors = [{name: getattr(result, name).numpy() for name in ("WS", "HC", "HI", "HO")} for result in results]
    reference_index = int(np.argmin([result.history[-1]["weighted_mean_squared_error"] for result in results]))
    reference = factors[reference_index]
    aligned = []
    orders = []
    for current in factors:
        order, _ = match_factors(current, reference)
        orders.append(order.tolist())
        aligned.append({name: current[name][order] if name.startswith("H") else current[name][:, order]
                        for name in ("WS", "HC", "HI", "HO")})
    pairs = {name: [] for name in ("HC", "HI", "HO", "WS")}
    for left in range(len(aligned)):
        for right in range(left + 1, len(aligned)):
            for name in pairs:
                count = aligned[left]["HC"].shape[0]
                values = [cosine_similarity(aligned[left][name][:, index], aligned[right][name][:, index])
                          if name == "WS" else cosine_similarity(aligned[left][name][index], aligned[right][name][index])
                          for index in range(count)]
                pairs[name].append(float(np.mean(values)))
    return {"reference_seed": results[reference_index].seed, "alignments": orders,
            "pairwise_mean": {name: float(np.mean(values)) for name, values in pairs.items()},
            "pairwise_min": {name: float(np.min(values)) for name, values in pairs.items()}}


def _evaluate_method(prepared, seeds):
    generated, split, scaling, _, spatial, truth = prepared
    results = _fit_ensemble(spatial, seeds)
    seed_recovery = [_compact_recovery(recovery_report({name: getattr(result, name) for name in ("WS", "HC", "HI", "HO")},
                                                        truth, generated.metadata["feature_mapping"])) for result in results]
    anchor, consensus = build_consensus(results, *spatial)
    recovery = _compact_recovery(recovery_report(anchor, truth, generated.metadata["feature_mapping"]))
    partitions = {name: _partition(prepared, getattr(split, name)) for name in ("train", "validation", "test")}
    truth_partitioned = _truth_for_partitions(truth, split)
    inner_audit = _inner_audit(anchor, partitions["train"])
    frozen, _ = _score_frozen(anchor, partitions, truth_partitioned, recovery["matching"], 50)
    first, second = recovery["matching"][:2]
    separation = 1 - pearson_correlation(anchor["WS"][:, first], anchor["WS"][:, second])
    return {"seeds": list(seeds), "seed_recovery": seed_recovery, "consensus_recovery": recovery,
            "stability": _aligned_stability(results), "consensus": consensus,
            "A_B_WS_separation": separation, "inner_inference_audit": inner_audit,
            "bulk_cox": frozen, "scaling": asdict(scaling)}, anchor


def _baseline_ccc_runtime(generated):
    config = StructuredCCCConfig(**generated.metadata["config"])
    cs = generated.data.CS.numpy()
    ws = generated.truth["WS"].numpy()
    hi = generated.truth["HI"].numpy()
    sender = np.repeat(np.arange(config.cell_types), config.cell_types * config.programs)
    receiver = np.tile(np.repeat(np.arange(config.cell_types), config.programs), config.cell_types)
    tracemalloc.start()
    started = time.perf_counter()
    opportunity = cs[:, sender] * cs[:, receiver]
    interaction = ws @ hi
    raw = opportunity * (config.communication_baseline + interaction)
    normalized = raw / (opportunity + config.opportunity_tau)
    elapsed = time.perf_counter() - started
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return {"construction_seconds": elapsed, "peak_python_memory_bytes": int(peak),
            "anchors": cs.shape[0], "features": normalized.shape[1]}


def _null_leakage(generated):
    composition = generated.structured.data.CS.numpy()
    opportunity = generated.structured.data.Opp_S.numpy()
    tau = generated.structured.metadata["opportunity_normalization"]["tau"]
    programs = generated.structured.metadata["config"]["programs"]
    raw_null = opportunity.copy()
    normalized = raw_null / (opportunity + tau)
    raw_correlations = []
    normalized_correlations = []
    for sender in range(8):
        for receiver in range(8):
            abundance = composition[:, sender] * composition[:, receiver]
            index = (sender * 8 + receiver) * programs
            if abundance.std() > 1e-8 and opportunity[:, index].std() > 1e-8:
                raw_correlations.append(abs(np.corrcoef(raw_null[:, index], abundance)[0, 1]))
                normalized_correlations.append(abs(np.corrcoef(normalized[:, index], abundance)[0, 1]))
    return {"mean_abs_raw_vs_abundance_correlation": float(np.mean(raw_correlations)),
            "mean_abs_normalized_vs_abundance_correlation": float(np.mean(normalized_correlations)),
            "correlation_reduction": float(np.mean(raw_correlations) - np.mean(normalized_correlations)),
            "features_audited": len(raw_correlations)}


def _summary_row(name, method):
    recovery = method["consensus_recovery"]
    records = recovery["per_niche"]
    return {"Method": name, "Overall_HC": recovery["mean_HC_recovery"], "Overall_HI": recovery["mean_HI_recovery"],
            "Overall_HO": recovery["mean_HO_recovery"], "Overall_WS": recovery["mean_WS_recovery"],
            "A_HI": records[0]["HI_recovery"], "A_WS": records[0]["WS_recovery"],
            "B_HI": records[1]["HI_recovery"], "B_WS": records[1]["WS_recovery"],
            "A_B_collision": recovery["HI_only_A_B_collision"],
            "A_B_WS_separation": method["A_B_WS_separation"],
            "Top_50_precision": float(np.mean([row["top_50_precision"] for row in records])),
            "Top_50_recall": float(np.mean([row["top_50_recall"] for row in records])),
            "HI_stability": method["stability"]["pairwise_mean"]["HI"],
            "WS_stability": method["stability"]["pairwise_mean"]["WS"]}


def _bulk_row(name, method):
    result = method["bulk_cox"]["per_niche"]
    return {"Method": name, "Risk_WB_recovery": result[0]["WB_recovery"],
            "Protective_WB_recovery": result[2]["WB_recovery"],
            "Validation_C": method["bulk_cox"]["validation_c_index"], "Test_C": method["bulk_cox"]["test_c_index"],
            "A_gamma": result[0]["gamma"], "B_gamma": result[1]["gamma"], "C_gamma": result[2]["gamma"]}


def _diagnostic_truth():
    rng = np.random.default_rng(8401)
    niches, anchors, cell_types, programs, topology = 6, 600, 4, 2, 12
    ws = np.full((anchors, niches), 0.01)
    for niche in range(niches):
        ws[niche * 100:(niche + 1) * 100, niche] = 1.0
    ws /= ws.sum(1, keepdims=True)
    hc = rng.uniform(0.02, 0.4, (niches, cell_types))
    hc /= hc.sum(1, keepdims=True)
    hc[1] = hc[0]
    ho = rng.uniform(0.01, 0.3, (niches, topology))
    ho /= ho.sum(1, keepdims=True)
    ho[1] = ho[0]
    center = np.full((niches, cell_types, cell_types, programs), 1e-5)
    center[:, 0, 1, 0] = 0.4
    center[:, 1, 0, 1] = 0.3
    for niche in range(2, niches):
        center[niche, 0, (niche % 3) + 1, niche % programs] += 1.0
    full = center.copy()
    full[0, 1, 2, 0] += 1.5
    full[1, 3, 2, 1] += 1.5
    for niche in range(2, niches):
        full[niche, (niche % 3) + 1, ((niche + 1) % 3) + 1, niche % programs] += 1.2
    center = center.reshape(niches, -1)
    full = full.reshape(niches, -1)
    cs = torch.tensor(ws @ hc, dtype=torch.float32)
    os = torch.tensor(ws @ ho, dtype=torch.float32)
    return cs, os, torch.tensor(ws @ center, dtype=torch.float32), torch.tensor(ws @ full, dtype=torch.float32), {
        "WS": torch.tensor(ws, dtype=torch.float32), "HC": torch.tensor(hc, dtype=torch.float32),
        "HI": torch.tensor(full, dtype=torch.float32), "HO": torch.tensor(ho, dtype=torch.float32)}, cell_types, programs


def _mechanism_method(cs, is_, os, truth, cell_types, programs):
    results = _fit_ensemble((cs, is_, os), SEEDS)
    anchor, _ = build_consensus(results, cs, is_, os)
    mapping = [{"index": index, "sender": index // (cell_types * programs),
                "receiver": (index // programs) % cell_types, "program": index % programs}
               for index in range(cell_types * cell_types * programs)]
    report = _compact_recovery(recovery_report(anchor, truth, mapping))
    rows = report["per_niche"]
    return {"X_HI": rows[0]["HI_recovery"], "Y_HI": rows[1]["HI_recovery"],
            "X_WS": rows[0]["WS_recovery"], "Y_WS": rows[1]["WS_recovery"],
            "collision": report["HI_only_A_B_collision"]}


def run_mechanism_diagnostic():
    cs, os, center_is, full_is, truth, cell_types, programs = _diagnostic_truth()
    return {"design": {"target_niches": ["X", "Y"], "composition_identical": True,
                       "center_related_CCC_identical": True, "peripheral_CCC_different": True,
                       "unchanged_factorizer_K": 6, "auxiliary_control_niches": 4},
            "V1000-center-only": _mechanism_method(cs, center_is, os, truth, cell_types, programs),
            "V1001-neighborhood-internal": _mechanism_method(cs, full_is, os, truth, cell_types, programs)}


def run_v1001_experiment(output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    baseline_prepared = _prepare(12)
    primary_prepared, generated = prepare_v1001(12)
    baseline, _ = _evaluate_method(baseline_prepared, SEEDS)
    primary, _ = _evaluate_method(primary_prepared, SEEDS)
    save_json(output / "v1000_baseline.json", baseline)
    save_json(output / "v1001_primary.json", primary)
    baseline_generated = baseline_prepared[0]
    truth_preserved = {name: bool(torch.equal(baseline_generated.truth[name], generated.structured.truth[name]))
                       for name in ("HC", "HI", "HO", "WB", "WS", "beta", "eta")}
    simulation_audit = {"dimensions": generated.structured.metadata["config"], "truth_preserved": truth_preserved,
                        "bulk_observations_preserved": {name: bool(torch.equal(getattr(baseline_generated.data, name),
                                                                               getattr(generated.structured.data, name)))
                                                        for name in ("CB", "IB", "Iraw_B", "Opp_B", "time", "event")},
                        "composition_expected_correlation": generated.audit["composition_expected_correlation"],
                        "feature_mapping_preserved": baseline_generated.metadata["feature_mapping"] == generated.structured.metadata["feature_mapping"]}
    save_json(output / "simulation_audit.json", simulation_audit)
    neighborhood_audit = {key: generated.audit[key] for key in ("anchors", "cells", "mean_cells_per_neighborhood",
                                                                 "mean_directed_pairs_per_neighborhood", "neighborhood",
                                                                 "composition_row_sum_max_error")}
    save_json(output / "neighborhood_audit.json", neighborhood_audit)
    construction_audit = {key: generated.audit[key] for key in ("local_directed_pair_instances", "peripheral_pair_instances",
                                                                 "peripheral_pair_fraction", "peripheral_signal_fraction",
                                                                 "peripheral_normalized_l1", "ordered_pairs", "self_pairs_excluded")}
    construction_audit["null_opportunity_audit"] = _null_leakage(generated)
    save_json(output / "ccc_construction_audit.json", construction_audit)
    mechanism = run_mechanism_diagnostic()
    mechanism["primary_peripheral_audit"] = construction_audit
    save_json(output / "peripheral_ccc_diagnostic.json", mechanism)
    sensitivity = []
    for programs in (2, 4, 8):
        seeds = SEEDS[:3]
        baseline_method, _ = _evaluate_method(_prepare(programs), seeds)
        v1001_prepared, _ = prepare_v1001(programs)
        v1001_method, _ = _evaluate_method(v1001_prepared, seeds)
        sensitivity.extend([{"Q": programs, "F": programs * 64, **_summary_row("V1000", baseline_method)},
                            {"Q": programs, "F": programs * 64, **_summary_row("V1001", v1001_method)}])
    sensitivity.extend([{"Q": 12, "F": 768, **_summary_row("V1000", baseline)},
                        {"Q": 12, "F": 768, **_summary_row("V1001", primary)}])
    save_json(output / "ccc_dimension_sensitivity.json", sensitivity)
    bulk = [_bulk_row("V1000", baseline), _bulk_row("V1001", primary)]
    save_json(output / "bulk_projection.json", {row["Method"]: {"Risk_WB_recovery": row["Risk_WB_recovery"],
                                                                  "Protective_WB_recovery": row["Protective_WB_recovery"]}
                                                   for row in bulk})
    save_json(output / "cox_results.json", bulk)
    baseline_runtime = _baseline_ccc_runtime(baseline_generated)
    runtime = {"V1000": baseline_runtime,
               "V1001": {"construction_seconds": generated.audit["construction_seconds"],
                          "peak_python_memory_bytes": generated.audit["peak_python_memory_bytes"],
                          "cells": generated.audit["cells"], "anchors": generated.audit["anchors"],
                          "local_directed_cell_pair_instances": generated.audit["local_directed_pair_instances"],
                          "mean_cells_per_neighborhood": generated.audit["mean_cells_per_neighborhood"],
                          "mean_directed_pairs_per_neighborhood": generated.audit["mean_directed_pairs_per_neighborhood"]}}
    runtime["construction_time_ratio_V1001_over_V1000"] = runtime["V1001"]["construction_seconds"] / baseline_runtime["construction_seconds"]
    save_json(output / "runtime_memory.json", runtime)
    table = [_summary_row("V1000", baseline), _summary_row("V1001", primary)]
    answers = {"1_entire_neighborhood_CCC_included": bool(construction_audit["peripheral_pair_instances"] > 0 and
                                                           construction_audit["ordered_pairs"] and construction_audit["self_pairs_excluded"]),
               "2_peripheral_information_added": bool(mechanism["V1001-neighborhood-internal"]["collision"] is False and
                                                       mechanism["V1000-center-only"]["collision"] is True),
               "3_same_composition_recovery_improved": bool(table[1]["A_HI"] + table[1]["B_HI"] > table[0]["A_HI"] + table[0]["B_HI"] and
                                                             table[1]["A_WS"] + table[1]["B_WS"] > table[0]["A_WS"] + table[0]["B_WS"]),
               "4_bulk_projection_maintained_or_improved": bool(bulk[1]["Risk_WB_recovery"] >= bulk[0]["Risk_WB_recovery"] and
                                                                 bulk[1]["Protective_WB_recovery"] >= bulk[0]["Protective_WB_recovery"]),
               "5_computationally_acceptable": bool(runtime["V1001"]["construction_seconds"] < 60 and
                                                     runtime["V1001"]["peak_python_memory_bytes"] < 2_000_000_000)}
    summary = {"ST_comparison": table, "bulk_comparison": bulk, "mechanism_diagnostic": mechanism, "runtime": runtime,
               "five_answers": answers}
    save_json(output / "summary.json", summary)
    return summary
