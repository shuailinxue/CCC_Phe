from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, replace
import hashlib
import json
import logging
import multiprocessing
from pathlib import Path
import numpy as np
import torch
from phenoniche.evaluation.challenging_benchmark import save_json
from phenoniche.evaluation.metrics import concordance_index, pearson_correlation
from phenoniche.evaluation.niche_stability import build_consensus, recovery_report, stability_report
from phenoniche.evaluation.patient_protocol import fit_frozen_cox, split_patients
from phenoniche.inference.bulk import infer_bulk_activities
from phenoniche.inference.nnls_reference import nnls_reference
from phenoniche.losses.survival import cox_breslow_loss
from phenoniche.simulation.structured_ccc import StructuredCCCConfig, generate_structured_ccc
from phenoniche.training.iterative_refinement import IterativeRefinementConfig, evaluate_refinement, run_iterative_refinement
from phenoniche.training.st_first import STFirstConfig, StructuredScaling, fit_st_only


def _fit_st_job(cs, is_, os, seed):
    torch.set_num_threads(1)
    return fit_st_only(cs, is_, os, STFirstConfig(seed=seed, iterations=500))


def _fit_st_ensemble(cs, is_, os, seeds, workers):
    results = []
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn")) as executor:
        futures = {executor.submit(_fit_st_job, cs, is_, os, seed): seed for seed in seeds}
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            logging.warning("ST-only Q=%s seed=%s complete (%s/%s)", is_.shape[1] // 64, result.seed, len(results), len(futures))
    return sorted(results, key=lambda result: result.seed)


def _prepare(programs):
    generated = generate_structured_ccc(StructuredCCCConfig(programs=programs))
    split = split_patients(generated.data.CB.shape[0], seed=90210)
    scaling = StructuredScaling.fit(generated.data.CB[split.train], generated.data.IB[split.train],
                                    generated.data.CS, generated.data.OS)
    cb, ib = scaling.bulk(generated.data.CB, generated.data.IB)
    cs, is_, os = scaling.spatial(generated.data.CS, generated.data.IS, generated.data.OS)
    truth = {name: value.clone() for name, value in generated.truth.items()}
    truth["HI"] /= scaling.communication
    truth["HO"] /= scaling.topology
    return generated, split, scaling, (cb, ib), (cs, is_, os), truth


def _partition(prepared, indices, time=None, event=None):
    generated, _, _, bulk, _, _ = prepared
    selected = torch.tensor(indices, dtype=torch.long)
    return (bulk[0][selected], bulk[1][selected],
            generated.data.time[selected] if time is None else time,
            generated.data.event[selected] if event is None else event)


def _bulk_objective(cb, ib, wb, hc, hi):
    return float((cb - wb @ hc).square().mean() + (ib - wb @ hi).square().mean())


def _inner_audit(anchor, training):
    cb, ib, time, event = training
    subset = slice(0, min(48, cb.shape[0]))
    cb_small, ib_small = cb[subset], ib[subset]
    reference = nnls_reference(cb_small, ib_small, anchor["HC"], anchor["HI"],
                               lambda_bc=1 / cb.shape[1], lambda_bi=1 / ib.shape[1])
    reference_objective = _bulk_objective(cb_small, ib_small, reference, anchor["HC"], anchor["HI"])
    records = []
    for steps in (10, 25, 50, 100, 200):
        inferred = infer_bulk_activities(cb_small, ib_small, anchor["HC"], anchor["HI"],
                                         lambda_bc=1 / cb.shape[1], lambda_bi=1 / ib.shape[1],
                                         steps=steps, create_graph=False)
        objective = _bulk_objective(cb_small, ib_small, inferred, anchor["HC"], anchor["HI"])
        excess = (objective - reference_objective) / max(reference_objective, np.finfo(float).tiny)
        records.append({"steps": steps, "objective": objective, "relative_excess_to_NNLS": excess,
                        "all_finite": bool(torch.isfinite(inferred).all())})
    eligible = [row["steps"] for row in records if row["relative_excess_to_NNLS"] <= 0.001 and row["all_finite"]]
    selected = min(eligible) if eligible else 200
    hc = anchor["HC"].detach().clone().requires_grad_(True)
    hi = anchor["HI"].detach().clone().requires_grad_(True)
    wb = infer_bulk_activities(cb, ib, hc, hi, lambda_bc=1 / cb.shape[1], lambda_bi=1 / ib.shape[1], steps=selected)
    gamma = torch.linspace(0.5, -0.5, anchor["HC"].shape[0])
    loss = cox_breslow_loss(wb @ gamma, time, event)
    gradients = torch.autograd.grad(loss, (hc, hi))
    return {"selection_rule": "smallest step count with relative objective excess to NNLS <= 0.001",
            "records": records, "selected_steps": selected, "reference_objective": reference_objective,
            "gradient_norm_HC": float(gradients[0].norm()), "gradient_norm_HI": float(gradients[1].norm()),
            "gradients_finite": all(torch.isfinite(value).all() for value in gradients)}


def _infer(anchor, partition, steps):
    cb, ib, _, _ = partition
    return infer_bulk_activities(cb, ib, anchor["HC"], anchor["HI"],
                                 lambda_bc=1 / cb.shape[1], lambda_bi=1 / ib.shape[1],
                                 steps=steps, create_graph=False)


def _score_frozen(anchor, partitions, truth, matching, steps, training_override=None):
    train = partitions["train"] if training_override is None else training_override
    train_wb = _infer(anchor, train, steps)
    gamma = fit_frozen_cox(train_wb, train[2], train[3], ridge=0.01)
    activities = {"train": train_wb,
                  "validation": _infer(anchor, partitions["validation"], steps),
                  "test": _infer(anchor, partitions["test"], steps)}
    scores = {name: concordance_index(partitions[name][2].numpy(), partitions[name][3].numpy(),
                                      (activities[name] @ gamma).numpy())
              for name in ("train", "validation", "test")}
    records = []
    for true_index, learned_index in enumerate(matching):
        records.append({"true_index": true_index, "learned_index": int(learned_index),
                        "WB_recovery": pearson_correlation(activities["test"][:, learned_index].numpy(), truth["WB_test"][:, true_index].numpy()),
                        "gamma": float(gamma[learned_index]),
                        "gamma_sign_correct": bool(gamma[learned_index] * truth["beta"][true_index] > 0) if truth["beta"][true_index] else None})
    return {"train_c_index": scores["train"], "validation_c_index": scores["validation"],
            "test_c_index": scores["test"], "gamma": gamma.tolist(), "per_niche": records}, gamma


def _truth_for_partitions(truth, split):
    result = {name: value for name, value in truth.items()}
    result["WB_train"] = truth["WB"][split.train]
    result["WB_validation"] = truth["WB"][split.validation]
    result["WB_test"] = truth["WB"][split.test]
    return result


def _run_methods(prepared, seeds, workers, inner_steps=None, stop_after_st=False):
    generated, split, scaling, _, spatial, truth = prepared
    results = _fit_st_ensemble(*spatial, seeds, workers)
    seed_recovery = [recovery_report({name: getattr(result, name) for name in ("WS", "HC", "HI", "HO")},
                                     truth, generated.metadata["feature_mapping"]) for result in results]
    anchor, consensus = build_consensus(results, *spatial)
    consensus_recovery = recovery_report(anchor, truth, generated.metadata["feature_mapping"])
    truth_partitioned = _truth_for_partitions(truth, split)
    partitions = {name: _partition(prepared, getattr(split, name)) for name in ("train", "validation", "test")}
    audit = _inner_audit(anchor, partitions["train"]) if inner_steps is None else None
    steps = audit["selected_steps"] if audit is not None else inner_steps
    run = {"results": results, "seed_recovery": seed_recovery, "anchor": anchor,
            "consensus": consensus, "consensus_recovery": consensus_recovery,
            "stability": stability_report(results), "partitions": partitions,
            "truth": truth_partitioned, "inner_audit": audit,
            "inner_steps": steps, "scaling": scaling}
    if stop_after_st:
        return run
    return _complete_methods(run, spatial)


def _complete_methods(run, spatial):
    frozen, gamma = _score_frozen(run["anchor"], run["partitions"], run["truth"],
                                  run["consensus_recovery"]["matching"], run["inner_steps"])
    adapter, history, diagnostics = run_iterative_refinement(
        run["anchor"], run["partitions"]["train"], run["partitions"]["validation"],
        spatial, gamma, run["inner_steps"])
    iterative = evaluate_refinement(
        adapter, run["anchor"], run["partitions"]["train"], run["partitions"]["validation"],
        run["partitions"]["test"], spatial, run["truth"],
        run["consensus_recovery"]["matching"], run["inner_steps"])
    iterative["training_diagnostics"] = diagnostics
    run.update({"frozen": frozen, "iterative": iterative, "adapter": adapter, "history": history})
    return run


def _sensitivity_row(programs, run):
    recovery = run["consensus_recovery"]
    a, b = recovery["per_niche"][0], recovery["per_niche"][1]
    frozen_a = run["frozen"]["per_niche"][0]
    iterative_a = run["iterative"]["per_niche"][0]
    return {"Q": programs, "F": programs * 64,
            "ST_A_HI_recovery": a["HI_recovery"], "ST_B_HI_recovery": b["HI_recovery"],
            "A_B_collision": recovery["HC_HI_A_B_collision"],
            "A_WS_recovery": a["WS_recovery"], "B_WS_recovery": b["WS_recovery"],
            "Frozen_WB_risk_recovery": frozen_a["WB_recovery"], "Frozen_test_c_index": run["frozen"]["test_c_index"],
            "Iterative_WB_risk_recovery": iterative_a["WB_recovery"], "Iterative_test_c_index": run["iterative"]["test_c_index"],
            "final_HC_anchor_cosine": run["iterative"]["HC_anchor_cosine"],
            "final_HI_anchor_cosine": run["iterative"]["HI_anchor_cosine"],
            "final_WS_anchor_correlation": run["iterative"]["WS_anchor_correlation"]}


def run_structured_st_first(output, workers=4):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    protocol = {"primary": asdict(StructuredCCCConfig()), "model_seeds": [31, 32, 33, 34, 35],
                "sensitivity": {"Q": [2, 4, 8], "model_seeds": [31, 32, 33]},
                "st_iterations": 500, "iterative": asdict(IterativeRefinementConfig()),
                "patient_split_seed": 90210, "inner_steps_candidates": [10, 25, 50, 100, 200],
                "inner_selection": "reconstruction accuracy only; no phenotype or truth",
                "success_gate_before_refinement": "consensus A/B HI>=0.65, WS>=0.60, no HI-only or HC+HI collision",
                "forbidden_mechanisms": ["anti-collapse", "dual-bank", "entropy", "orthogonality", "JSD", "contrastive", "GNN"]}
    save_json(output / "protocol.json", protocol)
    primary_prepared = _prepare(12)
    generated = primary_prepared[0]
    truth_output = {"specification": generated.metadata["config"], "A_B_truth": generated.metadata["A_B_truth"],
                    "active_edges_per_niche": generated.metadata["active_edges_per_niche"],
                    "active_edges": generated.metadata["active_edges"], "feature_mapping": generated.metadata["feature_mapping"],
                    "true_survival_c_index": generated.metadata["true_survival_c_index"],
                    "event_fraction": generated.metadata["event_fraction"],
                    "opportunity_normalization": generated.metadata["opportunity_normalization"]}
    save_json(output / "structured_truth.json", truth_output)
    primary = _run_methods(primary_prepared, (31, 32, 33, 34, 35), workers, stop_after_st=True)
    save_json(output / "inner_inference_audit.json", primary["inner_audit"])
    st_only = {"seeds": [result.seed for result in primary["results"]],
               "seed_recovery": primary["seed_recovery"], "consensus_recovery": primary["consensus_recovery"],
               "stability": primary["stability"], "consensus": primary["consensus"],
               "scaling": asdict(primary["scaling"])}
    save_json(output / "st_only_primary.json", st_only)
    a, b = primary["consensus_recovery"]["per_niche"][:2]
    discovery_passed = (a["HI_recovery"] >= 0.65 and b["HI_recovery"] >= 0.65 and
                        a["WS_recovery"] >= 0.60 and b["WS_recovery"] >= 0.60 and
                        not primary["consensus_recovery"]["HI_only_A_B_collision"] and
                        not primary["consensus_recovery"]["HC_HI_A_B_collision"])
    if not discovery_passed:
        summary = {"status": "stopped_after_ST_only", "failure_layer": "A. ST niche discovery",
                   "discovery_gate_passed": False, "st_only": st_only,
                   "simulation": truth_output, "inner_inference_audit": primary["inner_audit"]}
        save_json(output / "summary.json", summary)
        logging.warning("Structured experiment stopped: ST-only discovery gate failed")
        return summary
    primary = _complete_methods(primary, primary_prepared[4])
    save_json(output / "st_frozen_primary.json", primary["frozen"])
    save_json(output / "st_iterative_primary.json", primary["iterative"])
    save_json(output / "outer_iteration_history.json", primary["history"])
    edge_output = {row["niche"]: row["edges"] for row in primary["consensus_recovery"]["per_niche"]}
    save_json(output / "niche_edge_recovery.json", edge_output)
    np.savez_compressed(output / "stable_anchor.npz", **{name: value.numpy() for name, value in primary["anchor"].items()})
    torch.save({"state_dict": primary["adapter"].state_dict(), "config": asdict(IterativeRefinementConfig()),
                "inner_steps": primary["inner_steps"]}, output / "iterative_adapter.pt")
    permutation_records = []
    training = primary["partitions"]["train"]
    for seed in (101, 102, 103, 104, 105):
        order = torch.tensor(np.random.default_rng(seed).permutation(training[0].shape[0]), dtype=torch.long)
        permuted = (training[0], training[1], training[2][order], training[3][order])
        frozen, gamma = _score_frozen(primary["anchor"], primary["partitions"], primary["truth"],
                                      primary["consensus_recovery"]["matching"], primary["inner_steps"], permuted)
        adapter, history, diagnostics = run_iterative_refinement(primary["anchor"], permuted, primary["partitions"]["validation"],
                                                                 primary_prepared[4], gamma, primary["inner_steps"])
        evaluated = evaluate_refinement(adapter, primary["anchor"], permuted, primary["partitions"]["validation"],
                                        primary["partitions"]["test"], primary_prepared[4], primary["truth"],
                                        primary["consensus_recovery"]["matching"], primary["inner_steps"])
        permutation_records.append({"permutation_seed": seed, "frozen": frozen, "iterative": evaluated,
                                    "history": history, "diagnostics": diagnostics})
    permutation = {"runs": permutation_records,
                   "mean_test_c_index": float(np.mean([row["iterative"]["test_c_index"] for row in permutation_records])),
                   "std_test_c_index": float(np.std([row["iterative"]["test_c_index"] for row in permutation_records], ddof=1)),
                   "ST_anchor_sha256": hashlib.sha256(np.concatenate([primary["anchor"][name].numpy().ravel() for name in ("HC", "HI", "HO")]).tobytes()).hexdigest()}
    save_json(output / "phenotype_permutation.json", permutation)
    sensitivity = []
    for programs in (2, 4, 8):
        run = _run_methods(_prepare(programs), (31, 32, 33), workers, primary["inner_steps"])
        sensitivity.append(_sensitivity_row(programs, run))
    sensitivity.append(_sensitivity_row(12, primary))
    save_json(output / "ccc_dimension_sensitivity.json", sensitivity)
    summary = {"status": "complete", "discovery_gate_passed": True,
               "simulation": truth_output, "st_only": st_only,
               "frozen": primary["frozen"], "iterative": primary["iterative"],
               "outer_iterations": primary["history"], "permutation": permutation,
               "sensitivity": sensitivity, "inner_inference_audit": primary["inner_audit"]}
    save_json(output / "summary.json", summary)
    logging.warning("Structured ST-first benchmark complete")
    return summary
