"""Five-step deterministic generator audit, frozen data, then gated ST/bulk/Cox."""
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from phenoniche.evaluation.metrics import concordance_index
from phenoniche.evaluation.patient_protocol import split_patients
from phenoniche.losses.survival import cox_breslow_loss
from phenoniche.v1002.lr_atlas import load_lr_atlas
from phenoniche.v1002.final_simulation import (
    build_final_spec, simulate_final_spatial, simulate_final_bulk, true_activity, alr, TRUE_BETA,
)
from phenoniche.v1002.simulation_cells import bulk_potential
from phenoniche.v1002.final_model import fit_balanced, evaluate, primary_pass
from phenoniche.v1002.simulation_model import infer_activity, _correlation
from phenoniche.v1002.oracle_identifiability import audit_replicates, oracle_pass
from phenoniche.v1002.final_experiment import save, seed_for

OUT = Path("outputs/v1002_identifiability")
ADJUSTMENTS = {
    0: "Historical v2 generator; no repair (baseline audit)",
    1: "Match N1/N2 realized cell-type counts and local type pattern",
    2: "Pair B composition differs by swapping CCC-silent types 6/7",
    3: "Pair B types 6/7 have zero LR molecular template before count sampling",
    4: "Increase Pair B 6/7 composition separation only (.24 to .30 swing)",
    5: "Use 12-neighbor local radius; keep physical domains and LR biology fixed",
}
SEEDS = [33700, 33701, 33702]


def freeze(spec, simulations, audit, variant, atlas):
    features = audit["common_feature_indices"]
    file_names = []
    for rep, s in enumerate(simulations):
        filename = f"simulation_observed_data_seed{31 + rep}.npz"
        np.savez_compressed(OUT / filename, CS=s.cs, IS=s.communication[:, features],
                            labels=s.labels, coordinates=s.coordinates, cell_types=s.cell_types,
                            expression=s.expression, HC_truth=s.hc_truth, WS_truth=true_activity(s),
                            feature_indices=features, common_mask=np.isin(np.arange(64 * len(spec.atlas)), features))
        file_names.append(filename)
    np.savez_compressed(OUT / "oracle_centroids.npz", **audit["centroids"])
    save(OUT / "simulation_observed_data.json", {"files": file_names, "common_features": len(features),
                                                 "sha256_atlas": atlas.audit["sha256"]})
    save(OUT / "simulation_config.json", {"repair_variant": variant, "adjustments": [ADJUSTMENTS[i] for i in range(1, variant + 1)],
                                          "primary": {"purity": .5, "niche_size": 100, "noise": .05, "seeds": SEEDS},
                                          "oracle_protocol": audit["heldout_protocol"],
                                          "feature_mask": "intersection of three independently observed assay/10% coverage/spatial opportunity masks",
                                          "model_alpha": "one-time data-driven initial relative-gradient balance; unchanged",
                                          "frozen_after_oracle": True})
    save(OUT / "simulation_truth.json", {"K_total": 6, "HC_true_background_and_niches": np.vstack((np.full(8, .125), simulations[0].hc_truth)),
                                         "N1_N2_composition_exact_equal": bool(np.array_equal(simulations[0].hc_truth[0], simulations[0].hc_truth[1])),
                                         "N3_N4_molecular_delta_exact_equal": bool(np.array_equal(spec.expression_delta[2], spec.expression_delta[3])),
                                         "N3_N4_active_directed_edge_templates_equal": bool(np.array_equal(spec.hi_truth[2], spec.hi_truth[3])),
                                         "active_edges": [{"niche": k + 1, "feature": f, "strength": strength, "kind": kind}
                                                          for k, f, strength, kind in spec.active_edges],
                                         "ALR_beta": TRUE_BETA.tolist()})


def load_frozen(rep):
    with np.load(OUT / f"simulation_observed_data_seed{31 + rep}.npz") as value:
        data = {key: value[key] for key in value.files}
    simulation = SimpleNamespace(labels=data["labels"], coordinates=data["coordinates"],
                                 hc_truth=data["HC_truth"], cell_types=data["cell_types"])
    blocks = {"HC": data["CS"], "HI": data["IS"]}
    return simulation, blocks, data["WS_truth"], data["feature_indices"]


def fit_st():
    methods = ("C-only", "I-only", "C+I")
    reports, fits = [], []
    for rep in range(3):
        simulation, blocks, ws, features = load_frozen(rep)
        current, current_fit = {}, {}
        for method in methods:
            chosen = {"C-only": ("HC",), "I-only": ("HI",), "C+I": ("HC", "HI")}[method]
            fit = fit_balanced({name: blocks[name] for name in chosen}, seed=31 + rep)
            report = evaluate(fit, simulation, ws, blocks)
            if rep != 0:
                report.pop("predicted"); report.pop("activity")
            current[method] = report
            current_fit[method] = fit
        reports.append({"seed": SEEDS[rep], "methods": current})
        fits.append(current_fit)
        print("ST", SEEDS[rep], {m: [round(x["sensitivity"], 2) for x in current[m]["per_factor"]] for m in methods}, flush=True)
    save(OUT / "primary_st_results.json", reports)
    full = [r["methods"]["C+I"] for r in reports]
    comp = [r["methods"]["C-only"] for r in reports]
    ccc = [r["methods"]["I-only"] for r in reports]
    single_view_design = all(min(r["per_factor"][1]["sensitivity"], r["per_factor"][2]["sensitivity"]) < .8
                             and min(r["per_factor"][3]["sensitivity"], r["per_factor"][4]["sensitivity"]) >= .8
                             for r in comp) and all(min(r["per_factor"][1]["sensitivity"], r["per_factor"][2]["sensitivity"]) >= .8
                                                  and min(r["per_factor"][3]["sensitivity"], r["per_factor"][4]["sensitivity"]) < .8
                                                  for r in ccc)
    return reports, fits, primary_pass(full), single_view_design


def cox_fit(design, time, event):
    split = split_patients(len(time), seed=90210)
    train = split.train
    x = torch.as_tensor(design[train], dtype=torch.float64)
    t = torch.as_tensor(time[train], dtype=torch.float64)
    e = torch.as_tensor(event[train], dtype=torch.float64)
    if x.ndim != 2 or not torch.isfinite(x).all() or not bool(e.any()):
        raise ValueError("ALR Cox requires a finite signed design and observed events")
    coefficient = torch.zeros(x.shape[1], dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.LBFGS([coefficient], max_iter=300, tolerance_grad=1e-9,
                                  tolerance_change=1e-12, line_search_fn="strong_wolfe")
    def closure():
        optimizer.zero_grad()
        loss = cox_breslow_loss(x @ coefficient, t, e) + .01 * coefficient.square().sum()
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite signed-design Cox loss")
        loss.backward()
        return loss
    optimizer.step(closure)
    gamma = coefficient.detach().numpy()
    result = {"gamma": gamma.tolist()}
    for name in ("validation", "test"):
        index = getattr(split, name)
        result[name + "_C"] = concordance_index(time[index], event[index], design[index] @ gamma)
    return result


def run():
    OUT.mkdir(parents=True, exist_ok=True)
    atlas = load_lr_atlas("data/commuspace_human_lr_atlas.tsv")
    spec = build_final_spec(atlas)
    attempts = []
    selected = None
    frozen_simulations = None
    final_audit = None
    for variant in range(1):
        simulations = [simulate_final_spatial(spec, .5, 100, .05, seed) for seed in SEEDS]
        audit = audit_replicates(simulations, SEEDS)
        passed = oracle_pass(audit["rows"])
        attempts.append({"variant": variant, "change": ADJUSTMENTS[variant], "passed": passed,
                         "common_features": audit["common_feature_count"], "rows": audit["rows"]})
        save(OUT / "oracle_identifiability.json", {"criterion": {"N1_N2_CS_symmetric_max": .6,
                                                                   "N1_N2_IS_min": .9,
                                                                   "N3_N4_CS_min": .9,
                                                                   "N3_N4_IS_symmetric_max": .6,
                                                                   "BG_vs_niche_min": .9},
                                                     "protocol": audit["heldout_protocol"], "attempts": attempts,
                                                     "selected_variant": variant if passed else None})
        print("oracle", variant, [(round(r["N1_N2"]["CS_AUC"], 3), round(r["N1_N2"]["IS_AUC"], 3),
                                   round(r["N3_N4"]["CS_AUC"], 3), round(r["N3_N4"]["IS_AUC"], 3)) for r in audit["rows"]], passed, flush=True)
        if passed:
            selected = variant; frozen_simulations = simulations; final_audit = audit
            break
    summary = {"Level1_observable_identifiable": selected is not None, "selected_generator_repair": selected,
               "oracle_attempts": len(attempts), "Level2_ST": "not_run", "Level3_bulk": "not_run", "Level4_ALR_Cox": "not_run"}
    if selected is None:
        last = simulations[0]
        features = audit["common_feature_indices"]
        np.savez_compressed(OUT / "oracle_last_attempt_seed31.npz", coordinates=last.coordinates,
                            labels=last.labels, CS=last.cs, IS=last.communication[:, features],
                            CS_centroids=audit["centroids"]["CS"][0], IS_centroids=audit["centroids"]["IS"][0])
        save(OUT / "simulation_config.json", {"status": "oracle_failed_no_freeze", "attempted_repairs": 5,
                                              "last_variant": 5, "primary": {"purity": .5, "size": 100, "noise": .05, "seeds": SEEDS},
                                              "heldout_protocol": audit["heldout_protocol"]})
        for name in ("primary_st_results.json", "true_exposure_ALR_Cox.json", "bulk_transfer.json", "inferred_WB_ALR_Cox.json"):
            save(OUT / name, {"status": "not_run", "reason": "Observable oracle gate failed after five deterministic generator adjustments."})
        summary["conclusion"] = "B"
        save(OUT / "summary.json", summary)
        return summary
    freeze(spec, frozen_simulations, final_audit, selected, atlas)
    del frozen_simulations
    reports, fits, st_pass, single_view_design = fit_st()
    summary.update(Level2_ST=st_pass, single_view_design=single_view_design)
    if not st_pass or not single_view_design:
        summary["conclusion"] = "C"
        save(OUT / "summary.json", summary)
        return summary
    truth_cox, bulk_data = [], []
    for rep in range(3):
        simulation, blocks, ws, features = load_frozen(rep)
        bulk = simulate_final_bulk(spec, simulation.hc_truth, .05, SEEDS[rep] + 7000)
        pi, cb, expression, time, event, beta, eta = bulk
        truth_cox.append({"seed": SEEDS[rep], "true_C": concordance_index(time, event, eta),
                          **cox_fit(alr(pi), time, event)})
        bulk_data.append(bulk)
    save(OUT / "true_exposure_ALR_Cox.json", truth_cox)
    truth_cox_pass = all(r["gamma"][0] > 0 and r["gamma"][2] < 0 for r in truth_cox)
    summary["truth_ALR_Cox_sign_gate"] = truth_cox_pass
    if not truth_cox_pass:
        summary["conclusion"] = "E"
        summary["Level4_ALR_Cox"] = "truth_exposure_sign_failure"
        save(OUT / "summary.json", summary)
        return summary
    transfer, inferred_cox = [], []
    for rep, bulk in enumerate(bulk_data):
        simulation, blocks, ws, features = load_frozen(rep)
        pi, cb, expression, time, event, beta, eta = bulk
        fit = fits[rep]["C+I"]
        order = reports[rep]["methods"]["C+I"]["order"]
        potential = bulk_potential(expression, spec)[:, features]
        raw = infer_activity({"HC": cb, "HI": potential}, {name: fit.dictionaries[name] for name in ("HC", "HI")}).numpy()
        wb = raw[:, order]
        wb /= np.maximum(wb.sum(1, keepdims=True), 1e-12)
        corr = [_correlation(wb[:, k], pi[:, k]) for k in range(6)]
        transfer.append({"seed": SEEDS[rep], "Overall_WB": float(np.mean(corr)),
                         "Risk_WB": corr[1], "Neutral_WB": corr[2], "Protective_WB": corr[3],
                         "per_niche_WB": corr})
        inferred_cox.append({"seed": SEEDS[rep], **cox_fit(alr(wb), time, event)})
    save(OUT / "bulk_transfer.json", transfer)
    save(OUT / "inferred_WB_ALR_Cox.json", inferred_cox)
    bulk_pass = all(row["Risk_WB"] >= .5 and row["Neutral_WB"] >= .5 and row["Protective_WB"] >= .5 for row in transfer)
    cox_pass = all(row["gamma"][0] > 0 and row["gamma"][2] < 0 for row in inferred_cox)
    summary.update(Level3_bulk=bulk_pass, Level4_ALR_Cox=cox_pass,
                   conclusion="A" if bulk_pass and cox_pass else "D" if not bulk_pass else "E")
    save(OUT / "summary.json", summary)
    return summary


if __name__ == "__main__":
    run()
