"""Reproducible v2 primary gate and conditional bulk/robustness benchmark."""
import json
from pathlib import Path
import numpy as np
import torch
from phenoniche.evaluation.metrics import concordance_index
from phenoniche.evaluation.patient_protocol import fit_frozen_cox, split_patients
from phenoniche.v1002.lr_atlas import load_lr_atlas
from phenoniche.v1002.simulation_cells import bulk_potential
from phenoniche.v1002.simulation_model import infer_activity, _correlation
from phenoniche.v1002.final_simulation import (
    build_final_spec, simulate_final_spatial, simulate_final_bulk, true_activity, alr, TRUE_BETA,
)
from phenoniche.v1002.final_model import fit_balanced, evaluate, primary_pass

OUT = Path("outputs/v1002_final_sim")
METHODS = ("C-only", "I-only", "C+I")


def save(path, value):
    def encode(item):
        if isinstance(item, np.ndarray): return item.tolist()
        if isinstance(item, np.generic): return item.item()
        if isinstance(item, torch.Tensor): return item.tolist()
        raise TypeError(type(item))
    Path(path).write_text(json.dumps(value, indent=2, default=encode) + "\n", encoding="utf-8")


def seed_for(purity, size, noise, rep):
    return 31000 + rep + int(purity * 1000) + size * 17 + int(noise * 10000)


def make_blocks(s):
    features = np.flatnonzero(s.final_mask.reshape(-1))
    return {"HC": s.cs, "HI": s.communication[:, features]}, features


def fit_method(name, blocks, model_seed):
    if name == "C+I":
        return fit_balanced(blocks, model_seed)
    if name == "C-only":
        return fit_balanced({"HC": blocks["HC"]}, model_seed)
    if name == "I-only":
        return fit_balanced({"HI": blocks["HI"]}, model_seed)
    raise ValueError(f"Unknown C/I method: {name}")


def filtering(spec, s):
    retained = set(np.flatnonzero(s.final_mask.reshape(-1)).tolist())
    signal = {feature for _, feature, _, _ in spec.active_edges}
    return {"raw_LR": 8408, "raw_directed_CCC": 64 * 8408,
            "assay_measurable_LR": len(spec.atlas), "assay_directed_CCC": 64 * len(spec.atlas),
            "coverage_directed_CCC": int(s.coverage_mask.sum()),
            "retained_CCC": len(retained), "signal_edges": len(signal),
            "signal_retained": len(signal & retained),
            "signal_retention_recall": len(signal & retained) / max(len(signal), 1),
            "opportunity_min_anchors": max(20, int(np.ceil(.01 * len(s.labels)))),
            "feature_identity": "(sender, receiver, LR index); identical ST/bulk mask",
            "selection_uses": ["assay measurability", "10% sender/receiver coverage", "spatial pair opportunity"]}


def compact(report, keep_map=False):
    if not keep_map:
        report.pop("predicted")
        report.pop("activity")
    return report


def setting(spec, purity, size, noise, rep, methods, keep_map=False):
    data_seed = seed_for(purity, size, noise, rep)
    s = simulate_final_spatial(spec, purity, size, noise, data_seed)
    blocks, features = make_blocks(s)
    ws_truth = true_activity(s)
    reports = {}
    fits = {}
    for method in methods:
        fit = fit_method(method, blocks, 31 + rep)
        report = evaluate(fit, s, ws_truth, blocks)
        reports[method] = compact(report, keep_map)
        fits[method] = fit
    return {"purity": purity, "size": size, "noise": noise, "replicate": rep,
            "data_seed": data_seed, "methods": reports,
            "filter": filtering(spec, s)}, (s, blocks, features, ws_truth, fits)


def collinearity(x):
    x = np.asarray(x, dtype=np.float64)
    centered = x - x.mean(0)
    sd = centered.std(0)
    standardized = centered / np.maximum(sd, 1e-12)
    condition = float(np.linalg.cond(np.column_stack((np.ones(len(x)), x))))
    vifs = []
    for j in range(x.shape[1]):
        others = np.delete(standardized, j, axis=1)
        target = standardized[:, j]
        residual = target - others @ np.linalg.lstsq(others, target, rcond=None)[0]
        r2 = 1 - np.sum(residual ** 2) / max(np.sum(target ** 2), 1e-12)
        vifs.append(float(1 / max(1 - r2, 1e-12)))
    return {"condition_number_with_intercept": condition, "VIF": vifs,
            "pairwise_correlation": np.corrcoef(x, rowvar=False).tolist()}


def cox_compare(wb, pi, time, event, eta_true):
    split = split_patients(len(pi), seed=90210)
    train = split.train
    wb = wb / np.maximum(wb.sum(1, keepdims=True), 1e-12)
    designs = {"Raw-W": wb, "ALR": alr(wb)}
    result = {"true_C": concordance_index(time, event, eta_true), "beta_ALR": TRUE_BETA.tolist()}
    for name, x in designs.items():
        gamma = fit_frozen_cox(torch.tensor(x[train], dtype=torch.float32), torch.tensor(time[train]),
                               torch.tensor(event[train]), ridge=.01).numpy()
        stats = {"gamma": gamma.tolist(), "collinearity": collinearity(x[train])}
        for partition in ("validation", "test"):
            idx = getattr(split, partition)
            stats[partition + "_C"] = concordance_index(time[idx], event[idx], x[idx] @ gamma)
        result[name] = stats
    result["raw_interpretation"] = "Six compositional columns plus an intercept are rank deficient; absolute effects are not identified."
    return result


def bulk_transfer(spec, asset, report, noise, data_seed):
    s, blocks, features, _, fits = asset
    fit = fits["C+I"]
    order = report["order"]
    pi, cb, expression, time, event, beta, eta = simulate_final_bulk(spec, s.hc_truth, noise, data_seed + 7000)
    potential = bulk_potential(expression, spec)[:, features]
    raw = infer_activity({"HC": cb, "HI": potential}, {name: fit.dictionaries[name] for name in ("HC", "HI")}).numpy()
    wb = raw[:, order]
    wb /= np.maximum(wb.sum(1, keepdims=True), 1e-12)
    correlations = [_correlation(wb[:, k], pi[:, k]) for k in range(6)]
    result = {"Overall_WB": float(np.mean(correlations)), "Risk_WB": correlations[1],
              "Neutral_twin_WB": correlations[2], "Protective_WB": correlations[3],
              "per_niche_WB": correlations, "N1_N2_exposure_correlation": float(np.corrcoef(wb[:, 1], wb[:, 2])[0, 1]),
              "true_N1_N2_exposure_correlation": float(np.corrcoef(pi[:, 1], pi[:, 2])[0, 1])}
    return result, cox_compare(wb, pi, time, event, eta)


def summarize(primary, gate, single_view_ok):
    table = {}
    for method in METHODS:
        rows = [record["methods"][method] for record in primary]
        table[method] = {"Overall_WS": float(np.mean([r["overall_ws"] for r in rows])),
                         "sensitivity": [float(np.mean([r["per_factor"][k]["sensitivity"] for r in rows])) for k in range(6)],
                         "collision_12": sum(r["collision_12"] is True for r in rows),
                         "collision_34": sum(r["collision_34"] is True for r in rows)}
    return {"primary_ST_gate": gate, "single_view_complementarity_gate": single_view_ok,
            "primary_table": table, "final_conclusion": "pending" if gate and single_view_ok else "B"}


def run():
    OUT.mkdir(parents=True, exist_ok=True)
    spec = build_final_spec(load_lr_atlas("data/commuspace_human_lr_atlas.tsv"))
    primary = []
    assets = []
    for rep in range(3):
        record, asset = setting(spec, .5, 100, .05, rep, METHODS, keep_map=rep == 0)
        primary.append(record)
        assets.append(asset)
        print("primary", rep, {m: [round(v["sensitivity"], 2) for v in record["methods"][m]["per_factor"]] for m in METHODS}, flush=True)
    save(OUT / "filter_audit.json", primary[0]["filter"])
    s, blocks, features, ws, _ = assets[0]
    truth = {"generator": "cell-level negative-binomial observed expression; truth fixed before fitting",
             "n_cells": len(s.labels), "coordinates": s.coordinates, "labels": s.labels,
             "HC_background_and_niches": np.vstack((np.full(8, .125), s.hc_truth)),
             "N1_N2_composition_exact_equal": bool(np.array_equal(s.hc_truth[0], s.hc_truth[1])),
             "N3_N4_CCC_molecular_delta_exact_equal": bool(np.array_equal(spec.expression_delta[2], spec.expression_delta[3])),
             "N3_N4_CCC_truth_exact_equal": bool(np.array_equal(spec.hi_truth[2], spec.hi_truth[3])),
             "active_edges": [{"niche": k + 1, "feature": f, "strength": strength, "kind": kind}
                              for k, f, strength, kind in spec.active_edges],
             "beta_ALR": TRUE_BETA.tolist(), "ALR_reference": "Background"}
    save(OUT / "simulation_truth.json", truth)
    save(OUT / "primary_st_results.json", primary)
    save(OUT / "single_view_ablation.json", [{"replicate": r["replicate"],
                                                   "C-only": r["methods"]["C-only"],
                                                   "I-only": r["methods"]["I-only"]} for r in primary])
    balance = [{"seed": 31 + i, **a[4]["C+I"].audit} for i, a in enumerate(assets)]
    for row in balance:
        alpha, gradient = row["alpha"], row["final_gradient_norm"]
        denominator = alpha["HI"] * gradient["HI"]
        row["weighted_final_ratio_C_I"] = [alpha[k] * gradient[k] / max(denominator, 1e-30) for k in ("HC", "HI")]
    save(OUT / "view_balance_audit.json", {"current_primary": balance,
                                           "mean_alpha_C_I": [float(np.mean([r["alpha"][k] for r in balance])) for k in ("HC", "HI")],
                                           "mean_weighted_final_ratio_C_I": np.mean([r["weighted_final_ratio_C_I"] for r in balance], axis=0),
                                           "weight_rule": "geometric mean of two initial relative-loss WS gradients; mean(alpha)=1"})
    gate = primary_pass([r["methods"]["C+I"] for r in primary])
    comp = [r["methods"]["C-only"] for r in primary]
    ccc = [r["methods"]["I-only"] for r in primary]
    single_view_ok = all(min(r["per_factor"][1]["sensitivity"], r["per_factor"][2]["sensitivity"]) < .8 for r in comp) and all(
        min(r["per_factor"][3]["sensitivity"], r["per_factor"][4]["sensitivity"]) < .8 for r in ccc)
    summary = summarize(primary, gate, single_view_ok)
    summary["filtering"] = primary[0]["filter"]
    truth_audit = []
    for rep, (spatial, _, _, _, _) in enumerate(assets):
        pi, _, _, time, event, _, eta = simulate_final_bulk(spec, spatial.hc_truth, .05, primary[rep]["data_seed"] + 7000)
        truth_audit.append({"seed": primary[rep]["data_seed"], "patients": len(pi),
                            "all_positive": bool(np.all(pi > 0)),
                            "max_row_sum_error": float(np.max(np.abs(pi.sum(1) - 1))),
                            "risk_neutral_protective_correlation": np.corrcoef(pi[:, [1, 2, 3]], rowvar=False),
                            "eta_matches_ALR": bool(np.allclose(eta, alr(pi) @ TRUE_BETA, atol=1e-6)),
                            "true_C": concordance_index(time, event, eta), "event_fraction": float(event.mean())})
    save(OUT / "bulk_truth_audit.json", truth_audit)
    if gate and single_view_ok:
        grid = []
        bulk = []
        cox = []
        settings = [(p, size, n) for p in (.3, .5, .7) for size in (50, 100, 150) for n in (0, .05, .10)]
        for p, size, n in settings:
            for rep in range(3):
                is_primary = (p, size, n) == (.5, 100, .05)
                if is_primary:
                    record, asset = primary[rep], assets[rep]
                else:
                    record, asset = setting(spec, p, size, n, rep, ("C+I",))
                transfer, survival = bulk_transfer(spec, asset, record["methods"]["C+I"], n, record["data_seed"])
                grid.append(record)
                if is_primary:
                    bulk.append(transfer)
                    cox.append(survival)
            print("grid", p, size, n, flush=True)
        save(OUT / "robustness_grid.json", grid)
        save(OUT / "bulk_transfer.json", bulk)
        save(OUT / "cox_raw_vs_alr.json", cox)
        save(OUT / "cox_collinearity_audit.json", [row["ALR"]["collinearity"] | {"Raw-W": row["Raw-W"]["collinearity"]} for row in cox])
        summary["bulk"] = bulk
        summary["cox"] = cox
        bulk_ok = all(r["Risk_WB"] > .5 and r["Protective_WB"] > .5 for r in bulk)
        cox_ok = all(r["ALR"]["gamma"][0] > 0 and r["ALR"]["gamma"][2] < 0 for r in cox)
        summary["final_conclusion"] = "A" if bulk_ok and cox_ok else "C" if not bulk_ok else "D"
    else:
        reason = "Primary balanced joint ST failed predeclared gate; downstream inference and robustness were gated off."
        for name in ("robustness_grid.json", "bulk_transfer.json", "cox_raw_vs_alr.json", "cox_collinearity_audit.json"):
            save(OUT / name, {"status": "not_run", "reason": reason})
        summary["bulk"] = {"status": "not_run", "reason": reason}
        summary["cox"] = {"status": "not_run", "reason": reason}
    np.savez_compressed(OUT / "primary_figure_data.npz", coordinates=s.coordinates, labels=s.labels,
                        ws_truth=ws, hc=truth["HC_background_and_niches"],
                        hi1=blocks["HI"][s.labels == 1].mean(0), hi2=blocks["HI"][s.labels == 2].mean(0),
                        hi3=blocks["HI"][s.labels == 3].mean(0), hi4=blocks["HI"][s.labels == 4].mean(0),
                        **{m.replace("-", "_") + "_pred": primary[0]["methods"][m]["predicted"] for m in METHODS},
                        **{m.replace("-", "_") + "_activity": primary[0]["methods"][m]["activity"] for m in METHODS})
    save(OUT / "summary.json", summary)
    return summary


if __name__ == "__main__":
    run()
