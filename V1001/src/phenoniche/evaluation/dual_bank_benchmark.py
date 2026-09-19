from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
import hashlib
import json
import logging
import multiprocessing
from pathlib import Path
import numpy as np
import torch
from phenoniche.evaluation.bank_matching import bank_matching_report
from phenoniche.evaluation.challenging_benchmark import _c_index, _stats, save_json
from phenoniche.evaluation.inferred_benchmark import prepare_scenario, verify_frozen_data
from phenoniche.evaluation.patient_protocol import fit_frozen_cox, permute_training_phenotype, subset_bulk
from phenoniche.training.config import LossWeights
from phenoniche.training.dual_bank_trainer import DualBankTrainingConfig, DualBankTrainingResult, build_dual_bank_model, train_background, train_joint_scratch, train_residual_from_background


def _weights(data):
    return LossWeights.per_entry(data, ph=0.1, reg=0.0001, lambda_collapse=0.0)


def _configuration(scenario, seed, device="cpu"):
    return DualBankTrainingConfig(
        phenotype_niches=2, background_niches=4 if scenario == "same_composition" else 2,
        background_epochs=800, phenotype_epochs=800, jointft_epochs=200,
        scratch_epochs=1600, seed=seed, device=device, inner_steps=100, inner_lr=1.0)


def _cpu_state(model):
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def _background_job(data, config, weights):
    torch.set_num_threads(1)
    result = train_background(data, config, weights)
    return _cpu_state(result.model), result.history, result.diagnostics


def _residual_job(data, config, weights, state, jointft):
    torch.set_num_threads(1)
    frozen, fine_tuned = train_residual_from_background(data, config, weights, state, jointft=jointft)
    return frozen, fine_tuned


def _scratch_job(data, config, weights):
    torch.set_num_threads(1)
    return train_joint_scratch(data, config, weights)


def _background_hash(state):
    digest = hashlib.sha256()
    for name in sorted(state):
        if name.startswith("raw_") and name.endswith(("0",)):
            digest.update(name.encode())
            digest.update(state[name].detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def _labels(scenario, roles):
    if scenario == "same_composition":
        return ["A", "B", "nuisance_1", "nuisance_2", "neutral_1", "neutral_2"]
    counts = {}
    labels = []
    for role in roles:
        counts[role] = counts.get(role, 0) + 1
        labels.append(role if roles.count(role) == 1 else f"{role}_{counts[role]}")
    return labels


def _state_payload(result, config, seed, mode, background_history):
    return {"state_dict": _cpu_state(result.model), "config": asdict(config), "seed": seed,
            "mode": mode, "stage_history": background_history + result.history,
            "training_diagnostics": result.diagnostics}


def _load_result(path, data, config, weights):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model = build_dual_bank_model(data, config, weights)
    model.load_state_dict(payload["state_dict"])
    history = payload["stage_history"]
    if payload["mode"].startswith("residual"):
        history = history[config.background_epochs:]
    if payload["mode"] == "joint_scratch" and len(history) > config.scratch_epochs:
        history = history[-config.scratch_epochs:]
    return DualBankTrainingResult(model, history, payload["training_diagnostics"])


def _reconstruction(model, data):
    with torch.no_grad():
        output = model(data.CB, data.IB)
    result = {"blocks": {}, "conservation": {}}
    for name in ("CB", "IB", "CS", "OS", "IS"):
        background = output[f"{name}0"].cpu()
        phenotype = output[f"{name}p"].cpu()
        total = background + phenotype
        observed = getattr(data, name).cpu()
        result["blocks"][name] = {
            "background_mse": float((observed - background).square().mean()),
            "phenotype_contribution_mean_square": float(phenotype.square().mean()),
            "total_mse": float((observed - total).square().mean()),
        }
        result["conservation"][name] = {
            "shape": list(total.shape), "finite": bool(torch.isfinite(total).all()),
            "max_absolute_sum_error": float((total - (background + phenotype)).abs().max()),
        }
    result["bulk_reconstruction"] = result["blocks"]["CB"]["total_mse"] + result["blocks"]["IB"]["total_mse"]
    result["spatial_reconstruction"] = sum(result["blocks"][name]["total_mse"] for name in ("CS", "OS", "IS"))
    return result


def _score(result, prepared, scenario, mode, seed, state_path, background_hash, background_history):
    synthetic, split, scaler, training, validation, truth = prepared
    test = scaler.transform(subset_bulk(synthetic.data, split.test))
    model = result.model.cpu()
    with torch.no_grad():
        train_factors = {name: value.cpu() for name, value in model.factors(training.CB, training.IB, create_graph=False).items()}
        validation_factors = model.factors(validation.CB, validation.IB, create_graph=False)
        test_factors = model.factors(test.CB, test.IB, create_graph=False)
    labels = _labels(scenario, synthetic.metadata["roles"])
    matching = bank_matching_report(train_factors, truth, synthetic.metadata["roles"],
                                    train_factors["WB0"], train_factors["WBp"],
                                    truth["WB"][split.train], labels)
    beta0 = fit_frozen_cox(train_factors["WB0"], training.time, training.event, ridge=0.01)
    probes = {
        "background_beta": beta0.tolist(),
        "background_validation_c_index": _c_index(validation_factors["WB0"], beta0, validation.time, validation.event),
        "background_test_c_index": _c_index(test_factors["WB0"], beta0, test.time, test.event),
        "phenotype_validation_c_index": _c_index(validation_factors["WBp"], train_factors["gamma"], validation.time, validation.event),
        "phenotype_test_c_index": _c_index(test_factors["WBp"], train_factors["gamma"], test.time, test.event),
        "probe_used_for_training_or_selection": False,
    }
    return {
        "scenario": scenario, "mode": mode, "seed": seed, "lambda_ph": 0.1,
        "lambda_collapse": 0.0, "Kp": model.phenotype_niches,
        "Kb": model.background_niches, "K_total": model.phenotype_niches + model.background_niches,
        "train_c_index": _c_index(train_factors["WBp"], train_factors["gamma"], training.time, training.event),
        "validation_c_index": probes["phenotype_validation_c_index"],
        "test_c_index": probes["phenotype_test_c_index"],
        "matching": matching, "probes": probes,
        "reconstruction": _reconstruction(model, training),
        "training_diagnostics": result.diagnostics,
        "background_state_sha256": background_hash,
        "background_history_epochs": len(background_history),
        "state_path": str(state_path),
    }


def _aggregate(records, scenario):
    result = {key: _stats([row[key] for row in records]) for key in
              ("train_c_index", "validation_c_index", "test_c_index")}
    result["background_test_c_index"] = _stats([row["probes"]["background_test_c_index"] for row in records])
    result["phenotype_test_c_index"] = _stats([row["probes"]["phenotype_test_c_index"] for row in records])
    result["bulk_reconstruction"] = _stats([row["reconstruction"]["bulk_reconstruction"] for row in records])
    result["spatial_reconstruction"] = _stats([row["reconstruction"]["spatial_reconstruction"] for row in records])
    labels = _labels(scenario, [entry["role"] for entry in records[0]["matching"]["per_true_niche"]])
    for label in labels:
        matches = [row["matching"]["by_label"][label] for row in records]
        result[label] = {
            "phenotype_bank_rate": float(np.mean([match["best_bank"] == "phenotype" for match in matches])),
            "background_bank_rate": float(np.mean([match["best_bank"] == "background" for match in matches])),
            "best_bank_counts": {bank: sum(match["best_bank"] == bank for match in matches) for bank in ("phenotype", "background")},
            **{key: _stats([match[key] for match in matches]) for key in
               ("bank_margin", "HC_cosine", "HI_cosine", "HO_cosine", "WB_correlation", "WS_correlation",
                "best_phenotype_similarity", "best_background_similarity",
                "phenotype_HC_cosine", "phenotype_HI_cosine", "phenotype_HO_cosine",
                "phenotype_WB_correlation", "phenotype_WS_correlation", "phenotype_gamma",
                "background_HC_cosine", "background_HI_cosine", "background_HO_cosine",
                "background_WB_correlation", "background_WS_correlation")},
        }
    if scenario == "same_composition":
        result["A_B_different_bank_rate"] = float(np.mean([row["matching"]["A_B_different_bank"] for row in records]))
        result["A_B_different_factor_rate"] = float(np.mean([row["matching"]["A_B_different_factor"] for row in records]))
        result["risk_gamma_sign_consistency"] = float(np.mean([
            row["matching"]["by_label"]["A"]["phenotype_gamma_sign_correct"] is True for row in records]))
    else:
        for role in ("risk", "protective"):
            matches = [next(entry for entry in row["matching"]["per_true_niche"] if entry["role"] == role) for row in records]
            result[f"{role}_phenotype_bank_rate"] = float(np.mean([entry["best_bank"] == "phenotype" for entry in matches]))
            result[f"{role}_gamma_sign_consistency"] = float(np.mean([entry["phenotype_gamma_sign_correct"] is True for entry in matches]))
        nuisance = [[entry for entry in row["matching"]["per_true_niche"] if entry["role"] == "nuisance"] for row in records]
        result["nuisance_background_bank_rate"] = float(np.mean([entry["best_bank"] == "background" for group in nuisance for entry in group]))
    result["gradient_clip_events"] = sum(row["training_diagnostics"]["gradient_clip_count"] for row in records)
    result["max_gradient_norm"] = max(row["training_diagnostics"]["max_gradient_norm"] for row in records)
    return result


def _single_bank_baseline(root):
    same = json.loads((root / "outputs/anti_collapse/same_composition_sweep.json").read_text())["aggregates"]["0.0"]
    capacity = json.loads((root / "outputs/anti_collapse/capacity_sweep.json").read_text())["aggregates"]["0.0"]
    return {"same_composition": same, "capacity": capacity,
            "source": "Audited inferred-WB single-bank lambda_ph=0.1, lambda_collapse=0, seeds 31..35"}


def run_dual_bank_benchmark(output, root, workers=4, device="cpu"):
    output, root = Path(output), Path(root)
    output.mkdir(parents=True, exist_ok=True)
    states = output / "states"
    states.mkdir(exist_ok=True)
    scenarios = {name: prepare_scenario(name) for name in ("same_composition", "capacity")}
    audits = {name: verify_frozen_data(prepared, root / "outputs/challenging_synthetic" / name)
              for name, prepared in scenarios.items()}
    save_json(output / "frozen_data_audit.json", audits)
    protocol = {
        "seeds": [31, 32, 33, 34, 35], "permutation_seeds": [101, 102, 103, 104, 105],
        "allocations": {"same_composition": {"Kp": 2, "Kb": 4}, "capacity": {"Kp": 2, "Kb": 2}},
        "epochs": {"background": 800, "phenotype_residual": 800, "jointft": 200, "joint_scratch": 1600},
        "lambda_ph": 0.1, "lambda_collapse": 0.0, "regularization": 0.0001,
        "inner": {"steps": 100, "learning_rate": 1.0},
        "jointft_learning_rate_factor": 0.1,
        "background_probe": "Training-only Cox ridge=0.01; evaluation only",
        "single_bank_source": "outputs/anti_collapse lambda_collapse=0 records",
        "permutation_background": "Exact same_composition seed31 post-Stage-1 state reused for all five permutations",
        "selection": "No hyperparameter selection, no K allocation scan, no test-guided changes",
    }
    save_json(output / "protocol.json", protocol)
    seeds = (31, 32, 33, 34, 35)
    background = {}
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=workers, mp_context=context) as executor:
        futures = {}
        for scenario, prepared in scenarios.items():
            for seed in seeds:
                config = _configuration(scenario, seed, device)
                futures[executor.submit(_background_job, prepared[3], config, _weights(prepared[3]))] = (scenario, seed)
        for future in as_completed(futures):
            scenario, seed = futures[future]
            background[(scenario, seed)] = future.result()
            logging.warning("Background complete %s seed=%s (%s/10)", scenario, seed, len(background))
        jobs = {}
        completed = []
        for scenario, prepared in scenarios.items():
            for seed in seeds:
                config = _configuration(scenario, seed, device)
                state = background[(scenario, seed)][0]
                frozen_path = states / f"{scenario}_residual_frozen_seed{seed}.pt"
                jointft_path = states / f"{scenario}_residual_jointft_seed{seed}.pt"
                if frozen_path.exists() and jointft_path.exists():
                    completed.append(((scenario, "residual_frozen_loaded", seed, None),
                                      _load_result(frozen_path, prepared[3], config, _weights(prepared[3]))))
                    completed.append(((scenario, "residual_jointft_loaded", seed, None),
                                      _load_result(jointft_path, prepared[3], config, _weights(prepared[3]))))
                else:
                    jobs[executor.submit(_residual_job, prepared[3], config, _weights(prepared[3]), state, True)] = (scenario, "residual", seed, None)
        for seed in seeds:
            prepared = scenarios["same_composition"]
            config = _configuration("same_composition", seed, device)
            scratch_path = states / f"same_composition_joint_scratch_seed{seed}.pt"
            if scratch_path.exists():
                completed.append((("same_composition", "joint_scratch_loaded", seed, None),
                                  _load_result(scratch_path, prepared[3], config, _weights(prepared[3]))))
            else:
                jobs[executor.submit(_scratch_job, prepared[3], config, _weights(prepared[3]))] = ("same_composition", "joint_scratch", seed, None)
        same = scenarios["same_composition"]
        state31 = background[("same_composition", 31)][0]
        for permutation_seed in (101, 102, 103, 104, 105):
            config = _configuration("same_composition", 31, device)
            permuted = permute_training_phenotype(same[3], permutation_seed)
            permutation_path = states / f"same_composition_permutation_perm{permutation_seed}.pt"
            if permutation_path.exists():
                completed.append((("same_composition", "permutation_loaded", 31, permutation_seed),
                                  _load_result(permutation_path, permuted, config, _weights(same[3]))))
            else:
                jobs[executor.submit(_residual_job, permuted, config, _weights(same[3]), state31, False)] = ("same_composition", "permutation", 31, permutation_seed)
        for future in as_completed(jobs):
            key = jobs[future]
            completed.append((key, future.result()))
            logging.warning("Dual-bank job complete %s (%s/%s)", key, len(completed), len(jobs))
    records = {"same_composition": {"residual_frozen": [], "residual_jointft": [], "joint_scratch": []},
               "capacity": {"residual_frozen": [], "residual_jointft": []}}
    permutation_records = []
    matching_diagnostics = []
    for (scenario, kind, seed, permutation_seed), result in completed:
        prepared = scenarios[scenario]
        bg_state, bg_history, _ = background[(scenario, seed)]
        bg_hash = _background_hash(bg_state)
        if kind == "residual":
            candidates = [("residual_frozen", result[0]), ("residual_jointft", result[1])]
        elif kind == "permutation":
            candidates = [("permutation", result[0])]
        elif kind.endswith("_loaded"):
            candidates = [(kind.removesuffix("_loaded"), result)]
        else:
            candidates = [(kind, result)]
        for mode, training_result in candidates:
            if training_result is None:
                continue
            suffix = f"perm{permutation_seed}" if permutation_seed is not None else f"seed{seed}"
            state_path = states / f"{scenario}_{mode}_{suffix}.pt"
            history_prefix = [] if mode == "joint_scratch" else bg_history
            if not kind.endswith("_loaded"):
                torch.save(_state_payload(training_result, _configuration(scenario, seed, device), seed, mode, history_prefix), state_path)
            elif mode == "joint_scratch":
                torch.save(_state_payload(training_result, _configuration(scenario, seed, device), seed, mode, []), state_path)
            score_data = prepared
            if kind.startswith("permutation"):
                permuted_training = permute_training_phenotype(prepared[3], permutation_seed)
                score_data = (prepared[0], prepared[1], prepared[2], permuted_training, prepared[4], prepared[5])
            record = _score(training_result, score_data, scenario, mode, seed, state_path,
                            None if mode == "joint_scratch" else bg_hash, history_prefix)
            if kind.startswith("permutation"):
                record["permutation_seed"] = permutation_seed
                original_test = prepared[2].transform(subset_bulk(prepared[0].data, prepared[1].test))
                with torch.no_grad():
                    test_factors = training_result.model.cpu().factors(original_test.CB, original_test.IB, create_graph=False)
                record["test_c_index"] = _c_index(test_factors["WBp"], test_factors["gamma"], original_test.time, original_test.event)
                record["probes"]["phenotype_test_c_index"] = record["test_c_index"]
                permutation_records.append(record)
            else:
                records[scenario][mode].append(record)
            matching_diagnostics.append({"scenario": scenario, "mode": mode, "seed": seed,
                                         "permutation_seed": permutation_seed, "matching": record["matching"]})
    for scenario in records:
        for mode in records[scenario]:
            records[scenario][mode].sort(key=lambda row: row["seed"])
    permutation_records.sort(key=lambda row: row["permutation_seed"])
    for mode in ("residual_frozen", "residual_jointft", "joint_scratch"):
        save_json(output / f"same_composition_{mode}.json",
                  {"aggregate": _aggregate(records["same_composition"][mode], "same_composition"),
                   "runs": records["same_composition"][mode]})
    for mode in ("residual_frozen", "residual_jointft"):
        save_json(output / f"capacity_{mode}.json",
                  {"aggregate": _aggregate(records["capacity"][mode], "capacity"),
                   "runs": records["capacity"][mode]})
    permutation = {"aggregate": _aggregate(permutation_records, "same_composition"),
                   "runs": permutation_records,
                   "background_state_sha256": _background_hash(background[("same_composition", 31)][0]),
                   "unique_background_hashes": sorted(set(row["background_state_sha256"] for row in permutation_records))}
    save_json(output / "same_composition_permutation.json", permutation)
    save_json(output / "bank_matching_diagnostics.json", matching_diagnostics)
    probes = {scenario: {mode: {"aggregate": {
        "background_validation_c_index": _stats([row["probes"]["background_validation_c_index"] for row in rows]),
        "background_test_c_index": _stats([row["probes"]["background_test_c_index"] for row in rows]),
        "phenotype_validation_c_index": _stats([row["probes"]["phenotype_validation_c_index"] for row in rows]),
        "phenotype_test_c_index": _stats([row["probes"]["phenotype_test_c_index"] for row in rows])},
        "runs": [{"seed": row["seed"], **row["probes"]} for row in rows]}
        for mode, rows in modes.items()} for scenario, modes in records.items()}
    probes["same_composition_permutation"] = {"runs": [{"permutation_seed": row["permutation_seed"], **row["probes"]} for row in permutation_records]}
    save_json(output / "background_probe.json", probes)
    baseline = _single_bank_baseline(root)
    summary = {"single_bank": baseline,
               "same_composition": {mode: _aggregate(rows, "same_composition") for mode, rows in records["same_composition"].items()},
               "capacity": {mode: _aggregate(rows, "capacity") for mode, rows in records["capacity"].items()},
               "permutation": permutation["aggregate"], "fit_count": 30,
               "background_warmup_count": 10, "generator_and_split_unchanged": True,
               "all_lambda_collapse_zero": all(row["lambda_collapse"] == 0 for modes in records.values() for rows in modes.values() for row in rows) and all(row["lambda_collapse"] == 0 for row in permutation_records),
               "permutation_background_reused_exactly": len(permutation["unique_background_hashes"]) == 1,
               "frozen_data_audit": audits}
    save_json(output / "summary.json", summary)
    logging.warning("Dual-bank benchmark complete: 30 final models from 10 reusable background warmups")
    return summary
