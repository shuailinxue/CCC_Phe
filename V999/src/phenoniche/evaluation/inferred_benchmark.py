from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, replace
import hashlib
import json
import logging
import multiprocessing
from pathlib import Path
import numpy as np
import torch
from phenoniche.data.challenging_synthetic import ChallengingConfig, generate_challenging
from phenoniche.data.scaling import BlockScaler
from phenoniche.evaluation.challenging_benchmark import fit_candidate, score_candidate, aggregate_records, save_json, _stats
from phenoniche.evaluation.patient_protocol import split_patients, subset_bulk, select_positive_lambda
from phenoniche.evaluation.inference_diagnostics import scan_inner_inference, final_inference_audit
from phenoniche.evaluation.pair_diagnostics import pair_separation, truth_activity_correlations
from phenoniche.training.config import TrainingConfig, LossWeights


def prepare_scenario(scenario):
    synthetic = generate_challenging(ChallengingConfig(scenario=scenario))
    split = split_patients(synthetic.data.CB.shape[0])
    raw_train = subset_bulk(synthetic.data, split.train)
    scaler = BlockScaler.fit(raw_train)
    training = scaler.transform(raw_train)
    validation = scaler.transform(subset_bulk(synthetic.data, split.validation))
    truth = {name: value.clone() for name, value in synthetic.truth.items()}
    truth["HC"] /= scaler.composition
    truth["HI"] /= scaler.communication
    truth["HO"] /= scaler.topology
    return synthetic, split, scaler, training, validation, truth


def diagnostic_weights(data, phenotype, ccc_only=False):
    weights = LossWeights.per_entry(data, ph=phenotype, reg=0.0001)
    return replace(weights, bc=0.0, sc=0.0, so=0.0) if ccc_only else weights


def _fit_job(training, validation, config, weights):
    torch.set_num_threads(1)
    return fit_candidate(training, validation, config, weights)


def _fit_grid(executor, training, validation, tasks, inner, device):
    futures = {}
    for k, weight, seed, ccc_only in tasks:
        config = TrainingConfig(number_of_niches=k, seed=seed, device=device, inner_steps=inner["steps"], inner_lr=inner["learning_rate"])
        weights = diagnostic_weights(training, weight, ccc_only)
        future = executor.submit(_fit_job, training, validation, config, weights)
        futures[future] = (k, weight, seed, ccc_only)
    candidates = []
    for future in as_completed(futures):
        task = futures[future]
        candidates.append((task, future.result()))
        logging.warning("Completed K=%s lambda=%s seed=%s ccc_only=%s (%s/%s)", *task, len(candidates), len(tasks))
    return sorted(candidates, key=lambda item: item[0])


def _score_grid(candidates, prepared):
    synthetic, split, scaler, training, validation, truth = prepared
    test = scaler.transform(subset_bulk(synthetic.data, split.test))
    records = []
    for (k, weight, seed, ccc_only), candidate in candidates:
        weights = diagnostic_weights(training, weight, ccc_only)
        record = score_candidate(candidate, training, validation, test, truth, synthetic.metadata["roles"], split, weights)
        record.update({"K_fit": k, "ccc_only": ccc_only,
                       "pair": pair_separation(candidate.factors, truth, ccc_only),
                       "inner_audit": final_inference_audit(candidate, training, weights)})
        if ccc_only:
            record["recovery"] = None
            record["train_reconstruction"]["bulk_reconstruction"] = record["train_reconstruction"]["block_mse"]["IB"]
            record["train_reconstruction"]["spatial_reconstruction"] = record["train_reconstruction"]["block_mse"]["IS"]
        records.append(record)
    return records


def summarize_pairs(records):
    summary = {key: _stats([row[key] for row in records]) for key in ("train_c_index", "validation_c_index", "test_c_index")}
    summary["collision_rate"] = float(np.mean([row["pair"]["collision"] for row in records]))
    summary["HI_only_collision_rate"] = float(np.mean([row["pair"]["HI_only_collision"] for row in records]))
    for key in records[0]["pair"]:
        if key.endswith(("cosine", "correlation", "recovery", "gamma")):
            values = [row["pair"][key] for row in records if row["pair"][key] is not None]
            summary[key] = _stats(values) if values else None
    summary["risk_gamma_sign_consistency"] = float(np.mean([row["pair"]["A_gamma"] > 0 for row in records]))
    return summary


def verify_frozen_data(prepared, old_output):
    synthetic, split, _, _, _, _ = prepared
    with np.load(old_output / "ground_truth.npz") as previous:
        for name, value in synthetic.truth.items():
            if not np.array_equal(previous[name], value.numpy()):
                raise ValueError(f"Frozen ground truth changed: {name}")
    previous_split = json.loads((old_output / "splits.json").read_text())
    if any(previous_split[name] != getattr(split, name).tolist() for name in ("train", "validation", "test")):
        raise ValueError("Frozen patient split changed")
    return {"ground_truth_equal": True, "split_equal": True,
            "feature_sha256": {name: hashlib.sha256(value.numpy().tobytes()).hexdigest() for name, value in synthetic.data.blocks().items()}}


def run_inferred_benchmark(output, previous_output, workers=4, device="cpu"):
    output, previous_output = Path(output), Path(previous_output)
    output.mkdir(parents=True, exist_ok=True)
    capacity = prepare_scenario("capacity")
    same = prepare_scenario("same_composition")
    audits = {"capacity": verify_frozen_data(capacity, previous_output / "capacity"),
              "same_composition": verify_frozen_data(same, previous_output / "same_composition")}
    save_json(output / "frozen_data_audit.json", audits)
    numerical = scan_inner_inference(capacity[3], diagnostic_weights(capacity[3], 0))
    save_json(output / "inner_inference_diagnostics.json", numerical)
    inner = numerical["selected"]
    seeds, lambdas = (31, 32, 33, 34, 35), (0.0, 0.001, 0.005, 0.01, 0.05, 0.1)
    protocol = {"model": "inferred WB via unrolled accelerated projected gradient, same train/validation/test path",
                "seeds": seeds, "capacity_lambdas": lambdas, "diagnostic_positive_lambda": 0.1,
                "diagnostic_lambda_source": "Fixed from previous same_composition validation selection, identical across all K and modes",
                "inner": inner, "warmup_epochs": 800, "joint_epochs": 800, "workers": workers, "device": device,
                "regularization": 0.0001, "new_regularizers": [], "scaling": {"capacity": asdict(capacity[2]), "same_composition": asdict(same[2])},
                "ccc_only": "bc=sc=so=0; HC/HO excluded from existing L2; inference bi only; pair matching HI only",
                "old_results": str(previous_output), "nnls": "Numerical reference only, never used for training or held-out prediction"}
    save_json(output / "protocol.json", protocol)
    save_json(output / "same_composition_truth_diagnostics.json", truth_activity_correlations(same[0].truth, same[0].data.time, same[0].data.event))
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn")) as executor:
        logging.warning("Starting capacity grid: 30 fits")
        tasks = [(4, weight, seed, False) for weight in lambdas for seed in seeds]
        candidates = _fit_grid(executor, capacity[3], capacity[4], tasks, inner, device)
        selected = select_positive_lambda({weight: [candidate.validation_c_index for task, candidate in candidates if task[1] == weight] for weight in lambdas})
        save_json(output / "capacity_selection.json", {"lambda": selected, "test_labels_used": False})
        records = _score_grid(candidates, capacity)
        aggregates = {weight: aggregate_records([row for row in records if row["lambda_ph"] == weight]) for weight in lambdas}
        save_json(output / "capacity_lambda_sweep.json", {"aggregates": aggregates, "runs": records})
        summary = {"selected_lambda": selected, "new": {"unsupervised": aggregates[0.0], "supervised": aggregates[selected]},
                   "old": json.loads((previous_output / "capacity/summary.json").read_text())["comparison"]}
        for values in summary["old"].values():
            values.pop("train_fitted_c_index", None)
        save_json(output / "capacity_summary.json", summary)
        logging.warning("Capacity complete; selected lambda=%s. Starting full-model K diagnostic: 30 fits", selected)
        tasks = [(k, weight, seed, False) for k in (4, 6, 8) for weight in (0.0, 0.1) for seed in seeds]
        full = _fit_grid(executor, same[3], same[4], tasks, inner, device)
        records = _score_grid(full, same)
        groups = {f"K{k}_lambda{weight}": summarize_pairs([row for row in records if row["K_fit"] == k and row["lambda_ph"] == weight])
                  for k in (4, 6, 8) for weight in (0.0, 0.1)}
        save_json(output / "same_composition_k_sweep.json", {"aggregates": groups, "runs": records})
        logging.warning("Starting CCC-only diagnostic: 20 fits")
        tasks = [(k, weight, seed, True) for k in (4, 6) for weight in (0.0, 0.1) for seed in seeds]
        ccc = _fit_grid(executor, same[3], same[4], tasks, inner, device)
        records = _score_grid(ccc, same)
        groups = {f"K{k}_lambda{weight}": summarize_pairs([row for row in records if row["K_fit"] == k and row["lambda_ph"] == weight])
                  for k in (4, 6) for weight in (0.0, 0.1)}
        save_json(output / "same_composition_ccc_only.json", {"aggregates": groups, "runs": records})
    logging.warning("All 80 fits and numerical audits completed")
