from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, replace, asdict
import json
import logging
import multiprocessing
from pathlib import Path
import numpy as np
import torch
from phenoniche.evaluation.challenging_benchmark import fit_candidate, score_candidate, aggregate_records, save_json, _stats, _c_index
from phenoniche.evaluation.inferred_benchmark import prepare_scenario, verify_frozen_data
from phenoniche.evaluation.patient_protocol import subset_bulk, permute_training_phenotype
from phenoniche.evaluation.collapse_diagnostics import redundancy_diagnostics, detailed_pair_recovery, select_collapse_lambda
from phenoniche.evaluation.inference_diagnostics import final_inference_audit
from phenoniche.training.config import TrainingConfig, LossWeights


@dataclass(frozen=True)
class CollapseBenchmarkConfig:
    lambdas: tuple[float, ...] = (0.0, 0.0001, 0.001, 0.005, 0.01, 0.05, 0.1)
    seeds: tuple[int, ...] = (31, 32, 33, 34, 35)
    permutation_seeds: tuple[int, ...] = (101, 102, 103, 104, 105)
    warmup_epochs: int = 800
    joint_epochs: int = 800
    workers: int = 4
    device: str = "cpu"

    def __post_init__(self):
        if not self.lambdas or 0 not in self.lambdas or len(set(self.lambdas)) != len(self.lambdas):
            raise ValueError("Unique collapse grid must include zero")
        if not all(np.isfinite(value) and value >= 0 for value in self.lambdas):
            raise ValueError("Collapse weights must be finite and nonnegative")
        if not self.seeds or len(set(self.seeds)) != len(self.seeds):
            raise ValueError("Model seeds must be nonempty and unique")
        if not self.permutation_seeds or len(set(self.permutation_seeds)) != len(self.permutation_seeds):
            raise ValueError("Permutation seeds must be nonempty and unique")
        if not isinstance(self.workers, int) or self.workers < 1:
            raise ValueError("workers must be a positive integer")
        TrainingConfig(warmup_epochs=self.warmup_epochs, joint_epochs=self.joint_epochs)


def _config(k, seed, config):
    return TrainingConfig(number_of_niches=k, seed=seed, device=config.device, warmup_epochs=config.warmup_epochs,
                          joint_epochs=config.joint_epochs, inner_steps=100, inner_lr=1.0)


def _weights(training, weight):
    return LossWeights.per_entry(training, ph=0.1, reg=0.0001, lambda_collapse=weight)


def _fit(training, validation, training_config, weights):
    torch.set_num_threads(1)
    return fit_candidate(training, validation, training_config, weights)


def _collect_fits(executor, training, validation, k, config):
    futures = {executor.submit(_fit, training, validation, _config(k, seed, config), _weights(training, weight)): (weight, seed)
               for weight in config.lambdas for seed in config.seeds}
    candidates, failures = [], []
    for future in as_completed(futures):
        weight, seed = futures[future]
        try:
            candidate = future.result()
        except FloatingPointError as error:
            failures.append({"lambda_collapse": weight, "seed": seed, "error": str(error), "type": type(error).__name__})
            logging.error("Numerical failure K=%s collapse=%s seed=%s: %s", k, weight, seed, error)
            continue
        candidates.append((weight, candidate))
        logging.warning("Completed K=%s collapse=%s seed=%s (%s/%s)", k, weight, seed, len(candidates) + len(failures), len(futures))
    return sorted(candidates, key=lambda item: (item[0], item[1].seed)), failures


def _select(candidates, config):
    validation, collapse = {}, {}
    for weight in config.lambdas:
        matching = [candidate for value, candidate in candidates if value == weight]
        if len(matching) != len(config.seeds):
            continue
        validation[weight] = float(np.mean([candidate.validation_c_index for candidate in matching]))
        collapse[weight] = float(np.mean([redundancy_diagnostics(candidate.factors)["summary"]["mean_pair_collapse"] for candidate in matching]))
    selection = select_collapse_lambda(validation, collapse)
    selection["excluded_incomplete_lambdas"] = [weight for weight in config.lambdas if weight not in validation]
    return selection


def _score(weight, candidate, prepared):
    synthetic, split, scaler, training, validation, truth = prepared
    test = scaler.transform(subset_bulk(synthetic.data, split.test))
    weights = _weights(training, weight)
    record = score_candidate(candidate, training, validation, test, truth, synthetic.metadata["roles"], split, weights)
    pair = detailed_pair_recovery(candidate, truth, split.train)
    record.update({"lambda_collapse": weight, "pair": pair,
                   "redundancy": redundancy_diagnostics(candidate.factors, (pair["A_best_factor"], pair["B_best_factor"])),
                   "training_diagnostics": candidate.training_diagnostics,
                   "inner_audit": final_inference_audit(candidate, training, weights)})
    return record


def aggregate_collapse(records):
    result = aggregate_records(records)
    result["collision_rate"] = float(np.mean([row["pair"]["collision"] for row in records]))
    result["HI_only_collision_rate"] = float(np.mean([row["pair"]["HI_only_collision"] for row in records]))
    result["pair"] = {}
    for key in records[0]["pair"]:
        if key.endswith(("cosine", "correlation", "recovery", "gamma")):
            result["pair"][key] = _stats([row["pair"][key] for row in records])
    result["redundancy"] = {key: _stats([row["redundancy"]["summary"][key] for row in records]) for key in records[0]["redundancy"]["summary"]}
    result["matched_pair"] = {key: _stats([row["redundancy"]["matched_pair"][key] for row in records]) for key in records[0]["redundancy"]["matched_pair"]}
    result["final_losses"] = {key: _stats([row["training_diagnostics"]["final_losses"][key] for row in records]) for key in records[0]["training_diagnostics"]["final_losses"]}
    result["stability"] = {"total_clip_events": sum(row["training_diagnostics"]["gradient_clip_count"] for row in records),
                           "clipped_runs": sum(row["training_diagnostics"]["gradient_clip_count"] > 0 for row in records),
                           "max_gradient_norm": max(row["training_diagnostics"]["max_gradient_norm"] for row in records),
                           "nonfinite_runs": sum(row["training_diagnostics"]["nonfinite_loss"] or row["training_diagnostics"]["nonfinite_gradient"] for row in records)}
    return result


def _baseline_audit(scenario, records, previous_output):
    name = "same_composition_k_sweep.json" if scenario == "same_composition" else "capacity_lambda_sweep.json"
    previous = json.loads((previous_output / name).read_text())["runs"]
    k = 6 if scenario == "same_composition" else 4
    old = {row["seed"]: row for row in previous if row["lambda_ph"] == 0.1 and row["K_fit"] == k}
    comparisons = []
    for row in records:
        if row["lambda_collapse"] != 0:
            continue
        before = old[row["seed"]]
        differences = {key: abs(row[key] - before[key]) for key in ("train_c_index", "validation_c_index", "test_c_index")}
        differences["overall_dictionary_recovery"] = abs(row["recovery"]["overall_dictionary_recovery"] - before["recovery"]["overall_dictionary_recovery"])
        for key in ("bulk_reconstruction", "spatial_reconstruction"):
            differences[key] = abs(row["train_reconstruction"][key] - before["train_reconstruction"][key])
        comparisons.append({"seed": row["seed"], "absolute_differences": differences})
    maximum = max(value for comparison in comparisons for value in comparison["absolute_differences"].values())
    return {"runs": comparisons, "max_absolute_difference": maximum, "within_1e_minus7": maximum <= 1e-7}


def run_collapse_benchmark(output, previous_output, frozen_output, config=None):
    config = config or CollapseBenchmarkConfig()
    output, previous_output, frozen_output = Path(output), Path(previous_output), Path(frozen_output)
    output.mkdir(parents=True, exist_ok=True)
    prepared = {name: prepare_scenario(name) for name in ("same_composition", "capacity")}
    audits = {name: verify_frozen_data(values, frozen_output / name) for name, values in prepared.items()}
    save_json(output / "frozen_data_audit.json", audits)
    protocol = {"config": asdict(config), "K": {"same_composition": 6, "capacity": 4}, "lambda_ph": 0.1,
                "inner_steps": 100, "inner_lr": 1.0, "regularization": 0.0001,
                "collapse_definition": "mean over k<l of cosine(HC_k,HC_l)*cosine(HO_k,HO_l)*cosine(HI_k,HI_l), eps=1e-8",
                "collapse_schedule": "Same fixed lambda in warmup and joint stages; no other new loss",
                "selection": "Among lambdas within 0.02 of best mean validation C-index, minimize mean_pair_collapse; ties select smaller lambda",
                "selection_timing": "Selection JSON sealed before test or truth evaluation",
                "permutation": "same_composition only, selected collapse weight, lambda_ph=0.1, model seed=31, joint time/event shuffle seeds=101..105",
                "scaling": {name: asdict(values[2]) for name, values in prepared.items()},
                "old_inferred_results": str(previous_output), "frozen_data": str(frozen_output)}
    protocol_file = output / "protocol.json"
    if protocol_file.exists() and json.loads(protocol_file.read_text()) != json.loads(json.dumps(protocol)):
        raise ValueError("Output contains a different protocol; choose a new output directory")
    save_json(protocol_file, protocol)
    selections, all_records, all_aggregates, failures, baseline = {}, {}, {}, {}, {}
    with ProcessPoolExecutor(max_workers=config.workers, mp_context=multiprocessing.get_context("spawn")) as executor:
        for name, values in prepared.items():
            k = 6 if name == "same_composition" else 4
            logging.warning("Starting %s: %s fits", name, len(config.lambdas) * len(config.seeds))
            candidates, failures[name] = _collect_fits(executor, values[3], values[4], k, config)
            save_json(output / "numerical_failures.json", failures)
            selections[name] = _select(candidates, config)
            save_json(output / f"{name}_selection.json", selections[name])
            logging.warning("%s selection sealed: collapse=%s; starting held-out and truth evaluation", name, selections[name]["lambda_collapse"])
            records = [_score(weight, candidate, values) for weight, candidate in candidates]
            aggregates = {weight: aggregate_collapse([row for row in records if row["lambda_collapse"] == weight]) for weight in config.lambdas if any(row["lambda_collapse"] == weight for row in records)}
            all_records[name], all_aggregates[name] = records, aggregates
            save_json(output / f"{name}_sweep.json", {"selection": selections[name], "aggregates": aggregates, "runs": records, "failures": failures[name]})
            baseline[name] = _baseline_audit(name, records, previous_output)
            save_json(output / "zero_weight_reproduction.json", baseline)
        same = prepared["same_composition"]
        weight = selections["same_composition"]["lambda_collapse"]
        futures = {executor.submit(_fit, permute_training_phenotype(same[3], seed), same[4], _config(6, config.seeds[0], config), _weights(same[3], weight)): seed for seed in config.permutation_seeds}
        permutation_records = []
        for future in as_completed(futures):
            seed, candidate = futures[future], future.result()
            record = _score(weight, candidate, same)
            permuted = permute_training_phenotype(same[3], seed)
            record["permutation_seed"] = seed
            record["permuted_label_train_c_index"] = _c_index(candidate.train_inferred, candidate.factors["gamma"], permuted.time, permuted.event)
            permutation_records.append(record)
            logging.warning("Permutation %s complete (%s/%s)", seed, len(permutation_records), len(futures))
        permutation_records.sort(key=lambda row: row["permutation_seed"])
    permutation = {"lambda_collapse": weight, "lambda_ph": 0.1, "model_seed": config.seeds[0], "aggregate": aggregate_collapse(permutation_records), "runs": permutation_records}
    save_json(output / "same_composition_permutation.json", permutation)
    pairwise = {name: [{"lambda_collapse": row["lambda_collapse"], "seed": row["seed"], "pair": row["pair"], **row["redundancy"]} for row in records] for name, records in all_records.items()}
    save_json(output / "pairwise_similarity_diagnostics.json", pairwise)
    summary = {"selections": selections, "scenarios": {name: {"baseline": aggregates[0.0], "selected": aggregates[selections[name]["lambda_collapse"]]} for name, aggregates in all_aggregates.items()},
               "permutation": permutation["aggregate"], "zero_weight_reproduction": baseline,
               "numerical_failures": failures, "fit_count": sum(len(rows) for rows in all_records.values()) + len(permutation_records),
               "generator_and_split_unchanged": True}
    save_json(output / "summary.json", summary)
    logging.warning("Completed anti-collapse benchmark: %s successful fits", summary["fit_count"])
    return summary
