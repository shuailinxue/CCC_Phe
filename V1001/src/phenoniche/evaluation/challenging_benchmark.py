from dataclasses import asdict, dataclass, replace
from pathlib import Path
import hashlib
import json
import logging
import numpy as np
import torch
from phenoniche.data.challenging_synthetic import ChallengingConfig, generate_challenging
from phenoniche.data.scaling import BlockScaler
from phenoniche.evaluation.metrics import concordance_index, cosine_similarity, pearson_correlation
from phenoniche.evaluation.patient_protocol import split_patients, subset_bulk, permute_training_phenotype, fit_frozen_cox, select_positive_lambda
from phenoniche.evaluation.phenotype_recovery import phenotype_recovery
from phenoniche.inference.bulk import infer_bulk_activities
from phenoniche.training.config import TrainingConfig, LossWeights
from phenoniche.training.trainer import train


@dataclass(frozen=True)
class BenchmarkConfig:
    model_seeds: tuple[int, ...] = (31, 32, 33, 34, 35)
    permutation_seeds: tuple[int, ...] = (101, 102, 103, 104, 105)
    lambdas: tuple[float, ...] = (0.0, 0.001, 0.005, 0.01, 0.05, 0.1)
    warmup_epochs: int = 800
    joint_epochs: int = 800
    split_seed: int = 90210
    regularization: float = 0.0001
    frozen_cox_ridge: float = 0.01
    device: str = "cpu"

    def __post_init__(self):
        if not self.model_seeds or len(set(self.model_seeds)) != len(self.model_seeds):
            raise ValueError("Model seeds must be nonempty and unique")
        if not self.permutation_seeds or len(set(self.permutation_seeds)) != len(self.permutation_seeds):
            raise ValueError("Permutation seeds must be nonempty and unique")
        if len(set(self.lambdas)) != len(self.lambdas) or 0 not in self.lambdas or not any(v > 0 for v in self.lambdas):
            raise ValueError("Unique lambda candidates must include zero and positive values")
        if not all(np.isfinite(v) and v >= 0 for v in self.lambdas):
            raise ValueError("Lambda candidates must be finite and nonnegative")
        LossWeights(reg=self.regularization)
        TrainingConfig(warmup_epochs=self.warmup_epochs, joint_epochs=self.joint_epochs, device=self.device)
        if not np.isfinite(self.frozen_cox_ridge) or self.frozen_cox_ridge <= 0:
            raise ValueError("Frozen Cox ridge must be positive")


@dataclass
class Candidate:
    seed: int
    weight: float
    factors: dict
    train_inferred: torch.Tensor
    validation_inferred: torch.Tensor
    validation_c_index: float
    fit_epochs: int
    inference_options: dict
    training_diagnostics: dict


def _c_index(activity, gamma, time, event):
    return concordance_index(time.detach().cpu().numpy(), event.detach().cpu().numpy(),
                             (activity @ gamma).detach().cpu().numpy())


def fit_candidate(train_data, validation_data, training_config, weights, frozen_cox_ridge=0.01):
    result = train(train_data, training_config, weights)
    with torch.no_grad():
        factors = {name: value.cpu().clone() for name, value in result.model.factors().items()}
    inference_options = {"steps": training_config.inner_steps, "learning_rate": training_config.inner_lr,
                         "lambda_bc": weights.bc, "lambda_bi": weights.bi, "create_graph": False}
    factors["factorization_gamma"] = factors["gamma"].clone()
    inferred = infer_bulk_activities(train_data.CB.cpu(), train_data.IB.cpu(), factors["HC"], factors["HI"], **inference_options)
    if weights.ph == 0:
        factors["gamma"] = fit_frozen_cox(inferred, train_data.time.cpu(), train_data.event.cpu(), ridge=frozen_cox_ridge)
    validation = infer_bulk_activities(validation_data.CB.cpu(), validation_data.IB.cpu(), factors["HC"], factors["HI"], **inference_options)
    score = _c_index(validation, factors["gamma"], validation_data.time, validation_data.event)
    if not np.isfinite(score):
        raise ValueError("Validation partition has no comparable survival pairs")
    return Candidate(training_config.seed, weights.ph, factors, inferred, validation, score, len(result.history), inference_options, result.diagnostics)


def _reconstruction(factors, activity, data):
    errors = {"CB": float((data.CB - activity @ factors["HC"]).square().mean()),
              "IB": float((data.IB - activity @ factors["HI"]).square().mean())}
    for name, h in (("CS", "HC"), ("OS", "HO"), ("IS", "HI")):
        errors[name] = float((getattr(data, name) - factors["WS"] @ factors[h]).square().mean())
    return {"block_mse": errors, "bulk_reconstruction": errors["CB"] + errors["IB"],
            "spatial_reconstruction": errors["CS"] + errors["OS"] + errors["IS"]}


def score_candidate(candidate, train_data, validation_data, test_data, truth, roles, split, weights):
    factors = candidate.factors
    inferred_test = infer_bulk_activities(test_data.CB, test_data.IB, factors["HC"], factors["HI"], **candidate.inference_options)
    recovery = phenotype_recovery(factors, truth, roles, candidate.train_inferred, truth["WB"][split.train])
    test_recovery = phenotype_recovery(factors, truth, roles, inferred_test, truth["WB"][split.test])
    for record, test_record in zip(recovery["per_true_niche"], test_recovery["per_true_niche"]):
        record["WB_test_correlation"] = test_record["WB_correlation"]
    return {"seed": candidate.seed, "lambda_ph": candidate.weight, "epochs": candidate.fit_epochs,
            "train_c_index": _c_index(candidate.train_inferred, factors["gamma"], train_data.time, train_data.event),
            "validation_c_index": candidate.validation_c_index,
            "test_c_index": _c_index(inferred_test, factors["gamma"], test_data.time, test_data.event),
            "recovery": recovery,
            "train_reconstruction": _reconstruction(factors, candidate.train_inferred, train_data),
            "test_reconstruction": _reconstruction(factors, inferred_test, test_data),
            "factorization_gamma": factors["factorization_gamma"].tolist(),
            "evaluation_gamma": factors["gamma"].tolist(),
            "gamma_source": "training-only frozen Cox head" if candidate.weight == 0 else "joint training"}


def _stats(values):
    array = np.asarray(values, dtype=float)
    finite = array[np.isfinite(array)]
    return {"mean": float(finite.mean()) if len(finite) else None,
            "std": float(finite.std(ddof=1)) if len(finite) > 1 else None,
            "n": len(finite), "undefined": len(array) - len(finite)}


def aggregate_records(records):
    result = {key: _stats([row[key] for row in records])
              for key in ("train_c_index", "validation_c_index", "test_c_index")}
    for key in ("bulk_reconstruction", "spatial_reconstruction"):
        result[key] = _stats([row["train_reconstruction"][key] for row in records])
    result["overall_dictionary_recovery"] = _stats([row["recovery"]["overall_dictionary_recovery"] for row in records])
    for role in ("risk", "protective", "neutral", "nuisance"):
        key = f"{role}_niche_recovery"
        groups = [row["recovery"][key] for row in records if row["recovery"][key] is not None]
        result[key] = {metric: _stats([group[metric] for group in groups]) for metric in groups[0]} if groups else None
        signs = [niche["gamma_sign_correct"] for row in records for niche in row["recovery"]["per_true_niche"]
                 if niche["role"] == role and niche["gamma_sign_correct"] is not None]
        result[f"{role}_gamma_sign_consistency"] = float(np.mean(signs)) if signs else None
    return result


def _json_safe(value):
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return None
    return value


def save_json(path, value):
    Path(path).write_text(json.dumps(_json_safe(value), indent=2, allow_nan=False))


def _oracle(test_data, truth, roles, split, weights):
    activity = infer_bulk_activities(test_data.CB, test_data.IB, truth["HC"], truth["HI"], weights.bc, weights.bi)
    return {"test_c_index_true_risk": _c_index(truth["WB"][split.test], truth["gamma"], test_data.time, test_data.event),
            "test_c_index_true_dictionary_inference": _c_index(activity, truth["gamma"], test_data.time, test_data.event),
            "phenotype_activity_correlations": {str(k): pearson_correlation(activity[:, k].numpy(), truth["WB"][split.test, k].numpy())
                                                for k, role in enumerate(roles) if role in ("risk", "protective")}}


def run_benchmark(output, data_config=None, benchmark_config=None):
    data_config = data_config or ChallengingConfig()
    benchmark_config = benchmark_config or BenchmarkConfig()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(__name__)
    protocol = {"data": asdict(data_config), "benchmark": asdict(benchmark_config),
                "selection": "Highest mean validation C-index across fixed model seeds; positive lambdas only; ties choose smaller lambda.",
                "test_policy": "Test outcomes are scored only after selection.json is written; no refitting or selection uses them.",
                "inference": "Same differentiable unrolled projected-gradient implementation in train/validation/test; no phenotype arguments.",
                "unsupervised_head": "Additional training-only frozen-dictionary Cox evaluation head, ridge fixed in advance; factorization gamma remains zero.",
                "recovery": "Best match per true niche plus collisions and rectangular Hungarian coverage; unmatched true niches count zero overall.",
                "epochs": "Identical fixed epochs; no early stopping or ground-truth tuning.",
                "spatial": "Independent spatial cohort without phenotype; used in training and scaling only.",
                "scope": "One fixed synthetic cohort per scenario; model seeds measure initialization variation, not independent cohorts."}
    protocol_path = output / "protocol.json"
    if protocol_path.exists() and json.loads(protocol_path.read_text()) != _json_safe(protocol):
        raise ValueError("Output directory contains a different protocol; choose a new output directory")
    save_json(protocol_path, protocol)
    synthetic = generate_challenging(data_config)
    split = split_patients(data_config.patients, benchmark_config.split_seed)
    raw_train = subset_bulk(synthetic.data, split.train)
    scaler = BlockScaler.fit(raw_train)
    train_data = scaler.transform(raw_train)
    validation_data = scaler.transform(subset_bulk(synthetic.data, split.validation))
    truth = {key: value.clone() for key, value in synthetic.truth.items()}
    truth["HC"] /= scaler.composition
    truth["HI"] /= scaler.communication
    truth["HO"] /= scaler.topology
    weights = LossWeights.per_entry(train_data, reg=benchmark_config.regularization)
    candidates = []
    for seed in benchmark_config.model_seeds:
        config = TrainingConfig(number_of_niches=data_config.fit_niches, seed=seed, device=benchmark_config.device,
                                warmup_epochs=benchmark_config.warmup_epochs, joint_epochs=benchmark_config.joint_epochs)
        for weight in benchmark_config.lambdas:
            logger.warning("%s: fitting seed=%s lambda=%s", data_config.scenario, seed, weight)
            candidate = fit_candidate(train_data, validation_data, config, replace(weights, ph=weight), benchmark_config.frozen_cox_ridge)
            candidates.append(candidate)
    validation_scores = {weight: [c.validation_c_index for c in candidates if c.weight == weight] for weight in benchmark_config.lambdas}
    selected = select_positive_lambda(validation_scores)
    selection = {"lambda_ph": selected, "validation_scores": validation_scores,
                 "protocol_sha256": hashlib.sha256(protocol_path.read_bytes()).hexdigest(),
                 "test_outcomes_used": False}
    save_json(output / "selection.json", selection)
    logger.warning("%s: validation selection sealed, lambda=%s; beginning test scoring", data_config.scenario, selected)
    test_data = scaler.transform(subset_bulk(synthetic.data, split.test))
    records = [score_candidate(candidate, train_data, validation_data, test_data, truth, synthetic.metadata["roles"], split, weights)
               for candidate in candidates]
    sweeps = {weight: aggregate_records([record for record in records if record["lambda_ph"] == weight]) for weight in benchmark_config.lambdas}
    save_json(output / "lambda_sweep.json", {"aggregates": sweeps, "runs": records})
    chosen = {label: [row for row in records if row["lambda_ph"] == weight]
              for label, weight in (("unsupervised", 0.0), ("supervised", selected))}
    save_json(output / "seed_stability.json", {label: {"aggregate": aggregate_records(rows), "runs": rows} for label, rows in chosen.items()})
    for label, weight in (("unsupervised", 0.0), ("supervised", selected)):
        candidate = next(c for c in candidates if c.seed == benchmark_config.model_seeds[0] and c.weight == weight)
        np.savez_compressed(output / f"{label}_factors.npz", **{name: value.numpy() for name, value in candidate.factors.items()})
    permutation_records = []
    config = TrainingConfig(number_of_niches=data_config.fit_niches, seed=benchmark_config.model_seeds[0], device=benchmark_config.device,
                            warmup_epochs=benchmark_config.warmup_epochs, joint_epochs=benchmark_config.joint_epochs)
    for permutation_seed in benchmark_config.permutation_seeds:
        logger.warning("%s: phenotype permutation seed=%s", data_config.scenario, permutation_seed)
        permuted = permute_training_phenotype(train_data, permutation_seed)
        candidate = fit_candidate(permuted, validation_data, config, replace(weights, ph=selected), benchmark_config.frozen_cox_ridge)
        record = score_candidate(candidate, train_data, validation_data, test_data, truth, synthetic.metadata["roles"], split, weights)
        record["permutation_seed"] = permutation_seed
        record["permuted_label_train_c_index"] = _c_index(candidate.train_inferred, candidate.factors["gamma"], permuted.time, permuted.event)
        permutation_records.append(record)
    permutation = {"lambda_ph": selected, "model_seed": config.seed,
                   "aggregate": aggregate_records(permutation_records), "runs": permutation_records}
    save_json(output / "phenotype_permutation.json", permutation)
    summary = {"scenario": data_config.scenario, "selected_lambda": selected, "selection": selection,
               "scaling": asdict(scaler), "patient_counts": {name: len(getattr(split, name)) for name in ("train", "validation", "test")},
               "event_fractions": {name: float(data.event.mean()) for name, data in (("train", train_data), ("validation", validation_data), ("test", test_data))},
               "comparison": {label: aggregate_records(rows) for label, rows in chosen.items()},
               "permutation": permutation["aggregate"], "oracle": _oracle(test_data, truth, synthetic.metadata["roles"], split, weights),
               "truth_pair_similarity": {name: cosine_similarity(synthetic.truth[name][0].numpy(), synthetic.truth[name][1].numpy()) for name in ("HC", "HI", "HO")},
               "roles": synthetic.metadata["roles"], "torch_version": torch.__version__,
               "factor_export_seed": benchmark_config.model_seeds[0]}
    save_json(output / "summary.json", summary)
    save_json(output / "splits.json", {name: getattr(split, name).tolist() for name in ("train", "validation", "test")})
    save_json(output / "ground_truth_metadata.json", synthetic.metadata)
    np.savez_compressed(output / "ground_truth.npz", **{name: value.numpy() for name, value in synthetic.truth.items()})
    logger.warning("%s: completed; summary=%s", data_config.scenario, output / "summary.json")
    return summary
