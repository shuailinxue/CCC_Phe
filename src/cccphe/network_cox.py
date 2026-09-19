from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import zarr
from lifelines.utils import concordance_index

from .core import (
    ccc_metadata,
    ccc_smoothing_groups,
    cox_breslow_loss,
    device_from_name,
    feature_batch,
    feature_mask,
    stratified_survival_folds,
    structural_feature_indices,
)
from .config import configured_path, path_for
from .utils import atomic_json, write_provenance


@dataclass
class SparseCoxFit:
    coefficient: np.ndarray
    lambda_fraction: float
    graph_lambda: float
    lambda_value: float
    cox_loss: float
    objective: float
    nonzero: int
    iterations: int


def _safe_concordance_index(
    time: np.ndarray, risk: np.ndarray, event: np.ndarray
) -> float:
    """Return a neutral score when a split has no admissible survival pairs."""
    try:
        return float(concordance_index(time, -risk, event))
    except ZeroDivisionError:
        return 0.5


def _group_smoothness(
    coefficient: torch.Tensor,
    group: torch.Tensor,
    group_count: int,
) -> torch.Tensor:
    sums = torch.zeros(group_count, dtype=coefficient.dtype, device=coefficient.device)
    counts = torch.zeros(group_count, dtype=coefficient.dtype, device=coefficient.device)
    sums.index_add_(0, group, coefficient)
    counts.index_add_(0, group, torch.ones_like(coefficient))
    means = sums / counts.clamp(min=1.0)
    # This is a sum-form Laplacian penalty (w^T L w), not an average over
    # features. Averaging would dilute graph_lambda by the number of CCCs and
    # make values such as 0.01--0.05 effectively indistinguishable from zero.
    return (coefficient - means[group]).square().sum()


def _smooth_objective(
    coefficient: torch.Tensor,
    x: torch.Tensor,
    time: torch.Tensor,
    event: torch.Tensor,
    groups: list[tuple[torch.Tensor, int]],
    graph_lambda: float,
    ridge_lambda: float,
) -> torch.Tensor:
    value = cox_breslow_loss(x @ coefficient, time, event)
    if graph_lambda > 0 and groups:
        value = value + graph_lambda * torch.stack(
            [_group_smoothness(coefficient, group, count) for group, count in groups]
        ).mean()
    if ridge_lambda > 0:
        value = value + ridge_lambda * coefficient.square().mean()
    return value


def _lambda_max(
    x: torch.Tensor,
    time: torch.Tensor,
    event: torch.Tensor,
) -> float:
    coefficient = torch.zeros(x.shape[1], dtype=x.dtype, device=x.device, requires_grad=True)
    loss = cox_breslow_loss(x @ coefficient, time, event)
    gradient = torch.autograd.grad(loss, coefficient)[0]
    return max(float(gradient.abs().max().detach().cpu()), 1e-8)


def fit_network_sparse_cox(
    x: np.ndarray,
    time: np.ndarray,
    event: np.ndarray,
    group_indices: list[np.ndarray],
    cfg: dict,
    lambda_fraction: float,
    *,
    graph_lambda: float | None = None,
    max_iterations: int | None = None,
) -> SparseCoxFit:
    """Fit Cox + L1 + overlapping group-Laplacian smoothness by FISTA."""
    model_cfg = cfg["network_cox"]
    device = device_from_name(model_cfg["device"])
    x_t = torch.tensor(x, dtype=torch.float32, device=device)
    time_t = torch.tensor(time, dtype=torch.float32, device=device)
    event_t = torch.tensor(event, dtype=torch.float32, device=device)
    groups = []
    for values in group_indices:
        group = torch.tensor(values, dtype=torch.long, device=device)
        groups.append((group, int(values.max()) + 1))
    lambda_value = float(lambda_fraction) * _lambda_max(x_t, time_t, event_t)
    graph_lambda = float(
        model_cfg.get("graph_lambda", 0.05) if graph_lambda is None else graph_lambda
    )
    ridge_lambda = float(model_cfg.get("ridge_lambda", 1e-5))
    maximum = int(max_iterations or model_cfg["max_iterations"])
    step = float(model_cfg.get("initial_step", 0.1))
    minimum_step = float(model_cfg.get("minimum_step", 1e-7))
    maximum_step = float(model_cfg.get("maximum_step", 1.0))
    tolerance = float(model_cfg.get("tolerance", 1e-6))
    minimum_iterations = int(model_cfg.get("minimum_iterations", 30))
    coefficient = torch.zeros(x.shape[1], dtype=torch.float32, device=device)
    extrapolated = coefficient.clone()
    momentum = 1.0
    previous_objective = np.inf
    for iteration in range(maximum):
        point = extrapolated.detach().requires_grad_(True)
        smooth = _smooth_objective(
            point, x_t, time_t, event_t, groups, graph_lambda, ridge_lambda
        )
        gradient = torch.autograd.grad(smooth, point)[0]
        accepted = False
        local_step = min(step * 1.05, maximum_step)
        for _ in range(20):
            with torch.no_grad():
                proposal = point - local_step * gradient
                proposal = proposal.sign() * torch.clamp(
                    proposal.abs() - local_step * lambda_value, min=0.0
                )
                proposal_smooth = _smooth_objective(
                    proposal, x_t, time_t, event_t, groups, graph_lambda, ridge_lambda
                )
                difference = proposal - point
                majorizer = (
                    smooth.detach()
                    + gradient.detach() @ difference
                    + difference.square().sum() / (2.0 * local_step)
                )
                if float(proposal_smooth.cpu()) <= float(majorizer.cpu()) + 1e-7:
                    accepted = True
                    break
            local_step *= 0.5
            if local_step < minimum_step:
                break
        if not accepted:
            raise RuntimeError("Network sparse Cox backtracking failed")
        with torch.no_grad():
            objective = float(
                (proposal_smooth + lambda_value * proposal.abs().sum()).cpu()
            )
            new_momentum = (1.0 + np.sqrt(1.0 + 4.0 * momentum**2)) / 2.0
            extrapolated = proposal + ((momentum - 1.0) / new_momentum) * (
                proposal - coefficient
            )
            relative_change = float(
                torch.linalg.vector_norm(proposal - coefficient).cpu()
                / torch.linalg.vector_norm(coefficient).clamp(min=1e-8).cpu()
            )
            coefficient = proposal
            momentum = new_momentum
            step = local_step
        if iteration + 1 >= minimum_iterations and (
            relative_change < tolerance
            or abs(previous_objective - objective)
            <= tolerance * max(1.0, abs(previous_objective))
        ):
            break
        previous_objective = objective
    with torch.no_grad():
        cox_loss = float(
            cox_breslow_loss(x_t @ coefficient, time_t, event_t).cpu()
        )
        values = coefficient.cpu().numpy()
    zero_tolerance = float(model_cfg.get("coefficient_zero_tolerance", 1e-8))
    values[np.abs(values) <= zero_tolerance] = 0.0
    return SparseCoxFit(
        coefficient=values,
        lambda_fraction=float(lambda_fraction),
        graph_lambda=graph_lambda,
        lambda_value=float(lambda_value),
        cox_loss=cox_loss,
        objective=float(objective),
        nonzero=int(np.count_nonzero(values)),
        iterations=iteration + 1,
    )


def _standardize_from_training(
    x: np.ndarray,
    mask: np.ndarray,
    train: np.ndarray,
    apply: np.ndarray,
    clip: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    observed = mask[train].astype(np.float32)
    count = observed.sum(axis=0)
    mean = np.divide(
        (x[train] * observed).sum(axis=0),
        count,
        out=np.zeros(x.shape[1], dtype=np.float32),
        where=count > 0,
    )
    variance = np.divide(
        (((x[train] - mean) * observed) ** 2).sum(axis=0),
        count,
        out=np.ones(x.shape[1], dtype=np.float32),
        where=count > 0,
    )
    scale = np.sqrt(np.maximum(variance, 1e-8)).astype(np.float32)
    values = np.where(mask[apply], x[apply], mean)
    values = np.clip((values - mean) / scale, -clip, clip).astype(np.float32)
    return values, mean, scale


def _choose_hyperparameters(
    x: np.ndarray,
    mask: np.ndarray,
    time: np.ndarray,
    event: np.ndarray,
    train: np.ndarray,
    group_indices: list[np.ndarray],
    cfg: dict,
) -> tuple[float, float, list[dict]]:
    model_cfg = cfg["network_cox"]
    inner_assignment = stratified_survival_folds(
        event[train], int(model_cfg.get("inner_folds", 4)), int(cfg["project"]["random_seed"]) + len(train)
    )
    inner_train = train[inner_assignment != 0]
    validation = train[inner_assignment == 0]
    train_x, _, _ = _standardize_from_training(
        x, mask, inner_train, inner_train, float(model_cfg.get("standardization_clip", 8.0))
    )
    validation_x, _, _ = _standardize_from_training(
        x, mask, inner_train, validation, float(model_cfg.get("standardization_clip", 8.0))
    )
    rows = []
    graph_lambdas = model_cfg.get(
        "graph_lambdas", [model_cfg.get("graph_lambda", 0.05)]
    )
    for graph_lambda in graph_lambdas:
        for fraction in model_cfg["lambda_fractions"]:
            fit = fit_network_sparse_cox(
                train_x,
                time[inner_train],
                event[inner_train],
                group_indices,
                cfg,
                float(fraction),
                graph_lambda=float(graph_lambda),
                max_iterations=int(model_cfg.get("tuning_iterations", 120)),
            )
            risk = validation_x @ fit.coefficient
            cindex = _safe_concordance_index(
                time[validation], risk, event[validation]
            )
            rows.append(
                {
                    "lambda_fraction": float(fraction),
                    "graph_lambda": float(graph_lambda),
                    "validation_cindex": cindex,
                    "nonzero": fit.nonzero,
                    "iterations": fit.iterations,
                }
            )
    candidates = pd.DataFrame(rows)
    maximum_selected = int(model_cfg.get("maximum_selected_ccc", 500))
    eligible = candidates[candidates.nonzero.between(1, maximum_selected)]
    if eligible.empty:
        eligible = candidates[candidates.nonzero.gt(0)]
    if eligible.empty:
        chosen_fraction = float(max(model_cfg["lambda_fractions"]))
        chosen_graph = float(min(graph_lambdas))
    else:
        cindex_tolerance = float(model_cfg.get("tuning_cindex_tolerance", 0.01))
        near_best = eligible[
            eligible.validation_cindex.ge(eligible.validation_cindex.max() - cindex_tolerance)
        ]
        # Parsimony rule: within a small C-index margin, prefer fewer CCCs.
        chosen = near_best.sort_values(
            ["nonzero", "validation_cindex", "lambda_fraction", "graph_lambda"],
            ascending=[True, False, False, False],
        ).iloc[0]
        chosen_fraction = float(chosen.lambda_fraction)
        chosen_graph = float(chosen.graph_lambda)
    return chosen_fraction, chosen_graph, rows


def _stratified_half_sample(event: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    selected = []
    for status in (0, 1):
        candidates = np.flatnonzero(event == status)
        selected.extend(
            rng.choice(candidates, max(2, len(candidates) // 2), replace=False)
        )
    return np.sort(np.asarray(selected, dtype=int))


def run_network_sparse_cox(cfg: dict, smoke: bool = False) -> Path:
    model_cfg = cfg["network_cox"]
    run_label = str(cfg.get("project", {}).get("name", "network-Cox"))
    input_dir = configured_path(cfg, "communication_tensor", "processed", "communication_tensor")
    root = zarr.open_group(str(input_dir / "C_propensity.zarr"), mode="r")
    full_mask = root["mask"][:].astype(bool)
    structural = structural_feature_indices(full_mask)
    x = feature_batch(root["C"], np.arange(root["C"].shape[0]), structural).astype(np.float32)
    mask = feature_mask(full_mask, np.arange(root["C"].shape[0]), structural).astype(bool)
    coverage = mask.mean(axis=0)
    count = mask.sum(axis=0)
    mean = np.divide((x * mask).sum(axis=0), count, out=np.zeros(x.shape[1]), where=count > 0)
    variance = np.divide(
        (((x - mean) * mask) ** 2).sum(axis=0),
        count,
        out=np.zeros(x.shape[1]),
        where=count > 0,
    )
    keep = (coverage >= float(model_cfg.get("minimum_coverage", 0.4))) & (
        variance >= float(model_cfg.get("minimum_variance", 1e-8))
    )
    structural, x, mask, coverage, variance = (
        structural[keep], x[:, keep], mask[:, keep], coverage[keep], variance[keep]
    )
    cap = float(np.quantile(x[mask], float(model_cfg.get("winsor_quantile", 0.999))))
    np.minimum(x, cap, out=x)
    samples = pd.read_csv(input_dir / "samples.tsv", sep="\t")["Sample_ID"]
    clinical_path = configured_path(cfg, "clinical", "processed", "clinical.tsv")
    clinical = pd.read_csv(clinical_path, sep="\t").set_index("Sample_ID").reindex(samples)
    endpoint = cfg["phenotype"]["primary_endpoint"]
    duration = cfg["tcga"]["endpoints"][endpoint]["time"]
    event_column = cfg["tcga"]["endpoints"][endpoint]["event"]
    time_all = pd.to_numeric(clinical[duration], errors="coerce")
    event_all = pd.to_numeric(clinical[event_column], errors="coerce")
    valid = time_all.gt(0) & event_all.isin([0, 1])
    selected = np.flatnonzero(valid.to_numpy())
    if smoke:
        rng = np.random.default_rng(int(cfg["project"]["random_seed"]))
        status = event_all.iloc[selected].to_numpy()
        chosen = []
        for value in (0, 1):
            candidates = selected[status == value]
            chosen.extend(rng.choice(candidates, min(35, len(candidates)), replace=False))
        selected = np.sort(np.asarray(chosen, dtype=int))
    x, mask = x[selected], mask[selected]
    sample_ids = samples.iloc[selected].reset_index(drop=True)
    time = time_all.iloc[selected].to_numpy(np.float32)
    event = event_all.iloc[selected].to_numpy(np.float32)
    metadata = ccc_metadata(input_dir, structural, tuple(root["C"].shape))
    groups = ccc_smoothing_groups(metadata)
    output = path_for(cfg, "results", "network_cox")
    output.mkdir(parents=True, exist_ok=True)
    metadata = metadata.copy()
    metadata["coverage"] = coverage
    metadata["standard_deviation"] = np.sqrt(variance)
    metadata["filtered_column"] = np.arange(len(metadata))
    metadata.to_csv(output / "candidate_ccc.tsv.gz", sep="\t", index=False, compression="gzip")

    original = {
        "device": model_cfg["device"],
        "max_iterations": model_cfg["max_iterations"],
        "lambda_fractions": list(model_cfg["lambda_fractions"]),
        "graph_lambdas": list(
            model_cfg.get("graph_lambdas", [model_cfg.get("graph_lambda", 0.05)])
        ),
        "stability_replicates": model_cfg["stability_replicates"],
    }
    if smoke:
        model_cfg["device"] = "cpu"
        model_cfg["max_iterations"] = min(60, int(model_cfg["max_iterations"]))
        model_cfg["lambda_fractions"] = [0.1, 0.03]
        graph_values = original["graph_lambdas"]
        model_cfg["graph_lambdas"] = sorted(set([min(graph_values), max(graph_values)]))
        model_cfg["stability_replicates"] = min(4, int(model_cfg["stability_replicates"]))
    folds = 2 if smoke else int(model_cfg.get("outer_folds", 5))
    assignment = stratified_survival_folds(
        event, folds, int(cfg["project"]["random_seed"]) + 311
    )
    fold_rows, tuning_rows, risk_rows, chosen_parameters = [], [], [], []
    try:
        for fold in range(folds):
            train = np.flatnonzero(assignment != fold)
            test = np.flatnonzero(assignment == fold)
            fraction, graph_lambda, tuning = _choose_hyperparameters(
                x, mask, time, event, train, groups, cfg
            )
            chosen_parameters.append((fraction, graph_lambda))
            for row in tuning:
                row["outer_fold"] = fold
                tuning_rows.append(row)
            train_x, _, _ = _standardize_from_training(
                x, mask, train, train, float(model_cfg.get("standardization_clip", 8.0))
            )
            test_x, _, _ = _standardize_from_training(
                x, mask, train, test, float(model_cfg.get("standardization_clip", 8.0))
            )
            fit = fit_network_sparse_cox(
                train_x, time[train], event[train], groups, cfg, fraction,
                graph_lambda=graph_lambda,
            )
            risk = test_x @ fit.coefficient
            cindex = _safe_concordance_index(time[test], risk, event[test])
            fold_rows.append(
                {
                    "fold": fold,
                    "train_n": len(train),
                    "train_events": int(event[train].sum()),
                    "test_n": len(test),
                    "test_events": int(event[test].sum()),
                    "lambda_fraction": fraction,
                    "graph_lambda": graph_lambda,
                    "selected_ccc": fit.nonzero,
                    "iterations": fit.iterations,
                    "held_out_cindex": cindex,
                }
            )
            centered = (risk - risk.mean()) / max(float(risk.std(ddof=1)), 1e-8)
            for local, index in enumerate(test):
                risk_rows.append(
                    {
                        "Sample_ID": sample_ids.iloc[index],
                        "fold": fold,
                        "time": time[index],
                        "event": event[index],
                        "risk": risk[local],
                        "fold_standardized_risk": centered[local],
                    }
                )
            print(
                f"{run_label} fold complete\t{fold + 1}/{folds}\t"
                f"lambda={fraction:.3g}\tgraph={graph_lambda:.3g}\t"
                f"selected={fit.nonzero}\tcindex={cindex:.4f}",
                flush=True,
            )

        tuning_frame = pd.DataFrame(tuning_rows)
        aggregate = (
            tuning_frame.groupby(["lambda_fraction", "graph_lambda"], as_index=False)
            .agg(
                mean_validation_cindex=("validation_cindex", "mean"),
                median_nonzero=("nonzero", "median"),
                valid_outer_folds=("outer_fold", "nunique"),
            )
        )
        maximum_selected = int(model_cfg.get("maximum_selected_ccc", 500))
        eligible = aggregate[
            aggregate.median_nonzero.between(1, maximum_selected)
        ]
        if eligible.empty:
            eligible = aggregate[aggregate.median_nonzero.gt(0)]
        if eligible.empty:
            raise RuntimeError("Every Cox hyperparameter combination selected zero CCCs")
        cindex_tolerance = float(model_cfg.get("tuning_cindex_tolerance", 0.01))
        near_best = eligible[
            eligible.mean_validation_cindex.ge(
                eligible.mean_validation_cindex.max() - cindex_tolerance
            )
        ]
        final_parameters = near_best.sort_values(
            ["median_nonzero", "mean_validation_cindex", "lambda_fraction", "graph_lambda"],
            ascending=[True, False, False, False],
        ).iloc[0]
        lambda_fraction = float(final_parameters.lambda_fraction)
        graph_lambda = float(final_parameters.graph_lambda)
        full_indices = np.arange(len(x))
        full_x, _, _ = _standardize_from_training(
            x, mask, full_indices, full_indices, float(model_cfg.get("standardization_clip", 8.0))
        )
        full_fit = fit_network_sparse_cox(
            full_x, time, event, groups, cfg, lambda_fraction,
            graph_lambda=graph_lambda,
        )
        replicates = int(model_cfg["stability_replicates"])
        coefficients = np.zeros((replicates, x.shape[1]), dtype=np.float32)
        rng = np.random.default_rng(int(cfg["project"]["random_seed"]) + 5000)
        for replicate in range(replicates):
            subset = _stratified_half_sample(event, rng)
            fit = fit_network_sparse_cox(
                full_x[subset],
                time[subset],
                event[subset],
                groups,
                cfg,
                lambda_fraction,
                graph_lambda=graph_lambda,
                max_iterations=int(model_cfg.get("stability_iterations", 100)),
            )
            coefficients[replicate] = fit.coefficient
            if (replicate + 1) % max(1, replicates // 10) == 0:
                print(
                    f"{run_label} stability\t{replicate + 1}/{replicates}\tselected={fit.nonzero}",
                    flush=True,
                )
    finally:
        model_cfg.update(original)

    tolerance = float(model_cfg.get("coefficient_zero_tolerance", 1e-8))
    # Preserve replicate-level coefficients for auditing direction stability.
    np.save(output / "stability_beta_replicates.npy", coefficients)
    positive = np.mean(coefficients > tolerance, axis=0)
    negative = np.mean(coefficients < -tolerance, axis=0)
    mean_coefficient = coefficients.mean(axis=0)
    result = metadata.copy()
    result["full_coefficient"] = full_fit.coefficient
    result["positive_selection_probability"] = positive
    result["negative_selection_probability"] = negative
    result["selection_probability"] = positive + negative
    result["sign_selection_probability"] = np.maximum(positive, negative)
    result["mean_coefficient"] = mean_coefficient
    result["mean_absolute_coefficient"] = np.mean(np.abs(coefficients), axis=0)
    result["direction"] = np.where(positive >= negative, "risk", "protective")
    result["stability_importance"] = (
        result.sign_selection_probability * result.mean_absolute_coefficient
    )
    result.to_csv(output / "all_ccc_coefficients.tsv.gz", sep="\t", index=False, compression="gzip")
    stable = result[
        result.sign_selection_probability.ge(float(model_cfg.get("stability_threshold", 0.7)))
    ].copy()
    stable = stable.sort_values("stability_importance", ascending=False)
    stable = stable.head(int(model_cfg.get("maximum_stable_ccc", 500)))
    stable.to_csv(output / "stable_ccc.tsv", sep="\t", index=False)
    fold_frame = pd.DataFrame(fold_rows)
    fold_frame.to_csv(output / "fold_metrics.tsv", sep="\t", index=False)
    tuning_frame.to_csv(output / "lambda_tuning.tsv", sep="\t", index=False)
    aggregate.to_csv(output / "hyperparameter_grid_summary.tsv", sep="\t", index=False)
    risk_frame = pd.DataFrame(risk_rows)
    risk_frame.to_csv(output / "cross_fitted_risk.tsv", sep="\t", index=False)
    pooled = _safe_concordance_index(
        risk_frame.time.to_numpy(),
        risk_frame.fold_standardized_risk.to_numpy(),
        risk_frame.event.to_numpy(),
    )
    summary = {
        "model": "ST_anchored_exact_CCC_network_sparse_Cox",
        "endpoint": endpoint,
        "samples": len(x),
        "events": int(event.sum()),
        "structural_ccc": int(len(keep)),
        "candidate_ccc_after_outcome_independent_qc": int(x.shape[1]),
        "selected_lambda_fraction": lambda_fraction,
        "selected_graph_lambda": graph_lambda,
        "tuning_cindex_tolerance": float(model_cfg.get("tuning_cindex_tolerance", 0.01)),
        "outer_fold_selected_hyperparameters": [
            {"lambda_fraction": float(fraction), "graph_lambda": float(graph)}
            for fraction, graph in chosen_parameters
        ],
        "full_model_selected_ccc": full_fit.nonzero,
        "stable_ccc": int(len(stable)),
        "mean_fold_cindex": float(fold_frame.held_out_cindex.mean()),
        "pooled_fold_standardized_cindex": pooled,
        "fold_cindex": fold_frame.held_out_cindex.tolist(),
        "stability_replicates": int(coefficients.shape[0]),
        "stability_threshold": float(model_cfg.get("stability_threshold", 0.7)),
        "smoke": smoke,
    }
    atomic_json(summary, output / "summary.json")
    write_provenance(output / "provenance.json", cfg, summary)
    report = [
        f"# {run_label}: exact CCC network sparse Cox",
        "",
        f"- Samples/events: {len(x)}/{int(event.sum())}",
        f"- Structural CCC before QC: {len(keep):,}",
        f"- Candidate CCC after outcome-independent QC: {x.shape[1]:,}",
        f"- Selected lambda fraction / graph lambda: {lambda_fraction:g} / {graph_lambda:g}",
        f"- Full-model selected CCC: {full_fit.nonzero:,}",
        f"- Stable CCC: {len(stable):,}",
        f"- Mean held-out C-index: {fold_frame.held_out_cindex.mean():.3f}",
        f"- Fold-standardized pooled C-index: {pooled:.3f}",
        "",
        "No tensor/NMF decomposition and no clinical covariates are used. Each",
        "predictor is one ST-anchored directed sender->receiver/LR CCC.",
    ]
    (output / "REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    return output
