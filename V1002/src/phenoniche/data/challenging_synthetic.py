from dataclasses import dataclass, asdict
import numpy as np
import torch
from phenoniche.data.schema import CohortData


@dataclass(frozen=True)
class ChallengingConfig:
    scenario: str = "capacity"
    patients: int = 720
    anchors: int = 300
    cell_types: int = 12
    contacts: int = 18
    communications: int = 24
    true_niches: int = 6
    fit_niches: int = 4
    beta_risk: float = 1.8
    beta_protective: float = 1.8
    noise_fraction: float = 0.025
    seed: int = 20260917

    def __post_init__(self):
        dimensions = (self.patients, self.anchors, self.cell_types, self.contacts, self.communications, self.true_niches, self.fit_niches)
        if not all(isinstance(v, int) and v > 0 for v in dimensions):
            raise ValueError("Benchmark dimensions must be positive integers")
        if self.true_niches < 6 or self.fit_niches >= self.true_niches:
            raise ValueError("Benchmark requires K_true >= 6 and K_fit < K_true")
        if min(dimensions[:5]) < self.true_niches:
            raise ValueError("Every data dimension must be at least K_true")
        if self.scenario not in ("capacity", "same_composition"):
            raise ValueError("scenario must be capacity or same_composition")
        if not all(np.isfinite(v) and v > 0 for v in (self.beta_risk, self.beta_protective)):
            raise ValueError("Phenotype effect sizes must be finite and positive")
        if not np.isfinite(self.noise_fraction) or self.noise_fraction < 0:
            raise ValueError("noise_fraction must be finite and nonnegative")


@dataclass(frozen=True)
class ChallengingResult:
    data: CohortData
    truth: dict[str, torch.Tensor]
    metadata: dict


def _dictionary(rng, niches, features, amplitudes):
    values = rng.uniform(0.005, 0.02, (niches, features))
    for index in range(niches):
        values[index, index::niches] += rng.uniform(0.8, 1.2, len(range(index, features, niches)))
    return values / values.sum(1, keepdims=True) * amplitudes[:, None]


def _activities(rng, rows, niches):
    values = rng.gamma(shape=2.0, scale=0.5, size=(rows, niches))
    values[:, :2] = rng.gamma(shape=2.0, scale=0.4, size=(rows, 2))
    values[:, 2:4] = rng.lognormal(mean=0.4, sigma=0.8, size=(rows, 2))
    return values


def generate_challenging(config=None):
    config = config or ChallengingConfig()
    seeds = np.random.SeedSequence(config.seed).spawn(4)
    structural, noise_rng, survival_rng, censor_rng = [np.random.default_rng(seed) for seed in seeds]
    k = config.true_niches
    amplitudes = np.full(k, 1.3)
    amplitudes[:2], amplitudes[2:4] = 1.0, 2.5
    hc = _dictionary(structural, k, config.cell_types, amplitudes)
    ho = _dictionary(structural, k, config.contacts, amplitudes)
    hi = _dictionary(structural, k, config.communications, amplitudes)
    if config.scenario == "same_composition":
        hc[1] = hc[0]
    wb = _activities(structural, config.patients, k)
    ws = _activities(structural, config.anchors, k)
    gamma = np.zeros(k)
    gamma[0] = config.beta_risk
    if config.scenario == "capacity":
        gamma[1] = -config.beta_protective
    eta = wb @ gamma
    survival_draw = survival_rng.exponential(size=config.patients)
    failure = survival_draw / (0.04 * np.exp(eta))
    censoring = censor_rng.exponential(scale=50.0, size=config.patients)
    time = np.maximum(np.minimum(failure, censoring), np.finfo(np.float32).tiny)
    event = (failure <= censoring).astype(np.float32)

    def observe(w, h):
        values = w @ h
        deviation = config.noise_fraction * np.sqrt(np.mean(values ** 2))
        return np.maximum(values + noise_rng.normal(0, deviation, values.shape), 0)

    def tensor(values):
        return torch.tensor(values, dtype=torch.float32)

    data = CohortData(*[tensor(value) for value in (
        observe(wb, hc), observe(wb, hi), observe(ws, hc), observe(ws, ho), observe(ws, hi), time, event)])
    truth = {name: tensor(value) for name, value in {
        "WB": wb, "WS": ws, "HC": hc, "HO": ho, "HI": hi, "gamma": gamma, "eta": eta,
        "failure_time": failure, "censoring_time": censoring, "survival_draw": survival_draw,
    }.items()}
    roles = ["risk", "protective" if config.scenario == "capacity" else "neutral", "nuisance", "nuisance"] + ["neutral"] * (k - 4)
    senders = ["Tumor", "T_cell", "Macrophage", "Fibroblast"]
    receivers = ["Macrophage", "Tumor", "T_cell", "Tumor"]
    features = [{"sender": senders[i % 4], "receiver": receivers[i % 4], "LR": f"LR{i + 1}"}
                for i in range(config.communications)]
    metadata = {"config": asdict(config), "roles": roles, "directional_features": features,
                "baseline_hazard": 0.04, "nuisance_coefficients_exactly_zero": True,
                "independent_activity_columns": True}
    return ChallengingResult(data, truth, metadata)
