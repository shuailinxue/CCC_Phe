from dataclasses import dataclass
import numpy as np
import torch
from phenoniche.data.schema import CohortData


@dataclass(frozen=True)
class SyntheticResult:
    data: CohortData
    truth: dict[str, torch.Tensor]


def generate_synthetic(patients=256, anchors=180, cell_types=8, contacts=10,
                       communications=16, number_of_niches=4, noise=0.005, seed=17):
    sizes = (patients, anchors, cell_types, contacts, communications, number_of_niches)
    if not all(isinstance(v, int) and v > 0 for v in sizes):
        raise ValueError("Synthetic dimensions must be positive integers")
    if number_of_niches < 2 or min(sizes[:-1]) < number_of_niches:
        raise ValueError("Synthetic identifiable design requires K >= 2 and all dimensions >= K")
    if not np.isfinite(noise) or noise < 0:
        raise ValueError("noise must be finite and nonnegative")
    rng = np.random.default_rng(seed)
    k = number_of_niches

    def dictionary(features):
        values = rng.uniform(0.005, 0.025, (k, features))
        for index in range(k):
            values[index, index::k] += rng.uniform(0.8, 1.2, len(range(index, features, k)))
        return values / values.sum(axis=1, keepdims=True)

    def activity(rows):
        values = rng.dirichlet(np.full(k, 0.3), size=rows) * rng.uniform(1, 3, (rows, 1))
        values[:k] = np.eye(k) * 2
        return values

    hc, ho, hi = dictionary(cell_types), dictionary(contacts), dictionary(communications)
    wb, ws = activity(patients), activity(anchors)
    gamma = np.zeros(k)
    gamma[0], gamma[1] = 2.0, -2.0
    eta = wb @ gamma
    failure = rng.exponential(size=patients) / (0.1 * np.exp(eta))
    censoring = rng.exponential(scale=25, size=patients)
    time = np.maximum(np.minimum(failure, censoring), 1e-6)
    event = (failure <= censoring).astype(np.float32)

    def observed(w, h):
        signal = w @ h
        return np.maximum(signal + rng.normal(0, noise, signal.shape), 0)

    def tensor(values):
        return torch.tensor(values, dtype=torch.float32)

    data = CohortData(*[tensor(v) for v in (observed(wb, hc), observed(wb, hi),
                      observed(ws, hc), observed(ws, ho), observed(ws, hi), time, event)])
    truth = {name: tensor(value) for name, value in
             {"WB": wb, "WS": ws, "HC": hc, "HO": ho, "HI": hi, "gamma": gamma}.items()}
    return SyntheticResult(data, truth)
