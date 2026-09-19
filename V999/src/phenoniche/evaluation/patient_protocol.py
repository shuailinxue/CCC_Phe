from dataclasses import dataclass, replace
import math
import numpy as np
import torch
from phenoniche.data.schema import CohortData
from phenoniche.losses.survival import cox_breslow_loss
from phenoniche.utils.validation import matrix, survival_vectors


@dataclass(frozen=True)
class PatientSplit:
    train: np.ndarray
    validation: np.ndarray
    test: np.ndarray

    def validate(self, patients):
        arrays = (self.train, self.validation, self.test)
        if any(a.ndim != 1 or len(a) < 1 or not np.issubdtype(a.dtype, np.integer) for a in arrays):
            raise ValueError("Patient partitions must be nonempty integer vectors")
        joined = np.concatenate(arrays)
        if len(joined) != patients or not np.array_equal(np.sort(joined), np.arange(patients)):
            raise ValueError("Patient partitions must be disjoint and cover every patient once")


def split_patients(patients, seed=90210, train_fraction=0.6, validation_fraction=0.2):
    if not isinstance(patients, int) or patients < 5:
        raise ValueError("At least five patients are required")
    if not all(math.isfinite(v) and 0 < v < 1 for v in (train_fraction, validation_fraction)) or train_fraction + validation_fraction >= 1:
        raise ValueError("Split fractions must be positive with a sum below one")
    indices = np.random.default_rng(seed).permutation(patients)
    train_end = int(patients * train_fraction)
    val_end = train_end + int(patients * validation_fraction)
    result = PatientSplit(indices[:train_end], indices[train_end:val_end], indices[val_end:])
    result.validate(patients)
    return result


def subset_bulk(data, indices):
    indices = np.asarray(indices)
    if indices.ndim != 1 or indices.size == 0 or not np.issubdtype(indices.dtype, np.integer):
        raise ValueError("Patient indices must be a nonempty integer vector")
    if (indices < 0).any() or (indices >= data.CB.shape[0]).any() or len(np.unique(indices)) != len(indices):
        raise ValueError("Patient indices must be unique and within bounds")
    selected = torch.tensor(indices, device=data.CB.device, dtype=torch.long)
    return CohortData(data.CB[selected], data.IB[selected], data.CS, data.OS, data.IS,
                      data.time[selected], data.event[selected], data.laplacian)


def permute_training_phenotype(train_data, seed):
    order = torch.tensor(np.random.default_rng(seed).permutation(train_data.CB.shape[0]),
                         device=train_data.time.device, dtype=torch.long)
    return replace(train_data, time=train_data.time[order], event=train_data.event[order])


def fit_frozen_cox(activity, time, event, ridge=0.01, iterations=300):
    matrix(activity, "training activity")
    survival_vectors(activity[:, 0], time, event)
    if not event.any():
        raise ValueError("Frozen Cox fitting requires observed training events")
    if not math.isfinite(ridge) or ridge <= 0 or not isinstance(iterations, int) or iterations < 1:
        raise ValueError("Frozen Cox ridge and iterations must be positive")
    x = activity.detach().double()
    t, e = time.detach().double(), event.detach().double()
    gamma = torch.zeros(x.shape[1], dtype=x.dtype, device=x.device, requires_grad=True)
    optimizer = torch.optim.LBFGS([gamma], max_iter=iterations, tolerance_grad=1e-9,
                                  tolerance_change=1e-12, line_search_fn="strong_wolfe")

    def closure():
        optimizer.zero_grad()
        loss = cox_breslow_loss(x @ gamma, t, e) + ridge * gamma.square().sum()
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite frozen Cox objective")
        loss.backward()
        return loss

    optimizer.step(closure)
    return gamma.detach().to(dtype=activity.dtype)


def select_positive_lambda(validation_scores):
    candidates = {float(weight): np.asarray(scores, dtype=float) for weight, scores in validation_scores.items() if float(weight) > 0}
    if not candidates or any(v.size == 0 or not np.isfinite(v).all() for v in candidates.values()):
        raise ValueError("Selection requires finite validation scores for positive lambda candidates")
    return min(candidates, key=lambda weight: (-float(candidates[weight].mean()), weight))
