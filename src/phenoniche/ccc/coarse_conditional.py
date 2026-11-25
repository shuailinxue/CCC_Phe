"""Utilities for the isolated composition-first, conditional-CCC experiment."""

import numpy as np
import torch
from scipy import sparse


def within_class_graph(graph, labels):
    """Restrict the pre-existing statistical graph to the same coarse class."""
    coo = graph.tocoo()
    keep = labels[coo.row] == labels[coo.col]
    return sparse.csr_matrix((coo.data[keep], (coo.row[keep], coo.col[keep])), shape=graph.shape)


class ConditionalCoherence:
    """Spatial coherence after centering independently within composition classes."""

    def __init__(self, graph, labels, active, device):
        graph = graph.tocoo()
        indices = torch.as_tensor(np.vstack([graph.row, graph.col]), device=device)
        self.graph = torch.sparse_coo_tensor(indices, torch.as_tensor(graph.data, device=device), graph.shape).coalesce()
        self.labels = torch.as_tensor(labels, dtype=torch.long, device=device)
        self.indicator = torch.nn.functional.one_hot(self.labels).to(torch.float32)
        self.active = active.float()
        self.n_supported = self.indicator.T @ self.active
        self.denominator = self.indicator.T @ (self.active * torch.sparse.mm(self.graph, self.active))

    @torch.no_grad()
    def centered(self, z):
        centers = (self.indicator.T @ (z * self.active)) / self.n_supported.clamp_min(1)
        return (z - centers[self.labels]) * self.active

    @torch.no_grad()
    def score(self, z):
        z = self.centered(z)
        return (self.indicator.T @ (z * torch.sparse.mm(self.graph, z))) / self.denominator.clamp_min(1)


def empirical_max_test(observed, reference, calibration, valid):
    """One-sided max-statistic test; reference and calibration draws are disjoint.

    Exchangeability is required; maxT gives a global-null calibration, not an
    unconditional guarantee for every partially null feature configuration.
    """
    reference = np.asarray(reference, dtype=np.float64)
    calibration = np.asarray(calibration, dtype=np.float64)
    observed = np.asarray(observed, dtype=np.float64)
    valid = np.asarray(valid, dtype=bool)
    if reference.shape[1:] != observed.shape or calibration.shape[1:] != observed.shape:
        raise ValueError("Incompatible null and observed statistics")
    if not valid.any():
        raise ValueError("No CCC features have sufficient pair opportunities")
    mu = reference.mean(axis=0)
    sd = reference.std(axis=0, ddof=1)
    floor = max(float(np.median(sd[valid])) * 0.01, 1e-8)
    sd = np.maximum(sd, floor)
    z = (observed - mu) / sd
    null_z = (calibration - mu) / sd
    maxima = null_z[:, valid].max(axis=1)
    p = (1 + (maxima[:, None] >= z[valid][None, :]).sum(axis=0)) / (len(maxima) + 1)
    p_values = np.ones(observed.shape, dtype=np.float64)
    p_values[valid] = p
    return z, p_values, maxima, mu, sd
