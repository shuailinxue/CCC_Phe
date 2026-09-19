from dataclasses import dataclass
import math
import time
import numpy as np
import torch
from phenoniche.utils.seed import set_seed


@dataclass(frozen=True)
class SparseExpandedMatrix:
    base: torch.Tensor
    lr_count: int
    active_indices: torch.Tensor
    active_delta: torch.Tensor

    @property
    def shape(self):
        return self.base.shape[0], self.base.shape[1] * self.lr_count

    def to(self, device):
        return SparseExpandedMatrix(self.base.to(device), self.lr_count, self.active_indices.to(device), self.active_delta.to(device))

    def sum(self):
        return self.lr_count * self.base.sum() + self.active_delta.sum()

    def square_sum(self):
        pairs = torch.div(self.active_indices, self.lr_count, rounding_mode="floor")
        cross = (self.base[:, pairs] * self.active_delta).sum()
        return self.lr_count * self.base.square().sum() + 2 * cross + self.active_delta.square().sum()

    def wt_x(self, activity):
        base_product = activity.T @ self.base
        result = base_product[:, :, None].expand(-1, -1, self.lr_count).clone().reshape(activity.shape[1], -1)
        result.index_add_(1, self.active_indices, activity.T @ self.active_delta)
        return result

    def x_h_t(self, dictionary):
        shaped = dictionary.reshape(dictionary.shape[0], self.base.shape[1], self.lr_count)
        result = self.base @ shaped.sum(2).T
        active_h = dictionary[:, self.active_indices]
        return result + self.active_delta @ active_h.T

    def dense_chunk(self, lr_start, lr_stop):
        width = lr_stop - lr_start
        result = self.base[:, :, None].expand(-1, -1, width).clone()
        pairs = torch.div(self.active_indices, self.lr_count, rounding_mode="floor")
        lr = self.active_indices % self.lr_count
        mask = (lr >= lr_start) & (lr < lr_stop)
        if mask.any():
            linear = pairs[mask] * width + lr[mask] - lr_start
            result.reshape(result.shape[0], -1).index_add_(1, linear, self.active_delta[:, mask])
        return result


@dataclass
class LargeSTResult:
    WS: torch.Tensor
    HC: torch.Tensor
    HI: torch.Tensor
    HO: torch.Tensor
    history: list
    seed: int
    training_seconds: float
    epoch_seconds: float


def summarize_pairs(neighborhoods, cell_types, cell_niches, tau=0.0001, pair_sigma=0.8, cell_type_count=8):
    anchors = len(neighborhoods.offsets) - 1
    opportunity = np.zeros((anchors, cell_type_count * cell_type_count), dtype=np.float32)
    niche_numerator = np.zeros((anchors, cell_type_count * cell_type_count, cell_niches.shape[1]), dtype=np.float32)
    for anchor in range(anchors):
        cells, weights, _ = neighborhoods.members(anchor)
        size = len(cells)
        left = np.repeat(np.arange(size), size)
        right = np.tile(np.arange(size), size)
        valid = left != right
        left, right = left[valid], right[valid]
        sender, receiver = cells[left], cells[right]
        distance = np.linalg.norm(neighborhoods.cell_coordinates[sender] - neighborhoods.cell_coordinates[receiver], axis=1)
        pair_weight = weights[left] * weights[right] * np.exp(-(distance ** 2) / (2 * pair_sigma ** 2))
        pair = cell_types[sender] * cell_type_count + cell_types[receiver]
        local = 0.5 * (cell_niches[sender] + cell_niches[receiver])
        local /= local.sum(1, keepdims=True)
        opportunity[anchor] = np.bincount(pair, weights=pair_weight, minlength=cell_type_count ** 2)
        for niche in range(local.shape[1]):
            niche_numerator[anchor, :, niche] = np.bincount(pair, weights=pair_weight * local[:, niche],
                                                             minlength=cell_type_count ** 2)
    denominator = opportunity + tau
    return opportunity, niche_numerator / denominator[:, :, None]


def build_large_spatial_matrix(opportunity, local_niche, hi, baseline=0.001, tau=0.0001, noise_fraction=0.005, seed=20261002):
    niches, pairs, lr_count = hi.shape
    background = float(hi.min())
    base = baseline * opportunity / (opportunity + tau) + background * local_niche.sum(2)
    rng = np.random.default_rng(seed)
    noise = rng.normal(0, noise_fraction * np.sqrt(np.mean(base ** 2)), base.shape)
    base = np.maximum(base + noise, 0).astype(np.float32)
    active = np.flatnonzero(np.max(hi - background, axis=0).reshape(-1) > 1e-12)
    pair = active // lr_count
    values = hi.reshape(niches, -1)[:, active] - background
    delta = np.einsum("nmk,km->nm", local_niche[:, pair, :], values, optimize=True).astype(np.float32)
    return SparseExpandedMatrix(torch.tensor(base), lr_count, torch.tensor(active, dtype=torch.long), torch.tensor(delta))


def fit_large_st(cs, communication, os, niches=6, iterations=500, seed=31, epsilon=1e-8, device="cuda"):
    set_seed(seed)
    cs, os, communication = cs.to(device), os.to(device), communication.to(device)
    n, f = communication.shape
    scales = (cs.shape[1] ** -0.5, f ** -0.5, os.shape[1] ** -0.5)
    generator = torch.Generator(device=device).manual_seed(seed)
    w = torch.rand((n, niches), generator=generator, device=device) + 0.2
    hc = torch.rand((niches, cs.shape[1]), generator=generator, device=device) + 0.2
    hi = torch.rand((niches, f), generator=generator, device=device) + 0.2
    ho = torch.rand((niches, os.shape[1]), generator=generator, device=device) + 0.2
    elements = n * (cs.shape[1] + f + os.shape[1])
    total_sum = scales[0] * cs.sum() + scales[1] * communication.sum() + scales[2] * os.sum()
    mean = (total_sum / elements).clamp_min(epsilon).sqrt()
    w = w / w.mean() * mean
    factor_mean = torch.cat((hc, hi, ho), 1).mean()
    hc, hi, ho = [value / factor_mean * mean for value in (hc, hi, ho)]
    history = []
    started = time.perf_counter()
    xnorm = scales[0] ** 2 * cs.square().sum() + scales[1] ** 2 * communication.square_sum() + scales[2] ** 2 * os.square().sum()
    for iteration in range(iterations):
        gram = w.T @ w
        hc *= (scales[0] * (w.T @ cs)) / (gram @ hc + epsilon)
        hi *= (scales[1] * communication.wt_x(w)) / (gram @ hi + epsilon)
        ho *= (scales[2] * (w.T @ os)) / (gram @ ho + epsilon)
        dictionary_gram = hc @ hc.T + hi @ hi.T + ho @ ho.T
        target = scales[0] * (cs @ hc.T) + scales[1] * communication.x_h_t(hi) + scales[2] * (os @ ho.T)
        w *= target / (w @ dictionary_gram + epsilon)
        if iteration == 0 or (iteration + 1) % 25 == 0 or iteration + 1 == iterations:
            gram = w.T @ w
            cross = (w.T @ cs * hc).sum() * scales[0] + (communication.wt_x(w) * hi).sum() * scales[1] + (w.T @ os * ho).sum() * scales[2]
            hgram = hc @ hc.T + hi @ hi.T + ho @ ho.T
            mse = (xnorm - 2 * cross + (gram * hgram).sum()) / elements
            history.append({"iteration": iteration + 1, "weighted_mean_squared_error": float(mse)})
    seconds = time.perf_counter() - started
    hc, hi, ho = hc / scales[0], hi / scales[1], ho / scales[2]
    scale = hc.sum(1).clamp_min(epsilon)
    w *= scale
    hc, hi, ho = [value / scale[:, None] for value in (hc, hi, ho)]
    return LargeSTResult(w.cpu(), hc.cpu(), hi.cpu(), ho.cpu(), history, seed, seconds, seconds / iterations)
