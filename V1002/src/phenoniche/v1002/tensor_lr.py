from dataclasses import dataclass
import math
import time
import torch
from phenoniche.utils.seed import set_seed
from phenoniche.v1002.large_lr import SparseExpandedMatrix


@dataclass
class TensorLRResult:
    WS: torch.Tensor
    HC: torch.Tensor
    G: torch.Tensor
    V: torch.Tensor
    HO: torch.Tensor
    history: list
    seed: int
    training_seconds: float
    epoch_seconds: float

    def raw_lr_dictionary(self):
        return torch.einsum("kpr,lr->kpl", self.G, self.V)


def normalize_lr_programs(g, v, epsilon=1e-12):
    scale = v.sum(0).clamp_min(epsilon)
    return g * scale[None, None, :], v / scale[None, :]


def dense_tensor_sse(observed, w, g, v):
    predicted = torch.einsum("nk,kpr,lr->npl", w, g, v)
    return (observed - predicted).square().sum()


def exact_tensor_sse_from_statistics(x_square_sum, x_base, active_indices, active_delta, lr_count, w, g, v):
    pairs = torch.div(active_indices, lr_count, rounding_mode="floor")
    lr = active_indices % lr_count
    wt_base = w.T @ x_base
    wt_delta = w.T @ active_delta
    cross_base = (wt_base[:, :, None] * g).sum((0, 1)) @ v.sum(0)
    selected = g[:, pairs, :] * v[lr, :][None, :, :]
    cross_active = (wt_delta[:, :, None] * selected).sum()
    v_gram = v.T @ v
    w_gram = w.T @ w
    h_gram = torch.einsum("kpr,jps,rs->kj", g, g, v_gram)
    prediction_square = (w_gram * h_gram).sum()
    return x_square_sum - 2 * (cross_base + cross_active) + prediction_square


def exact_tensor_mse(communication, w, g, v):
    if not isinstance(communication, SparseExpandedMatrix):
        raise TypeError("communication must use SparseExpandedMatrix")
    sse = exact_tensor_sse_from_statistics(
        communication.square_sum(), communication.base, communication.active_indices,
        communication.active_delta, communication.lr_count, w, g, v
    )
    return sse / math.prod(communication.shape)


def _scatter_g_numerator(wt_base, wt_delta, g, v, pairs, lr):
    numerator = wt_base[:, :, None] * v.sum(0)[None, None, :]
    addition = wt_delta[:, :, None] * v[lr, :][None, :, :]
    for pair in torch.unique(pairs):
        mask = pairs == pair
        numerator[:, pair, :] += addition[:, mask, :].sum(1)
    return numerator


def _v_numerator(wt_base, wt_delta, g, v, pairs, lr):
    base = torch.einsum("kp,kpr->r", wt_base, g)
    numerator = base[None, :].expand(v.shape[0], -1).clone()
    selected = g[:, pairs, :]
    addition = torch.einsum("km,kmr->mr", wt_delta, selected)
    numerator.index_add_(0, lr, addition)
    return numerator


def _ccc_target(communication, g, v):
    h_sum = torch.einsum("kpr,r->kp", g, v.sum(0))
    target = communication.base @ h_sum.T
    pairs = torch.div(communication.active_indices, communication.lr_count, rounding_mode="floor")
    lr = communication.active_indices % communication.lr_count
    selected = torch.einsum("kmr,mr->km", g[:, pairs, :], v[lr, :])
    return target + communication.active_delta @ selected.T


def _ccc_gram(g, v):
    return torch.einsum("kpr,jps,rs->kj", g, g, v.T @ v)


def _initialize(cs, communication, os, niches, rank, seed, device, epsilon):
    generator = torch.Generator(device=device).manual_seed(seed)
    random = lambda shape: torch.rand(shape, generator=generator, device=device) + 0.2
    w = random((cs.shape[0], niches))
    w /= w.sum(1, keepdim=True)
    hc = random((niches, cs.shape[1]))
    ho = random((niches, os.shape[1]))
    g = random((niches, communication.base.shape[1], rank))
    v = random((communication.lr_count, rank))
    g, v = normalize_lr_programs(g, v, epsilon)
    hc *= cs.mean() / (w @ hc).mean().clamp_min(epsilon)
    ho *= os.mean() / (w @ ho).mean().clamp_min(epsilon)
    g *= (communication.sum() / math.prod(communication.shape)) / (
        torch.einsum("nk,kpr,lr->", w, g, v) / math.prod(communication.shape)
    ).clamp_min(epsilon)
    return w, hc, g, v, ho


def fit_tensor_st(cs, communication, os, niches=6, rank=64, iterations=500, seed=31, epsilon=1e-8, device="cuda"):
    if rank != 64:
        raise ValueError("V1002 fixes the LR-program rank at R=64")
    set_seed(seed)
    cs = cs.to(device)
    os = os.to(device)
    communication = communication.to(device)
    w, hc, g, v, ho = _initialize(cs, communication, os, niches, rank, seed, device, epsilon)
    pairs = torch.div(communication.active_indices, communication.lr_count, rounding_mode="floor")
    lr = communication.active_indices % communication.lr_count
    c_weight = 1 / cs.shape[1]
    i_weight = 1 / math.prod(communication.shape[1:])
    o_weight = 1 / os.shape[1]
    history = []
    started = time.perf_counter()
    for iteration in range(iterations):
        w_gram = w.T @ w
        hc *= (w.T @ cs) / (w_gram @ hc + epsilon)
        ho *= (w.T @ os) / (w_gram @ ho + epsilon)
        wt_base = w.T @ communication.base
        wt_delta = w.T @ communication.active_delta
        g_numerator = _scatter_g_numerator(wt_base, wt_delta, g, v, pairs, lr)
        g_denominator = torch.einsum("kj,jps,sr->kpr", w_gram, g, v.T @ v)
        g *= g_numerator / (g_denominator + epsilon)
        v_numerator = _v_numerator(wt_base, wt_delta, g, v, pairs, lr)
        activity_gram = torch.einsum("kj,kpr,jps->rs", w_gram, g, g)
        v *= v_numerator / (v @ activity_gram + epsilon)
        g, v = normalize_lr_programs(g, v, epsilon)
        dictionary_gram = c_weight * (hc @ hc.T) + i_weight * _ccc_gram(g, v) + o_weight * (ho @ ho.T)
        target = c_weight * (cs @ hc.T) + i_weight * _ccc_target(communication, g, v) + o_weight * (os @ ho.T)
        w *= target / (w @ dictionary_gram + epsilon)
        if (iteration + 1) % 25 == 0:
            scale = hc.sum(1).clamp_min(epsilon)
            w *= scale
            hc /= scale[:, None]
            ho /= scale[:, None]
            g /= scale[:, None, None]
        if iteration == 0 or (iteration + 1) % 25 == 0 or iteration + 1 == iterations:
            losses = {
                "composition": float((cs - w @ hc).square().mean()),
                "communication": float(exact_tensor_mse(communication, w, g, v)),
                "topology": float((os - w @ ho).square().mean())
            }
            history.append({"iteration": iteration + 1, **losses, "total": sum(losses.values())})
    seconds = time.perf_counter() - started
    scale = hc.sum(1).clamp_min(epsilon)
    w *= scale
    hc /= scale[:, None]
    ho /= scale[:, None]
    g /= scale[:, None, None]
    values = (w, hc, g, v, ho)
    if not all(torch.isfinite(value).all() and (value >= 0).all() for value in values):
        raise FloatingPointError("Tensor LR training produced invalid factors")
    return TensorLRResult(*(value.detach().cpu() for value in values), history, seed, seconds, seconds / iterations)


def infer_tensor_activity(composition, potential_chunks, hc, g, v, communication_scale, steps=50):
    k, pairs, _ = g.shape
    h_gram = _ccc_gram(g, v)
    gram = hc @ hc.T / composition.shape[1] + h_gram / (pairs * v.shape[0])
    target = composition @ hc.T / composition.shape[1]
    for start, potential in potential_chunks:
        width = potential.shape[2]
        h = torch.einsum("kpr,lr->kpl", g, v[start:start + width])
        target += (potential / communication_scale).reshape(potential.shape[0], -1) @ h.reshape(k, -1).T / (pairs * v.shape[0])
    step = 1 / gram.abs().sum(1).max().clamp_min(torch.finfo(gram.dtype).tiny)
    activity = torch.zeros_like(target)
    extrapolated = activity
    momentum = 1.0
    for _ in range(steps):
        updated = torch.relu(extrapolated - step * (extrapolated @ gram - target))
        next_momentum = (1 + math.sqrt(1 + 4 * momentum ** 2)) / 2
        extrapolated = updated + (momentum - 1) / next_momentum * (updated - activity)
        activity, momentum = updated, next_momentum
    return activity
