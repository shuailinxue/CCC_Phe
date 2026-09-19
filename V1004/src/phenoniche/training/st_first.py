from dataclasses import dataclass
import math
import torch
from phenoniche.utils.seed import set_seed


@dataclass(frozen=True)
class StructuredScaling:
    communication: float
    topology: float

    @classmethod
    def fit(cls, cb_train, ib_train, cs, os):
        rms = lambda value: float(value.square().mean().sqrt())
        composition_scale = rms(cb_train)
        communication = rms(ib_train) / composition_scale
        topology = rms(os) / rms(cs)
        if not all(math.isfinite(value) and value > 0 for value in (communication, topology)):
            raise ValueError("Structured block scaling requires positive finite blocks")
        return cls(communication, topology)

    def bulk(self, cb, ib):
        return cb, ib / self.communication

    def spatial(self, cs, is_, os):
        return cs, is_ / self.communication, os / self.topology


@dataclass(frozen=True)
class STFirstConfig:
    niches: int = 6
    iterations: int = 500
    seed: int = 31
    epsilon: float = 1e-8

    def __post_init__(self):
        if self.niches != 6 or not isinstance(self.iterations, int) or self.iterations < 1:
            raise ValueError("ST-first primary requires K=6 and positive iterations")
        if not math.isfinite(self.epsilon) or self.epsilon <= 0:
            raise ValueError("epsilon must be positive and finite")


@dataclass
class STFirstResult:
    WS: torch.Tensor
    HC: torch.Tensor
    HI: torch.Tensor
    HO: torch.Tensor
    history: list[dict]
    seed: int


def _weighted_blocks(cs, is_, os):
    scales = (cs.shape[1] ** -0.5, is_.shape[1] ** -0.5, os.shape[1] ** -0.5)
    return torch.cat((cs * scales[0], is_ * scales[1], os * scales[2]), dim=1), scales


def _normalize_solution(w, h, composition_features, epsilon):
    scale = h[:, :composition_features].sum(1).clamp_min(epsilon)
    return w * scale, h / scale[:, None]


def fit_st_only(cs, is_, os, config=None):
    config = config or STFirstConfig()
    if cs.ndim != 2 or is_.ndim != 2 or os.ndim != 2 or len({cs.shape[0], is_.shape[0], os.shape[0]}) != 1:
        raise ValueError("ST-only requires aligned rank-2 spatial blocks")
    if any((value < 0).any() or not torch.isfinite(value).all() for value in (cs, is_, os)):
        raise ValueError("ST-only blocks must be finite and nonnegative")
    set_seed(config.seed)
    x, scales = _weighted_blocks(cs, is_, os)
    generator = torch.Generator(device=x.device).manual_seed(config.seed)
    w = torch.rand((x.shape[0], config.niches), generator=generator, device=x.device, dtype=x.dtype) + 0.2
    h = torch.rand((config.niches, x.shape[1]), generator=generator, device=x.device, dtype=x.dtype) + 0.2
    mean = x.mean().clamp_min(config.epsilon).sqrt()
    w = w / w.mean() * mean
    h = h / h.mean() * mean
    history = []
    for iteration in range(config.iterations):
        h *= (w.T @ x) / ((w.T @ w) @ h + config.epsilon)
        w *= (x @ h.T) / (w @ (h @ h.T) + config.epsilon)
        if iteration == 0 or (iteration + 1) % 25 == 0 or iteration + 1 == config.iterations:
            residual = x - w @ h
            history.append({"iteration": iteration + 1, "weighted_mean_squared_error": float(residual.square().mean())})
    w, h = _normalize_solution(w, h, cs.shape[1], config.epsilon)
    c_end = cs.shape[1]
    i_end = c_end + is_.shape[1]
    hc = h[:, :c_end] / scales[0]
    hi = h[:, c_end:i_end] / scales[1]
    ho = h[:, i_end:] / scales[2]
    if not all(torch.isfinite(value).all() for value in (w, hc, hi, ho)):
        raise FloatingPointError("ST-only training produced nonfinite factors")
    return STFirstResult(w, hc, hi, ho, history, config.seed)


def infer_spatial_activities(cs, is_, os, hc, hi, ho, steps=200, learning_rate=1.0, create_graph=False):
    if not isinstance(steps, int) or steps < 1 or not math.isfinite(learning_rate) or not 0 < learning_rate <= 1:
        raise ValueError("Spatial inference requires positive steps and relative learning rate in (0,1]")
    blocks = ((cs, hc, 1 / cs.shape[1]), (is_, hi, 1 / is_.shape[1]), (os, ho, 1 / os.shape[1]))
    with torch.set_grad_enabled(torch.is_grad_enabled() and create_graph):
        gram = sum(weight * dictionary @ dictionary.T for _, dictionary, weight in blocks)
        target = sum(weight * observed @ dictionary.T for observed, dictionary, weight in blocks)
        lipschitz = gram.abs().sum(1).max().clamp_min(torch.finfo(gram.dtype).tiny)
        step = learning_rate / lipschitz
        activity = torch.zeros_like(target)
        extrapolated = activity
        momentum = 1.0
        for _ in range(steps):
            updated = torch.relu(extrapolated - step * (extrapolated @ gram - target))
            next_momentum = (1 + math.sqrt(1 + 4 * momentum ** 2)) / 2
            extrapolated = updated + ((momentum - 1) / next_momentum) * (updated - activity)
            activity, momentum = updated, next_momentum
    if not torch.isfinite(activity).all():
        raise FloatingPointError("Spatial inference produced nonfinite activities")
    return activity


def spatial_reconstruction(cs, is_, os, ws, hc, hi, ho):
    blocks = ((cs, ws @ hc), (is_, ws @ hi), (os, ws @ ho))
    values = [float((observed - predicted).square().mean()) for observed, predicted in blocks]
    return {"composition": values[0], "communication": values[1], "topology": values[2],
            "total": sum(values)}
