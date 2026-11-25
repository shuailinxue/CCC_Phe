"""Direct nonnegative factors with matched CCC/composition decoder constraints.

This is a joint-factorization control, not the complete SpaNiche algorithm.
Z is fitted per observed microenvironment; P is its row-normalized version.
No graph, encoder or phenotype labels are used in this control.
"""
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from sklearn.decomposition import NMF
from threadpoolctl import threadpool_limits

from ..shared.objectives import ccc_loss, composition_loss
from .regularization import regularization


@dataclass(frozen=True)
class JointNMFConfig:
    programs: int = 32
    composition_weight: float = 1.0
    regularization_weight: float = 0.005
    epochs: int = 800
    learning_rate: float = 0.01
    dictionary_lr: float = 0.00005
    initialization_iterations: int = 200
    ccc_nonzero_weight_max: float = 1.0
    seed: int = 42
    threads: int = 4
    device: str = 'cpu'


def _inverse_softplus(value):
    return value + torch.log(-torch.expm1(-value))


class JointNMF(nn.Module):
    def __init__(self, z, h_scaled, b, feature_scale):
        super().__init__()
        z = torch.as_tensor(z, dtype=torch.float32)
        # Separate numerical units from free activities, without changing Z@H.
        units = z.mean(0).clamp_min(1e-4)
        self.register_buffer('activity_units', units)
        self.register_buffer('feature_scale', torch.as_tensor(feature_scale, dtype=torch.float32))
        self.raw_activity = nn.Parameter(_inverse_softplus((z/units).clamp_min(1e-4)))
        h = torch.as_tensor(h_scaled, dtype=torch.float32).clamp_min(0)
        self.raw_dictionary = nn.Parameter(h/h.sum(1, keepdim=True).clamp_min(1e-8))
        self.raw_composition = nn.Parameter(torch.as_tensor(b, dtype=torch.float32).clamp_min(1e-8).log())

    def forward(self):
        z = F.softplus(self.raw_activity)*self.activity_units
        p = z/z.sum(1, keepdim=True).clamp_min(1e-8)
        h = self.raw_dictionary.clamp_min(0)*self.feature_scale
        h = h/h.sum(1, keepdim=True).clamp_min(1e-8)
        b = self.raw_composition.softmax(1)
        return dict(Z=z, P=p, H=h, B=b, reconstruction=z@h, composition=p@b)

    @torch.no_grad()
    def project(self):
        self.raw_dictionary.clamp_(min=0)
        empty = self.raw_dictionary.sum(1) <= 1e-12
        self.raw_dictionary[empty] = 1/self.raw_dictionary.shape[1]
        self.raw_dictionary.div_(self.raw_dictionary.sum(1, keepdim=True))


def initialize_factors(x, c, config, feature_scale=None):
    scale = (np.sqrt(np.mean(x*x, axis=0)).clip(1e-4) if feature_scale is None
             else np.asarray(feature_scale)).astype(np.float32)
    if scale.shape != (x.shape[1],) or not np.isfinite(scale).all() or (scale <= 0).any():
        raise ValueError('Invalid feature scale')
    with threadpool_limits(limits=config.threads):
        nmf = NMF(n_components=config.programs, init='nndsvda', random_state=config.seed,
                  max_iter=config.initialization_iterations)
        w = nmf.fit_transform(x/scale)
    h = nmf.components_ + 1e-8
    z = w*(h*scale).sum(1)
    p = z/z.sum(1, keepdims=True).clip(1e-8)
    b = (p.T@c)/p.sum(0)[:, None].clip(1e-8)
    return dict(z=z, h_scaled=h, b=b, feature_scale=scale)


def fit_joint_nmf(x, c, config=JointNMFConfig(), initialization=None):
    """Fit both reconstruction targets using one free nonnegative activity matrix.

    CCC-only control uses composition_weight=0 with identical initialization.
    Uses all input rows; training error is not a held-out predictive assessment.
    """
    x = np.asarray(x, dtype=np.float32)
    c = np.asarray(c, dtype=np.float32)
    if x.ndim != 2 or c.ndim != 2 or len(x) != len(c):
        raise ValueError('Target dimensions do not match')
    if not np.isfinite(x).all() or not np.isfinite(c).all() or (x < 0).any() or (c < 0).any():
        raise ValueError('Targets must be finite and nonnegative')
    if not np.allclose(c.sum(1), 1, atol=1e-5):
        raise ValueError('C must contain row-normalized composition')
    if not 2 <= config.programs <= min(x.shape) or config.epochs < 1:
        raise ValueError('Invalid rank or epoch budget')
    if config.composition_weight < 0 or config.regularization_weight < 0:
        raise ValueError('Loss weights must be nonnegative')
    torch.set_num_threads(config.threads)
    torch.manual_seed(config.seed)
    if initialization is None:
        initialization = initialize_factors(x, c, config)
    model = JointNMF(**initialization).to(config.device)
    target = torch.as_tensor(x, device=config.device)
    composition = torch.as_tensor(c, device=config.device)
    mask = torch.ones(x.shape[1], dtype=torch.bool, device=config.device)
    nonzero_fraction = (x > 0).mean(0)
    optimizer = torch.optim.Adam([
        {'params': [model.raw_activity, model.raw_composition], 'lr': config.learning_rate},
        {'params': [model.raw_dictionary], 'lr': config.dictionary_lr},
    ])
    history = []
    best = float('inf')
    best_state = None
    best_epoch = 0
    for epoch in range(config.epochs+1):
        result = model()
        primary = ccc_loss(result['reconstruction'], target, model.feature_scale, mask,
                           nonzero_fraction=nonzero_fraction,
                           nonzero_weight_max=config.ccc_nonzero_weight_max)
        auxiliary = composition_loss(result['composition'], composition)
        penalty = regularization(result)
        total = primary + config.composition_weight*auxiliary + config.regularization_weight*penalty
        if not torch.isfinite(total):
            raise FloatingPointError('Nonfinite joint NMF objective')
        record = dict(epoch=epoch, ccc=float(primary.detach()), composition=float(auxiliary.detach()),
                      regularization=float(penalty.detach()), total=float(total.detach()))
        history.append(record)
        if record['total'] < best:
            best, best_epoch = record['total'], epoch
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
        if epoch % 100 == 0 or epoch == config.epochs:
            print(f'joint-NMF lambda={config.composition_weight:g} {epoch}/{config.epochs}: '
                  f'CCC={record["ccc"]:.5f}, composition={record["composition"]:.5f}', flush=True)
        if epoch == config.epochs:
            break
        optimizer.zero_grad(set_to_none=True)
        total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
        optimizer.step()
        model.project()
    model.load_state_dict(best_state)
    with torch.no_grad():
        result = {k: v.cpu().numpy() for k, v in model().items()}
    result.update(history=history, best_epoch=best_epoch, config=asdict(config))
    return result


def reconstruction_metrics(x, c, result, scale):
    """One evaluation definition for saved GNN and direct factorization."""
    xhat = result['Z']@result['H']
    chat = result['P']@result['B']
    error = (xhat-x)/scale
    baseline = (x.mean(0)-x)/scale
    mse, denominator = float(np.mean(error**2)), float(np.mean(baseline**2))
    return dict(ccc_scaled_mse=mse, mean_scaled_mse=denominator,
                fraction_mse_removed=1-mse/denominator,
                ccc_huber=float(np.where(abs(error) <= 1, .5*error**2, abs(error)-.5).mean()),
                composition_KL=float(np.mean(np.sum(c*np.log(c.clip(1e-8)/chat.clip(1e-8)), axis=1))))
