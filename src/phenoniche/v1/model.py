"""MAP mixture of multinomial nodes and hurdle-network observations."""
from dataclasses import dataclass
import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class Settings:
    k: int = 4
    rank: int = 12
    epochs: int = 18
    batch: int = 48
    centers_per_epoch: int = 768
    spacing: float = 50.
    learning_rate: float = .025
    local_rate: float = .15
    local_steps: int = 10
    sigma_x: float = 1.
    sigma_a: float = 2.
    mass_cap: float = 16.
    support_cap: float = 8.
    sparsity: float = .02
    shrinkage: float = .05
    scale: str = "calibrated"
    mode: str = "joint"
    seed: int = 73

    def __post_init__(self):
        for name in ("k","rank","epochs","batch","centers_per_epoch","local_steps"):
            if type(getattr(self,name)) is not int or getattr(self,name) < 1:
                raise ValueError(f"Invalid {name}")
        for name in ("spacing","learning_rate","local_rate","sigma_x","sigma_a",
                     "mass_cap","support_cap","sparsity","shrinkage"):
            v = getattr(self,name)
            if not math.isfinite(v) or v < 0 or (v == 0 and name not in ("spacing","sparsity","shrinkage")):
                raise ValueError(f"Invalid {name}")
        if self.mode not in {"joint","composition","communication"} or self.scale not in {"calibrated","raw"}:
            raise ValueError("Invalid mode or evidence scale")


class Batch:
    def __init__(self, data, ids, device):
        ten = lambda x: torch.as_tensor(x, device=device)
        self.mass = ten(data.mass[ids]).float()
        self.n = len(ids)
        self.nt = data.nt
        self.nl = data.nl
        self.s = ten(data.support[ids])[...,ten(data.group)].reshape(self.n,-1).float()
        r,e,v = data.entries(ids)
        flat = ten(r * self.s.shape[1] + e)
        stats = torch.zeros((self.s.numel(),3),device=device)
        stats[flat] = ten(v[:,:3])
        self.c,self.logs,self.squares = stats.reshape(self.n,-1,3).unbind(-1)

    def weights(self, cfg, mask=None):
        s,c = self.s,self.c
        valid = (s > 0) if mask is None else ((s > 0) & mask)
        if cfg.scale == "raw":
            return valid, valid
        def calibrated(count):
            observed = ((count > 0) & valid).reshape(self.n,self.nt,-1)
            per_sender = observed.sum(-1)
            denom = per_sender * (per_sender > 0).sum(-1,keepdim=True)
            weight = (observed / denom.clamp_min(1)[...,None]).reshape_as(count)
            return weight * (cfg.support_cap / count.clamp_min(1)).clamp_max(1)
        return calibrated(s),calibrated(c)


def hurdle(batch, logits, mu, weights, sigma):
    occ = batch.s * F.softplus(logits) - batch.c * logits
    # Sums of squared residuals of positive sender-level log activities.
    rss = batch.squares - 2 * mu * batch.logs + batch.c * mu.square()
    intensity = rss / (2*sigma*sigma) + batch.c * math.log(sigma*math.sqrt(2*math.pi))
    return occ*weights[0],intensity*weights[1]


class PhenoMap(nn.Module):
    def __init__(self,cfg,nt,nl):
        super().__init__()
        self.cfg = cfg
        d = nt*nt*nl
        self.dictionary = nn.Parameter(torch.rand(2,cfg.rank,d)*.15)
        self.base = nn.Parameter(torch.zeros(2,d))
        self.register_buffer("anchor",torch.zeros(2,d))
        self.pi = nn.Parameter(torch.zeros(cfg.k,nt))
        self.eta = nn.Parameter(torch.zeros(cfg.k,cfg.rank))
        self.omega = nn.Parameter(torch.zeros(cfg.k))

    def weights(self):
        w = F.relu(self.dictionary)
        return w / w.amax(dim=(0,2),keepdim=True).clamp_min(1.e-8)

    def predict(self,u,w):
        a = F.softplus(u)
        return self.base[0] + a @ w[0], self.base[1] + a @ w[1]

    def evidence(self,mass,u):
        cfg = self.cfg
        comp = mass @ self.pi.log_softmax(1).T
        if cfg.scale == "calibrated":
            comp = comp * (cfg.mass_cap / mass.sum(1,keepdim=True)).clamp_max(1)
        latent = -(u[:,None] - self.eta).square().sum(-1) / (2*cfg.sigma_a**2)
        if cfg.mode == "communication":
            comp = torch.zeros_like(comp)
        if cfg.mode == "composition":
            latent = torch.zeros_like(latent)
        return comp, latent, self.omega.log_softmax(0)

    def objective(self,batch,u,w,weights,regularize=False):
        comp,latent,prior = self.evidence(batch.mass,u)
        mix = comp + latent + prior
        soft = mix.softmax(1)
        occ = intensity = comp.new_zeros(())
        reg = comp.new_zeros(())
        if self.cfg.mode != "composition":
            occ_i,int_i = hurdle(batch,*self.predict(u,w),weights,self.cfg.sigma_x)
            occ,intensity = occ_i.sum(1).mean(),int_i.sum(1).mean()
            if regularize:
                reg = self.cfg.sparsity*w.mean() + self.cfg.shrinkage*(self.base-self.anchor).square().mean()
        loss = occ + intensity - mix.logsumexp(1).mean() + reg
        parts = dict(composition=-(soft*comp).sum(1).mean(), latent=-(soft*latent).sum(1).mean(),
                     prior=-(soft*prior).sum(1).mean(), entropy=(soft*soft.clamp_min(1e-30).log()).sum(1).mean(),
                     occurrence=occ,intensity=intensity,regularization=reg,total=loss)
        return loss,parts

    @torch.no_grad()
    def start(self,batch,w,weights):
        """Score each prototype using observed edges before local MAP refinement."""
        cfg = self.cfg
        comp,_,prior = self.evidence(batch.mass, self.eta[:1].expand(batch.n,-1))
        scores = comp + prior
        if cfg.mode != "composition":
            logits,mu = self.predict(self.eta,w)
            scores -= (batch.s*weights[0]) @ F.softplus(logits).T - (batch.c*weights[0]) @ logits.T
            # Only candidate-dependent terms of the positive log-intensity NLL.
            scores -= ((batch.c*weights[1]) @ mu.square().T -
                       2*(batch.logs*weights[1]) @ mu.T) / (2*cfg.sigma_x**2)
        return self.eta[scores.argmax(1)].clone()


@torch.no_grad()
def local_fit(model,batch,w,weights,steps,initial=None):
    """Adam on the exact local MAP gradient with globals fixed."""
    u = model.start(batch,w,weights) if initial is None else initial.detach().clone()
    cfg = model.cfg
    if cfg.mode == "composition":
        return u
    opt = torch.optim.Adam([u],lr=cfg.local_rate)
    # The positive-intensity score is affine in a. Cache its Hessian-vector
    # coefficients implicitly; no cell-by-edge tensors are saved to disk.
    support = batch.s*weights[0]
    positives = batch.c*weights[1]/cfg.sigma_x**2
    offset = (batch.c*weights[0]) @ w[0].T + (batch.logs*weights[1]) @ w[1].T / cfg.sigma_x**2
    for _ in range(steps):
        logits,mu = model.predict(u,w)
        comp,latent,prior = model.evidence(batch.mass,u)
        soft = (comp+latent+prior).softmax(1)
        score_a = (logits.sigmoid()*support) @ w[0].T + (mu*positives) @ w[1].T - offset
        u.grad = (u.sigmoid()*score_a + (u-soft@model.eta)/cfg.sigma_a**2) / batch.n
        opt.step()
    return u
