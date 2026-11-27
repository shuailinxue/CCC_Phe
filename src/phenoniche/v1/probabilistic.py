"""Independent hurdle dictionary and direct mixture of typed network prototypes."""
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .model import Batch


@dataclass(frozen=True)
class Config:
    rank: int = 12
    k: int = 4
    seed: int = 73
    batch: int = 32
    centers: int = 512
    epochs: int = 30
    min_epochs: int = 12
    patience: int = 3
    spacing: float = 50
    local_steps: int = 16
    niche_epochs: int = 24
    tune_epochs: int = 6
    local_ridge: float = .04

    def __post_init__(self):
        if min(self.rank, self.k, self.batch, self.centers, self.epochs,
               self.patience, self.local_steps, self.niche_epochs) < 1:
            raise ValueError("Positive rank, K, training budgets and patience are required")
        if not 0 <= self.min_epochs <= self.epochs or self.tune_epochs < 1:
            raise ValueError("Invalid epoch budget")
        if self.spacing < 0 or self.local_ridge <= 0:
            raise ValueError("Invalid spacing or local ridge")


class Dictionary(nn.Module):
    def __init__(self, base, support, positive, rank):
        super().__init__()
        self.base = nn.Parameter(base.clone())
        self.register_buffer("anchor", base)
        self.register_buffer("support", support)
        self.register_buffer("positive", positive)
        self.loading = nn.Parameter(torch.rand(2, rank, base.shape[-1], device=base.device) * .1 + .01)

    def weights(self):
        w = self.loading.clamp_min(0)
        return w / w.amax((0,2), keepdim=True).clamp_min(1e-8)

    def predict(self, activity, w=None):
        if w is None:
            w = self.weights()
        return self.base[:,None] + (activity - 1) @ w

    def penalty(self, w):
        return .15*w.mean() + .15*(self.base-self.anchor).square().mean()

    def features(self, batch, mask=None):
        """Outcome-independent, support- and sender-mass-weighted sufficient statistics."""
        valid = batch.s > 0
        if mask is not None:
            valid = valid & mask
        mass = batch.mass / batch.mass.sum(1, keepdim=True)
        mass = mass.repeat_interleave(batch.nt*batch.nl, dim=1)
        # Reliability of the training reference and opportunity count for this edge.
        support = batch.s * valid
        stability = self.support/(self.support+32)
        lr_p = (self.positive.reshape(-1,batch.nl).sum(0)+.5)/(
            self.support.reshape(-1,batch.nl).sum(0)+1)
        # Mild Fisher-information scaling, fixed for both detections and zeros.
        lr_weight = (lr_p/.01).clamp(.25,4).pow(-.25).repeat(batch.nt**2)
        w = mass * stability * lr_weight / support.clamp_min(1).sqrt() * valid
        wi = w * (self.positive/(self.positive+12))
        c = batch.c
        # One evidence budget per head. Square-root total support provides diminishing returns.
        detection = w * (32 / (support*w).sum(1,keepdim=True).clamp_min(1).sqrt())
        intensity = wi * (8 / (c*wi).sum(1,keepdim=True).clamp_min(1).sqrt())
        return support*detection, c*detection, c*intensity, batch.logs*intensity

    def score(self, features, activity, w=None):
        """Each candidate receives actual observation log gain against the fixed pooled reference."""
        h, m = self.predict(activity,w)
        h0,m0 = self.anchor
        s,c,n,l = features
        occ = c @ (h-h0).T - s @ (F.softplus(h)-F.softplus(h0)).T
        intensity = l @ (m-m0).T - .5 * (n @ (m.square()-m0.square()).T)
        return occ,intensity

    def local_loss(self, features, activity, w):
        h,m = self.predict(activity,w)
        h0,m0 = self.anchor
        s,c,n,l = features
        occ = (s*(F.softplus(h)-F.softplus(h0))-c*(h-h0)).sum(1)
        intensity = (.5*n*(m.square()-m0.square())-l*(m-m0)).sum(1)
        return occ,intensity


@torch.no_grad()
def infer_activity(model, features, cfg, initial=None, steps=None):
    """Projected accelerated gradient, with a per-center bound on local curvature."""
    s,c,n,l = features
    w = model.weights()
    ww = w * w.sum(1,keepdim=True)
    curvature = (.25*(s @ ww[0].T) + n @ ww[1].T).amax(1,keepdim=True) + cfg.local_ridge
    offset = c @ w[0].T + l @ w[1].T
    a = s.new_ones((len(s),cfg.rank)) if initial is None else initial
    extrapolated = a
    for j in range(cfg.local_steps if steps is None else steps):
        h,m = model.predict(extrapolated,w)
        gradient = (s*h.sigmoid()) @ w[0].T + (n*m) @ w[1].T - offset
        gradient += cfg.local_ridge*(extrapolated-1)
        update = (extrapolated-gradient/curvature).clamp_min(0)
        extrapolated = update + j/(j+3)*(update-a)
        a = update
    return a


class Niches(nn.Module):
    def __init__(self,k,nt,rank,mode,temperature):
        super().__init__()
        if mode not in ("joint","composition","communication"):
            raise ValueError(mode)
        self.pi = nn.Parameter(torch.zeros(k,nt))
        self.activity = nn.Parameter(torch.ones(k,rank))
        self.omega = nn.Parameter(torch.zeros(k))
        self.mode = mode
        self.temperature = temperature

    def evidence(self, mass, dictionary, features, uniform=False):
        n = mass.sum(1,keepdim=True)
        comp = (mass @ self.pi.log_softmax(1).T) * (16/n.clamp_min(16))
        occ = inten = torch.zeros_like(comp)
        if self.mode != "composition":
            occ,inten = dictionary.score(features,self.activity.clamp_min(0))
        if self.mode == "communication":
            comp = torch.zeros_like(comp)
        prior = self.omega.log_softmax(0)
        if uniform:
            prior = torch.full_like(prior,-np.log(len(prior)))
        return comp, self.temperature*occ, self.temperature*inten, prior

    def loss(self, mass, dictionary, features, uniform=False):
        terms = self.evidence(mass,dictionary,features,uniform)
        score = sum(terms)
        ridge = .025*(self.activity-1).square().mean()
        balance = -.035*self.omega.log_softmax(0).mean()
        return -score.logsumexp(1).mean() + ridge + balance, terms


def pooled(data,ids,device):
    support = data.support[ids].sum(0)[...,data.group].ravel()
    _,edge,value = data.entries(ids)
    positive = np.bincount(edge,weights=value[:,0],minlength=len(support))
    logs = np.bincount(edge,weights=value[:,1],minlength=len(support))
    rp = (positive.sum()+.5)/(support.sum()+1)
    rm = logs.sum()/max(positive.sum(),1)
    p = (positive+16*rp)/(support+16)
    base = np.stack([np.log(p)-np.log1p(-p),(logs+12*rm)/(positive+12)])
    return [torch.as_tensor(x,dtype=torch.float32,device=device) for x in (base,support,positive)]


def batched(ids,size):
    for start in range(0,len(ids),size):
        yield ids[start:start+size]


def batch_data(data,ids,device):
    return Batch(data,ids,device)
