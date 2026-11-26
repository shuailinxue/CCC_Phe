"""Dynamic specimen-local sampling, joint fitting and fixed-global inference."""
import numpy as np
from scipy.spatial import cKDTree
from sklearn.cluster import KMeans
import torch
from torch.nn import functional as F

from .model import Batch, PhenoMap, local_fit


class Sampler:
    def __init__(self,data):
        self.ids = np.flatnonzero(data.train)
        self.groups = []
        for name in np.unique(data.samples[self.ids]):
            idx = self.ids[data.samples[self.ids] == name]
            self.groups.append((idx,cKDTree(data.xyz[idx])))
        self.visits = np.zeros(len(data.ids),np.int32)
        self.xyz = data.xyz

    def draw(self,rng,spacing,limit):
        # Random priority among the least visited centers, globally across specimens.
        order = self.ids[np.lexsort((rng.random(len(self.ids)),self.visits[self.ids]))]
        available = np.ones(len(self.xyz),bool)
        group_for = np.zeros(len(self.xyz),np.int32)
        for g,(idx,_) in enumerate(self.groups):
            group_for[idx] = g
        chosen = []
        for i in order:
            if available[i]:
                chosen.append(i)
                if len(chosen) >= limit:
                    break
                if spacing:
                    idx,tree = self.groups[group_for[i]]
                    available[idx[tree.query_ball_point(self.xyz[i],spacing)]] = False
        self.visits[chosen] += 1
        return np.array(chosen)


def initialize(model,data,ids,rng):
    """Empirical Bayes edge intercepts; exploratory summaries only seed joint Adam."""
    cfg = model.cfg
    t,l = data.nt,data.nl
    if len(ids) < cfg.k:
        raise ValueError("Fewer training centers than K")
    mass = data.mass[ids].astype(float)
    mass /= mass.sum(1,keepdims=True)
    profile = np.zeros((len(ids),l),np.float32)
    support = data.support[ids].sum((1,2))[:,data.group]
    row,edge,v = data.entries(ids)
    np.add.at(profile,(row,edge%l),v[:,3])
    profile /= np.maximum(support,1)
    def normalized(x):
        return (x-x.mean(0))/max(float(np.sqrt(x.var(0).sum())),1e-6)
    node,comm = normalized(np.sqrt(mass)),normalized(np.log1p(profile))
    init_features = {"composition":node,"communication":comm,
                     "joint":np.column_stack([node,comm])}[cfg.mode]
    clusters = KMeans(cfg.k,n_init=5,random_state=cfg.seed).fit_predict(init_features)
    programs = KMeans(min(cfg.rank,len(ids)),n_init=2,random_state=cfg.seed).fit_predict(comm)
    def summarize(mask):
        s = data.support[ids[mask]].sum(0)[...,data.group].ravel()
        keep = mask[row]
        c = np.bincount(edge[keep],weights=v[keep,0],minlength=len(s))
        logs = np.bincount(edge[keep],weights=v[keep,1],minlength=len(s))
        return s,c,logs
    s,c,logs = summarize(np.ones(len(ids),bool))
    p = (c.sum()+.5)/(s.sum()+1)
    m = logs.sum()/max(c.sum(),1)
    prob = (c+8*p)/(s+8)
    base = np.stack([np.log(prob)-np.log1p(-prob),(logs+8*m)/(c+8)])
    # Positive-LR mean intensity conditional on opportunity, shared reference mass.
    activity = np.bincount(row*t+edge//(t*l),weights=v[:,3]/np.maximum(
        data.support[ids[row],edge//(t*l),edge//l%t,data.group[edge%l]],1),minlength=len(ids)*t)
    tau = float(np.median(activity[activity>0])) if np.any(activity>0) else 1.
    with torch.no_grad():
        def assign(param,x):
            param.copy_(torch.as_tensor(x,dtype=param.dtype,device=param.device))
        assign(model.base,base)
        assign(model.anchor,base)
        for r in range(cfg.rank):
            sr,cr,lr = summarize(programs == r)
            pr = (cr+8*prob)/(sr+8)
            shifted = np.stack([np.log(pr)-np.log1p(-pr),(lr+8*base[1])/(cr+8)])-base
            w = np.clip(shifted,0,3) + rng.uniform(.002,.04,shifted.shape)
            assign(model.dictionary[:,r],w / w.max())
        for k in range(cfg.k):
            mask = clusters == k
            prop = data.mass[ids[mask]].sum(0)+.5
            assign(model.pi[k],np.log(prop/prop.sum()))
            freq = np.bincount(programs[mask],minlength=cfg.rank) / max(mask.sum(),1)
            assign(model.eta[k],-1 + 3 * freq/max(freq.max(),1e-6))
        assign(model.omega,np.log(np.bincount(clusters,minlength=cfg.k)+1))
    return tau


def train(data,cfg,device):
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    sampler = Sampler(data)
    model = PhenoMap(cfg,data.nt,data.nl).to(device)
    init = rng.choice(sampler.ids,min(len(sampler.ids),2048),replace=False)
    tau = initialize(model,data,init,rng)
    opt = torch.optim.Adam(model.parameters(),lr=cfg.learning_rate)
    history = []
    local = torch.full((len(data.ids),cfg.rank),float("nan"),device=device)
    for epoch in range(cfg.epochs):
        chosen = sampler.draw(rng,cfg.spacing,cfg.centers_per_epoch)
        totals = {}
        for start in range(0,len(chosen),cfg.batch):
            ids = chosen[start:start+cfg.batch]
            batch = Batch(data,ids,device)
            weights = batch.weights(cfg)
            w = model.weights()
            u0 = model.start(batch,w.detach(),weights)
            idx = torch.as_tensor(ids,device=device)
            previous = local[idx]
            u0 = torch.where(previous.isfinite(),previous,u0)
            u = local_fit(model,batch,w.detach(),weights,min(4,cfg.local_steps),u0)
            local[idx] = u
            opt.zero_grad()
            loss,parts = model.objective(batch,u,w,weights,regularize=True)
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite training objective")
            loss.backward()
            opt.step()
            for name,value in parts.items():
                totals[name] = totals.get(name,0.) + value.detach()*len(ids)
        history.append({name:float(v)/len(chosen) for name,v in totals.items()} |
                       dict(epoch=epoch+1,centers=len(chosen),unique_seen=int((sampler.visits>0).sum())))
    model.requires_grad_(False)
    return model,tau,history,sampler.visits


def infer(model,data,device):
    model.requires_grad_(False)
    w = model.weights()
    cfg = model.cfg
    result = {name:[] for name in ("u","soft","composition","latent")}
    for start in range(0,len(data.ids),cfg.batch):
        ids = np.arange(start,min(start+cfg.batch,len(data.ids)))
        b = Batch(data,ids,device)
        u = local_fit(model,b,w,b.weights(cfg),cfg.local_steps)
        comp,latent,prior = model.evidence(b.mass,u)
        for key,v in zip(result,(u,(comp+latent+prior).softmax(1),comp,latent),strict=True):
            result[key].append(v.detach().cpu().numpy())
    out = {key:np.concatenate(v) for key,v in result.items()}
    out["activity"] = np.logaddexp(0,out["u"])
    out["niche"] = out["soft"].argmax(1)
    return out
