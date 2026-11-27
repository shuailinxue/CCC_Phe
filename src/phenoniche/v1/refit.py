"""Spatial validation, dictionary training and constrained network-mixture fitting."""
from copy import deepcopy
from dataclasses import replace

import numpy as np
from sklearn.cluster import KMeans
import torch
from torch.nn import functional as F

from .fit import Sampler
from .probabilistic import Dictionary, Niches, batched, batch_data, infer_activity, pooled


def spatial_split(data):
    train = data.train.copy()
    validation = np.zeros_like(train)
    for sample in np.unique(data.samples[train]):
        idx = np.flatnonzero(train & (data.samples==sample))
        cut = np.quantile(data.xyz[idx,0],.83)
        validation[idx] = data.xyz[idx,0] >= cut
        train[idx] = data.xyz[idx,0] < cut - 2*float(data.window)
    if not train.any() or not validation.any():
        raise ValueError("Need disjoint buffered training and validation regions")
    return train,validation


def hidden_edges(data,device):
    mask = np.random.default_rng(813).random(data.nt**2*data.nl)<.2
    if not mask.any():
        mask[0] = True
    return torch.tensor(mask,device=device)


@torch.no_grad()
def prediction(data,ids,dictionary,cfg,device,niches=None,legacy=None,baseline=False):
    mask = hidden_edges(data,device)
    sums = np.zeros(7)
    for idx in batched(ids,cfg.batch):
        b = batch_data(data,idx,device)
        if baseline:
            h,m = dictionary.anchor[:,mask].unbind()
            p = h.sigmoid()
        elif legacy is not None:
            from .model import local_fit
            w = legacy.weights()
            u = local_fit(legacy,b,w,b.weights(legacy.cfg,~mask),legacy.cfg.local_steps)
            h,m = (value[:,mask] for value in legacy.predict(u,w))
            p = h.sigmoid()
        else:
            features = dictionary.features(b,~mask)
            if niches is None:
                a = infer_activity(dictionary,features,cfg,steps=cfg.local_steps*2)
                h,m = dictionary.predict(a)
                p,m = h[:,mask].sigmoid(),m[:,mask]
            else:
                r = sum(niches.evidence(b.mass,dictionary,features)).softmax(1)
                h,m = dictionary.predict(niches.activity.clamp_min(0))
                probs = h[:,mask].sigmoid()
                p = r @ probs
                m = (r @ (probs*m[:,mask])) / p.clamp_min(1e-10)
        p = p.clamp(1e-7,1-1e-7)
        s,c,l,q = (x[:,mask] for x in (b.s,b.c,b.logs,b.squares))
        zero = s-c
        values = (s,c,-c*p.log(),-zero*torch.log1p(-p),q-2*m*l+c*m.square(),c*p,zero*p)
        sums += [float(x.double().sum()) for x in values]
    s,c,pos,neg,se,pp,pn = sums
    return dict(centers=len(ids),trials=int(s),positive=int(c),
                detection=(pos+neg)/max(s,1),positive_nll=pos/max(c,1),zero_nll=neg/max(s-c,1),
                rmse=float(np.sqrt(max(se,0)/max(c,1))),
                detected_mean=pp/max(c,1),zero_mean=pn/max(s-c,1))


def utility(pred, baseline):
    """Relative raw detection NLL and log-intensity MSE, equal heads."""
    return pred["detection"]/max(baseline["detection"],1e-8) + (
        pred["rmse"]/max(baseline["rmse"],1e-8))**2


def init_dictionary(data,train,cfg,device):
    torch.manual_seed(cfg.seed)
    base,s,c = pooled(data,train,device)
    model = Dictionary(base,s,c,cfg.rank)
    rng = np.random.default_rng(cfg.seed)
    ids = rng.choice(train,min(1536,len(train)),replace=False)
    row,edge,v = data.entries(ids)
    profile = np.zeros((len(ids),data.nl))
    np.add.at(profile,(row,edge%data.nl),v[:,3])
    profile /= data.support[ids].sum((1,2))[:,data.group].clip(1)
    group = KMeans(min(len(ids),cfg.rank),n_init=3,random_state=cfg.seed).fit_predict(profile)
    with torch.no_grad():
        for r in range(cfg.rank):
            if np.any(group==r):
                contrast = (pooled(data,ids[group==r],device)[0]-base).clamp(0,2.5)
                # Avoid initializing unsupported edges from differences between pooled backoffs.
                contrast *= (s > 0)
                w = contrast + torch.rand_like(contrast)*.025
                model.loading[:,r] = w/w.max().clamp_min(1e-8)
    return model


def sampler_for(data,train):
    from types import SimpleNamespace
    return Sampler(SimpleNamespace(train=train,ids=data.ids,xyz=data.xyz,samples=data.samples))


def dictionary_fit(data,train,valid,cfg,device,progress):
    ids = np.flatnonzero(train)
    model = init_dictionary(data,ids,cfg,device)
    sampler = sampler_for(data,train)
    rng = np.random.default_rng(cfg.seed)
    opt = torch.optim.Adam(model.parameters(),lr=.012)
    activity = torch.ones((len(data.ids),cfg.rank),device=device)
    baseline = prediction(data,valid,model,cfg,device,baseline=True)
    best,wait = None,0
    history = []
    for epoch in range(1,cfg.epochs+1):
        selected = sampler.draw(rng,cfg.spacing,cfg.centers)
        measured = []
        for ix in batched(selected,cfg.batch):
            b = batch_data(data,ix,device)
            f = model.features(b)
            a = infer_activity(model,f,cfg,activity[ix],steps=6)
            activity[ix] = a
            w = model.weights()
            occ,inten = model.local_loss(f,a,w)
            loss = (occ+inten).mean() + model.penalty(w)
            opt.zero_grad()
            loss.backward()
            opt.step()
            measured.append([float(loss.detach()),float(occ.mean().detach()),float(inten.mean().detach())])
        record = dict(epoch=epoch,visited=int(np.count_nonzero(sampler.visits)),
                      **dict(zip(("objective","occurrence","intensity"),np.mean(measured,0))))
        if epoch%3==0 or epoch==cfg.epochs:
            val = prediction(data,valid,model,cfg,device)
            score = utility(val,baseline)
            record.update(valid_utility=score,valid_detection=val["detection"],valid_rmse=val["rmse"])
            if best is None or score < best[0]-1e-4:
                best = score,epoch,deepcopy(model.state_dict())
                wait = 0
            else:
                wait += 1
            progress("dictionary",record)
        history.append(record)
        if epoch>=cfg.min_epochs and sampler.visits[train].min()>=2 and wait>=cfg.patience:
            break
    model.load_state_dict(best[2])
    return model,history,sampler.visits,best[1]


@torch.no_grad()
def activities(data,model,ids,cfg,device):
    out = []
    for idx in batched(ids,cfg.batch):
        b = batch_data(data,idx,device)
        out.append(infer_activity(model,model.features(b),cfg,steps=2*cfg.local_steps).cpu().numpy())
    return np.concatenate(out)


def initialize_niches(data,ids,a,mode,temperature,cfg,device,seed):
    mass = data.mass[ids].astype(float)
    prop = np.sqrt(mass/mass.sum(1,keepdims=True))
    def norm(x):
        x = x-x.mean(0)
        return x/max(np.sqrt(x.var(0).sum()),1e-8)
    feat = {"composition":norm(prop),"communication":norm(np.log1p(a)),
            "joint":np.column_stack([norm(prop),norm(np.log1p(a))])}[mode]
    labels = KMeans(cfg.k,n_init=4,random_state=seed).fit_predict(feat)
    model = Niches(cfg.k,data.nt,cfg.rank,mode,temperature).to(device)
    with torch.no_grad():
        for k in range(cfg.k):
            selected = labels==k
            if selected.any():
                model.activity[k].copy_(torch.as_tensor(a[selected].mean(0),device=device))
                model.pi[k].copy_(torch.as_tensor(np.log(mass[selected].sum(0)+.5),device=device))
    return model


@torch.no_grad()
def niche_validation(data,valid,dictionary,model,cfg,device):
    # Compare temperatures on unseen centers and unseen edges with an unweighted prediction score.
    pred = prediction(data,valid,dictionary,cfg,device,niches=model)
    return pred


def niche_fit(data,dictionary,train,valid,a,mode,cfg,device,progress):
    ids = np.flatnonzero(train)
    baseline = prediction(data,valid,dictionary,cfg,device,baseline=True)
    best = None
    summaries = []
    dictionary.requires_grad_(False)
    for i,temp in enumerate((.5,1.) if mode=="joint" else (1.,)):
        model = initialize_niches(data,ids,a,mode,temp,cfg,device,cfg.seed+i)
        opt = torch.optim.Adam(model.parameters(),lr=.04)
        rng = np.random.default_rng(cfg.seed)
        history = []
        for epoch in range(cfg.niche_epochs):
            values = []
            for ix in batched(rng.choice(ids,min(cfg.centers,len(ids)),replace=False),cfg.batch):
                b = batch_data(data,ix,device)
                f = dictionary.features(b) if mode!="composition" else None
                loss,terms = model.loss(b.mass,dictionary,f,uniform=epoch<3)
                opt.zero_grad()
                loss.backward()
                opt.step()
                with torch.no_grad():
                    model.activity.clamp_min_(0)
                    soft = sum(terms).softmax(1)
                    values.append([float(loss)]+[float((x*soft).sum(1).mean()) for x in terms])
            history.append(dict(epoch=epoch+1,**dict(zip(
                ("objective","composition","occurrence_gain","intensity_gain","log_prior"),np.mean(values,0)))))
        val = niche_validation(data,valid,dictionary,model,cfg,device) if mode!="composition" else None
        score = utility(val,baseline) if val else 0.
        result = dict(mode=mode,temperature=temp,utility=score,validation=val,history=history)
        progress("niches",{k:v for k,v in result.items() if k!="history"})
        summaries.append(result)
        if best is None or score<best[0]:
            best = score,model
    return best[1],summaries


def joint_refine(data,model,niches,train,valid,cfg,device,progress):
    original, original_niche = model,niches
    model,niches = deepcopy(model),deepcopy(niches)
    model.requires_grad_(True)
    w0,base0 = model.weights().detach(),model.base.detach().clone()
    opt = torch.optim.Adam([{"params":model.parameters(),"lr":.001},
                            {"params":niches.parameters(),"lr":.008}])
    rng = np.random.default_rng(cfg.seed+100)
    sampler = sampler_for(data,train)
    history = []
    for epoch in range(cfg.tune_epochs):
        values = []
        for ix in batched(sampler.draw(rng,cfg.spacing,cfg.centers),cfg.batch):
            b = batch_data(data,ix,device)
            f = model.features(b)
            a = infer_activity(model,f,cfg)
            w = model.weights()
            occ,inten = model.local_loss(f,a,w)
            mixture,_ = niches.loss(b.mass,model,f)
            anchor = 3*(w-w0).square().mean()+2*(model.base-base0).square().mean()
            loss = (occ+inten).mean()+.15*mixture+model.penalty(w)+anchor
            opt.zero_grad()
            loss.backward()
            opt.step()
            with torch.no_grad():
                niches.activity.clamp_min_(0)
            values.append(float(loss.detach()))
        history.append(float(np.mean(values)))
    baseline = prediction(data,valid,original,cfg,device,baseline=True)
    original_local = utility(prediction(data,valid,original,cfg,device),baseline)
    new_local = utility(prediction(data,valid,model,cfg,device),baseline)
    original_mix = utility(prediction(data,valid,original,cfg,device,niches=original_niche),baseline)
    new_mix = utility(prediction(data,valid,model,cfg,device,niches=niches),baseline)
    def vectors(w):
        return F.normalize(w.transpose(0,1).reshape(cfg.rank,-1),dim=1)
    cosine = float((vectors(w0)*vectors(model.weights().detach())).sum(1).min())
    record = dict(history=history,old_local=original_local,new_local=new_local,old_mix=original_mix,
                  new_mix=new_mix,min_program_cosine=cosine,
                  accept=bool(new_local<=original_local and new_mix<=original_mix and cosine>=.90))
    progress("refinement",record)
    return model,niches,record
