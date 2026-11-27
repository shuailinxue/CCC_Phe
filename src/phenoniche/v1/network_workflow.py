"""Persist each stage and score frozen fits on an independent test region."""
from dataclasses import asdict
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit
import torch

from . import probabilistic, refit
from .evaluate import decode
from .probabilistic import Dictionary, Niches, batched, batch_data
from .storage import digest, key, lock, save_npz
from .workflow import checkpoint, runtime, write_cells


def save_checkpoint(path,**kwargs):
    tmp = path.with_suffix(".partial")
    torch.save(kwargs,tmp)
    os.replace(tmp,path)


def dictionary_from(state,cfg):
    m = Dictionary(state["anchor"],state["support"],state["positive"],cfg.rank)
    m.load_state_dict(state)
    return m.requires_grad_(False)


def niche_from(record,cfg,nt,device):
    model = Niches(cfg.k,nt,cfg.rank,record["mode"],record["temperature"]).to(device)
    model.load_state_dict(record["state"])
    return model


@torch.no_grad()
def export(data,model,niches,a,cfg,device,out):
    result = {k:[] for k in ("composition","occurrence_gain","intensity_gain","prior")}
    for ix in batched(np.arange(len(data.ids)),cfg.batch):
        b = batch_data(data,ix,device)
        terms = niches.evidence(b.mass,model,model.features(b))
        for name,term in zip(result,terms):
            result[name].append(term.expand(b.n,-1).cpu().numpy())
    result={k:np.concatenate(v) for k,v in result.items()}
    score=sum(result.values())
    pr=np.exp(score-score.max(1,keepdims=True))
    soft=pr/pr.sum(1,keepdims=True)
    labels=soft.argmax(1)
    write_cells(out,data,dict(**result,niche=labels,soft=soft,activity=a))
    w=model.weights().cpu().numpy()
    prop=niches.pi.softmax(1).cpu().numpy()
    h,m=model.predict(niches.activity).detach().cpu().numpy()
    save_npz(out/"programs.npz",occurrence=w[0],intensity=w[1],base=model.base.cpu().numpy(),
             reference=model.anchor.cpu().numpy(),support=model.support.cpu().numpy(),
             lr=data.lr,names=data.names)
    save_npz(out/"niches.npz",pi=prop,activity=niches.activity.cpu().numpy(),
             omega=niches.omega.softmax(0).cpu().numpy(),detection=expit(h),log_intensity=m)
    tables=[]
    for i in range(cfg.rank):
        magnitude=np.hypot(w[0,i],w[1,i])
        e=np.argsort(np.where(model.support.cpu().numpy()>0,magnitude,-1))[-20:][::-1]
        table=decode(e,data.names,data.lr)
        table["program"]=i
        table["occurrence"],table["intensity"]=w[:,i,e]
        tables.append(table)
    pd.concat(tables).to_csv(out/"top_edges.tsv",sep="\t",index=False)
    vector=w.transpose(1,0,2).reshape(cfg.rank,-1)
    norm=np.linalg.norm(vector,axis=1)
    vector/=np.maximum(norm[:,None],1e-12)
    cosine=vector@vector.T
    from scipy.spatial.distance import pdist
    return dict(sizes=np.bincount(labels,minlength=cfg.k).tolist(),activity_sd=a.std(0).tolist(),
                program_norm=norm.tolist(),maximum_program_cosine=float(cosine[np.triu_indices(cfg.rank,1)].max()) if cfg.rank>1 else None,
                min_composition_l1=float(pdist(prop,metric="cityblock").min()) if cfg.k>1 else None,
                min_activity_distance=float(pdist(niches.activity.cpu().numpy()).min()) if cfg.k>1 else None,
                evidence_change=float(np.mean(labels!=(result["composition"]+result["prior"]).argmax(1))))


@torch.no_grad()
def explain(data,idx,dictionary,niches,activity,device):
    """Return exact edge contributions to the winning candidate's log-score margin."""
    b=batch_data(data,np.array([idx]),device)
    s,c,n,l = dictionary.features(b)
    comp,occ,inten,prior=niches.evidence(b.mass,dictionary,(s,c,n,l))
    score=(comp+occ+inten+prior)[0]
    winner=int(score.argmax())
    other=comp[0].clone()
    other[winner]=-torch.inf
    competitor=int(other.argmax())
    h,m=dictionary.predict(niches.activity)
    edge_occ=(c*(h[winner]-h[competitor])-s*(torch.nn.functional.softplus(h[winner])-torch.nn.functional.softplus(h[competitor])))[0]
    edge_int=(l*(m[winner]-m[competitor])-.5*n*(m[winner].square()-m[competitor].square()))[0]
    weight=niches.temperature if niches.mode!="composition" else 0
    local_h,local_m=dictionary.predict(torch.as_tensor(activity,device=device)[None])
    table=decode(np.arange(b.s.shape[1]),data.names,data.lr)
    support=b.s[0].cpu().numpy()
    table["support"]=support
    table["positive_count"]=b.c[0].cpu().numpy()
    obs=np.full(len(table),np.nan)
    obs[support>0]=0
    _,e,v=data.entries(np.array([idx]))
    obs[e]=v[:,3]/support[e]
    table["observed_mean"]=obs
    table["predicted_detection"]=local_h[0].sigmoid().cpu().numpy()
    table["predicted_positive_log_mean"]=local_m[0].cpu().numpy()
    table["occurrence_margin"]=weight*edge_occ.cpu().numpy()
    table["intensity_margin"]=weight*edge_int.cpu().numpy()
    for j in range(len(activity)):
        table[f"program_{j}_occurrence"],table[f"program_{j}_intensity"]=(
            (activity[j]-1)*dictionary.weights()[:,j]).cpu().numpy()
    return dict(winner=winner,competitor=competitor,total_margin=float(score[winner]-score[competitor]),
                composition_margin=float(comp[0,winner]-comp[0,competitor]),
                prior_margin=float(prior[winner]-prior[competitor]),
                communication_margin=float((occ+inten)[0,winner]-(occ+inten)[0,competitor])),table


def fit_run(data,real,root,cfg,device,detailed=True,legacy_run=None):
    """Run one seed. The true target is used exclusively after model selection."""
    for name in ("ids","xyz","mass","support","group","scales","lr","names","train","test"):
        if not np.array_equal(getattr(data,name),getattr(real,name)):
            raise ValueError(f"Paired datasets disagree in {name}")
    meta = dict(settings=asdict(cfg),input=data.meta,real=real.meta,runtime=runtime(device),
                stages={Path(s).name:digest(s) for s in [__file__,probabilistic.__file__,refit.__file__]},
                detailed=detailed,legacy=digest(Path(legacy_run)/"model.pt") if legacy_run else None)
    out=Path(root)/("fit-"+key(meta))
    out.mkdir(parents=True,exist_ok=True)
    def progress(stage,record):
        print(out.name,stage,json.dumps(record),flush=True)
    with lock(out):
        if (out/"complete").exists():
            return out
        (out/"manifest.json").write_text(json.dumps(meta,indent=2))
        train,valid=refit.spatial_split(data)
        ids=np.arange(len(data.ids))
        valid_ids=np.flatnonzero(valid)
        valid_ids=np.sort(np.random.default_rng(204).choice(valid_ids,min(128,len(valid_ids)),replace=False))
        save_npz(out/"split.npz",ids=data.ids,train=train,validation=valid,validation_scored=valid_ids,test=data.test)
        if not (out/"dictionary.pt").exists():
            model,history,visits,best=refit.dictionary_fit(data,train,valid_ids,cfg,device,progress)
            pd.DataFrame(history).to_csv(out/"dictionary_loss.tsv",sep="\t",index=False)
            save_npz(out/"visits.npz",ids=data.ids,visits=visits,best_epoch=best)
            save_checkpoint(out/"dictionary.pt",state=model.state_dict())
        model=dictionary_from(torch.load(out/"dictionary.pt",map_location=device,weights_only=True)["state"],cfg)
        if not (out/"activity.npz").exists():
            save_npz(out/"activity.npz",ids=data.ids,activity=refit.activities(data,model,ids,cfg,device))
        with np.load(out/"activity.npz") as local:
            if not np.array_equal(local["ids"],data.ids):
                raise ValueError("Local activity cache ID mismatch")
            a=local["activity"]
        test=np.flatnonzero(real.test)
        scores={}
        scores["pooled"]=refit.prediction(real,test,model,cfg,device,baseline=True)
        scores["dictionary"]=refit.prediction(real,test,model,cfg,device)
        if legacy_run and detailed:
            legacy,_=checkpoint(legacy_run,device)
            scores["legacy_historical"]=refit.prediction(real,test,model,cfg,device,legacy=legacy)
        modes=("joint","communication","composition") if detailed else ("joint",)
        for mode in modes:
            run=out/mode
            run.mkdir(exist_ok=True)
            if not (run/"model.pt").exists():
                n,history=refit.niche_fit(data,model,train,valid_ids,a[train],mode,cfg,device,progress)
                (run/"optimization.json").write_text(json.dumps(history,indent=2))
                save_checkpoint(run/"model.pt",state=n.state_dict(),temperature=n.temperature,mode=mode)
            n=niche_from(torch.load(run/"model.pt",map_location=device,weights_only=True),cfg,data.nt,device)
            metrics=export(data,model,n,a,cfg,device,run)
            scores[mode]=refit.prediction(real,test,model,cfg,device,niches=n) if mode!="composition" else scores["pooled"]
            (run/"metrics.json").write_text(json.dumps(metrics,indent=2))
            if mode=="joint" and detailed:
                tune=out/"refined"
                tune.mkdir(exist_ok=True)
                if not (tune/"model.pt").exists():
                    updated,prototypes,decision=refit.joint_refine(data,model,n,train,valid_ids,cfg,device,progress)
                    (tune/"selection.json").write_text(json.dumps(decision,indent=2))
                    save_checkpoint(tune/"model.pt",dictionary=updated.state_dict(),state=prototypes.state_dict(),
                                    mode=mode,temperature=prototypes.temperature)
                rec=torch.load(tune/"model.pt",map_location=device,weights_only=True)
                updated=dictionary_from(rec["dictionary"],cfg)
                prototypes=niche_from(rec,cfg,data.nt,device)
                if not (tune/"activity.npz").exists():
                    save_npz(tune/"activity.npz",ids=data.ids,activity=refit.activities(data,updated,ids,cfg,device))
                with np.load(tune/"activity.npz") as local:
                    metrics=export(data,updated,prototypes,local["activity"],cfg,device,tune)
                (tune/"metrics.json").write_text(json.dumps(metrics,indent=2))
                scores["refined_dictionary"]=refit.prediction(real,test,updated,cfg,device)
                scores["refined"]=refit.prediction(real,test,updated,cfg,device,niches=prototypes)
        (out/"predictions.json").write_text(json.dumps(scores,indent=2))
        (out/"complete").touch()
    return out
