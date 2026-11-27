"""Sequential validation-only dictionary and niche search."""
from dataclasses import asdict, replace
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score
import torch
from torch.nn import functional as F

from .network_workflow import export, save_checkpoint
from .polish import Options, StructuredDictionary, biological_graph
from .polish_metrics import benchmark, gradient_parts, utility
from .probabilistic import Config, batched, batch_data, infer_activity
from .refit import activities, init_dictionary, initialize_niches, sampler_for, spatial_split
from .storage import digest, key, lock, save_npz
from .workflow import runtime


KINDS=("random","grouped")


def load_dictionary(path,cfg,opt,device):
    state=torch.load(path,map_location=device,weights_only=True)["state"]
    model=StructuredDictionary(state["anchor"],state["support"],state["positive"],cfg.rank,opt)
    model.load_state_dict(state)
    return model.requires_grad_(False)


def dictionary_fit(data,out,cfg,opt,graph,device,train,valid,baseline):
    rng=np.random.default_rng(cfg.seed)
    start=init_dictionary(data,np.flatnonzero(train),cfg,device)
    model=StructuredDictionary(start.anchor,start.support,start.positive,cfg.rank,opt,graph)
    model.load_state_dict(start.state_dict())
    optimizer=torch.optim.Adam(model.parameters(),lr=.012)
    local=torch.ones((len(data.ids),cfg.rank),device=device)
    sampler=sampler_for(data,train)
    best=None
    history=[]
    stale=0
    for epoch in range(1,cfg.epochs+1):
        selected=sampler.draw(rng,cfg.spacing,cfg.centers)
        loss_sum=[]
        for ix in batched(selected,cfg.batch):
            batch=batch_data(data,ix,device)
            features=model.features(batch)
            a=infer_activity(model,features,cfg,local[ix],steps=6)
            local[ix]=a
            w=model.weights()
            lo,li=model.local_loss(features,a,w)
            lr,div=model.regularizers(w)
            penalty=.15*w.mean()+.15*(model.base-model.anchor).square().mean()
            loss=(lo+li).mean()+penalty+opt.lr_strength*lr+opt.diversity*div
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_sum.append([float(x.detach()) for x in (loss,lo.mean(),li.mean(),lr,div)])
        row=dict(epoch=epoch,coverage=int((sampler.visits>0).sum()),
                 **dict(zip(("loss","occurrence","intensity","lr","diversity"),np.mean(loss_sum,0))))
        if epoch%6==0 or epoch==cfg.epochs:
            metrics=[benchmark(data,valid,model,cfg,device,k) for k in KINDS]
            row["utility"]=utility(metrics,baseline)
            print(out.name,cfg.seed,epoch,round(row["utility"],5),flush=True)
            if best is None or row["utility"]<best[0]-1e-4:
                best=row["utility"],epoch,deepcopy(model.state_dict())
                stale=0
            else:
                stale+=1
        history.append(row)
        if epoch>=cfg.min_epochs and sampler.visits[train].min()>=2 and stale>=2:
            break
    pd.DataFrame(history).to_csv(out/"dictionary_loss.tsv",sep="\t",index=False)
    save_npz(out/"visits.npz",ids=data.ids,visits=sampler.visits,best_epoch=best[1])
    save_checkpoint(out/"dictionary.pt",state=best[2])


def diagnostic_dictionary(model,activity):
    w=model.weights().detach().cpu().numpy()
    v=w.transpose(1,0,2).reshape(w.shape[1],-1)
    norm=np.linalg.norm(v,axis=1)
    v=v/np.maximum(norm[:,None],1e-10)
    cosine=v@v.T
    z=activity-activity.mean(0)
    z/=np.maximum(np.linalg.norm(z,axis=0),1e-10)
    support=model.support.cpu().numpy()>0
    positive=model.positive.cpu().numpy()>0
    weighted=lambda x:(v[:,np.tile(x,2)]**2).sum(1)
    return dict(cosine=cosine.tolist(),activity_correlation=(z.T@z).tolist(),
                sd=activity.std(0).tolist(),norm=norm.tolist(),supported_loading_fraction=weighted(support).tolist(),
                detected_loading_fraction=weighted(positive).tolist(),
                effective_programs=int(np.sum((activity.std(0)>.05)&(norm>1e-6))),
                duplicate_pairs=int((np.triu(cosine,1)>.90).sum()))


def match_dictionaries(first,second):
    def x(model):
        w=model.weights().detach()
        return F.normalize(w.transpose(0,1).reshape(w.shape[1],-1),dim=1).cpu().numpy()
    c=x(first)@x(second).T
    a,b=linear_sum_assignment(-c)
    return c[a,b].tolist()


def fit_niches(data,out,model,a,train,valid,cfg,temp,mode,baseline,device):
    p=out/"niche.pt"
    if p.is_file():
        from .network_workflow import niche_from
        record=torch.load(p,map_location=device,weights_only=True)
        return niche_from(record,cfg,data.nt,device),record["summary"]
    ids=np.flatnonzero(train)
    n=initialize_niches(data,ids,a[ids],mode,temp,cfg,device,cfg.seed)
    optimizer=torch.optim.Adam(n.parameters(),lr=.04)
    rng=np.random.default_rng(cfg.seed)
    history=[]
    for e in range(cfg.niche_epochs):
        losses=[]
        for ix in batched(rng.choice(ids,min(cfg.centers,len(ids)),replace=False),cfg.batch):
            b=batch_data(data,ix,device)
            f=model.features(b) if mode!="composition" else None
            loss,_=n.loss(b.mass,model,f,uniform=e<3)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                n.activity.clamp_min_(0)
            losses.append(float(loss.detach()))
        history.append(float(np.mean(losses)))
    scores=[benchmark(data,valid,model,cfg,device,k,niche=n) for k in KINDS] if mode!="composition" else baseline
    with torch.no_grad():
        labels=[]
        for ix in batched(valid,cfg.batch):
            b=batch_data(data,ix,device)
            labels.extend(sum(n.evidence(b.mass,model,model.features(b))).argmax(1).cpu().tolist())
        h,m=model.predict(n.activity)
        v=torch.cat((h-model.base[0],m-model.base[1]),1)
        d=torch.pdist(v)/np.sqrt(v.shape[1])
        summary=dict(scores=scores,utility=utility(scores,baseline),history=history,
                     labels=labels,sizes=np.bincount(labels,minlength=cfg.k).tolist(),
                     omega=n.omega.softmax(0).cpu().tolist(),min_prototype_rms=float(d.min()),
                     min_node_distance=float(torch.pdist(n.pi.softmax(1),p=1).min()))
    summary=json.loads(json.dumps(summary))
    save_checkpoint(p,state=n.state_dict(),temperature=temp,mode=mode,summary=summary)
    return n,summary


class Experiment:
    def __init__(self,data,root,baseline_root,reference,device,base_cfg=None):
        self.data=data
        self.root=Path(root)
        self.device=device
        self.reference=[Path(p) for p in reference]
        self.cfg=base_cfg or Config()
        self.train,v=spatial_split(data)
        ids=np.flatnonzero(v)
        self.valid=np.sort(np.random.default_rng(204).choice(ids,min(128,len(ids)),replace=False))
        resources=Path(baseline_root)
        paths=[resources/"cellchat_interactions.csv",resources/"cellchat_complex.csv"]
        self.graph,self.pathways=biological_graph(data.lr,data.signaling,*[pd.read_csv(p,keep_default_na=False) for p in paths])
        import inspect
        from . import polish,polish_metrics,probabilistic,refit
        self.meta=dict(network=data.meta,resources={p.name:digest(p) for p in paths},runtime=runtime(device),
                       algorithms={m.__name__:digest(inspect.getfile(m)) for m in (polish,polish_metrics,probabilistic,refit)})
        self.root.mkdir(parents=True,exist_ok=True)
        if self.reference:
            model=load_dictionary(self.reference[0]/"dictionary.pt",self.cfg,Options(),device)
        else:
            model=init_dictionary(data,np.flatnonzero(self.train),self.cfg,device)
        base_path=self.root/("baseline-"+key(self.meta)+".json")
        if not base_path.is_file():
            base=[benchmark(data,self.valid,model,self.cfg,device,k,baseline=True) for k in KINDS]
            base_path.write_text(json.dumps(base))
        self.baseline=json.loads(base_path.read_text())

    def dictionary(self,rank,opt,seed,cached=None):
        cfg=replace(self.cfg,seed=seed,rank=rank)
        meta=dict(**self.meta,cfg=asdict(cfg),options=asdict(opt),
                  code=digest(__file__),reference=digest(cached/"dictionary.pt") if cached else None)
        out=self.root/("dictionary-"+key(meta))
        out.mkdir(exist_ok=True)
        with lock(out):
            p=cached/"dictionary.pt" if cached else out/"dictionary.pt"
            if not p.is_file():
                dictionary_fit(self.data,out,cfg,opt,self.graph,self.device,self.train,self.valid,self.baseline)
            model=load_dictionary(p,cfg,opt,self.device)
            act=out/"activity.npz"
            if not act.is_file():
                # No final-test centers are evaluated while selecting models.
                ids=np.flatnonzero(self.train)
                a=np.zeros((len(self.data.ids),cfg.rank),np.float32)
                a[ids]=activities(self.data,model,ids,cfg,self.device)
                save_npz(act,ids=self.data.ids,activity=a)
            with np.load(act) as z:
                if not np.array_equal(z["ids"],self.data.ids):
                    raise ValueError("Activity IDs disagree")
                a=z["activity"]
            m=out/"validation.json"
            if not m.is_file():
                score=[benchmark(self.data,self.valid,model,cfg,self.device,k) for k in KINDS]
                diagnostic=diagnostic_dictionary(model,a[self.train])
                m.write_text(json.dumps(dict(scores=score,utility=utility(score,self.baseline),diagnostic=diagnostic)))
                (out/"manifest.json").write_text(json.dumps(meta,indent=2))
            return model,a,json.loads(m.read_text()),out

    def niche(self,item,k,temp,mode,seed):
        model,a,metrics,path=item
        cfg=replace(self.cfg,seed=seed,k=k,rank=a.shape[1])
        where=path/f"{mode}-k{k}-t{temp:g}-s{seed}"
        where.mkdir(exist_ok=True)
        with lock(where):
            n,m=fit_niches(self.data,where,model,a,self.train,self.valid,cfg,temp,mode,self.baseline,self.device)
        return n,m,where
