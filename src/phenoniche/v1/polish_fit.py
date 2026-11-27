"""Fit and reuse the full-slice V1, with historical validation utilities."""
from dataclasses import asdict, replace
from copy import deepcopy
import json
from pathlib import Path
import time
from types import SimpleNamespace

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
from .storage import Network, digest, key, lock, save_npz
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


class _ResidentWindows:
    """Read the sparse window statistics on the fit device."""
    def __init__(self,d,device):
        self.device=device
        def to(x):
            return torch.as_tensor(x,device=self.device)
        self.mass=to(d.mass).float()
        self.support=to(d.support)
        self.group=to(d.group)
        self.ptr=to(d.ptr)
        self.edges=to(d.edge)
        self.values=to(np.ascontiguousarray(d.stats[:,:2]))
        self.positions=torch.arange(int(np.diff(d.ptr).max()),device=self.device)
        self.nt,self.nl=d.nt,d.nl
        self.width=d.nt*d.nt*d.nl

    def batch(self,ids):
        support=self.support[ids][...,self.group].reshape(len(ids),self.width).float()
        start=self.ptr[ids]
        count=self.ptr[ids+1]-start
        valid=self.positions[None,:]<count[:,None]
        if self.positions.numel()==0:
            zeros=torch.zeros_like(support)
            return SimpleNamespace(s=support,c=zeros,logs=zeros,mass=self.mass[ids],
                                   n=len(ids),nt=self.nt,nl=self.nl)
        gather=(start[:,None]+self.positions).clamp_max(len(self.edges)-1)
        destination=torch.where(valid,self.edges[gather].long(),self.width)
        values=self.values[gather]*valid[:,:,None]
        dense=torch.zeros((len(ids),self.width+1,2),device=self.device)
        dense.scatter_(1,destination[:,:,None].expand(-1,-1,2),values)
        c,logs=dense[:,:self.width].unbind(2)
        return SimpleNamespace(s=support,c=c,logs=logs,mass=self.mass[ids],
                               n=len(ids),nt=self.nt,nl=self.nl)


class _FullSliceTrainer:
    def __init__(self,resident,net,optimizer,local,stage,dictionary,cfg):
        self.cfg=cfg
        self.device=local.device
        self.resident,self.net,self.opt,self.local=resident,net,optimizer,local
        self.stage,self.dictionary=stage,dictionary
        self.visits=torch.zeros(len(local),dtype=torch.int32,device=self.device)
        self.one=torch.ones(cfg.batch,dtype=torch.int32,device=self.device)
        self.static=torch.arange(cfg.batch,device=self.device)
        self.stats=torch.zeros(5,dtype=torch.float64,device=self.device)
        self.graphs={}

    def step(self,ids,uniform=False):
        b=self.resident.batch(ids)
        f=self.dictionary.features(b)
        if self.stage==1:
            a=infer_activity(self.dictionary,f,self.cfg,self.local[ids],steps=6)
            self.local[ids]=a
            w=self.dictionary.weights()
            occ,inten=self.dictionary.local_loss(f,a,w)
            lr,div=self.dictionary.regularizers(w)
            loss = ((occ + inten).mean() + .15 * w.mean()
                    + .15 * (self.dictionary.base - self.dictionary.anchor).square().mean()
                    + self.dictionary.options.lr_strength * lr
                    + self.dictionary.options.diversity * div)
            parts=(loss,occ.mean(),inten.mean(),lr,div)
        else:
            loss,terms=self.net.loss(b.mass,self.dictionary,f,uniform=uniform)
            parts=(loss,*[v.mean() for v in terms])
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        self.opt.step()
        if self.stage==2:
            with torch.no_grad():
                self.net.activity.clamp_min_(0)
        with torch.no_grad():
            self.stats.add_(torch.stack(parts).detach().double()*len(ids))
            self.visits.index_add_(0,ids,self.one[:len(ids)])

    def graph(self,uniform):
        # Warm up on a side stream, then restore the same initial state in-place.
        self.static.copy_(torch.arange(self.cfg.batch,device=self.device))
        net=deepcopy(self.net.state_dict())
        local=self.local[:self.cfg.batch].clone()
        visits=self.visits[:self.cfg.batch].clone()
        stats=self.stats.clone()
        opt=deepcopy(self.opt.state_dict())
        stream=torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                self.step(self.static,uniform)
        torch.cuda.current_stream().wait_stream(stream)
        with torch.no_grad():
            self.net.load_state_dict(net)
            self.local[:self.cfg.batch].copy_(local)
            self.visits[:self.cfg.batch].copy_(visits)
            self.stats.copy_(stats)
            for group_now,group_saved in zip(self.opt.param_groups,opt['param_groups'],strict=True):
                for param, ident in zip(group_now['params'],group_saved['params'],strict=True):
                    for name,value in self.opt.state[param].items():
                        saved=opt['state'].get(ident,{}).get(name)
                        if torch.is_tensor(value):
                            value.zero_() if saved is None else value.copy_(saved)
        torch.cuda.synchronize()
        graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            self.step(self.static,uniform)
        self.graphs[uniform]=graph
        return graph

    def epoch(self,order,uniform):
        self.stats.zero_()
        perm=torch.as_tensor(order,device=self.device)
        end = len(order) // self.cfg.batch * self.cfg.batch
        if self.device.type != "cuda" or not end:
            for batch in batched(perm,self.cfg.batch):
                self.step(batch,uniform)
            return (self.stats/len(order)).cpu().tolist()
        graph=self.graphs.get(uniform)
        if graph is None:
            graph=self.graph(uniform)
        for start in range(0,end,self.cfg.batch):
            self.static.copy_(perm[start:start+self.cfg.batch])
            graph.replay()
        if end<len(order):
            self.step(perm[end:],uniform)
        return (self.stats/len(order)).cpu().tolist()


def _fit_full_stage(stage,resident,net,dictionary,local,epochs,cfg,out):
    device=local.device
    lr=.012 if stage==1 else .04
    opt=torch.optim.Adam(net.parameters(),lr=lr,fused=device.type=="cuda",capturable=device.type=="cuda")
    trainer=_FullSliceTrainer(resident,net,opt,local,stage,dictionary,cfg)
    resume=out/f'stage{stage}_resume.pt'
    rng=np.random.default_rng(cfg.seed)
    start=0
    history=[]
    if resume.exists():
        saved=torch.load(resume,map_location=device,weights_only=True)
        net.load_state_dict(saved['state'])
        opt.load_state_dict(saved['optimizer'])
        trainer.local.copy_(saved['local'])
        trainer.visits.copy_(saved['visits'])
        history=saved['history']
        start=saved['epoch']
        rng.bit_generator.state=json.loads(saved['rng'])
    for i in range(start,epochs):
        before=time.monotonic()
        metrics=trainer.epoch(rng.permutation(len(local)),uniform=stage==2 and i<3)
        if not np.isfinite(metrics).all():
            raise ValueError("Nonfinite loss in full-slice fit")
        minimum,maximum=int(trainer.visits.min()),int(trainer.visits.max())
        if minimum!=i+1 or maximum!=i+1:
            raise RuntimeError(f'Unbalanced coverage at epoch {i+1}: {minimum}, {maximum}')
        history.append(dict(epoch=i+1,total=metrics[0],component_1=metrics[1],component_2=metrics[2],
                            component_3=metrics[3],component_4=metrics[4],visits_min=minimum,visits_max=maximum))
        save_checkpoint(resume,state=net.state_dict(),optimizer=opt.state_dict(),local=local,
                        visits=trainer.visits,epoch=i+1,history=history,rng=json.dumps(rng.bit_generator.state))
        print(f'Stage {stage} epoch {i+1}/{epochs} • {time.monotonic()-before:.1f}s • loss {metrics[0]:.4f}',flush=True)
    pd.DataFrame(history).to_csv(out/f'stage{stage}_loss.tsv',sep='\t',index=False)
    return trainer.visits.cpu().numpy()


@torch.no_grad()
def _full_activity(data,resident,dictionary,cfg,out,device):
    path=out/"activity.npz"
    blocks=out/"activity_blocks"
    if not path.exists():
        blocks.mkdir(exist_ok=True)
        chunks=[]
        for start in range(0,len(data.ids),8192):
            stop=min(start+8192,len(data.ids))
            block=blocks/f"{start:08d}.npz"
            if not block.exists():
                values=[]
                for ix in batched(torch.arange(start,stop,device=device),128):
                    batch=resident.batch(ix)
                    value=infer_activity(dictionary,dictionary.features(batch),cfg,steps=2*cfg.local_steps)
                    values.append(value.cpu().numpy())
                save_npz(block,ids=data.ids[start:stop],activity=np.concatenate(values))
            with np.load(block,allow_pickle=False) as z:
                np.testing.assert_array_equal(z["ids"],data.ids[start:stop])
                chunks.append(z["activity"])
        save_npz(path,ids=data.ids,activity=np.concatenate(chunks))
    with np.load(path,allow_pickle=False) as z:
        np.testing.assert_array_equal(z["ids"],data.ids)
        return z["activity"]


def _full_identity(network,cfg,options,temperature):
    from . import model, network_workflow, networks, polish, probabilistic, refit, storage
    from phenoniche.spatial import channels

    manifest=network/"manifest.json"
    schema=network/"schema.npz"
    observed=json.loads(manifest.read_text())
    radii={"Contact":20.,"ECM":20.,"Secreted":100.,"Ambiguous":50.,"Unknown":50.}
    expected_code={Path(m.__file__).name:digest(m.__file__) for m in (storage,networks,channels)}
    if (observed.get("window")!=50 or observed.get("radii")!=radii or observed.get("holdout")!=0
        or observed.get("null_seed") is not None or observed.get("source_code")!=expected_code):
        raise ValueError("Expected a closed 50 µm network cache from the full, unpermuted slice")
    with np.load(schema,allow_pickle=False) as z:
        if (float(z["window"])!=50 or not len(z["ids"]) or not z["train"].all() or z["test"].any()
            or not np.array_equal(z["radii"][z["group"]], [radii[x] for x in z["signaling"]])):
            raise ValueError("Invalid full-slice cache schema")
    return dict(network=digest(manifest),schema=digest(schema),config=asdict(cfg),
                options=asdict(options),temperature=temperature,
                model_source={m.__name__:digest(m.__file__) for m in
                              (polish,probabilistic,refit,network_workflow,storage,model)})


def _verify_full_run(path,network,cfg,identity):
    if not (path/"complete").is_file():
        raise ValueError(f"Incomplete full-slice fit: {path}")
    recorded=json.loads((path/"manifest.json").read_text())
    if any(recorded.get(k)!=v for k,v in identity.items()):
        raise ValueError(f"Full-slice fit provenance mismatch: {path}")
    for name in ("dictionary.pt","niche.pt","programs.npz","niches.npz","metrics.json"):
        if not (path/name).is_file():
            raise ValueError(f"Missing full-slice result: {path/name}")
    dictionary = torch.load(path/"dictionary.pt", map_location="cpu", weights_only=True)["state"]
    niche = torch.load(path/"niche.pt", map_location="cpu", weights_only=True)
    if (dictionary["loading"].shape[1] != cfg.rank or niche["state"]["activity"].shape != (cfg.k, cfg.rank)
        or niche["mode"] != "joint" or niche["temperature"] != identity["temperature"]):
        raise ValueError("Full-slice checkpoint settings disagree with its manifest")
    with np.load(network/"schema.npz",allow_pickle=False) as schema, np.load(
        path/"cells.npz",allow_pickle=False
    ) as cells:
        ids=schema["ids"]
        for name in ("ids","xyz","train","test"):
            np.testing.assert_array_equal(cells[name],schema[name])
        if cells["activity"].shape!=(len(ids),cfg.rank) or cells["soft"].shape!=(len(ids),cfg.k):
            raise ValueError("Full-slice results have a different rank or K")
    for stage,epochs in (("dictionary",cfg.epochs),("niche",cfg.niche_epochs)):
        with np.load(path/f"{stage}_coverage.npz",allow_pickle=False) as coverage:
            np.testing.assert_array_equal(coverage["ids"],ids)
            if coverage["visits"].shape!=ids.shape or not (coverage["visits"]==epochs).all():
                raise ValueError(f"Incomplete {stage} visits")
    return path


def fit_full_slice(network,root,cfg,options,temperature,device,reuse_only=False):
    """Reuse an aligned census fit, otherwise fit all cells without a split."""
    from .network_workflow import niche_from

    if options.lr_strength != 0 or not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("Full-slice V1 requires zero LR-prior strength and a positive temperature")
    network,root=Path(network),Path(root)
    identity=_full_identity(network,cfg,options,temperature)
    completed=[]
    for path in sorted(root.glob("fullfit-*/complete")):
        manifest=path.parent/"manifest.json"
        if manifest.is_file():
            recorded=json.loads(manifest.read_text())
            if all(recorded.get(k)==v for k,v in identity.items()):
                completed.append(path.parent)
    if completed:
        if len(completed)!=1:
            raise ValueError("Multiple compatible full-slice fits in the output root")
        return _verify_full_run(completed[0],network,cfg,identity)
    if reuse_only:
        raise ValueError("No compatible completed full-slice fit")
    device=torch.device(device)
    if device.type=="cuda":
        torch.cuda.set_device(device)
    meta=dict(**identity,data_access="all centers once per epoch",stopping="fixed final epoch",
              source=digest(__file__),runtime=runtime(str(device)))
    out=root/("fullfit-"+key(meta))
    out.mkdir(parents=True,exist_ok=True)
    with lock(out):
        if (out/"complete").is_file():
            return _verify_full_run(out,network,cfg,identity)
        (out/"manifest.json").write_text(json.dumps(meta,indent=2))
        data=Network(network)
        ids=np.arange(len(data.ids))
        resident=_ResidentWindows(data,device)
        dictionary_path=out/"dictionary.pt"
        if not dictionary_path.is_file():
            checkpoint=out/"stage1_resume.pt"
            if checkpoint.is_file():
                state=torch.load(checkpoint,map_location=device,weights_only=True)["state"]
                dictionary=StructuredDictionary(state["anchor"],state["support"],state["positive"],cfg.rank,options)
            else:
                init=init_dictionary(data,ids,cfg,device)
                dictionary=StructuredDictionary(init.anchor,init.support,init.positive,cfg.rank,options)
                dictionary.load_state_dict(init.state_dict())
            local=torch.ones((len(ids),cfg.rank),device=device)
            visits=_fit_full_stage(1,resident,dictionary,dictionary,local,cfg.epochs,cfg,out)
            save_npz(out/"dictionary_coverage.npz",ids=data.ids,visits=visits)
            save_checkpoint(dictionary_path,state=dictionary.state_dict())
            del local
        dictionary=load_dictionary(dictionary_path,cfg,options,device)
        activity=_full_activity(data,resident,dictionary,cfg,out,device)
        niche_path=out/"niche.pt"
        if not niche_path.is_file():
            niche=initialize_niches(data,ids,activity,"joint",temperature,cfg,device,cfg.seed)
            local=torch.as_tensor(activity,device=device)
            visits=_fit_full_stage(2,resident,niche,dictionary,local,cfg.niche_epochs,cfg,out)
            save_npz(out/"niche_coverage.npz",ids=data.ids,visits=visits)
            save_checkpoint(niche_path,state=niche.state_dict(),temperature=temperature,mode="joint")
        niche=niche_from(torch.load(niche_path,map_location=device,weights_only=True),cfg,data.nt,device)
        del resident
        if device.type=="cuda":
            torch.cuda.empty_cache()
        metrics=export(data,dictionary,niche,activity,replace(cfg,batch=128),device,out)
        (out/"metrics.json").write_text(json.dumps(metrics,indent=2))
        (out/"complete").touch()
    return out
