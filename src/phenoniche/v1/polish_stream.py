"""Resumable, bounded-memory fixed-global inference over existing window shards."""
from pathlib import Path
from shutil import copyfile

import numpy as np
import torch

from .probabilistic import batched,batch_data,infer_activity
from .storage import Network,save_npz


def load_group(files,schema,start):
    data=Network.__new__(Network)
    for name,value in schema.items():
        setattr(data,name,value)
    data.nt,data.nl=len(data.names),len(data.lr)
    mass,support,edge,stats,lengths=[],[],[],[],[]
    stop=start
    for f in files:
        with np.load(f,allow_pickle=False) as row:
            ids=row["ids"]
            end=stop+len(ids)
            if not np.array_equal(ids,schema["ids"][stop:end]):
                raise ValueError(f"Full-slice shard ID mismatch: {f}")
            stop=end
            mass.append(row["mass"])
            support.append(row["support"])
            edge.append(row["edge"])
            stats.append(row["stats"])
            lengths.append(np.diff(row["ptr"]))
    for name in ("ids","samples","xyz","train","test"):
        setattr(data,name,getattr(data,name)[start:stop])
    data.mass=np.concatenate(mass)
    data.support=np.concatenate(support)
    data.edge=np.concatenate(edge)
    data.stats=np.concatenate(stats)
    data.ptr=np.r_[0,np.cumsum(np.concatenate(lengths))]
    return data


@torch.no_grad()
def predict_group(data,target,model,niche,cfg,device):
    values={k:[] for k in ("composition","occurrence_gain","intensity_gain","prior","niche","soft","activity")}
    for ix in batched(np.arange(len(data.ids)),cfg.batch):
        b=batch_data(data,ix,device)
        f=model.features(b)
        a=infer_activity(model,f,cfg,steps=cfg.local_steps*2)
        terms=niche.evidence(b.mass,model,f)
        soft=sum(terms).softmax(1)
        for k,v in zip(values,(*[x.expand(b.n,-1) for x in terms],soft.argmax(1),soft,a),strict=True):
            values[k].append(v.cpu().numpy())
    save_npz(target,ids=data.ids,mass=data.mass,**{k:np.concatenate(v) for k,v in values.items()})


@torch.no_grad()
def stream(path,dest,fitted,model,niche,cfg,device):
    path,dest=Path(path),Path(dest)
    chunks=dest/"blocks"
    chunks.mkdir(exist_ok=True)
    with np.load(path/"schema.npz",allow_pickle=False) as z:
        schema={k:z[k] for k in z.files}
    files=sorted(path.glob("part-*.npz"))
    start=0
    paths=[]
    for part in range(0,len(files),8):
        selected=files[part:part+8]
        target=chunks/f"part-{part:08}.npz"
        paths.append(target)
        if target.is_file():
            with np.load(target,allow_pickle=False) as z:
                ids=z["ids"]
                if not np.array_equal(ids,schema["ids"][start:start+len(ids)]):
                    raise ValueError(f"Cached prediction IDs disagree: {target}")
                start+=len(ids)
            continue
        data=load_group(selected,schema,start)
        predict_group(data,target,model,niche,cfg,device)
        start+=len(data.ids)
        if part%80==0 or start==len(schema["ids"]):
            print(f"{dest.name} {start}/{len(schema['ids'])} centers",flush=True)
    if start!=len(schema["ids"]):
        raise ValueError("Incomplete network shards")
    collected={}
    for target in paths:
        with np.load(target,allow_pickle=False) as chunk:
            for k in chunk.files:
                collected.setdefault(k,[]).append(chunk[k])
    merged={k:np.concatenate(v) for k,v in collected.items()}
    np.testing.assert_array_equal(merged["ids"],schema["ids"])
    save_npz(dest/"cells.npz",xyz=schema["xyz"],specimens=schema["samples"],types=schema["names"],
             train=np.zeros(len(merged["ids"]),bool),test=np.zeros(len(merged["ids"]),bool),**merged)
    for name in ("niches.npz","programs.npz","top_edges.tsv"):
        copyfile(Path(fitted)/name,dest/name)
    sizes=np.bincount(merged["niche"],minlength=cfg.k)
    node=(merged["composition"]+merged["prior"]).argmax(1)
    return dict(centers=len(node),niche_sizes=sizes.tolist(),
                activity_sd=merged["activity"].std(0).tolist(),
                median_max_soft=float(np.median(merged["soft"].max(1))),
                evidence_change=float(np.mean(node!=merged["niche"])))
