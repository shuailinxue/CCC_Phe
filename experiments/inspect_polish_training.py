"""Measure detector class and positive-intensity contributions on training centers."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from phenoniche.v1.polish import Options
from phenoniche.v1.polish_fit import load_dictionary
from phenoniche.v1.polish_metrics import gradient_parts
from phenoniche.v1.probabilistic import Config,batched,batch_data
from phenoniche.v1.storage import Network,digest,key,save_npz,versions


@torch.no_grad()
def main():
    p=argparse.ArgumentParser()
    p.add_argument("--selected",type=Path,required=True)
    p.add_argument("--network",type=Path,required=True)
    a=p.parse_args()
    own=a.selected/"final73"
    run=json.loads((own/"metrics.json").read_text())
    fit=Path(run["dictionary_fit"])
    meta=json.loads((fit/"manifest.json").read_text())
    f=fit/"dictionary.pt"
    if not f.is_file():
        f=next(p for p in (a.selected.parent.parent/"phenomap_network").glob("*/dictionary.pt") if digest(p)==meta["reference"])
    net=Network(a.network)
    if net.meta!=meta["network"]:
        raise ValueError("Network provenance mismatch")
    fingerprint=dict(script=digest(__file__),fit=digest(f),network=digest(a.network/"manifest.json"),
                     versions=versions(["torch","numpy"]))
    root=a.selected/("training-diagnostic-"+key(fingerprint))
    root.mkdir(exist_ok=True)
    if (root/"metrics.json").exists():
        print(root,flush=True)
        return
    with np.load(a.selected/"split.npz") as z:
        where=np.flatnonzero(z["train"])
    ids=np.sort(np.random.default_rng(99).choice(where,min(128,len(where)),replace=False))
    with np.load(own/"cells.npz") as z:
        np.testing.assert_array_equal(z["ids"],net.ids)
        activity=z["activity"][ids]
    model=load_dictionary(f,Config(**meta["cfg"]),Options(**run["options"]),"cpu")
    out={}
    for kind in ("raw","weighted"):
        gradients=[]
        losses=[]
        for ix in batched(np.arange(len(ids)),16):
            b=batch_data(net,ids[ix],"cpu")
            aa=torch.tensor(activity[ix])
            h,m=model.predict(aa)
            ft=model.features(b) if kind=="weighted" else (b.s,b.c,b.c,b.logs)
            s,c,n,l=ft
            gradient,_=gradient_parts(model,b,ft,aa)
            scale=n/b.c.clamp_min(1)
            positive=scale*(b.squares-2*m*b.logs+b.c*m.square())*.5
            loss=((s-c)*F.softplus(h),c*F.softplus(-h),positive)
            gradients.append(gradient.numpy())
            losses.append(torch.stack([v.sum(1) for v in loss],1).numpy())
        out[f"{kind}_gradient"]=np.concatenate(gradients)
        out[f"{kind}_nll"]=np.concatenate(losses)
    save_npz(root/"diagnostic.npz",ids=net.ids[ids],**out)
    (root/"manifest.json").write_text(json.dumps(fingerprint,indent=2))
    (root/"metrics.json").write_text(json.dumps(dict(order=["effective_zero","positive_detection","positive_intensity"],
        centers=len(ids),**{k:np.quantile(v,[.1,.5,.9],axis=0).tolist() for k,v in out.items()}),indent=2))
    print(root,flush=True)


if __name__=="__main__":
    main()
