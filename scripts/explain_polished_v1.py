"""Decompose a chosen pair of candidate scores over supported typed LR edges."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
import numpy as np
import torch
from torch.nn import functional as F

from phenoniche.v1.evaluate import decode
from phenoniche.v1.polish_io import exported_niche
from phenoniche.v1.polish import Options
from phenoniche.v1.polish_fit import load_dictionary
from phenoniche.v1.probabilistic import Config,batch_data
from phenoniche.v1.storage import Network,digest


@torch.no_grad()
def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ("selected","network","output"):
        p.add_argument("--"+name,type=Path,required=True)
    p.add_argument("--cell",required=True)
    p.add_argument("--candidate",type=int)
    p.add_argument("--against",type=int)
    args=p.parse_args()
    run=args.selected/"final73"
    record=json.loads((run/"metrics.json").read_text())
    folder=Path(record["dictionary_fit"])
    meta=json.loads((folder/"manifest.json").read_text())
    data=Network(args.network)
    if data.meta!=meta["network"]:
        raise ValueError("Network and fitted schema differ")
    f=folder/"dictionary.pt"
    if not f.is_file():
        f=next(p for p in (args.selected.parent.parent/"phenomap_network").glob("*/dictionary.pt")
               if digest(p)==meta["reference"])
    cfg=Config(**meta["cfg"])
    model=load_dictionary(f,cfg,Options(**record["options"]),"cpu")
    from dataclasses import replace
    selected=json.loads((args.selected/"selection.json").read_text())["niche"]
    niche=exported_niche(run,"joint",selected["temperature"],"cpu")
    cfg=replace(cfg,k=len(niche.pi))
    ix=np.flatnonzero(data.ids==args.cell)
    if len(ix)!=1:
        raise ValueError(f"Invalid center {args.cell!r}")
    b=batch_data(data,ix,"cpu")
    s,c,n,l=model.features(b)
    terms=niche.evidence(b.mass,model,(s,c,n,l))
    score=sum(terms)[0]
    winner=int(score.argmax()) if args.candidate is None else args.candidate
    if not 0<=winner<cfg.k:
        raise ValueError("Candidate outside fitted vocabulary")
    alternate=score.clone()
    alternate[winner]=-torch.inf
    j=int(alternate.argmax()) if args.against is None else args.against
    if not 0<=j<cfg.k or j==winner:
        raise ValueError("Choose a different existing comparison")
    h,m=model.predict(niche.activity)
    occ=niche.temperature*(c*(h[winner]-h[j])-s*(F.softplus(h[winner])-F.softplus(h[j])))
    inten=niche.temperature*(l*(m[winner]-m[j])-.5*n*(m[winner].square()-m[j].square()))
    observed=b.s[0]>0
    edge=observed.nonzero().flatten().numpy()
    table=decode(edge,data.names,data.lr)
    table["supports"]=b.s[0,observed].numpy()
    table["detects"]=b.c[0,observed].numpy()
    table["occurrence_margin"]=occ[0,observed].numpy()
    table["intensity_margin"]=inten[0,observed].numpy()
    import pandas as pd
    annot=pd.read_csv(args.selected/"lr_annotations.tsv",sep="\t",keep_default_na=False)
    table["CellChat_pathways"]=annot.CellChat_pathways.to_numpy()[edge%data.nl]
    with np.load(run/"cells.npz") as z:
        np.testing.assert_array_equal(z["ids"],data.ids)
        a=torch.as_tensor(z["activity"][ix])
    pred_h,pred_m=model.predict(a)
    table["denoised_detection"]=pred_h[0,observed].sigmoid().numpy()
    table["denoised_log_intensity"]=pred_m[0,observed].numpy()
    row,e,v=data.entries(ix)
    raw=np.zeros(b.s.shape[1])
    raw[e]=v[:,3]/b.s[0,e].numpy()
    table["observed_mean"]=raw[edge]
    w=(a[0,None,:,None]-1)*model.weights()
    for r in range(cfg.rank):
        table[f"program_{r}_occurrence"],table[f"program_{r}_intensity"]=w[:,r,observed].numpy()
    node=float(terms[0][0,winner]-terms[0][0,j])
    prior=float(terms[-1][winner]-terms[-1][j])
    summary=dict(cell=args.cell,candidate=winner,against=j,node=node,prior=prior,
                 occurrence=float(occ.sum()),intensity=float(inten.sum()),total=float(score[winner]-score[j]))
    np.testing.assert_allclose(summary["total"],node+prior+summary["occurrence"]+summary["intensity"],atol=5e-4)
    args.output.mkdir(parents=True,exist_ok=False)
    (args.output/"margin.json").write_text(json.dumps(summary,indent=2))
    table.to_csv(args.output/"edges.tsv.gz",sep="\t",index=False)


if __name__=="__main__":
    main()
