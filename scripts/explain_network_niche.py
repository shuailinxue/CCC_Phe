"""Export exact typed-edge contributions for a center in a frozen V1 fit."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from phenoniche.v1.network_workflow import dictionary_from, explain, niche_from
from phenoniche.v1.probabilistic import Config
from phenoniche.v1.storage import Network


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--network",required=True,type=Path)
    p.add_argument("--fit",required=True,type=Path)
    p.add_argument("--stage",choices=["joint","refined","composition","communication"],default="joint")
    p.add_argument("--cell",required=True)
    p.add_argument("--output",required=True,type=Path)
    p.add_argument("--device",default="cpu")
    a=p.parse_args()
    meta=json.loads((a.fit/"manifest.json").read_text())
    data=Network(a.network)
    if data.meta!=meta["input"]:
        raise ValueError("The fit and network manifests disagree")
    stage=a.fit/a.stage
    cfg=Config(**meta["settings"])
    with np.load(stage/"cells.npz") as z:
        np.testing.assert_array_equal(z["ids"],data.ids)
        match=np.flatnonzero(data.ids==a.cell)
        if len(match)!=1:
            raise ValueError(f"No unique center named {a.cell!r}")
        idx=int(match[0])
        activity=z["activity"][idx]
    params=torch.load(stage/"model.pt",map_location=a.device,weights_only=True)
    state=(params["dictionary"] if a.stage=="refined" else
           torch.load(a.fit/"dictionary.pt",map_location=a.device,weights_only=True)["state"])
    dictionary=dictionary_from(state,cfg)
    niche=niche_from(params,cfg,data.nt,a.device)
    info,table=explain(data,idx,dictionary,niche,activity,a.device)
    a.output.mkdir(parents=True,exist_ok=False)
    (a.output/"assignment.json").write_text(json.dumps(dict(cell_id=a.cell,**info),indent=2))
    table.to_csv(a.output/"edges.tsv.gz",sep="\t",index=False)


if __name__=="__main__":
    main()
