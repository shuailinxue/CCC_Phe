"""Compute a disjoint tail of a running frozen inference, with hash verification."""
import argparse
from dataclasses import replace
import json
from pathlib import Path

import numpy as np

from phenoniche.v1.polish import Options
from phenoniche.v1.polish_fit import load_dictionary
from phenoniche.v1.polish_io import exported_niche
from phenoniche.v1.polish_stream import load_group,predict_group
from phenoniche.v1.probabilistic import Config
from phenoniche.v1.storage import digest


def main():
    p=argparse.ArgumentParser()
    for k in ("selected","network","inference"):
        p.add_argument("--"+k,required=True,type=Path)
    p.add_argument("--device",default="cuda:2")
    p.add_argument("--start-fraction",type=float,default=.55)
    a=p.parse_args()
    run=a.selected/"final73"
    selected=json.loads((a.selected/"selection.json").read_text())
    provenance=json.loads((a.inference/"manifest.json").read_text())
    record=json.loads((run/"metrics.json").read_text())
    d=Path(record["dictionary_fit"])
    manifest=json.loads((d/"manifest.json").read_text())
    ckpt=d/"dictionary.pt"
    if not ckpt.is_file():
        raise ValueError("Expected an independently fitted dictionary")
    if any(digest(p)!=provenance[k] for k,p in (("checkpoint",ckpt),
           ("niche",run/"niches.npz"),("network",a.network/"manifest.json"))):
        raise ValueError("Running inference uses different fitted inputs")
    cfg=replace(Config(**manifest["cfg"]),batch=64,k=selected["niche"]["k"])
    model=load_dictionary(ckpt,cfg,Options(**record["options"]),a.device)
    niche=exported_niche(run,"joint",selected["niche"]["temperature"],a.device)
    with np.load(a.network/"schema.npz") as z:
        schema={k:z[k] for k in z.files}
    files=sorted(a.network.glob("part-*.npz"))
    blocks=list(range(0,len(files),8))
    done=0
    tail=blocks[int(len(blocks)*a.start_fraction):][::-1]
    for start in tail:
        paths=files[start:start+8]
        offset=int(paths[0].stem.split("-")[-1])
        target=a.inference/"blocks"/f"part-{start:08}.npz"
        if target.is_file():
            continue
        data=load_group(paths,schema,offset)
        predict_group(data,target,model,niche,cfg,a.device)
        done+=len(data.ids)
        if done//2048%10==1:
            print(f"Completed {done} tail centers",flush=True)
    receipt=dict(device=a.device,algorithm=digest(__file__),completed_centers=done)
    (a.inference/f"tail_worker_{a.start_fraction:g}.json").write_text(json.dumps(receipt,indent=2))
    print(receipt,flush=True)


if __name__=="__main__":
    main()
