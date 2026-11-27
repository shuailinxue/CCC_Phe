"""Fit the preserved legacy V1 within the new training region for paired evaluation."""
import argparse
from pathlib import Path
import json

import numpy as np
import pandas as pd
from phenoniche.v1 import fit, refit
from phenoniche.v1.model import Settings
from phenoniche.v1.network_workflow import save_checkpoint
from phenoniche.v1.probabilistic import Config, Dictionary, pooled
from phenoniche.v1.storage import Network, digest, key, lock, save_npz
from phenoniche.v1.workflow import checkpoint, runtime, write_cells


def main():
    p=argparse.ArgumentParser()
    p.add_argument("network",type=Path)
    p.add_argument("root",type=Path)
    p.add_argument("--device",default="cuda:2")
    args=p.parse_args()
    d=Network(args.network)
    d.train,validation=refit.spatial_split(d)
    cfg=Settings(
        batch=32,epochs=18,centers_per_epoch=512)
    meta=dict(runtime=runtime(args.device),code=digest(__file__),data=d.meta,settings=vars(cfg),
              spatial_split=digest(refit.__file__))
    out=args.root/("legacy-matched-"+key(meta))
    out.mkdir(parents=True,exist_ok=True)
    with lock(out):
        if not (out/"complete").exists():
            if not (out/"model.pt").is_file():
                model,tau,history,visits=fit.train(d,cfg,args.device)
                pd.DataFrame(history).to_csv(out/"loss.tsv",sep="\t",index=False)
                save_npz(out/"visits.npz",ids=d.ids,visits=visits)
                save_checkpoint(out/"model.pt",settings=vars(cfg),nt=d.nt,nl=d.nl,state=model.state_dict(),tau=tau)
            model,_=checkpoint(out,args.device)
            write_cells(out,d,fit.infer(model,d,args.device))
            baseline=Dictionary(*pooled(d,np.flatnonzero(d.train),args.device),cfg.rank)
            score=refit.prediction(d,np.flatnonzero(d.test),baseline,Config(),args.device,legacy=model)
            (out/"predictions.json").write_text(json.dumps(score,indent=2))
            (out/"manifest.json").write_text(json.dumps(meta,indent=2))
            (out/"complete").touch()
    print(out,flush=True)


if __name__=="__main__":
    main()
