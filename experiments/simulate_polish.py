"""Use controlled windows to verify that polishing retains joint signal recovery."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score

from phenoniche.v1.network_workflow import export
from phenoniche.v1.polish import Options
from phenoniche.v1.polish_fit import Experiment
from phenoniche.v1.probabilistic import Config
from phenoniche.v1.refit import activities
from phenoniche.v1.simulated import SCENARIOS,simulate

p=argparse.ArgumentParser()
p.add_argument("--root",type=Path,required=True)
p.add_argument("--resources",type=Path,required=True)
p.add_argument("--device",default="cuda:2")
a=p.parse_args()
cfg=Config(rank=8,k=2,epochs=18,min_epochs=12,niche_epochs=30,centers=192,batch=48)
rows=[]
for scenario in SCENARIOS:
    data=simulate(a.root,scenario)
    exp=Experiment(data,a.root/scenario,a.resources,[],a.device,cfg)
    for name,opt in (("control",Options()),("div_calibrated",Options(diversity=4,calibration=True))):
        item=exp.dictionary(cfg.rank,opt,73)
        model,_,m,path=item
        n,summary,out=exp.niche(item,2,.5,"joint",73)
        if not (out/"cells.npz").exists():
            local=activities(data,model,np.arange(len(data.ids)),cfg,a.device)
            export(data,model,n,local,cfg,a.device,out)
        with np.load(out/"cells.npz") as z,np.load(data.path/"truth.npz") as truth:
            labels=z["niche"][data.test]
            y=truth["labels"][data.test]
        rows.append(dict(scenario=scenario,model=name,test_ari=adjusted_rand_score(y,labels),validation=m["utility"],
                         isolated_prediction=summary["utility"],output=str(out)))
frame=pd.DataFrame(rows)
frame.to_csv(a.root/"simulation_polish.tsv",sep="\t",index=False)
print(frame.to_string(index=False),flush=True)
