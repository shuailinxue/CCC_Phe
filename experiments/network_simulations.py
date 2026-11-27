"""Assess direct-network niches against known labels after unsupervised fitting."""
import argparse
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score, balanced_accuracy_score
from scipy.optimize import linear_sum_assignment
from phenoniche.v1.network_workflow import fit_run
from phenoniche.v1.probabilistic import Config
from phenoniche.v1.simulated import SCENARIOS,simulate

p=argparse.ArgumentParser()
p.add_argument('--root',type=Path,required=True)
p.add_argument('--device',default='cuda:2')
args=p.parse_args()
cfg=Config(k=2,rank=12,batch=48,centers=192,epochs=36,min_epochs=12,niche_epochs=40)
rows=[]
for scenario in SCENARIOS:
    d=simulate(args.root,scenario)
    path=fit_run(d,d,args.root,cfg,args.device)
    with np.load(d.path/'truth.npz') as z:
        np.testing.assert_array_equal(z['ids'],d.ids)
        truth=z['labels'][d.test]
    for mode in ("joint","composition","communication","refined"):
        with np.load(path/mode/'cells.npz') as z:
            pred=z['niche'][d.test]
        mat=np.bincount(truth*2+pred,minlength=4).reshape(2,2)
        i,j=linear_sum_assignment(-mat)
        lookup=np.zeros(2,int)
        lookup[j]=i
        rows.append(dict(scenario=scenario,model=mode,ari=adjusted_rand_score(truth,pred),
                         balanced_accuracy=balanced_accuracy_score(truth,lookup[pred]),test_n=len(truth),
                         rare_n=int(truth.sum()),run=str(path)))
pd.DataFrame(rows).to_csv(args.root/'simulation_metrics.tsv',sep='\t',index=False)
