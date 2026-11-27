"""Test communication-only local refits against concealed edges in a disjoint Pt73 slab."""
import argparse
from pathlib import Path
import json
import numpy as np
import torch
from diagnose_pt73_v1 import evaluate_batch
from phenoniche.v1.storage import Network,digest,key,save_npz,lock
from phenoniche.v1.model import Batch,local_fit
from phenoniche.v1.workflow import checkpoint,runtime

p=argparse.ArgumentParser()
p.add_argument('--run',required=True,type=Path)
p.add_argument('--network',required=True,type=Path)
p.add_argument('--device',default='cuda:2')
a=p.parse_args()
meta=dict(check=digest(__file__),diagnostic=digest(Path(__file__).with_name('diagnose_pt73_v1.py')),
          model=digest(a.run/'model.pt'),upstream=digest(a.network/'manifest.json'),seed=813,**runtime(a.device))
dest=a.network.parent/('free-prediction-'+key(meta))
dest.mkdir(exist_ok=True)
with lock(dest):
    if (dest/'metrics.json').exists():
        print(dest)
        raise SystemExit
    d=Network(a.network)
    rng=np.random.default_rng(813)
    ids=np.sort(rng.choice(np.flatnonzero(d.test),min(d.test.sum(),1024),replace=False))
    hidden=rng.random(d.nt*d.nt*d.nl)<.2
    mask=torch.tensor(hidden,device=a.device)
    m,_=checkpoint(a.run,a.device)
    w=m.weights()
    totals=np.zeros((2,6))
    activities=[]
    for start in range(0,len(ids),64):
        b=Batch(d,ids[start:start+64],a.device)
        s,c,logs,sq=[v[:,mask].clone() for v in (b.s,b.c,b.logs,b.squares)]
        for v in (b.s,b.c,b.logs,b.squares):
            v[:,mask]=0
        u=local_fit(m,b,w,b.weights(m.cfg),m.cfg.local_steps)
        diagnostic=evaluate_batch(m,b,u)
        activities.append(diagnostic['a_free'].cpu().numpy())
        for i,act in enumerate([torch.nn.functional.softplus(u),diagnostic['a_free']]):
            lo=m.base[0,mask]+act@w[0,:,mask]
            mu=m.base[1,mask]+act@w[1,:,mask]
            prob=lo.sigmoid()
            for j,value in enumerate([s,c,s*torch.nn.functional.softplus(lo)-c*lo,
                                      c*(1-prob)**2,(s-c)*prob**2,sq-2*mu*logs+c*mu**2]):
                totals[i,j]+=value.double().sum().item()
    result={}
    for label,(s,c,nll,pos,zero,sse) in zip(['shared','free'],totals):
        result[label]=dict(detection_nll=nll/s,brier=(pos+zero)/s,positive_log_rmse=np.sqrt(sse/c))
    save_npz(dest/'refit.npz',ids=d.ids[ids],hidden=hidden,activity=np.vstack(activities),sums=totals)
    (dest/'manifest.json').write_text(json.dumps(meta,indent=2))
    (dest/'metrics.json').write_text(json.dumps(result,indent=2))
    print(dest)
