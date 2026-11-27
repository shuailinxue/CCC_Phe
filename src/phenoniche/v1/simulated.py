"""Controlled independent windows with typed directional hurdle edges."""
from pathlib import Path
import json
import numpy as np

from .storage import Network, key, lock, save_npz


SCENARIOS = ("communication", "composition", "both", "sparse", "rare")


def simulate(root, scenario, seed=91):
    if scenario not in SCENARIOS:
        raise ValueError(scenario)
    meta=dict(simulation=scenario,seed=seed,version=1)
    path=Path(root)/("network-"+key(meta))
    path.mkdir(parents=True,exist_ok=True)
    with lock(path):
        if not (path/"manifest.json").exists():
            rng=np.random.default_rng(seed)
            n,nt,nl=300,2,8
            ids=np.array([f"window-{i}" for i in range(n)])
            truth=(rng.random(n)<(.15 if scenario=="rare" else .5)).astype(int)
            xyz=np.c_[np.arange(n)*120,np.zeros(n)]
            mass=np.tile([24,24],(n,1))
            if scenario in ("composition","both"):
                mass[truth==0]=[38,10]
                mass[truth==1]=[10,38]
            support=np.repeat(mass[:, :, None, None],nt,axis=2).astype(np.int32)
            s=support[...,np.zeros(nl,dtype=int)].reshape(n,-1)
            p=np.full(s.shape,.004 if scenario=="sparse" else .018)
            mean=np.full_like(p,-1.45)
            if scenario!="composition":
                for k,edge in ((0,slice(nl,nl+4)),(1,slice(2*nl+4,3*nl))):
                    mask=truth==k
                    p[mask,edge]=.08 if scenario=="sparse" else .34
                    mean[mask,edge]=-.38
            c=rng.binomial(s,p)
            row,edge=np.nonzero(c)
            stat=[]
            for i,e in zip(row,edge,strict=True):
                log=rng.normal(mean[i,e],.45 if scenario=="sparse" else .25,c[i,e])
                stat.append((len(log),log.sum(),log@log,np.exp(log).sum()))
            stats=np.asarray(stat,np.float32)
            for start in range(0,n,256):
                stop=min(start+256,n)
                select=(row>=start)&(row<stop)
                lengths=np.bincount(row[select]-start,minlength=stop-start)
                save_npz(path/f"part-{start:08}.npz",ids=ids[start:stop],mass=mass[start:stop],
                         support=support[start:stop],ptr=np.r_[0,np.cumsum(lengths)],
                         edge=edge[select].astype(np.int32),stats=stats[select])
            lr=np.array([(f"L{i}",f"R{i}") for i in range(nl)])
            save_npz(path/"schema.npz",ids=ids,xyz=xyz,names=["A","B"],samples=np.repeat("sim",n),
                     genes=lr.ravel(),scales=np.ones(nl*2),lr=lr,signaling=np.repeat("Unknown",nl),
                     group=np.zeros(nl,dtype=int),radii=[50],window=50,train=np.arange(n)<240,test=np.arange(n)>=240)
            save_npz(path/"truth.npz",ids=ids,labels=truth)
            (path/"manifest.json").write_text(json.dumps(meta,indent=2))
    return Network(path)
