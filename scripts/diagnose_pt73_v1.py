"""Frozen Pt73 checkpoint diagnostics. Run with PYTHONPATH=src."""
from pathlib import Path
import argparse
from dataclasses import replace
import json
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from phenoniche.v1.model import Batch,hurdle
from phenoniche.v1.storage import Network,digest,key,save_npz,lock
from phenoniche.v1.workflow import checkpoint,runtime


def evaluate_batch(m,b,u0):
    w=m.weights()
    cfg=m.cfg
    weight=b.weights(cfg)
    def comm(u):
        return sum(x.sum(1) for x in hurdle(b,*m.predict(u,w),weight,cfg.sigma_x))
    def derivative(u,wt):
        lo,mu=m.predict(u,w)
        return ((b.s*lo.sigmoid()-b.c)*wt[0])@w[0].T+((mu*b.c-b.logs)*wt[1])@w[1].T/cfg.sigma_x**2
    def priorgrad(u):
        c,l,p=m.evidence(b.mass,u)
        return (u-(c+l+p).softmax(1)@m.eta)/cfg.sigma_a**2
    def optimize(u,k):
        u=u.clone()
        opt=torch.optim.Adam([u],lr=.08)
        for step in range(160):
            pr=priorgrad(u) if k is None else (u-m.eta[k])/cfg.sigma_a**2
            u.grad=u.sigmoid()*derivative(u,weight)+pr
            opt.step()
        return u
    g=u0.sigmoid()*derivative(u0,weight)
    raw=u0.sigmoid()*derivative(u0,b.weights(replace(cfg,scale="raw")))
    pg=priorgrad(u0)
    shared=optimize(u0,None)
    def score(u):
        c,l,p=m.evidence(b.mass,u)
        return c+l+p-comm(u)[:,None]
    candidates=[]
    for k in range(cfg.k):
        # Fixed-k objective can be nonconvex in u. Retain better of two starts.
        first=optimize(m.eta[k].expand_as(u0),k)
        second=optimize(u0,k)
        candidates.append(torch.maximum(score(first)[:,k],score(second)[:,k]))
    # Convex communication-only objective in a=softplus(u); a=0 is the u -> -inf limit.
    # Row-sum bound of the nonnegative Hessian bounds its spectral radius.
    lipschitz=(.25*(b.s*weight[0])@(w[0]*w[0].sum(0)).T +
               (b.c*weight[1])@(w[1]*w[1].sum(0)).T/cfg.sigma_x**2).amax(1,keepdim=True).clamp_min(1e-8)
    a=F.softplus(u0)
    y=a.clone()
    def agrad(a):
        lo=m.base[0]+a@w[0]
        mu=m.base[1]+a@w[1]
        return ((b.s*lo.sigmoid()-b.c)*weight[0])@w[0].T+((b.c*mu-b.logs)*weight[1])@w[1].T/cfg.sigma_x**2
    for j in range(2000):
        new=(y-agrad(y)/lipschitz).clamp_min(0)
        y=new+(j/(j+3))*(new-a)
        a=new
        if j%25==0 and (a-(a-agrad(a)).clamp_min(0)).norm(dim=1).max()<1e-5:
            break
    free=sum(x.sum(1) for x in hurdle(b,m.base[0]+a@w[0],m.base[1]+a@w[1],weight,cfg.sigma_x))
    return dict(a0=F.softplus(u0),a_free=a,grad_comm=g.norm(dim=1),grad_raw=raw.norm(dim=1),
                grad_prior=pg.norm(dim=1),grad_total=(g+pg).norm(dim=1),
                free_kkt=(a-(a-agrad(a)).clamp_min(0)).norm(dim=1),
                comm0=comm(u0),comm_free=free,shared0=score(u0),shared_refined=score(shared),
                candidate=torch.stack(candidates,1),refined_grad=(shared.sigmoid()*derivative(shared,weight)+priorgrad(shared)).norm(dim=1))


def main():
    p=argparse.ArgumentParser()
    for name in ("joint","communication","network"):
        p.add_argument("--"+name,type=Path,required=True)
    p.add_argument("--device",default="cuda:2")
    args=p.parse_args()
    meta=dict(sample=1024,seed=2026,diagnostic_code=digest(__file__),network=digest(args.network/"manifest.json"),
              checkpoints=[digest(q/"model.pt") for q in (args.joint,args.communication)],**runtime(args.device))
    dest=args.network.parent/("diagnosis-"+key(meta))
    dest.mkdir(exist_ok=True)
    with lock(dest):
        (dest/"manifest.json").write_text(json.dumps(meta,indent=2))
        if (dest/"local.npz").exists(): return
        d=Network(args.network)
        ids=np.sort(np.random.default_rng(2026).choice(len(d.ids),1024,replace=False))
        arrays={}
        for name,run in [('joint',args.joint),('communication',args.communication)]:
            if json.loads((run/"manifest.json").read_text())['network_key']!=key(d.meta):
                raise ValueError('Network mismatch')
            m,_=checkpoint(run,args.device)
            with np.load(run/'cells.npz') as cells:
                assert np.array_equal(cells['ids'],d.ids)
                u=cells['u'][ids]
            if name=='joint':
                chunks=[]
                for i in range(0,len(ids),64):
                    b=Batch(d,ids[i:i+64],args.device)
                    chunks.append({k:v.cpu().numpy() for k,v in evaluate_batch(m,b,torch.tensor(u[i:i+64],device=args.device)).items()})
                arrays.update({k:np.concatenate([c[k] for c in chunks]) for k in chunks[0]})
            else:
                b=Batch(d,ids[:64],args.device)
                w=m.weights()
                # Soft mixture at the first and final checkpoint is unavailable;
                # the saved epochwise prior/entropy terms quantify concentration.
                arrays['comm_eta']=m.eta.cpu().numpy()
                arrays['comm_omega']=m.omega.softmax(0).cpu().numpy()
                ww=w.transpose(0,1).reshape(m.cfg.rank,-1)
                vv=F.normalize(ww,dim=1)
                arrays['comm_cosine']=(vv@vv.T).cpu().numpy()
        save_npz(dest/'local.npz',ids=d.ids[ids],index=ids,**arrays)
        print(dest,flush=True)

if __name__=='__main__':
    main()
