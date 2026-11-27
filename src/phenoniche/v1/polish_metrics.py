"""Trial-weighted scoring, calibration and dependence diagnostics."""
import numpy as np
import torch
from torch.nn import functional as F
from .probabilistic import batched,batch_data,infer_activity
from .refit import hidden_edges
from .polish import grouped_masks


def masks(data,device,kind):
    if kind=="grouped":
        return grouped_masks(data.lr,data.nt,device)
    scored=hidden_edges(data,device)
    return scored,~scored


@torch.no_grad()
def benchmark(data,ids,model,cfg,device,kind,niche=None,baseline=False):
    scored,context=masks(data,device,kind)
    totals=np.zeros((7,7))
    roc=np.zeros((512,2))
    reliability=np.zeros((12,3))
    lr_p=(model.positive.reshape(-1,data.nl).sum(0)+.5)/(model.support.reshape(-1,data.nl).sum(0)+1)
    lr_group=torch.bucketize(lr_p,torch.tensor([.003,.02],device=device)).repeat(data.nt**2)[scored]
    support_cut=torch.tensor([4.,16.],device=device)
    def add(x,g,n):
        return torch.bincount(g.flatten(),weights=x.double().flatten(),minlength=n).cpu().numpy()
    for ix in batched(ids,cfg.batch):
        b=batch_data(data,ix,device)
        if baseline:
            h,m=model.anchor[:,scored].unbind()
            p=h.sigmoid()
        else:
            ft=model.features(b,context)
            if niche is None:
                a=infer_activity(model,ft,cfg,steps=2*cfg.local_steps)
                h,m=model.predict(a)
                p,m=h[:,scored].sigmoid(),m[:,scored]
            else:
                prob=sum(niche.evidence(b.mass,model,ft)).softmax(1)
                h,m=model.predict(niche.activity)
                pp=h[:,scored].sigmoid()
                p=prob@pp
                m=(prob@(pp*m[:,scored]))/p.clamp_min(1e-9)
        s,c,l,q=[x[:,scored] for x in (b.s,b.c,b.logs,b.squares)]
        z=s-c
        p=p.expand_as(s).contiguous().clamp(1e-7,1-1e-7)
        groups=[torch.zeros_like(s,dtype=torch.long),lr_group.expand_as(s)+1,
                torch.bucketize(s.contiguous(),support_cut)+4]
        for j,v in enumerate((s,c,-c*p.log(),-z*torch.log1p(-p),q-2*m*l+c*m.square(),c*p,z*p)):
            totals[:,j]+=sum(add(v,g,7) for g in groups)
        bins=((p.log()+16)/16*512).long().clamp(0,511)
        roc[:,0]+=add(c,bins,512)
        roc[:,1]+=add(z,bins,512)
        cal=torch.bucketize(p,torch.tensor([.001,.002,.004,.008,.015,.03,.06,.12,.24,.48,.75],device=device))
        for j,v in enumerate((s,c,s*p)):
            reliability[:,j]+=add(v,cal,12)
    def summary(v):
        s,c,p,z,e,cp,zp=v
        return dict(trials=int(s),positive=int(c),detection=(p+z)/max(s,1),
                    positive_nll=p/max(c,1),zero_nll=z/max(s-c,1),
                    rmse=float(np.sqrt(max(e,0)/max(c,1))),positive_mean=cp/max(c,1),zero_mean=zp/max(s-c,1))
    pos,neg=roc[::-1].T
    precision=pos.cumsum()/np.maximum((pos+neg).cumsum(),1)
    return dict(**summary(totals[0]),auprc=float((precision*pos).sum()/max(pos.sum(),1)),
                calibration_ece=float(np.abs(reliability[:,1]-reliability[:,2]).sum()/max(totals[0,0],1)),
                calibration=reliability.tolist(),score_lr=int(scored.sum()/data.nt**2),
                context_lr_any=int(context.reshape(data.nt**2,-1).any(0).sum()),
                groups={k:summary(v) for k,v in zip(
                    ("all","rare_lr","medium_lr","common_lr","support_le4","support_5to16","support_gt16"),totals)})


def utility(scores,baseline):
    # Guard the minority class without reweighting the actual evaluated trials.
    return float(np.mean([.35*s["detection"]/max(b["detection"],1e-8)
                          +.15*s["positive_nll"]/max(b["positive_nll"],1e-8)
                          +.50*(s["rmse"]/max(b["rmse"],1e-8))**2
                          for s,b in zip(scores,baseline,strict=True)]))


def gradient_parts(model,b,ft,a):
    """Gradient norms for zero, detection and positive-intensity terms."""
    s,c,n,l=ft
    h,m=model.predict(a)
    w=model.weights()
    terms=((s-c)*h.sigmoid()@w[0].T,c*(h.sigmoid()-1)@w[0].T,(n*m-l)@w[1].T)
    loss=((s-c)*F.softplus(h),c*F.softplus(-h),.5*n*m.square()-l*m)
    return torch.stack([x.norm(dim=1) for x in terms],1),torch.stack([x.sum(1) for x in loss],1)
