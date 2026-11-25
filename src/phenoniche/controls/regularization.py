"""Dictionary-control regularizer; not used by V1."""
import torch
from torch.nn import functional as F

def regularization(result):
    h=result['H']; p=result['P']; epsilon=1e-8
    active=(h.sum(0)>0).sum().clamp_min(2)
    entropy=-(h*h.clamp_min(epsilon).log()).sum(1).mean()/active.float().log()
    normalized=F.normalize(h,dim=1); cosine=normalized@normalized.T
    upper=torch.triu(torch.ones_like(cosine,dtype=torch.bool),1)
    diversity=F.relu(cosine[upper]-.7).square().mean()
    minimum=.05/p.shape[1]
    usage=(F.relu(minimum-p.mean(0))/minimum).square().mean()
    return entropy+diversity+usage
