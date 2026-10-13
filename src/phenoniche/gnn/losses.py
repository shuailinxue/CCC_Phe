"""Feature-balanced CCC loss, restrained regularization and optional gated nulls."""
import math
import numpy as np
import torch
from torch.nn import functional as F
from .model import physical_weights
from ..ccc.lr_atlas import canonicalize_complex


def ccc_loss(prediction, target, scale, mask):
    mask = torch.as_tensor(mask,device=prediction.device,dtype=torch.bool)
    scale = torch.as_tensor(scale,device=prediction.device)
    if not mask.any(): raise ValueError('No effective CCC supervision')
    if not torch.isfinite(target).all() or (target<0).any(): raise ValueError('Invalid CCC target')
    return F.smooth_l1_loss(prediction[:,mask]/scale[mask],target[:,mask]/scale[mask])


def regularization(result):
    h=result['H']; p=result['P']; epsilon=1e-8
    active = (h.sum(0)>0).sum().clamp_min(2)
    entropy=-(h*h.clamp_min(epsilon).log()).sum(1).mean()/active.float().log()
    normalized=F.normalize(h,dim=1); cosine=normalized@normalized.T
    upper=torch.triu(torch.ones_like(cosine,dtype=torch.bool),1)
    diversity=F.relu(cosine[upper]-.7).square().mean()
    minimum=.05/p.shape[1]
    usage=(F.relu(minimum-p.mean(0))/minimum).square().mean()
    return entropy+diversity+usage


def guarded_backward(primary, auxiliary, parameters, clip=5.):
    """CCC is primary: project conflicting auxiliary gradients off its direction.

    Records cosine/norms before projection; never silently rescales CCC gradients.
    """
    parameters=list(parameters)
    g=torch.autograd.grad(primary,parameters,retain_graph=True,allow_unused=True)
    a=torch.autograd.grad(auxiliary,parameters,allow_unused=True) if auxiliary.requires_grad else (None,)*len(parameters)
    g=[torch.zeros_like(p) if v is None else v for p,v in zip(parameters,g)]
    a=[torch.zeros_like(p) if v is None else v for p,v in zip(parameters,a)]
    dot=sum((x*y).sum() for x,y in zip(g,a)); norm=sum(x.square().sum() for x in g); anorm=sum(x.square().sum() for x in a)
    correction=torch.minimum(dot,torch.zeros_like(dot))/norm.clamp_min(1e-12)
    for p,x,y in zip(parameters,g,a): p.grad=x+y-correction*x
    torch.nn.utils.clip_grad_norm_(parameters,clip)
    return {'gradient_cosine':float((dot/(norm*anorm).sqrt().clamp_min(1e-12)).detach()),
            'ccc_gradient_norm':float(norm.sqrt().detach()),'aux_gradient_norm':float(anorm.sqrt().detach())}


def pair_ccc(ligand,receptor,types,features,categories,weights,chunk=32):
    # Exact weighted sqrt(L_s R_r)/(opportunity+1e-4), feature order unchanged.
    output=weights.new_empty((len(weights),len(features)))
    send=torch.as_tensor([categories.index(t) for t in features.sender],device=weights.device)
    receive=torch.as_tensor([categories.index(t) for t in features.receiver],device=weights.device)
    for start in range(0,len(features),chunk):
        stop=min(start+chunk,len(features))
        s=(types[:,:,None]==send[None,None,start:stop]).float()
        r=(types[:,:,None]==receive[None,None,start:stop]).float()
        raw=torch.einsum('bsf,bsr,brf->bf',ligand[:,:,start:stop].clamp_min(0).sqrt()*s,weights,receptor[:,:,start:stop].clamp_min(0).sqrt()*r)
        opportunity=torch.einsum('bsf,bsr,brf->bf',s,weights,r)
        output[:,start:stop]=raw/(opportunity+1e-4)
    return output



def composition_loss(prediction,target):
    """KL(target || prediction); zero-sum rows represent missing composition."""
    if prediction.shape!=target.shape: raise ValueError('Composition shape mismatch')
    if not torch.isfinite(target).all() or (target<0).any(): raise ValueError('Invalid composition target')
    mass=target.sum(1);valid=mass>0
    if not torch.allclose(mass[valid],torch.ones_like(mass[valid]),atol=1e-5): raise ValueError('Composition target is not a probability distribution')
    if not valid.any():return prediction.sum()*0
    return F.kl_div(prediction[valid].clamp_min(1e-8).log(),target[valid],reduction='batchmean')
