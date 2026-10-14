"""Feature-balanced CCC loss, restrained regularization and optional gated nulls."""
import math
import numpy as np
import torch
from torch.nn import functional as F
from .model import physical_weights
from ..ccc.lr_atlas import canonicalize_complex


def ccc_loss(prediction, target, scale, mask,kind='huber',delta=1.,nonzero_fraction=None,nonzero_weight_max=1.,entry_weights=None):
    mask = torch.as_tensor(mask,device=prediction.device,dtype=torch.bool)
    scale = torch.as_tensor(scale,device=prediction.device)
    if not mask.any(): raise ValueError('No effective CCC supervision')
    if not torch.isfinite(target).all() or (target<0).any(): raise ValueError('Invalid CCC target')
    if entry_weights is not None:
        entry_weights=torch.as_tensor(entry_weights,device=prediction.device,dtype=prediction.dtype)
        if entry_weights.shape!=target.shape or not torch.isfinite(entry_weights).all() or (entry_weights<0).any():raise ValueError('Invalid CCC observation weights')
        entry_weights=entry_weights[:,mask]
    prediction=prediction[:,mask]/scale[mask];target=target[:,mask]/scale[mask]
    if kind=='mse':error=F.mse_loss(prediction,target,reduction='none')
    elif kind=='huber' and delta>0:error=F.huber_loss(prediction,target,delta=delta,reduction='none')
    else:raise ValueError('Invalid CCC error objective')
    if entry_weights is not None:error=error*entry_weights
    if nonzero_weight_max==1:return error.mean()
    if not 1<nonzero_weight_max<=20 or nonzero_fraction is None:raise ValueError('Weighted CCC requires audited training feature support')
    rho=torch.as_tensor(nonzero_fraction,device=prediction.device,dtype=prediction.dtype)[mask]
    if not torch.isfinite(rho).all() or ((rho<0)|(rho>1)).any():raise ValueError('Invalid training nonzero fractions')
    positive_weight=((1-rho)/rho.clamp_min(1e-8)).sqrt().clamp(min=1,max=nonzero_weight_max)
    normalizer=1-rho+rho*positive_weight
    weights=torch.where(target>0,positive_weight,torch.ones_like(positive_weight))
    return (error*weights/normalizer).mean()


def ccc_error_sums(prediction,target,scale,mask):
    """Unweighted diagnostics, with sums for correct epoch-wide aggregation."""
    mask=torch.as_tensor(mask,device=prediction.device,dtype=torch.bool)
    scale=torch.as_tensor(scale,device=prediction.device)
    p=prediction.detach()[:,mask];t=target.detach()[:,mask];error=F.huber_loss(p/scale[mask],t/scale[mask],reduction='none')
    positive=t>0
    return {'huber_sum':float(error.sum()),'entries':error.numel(),
            'positive_huber_sum':float(error[positive].sum()),'positive_entries':int(positive.sum()),
            'zero_huber_sum':float(error[~positive].sum()),'zero_entries':int((~positive).sum()),
            'native_sse':float((p-t).square().sum()),'native_target_square':float(t.square().sum())}


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
    parameters=[p for p in parameters if p.requires_grad]
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
