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


def rewire_weights(batch,sigma,generator=None):
    weight=physical_weights(batch['coordinates'],batch['anchor_weights'],sigma,batch['cell_ids'])
    local_types=batch['types'][batch['local_index']]
    rewired=weight.clone()
    # Type-stratified receiver rewiring preserves per-sender type-pair weight multisets,
    # apart from newly forbidden self pairs (explicitly measured below).
    for b in range(len(weight)):
        permutation=torch.arange(weight.shape[1],device=weight.device)
        for t in local_types[b].unique():
            indices=torch.nonzero(local_types[b]==t).flatten()
            permutation[indices]=indices[torch.randperm(len(indices),device=weight.device,generator=generator)]
        rewired[b]=weight[b,:,permutation]
    distinct=batch['cell_ids'][:,:,None]!=batch['cell_ids'][:,None,:]
    rewired*=distinct
    drift=(rewired.sum((1,2))-weight.sum((1,2))).abs()/weight.sum((1,2)).clamp_min(1e-8)
    return weight,rewired,drift


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


def contrastive_loss(data,model,batch,result,scale,threshold=.05):
    original,negative,drift=rewire_weights(batch,model.sigma)
    unique=batch['unique']; raw=data.expression[unique].toarray()
    lookup={g:i for i,g in enumerate(data.genes)}
    sides=[]
    for column in ('ligand','receptor'):
        values=np.column_stack([np.prod(raw[:,[lookup[g] for g in canonicalize_complex(value)]],axis=1)**(1/len(canonicalize_complex(value))) for value in data.features[column]])
        sides.append(torch.as_tensor(values,device=original.device)[batch['local_index']])
    types=batch['types'][batch['local_index']]
    actual=pair_ccc(*sides,types,data.features,list(data.categories),original)
    counter=pair_ccc(*sides,types,data.features,list(data.categories),negative)
    scale=torch.as_tensor(scale,device=original.device)
    mask=torch.as_tensor(data.feature_mask,device=original.device)
    difference=((actual[:,mask]-counter[:,mask])/scale[mask]).square().mean(1).sqrt()
    gate=(difference>threshold)&(drift<.1)
    if not gate.any(): return result['Z'].sum()*0,0
    # Positive: mild directed-edge thinning of the same physical microenvironment.
    positive=original*(torch.rand_like(original)>.1)
    pos=data.forward(model,batch,pair_override=positive)
    neg=data.forward(model,batch,pair_override=negative)
    embedding=lambda value:torch.cat((value['P'].sqrt(),.25*torch.log1p(value['activity'])[:,None]/10),dim=1)
    pos_cos=F.cosine_similarity(embedding(result),embedding(pos)); neg_cos=F.cosine_similarity(embedding(result),embedding(neg))
    return ((1-pos_cos)+F.relu(neg_cos-pos_cos+.1))[gate].mean(),int(gate.sum())
