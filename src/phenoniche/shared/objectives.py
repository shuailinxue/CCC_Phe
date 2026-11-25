"""Identical native-CCC and composition objectives for both backends."""
import torch
from torch.nn import functional as F


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


def composition_loss(prediction,target):
    """KL(target || prediction); zero-sum rows represent missing composition."""
    if prediction.shape!=target.shape: raise ValueError('Composition shape mismatch')
    if not torch.isfinite(target).all() or (target<0).any(): raise ValueError('Invalid composition target')
    mass=target.sum(1);valid=mass>0
    if not torch.allclose(mass[valid],torch.ones_like(mass[valid]),atol=1e-5): raise ValueError('Composition target is not a probability distribution')
    if not valid.any():return prediction.sum()*0
    return F.kl_div(prediction[valid].clamp_min(1e-8).log(),target[valid],reduction='batchmean')
