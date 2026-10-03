import torch

def nonconvex_lowrank(z,epsilon=1e-3):
    return torch.log1p(torch.linalg.svdvals(z)/epsilon).sum()

def geometry_loss(z_i,z_j,weights):
    return (weights*(z_i-z_j).square().sum(-1)).mean()

def group_sparsity(h,pair_ids,n_pairs):
    accum=torch.zeros((h.shape[0],n_pairs),device=h.device,dtype=h.dtype)
    accum.scatter_add_(1,pair_ids[None].expand(h.shape[0],-1),h.square())
    return (torch.sqrt(accum+1e-12)-1e-6).sum()
