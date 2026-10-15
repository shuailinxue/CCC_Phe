"""Optional clustering features; original expression, CCC, Z and H stay intact."""
import numpy as np


def physical_fold_weights(members,weights,rows):
    members=np.asarray(members);weights=np.asarray(weights)
    if members.shape!=weights.shape or members.ndim!=2:raise ValueError('Invalid neighborhood weights')
    owner=np.full(len(members),-1,np.int8)
    for fold,r in enumerate(rows):
        cells=np.unique(members[r])
        if (owner[cells]>=0).any():raise ValueError('Physical folds overlap')
        owner[cells]=fold
    donor_owner=owner[members]
    pure=np.where((donor_owner==donor_owner[:,:1]).all(1),donor_owner[:,0],-1)
    valid=members!=np.arange(len(members))[:,None]
    valid &= (owner[:,None]<0)|((pure[members]==owner[:,None])&(pure[members]>=0))
    return weights*valid,valid


def spatial_cluster_features(values,members,weights,rows,mixing=.2,min_donors=5):
    values=np.asarray(values)
    if values.ndim!=2 or len(values)!=len(members) or not np.isfinite(values).all():raise ValueError('Invalid clustering representation')
    if not 0<=mixing<=.5:raise ValueError('Local mixing must be between zero and one half')
    result=values.copy()
    if mixing==0:return result
    w,valid=physical_fold_weights(members,weights,rows)
    for start in range(0,len(values),4096):
        r=slice(start,start+4096)
        neighbor=(values[np.asarray(members)[r]]*w[r,:,None]).sum(1)/np.maximum(w[r].sum(1,keepdims=True),1e-12)
        result[r]=np.where((valid[r].sum(1)>=min_donors)[:,None],(1-mixing)*values[r]+mixing*neighbor,values[r])
    return result
