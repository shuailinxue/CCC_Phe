"""Outcome-blind per-section union of the existing CCC eligibility rules."""
import numpy as np


def eligible_ccc_features(ligand,receptor,types,physical_groups,opportunity,center_groups,
                          categories,coverage=.1,opportunity_fraction=.01,scope='pooled_training'):
    if scope not in ('pooled_training','sample_union'):raise ValueError('Invalid CCC feature scope')
    if ligand.shape!=receptor.shape or len(types)!=len(ligand):raise ValueError('Coverage dimensions differ')
    groups=[None] if scope=='pooled_training' else np.unique(physical_groups)
    selected=np.zeros((categories*categories,ligand.shape[1]),bool);records=[]
    for group in groups:
        physical=np.ones(len(types),bool) if group is None else physical_groups==group
        centers=np.ones(len(opportunity),bool) if group is None else center_groups==group
        if not centers.any():continue
        left=np.zeros((categories,ligand.shape[1]));right=np.zeros_like(left)
        for t in range(categories):
            ids=physical&(types==t)
            if ids.any():left[t]=(ligand[ids]>0).mean(0);right[t]=(receptor[ids]>0).mean(0)
        support=(opportunity[centers]>0).sum(0)>=max(20,int(np.ceil(opportunity_fraction*centers.sum())))
        mask=((left[:,None,:]>=coverage)&(right[None,:,:]>=coverage)).reshape(selected.shape)&support[:,None]
        selected|=mask;records.append({'sample':'pooled' if group is None else str(group),'training_centers':int(centers.sum()),'eligible_features':int(mask.sum())})
    return selected,records
