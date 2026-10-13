"""MI signal enters expression only. Never alter computed CCC."""
import numpy as np
from .generation import TYPES

def inject(adata,lr,config):
    a=adata.copy();x=np.asarray(a.X).copy();lookup={g:i for i,g in enumerate(a.var_names)};n=len(a)//2
    radius=a.obs.radius_um.to_numpy();edge=np.isin(a.obs.true_niche.to_numpy(),[2,3]);types=a.obs.cell_type.to_numpy()
    act=np.zeros((len(a),2),np.float32)
    mid=config.core_radius_um+config.edge_width_um/2
    profile=np.exp(-.5*((radius-mid)/(config.edge_width_um/3))**2)*edge
    if config.mode=='expression':
        act[:,0]=profile*(a.obs.true_niche.to_numpy()==2);act[:,1]=profile*(a.obs.true_niche.to_numpy()==3)
        for mi in (1,2):
            links=lr[lr.mi==mi]
            for role,genes in [(config.sender_type,links.ligand),(config.receiver_type,links.receptor)]:
                ids=np.flatnonzero(types==role)
                for gene,loading in zip(genes,links.loading):x[ids,lookup[gene]]+=config.signal_strength*act[ids,mi-1]*loading*config.mi_lr_count
        sender=act.copy();receiver=act.copy()
    else:
        # A: both sender ligand groups in one angular state; receiver groups
        # occupy complementary angular states. B: permute whole fibroblast
        # expression profiles within sample/radial strata, preserving full
        # cell-type-conditional expression marginals in each radial bin.
        sender=np.zeros_like(act);receiver=np.zeros_like(act)
        bins=np.clip(((radius-config.core_radius_um)/config.edge_width_um*config.radial_bins).astype(int),0,config.radial_bins-1)
        for sample in np.unique(a.obs.sample_id):
            for band in range(config.radial_bins):
                for role in (config.sender_type,config.receiver_type):
                    ids=np.flatnonzero(edge[:n]&(types[:n]==role)&(a.obs.sample_id.to_numpy()[:n]==sample)&(bins[:n]==band))
                    ids=ids[np.argsort(a.obs.local_y.to_numpy()[ids])]
                    high=ids[len(ids)//2:];low=ids[:len(ids)//2]
                    if role==config.sender_type:sender[high]=1
                    else:receiver[high,0]=1;receiver[low,1]=1
        sender[n:]=sender[:n];receiver[n:]=receiver[:n]
        for mi in (1,2):
            for role,genes,active in [(config.sender_type,lr[lr.mi==mi].ligand,sender),(config.receiver_type,lr[lr.mi==mi].receptor,receiver)]:
                ids=np.flatnonzero(types==role)
                for gene,loading in zip(genes,lr[lr.mi==mi].loading):x[ids,lookup[gene]]+=config.signal_strength*active[ids,mi-1]*loading*config.mi_lr_count
        for sample in np.unique(a.obs.sample_id):
            for band in range(config.radial_bins):
                ids=np.flatnonzero(edge[:n]&(types[:n]==config.receiver_type)&(a.obs.sample_id.to_numpy()[:n]==sample)&(bins[:n]==band))
                ids=ids[np.argsort(a.obs.local_y.to_numpy()[ids])]
                x[ids+n]=x[ids[::-1]];receiver[ids+n]=receiver[ids[::-1]]
        act[:,0]=profile*(a.obs.true_niche.to_numpy()==2);act[:,1]=profile*(a.obs.true_niche.to_numpy()==3)
    a.X=x.astype(np.float32);a.obsm['injected_sender_activity']=sender;a.obsm['injected_receiver_activity']=receiver
    a.obsm['designed_MI_activity']=act
    a.uns['injection_mode']=config.mode
    return a
