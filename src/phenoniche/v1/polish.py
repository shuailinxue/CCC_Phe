"""Sparse biological regularization and reliability for the direct-network model."""
from collections import defaultdict
from dataclasses import dataclass, replace
from itertools import combinations

import numpy as np
import pandas as pd
from scipy import sparse
import torch
from torch.nn import functional as F

from .probabilistic import Config, Dictionary
from .refit import init_dictionary, hidden_edges


@dataclass(frozen=True)
class Options:
    lr_strength: float = 0.
    diversity: float = 0.
    calibration: bool = False

    def __post_init__(self):
        if not np.isfinite(self.lr_strength+self.diversity) or min(self.lr_strength,self.diversity)<0:
            raise ValueError("Regularization must be finite and nonnegative")


def biological_graph(lr, signaling, interactions, complexes):
    def arm(name):
        return "+".join(sorted(set(str(name).split("+"))))
    groups = complexes.set_index(complexes.columns[0]).to_dict("index")
    def resolve(name):
        return arm("+".join(str(x) for x in groups[name].values() if x)) if name in groups else arm(name)
    pathway = defaultdict(set)
    for item in interactions[["ligand","receptor","pathway_name"]].itertuples(index=False,name=None):
        l,r,p = item
        if l and r and p:
            pathway[resolve(l),resolve(r)].add(p)
    lr = [(arm(l),arm(r)) for l,r in lr]
    labels = [pathway[x] for x in lr]
    edges = defaultdict(float)
    for family,multiplier in ((0,1.),(1,.75)):
        families=defaultdict(list)
        for i,pair in enumerate(lr):
            families[pair[family],signaling[i]].append(i)
        for members in families.values():
            scale=multiplier/max(1,len(members)-1)
            for i,j in combinations(members,2):
                shared=labels[i]&labels[j]
                if labels[i] and labels[j] and not shared:
                    continue
                # Pathway membership alone is insufficient to assume concordance.
                edges[i,j]+=scale+(.35*scale if shared else 0.)
    pair=np.array(list(edges),dtype=int).reshape(-1,2)
    weight=np.array(list(edges.values()),dtype=np.float32)
    n=len(lr)
    a=sparse.coo_matrix((weight,(pair[:,0],pair[:,1])),shape=(n,n)).tocsr()
    a=a+a.T
    degree=np.asarray(a.sum(1)).ravel()
    inv=sparse.diags(1/np.maximum(degree,1)**.5)
    return (inv@a@inv).tocsr(),[";".join(sorted(p)) for p in labels]


def grouped_masks(lr,nt,device,seed=145):
    """Score target LRs only and exclude their complete gene neighborhoods from context."""
    genes=[set(l.split("+"))|set(r.split("+")) for l,r in lr]
    used=set()
    target=[]
    excluded=np.zeros(len(lr),bool)
    for i in np.random.default_rng(seed).permutation(len(lr)):
        more=used|genes[i]
        mask=np.array([bool(g&more) for g in genes])
        if mask.mean()>.38 and target:
            continue
        if mask.mean()>.65:
            continue
        used=more
        target.append(i)
        excluded=mask
        if len(target)>=max(1,int(.12*len(lr))):
            break
    if not target or excluded.all():
        raise ValueError("Unable to construct a nonempty gene-disjoint context")
    s=np.isin(np.arange(len(lr)),target)
    return tuple(torch.as_tensor(np.tile(x,nt**2),device=device) for x in (s,~excluded))


class StructuredDictionary(Dictionary):
    def __init__(self, base,support,positive,rank,options,graph=None):
        super().__init__(base,support,positive,rank)
        self.options=options
        if graph is not None and options.lr_strength:
            coo=graph.tocoo()
            idx=np.stack([coo.row,coo.col])
            adj=torch.sparse_coo_tensor(idx,coo.data,coo.shape,device=base.device).coalesce()
            support_mask=(support.reshape(-1,graph.shape[0])>0).float()
            degree=torch.sparse.mm(adj,support_mask.T).T
            self.register_buffer("adjacency",adj,persistent=False)
            self.register_buffer("lr_mask",support_mask,persistent=False)
            self.register_buffer("lr_degree",degree,persistent=False)

    def regularizers(self,w):
        lr=w.new_zeros(())
        diversity=lr
        if self.options.lr_strength and hasattr(self,"adjacency"):
            shape=(2,w.shape[1],*self.lr_mask.shape)
            weights=w.reshape(shape)*self.lr_mask
            axes=weights.reshape(-1,shape[-1])
            neighbors=torch.sparse.mm(self.adjacency,axes.T).T.reshape(shape)
            lr=(weights*(weights*self.lr_degree-neighbors)).sum()/(
                (self.lr_mask*self.lr_degree).sum().clamp_min(1)*2*w.shape[1])
        if self.options.diversity and w.shape[1]>1:
            vectors=F.normalize(w.transpose(0,1).reshape(w.shape[1],-1),dim=1)
            i,j=torch.triu_indices(len(vectors),len(vectors),1,device=w.device)
            diversity=((vectors@vectors.T)[i,j]-.80).clamp_min(0).square().mean()
        return lr,diversity

    def penalty(self,w):
        lr,div=self.regularizers(w)
        return super().penalty(w)+self.options.lr_strength*lr+self.options.diversity*div

    def features(self,batch,mask=None):
        if not self.options.calibration:
            return super().features(batch,mask)
        valid=batch.s>0
        if mask is not None:
            valid=valid&mask
        s=batch.s*valid
        mass=(batch.mass/batch.mass.sum(1,keepdim=True)).repeat_interleave(batch.nt*batch.nl,dim=1)
        # Modest, label-independent reweighting within each edge. Sparse
        # edges still shrink, with square-root rather than linear shrinkage.
        lr_p=(self.positive.reshape(-1,batch.nl).sum(0)+.5)/(self.support.reshape(-1,batch.nl).sum(0)+1)
        freq=(lr_p/.01).clamp(.5,2).pow(-.25).repeat(batch.nt**2)
        wt=mass*(self.support/(self.support+32)).sqrt()*freq/s.clamp_min(1).sqrt()*valid
        wi=wt*(self.positive/(self.positive+12)).sqrt()
        wo=wt*32/(s*wt).sum(1,keepdim=True).clamp_min(1).sqrt()
        wi=wi*8/(batch.c*wi).sum(1,keepdim=True).clamp_min(1).sqrt()
        return s*wo,batch.c*wo,batch.c*wi,batch.logs*wi
