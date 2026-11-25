"""Conditional RNA-profile randomization and fixed-geometry CCC statistics.

No network, preselected mechanism, planted state or known niche is accepted by
these functions. This is an exploratory data-layer experiment, not V1's default.
"""
import numpy as np
import torch
from scipy.spatial import cKDTree
from scipy import sparse
from scipy.stats import t as student_t


def depth_strata(types,totals,bins=10):
    groups=np.empty(len(types),np.int64);offset=0
    for kind in np.unique(types):
        ids=np.flatnonzero(types==kind);ids=ids[np.argsort(totals[ids],kind='stable')]
        for part in np.array_split(ids,min(bins,max(1,len(ids)//20))):
            groups[part]=offset;offset+=1
    return groups


def profile_permutation(groups,rng):
    permutation=np.arange(len(groups))
    for group in np.unique(groups):
        ids=np.flatnonzero(groups==group);permutation[ids]=rng.permutation(ids)
    return permutation


class FixedPhysicalCCC:
    """Batched bilinear form of the exact existing weighted sqrt(L*R) CCC.

    A single donor permutation is shared by every ligand/receptor, preserving
    whole-cell expression correlations. Physical positions/types never move.
    """
    def __init__(self,coordinates,types,members,weights,ligand,receptor,
                 feature_lr,feature_sender,feature_receiver,opportunity,
                 sigma=20.,device='cpu',anchor_batch=256,feature_batch=256):
        self.device=torch.device(device);self.anchor_batch=anchor_batch;self.feature_batch=feature_batch
        tensor=lambda x:torch.as_tensor(np.asarray(x).copy(),device=self.device)
        self.members=tensor(members).long();self.types=tensor(types).long();self.local_types=self.types[self.members]
        xy=tensor(coordinates).float()[self.members];w=tensor(weights).float()
        distance2=(xy[:,:,None]-xy[:,None,:]).square().sum(-1)
        self.pair=w[:,:,None]*w[:,None,:]*torch.exp(-distance2/(2*sigma*sigma))
        self.pair*=self.members[:,:,None]!=self.members[:,None,:]
        self.l=tensor(ligand).float().clamp_min(0).sqrt();self.r=tensor(receptor).float().clamp_min(0).sqrt()
        self.lr=tensor(feature_lr).long();self.sender=tensor(feature_sender).long();self.receiver=tensor(feature_receiver).long()
        nt=int(np.max(types))+1;self.pair_index=self.sender*nt+self.receiver
        self.opportunity=tensor(opportunity).float();self.n=len(members);self.cells=len(types);self.f=len(feature_lr)
        self.active=self.opportunity[:,self.pair_index]>0

    @torch.no_grad()
    def evaluate(self,permutation=None,rows=None):
        ids=torch.arange(self.n,device=self.device) if rows is None else torch.as_tensor(rows,device=self.device)
        permutation=torch.arange(self.cells,device=self.device) if permutation is None else torch.as_tensor(permutation,device=self.device)
        out=torch.empty((len(ids),self.f),device=self.device)
        for begin in range(0,len(ids),self.anchor_batch):
            index=ids[begin:begin+self.anchor_batch];m=self.members[index];donor=permutation[m];ty=self.local_types[index]
            pair=self.pair[index]
            for start in range(0,self.f,self.feature_batch):
                stop=min(start+self.feature_batch,self.f);q=self.lr[start:stop]
                left=self.l[donor[:,:,None],q[None,None,:]]*(ty[:,:,None]==self.sender[None,None,start:stop])
                right=self.r[donor[:,:,None],q[None,None,:]]*(ty[:,:,None]==self.receiver[None,None,start:stop])
                signal=(left*torch.bmm(pair,right)).sum(1)
                out[begin:begin+len(index),start:stop]=signal/(self.opportunity[index[:,None],self.pair_index[None,start:stop]]+1e-4)
        return out


def nonoverlap_graph(coordinates,members,composition,cfg):
    """Center graph for testing: each connected physical window is disjoint.

    This is a statistical graph over fixed observations, not a replacement of
    their physical members. Geometry and composition alone determine its edges.
    """
    xy=np.asarray(coordinates);n=len(xy)
    radius=np.sqrt(np.square(xy[members]-xy[:,None]).sum(-1)).max(1)
    distance,neighbor=cKDTree(xy).query(xy,k=min(cfg['graph_candidates'],n),workers=4)
    mix=np.sqrt(np.maximum(composition,0));u=[];v=[];values=[]
    for i in range(n):
        js=neighbor[i];ds=distance[i]
        good=(js!=i)&(ds>radius[i]+radius[js]+1e-6)&(ds<=cfg['graph_max_distance_um'])
        js,ds=js[good],ds[good]
        if not len(js):continue
        dc=np.square(mix[js]-mix[i]).sum(1)
        w=np.exp(-ds**2/(2*cfg['graph_bandwidth_um']**2)-dc/(2*cfg['composition_bandwidth']**2))
        order=np.argsort(-w,kind='stable')[:cfg['graph_neighbors']];js,w=js[order],w[order]
        w=w/w.sum();u.extend([i]*len(js));v.extend(js);values.extend(w)
    graph=sparse.csr_matrix((np.asarray(values,np.float32),(u,v)),shape=(n,n))
    folds=(np.floor(xy[:,0]/cfg['spatial_block_um']).astype(int)+np.floor(xy[:,1]/cfg['spatial_block_um']).astype(int))%2
    return graph,folds


def torch_graph(graph,device):
    coo=graph.tocoo()
    return torch.sparse_coo_tensor(torch.as_tensor(np.vstack([coo.row,coo.col]),device=device),
        torch.as_tensor(coo.data,device=device),coo.shape).coalesce()


def calibrated_values(matrix,mean,sd,active):
    z=(matrix-mean)/sd
    # Remove a feature-wide shift among physically supported windows: this
    # screen targets changing states, not uniform expression/communication.
    center=(z*active).sum(0)/active.sum(0).clamp_min(1)
    return (z-center)*active


def coherence(values,graph,folds,denominators):
    product=values*torch.sparse.mm(graph,values)
    return torch.stack([product.sum(0),product[folds==0].sum(0),product[folds==1].sum(0)])/denominators


def bh_adjust(p):
    p=np.asarray(p);order=np.argsort(p,kind='stable');rank=p[order]*len(p)/np.arange(1,len(p)+1)
    result=np.empty_like(p);result[order]=np.minimum.accumulate(rank[::-1])[::-1].clip(0,1)
    return result


def select_features(observed,null_scores,cfg):
    """Predictive t calibration, BH, and a predeclared two-block stability gate.

    The t reference is an approximation for these quadratic statistics. Its
    negative-control behavior is reported; no exact FDR guarantee is claimed.
    """
    observed=np.asarray(observed);null_scores=np.asarray(null_scores)
    b=len(null_scores);mean=null_scores.mean(0);sd=null_scores.std(0,ddof=1)
    sd=np.maximum(sd,np.maximum(np.median(sd,axis=1,keepdims=True)*1e-3,1e-8))
    z=(observed-mean)/(sd*np.sqrt(1+1/b));p=student_t.sf(z[0],df=b-1);q=bh_adjust(p)
    keep=(q<=cfg['feature_q_threshold'])&(observed[0]>0)&(z[1]>=cfg['minimum_fold_z'])&(z[2]>=cfg['minimum_fold_z'])
    return keep,z,p,q
