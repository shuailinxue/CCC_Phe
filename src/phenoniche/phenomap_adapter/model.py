"""Reuse PhenoMap GNN; zero-offset scaling and nonnegative cell outputs.

The only architectural adapter is the nonnegative final activation. No spot
minimum is ever added to individual cells. Original code is imported read-only.
"""
import sys
from pathlib import Path
import numpy as np,torch
from scipy.spatial import cKDTree
from torch_geometric.data import Data


def load_upstream(path):
    path=str(Path(path).resolve())
    if path not in sys.path:sys.path.insert(0,path)
    from step1.models.spatial_model import SpatialGeneExpressionModel
    return SpatialGeneExpressionModel


def make_model(c,features,genes):
    base=load_upstream(c['phenomap_repo'])
    model=base(feat_dim=features,gene_dim=genes,lr=c['learning_rate'],embed_dim=c['hidden_dim'],num_layers=c['layers'],heads=c['heads'],dropout=c['dropout'])
    model.prediction_head.layers[-1]=torch.nn.Softplus()
    with torch.no_grad():model.prediction_head.layers[-2].bias.fill_(-6.)
    model.register_buffer('active_gene_mask',torch.ones(genes))
    model.register_forward_hook(lambda module,inputs,output:output*module.active_gene_mask)
    return model


def zero_offset_scale(counts):
    scale=np.maximum(np.asarray(counts).max(0),1.).astype(np.float32)
    active=np.asarray(counts).max(0)>0
    return (np.asarray(counts,dtype=np.float32)/scale),scale,active


def radius_edges(xy,radius,maximum=32):
    """Deterministic distance/ID ordering, same policy for training and inference."""
    tree=cKDTree(xy);src=[];dst=[]
    for v,neighbors in enumerate(tree.query_ball_point(xy,radius)):
        candidates=np.asarray([u for u in neighbors if u!=v],dtype=np.int64)
        if not len(candidates):continue
        distance=((xy[candidates]-xy[v])**2).sum(1)
        chosen=candidates[np.lexsort((candidates,distance))[:maximum]]
        src.extend(chosen);dst.extend([v]*len(chosen))
    return torch.tensor(np.asarray([src,dst]),dtype=torch.long).reshape(2,-1)


def graph(local,context,xy,radius,maximum=32):
    return Data(x=torch.as_tensor(local,dtype=torch.float32),x_context=torch.as_tensor(context,dtype=torch.float32),
        edge_index=radius_edges(xy,radius,maximum),pos=torch.as_tensor(xy,dtype=torch.float32))


def halo_ids(xy,core_ids,radius,layers):
    lo=xy[core_ids].min(0)-radius*layers-1e-3;hi=xy[core_ids].max(0)+radius*layers+1e-3
    return np.flatnonzero(((xy>=lo)&(xy<=hi)).all(1))
