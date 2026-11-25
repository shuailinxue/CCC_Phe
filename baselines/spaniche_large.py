"""Native SpaNiche/IntNMF R solver, bounded-memory input/graph interfaces."""
from pathlib import Path
import json
import subprocess
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.spatial import cKDTree
from scipy.io import mmwrite
from .adapters import log_expression
from .protocol import panel_lr_prior,row_normalize


def views(xy,samples,radius):
    arrays=[([],[],[]) for _ in range(3)]
    for sample in np.unique(samples):
        ids=np.flatnonzero(samples==sample);tree=cKDTree(xy[ids])
        for begin in range(0,len(ids),1024):
            lists=tree.query_ball_point(xy[ids[begin:begin+1024]],2*radius)
            for offset,nearby in enumerate(lists):
                i=ids[begin+offset];j=ids[np.asarray(nearby,np.int64)]
                distance=np.linalg.norm(xy[j]-xy[i],axis=1)
                groups=[j[distance<=radius],j,np.r_[i,j[distance>radius]]]
                for (rows,cols,values),donors in zip(arrays,groups):
                    rows.extend([i]*len(donors));cols.extend(donors);values.extend([1./len(donors)]*len(donors))
    return [sp.csr_matrix((v,(r,c)),shape=(len(xy),len(xy))) for r,c,v in arrays]


def gaussian(xy,samples,sigma,output):
    path=Path(output)/'gaussian.mtx'
    if path.exists():return
    rows=[];cols=[];weights=[]
    for sample in np.unique(samples):
        ids=np.flatnonzero(samples==sample);tree=cKDTree(xy[ids])
        for begin in range(0,len(ids),512):
            lists=tree.query_ball_point(xy[ids[begin:begin+512]],8*sigma)
            rr=np.repeat(ids[begin:begin+len(lists)],[len(v) for v in lists])
            cc=ids[np.concatenate(lists)]
            ww=np.exp(-((xy[rr]-xy[cc])**2).sum(1)/(2*sigma**2))
            rows.append(rr);cols.append(cc);weights.append(ww)
    g=sp.coo_matrix((np.concatenate(weights),(np.concatenate(rows),np.concatenate(cols))),shape=(len(xy),len(xy)))
    mmwrite(path,g)


def spaniche(a,members,types,categories,cfg,output):
    output=Path(output);prior=panel_lr_prior(a.var_names,output)
    xy=np.asarray(a.obsm['spatial'],np.float64);samples=a.obs.sample_id.to_numpy()
    radius=float(np.median(np.linalg.norm(xy[members]-xy[:,None],axis=2).max(1)))
    if not (output/'input_shapes.json').exists():
        print('SpaNiche preparing full-cell multiscale matrices',len(a),flush=True)
        near,far,annulus=views(xy,samples,radius)
        onehot=np.eye(len(categories))[types]
        abundance=np.column_stack((onehot,near@onehot,far@onehot)).astype(np.float64)
        counts=log_expression(a);lookup={g:i for i,g in enumerate(a.var_names)}
        ligand=counts[:,[lookup[g] for g in prior.ligand]];receptor=counts[:,[lookup[g] for g in prior.receptor]]
        del counts
        blocks=[ligand*(near@receptor),receptor*(near@ligand),ligand*(annulus@receptor),receptor*(annulus@ligand)]
        variance=np.concatenate([v.var(0,ddof=1) for v in blocks]);lo,hi=np.quantile(variance,cfg['spaniche_variance_quantiles'])
        nonzero=np.concatenate([v.sum(0)>0 for v in blocks]);mask=(variance>lo)&(variance<hi)&nonzero
        if not mask.any():raise ValueError('No variable SpaNiche LR features')
        # Binary little-endian float64 row-major input avoids enormous text CSVs.
        abundance.astype('<f8').tofile(output/'abundance.bin')
        width=int(mask.sum());value=np.memmap(output/'LR_views.bin',mode='w+',dtype='<f8',shape=(len(a),width))
        position=0
        for j,block in enumerate(blocks):
            keep=mask[j*len(prior):(j+1)*len(prior)];stop=position+int(keep.sum())
            value[:,position:stop]=block[:,keep];position=stop
        value.flush();del value,blocks,ligand,receptor
        np.save(output/'LR_feature_mask.npy',mask)
        shapes=dict(abundance=list(abundance.shape),LR_views=[len(a),width])
        (output/'input_shapes.json').write_text(json.dumps(shapes))
    gaussian(xy,samples,cfg['spaniche_sigma_um'],output)
    args=[str(output),str(cfg['seed']),str(cfg['spaniche_topics']),str(cfg['spaniche_lambda']),
        str(cfg['spaniche_sigma_um']),str(cfg['spaniche_maxiter']),str(cfg['spaniche_initial_maxiter']),str(cfg['spaniche_initial_restarts'])]
    with (output/'R_execution.log').open('a') as stream:
        subprocess.run([cfg['rscript'],str(Path(__file__).with_name('spaniche_large_runner.R')),*args],stdout=stream,stderr=subprocess.STDOUT,check=True)
    embedding=pd.read_csv(output/'W.csv').to_numpy(np.float32)
    if embedding.shape!=(len(a),cfg['spaniche_topics']):raise ValueError('SpaNiche row count mismatch')
    (output/'adapter_audit.json').write_text(json.dumps(dict(
        variant='Official SpaNiche and IntNMF R core; sparse spatial Gaussian and scalable NNDSVD interface',
        all_cells=len(a),validation_cells=0,test_cells=0,radius_um=radius,
        within_sample_graphs_only=True,cell_type_input=True,condition_or_truth_input=False,
        gaussian_cutoff_sigma=8,maximum_discarded_degree_bound=float(len(a)*np.exp(-32)),
        initialization='Native NNDSVD/NNLS, truncated irlba SVD at1e-6; same native restarts',
        consensus_memory='Exact co-membership stability via canonical partition IDs; unused N×N consensus export omitted',
        native_NMF_objective_and_updates_unchanged=True),indent=2)+'\n')
    return embedding
