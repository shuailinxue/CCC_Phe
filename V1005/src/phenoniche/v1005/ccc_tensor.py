"""Sparse retained entries with explicit sender/receiver/LR identities; V1002 semantics."""
from pathlib import Path
import json,time
import numpy as np
import pandas as pd
from scipy import sparse
from phenoniche.v1001.neighborhoods import build_neighborhoods
from phenoniche.v1002.lr_atlas import load_lr_atlas,LRAtlas
from phenoniche.v1002.simulation_cells import _side_expression,aggregate_pairwise_ccc
from .data import ROOT,REPO,TAXONOMY,PATHS,load_data,json_write


def entry_holdout(rows,cols,seed=991):
    """Fixed outcome-blind entry split, including retained zero entries."""
    r=np.asarray(rows,dtype=np.uint64);c=np.asarray(cols,dtype=np.uint64)
    h=(r*np.uint64(2654435761)) ^ (c*np.uint64(2246822519)) ^ np.uint64(seed)
    h ^= h >> np.uint64(13)
    return (h % np.uint64(10))==0

def training_only(x,row_offset=0):
    v=x.tocoo();keep=~entry_holdout(v.row+row_offset,v.col)
    return sparse.csr_matrix((v.data[keep],(v.row[keep],v.col[keep])),shape=x.shape)

def prepare(dataset,config):
    folder=ROOT/'outputs'/dataset;cache=folder/'cache';cache.mkdir(parents=True,exist_ok=True)
    done=cache/'complete.json'
    if done.exists(): return load_cache(dataset)
    a,coords,ct,types=load_data(dataset);n=len(coords);C=len(types)
    atlas=load_lr_atlas(REPO/'V1002/data/commuspace_human_lr_atlas.tsv')
    lookup={g.upper():g for g in a.var_names}
    rows=tuple(row for row in atlas.interactions if all(g in lookup for g in row.ligand_components+row.receptor_components))
    atlas=LRAtlas(rows,{},atlas.source_path);L=len(rows)
    genes=sorted({g for row in rows for g in row.ligand_components+row.receptor_components})
    expression=a[:,[lookup[g] for g in genes]].X
    if sparse.issparse(expression):expression=expression.toarray()
    # Contiguous row blocks avoid strided writes across multi-GB LR arrays.
    # The validated complex expression function and arithmetic are unchanged.
    ligand=np.lib.format.open_memmap(cache/'ligand.npy',mode='w+',dtype=np.float32,shape=(n,L))
    receptor=np.lib.format.open_memmap(cache/'receptor.npy',mode='w+',dtype=np.float32,shape=(n,L))
    for start in range(0,n,8192):
        stop=min(start+8192,n)
        a_l,a_r=_side_expression(np.asarray(expression[start:stop],dtype=np.float32),genes,atlas)
        ligand[start:stop]=a_l;receptor[start:stop]=a_r
        if start%81920==0:print(dataset,'complex expression',stop,'/',n,flush=True)
    ligand.flush();receptor.flush()
    del expression,a_l,a_r
    neighborhoods=build_neighborhoods(coords,k=config['neighbors'],sigma=config['sigma'])
    members=neighborhoods.indices.reshape(n,config['neighbors']);weights=neighborhoods.weights.reshape(members.shape).astype(np.float32)
    dummy=np.zeros((n,1),np.float32)
    P,_,opp=aggregate_pairwise_ccc(coords,ct,dummy,dummy,C,sigma=config['sigma'],members=members,anchor_weights=weights,
                                  batch_size=config['ccc_batch'],device=config['device'],compute_signal=False)
    cov_l=np.stack([(ligand[ct==c]>0).mean(0) for c in range(C)])
    cov_r=np.stack([(receptor[ct==c]>0).mean(0) for c in range(C)])
    mask=(cov_l[:,None,:]>=.1)&(cov_r[None,:,:]>=.1)
    mask &= ((opp>0).sum(0)>=max(20,int(np.ceil(.01*n)))).reshape(C,C,1)
    selected=np.flatnonzero(mask);s,r,l=np.unravel_index(selected,(C,C,L));F=len(selected)
    if not F:raise ValueError('No retained features')
    audit={'dataset':dataset,'N':n,'C':C,'L':L,'retained_features':F,'conceptual_shape':[n,C,C,L],
           'dense_tensor_bytes_avoided':int(n*C*C*L*4),'types':types,'original_label_counts':a.obs.cell_type_coarse.value_counts().to_dict(),
           'taxonomy':TAXONOMY,'annotation_status':'official observed labels' if dataset=='hbc1' else 'existing provisional marker-based annotation; not ground truth',
           'source':str(PATHS[dataset]),'source_size':PATHS[dataset].stat().st_size,'seed':config['seed'],
           'neighbors':config['neighbors'],'sigma':config['sigma'],'coverage':.1,'pair_support_fraction':.01,'pair_support_min':20}
    json_write(folder/'tensor_audit.json',audit)
    print(dataset,'tensor audit',n,C,L,F,flush=True)
    features=pd.DataFrame({'sender_id':s,'receiver_id':r,'lr_index':l,'sender':np.array(types)[s],'receiver':np.array(types)[r],
         'ligand':[rows[q].ligand for q in l],'receptor':[rows[q].receptor for q in l],'lr_id':[rows[q].lr_id for q in l]})
    features.to_csv(folder/'features.csv',index=False)
    np.savez(cache/'context.npz',P=P,coords=coords,cell_types=ct,selected=selected,shape=np.array([C,C,L]),
             spatial_neighbors=members)
    a.obs[['cell_id','cell_type_coarse']].to_csv(folder/'cells.csv',index=False)
    del a,opp,neighborhoods
    block_rows=max(1,min(config['ccc_store_rows'],int(256*2**20/max(F*4,1))))
    blocks=[];nnz=0;start_time=time.time()
    for start in range(0,n,block_rows):
        stop=min(n,start+block_rows);path=cache/f'ccc_{start:09d}.npz'
        if path.exists():x=sparse.load_npz(path)
        else:
            _,dense,_=aggregate_pairwise_ccc(coords,ct,ligand,receptor,C,sigma=config['sigma'],
                members=members[start:stop],anchor_weights=weights[start:stop],feature_mask=mask,
                batch_size=config['ccc_batch'],lr_batch_size=256,device=config['device'])
            x=sparse.csr_matrix(dense);sparse.save_npz(path,x,compressed=False);del dense
        assert x.shape==(stop-start,F)
        nnz+=x.nnz;blocks.append(path.name)
        if (start//block_rows)%20==0:print(dataset,'CCC',stop,'/',n,'seconds',round(time.time()-start_time),flush=True)
    audit['nnz']=nnz;audit['sparsity']=1-nnz/(n*F);json_write(folder/'tensor_audit.json',audit)
    json_write(done,{'blocks':blocks,'rows':n,'features':F,'config':config})
    return load_cache(dataset)

def load_cache(dataset):
    folder=ROOT/'outputs'/dataset;cache=folder/'cache';meta=json.loads((cache/'complete.json').read_text())
    paths=[cache/p for p in meta['blocks']]
    if sum(p.stat().st_size for p in paths)>96*2**30:raise MemoryError('Sparse cache exceeds guarded 96 GiB load; use on-disk batches')
    x=sparse.vstack([sparse.load_npz(p) for p in paths],format='csr')
    context=dict(np.load(cache/'context.npz'));context['features']=pd.read_csv(folder/'features.csv');context['X']=x
    return context
