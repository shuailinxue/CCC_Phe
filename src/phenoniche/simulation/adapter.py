"""Adapter to existing exact CCC, GraphData, expression-only GNN and spatial splits."""
from pathlib import Path
import json,hashlib
from dataclasses import replace
import numpy as np,pandas as pd,torch,scipy.sparse as sp
from ..ccc.pairwise import _side_expression,aggregate_pairwise_ccc
from ..ccc.lr_atlas import LRInteraction,LRAtlas
from ..gnn.data import GraphData,Targets,neighborhoods,spatial_split,expression_block,pair_composition
from ..gnn.model import physical_weights
from ..gnn.config import GNNConfig
from ..data import feature_statistics,resolve_device
from .generation import TYPES
from .programs import program_specs

def adapt(adata,lr,config,output):
    output=Path(output);cache=output/'input';cache.mkdir(exist_ok=True,parents=True)
    xy=np.asarray(adata.obsm['spatial'],np.float32);n=len(adata)
    types=pd.Categorical(adata.obs.cell_type,categories=TYPES).codes.astype(np.int64)
    if (types<0).any() or not adata.obs.cell_id.is_unique:raise ValueError('Invalid simulation cell/type identity')
    members,weights=neighborhoods(xy,30,20.,cache)
    samples=adata.obs.sample_id.to_numpy()
    if not (samples[members]==samples[:,None]).all():raise ValueError('Physical neighborhoods cross simulation replicates')
    if config.replicates>=3:groups=samples.astype(str)
    elif config.split_strategy=='mirrored_sectors':
        # Full-radius sectors avoid exterior-only test blocks. Mirrored
        # donors share a sector; geometry alone defines folds, never truth.
        angle=np.arctan2(adata.obs.local_y.to_numpy(),np.abs(adata.obs.local_x.to_numpy()))
        sectors=np.clip(np.floor((angle+np.pi/2)/np.pi*config.angular_sectors).astype(int),0,config.angular_sectors-1)
        groups=sectors.astype(str)
    else:
        # Mirror-paired counterfactual donors stay in the same fold; blocks
        # use geometry only and never true niche/MI labels.
        bx=np.floor(np.abs(adata.obs.local_x.to_numpy())/config.block_um).astype(int)
        by=np.floor((adata.obs.local_y.to_numpy()-adata.obs.local_y.min())/config.block_um).astype(int)
        groups=np.asarray([f'{a}:{b}' for a,b in zip(bx,by)])
    rows,folds,blocks=spatial_split(xy,members,config.seed,config.block_um,samples=groups)
    genes=tuple(str(g) for g in adata.var_names);raw=np.asarray(adata.X,np.float32)
    atlas=LRAtlas(tuple(LRInteraction(str(v.lr_id),v.ligand,v.receptor,(v.ligand,),(v.receptor,)) for v in lr.itertuples()),{},'simulated subset of real LR atlas')
    features=pd.DataFrame([{'feature_id':i,'sender':s,'receiver':r,'lr_id':v.lr_id,'ligand':v.ligand,'receptor':v.receptor,'ccc':f'{s} → {r} | {v.ligand}–{v.receptor}'} for i,(s,r,v) in enumerate((s,r,v) for s in TYPES for r in TYPES for v in lr.itertuples())])
    ccc_path=cache/'ccc.npy';shape=(n,len(features))
    if ccc_path.is_file():
        matrix=np.load(ccc_path,mmap_mode='r');assert matrix.shape==shape
    else:
        ligand,receptor=_side_expression(raw,genes,atlas)
        matrix=np.lib.format.open_memmap(cache/'ccc.incomplete.npy',mode='w+',dtype=np.float32,shape=shape)
        aggregate_pairwise_ccc(xy,types,ligand,receptor,5,20.,members,weights,batch_size=config.batch_size,lr_batch_size=16,device=resolve_device(config.device),communication_out=matrix)
        matrix.flush();del matrix;(cache/'ccc.incomplete.npy').replace(ccc_path);matrix=np.load(ccc_path,mmap_mode='r')
    if not np.isfinite(matrix).all() or (matrix<0).any():raise ValueError('Invalid computed CCC')
    mean,ms,var,nz=feature_statistics(matrix,rows[0]);mask=(nz>=.005)&(var>np.maximum(np.finfo(np.float32).eps*ms,np.finfo(np.float32).tiny))
    if not mask.any():raise ValueError('No effective CCC features; adjust simulation strength/support')
    rms=np.maximum(np.sqrt(ms),1e-6).astype(np.float32)
    global_scale=max(float(np.median(rms[mask])),1e-6)
    scale=(global_scale*(rms/global_scale)**config.ccc_scale_power).astype(np.float32)
    train_cells=np.unique(members[rows[0]]);totals=raw.sum(1);x=sp.csr_matrix(raw)
    sample=expression_block(x,train_cells,np.arange(len(genes)),totals)
    encoder=np.argsort(-sample.var(0),kind='stable')[:min(len(genes),512)];selected=sample[:,encoder]
    specs=program_specs(config);send_types=np.asarray([TYPES.index(s['sender']) for s in specs]);receive_types=np.asarray([TYPES.index(s['receiver']) for s in specs])
    comp=np.empty((n,5),np.float32);mi_activity=np.empty((n,config.mi_count),np.float32);mi_opportunity=np.empty_like(mi_activity)
    for start in range(0,n,config.batch_size):
        stop=min(start+config.batch_size,n);local=np.asarray(members[start:stop]);local_type=types[local]
        pair=physical_weights(torch.from_numpy(xy[local]),torch.from_numpy(np.asarray(weights[start:stop]).copy()),20.,torch.from_numpy(local))
        comp[start:stop]=pair_composition(torch.from_numpy(local_type),pair,5).numpy()
        # Ground truth is computed from known injection exposures + geometry,
        # not from final CCC and never passed to GraphData or clustering.
        sender_mask=(local_type[...,None]==send_types).astype(np.float32)
        receiver_mask=(local_type[...,None]==receive_types).astype(np.float32)
        sending=adata.obsm['injected_sender_activity'][local]*sender_mask
        receiving=adata.obsm['injected_receiver_activity'][local]*receiver_mask
        opp=torch.einsum('bsm,bsr,brm->bm',torch.from_numpy(sender_mask),pair,torch.from_numpy(receiver_mask))
        exposure=torch.einsum('bsm,bsr,brm->bm',torch.from_numpy(np.sqrt(sending)),pair,torch.from_numpy(np.sqrt(receiving)))
        mi_activity[start:stop]=(exposure/(opp+1e-4)).numpy();mi_opportunity[start:stop]=opp.numpy()
    true_h=np.zeros((config.mi_count,len(features)),np.float32)
    for spec in specs:
        mi=spec['MI'];links=lr[lr.mi==mi]
        ids=np.flatnonzero((features.sender==spec['sender'])&(features.receiver==spec['receiver'])&features.lr_id.isin(links.lr_id))
        loading=links.set_index('lr_id').loading
        true_h[mi-1,ids]=[loading.loc[v] for v in features.lr_id.iloc[ids]]
    np.save(cache/'composition_target.npy',comp);np.save(cache/'folds.npy',folds);features.to_csv(cache/'features.csv',index=False)
    np.savez(cache/'preprocessing.npz',encoder_indices=encoder,gene_mean=selected.mean(0),gene_scale=np.maximum(selected.std(0),.1),feature_scale=scale,feature_mask=mask,columns=np.arange(len(features)),folds=folds,blocks=blocks)
    truth=output/'ground_truth';truth.mkdir(exist_ok=True)
    np.save(truth/'MI_activity.npy',mi_activity);np.save(truth/'MI_dictionary.npy',true_h)
    np.save(truth/'MI_opportunity.npy',mi_opportunity);np.save(truth/'MI_cell_activity.npy',adata.obsm['designed_MI_activity'])
    pd.DataFrame(specs).to_csv(truth/'MI_programs.csv',index=False)
    mi_weights=mi_activity.sum(0)
    mi_composition=(mi_activity.T@comp)/np.maximum(mi_weights[:,None],1e-8)
    pd.DataFrame(mi_composition,index=[s['name'] for s in specs],columns=TYPES).to_csv(truth/'MI_composition.csv')
    np.save(truth/'niche_labels.npy',adata.obs.true_niche.to_numpy());np.save(truth/'composition.npy',comp)
    adata.obs.to_csv(truth/'cells.csv.gz',index=False);lr.to_csv(truth/'MI_LR_loading.csv',index=False)
    audit={'scope':'synthetic downstream simulation only','architecture':GNNConfig().architecture,'source_signature':{'simulation_signature':config.signature()},'cell_order_sha256':hashlib.sha256(pd.util.hash_pandas_object(adata.obs.cell_id,index=False).to_numpy().tobytes()).hexdigest(),'CCC_matrix_path':str(ccc_path),'categories':list(TYPES),'fixed_features':len(features),'effective_features':int(mask.sum()),'truth_entered_training':False,'clinical_Cox_analysis':False,'split':'sample or mirror-paired spatial blocks with all-member boundary purge','split_sizes':[len(v) for v in rows]}
    audit['split_strategy']=config.split_strategy if config.replicates==1 else 'sample'
    audit['composition_initialization']=config.initialize_composition
    audit['rescale_activity']=config.rescale_activity
    audit['ccc_scale_power']=config.ccc_scale_power
    audit['ccc_error']=config.ccc_error
    audit['n_ground_truth_programs']=config.mi_count
    audit['ground_truth_policy']='Four/edge-only dictionaries predeclared before injection; paired-cell activity from known injected exposures, independent of observed CCC and training'
    (cache/'audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    cells=adata.obs[['cell_id','cell_type']].reset_index(drop=True).copy()
    data=GraphData(x,totals,genes,encoder,selected.mean(0),np.maximum(selected.std(0),.1),types,TYPES,xy,members,weights,Targets(matrix,np.arange(len(features))),scale,mask,features,cells,rows,folds,blocks,comp,audit)
    data.original_feature_scale=rms
    return data

def gnn_config(config,output):
    return replace(GNNConfig(),seed=config.seed,split_seed=config.seed,hidden=config.hidden,programs=config.programs,batch_size=config.batch_size,max_epochs=config.gnn_epochs,patience=config.patience,device=config.device,output_dir=str(output),lambda_comp=config.lambda_comp,lambda_reg=config.lambda_reg,k_min=config.k_min,k_max=config.k_max,clustering_sample_size=10000,silhouette_sample_size=1500)
