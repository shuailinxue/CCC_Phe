"""Raw-count, fragment-aware colitis analysis; clinical CCC/Cox is untouched.

Only the cohort adapter and descriptive interpretation are new. The production
physical-pair CCC, GNN, NMF initialization and niche discovery are reused.
"""
from pathlib import Path
from functools import partial
import hashlib,json
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.spatial import cKDTree
from scipy.sparse.csgraph import connected_components
import anndata as ad
import torch
from ..ccc.lr_atlas import load_lr_atlas,LRAtlas
from ..ccc.pairwise import _side_expression,aggregate_pairwise_ccc
from ..gnn.data import GraphData,Targets,neighborhoods,expression_block,pair_composition
from ..gnn.model import physical_weights,model_from_config
from ..gnn.config import GNNConfig
from ..gnn.genes import select_encoder_genes
from ..gnn.training import fit,infer
from ..gnn.losses import ccc_loss
from ..gnn.evaluation import discover
from ..gnn.spatial_refinement import spatial_cluster_features
from ..simulation.baselines import dictionary_program_projection
from ..consolidation import _latent_block
from ..data import feature_statistics
from ..storage import STORAGE_ROOT
from .feature_scope import eligible_ccc_features

CONDITIONS=('HC','UC','ICI')
PATIENT='25_06_11_Patient_ID_HS'
CONDITION='240927_Condition'
COARSE='25_06_11_ICI_5K_Coarse_annotations'
FINE='24_11_12_ICI_5K_Fine_annotations'


def read_config(path):
    c=json.loads(Path(path).read_text());out=Path(c['output_dir'])
    if not out.resolve().is_relative_to(STORAGE_ROOT.resolve()) or out.is_symlink():
        raise ValueError('Direct NAS output required')
    return c


def tissue_components(xy,patients,link_um=100.,minimum=100):
    """Find disconnected tissue fragments within patients, never across donors."""
    core=np.full(len(xy),'',dtype=object);keep=np.zeros(len(xy),bool)
    for patient in sorted(np.unique(patients)):
        ids=np.flatnonzero(patients==patient);v=xy[ids]
        if len(v)<30:continue
        distance,index=cKDTree(v).query(v,k=min(17,len(v)))
        good=distance[:,1:]<=link_um
        row=np.broadcast_to(np.arange(len(v))[:,None],index[:,1:].shape)[good]
        graph=sp.coo_matrix((np.ones(good.sum(),np.int8),(row,index[:,1:][good])),shape=(len(v),len(v))).tocsr()
        _,labels=connected_components(graph,directed=False)
        # Stable coordinate order gives fragments reproducible identifiers.
        groups=sorted(np.unique(labels),key=lambda j:tuple(v[labels==j].mean(0)))
        for j,label in enumerate(groups):
            local=ids[labels==label]
            if len(local)>=minimum:core[local]=f'{patient}_fragment{j+1:02d}';keep[local]=True
    return core,keep


def patient_folds(obs,seed):
    """One patient per condition per fold; labels balance splits, never model input."""
    meta=obs[['patient','condition']].drop_duplicates().sort_values('patient')
    if meta.patient.duplicated().any():raise ValueError('Patient has inconsistent condition')
    rng=np.random.default_rng(seed);mapping={}
    for condition in CONDITIONS:
        patients=meta.loc[meta.condition==condition,'patient'].to_numpy()
        if len(patients)!=3:raise ValueError('Expected three independent patients per condition')
        for fold,p in enumerate(rng.permutation(patients)):mapping[p]=fold
    return obs.patient.map(mapping).to_numpy(np.int8),mapping


def fragment_folds(obs,seed):
    """All donors contribute training tissue; whole fragments remain disjoint."""
    mapping={};rng=np.random.default_rng(seed)
    for patient in sorted(obs.patient.unique()):
        counts=obs.loc[obs.patient==patient].groupby('core',observed=True).size().sort_values(ascending=False)
        for core in counts.index:mapping[str(core)]=0
        if len(counts)>=3:
            mapping[str(counts.index[1])]=1;mapping[str(counts.index[2])]=2
        elif len(counts)==2:mapping[str(counts.index[1])]=int(rng.integers(1,3))
        else:raise ValueError('Joint discovery needs at least two fragments per donor')
    return obs.core.map(mapping).to_numpy(np.int8),mapping


def load_tissue(c,out):
    target=out/'input/tissue.h5ad'
    if target.exists():return ad.read_h5ad(target)
    root=Path(c['data_dir']);print('Aligning original raw counts and published annotations',flush=True)
    annotated=ad.read_h5ad(root/'25_11_12_Xenium_Dataset2_5K_Annotated.h5ad',backed='r')
    observed=annotated.obs.copy();xy=np.asarray(annotated.obsm['spatial'],np.float32)
    selected=observed[PATIENT].astype(str).ne('unassigned')&observed[CONDITION].astype(str).isin(CONDITIONS)
    observed=observed.loc[selected];xy=xy[selected]
    obs=pd.DataFrame({'cell_id':observed.index.astype(str),'patient':observed[PATIENT].astype(str).to_numpy(),
        'condition':observed[CONDITION].astype(str).to_numpy(),'cell_type':observed[COARSE].astype(str).to_numpy(),
        'cell_type_fine':observed[FINE].astype(str).to_numpy()},index=observed.index)
    core,keep=tissue_components(xy,obs.patient.to_numpy(),c['core_link_um'],c['minimum_core_cells'])
    obs=obs.loc[keep].copy();obs['core']=core[keep];xy=xy[keep]
    annotated.file.close();del annotated
    raw=ad.read_h5ad(root/'25_11_12_Xenium_Dataset2_5K_Raw.h5ad')
    order=pd.Index(raw.obs_names).get_indexer(obs.index)
    if (order<0).any() or not np.allclose(raw.obsm['spatial'][order],xy,atol=1e-3):raise ValueError('Raw cell/coordinate mismatch')
    genes=raw.var_names.astype(str).str.upper();valid=raw.var.feature_types.astype(str).eq('Gene Expression').to_numpy() if 'feature_types' in raw.var else np.ones(raw.n_vars,bool)
    values=raw.X[order][:,valid].tocsr().astype(np.float32)
    if not np.isfinite(values.data).all() or (values.data<0).any() or not np.allclose(values.data,np.round(values.data)):raise ValueError('Raw X is not original counts')
    a=ad.AnnData(values,obs=obs,var=pd.DataFrame(index=genes[valid]));a.obsm['spatial']=xy
    if not a.var_names.is_unique:raise ValueError('Nonunique gene symbols')
    if c.get('split_unit')=='joint_all':
        fold=np.zeros(len(obs),np.int8);mapping={str(core):0 for core in obs.core.unique()}
    else:
        splitter=fragment_folds if c.get('split_unit')=='tissue_fragment' else patient_folds
        fold,mapping=splitter(obs,c['seed'])
    a.obs['fold']=fold
    a.uns['patient_split']=mapping;a.uns['coordinate_units']='micrometers; original x_centroid/y_centroid'
    temporary=target.with_name('tissue.incomplete.h5ad')
    a.write_h5ad(temporary,compression='gzip');temporary.replace(target)
    print('Prepared',a.shape,'fragments',a.obs.core.nunique(),'split',c.get('split_unit','patient'),flush=True)
    return a


def sample_rows(a,c):
    if c.get('split_unit')=='joint_all':
        # Every retained center is optimized; monitoring is a known training subset.
        rng=np.random.default_rng(c['seed']);monitor=[]
        valid=a.obs.analysis_center.to_numpy(bool) if 'analysis_center' in a.obs else np.ones(len(a),bool)
        for patient in sorted(a.obs.patient.unique()):
            ids=np.flatnonzero((a.obs.patient.to_numpy()==patient)&valid)
            monitor.extend(rng.choice(ids,min(len(ids),c['evaluation_centers_per_patient']),replace=False))
        return [np.flatnonzero(valid),np.sort(monitor).astype(np.int64),np.empty(0,np.int64)]
    rng=np.random.default_rng(c['seed']);rows=[]
    for fold in range(3):
        chosen=[];budget=c['train_centers_per_patient'] if fold==0 else c['evaluation_centers_per_patient']
        for patient in sorted(a.obs.loc[a.obs.fold==fold,'patient'].unique()):
            ids=np.flatnonzero((a.obs.patient.to_numpy()==patient)&(a.obs.fold.to_numpy()==fold))
            chosen.extend(rng.choice(ids,min(len(ids),budget),replace=False))
        rows.append(np.sort(chosen))
    return rows


def build_graph(a):
    xy=np.asarray(a.obsm['spatial'],np.float32);n=len(a)
    members=np.empty((n,30),np.int64);weights=np.empty((n,30),np.float32)
    for core in sorted(a.obs.core.unique()):
        ids=np.flatnonzero(a.obs.core.to_numpy()==core);m,w=neighborhoods(xy[ids])
        members[ids]=ids[m];weights[ids]=w
    if not (a.obs.core.to_numpy()[members]==a.obs.core.to_numpy()[:,None]).all():raise AssertionError('Cross-fragment edge')
    return members,weights


def prepare(c,out,tissue=None):
    cache=out/'input';cache.mkdir(parents=True,exist_ok=True)
    a=load_tissue(c,out) if tissue is None else tissue
    rows=sample_rows(a,c);xy=np.asarray(a.obsm['spatial'],np.float32)
    if (cache/'members.npy').exists():members=np.load(cache/'members.npy');weights=np.load(cache/'weights.npy')
    else:
        members,weights=build_graph(a);np.save(cache/'members.npy',members);np.save(cache/'weights.npy',weights)
    fold=a.obs.fold.to_numpy(np.int8);physical=np.unique(members[rows[0]])
    if c.get('split_unit')!='joint_all':
        for i in range(3):
            if not (fold[members[rows[i]]]==i).all():raise AssertionError('Patient leakage')
    types=tuple(sorted(a.obs.cell_type.unique()));codes=pd.Categorical(a.obs.cell_type,categories=types).codes.astype(np.int64)
    x=a.X;genes=tuple(a.var_names);n=len(a);t=len(types)
    atlas=load_lr_atlas(c['lr_atlas']);panel=set(genes)
    measurable=tuple(v for v in atlas.interactions if set(v.ligand_components+v.receptor_components)<=panel)
    lrgenes=tuple(sorted({g for v in measurable for g in v.ligand_components+v.receptor_components}))
    indices=a.var_names.get_indexer(lrgenes);trainx=x[physical][:,indices].toarray()
    ligand_train,receptor_train=_side_expression(trainx,lrgenes,LRAtlas(measurable,{},c['lr_atlas']));del trainx
    coverage_l=np.zeros((t,len(measurable)));coverage_r=coverage_l.copy()
    for j in range(t):
        ids=codes[physical]==j
        if ids.any():coverage_l[j]=(ligand_train[ids]>0).mean(0);coverage_r[j]=(receptor_train[ids]>0).mean(0)
    dummy=np.zeros((n,1),np.float32)
    print('Physical-pair opportunities',flush=True)
    _,_,opp=aggregate_pairwise_ccc(xy,codes,dummy,dummy,t,20.,members,weights,batch_size=256,device=c['device'],compute_signal=False)
    group_column='image' if 'image' in a.obs else 'patient'
    groups=a.obs[group_column].astype(str).to_numpy()
    mask,scope_audit=eligible_ccc_features(ligand_train,receptor_train,codes[physical],groups[physical],
        opp[rows[0]],groups[rows[0]],t,c['coverage_fraction'],c['opportunity_fraction'],c.get('feature_scope','pooled_training'))
    excluded=np.isin(np.asarray(types),c.get('excluded_ccc_types',[]))
    if excluded.any():mask&=~(excluded[:,None]|excluded[None,:]).reshape(-1,1)
    del ligand_train,receptor_train
    (cache/'feature_scope_audit.json').write_text(json.dumps({'scope':c.get('feature_scope','pooled_training'),
        'uses_condition_labels':False,'coverage':c['coverage_fraction'],'samples':scope_audit},indent=2))
    used=mask.any(0);selected_lr=tuple(v for v,b in zip(measurable,used) if b);mask=mask[:,used]
    if mask.sum()<32:raise ValueError('Fewer than 32 supported CCC features')
    lrgenes=tuple(sorted({g for v in selected_lr for g in v.ligand_components+v.receptor_components}));li=a.var_names.get_indexer(lrgenes)
    signal=x[:,li].toarray();ligand,receptor=_side_expression(signal,lrgenes,LRAtlas(selected_lr,{},c['lr_atlas']));del signal
    flat=np.flatnonzero(mask.ravel());pair,lrid=np.divmod(flat,len(selected_lr));sender,receiver=np.divmod(pair,t)
    features=pd.DataFrame({'feature_id':np.arange(len(flat)),'flat_feature_index':flat,
        'sender':np.asarray(types)[sender],'receiver':np.asarray(types)[receiver],
        'lr_id':[selected_lr[j].lr_id for j in lrid],'ligand':[selected_lr[j].ligand for j in lrid],'receptor':[selected_lr[j].receptor for j in lrid]})
    features['ccc']=features.sender+' → '+features.receiver+' | '+features.ligand+'–'+features.receptor
    path=cache/'ccc.npy'
    if not path.exists():
        print('CCC:',n,'centers ×',len(features),'features;',len(selected_lr),'LR pairs',flush=True)
        value=np.lib.format.open_memmap(cache/'ccc.incomplete.npy',mode='w+',dtype=np.float32,shape=(n,len(features)))
        # The original kernel scatters LR chunks across columns. Scatter into
        # RAM first, then write contiguous NAS rows to avoid random mmap I/O.
        for start in range(0,n,512):
            stop=min(start+512,n)
            _,local,_=aggregate_pairwise_ccc(xy,codes,ligand,receptor,t,20.,members[start:stop],weights[start:stop],mask,
                batch_size=128,lr_batch_size=32,device=c['device'])
            value[start:stop]=local
            if start%5120==0:print(f'CCC rows {stop:,}/{n:,}',flush=True)
        value.flush();del value;(cache/'ccc.incomplete.npy').replace(path)
    matrix=np.load(path,mmap_mode='r');del ligand,receptor,opp
    if matrix.shape!=(n,len(features)):raise ValueError('CCC cache feature mismatch')
    mean,ms,var,nz=feature_statistics(matrix,rows[0]);effective=(nz>=c['feature_nonzero_fraction'])&(var>np.maximum(np.finfo(np.float32).eps*ms,np.finfo(np.float32).tiny))
    if effective.sum()<32:raise ValueError('Too few variable CCC supervision features')
    rms=np.maximum(np.sqrt(ms),1e-6).astype(np.float32);global_scale=max(float(np.median(rms[effective])),1e-6)
    scale=(global_scale*(rms/global_scale)**.5).astype(np.float32)
    totals=np.asarray(x.sum(1)).ravel();expression=expression_block(x,physical,np.arange(len(genes)),totals)
    encoder,gene_audit=select_encoder_genes(genes,features.loc[effective],expression.var(0),c['encoder_genes'],'lr_plus_variance')
    e=expression[:,encoder];del expression
    comp=np.empty((n,t),np.float32)
    for start in range(0,n,256):
        m=members[start:start+256]
        w=physical_weights(torch.from_numpy(xy[m]),torch.from_numpy(weights[start:start+len(m)]),20.,torch.from_numpy(m))
        comp[start:start+len(m)]=pair_composition(torch.from_numpy(codes[m]),w,t).numpy()
    audit={'scope':c.get('audit_scope','Independent downstream colitis niche cohort; no clinical CCC/Cox changes'),
        'cells':n,'genes':len(genes),'patients':a.obs.patient.nunique(),'fragments':a.obs.core.nunique(),
        'cell_types':types,'atlas_LR':len(atlas),'panel_measurable_LR':len(measurable),'retained_LR':len(selected_lr),
        'CCC_features':len(features),'effective_CCC_features':int(effective.sum()),'encoder_gene_selection':gene_audit,
        'split':('All cells and all fragments optimized; fixed training-subset monitor; no held-out test'
                 if c.get('split_unit')=='joint_all' else 'All nine donors in training; whole tissue fragments held out; not independent-patient evaluation'
                 if c.get('split_unit')=='tissue_fragment' else 'One patient per condition per train/validation/test; phenotype never supervised'),
        'monitor_role':'training_subset' if c.get('split_unit')=='joint_all' else 'validation',
        'split_unit':c.get('split_unit','patient'),
        'split_sizes':[len(v) for v in rows],'patient_split':{str(k):int(v) for k,v in a.uns['patient_split'].items()},
        'counts':'Original integer counts, aligned by cell ID; no imputation',
        'CCC_definition':'Original all distinct directed physical pairs, KNN30, sigma20, opportunity normalization',
        'coverage_fit':'Training physical cells only, 10%; opportunity support training centers only',
        'feature_scope':c.get('feature_scope','pooled_training'),
        'cell_type_encoder_input':False,'composition_loss':False,'biological_alignment_loss':False,
        'BayesPrism_clinical_endpoint_matching':'Not applicable to independent observational spatial niche cohort'}
    data=GraphData(x,totals,genes,encoder,e.mean(0),np.maximum(e.std(0),.1),codes,types,xy,members,weights,
        Targets(matrix,np.arange(len(features))),scale,effective,features,a.obs.reset_index(drop=True),rows,fold,
        pd.Categorical(a.obs.patient).codes,comp,audit)
    data.feature_mean=mean;data.feature_nonzero_fraction=nz
    features.to_csv(cache/'features.csv',index=False);pd.DataFrame({'gene':np.asarray(genes)[encoder]}).to_csv(cache/'encoder_genes.csv',index=False)
    np.save(cache/'composition.npy',comp);np.savez(cache/'preprocessing.npz',feature_mask=effective,feature_scale=scale,feature_nonzero_fraction=nz,encoder_indices=encoder,gene_mean=data.gene_mean,gene_scale=data.gene_scale)
    (cache/'audit.json').write_text(json.dumps(audit,indent=2)+'\n');print('Input ready',audit['effective_CCC_features'],'effective features',flush=True)
    return a,data


def summarize(a,data,z,h,labels,out,k):
    frame=a.obs.reset_index(drop=True).copy();frame['niche']=labels+1;frame['x']=data.coordinates[:,0];frame['y']=data.coordinates[:,1]
    frame.to_csv(out/'assignments.csv.gz',index=False)
    freq=pd.crosstab([frame.patient,frame.condition],frame.niche,normalize='index').reindex(columns=np.arange(1,k+1),fill_value=0)
    patient_fold=frame.groupby('patient',observed=True).fold.agg(lambda x:int(x.iloc[0]) if x.nunique()==1 else -1)
    freq.insert(0,'fold',[int(patient_fold.loc[p]) for p,_ in freq.index])
    freq.to_csv(out/'patient_niche_fractions.csv')
    np.save(out/'niche_composition.npy',np.stack([data.composition_target[labels==j].mean(0) for j in range(k)]))
    mean=np.stack([z[labels==j].mean(0) for j in range(k)])
    pd.DataFrame(mean,index=np.arange(1,k+1),columns=np.arange(1,33)).to_csv(out/'niche_program_activity.csv',index_label='niche')
    ccc=mean@h;top=[]
    for j in range(k):
        for rank,f in enumerate(np.argsort(-ccc[j])[:10],1):
            r=data.features.iloc[f]
            top.append({'niche':j+1,'rank':rank,'sender':r.sender,'receiver':r.receiver,'LR':r.ligand+'–'+r.receptor,
                'mean_reconstructed_CCC':float(ccc[j,f]),'main_program':int(np.argmax(mean[j]*h[:,f]))+1})
    pd.DataFrame(top).to_csv(out/'niche_top10_CCC.csv',index=False)
    program=[]
    for j in range(32):
        for rank,f in enumerate(np.argsort(-h[j])[:8],1):
            r=data.features.iloc[f];program.append({'program':j+1,'rank':rank,'sender':r.sender,'receiver':r.receiver,
                'LR':r.ligand+'–'+r.receptor,'loading':float(h[j,f])})
    pd.DataFrame(program).to_csv(out/'program_top_CCC.csv',index=False)
    patient_meta=frame[['patient','condition']].drop_duplicates().set_index('patient');patient_meta['fold']=patient_fold
    activities=pd.DataFrame(np.log1p(z),index=frame.patient).groupby(level=0).mean().join(patient_meta)
    activities.to_csv(out/'patient_program_activity.csv')
    # Post-fit disease enrichment is descriptive, with patients equally weighted.
    f=freq.reset_index();group=f.groupby('condition')[np.arange(1,k+1)].mean().reindex(CONDITIONS)
    group.to_csv(out/'condition_niche_fractions.csv')
    disease=[]
    for condition in ('UC','ICI'):
        for niche in range(1,k+1):
            observed=float(group.loc[condition,niche]);healthy=float(group.loc['HC',niche])
            disease.append({'condition':condition,'niche':niche,'mean_patient_fraction':observed,'HC_mean_fraction':healthy,
                'log2_ratio_vs_HC':float(np.log2((observed+.005)/(healthy+.005)))})
    pd.DataFrame(disease).to_csv(out/'condition_niche_enrichment.csv',index=False)


def load_prepared(c,out):
    """Reuse the exact training inputs without rebuilding CCC or gene selection."""
    cache=out/'input'
    if json.loads((out/'cohort_config.json').read_text())!=c:
        raise ValueError('Prepared input configuration differs from this run')
    a=ad.read_h5ad(cache/'tissue.h5ad');audit=json.loads((cache/'audit.json').read_text())
    p=np.load(cache/'preprocessing.npz');features=pd.read_csv(cache/'features.csv')
    matrix=np.load(cache/'ccc.npy',mmap_mode='r');members=np.load(cache/'members.npy',mmap_mode='r')
    weights=np.load(cache/'weights.npy',mmap_mode='r');comp=np.load(cache/'composition.npy',mmap_mode='r')
    types=tuple(audit['cell_types']);codes=pd.Categorical(a.obs.cell_type,categories=types).codes.astype(np.int64)
    rows=sample_rows(a,c)
    if (a.shape!=(audit['cells'],audit['genes']) or matrix.shape!=(len(a),len(features))
        or members.shape!=(len(a),30) or weights.shape!=members.shape or comp.shape!=(len(a),len(types))
        or [len(r) for r in rows]!=audit['split_sizes'] or (codes<0).any()):
        raise ValueError('Prepared training input dimensions/labels differ')
    data=GraphData(a.X,np.asarray(a.X.sum(1)).ravel(),tuple(a.var_names),p['encoder_indices'],
        p['gene_mean'],p['gene_scale'],codes,types,np.asarray(a.obsm['spatial'],np.float32),members,weights,
        Targets(matrix,np.arange(len(features))),p['feature_scale'],p['feature_mask'],features,
        a.obs.reset_index(drop=True),rows,a.obs.fold.to_numpy(np.int8),pd.Categorical(a.obs.patient).codes,comp,audit)
    if 'feature_nonzero_fraction' in p:data.feature_nonzero_fraction=p['feature_nonzero_fraction']
    else:data.feature_nonzero_fraction=feature_statistics(matrix,rows[0])[3]
    print('Reused exact prepared training inputs; no CCC recalculation',flush=True)
    return a,data


def recover_training_best(data,config):
    """Finalize the already fused native-scale best state, with zero optimizer steps."""
    output=Path(config.output_dir)
    if (output/'checkpoint.pt').exists():return restore_checkpoint(data,config)
    best=torch.load(output/'training_best.pt',map_location='cpu',weights_only=False)
    if best['config']!=config.to_dict():raise ValueError('Saved best-weight model configuration differs')
    history=pd.read_csv(output/'training.csv');row=history.loc[history.epoch==best['epoch']]
    if len(row)!=1:raise ValueError('Saved best epoch is missing from training history')
    model=model_from_config(data,config);model.load_state_dict(best['model_state'])
    # training_best.pt already folds activity_parameter_scale into the head.
    # Reapplying that scale would corrupt Z and the CCC reconstruction.
    payload={'model_state':best['model_state'],'config':config.to_dict(),'genes':data.genes,
        'categories':data.categories,'encoder_indices':data.encoder_indices,'features':data.features.to_dict('list'),
        'gene_mean':data.gene_mean,'gene_scale':data.gene_scale,'feature_scale':data.feature_scale,
        'feature_mask':data.feature_mask,'feature_mean':data.feature_mean,
        'feature_nonzero_fraction':data.feature_nonzero_fraction,'folds':data.folds,
        'architecture':config.architecture,'best_epoch':int(best['epoch']),'data_audit':data.audit,
        'activity_parameter_scale':best.get('activity_parameter_scale',1.),
        'expression_normalization':data.expression_normalization,'library_mean':data.library_mean,
        'library_scale':data.library_scale,'recovered_from':'training_best.pt','additional_training_epochs':0}
    temporary=output/'checkpoint.incomplete.pt';torch.save(payload,temporary);temporary.replace(output/'checkpoint.pt')
    (output/'config.json').write_text(json.dumps(config.to_dict(),indent=2)+'\n')
    metrics={'best_epoch':int(best['epoch']),'monitor_loss':float(row.iloc[0].selection_loss),
        'monitor_role':'training_subset','test':None,'training_centers':len(data.rows[0]),
        'monitor_centers':len(data.rows[1]),'completed_epochs':int(history.epoch.max()),
        'additional_training_epochs':0,'training_stop':'user_request'}
    (output/'training_monitor_metrics.json').write_text(json.dumps(metrics,indent=2)+'\n')
    print('Recovered best epoch',best['epoch'],'without further training',flush=True)
    return model.to(config.device)


def restore_checkpoint(data,config):
    """Restore the shared model with cohort-specific input identity checks."""
    p=torch.load(Path(config.output_dir)/'checkpoint.pt',map_location='cpu',weights_only=False)
    if p['config']!=config.to_dict() or tuple(p['genes'])!=data.genes or tuple(p['categories'])!=data.categories:
        raise ValueError('Checkpoint configuration/panel differs; refusing overwrite')
    for key in ('encoder_indices','gene_mean','gene_scale','feature_scale','feature_mask','folds'):
        if not np.array_equal(p[key],getattr(data,key)):raise ValueError('Checkpoint input changed: '+key)
    if p['features']!=data.features.to_dict('list') or p['data_audit']['patient_split']!=data.audit['patient_split']:
        raise ValueError('Checkpoint directed features or patient split differs')
    model=model_from_config(data,config);model.load_state_dict(p['model_state'])
    return model.to(config.device)


def clustering_representation(data,arrays,h,c):
    if c.get('clustering_representation')=='program_mixture_activity':
        activity=np.log1p(np.asarray(arrays['activity'][data.rows[0]]))
        values=_latent_block(arrays['P'],arrays['activity'],slice(None),float(np.median(activity)),
                             max(float(np.std(activity)),.1),.25)
        projection=None
    else:
        projection=dictionary_program_projection(h,data.feature_scale)
        values=np.asarray(arrays['Z'])@projection
    # Monitor centers also belong to training: pass physical ownership only once.
    ownership=[data.rows[0],np.empty(0,np.int64),np.empty(0,np.int64)] if c.get('split_unit')=='joint_all' else data.rows
    return spatial_cluster_features(values,data.members,data.weights,ownership,c['spatial_mixing']),projection


def run(path,postprocess_only=False):
    torch.set_num_threads(8)
    c=read_config(path);out=Path(c['output_dir']);out.mkdir(parents=True,exist_ok=True)
    signature=hashlib.sha256(json.dumps(c,sort_keys=True).encode()).hexdigest()
    complete=out/'complete.json'
    stopped=(out/'training_stopped.json').exists()
    if complete.exists():
        saved=json.loads(complete.read_text())
        if saved['config_signature']!=signature:raise ValueError('Configuration differs from completed results; refusing overwrite')
        required=('summary.json','assignments.csv.gz','gnn/checkpoint.pt','gnn/H.npy','gnn/Z.npy',
                  'ccc_reconstruction.npy','patient_niche_fractions.csv','program_top_CCC.csv')
        if all((out/f).is_file() for f in required):return out
    if stopped or postprocess_only:
        if not any((out/'gnn'/name).exists() for name in ('checkpoint.pt','training_best.pt')):
            raise RuntimeError('No saved weights for inference; postprocessing cannot start training')
        a,data=load_prepared(c,out)
    else:
        (out/'cohort_config.json').write_text(json.dumps(c,indent=2)+'\n')
        a,data=prepare(c,out)
    torch.set_num_threads(8)
    g=GNNConfig(seed=c['seed'],programs=32,hidden=c['hidden'],encoder_genes=len(data.encoder_indices),
        architecture='expression_gnn_residual_v2' if c['message_layers']>1 else 'expression_gnn_composition_v1',
        message_layers=c['message_layers'],batch_size=c['batch_size'],max_epochs=c['epochs'],patience=c['patience'],
        device=c['device'],lambda_comp=0.,lambda_reg=.005,dictionary_initialization='nmf',
        output_dir=str(out/'gnn'),k_min=4,k_max=15,clustering_sample_size=18000,silhouette_sample_size=1500,near_optimal_score_tolerance=0.)
    objective=partial(ccc_loss,kind='huber',delta=3.)
    if (out/'gnn/checkpoint.pt').exists():model=restore_checkpoint(data,g)
    elif stopped or postprocess_only:model=recover_training_best(data,g)
    else:model=fit(data,g,rescale_activity=True,ccc_objective=objective)
    if getattr(data,'_resident',None) is None:
        data.prepare_device(next(model.parameters()).device,out/'gnn/cache')
    arrays=infer(model,data,out/'gnn',c['batch_size']);z=np.asarray(arrays['Z']);h=np.load(out/'gnn/H.npy')
    values,projection=clustering_representation(data,arrays,h,c)
    k,labels,_=discover(data,arrays,h,g,program_projection=projection,clustering_values=values,full_kmeans=True)
    pred=np.lib.format.open_memmap(out/'ccc_reconstruction.npy',mode='w+',dtype=np.float32,shape=data.targets.shape)
    for start in range(0,len(z),1024):pred[start:start+1024]=z[start:start+1024]@h
    pred.flush();del pred
    summarize(a,data,z,h,labels,out,k)
    summary={**data.audit,'selected_K':k,'programs':32,'loss':'Feature-scaled Huber(delta=3) CCC + 0.005 regularization',
        'expression_decoder':False,'training_config':g.to_dict(),'spatial_mixing':c['spatial_mixing'],
        'clustering_representation':c.get('clustering_representation','native_CCC_dictionary_metric')}
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    complete.write_text(json.dumps({'config_signature':signature,'model_source':str(Path(__file__))},indent=2)+'\n')
    print('ANALYSIS COMPLETE',out,flush=True);return out
