"""Joint, expression-only GNN learning with sample-held-out evaluation."""
from pathlib import Path
from types import SimpleNamespace
from functools import partial
import json
import hashlib
import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from scipy.optimize import linear_sum_assignment
from scipy.stats import spearmanr
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, f1_score
from ..ccc.pairwise import _side_expression, aggregate_pairwise_ccc
from ..ccc.lr_atlas import LRInteraction, LRAtlas
from ..data import feature_statistics, resolve_device
from ..gnn.data import GraphData, Targets, neighborhoods, expression_block, pair_composition
from ..gnn.model import physical_weights
from ..gnn.config import GNNConfig
from ..gnn.training import fit, infer
from ..gnn.pipeline import load_checkpoint
from ..gnn.losses import ccc_loss
from ..gnn.evaluation import discover
from ..gnn.spatial_refinement import spatial_cluster_features
from .paired_tissues import PairedConfig,TYPES,NICHES,PROGRAMS,generate
from .counterfactual_controls import coverage_supervision_mask
from .baselines import dictionary_program_projection
from .response_simulation import checksum


def encoder_panel(a,lr,training_cells,config):
    """Retain all LR; optional declared markers, audited on training counts.

    Neither cell labels, condition, truth, nor held-out counts select genes.
    This simulator's non-LR state effects are nuisance, not true CCC drivers.
    """
    genes=tuple(a.var_names);values=np.asarray(a.X[training_cells],np.float32)
    required=set(lr.ligand)|set(lr.receptor)
    if not required<=set(genes):raise ValueError('LR gene missing from panel')
    if config.encoder_gene_selection=='all':selected=np.arange(len(genes))
    else:
        coverage=(values>0).mean(0);variance=values.var(0)
        selected=np.array([i for i,g in enumerate(genes) if g in required or
                           (g.startswith('SYN_MARKER_') and coverage[i]>=.01 and variance[i]>1e-6)])
    audit={'selection':config.encoder_gene_selection,'panel_genes':len(genes),'encoder_genes':len(selected),
           'LR_genes_required':len(required),'LR_genes_retained':sum(genes[i] in required for i in selected),
           'additional_marker_genes':sum(genes[i] not in required for i in selected),
           'marker_evidence':'Predeclared synthetic cell-type markers; training-only detection >=1% and nonzero variance',
           'state_policy':'Background state genes are not injected CCC drivers; no unsupported state genes are forced in',
           'condition_celltype_or_niche_labels_used_for_selection':False}
    return selected,audit


def prepare(a,lr,config,out):
    """Only the adapter is new: exact production CCC and GNN remain unchanged."""
    cache=Path(out)/'input';cache.mkdir(parents=True,exist_ok=True)
    xy=np.asarray(a.obsm['spatial'],np.float32);n=len(a)
    types=pd.Categorical(a.obs.cell_type,categories=TYPES).codes.astype(np.int64)
    if not a.obs.cell_id.is_unique or (types<0).any():raise ValueError('Invalid cell identities/types')
    # Coordinates are local to slices; explicit sample-wise queries prevent
    # accidental edges between overlapping healthy/cancer coordinate systems.
    members=np.empty((n,30),np.int64);weights=np.empty((n,30),np.float32)
    samples=a.obs.sample_id.to_numpy()
    for sample in np.unique(samples):
        ids=np.flatnonzero(samples==sample);m,w=neighborhoods(xy[ids])
        members[ids]=ids[m];weights[ids]=w
    folds=a.obs.fold.to_numpy(np.int8);rows=[np.flatnonzero(folds==f) for f in range(3)]
    if config.training_centers_per_slice:
        chosen=[]
        for pair in np.unique(a.obs.pair_id.to_numpy()[folds==0]):
            pair_rng=np.random.default_rng(config.seed+int(pair)*589+73)
            local=pair_rng.choice(config.cells_per_slice,config.training_centers_per_slice,replace=False)
            # Matched local indices preserve paired composition at the
            # supervised-center level as well as the whole-slice level.
            for sample in np.unique(samples[(a.obs.pair_id.to_numpy()==pair)&(folds==0)]):
                chosen.extend(np.flatnonzero(samples==sample)[local])
        rows[0]=np.sort(chosen)
    if not (samples[members]==samples[:,None]).all() or not (folds[members]==folds[:,None]).all():
        raise AssertionError('Cross-sample or cross-fold physical pair')
    genes=tuple(a.var_names);raw=np.asarray(a.X,np.float32)
    features=pd.DataFrame([{'feature_id':i,'sender':s,'receiver':r,'lr_id':v.lr_id,
                            'ligand':v.ligand,'receptor':v.receptor,'ccc':f'{s} → {r} | {v.ligand}–{v.receptor}'}
                           for i,(s,r,v) in enumerate((s,r,v) for s in TYPES for r in TYPES for v in lr.itertuples())])
    path=cache/'ccc.npy'
    if not path.exists():
        atlas=LRAtlas(tuple(LRInteraction(str(v.lr_id),v.ligand,v.receptor,(v.ligand,),(v.receptor,)) for v in lr.itertuples()),{},'Public LR identities; synthetic roles')
        ligand,receptor=_side_expression(raw,genes,atlas)
        c=np.lib.format.open_memmap(cache/'ccc.incomplete.npy',mode='w+',dtype=np.float32,shape=(n,len(features)))
        aggregate_pairwise_ccc(xy,types,ligand,receptor,5,20.,members,weights,
                              batch_size=config.batch_size,lr_batch_size=16,
                              device=resolve_device(config.device),communication_out=c)
        c.flush();del c;(cache/'ccc.incomplete.npy').replace(path)
    matrix=np.load(path,mmap_mode='r')
    if matrix.shape!=(n,len(features)) or not np.isfinite(matrix).all() or (matrix<0).any():
        raise ValueError('CCC shape/values invalid')
    mean,ms,var,nz=feature_statistics(matrix,rows[0])
    mask=(nz>=.005)&(var>np.maximum(np.finfo(np.float32).eps*ms,np.finfo(np.float32).tiny))
    labels=SimpleNamespace(members=members,rows=rows,types=types,categories=TYPES,features=features,feature_mask=mask)
    mask=coverage_supervision_mask(a,labels,.1)
    if mask.sum()<32:raise ValueError('Too few effective CCC features')
    rms=np.maximum(np.sqrt(ms),1e-6).astype(np.float32);global_scale=max(float(np.median(rms[mask])),1e-6)
    scale=(global_scale*(rms/global_scale)**.5).astype(np.float32)
    totals=raw.sum(1);x=sp.csr_matrix(raw);train=np.unique(members[rows[0]])
    e=expression_block(x,train,np.arange(len(genes)),totals)
    panel,gene_audit=encoder_panel(a,lr,train,config)
    encoder=panel[np.argsort(-e[:,panel].var(0),kind='stable')];selected=e[:,encoder]
    comp=np.empty((n,5),np.float32)
    for start in range(0,n,config.batch_size):
        local=members[start:start+config.batch_size]
        w=physical_weights(torch.from_numpy(xy[local]),torch.from_numpy(weights[start:start+len(local)]),20.,torch.from_numpy(local))
        comp[start:start+len(local)]=pair_composition(torch.from_numpy(types[local]),w,5).numpy()
    audit={'scope':'Independent downstream synthetic experiment; clinical CCC/Cox untouched',
           'source_signature':config.to_dict(),
           'cell_order_sha256':hashlib.sha256(pd.util.hash_pandas_object(a.obs.cell_id,index=False).to_numpy().tobytes()).hexdigest(),
           'encoder_inputs':'Expression + within-slice physical graph only',
           'available_expression_genes':len(genes),'expression_genes':len(encoder),'all_LR_genes_in_encoder':True,
           'encoder_gene_selection':gene_audit,
           'fixed_CCC_features':len(features),'effective_CCC_features':int(mask.sum()),
           'CCC':'Original directed all-distinct-physical-pairs formula, KNN30, sigma20, opportunity normalization',
           'split':'Predeclared whole-pair holdout, both conditions in each fold',
           'split_sizes':[len(r) for r in rows], 'feature_mask_fit':'Training only; 10% type-specific expression, 0.5% nonzero and variance',
           'training_slices':int(a.obs.loc[folds==0,'sample_id'].nunique()),
           'training_physical_cells':len(train),'training_centers_per_slice':config.training_centers_per_slice,
           'condition_or_true_niche_used_in_training':False,'composition_loss':0,'regularization_loss':.005}
    cells=a.obs[['cell_id','cell_type']].reset_index(drop=True)
    d=GraphData(x,totals,genes,encoder,selected.mean(0),np.maximum(selected.std(0),.1),types,TYPES,xy,members,weights,
                Targets(matrix,np.arange(len(features))),scale,mask,features,cells,rows,folds,a.obs.pair_id.to_numpy(),comp,audit)
    d.original_feature_scale=rms;d.feature_mean=mean;d.feature_nonzero_fraction=nz
    for name,value in [('members',members),('weights',weights),('folds',folds),('composition',comp)]:np.save(cache/(name+'.npy'),value)
    np.save(cache/'training_rows.npy',rows[0])
    features.to_csv(cache/'features.csv',index=False)
    np.savez(cache/'preprocessing.npz',feature_mask=mask,feature_scale=scale,encoder_indices=encoder,gene_mean=d.gene_mean,gene_scale=d.gene_scale)
    (cache/'audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    pd.DataFrame({'gene':[genes[i] for i in encoder],
                  'reason':['LR endpoint' if genes[i] in (set(lr.ligand)|set(lr.receptor)) else 'Declared marker' if config.encoder_gene_selection=='lr_markers' else 'All-panel baseline' for i in encoder]}).to_csv(cache/'encoder_genes.csv',index=False)
    return d


def evaluation(a,lr,data,z,h,labels,out,k):
    """Truth/condition are first consumed after model and clustering are fixed."""
    out=Path(out);truth=a.obs.true_niche.to_numpy();train=data.rows[0];test=data.rows[2]
    counts=np.zeros((k,len(NICHES)),int)
    np.add.at(counts,(labels[train],truth[train]),1)
    ri,ci=linear_sum_assignment(-counts);mapping={int(r):int(c) for r,c in zip(ri,ci)}
    aligned=np.array([mapping.get(int(v),-1) for v in labels])
    frame=a.obs.copy();frame['niche']=labels+1;frame['evaluation_aligned_niche']=aligned
    frame.to_csv(out/'assignments.csv.gz',index=False)
    frequency=pd.crosstab([frame.sample_id,frame.condition,frame.fold],frame.niche,normalize='index').reindex(columns=np.arange(1,k+1),fill_value=0)
    frequency.to_csv(out/'sample_niche_fractions.csv')
    f=frequency.reset_index();discovery=f[f.fold==0]
    healthy=discovery[discovery.condition=='Healthy'][np.arange(1,k+1)].mean().to_numpy()
    cancer=discovery[discovery.condition=='Cancer'][np.arange(1,k+1)].mean().to_numpy()
    # Candidate selection uses training slice frequencies only, never truth or
    # held-out group labels. Specific means near absent in training controls.
    candidate=(healthy<.02)&(cancer>.03)&((cancer+.005)/(healthy+.005)>3)
    enrichment=pd.DataFrame({'niche':np.arange(1,k+1),'healthy_train_fraction':healthy,'cancer_train_fraction':cancer,
                             'log2_enrichment':np.log2((cancer+.005)/(healthy+.005)),
                             'disease_specific_candidate':candidate})
    for condition in ('Healthy','Cancer'):
        enrichment[condition.lower()+'_test_fraction']=f[(f.fold==2)&(f.condition==condition)][np.arange(1,k+1)].mean().to_numpy()
    enrichment.to_csv(out/'disease_enrichment.csv',index=False)
    pred_disease=np.isin(labels+1,enrichment.loc[candidate,'niche']);true_disease=np.isin(truth,[4,5])
    metrics=[]
    for scope,rows in [('all_slices_descriptive',np.arange(len(a))),('heldout_slices',test)]:
        metrics.append({'scope':scope,'K':k,'cells':len(rows),'ARI':adjusted_rand_score(truth[rows],labels[rows]),
                        'NMI':normalized_mutual_info_score(truth[rows],labels[rows]),
                        'disease_niche_F1':f1_score(true_disease[rows],pred_disease[rows],zero_division=0),
                        'candidate_niches':int(candidate.sum())})
    pd.DataFrame(metrics).to_csv(out/'metrics.csv',index=False)
    # No forced 1:1 match: 32 learned dimensions may split a true MI.
    true_h=np.zeros((8,len(data.features)),np.float32)
    for p,(_,sender,receiver) in enumerate(PROGRAMS):
        group=lr[lr.program_id==p]
        ids=np.flatnonzero((data.features.sender==TYPES[sender])&(data.features.receiver==TYPES[receiver])&data.features.lr_id.isin(group.lr_id))
        true_h[p,ids]=1/len(group)
    # Opportunity-weighted activities use the same physical cells and weights.
    truth_activity=np.zeros((len(a),8),np.float32);injected=np.asarray(a.obsm['true_program_activity'])
    for start in range(0,len(a),128):
        local=data.members[start:start+128];t=data.types[local]
        w=physical_weights(torch.from_numpy(data.coordinates[local]),torch.from_numpy(data.weights[start:start+len(local)]),20.,torch.from_numpy(local)).numpy()
        for p,(_,sender,receiver) in enumerate(PROGRAMS):
            allowed=w*(t[:,:,None]==sender)*(t[:,None,:]==receiver)
            activity=np.sqrt(injected[local,p][:,:,None]*injected[local,p][:,None,:])
            truth_activity[start:start+len(local),p]=(allowed*activity).sum((1,2))/(allowed.sum((1,2))+1e-4)
    np.save(out/'ground_truth/microenvironment_activity.npy',truth_activity);np.save(out/'ground_truth/dictionary.npy',true_h)
    similarity=true_h@h.T/np.maximum(np.linalg.norm(true_h,axis=1)[:,None]*np.linalg.norm(h,axis=1)[None,:],1e-12)
    matches=[]
    for p in range(8):
        j=int(similarity[p].argmax())
        correlation=spearmanr(truth_activity[test,p],z[test,j]).statistic
        matches.append({'true_program':PROGRAMS[p][0],'learned_program':j+1,'dictionary_cosine':float(similarity[p,j]),
                        'heldout_activity_spearman':float(correlation) if np.isfinite(correlation) else None})
    pd.DataFrame(matches).to_csv(out/'program_recovery.csv',index=False)
    top=[];gene=[]
    for niche in range(k):
        r=train[labels[train]==niche]
        if not len(r):continue
        mean=z[r].mean(0);profile=mean@h
        for rank,j in enumerate(np.argsort(-profile)[:10],1):
            v=data.features.iloc[j];top.append({'niche':niche+1,'rank':rank,'sender':v.sender,'receiver':v.receiver,
                                               'LR':f'{v.ligand}–{v.receptor}','mean_reconstructed_CCC':float(profile[j]),
                                               'main_program':int(np.argmax(mean*h[:,j]))+1})
    pd.DataFrame(top).to_csv(out/'niche_top10_CCC.csv',index=False)
    pd.DataFrame([z[train[labels[train]==j]].mean(0) for j in range(k)],index=np.arange(1,k+1)).to_csv(out/'niche_program_activity.csv')
    pd.DataFrame([data.composition_target[train[labels[train]==j]].mean(0) for j in range(k)],index=np.arange(1,k+1),columns=TYPES).to_csv(out/'niche_composition.csv')
    # Exact local type matching audit, with independently drawn expression.
    audits=[]
    for pair in np.unique(a.obs.pair_id):
        ids=np.flatnonzero(a.obs.pair_id.to_numpy()==pair);health=ids[a.obs.condition.to_numpy()[ids]=='Healthy'];case=ids[a.obs.condition.to_numpy()[ids]=='Cancer']
        audits.append({'pair_id':int(pair),'matched_types':bool(np.array_equal(data.types[health],data.types[case])),
                       'matched_coordinates':bool(np.array_equal(data.coordinates[health],data.coordinates[case])),
                       'max_weighted_composition_difference':float(abs(data.composition_target[health]-data.composition_target[case]).max()),
                       'CCC_difference_RMS':float(np.sqrt(np.mean((data.targets[health]-data.targets[case])**2)))})
    pd.DataFrame(audits).to_csv(out/'pair_audit.csv',index=False)


def run(config):
    out=generate(config);identity=out/'analysis_manifest.json'
    sources=[Path(__file__),Path(__file__).with_name('paired_tissues.py')]
    sources += [Path(__file__).parents[1]/'gnn'/f for f in ('model.py','training.py','losses.py','evaluation.py','data.py','spatial_refinement.py')]
    manifest={'config':config.to_dict(),'data':checksum(out/'data_complete.json'),'implementation':{str(p):checksum(p) for p in sources}}
    if identity.exists() and json.loads(identity.read_text())!=manifest:raise ValueError('Analysis implementation/config changed; refusing to overwrite completed model')
    identity.write_text(json.dumps(manifest,indent=2)+'\n')
    required=('summary.json','metrics.csv','assignments.csv.gz','disease_enrichment.csv','gnn/Z.npy','gnn/H.npy','reconstruction.npy')
    if (out/'complete.json').exists():
        done=json.loads((out/'complete.json').read_text())
        if not all((out/f).exists() and checksum(out/f)==done['files'][f] for f in required):raise ValueError('Completed result cache invalid')
        return out
    torch.set_num_threads(6)
    a=ad.read_h5ad(out/'tissues.h5ad');lr=pd.read_csv(out/'lr.csv');data=prepare(a,lr,config,out)
    cfg=GNNConfig(seed=config.seed,programs=32,hidden=config.hidden,encoder_genes=len(data.encoder_indices),
                  max_epochs=config.epochs,patience=config.patience,batch_size=config.batch_size,
                  lambda_comp=0,lambda_reg=.005,device=config.device,output_dir=str(out/'gnn'),
                  dictionary_initialization='nmf',nmf_initialization_max_iter=config.nmf_iterations,
                  k_min=config.k_min,k_max=config.k_max,clustering_sample_size=10000,silhouette_sample_size=1500)
    if (out/'gnn/checkpoint.pt').exists():model,_=load_checkpoint(data,cfg)
    else:model=fit(data,cfg,rescale_activity=True,ccc_objective=partial(ccc_loss,kind='huber',delta=3.))
    arrays=infer(model,data,out/'gnn',config.batch_size);z=np.asarray(arrays['Z']);h=np.load(out/'gnn/H.npy')
    reconstruction=np.lib.format.open_memmap(out/'reconstruction.npy',mode='w+',dtype=np.float32,shape=data.targets.shape)
    for start in range(0,len(a),2048):reconstruction[start:start+2048]=z[start:start+2048]@h
    reconstruction.flush();del reconstruction
    projection=dictionary_program_projection(h,data.feature_scale)
    pooled=spatial_cluster_features(z@projection,data.members,data.weights,data.rows,.2)
    k,labels,_=discover(data,arrays,h,cfg,program_projection=projection,clustering_values=pooled,full_kmeans=True)
    evaluation(a,lr,data,z,h,labels,out,k)
    summary={**data.audit,'selected_K':k,'slices':int(a.obs.sample_id.nunique()),'cells':len(a),'genes':a.n_vars,'programs':32,
             'method':'Existing NMF-initialized CCC-only GNN; CCC Huber + 0.005 regularization; existing 20% local niche pooling',
             'disease_enrichment':'Training slice proportions nominate candidates; held-out slices report their frequencies',
             'limitations':'Controlled expression-state simulation, not clinical validation; only one held-out pair; no population P values or real pathway claims'}
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    (out/'complete.json').write_text(json.dumps({'manifest':manifest,'files':{f:checksum(out/f) for f in required}},indent=2)+'\n')
    return out
