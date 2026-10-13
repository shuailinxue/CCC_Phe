"""Synthetic fixtures are mechanism tests, not evidence of clinical/biological efficacy."""
from dataclasses import replace
from pathlib import Path
import json
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
import pytest
import anndata as ad
from phenoniche.ccc.neighborhoods import build_neighborhoods
from phenoniche.ccc.pairwise import aggregate_pairwise_ccc
from phenoniche.gnn.config import GNNConfig
from phenoniche.gnn.model import MicroenvironmentGNN,physical_weights
from phenoniche.gnn.biology import (compile_prior,activity_scores,aggregate_activity,BioAlignment,read_prior,checksum)
from phenoniche.gnn.data import GraphData,Targets,neighborhoods,spatial_split,align_cells
from phenoniche.gnn.training import fit,infer
from phenoniche.gnn.losses import ccc_loss,pair_ccc,guarded_backward,contrastive_loss
from phenoniche.gnn.evaluation import discover,diagnostics,match_programs
from phenoniche.gnn.audit import protect_output
from phenoniche.gnn.pipeline import apply_control


@pytest.fixture
def synthetic(tmp_path):
    rng=np.random.default_rng(17);n=180
    xy=np.concatenate([rng.normal(size=(60,2))*15+[i*1200,0] for i in range(3)]).astype(np.float32)
    types=np.tile([0,1],90);categories=('A','B')
    genes=('L0','R0','L1','R1','TF','U1','U2','U3','D1','D2','D3','STATE')
    raw=rng.gamma(2,2,(n,len(genes))).astype(np.float32)
    raw[:,5:8]+=raw[:,0,None]*.5;raw[:,8:11]+=raw[:,1,None]*.5
    expression=sp.csr_matrix(raw);totals=raw[:,[0,1,2,3,4,11]].sum(1)
    members,weights=neighborhoods(xy)
    rows,folds,blocks=spatial_split(xy,members,block_um=500.)
    train_cells=np.unique(members[rows[0]])
    features=pd.DataFrame([{'feature_id':f,'sender':categories[s],'receiver':categories[r],'lr_id':f'LR{l}',
        'ligand':f'L{l}','receptor':f'R{l}','ccc':f'{categories[s]} → {categories[r]} | L{l}–R{l}'} for f,(s,r,l) in enumerate((s,r,l) for s in range(2) for r in range(2) for l in range(2))])
    _,ccc,_=aggregate_pairwise_ccc(xy,types,raw[:,[0,2]],raw[:,[1,3]],2,20.,members,weights,batch_size=16)
    priors={};observations={};masks={}
    up=pd.DataFrame({'tf':['TF']*4,'target':['L0','U1','U2','U3'],'weight':[1.,1.,-1.,1.], 'source':'synthetic_test','evidence':'artificial fixture'})
    down=pd.DataFrame({'ligand':['L0']*3,'target':['D1','D2','D3'],'weight':[1.,1.,1.], 'source':'synthetic_test','evidence':'artificial fixture'})
    for branch,table in [('upstream',up),('downstream',down)]:
        prior=compile_prior(table,features,genes,branch,min_targets=2);priors[branch]=prior
        scores=activity_scores(expression,prior,train_cells)
        observed,mask=aggregate_activity(torch.from_numpy(scores[members]),torch.from_numpy(types[members]),torch.from_numpy(xy[members]),torch.from_numpy(weights),torch.from_numpy(members),features,prior,categories)
        observations[branch]=observed.numpy();masks[branch]=mask.numpy()
    encoder=np.array([0,1,2,3,4,11]);sample=np.log1p(raw[train_cells][:,encoder]*10000/totals[train_cells,None])
    local=types[members];composition=np.column_stack([((local==t)*weights).sum(1)/weights.sum(1) for t in range(2)])
    nuisance=np.column_stack((np.ones(n),composition,np.log1p(totals),np.log1p(raw[:,11]))).astype(np.float32)
    cells=pd.DataFrame({'cell_id':[f'c{i}' for i in range(n)],'cell_type':[categories[i] for i in types]})
    return GraphData(expression,totals,genes,encoder,sample.mean(0),np.maximum(sample.std(0),.1),types,categories,xy,members,weights,Targets(ccc,np.arange(8)),np.maximum(np.sqrt((ccc[rows[0]]**2).mean(0)),.1),np.ones(8,bool),features,cells,rows,folds,blocks,nuisance,priors,observations,masks,{})


def make_model(data): return MicroenvironmentGNN(7,2,8,4,hidden=12)


def test_one_nonnegative_vector_per_environment_and_unique_cell_reuse(synthetic):
    batch=synthetic.batch(synthetic.rows[0][:5]);model=make_model(synthetic)
    result=synthetic.forward(model,batch)
    assert result['Z'].shape==(5,4) and result['reconstruction'].shape==(5,8)
    assert len(batch['expression'])<5*30
    assert torch.all(result['Z']>=0) and torch.all(result['H']>=0)
    assert torch.allclose(result['H'].sum(1),torch.ones(4))
    assert torch.allclose(result['Z'].sum(1),result['activity'])
    assert torch.allclose(result['reconstruction'],result['Z']@result['H'])
    assert torch.allclose(result['reconstruction'].sum(1),result['activity'])
    loss=ccc_loss(result['reconstruction'],batch['target'],synthetic.feature_scale,synthetic.feature_mask)
    loss.backward()
    assert model.cell_encoder[0].weight.grad.abs().sum()>0
    assert model.sender.weight.grad.abs().sum()>0
    assert model.receiver.weight.grad.abs().sum()>0
    assert model.raw_dictionary.grad.abs().sum()>0
    # CCC is only supervision: modifying targets cannot change an encoder forward pass.
    batch['target']=torch.zeros_like(batch['target'])
    assert torch.equal(result['Z'],synthetic.forward(model,batch)['Z'])


def test_exact_neighborhood_and_pairwise_CCC(synthetic):
    old=build_neighborhoods(synthetic.coordinates,k=30,sigma=20.)
    assert np.array_equal(old.indices.reshape(-1,30),synthetic.members)
    assert np.array_equal(old.weights.reshape(-1,30).astype(np.float32),synthetic.weights)
    rows=synthetic.rows[0][:4];local=synthetic.members[rows]
    raw=synthetic.expression.toarray()
    ligand=np.column_stack([raw[:,0] if l=='L0' else raw[:,2] for l in synthetic.features.ligand])
    receptor=np.column_stack([raw[:,1] if r=='R0' else raw[:,3] for r in synthetic.features.receptor])
    weights=physical_weights(torch.from_numpy(synthetic.coordinates[local]),torch.from_numpy(synthetic.weights[rows]),20.,torch.from_numpy(local))
    predicted=pair_ccc(torch.from_numpy(ligand[local]),torch.from_numpy(receptor[local]),torch.from_numpy(synthetic.types[local]),synthetic.features,list(synthetic.categories),weights)
    assert np.allclose(predicted.numpy(),synthetic.targets[rows],rtol=1e-5,atol=1e-5)
    assert torch.all(torch.diagonal(weights,dim1=1,dim2=2)==0)


def test_cell_order_alignment_and_permutation_invariance(synthetic):
    d=synthetic;adata=ad.AnnData(d.expression,obs=pd.DataFrame({'cell_id':d.cells.cell_id.to_numpy(),'cell_type_coarse':d.cells.cell_type.to_numpy()},index=d.cells.cell_id.to_numpy()));adata.obsm['spatial']=d.coordinates
    perm=np.random.default_rng(9).permutation(len(d.cells));shuffled=adata[perm].copy()
    cached={'cells':d.cells,'coordinates':d.coordinates}
    order=align_cells(shuffled,cached)
    assert np.array_equal(shuffled.obs.cell_id.to_numpy()[order],d.cells.cell_id)
    bad=shuffled.copy();bad.obs.iloc[0,bad.obs.columns.get_loc('cell_id')]='missing'
    with pytest.raises(ValueError,match='Missing'): align_cells(bad,cached)
    batch=d.batch(d.rows[0][:3]);model=make_model(d)
    original=d.forward(model,batch)['Z']
    perm=torch.randperm(30)
    for key in ('local_index','coordinates','anchor_weights','cell_ids'): batch[key]=batch[key][:,perm]
    assert torch.allclose(original,d.forward(model,batch)['Z'],atol=1e-6)


def test_spatial_split_has_no_shared_physical_cells(synthetic):
    d=synthetic;used=[np.unique(d.members[r]) for r in d.rows]
    for i in range(3):
        for j in range(i+1,3): assert not np.intersect1d(used[i],used[j]).size
    again=spatial_split(d.coordinates,d.members,block_um=500.)
    assert np.array_equal(again[1],d.folds)
    xy=np.column_stack((np.arange(300),np.zeros(300))).astype(np.float32)
    members,_=neighborhoods(xy)
    rows,folds,_=spatial_split(xy,members,block_um=50.)
    assert (folds<0).any()
    used=[np.unique(members[r]) for r in rows]
    assert not np.intersect1d(used[0],used[1]).size


def test_regulatory_masks_and_no_TF_mRNA_surrogate(synthetic):
    d=synthetic;prior=d.priors['upstream']
    scores=activity_scores(d.expression,prior,np.unique(d.members[d.rows[0]]))
    altered=d.expression.toarray();altered[:,4]+=1000;altered[:,0]+=500
    assert np.allclose(scores,activity_scores(sp.csr_matrix(altered),prior,np.unique(d.members[d.rows[0]])))
    assert prior.weights[4].sum()==0 and prior.weights[0].sum()==0
    missing=compile_prior(None,d.features,d.genes,'upstream')
    assert missing.mapping.shape==(8,0)
    low=compile_prior(pd.DataFrame({'tf':['TF'],'target':['MISSING'],'weight':[1.],'source':['synthetic'],'evidence':['fixture']}),d.features,d.genes,'upstream')
    assert not low.mapping.shape[1]
    # No receiver population B -> no observed mask for B response modules.
    down=d.priors['downstream'];rows=d.rows[0][:2];local=d.members[rows]
    scores=torch.ones((2,30,down.mapping.shape[1]))
    value,mask=aggregate_activity(scores,torch.zeros((2,30),dtype=torch.long),torch.from_numpy(d.coordinates[local]),torch.from_numpy(d.weights[rows]),torch.from_numpy(local),d.features,down,d.categories)
    b=down.modules.cell_type.eq('B').to_numpy()
    assert not mask[:,b].any() and torch.isfinite(value).all()


def test_prior_provenance_rejects_missing_and_false_sources(tmp_path):
    path=tmp_path/'up.tsv';path.write_text('tf\ttarget\tweight\tsource\tevidence\nTF\tT\t1\tsynthetic_test\tfixture\n')
    with pytest.raises(ValueError,match='provenance'): read_prior(path,{},'upstream')
    manifest={'upstream':{'database':'synthetic_test','version':'fixture','url':'https://example.invalid/test','sha256':checksum(path),'species':'human'}}
    with pytest.raises(ValueError,match='database'): read_prior(path,manifest,'upstream')
    assert len(read_prior(path,manifest,'upstream',synthetic=True)[0])==1
    manifest['upstream']['sha256']='wrong'
    with pytest.raises(ValueError,match='checksum'): read_prior(path,manifest,'upstream',synthetic=True)


def test_bio_gradient_and_training_only_calibration(synthetic):
    d=synthetic;branch='downstream'
    alignment=BioAlignment.fit(d.priors[branch],d.targets[:],d.nuisance,d.observations[branch],d.observed_masks[branch],d.rows[0])
    assert alignment.valid.any()
    observed=d.observations[branch].copy();observed[d.rows[2]]+=1e4
    other=BioAlignment.fit(d.priors[branch],d.targets[:],d.nuisance,observed,d.observed_masks[branch],d.rows[0])
    assert np.array_equal(alignment.slope,other.slope)
    batch=d.batch(d.rows[0][:16]);model=make_model(d);result=d.forward(model,batch)
    loss=alignment.loss(result['reconstruction'],batch['nuisance'],batch[branch],batch[branch+'_mask']);loss.backward()
    assert model.cell_encoder[0].weight.grad.abs().sum()>0 and model.raw_dictionary.grad.abs().sum()>0
    zero=alignment.loss(result['reconstruction'],batch['nuisance'],batch[branch],torch.zeros_like(batch[branch+'_mask']))
    assert float(zero)==0


def test_gradient_conflict_projection():
    p=torch.nn.Parameter(torch.tensor([1.,1.]));primary=p.square().sum();auxiliary=-10*p.square().sum()
    metrics=guarded_backward(primary,auxiliary,[p],clip=100)
    assert metrics['gradient_cosine']<-.99
    assert torch.allclose(p.grad,torch.tensor([2.,2.]),atol=1e-5)


@pytest.mark.parametrize('mode',['ccc_only','upstream','downstream','both'])
def test_ablation_training_inference_and_reconstruction(synthetic,tmp_path,mode):
    cfg=replace(GNNConfig(),programs=4,hidden=12,batch_size=16,max_epochs=4,patience=3,mode=mode,device='cpu',output_dir=str(tmp_path/mode))
    model,alignments=fit(synthetic,cfg)
    assert set(alignments)==({'upstream','downstream'} if mode=='both' else {mode} if mode!='ccc_only' else set())
    history=pd.read_csv(Path(cfg.output_dir)/'training.csv')
    assert np.isfinite(history.select_dtypes(include='number')).all().all()
    if mode=='ccc_only': assert history.train_ccc.iloc[-1]<history.train_ccc.iloc[0]
    arrays=infer(model,synthetic,cfg.output_dir,16)
    assert arrays['Z'].shape==(180,4) and np.all(arrays['Z']>=0)
    assert np.allclose(arrays['P'].sum(1),1)
    assert np.allclose(arrays['Z'].sum(1),arrays['activity'])
    with pytest.raises(FileExistsError): fit(synthetic,cfg)


def test_reproducibility_and_K_selection(synthetic,tmp_path):
    base=replace(GNNConfig(),programs=4,hidden=12,batch_size=16,max_epochs=3,patience=2,device='cpu',k_min=4,k_max=15,clustering_sample_size=60,silhouette_sample_size=60)
    first,_=fit(synthetic,replace(base,output_dir=str(tmp_path/'a')))
    second,_=fit(synthetic,replace(base,output_dir=str(tmp_path/'b')))
    for a,b in zip(first.parameters(),second.parameters()): assert torch.equal(a,b)
    cfg=replace(base,output_dir=str(tmp_path/'a'));arrays=infer(first,synthetic,cfg.output_dir,16);h=first.dictionary.detach().numpy()
    k,labels,table=discover(synthetic,arrays,h,cfg)
    assert table.K.tolist()==list(range(4,16)) and table.selected.sum()==1
    assert len(labels)==180 and k>=4
    result=diagnostics(synthetic,arrays,h,labels,{},cfg)
    assert 'composition_to_Z_test_R2' in result
    assert match_programs(h,h)['matched_program_cosine_mean']>.999


def test_feature_mask_and_counterfactual_gate(synthetic):
    d=synthetic;mask=np.ones(8,bool);mask[0]=False
    model=MicroenvironmentGNN(7,2,8,4,12,feature_mask=mask);batch=d.batch(d.rows[0][:3]);result=d.forward(model,batch)
    assert torch.all(result['H'][:,0]==0)
    original=ccc_loss(result['reconstruction'],batch['target'],d.feature_scale,mask)
    batch['target'][:,0]=10000
    assert torch.equal(original,ccc_loss(result['reconstruction'],batch['target'],d.feature_scale,mask))
    # All-identical ligand/receptor signals make opportunity-normalized CCC invariant
    # up to the existing tau. A high threshold must suppress every negative example.
    contrast,valid=contrastive_loss(d,model,batch,result,d.feature_scale,threshold=1e6)
    assert valid==0 and float(contrast)==0
    contrast,valid=contrastive_loss(d,model,batch,result,d.feature_scale,threshold=0)
    assert torch.isfinite(contrast) and valid>=0


def test_storage_protection_and_negative_controls(synthetic):
    with pytest.raises(ValueError): protect_output(replace(GNNConfig(),output_dir=GNNConfig().legacy_output))
    with pytest.raises(ValueError): protect_output(replace(GNNConfig(),output_dir='/tmp/real_training'))
    data=synthetic;before=data.observations['upstream'].copy()
    apply_control(data,replace(GNNConfig(),mode='both',prior_control='shuffle_activity'))
    for rows in data.rows:
        assert np.allclose(np.sort(before[rows],axis=0),np.sort(data.observations['upstream'][rows],axis=0))


def test_synthetic_preparation_training_and_fair_AE_baseline(synthetic,tmp_path,monkeypatch):
    import phenoniche.gnn.data as graph_module
    import phenoniche.gnn.pipeline as pipeline
    from phenoniche.ccc.config import CCCInputConfig
    from phenoniche.ccc.data import _manifest_signature
    d=synthetic
    obs=pd.DataFrame({'cell_id':d.cells.cell_id,'cell_type_coarse':d.cells.cell_type},index=d.cells.cell_id.to_numpy())
    # Assign columns positionally: Pandas index alignment must not create missing labels.
    obs['cell_id']=d.cells.cell_id.to_numpy();obs['cell_type_coarse']=d.cells.cell_type.to_numpy()
    adata=ad.AnnData(d.expression,obs=obs,var=pd.DataFrame(index=d.genes));adata.obsm['spatial']=d.coordinates
    adata.uns['dataset']='Xenium_Prime_Breast_Cancer_FFPE';adata.uns['cell_type_annotation']='synthetic test fixture'
    input_file=tmp_path/'input.h5ad';adata.write_h5ad(input_file)
    atlas_file=tmp_path/'lr.tsv';atlas_file.write_text('ligand\treceptor\nL0\tR0\nL1\tR1\n')
    cache=tmp_path/'source/cache';cache.mkdir(parents=True)
    np.save(cache/'ccc.npy',d.targets[:]);np.save(cache/'coordinates.npy',d.coordinates);np.save(cache/'magnitude.npy',d.targets[:].sum(1))
    d.cells.to_csv(cache/'cells.csv.gz',index=False);d.features.to_csv(cache/'features.csv',index=False)
    inp=replace(CCCInputConfig(),processed_h5ad=str(input_file),lr_atlas=str(atlas_file),output_dir=str(cache.parent))
    (cache/'input_manifest.json').write_text(json.dumps({'signature':_manifest_signature(inp),'n_cells':180,'n_features':8}))
    legacy=tmp_path/'legacy/cache';legacy.mkdir(parents=True);np.save(legacy/'retained_indices.npy',np.arange(8))
    cfg=replace(GNNConfig(),programs=4,hidden=12,encoder_genes=6,batch_size=16,max_epochs=8,patience=2,block_um=500.,device='cpu',output_dir=str(tmp_path/'gnn'),legacy_output=str(legacy.parent),upstream_prior=str(tmp_path/'absent_upstream.tsv'),downstream_prior=str(tmp_path/'absent_downstream.tsv'),prior_manifest=str(tmp_path/'absent_manifest.json'))
    monkeypatch.setattr(graph_module,'ccc_config',lambda:inp)
    prepared=graph_module.prepare_graph_data(cfg)
    assert len(prepared.encoder_indices)==6 and prepared.targets.shape==(180,8)
    assert prepared.audit['upstream']['status']=='missing'
    model,_=fit(prepared,cfg);arrays=infer(model,prepared,cfg.output_dir,16)
    k,labels,table=discover(prepared,arrays,model.dictionary.detach().numpy(),cfg)
    assert 4<=k<=15
    diagnostics(prepared,arrays,model.dictionary.detach().numpy(),labels,{},cfg)
    monkeypatch.setattr(pipeline,'require_audit',lambda c:{'synthetic_test':True})
    monkeypatch.setattr(pipeline,'protect_output',lambda c:None)  # Tiny synthetic fixture only.
    restored=pipeline.inference(cfg)
    assert np.array_equal(restored['Z'],arrays['Z'])
    baseline=pipeline.retrain_vector_baseline(cfg)
    assert np.isfinite(baseline['test_ccc'])
    report=pipeline.compare_outputs(cfg)
    assert report['comparison']=='original AE retrained on identical spatial folds'


def test_audit_gate_blocks_real_training_before_preparation(monkeypatch):
    import phenoniche.gnn.pipeline as pipeline
    monkeypatch.setattr(pipeline,'audit_study',lambda cfg:{'ready_for_training':False,'blockers':['missing prior/audit']})
    def forbidden(*args): pytest.fail('Preparation/training must not run after failed audit')
    monkeypatch.setattr(pipeline,'prepare_graph_data',forbidden)
    with pytest.raises(RuntimeError,match='missing prior/audit'): pipeline.run(GNNConfig())

@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA runtime unavailable')
def test_resident_gpu_inputs_and_regulatory_aggregation_match_cpu(synthetic,tmp_path):
    d=synthetic;rows=d.rows[0][:12];cpu=d.batch(rows);model=make_model(d).eval()
    with torch.no_grad():expected=d.forward(model,cpu)['Z']
    d.prepare_device('cuda:0',tmp_path)
    gpu=d.batch(rows,'cuda:0');model.cuda(0)
    with torch.no_grad():actual=d.forward(model,gpu)['Z'].cpu()
    torch.testing.assert_close(actual,expected,rtol=1e-4,atol=1e-5)
    for key in ('expression','target','nuisance','upstream','downstream','upstream_mask','downstream_mask'):
        torch.testing.assert_close(gpu[key].cpu(),cpu[key])
    branch='upstream';prior=d.priors[branch];scores=activity_scores(d.expression,prior,np.unique(d.members[d.rows[0]]));local=d.members[rows]
    args=[torch.from_numpy(a) for a in (scores[local],d.types[local],d.coordinates[local],d.weights[rows],local)]
    expected,mask=aggregate_activity(*args,d.features,prior,d.categories)
    actual,gpu_mask=aggregate_activity(*(a.cuda(0) for a in args),d.features,prior,d.categories)
    torch.testing.assert_close(actual.cpu(),expected,rtol=1e-4,atol=1e-5)
    assert torch.equal(gpu_mask.cpu(),mask)


def test_embedding_inference_does_not_read_CCC_targets(synthetic):
    rows=synthetic.rows[0][:7];model=make_model(synthetic).eval()
    with torch.no_grad(): expected=synthetic.forward(model,synthetic.batch(rows))
    class UnreadableTargets:
        def __getitem__(self,rows): raise AssertionError('Inference read CCC supervision')
    synthetic.targets=UnreadableTargets()
    batch=synthetic.batch(rows,include_target=False,include_bio=False)
    assert 'target' not in batch and 'nuisance' not in batch
    with torch.no_grad(): actual=synthetic.forward(model,batch,reconstruct=False)
    assert 'reconstruction' not in actual and 'H' not in actual
    torch.testing.assert_close(actual['Z'],expected['Z'],rtol=0,atol=0)


def test_historical_input_configuration_is_frozen(tmp_path):
    from phenoniche.gnn.inputs import resolve_input
    from phenoniche.ccc.config import CCCInputConfig
    source=tmp_path/'old/cache';source.mkdir(parents=True)
    (source/'input_manifest.json').write_text(json.dumps({'signature':{'expression_coverage_fraction':.1,'n_neighbors':30,'sigma_um':20.}}))
    legacy=tmp_path/'legacy/cache';legacy.mkdir(parents=True)
    (legacy/'input_manifest.json').write_text(json.dumps({'signature':{'source':str(source/'ccc.npy')}}))
    current=replace(CCCInputConfig(),expression_coverage_fraction=.05,output_dir=str(tmp_path/'new'))
    old=replace(GNNConfig(),legacy_output=str(legacy.parent))
    pinned=resolve_input(old,current)
    assert pinned.expression_coverage_fraction==.1 and pinned.output_dir==str(source.parent)
    assert resolve_input(replace(old,use_legacy_feature_mask=False),current)==current
