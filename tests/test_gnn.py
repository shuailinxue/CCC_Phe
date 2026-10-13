"""Synthetic mechanism tests; never read/train on real Xenium data."""
from dataclasses import replace
import json
import numpy as np,pandas as pd,scipy.sparse as sp,torch,pytest,anndata as ad
from phenoniche.gnn.config import GNNConfig
from phenoniche.gnn.model import MicroenvironmentGNN,physical_weights
from phenoniche.gnn.data import GraphData,Targets,neighborhoods,spatial_split,pair_composition,align_cells
from phenoniche.gnn.losses import ccc_loss,composition_loss,guarded_backward,pair_ccc
from phenoniche.gnn.training import fit,infer
from phenoniche.gnn.pipeline import export_reconstruction,results_complete
from phenoniche.ccc.pairwise import aggregate_pairwise_ccc

@pytest.fixture
def synthetic():
    rng=np.random.default_rng(17);n=120
    xy=np.concatenate([rng.normal(size=(40,2))*12+[1200*i,0] for i in range(3)]).astype(np.float32)
    types=rng.integers(0,2,n);categories=('A','B');raw=rng.gamma(2,2,(n,10)).astype(np.float32)
    genes=('L0','R0','L1','R1','G4','G5','G6','G7','G8','G9');members,weights=neighborhoods(xy)
    rows,folds,blocks=spatial_split(xy,members,block_um=500)
    features=pd.DataFrame([{'sender':categories[s],'receiver':categories[r],'ligand':f'L{l}','receptor':f'R{l}','lr_id':str(l),'ccc':f'{categories[s]}→{categories[r]}|L{l}–R{l}'} for s in range(2) for r in range(2) for l in range(2)])
    _,ccc,_=aggregate_pairwise_ccc(xy,types,raw[:,[0,2]],raw[:,[1,3]],2,20.,members,weights,batch_size=16)
    pair=physical_weights(torch.from_numpy(xy[members]),torch.from_numpy(weights),20.,torch.from_numpy(members))
    comp=pair_composition(torch.from_numpy(types[members]),pair,2).numpy()
    totals=raw.sum(1);encoder=np.arange(6);train=np.unique(members[rows[0]])
    sample=np.log1p(raw[train][:,encoder]*10000/totals[train,None])
    cells=pd.DataFrame({'cell_id':[f'c{i}' for i in range(n)],'cell_type':np.asarray(categories)[types]})
    return GraphData(sp.csr_matrix(raw),totals,genes,encoder,sample.mean(0),np.maximum(sample.std(0),.1),types,categories,xy,members,weights,Targets(ccc,np.arange(8)),np.maximum(np.sqrt((ccc[rows[0]]**2).mean(0)),.1),np.ones(8,bool),features,cells,rows,folds,blocks,comp,{'source_signature':{'synthetic':True},'cell_order_sha256':'synthetic','CCC_matrix_path':'synthetic'})


def make_model(d):return MicroenvironmentGNN(len(d.encoder_indices)+1,2,8,4,hidden=12)


def test_expression_only_forward_nonnegative_dictionaries_and_gradient(synthetic):
    d=synthetic;b=d.batch(d.rows[0][:8]);model=make_model(d);out=d.forward(model,b)
    assert not hasattr(model,'type_embedding') and 'types' not in b
    assert out['Z'].shape==(8,4) and out['composition'].shape==(8,2)
    for key in ('Z','H','B','composition'):assert (out[key]>=0).all()
    torch.testing.assert_close(out['H'].sum(1),torch.ones(4));torch.testing.assert_close(out['B'].sum(1),torch.ones(4))
    torch.testing.assert_close(out['composition'].sum(1),torch.ones(8))
    torch.testing.assert_close(out['reconstruction'],out['Z']@out['H'])
    torch.testing.assert_close(out['composition'],out['P']@out['B'])
    loss=ccc_loss(out['reconstruction'],b['target'],d.feature_scale,d.feature_mask)+.05*composition_loss(out['composition'],b['composition_target'])
    loss.backward()
    for p in (model.cell_encoder[0].weight,model.sender.weight,model.receiver.weight,model.raw_dictionary,model.raw_composition):assert p.grad is not None and p.grad.abs().sum()>0
    # Arbitrary type changes do not change expression/geometry or encoder outputs.
    d.types=1-d.types;d.composition_target=d.composition_target[:,::-1].copy()
    again=d.forward(model,d.batch(d.rows[0][:8]));torch.testing.assert_close(out['Z'],again['Z'],rtol=0,atol=0)


def test_composition_uses_exact_outgoing_pair_weights():
    w=torch.tensor([[[0.,1.,2.],[3.,0.,4.],[5.,6.,0.]]]);types=torch.tensor([[0,1,0]])
    actual=pair_composition(types,w,2)
    torch.testing.assert_close(actual,torch.tensor([[14/21,7/21]]))
    torch.testing.assert_close(pair_composition(types,torch.zeros_like(w),2),torch.zeros(1,2))
    assert float(composition_loss(actual,actual))==pytest.approx(0,abs=1e-7)


def test_undefined_composition_mask_and_isolated_graph():
    xy=torch.tensor([[[0.,0.],[1000.,0.],[0.,1000.]]])
    weights=physical_weights(xy,torch.tensor([[1.,0.,0.]]),20.)
    target=pair_composition(torch.tensor([[0,1,0]]),weights,2)
    assert target.sum()==0
    prediction=torch.tensor([[.2,.8],[.6,.4]],requires_grad=True)
    mixed=torch.cat((target,torch.tensor([[.7,.3]])))
    actual=composition_loss(prediction,mixed)
    torch.testing.assert_close(actual,composition_loss(prediction[1:],mixed[1:]))
    actual.backward();torch.testing.assert_close(prediction.grad[0],torch.zeros(2))
    empty=torch.tensor([[.2,.8]],requires_grad=True)
    composition_loss(empty,target).backward();torch.testing.assert_close(empty.grad,torch.zeros_like(empty))
    model=MicroenvironmentGNN(4,2,3,programs=4,hidden=8)
    out=model(torch.randn(3,4),torch.arange(3)[None],xy,torch.tensor([[1.,0.,0.]]))
    loss=ccc_loss(out['reconstruction'],torch.zeros(1,3),np.ones(3),np.ones(3,bool))+composition_loss(out['composition'],target)
    loss.backward();assert torch.isfinite(loss) and torch.isfinite(model.cell_encoder[0].weight.grad).all()


def test_export_excludes_undefined_composition(synthetic,tmp_path):
    d=synthetic;d.composition_target[:]=.5;d.composition_target[d.rows[2][:3]]=0
    arrays={'Z':np.ones((120,4),np.float32),'P':np.full((120,4),.25,np.float32)}
    h=np.full((4,8),1/8,np.float32);b=np.full((4,2),.5,np.float32)
    cfg=replace(GNNConfig(),output_dir=str(tmp_path))
    export_reconstruction(d,arrays,h,b,cfg)
    metrics=json.loads((tmp_path/'reconstruction_metrics.json').read_text())['test']
    assert metrics['n']==len(d.rows[2]) and metrics['composition_valid_n']==len(d.rows[2])-3
    assert metrics['composition_RMSE']==0 and metrics['composition_KL']==0


def test_composition_initialization_is_train_only(synthetic):
    from phenoniche.gnn.training import initialize
    d=synthetic;cfg=replace(GNNConfig(),programs=4,hidden=12,device='cpu')
    first=initialize(d,cfg,initialize_composition=True)
    d.composition_target[d.rows[1]]=d.composition_target[d.rows[1]][:,::-1]
    d.composition_target[d.rows[2]]=d.composition_target[d.rows[2]][:,::-1]
    second=initialize(d,cfg,initialize_composition=True)
    torch.testing.assert_close(first.composition_dictionary,second.composition_dictionary,rtol=0,atol=0)
    assert first.composition_dictionary.std(0).max()>0


def test_activity_reparameterization_exports_original_architecture(synthetic,tmp_path):
    d=synthetic;d.targets.matrix*=100;d.feature_scale*=100
    cfg=replace(GNNConfig(),programs=4,hidden=12,batch_size=16,max_epochs=2,patience=2,device='cpu',output_dir=str(tmp_path))
    model=fit(d,cfg,rescale_activity=True)
    payload=torch.load(tmp_path/'checkpoint.pt',map_location='cpu',weights_only=False)
    assert payload['activity_parameter_scale']>10 and not model.activity_head._forward_hooks
    restored=make_model(d);restored.load_state_dict(payload['model_state'])
    batch=d.batch(d.rows[2][:6])
    torch.testing.assert_close(d.forward(model,batch)['Z'],d.forward(restored,batch)['Z'],rtol=1e-6,atol=1e-6)
    best=torch.load(tmp_path/'training_best.pt',map_location='cpu',weights_only=False)
    restored.load_state_dict(best['model_state'])
    torch.testing.assert_close(d.forward(model,batch)['Z'],d.forward(restored,batch)['Z'],rtol=1e-6,atol=1e-6)


def test_exact_ccc_and_spatial_split(synthetic):
    d=synthetic;local=d.members[:7];raw=d.expression.toarray()
    l=np.column_stack([raw[:,0 if v=='L0' else 2] for v in d.features.ligand]);r=np.column_stack([raw[:,1 if v=='R0' else 3] for v in d.features.receptor])
    w=physical_weights(torch.from_numpy(d.coordinates[local]),torch.from_numpy(d.weights[:7]),20.,torch.from_numpy(local))
    actual=pair_ccc(torch.from_numpy(l[local]),torch.from_numpy(r[local]),torch.from_numpy(d.types[local]),d.features,d.categories,w).numpy()
    np.testing.assert_allclose(actual,d.targets[:7],rtol=1e-4,atol=1e-5)
    used=[np.unique(d.members[r]) for r in d.rows]
    for i in range(3):
        for j in range(i):assert not np.intersect1d(used[i],used[j]).size


def test_permutation_and_target_free_inference(synthetic):
    d=synthetic;model=make_model(d).eval();b=d.batch(d.rows[0][:6]);out=d.forward(model,b)
    perm=torch.randperm(30);permuted={k:v for k,v in b.items()}
    for key in ('local_index','coordinates','anchor_weights','cell_ids'):permuted[key]=b[key][:,perm]
    torch.testing.assert_close(out['Z'],d.forward(model,permuted)['Z'],rtol=1e-5,atol=1e-5)
    class Unreadable:
        def __getitem__(self,rows):raise AssertionError('Read CCC targets during inference')
    d.targets=Unreadable();b=d.batch(d.rows[0][:6],include_target=False)
    assert 'target' not in b and 'composition_target' not in b
    actual=d.forward(model,b,reconstruct=False);assert 'H' not in actual
    torch.testing.assert_close(actual['Z'],out['Z'],rtol=0,atol=0)


@pytest.mark.parametrize('lambda_comp',[0.,.05])
def test_synthetic_training_inference_and_export(synthetic,tmp_path,lambda_comp):
    d=synthetic;cfg=replace(GNNConfig(),programs=4,hidden=12,batch_size=16,max_epochs=2,patience=2,device='cpu',output_dir=str(tmp_path),lambda_comp=lambda_comp,k_min=4,k_max=5)
    model=fit(d,cfg);arrays=infer(model,d,tmp_path,16)
    assert arrays['Z'].shape==(120,4) and np.load(tmp_path/'B.npy').shape==(4,2)
    export_reconstruction(d,arrays,np.load(tmp_path/'H.npy'),np.load(tmp_path/'B.npy'),cfg)
    np.testing.assert_allclose(np.load(tmp_path/'composition_prediction.npy').sum(1),1,rtol=1e-5)
    history=pd.read_csv(tmp_path/'training.csv');assert 'train_composition' in history and np.isfinite(history.select_dtypes('number')).all().all()
    assert not any('upstream' in c or 'downstream' in c for c in history)
    from phenoniche.gnn.evaluation import discover,diagnostics
    k,labels,table=discover(d,arrays,np.load(tmp_path/'H.npy'),cfg)
    assert k in (4,5) and len(labels)==120 and len(table)==2
    assert 'composition_to_Z_test_R2' in diagnostics(d,arrays,np.load(tmp_path/'H.npy'),labels,cfg)
    with pytest.raises(FileExistsError):fit(d,cfg)


def test_synthetic_preparation_without_priors(synthetic,tmp_path,monkeypatch):
    from phenoniche.gnn import data as module
    from phenoniche.ccc.config import CCCInputConfig
    from phenoniche.ccc.data import _manifest_signature
    d=synthetic;obs=pd.DataFrame({'cell_id':d.cells.cell_id.to_numpy(),'cell_type_coarse':d.cells.cell_type.to_numpy()})
    a=ad.AnnData(d.expression,obs=obs,var=pd.DataFrame(index=d.genes));a.obsm['spatial']=d.coordinates
    file=tmp_path/'input.h5ad';a.write_h5ad(file);atlas=tmp_path/'lr.tsv';atlas.write_text('ligand\treceptor\nL0\tR0\nL1\tR1\n')
    cache=tmp_path/'source/cache';cache.mkdir(parents=True)
    np.save(cache/'ccc.npy',d.targets[:]);np.save(cache/'coordinates.npy',d.coordinates);np.save(cache/'magnitude.npy',d.targets[:].sum(1))
    d.cells.to_csv(cache/'cells.csv.gz',index=False);d.features.to_csv(cache/'features.csv',index=False)
    inp=replace(CCCInputConfig(),processed_h5ad=str(file),lr_atlas=str(atlas),output_dir=str(cache.parent))
    (cache/'input_manifest.json').write_text(json.dumps({'signature':_manifest_signature(inp),'n_cells':120,'n_features':8}))
    monkeypatch.setattr(module,'ccc_config',lambda:inp)
    cfg=replace(GNNConfig(),encoder_genes=6,batch_size=16,block_um=500,output_dir=str(tmp_path/'gnn'))
    actual=module.prepare_graph_data(cfg)
    assert actual.targets.shape==(120,8) and actual.composition_target.shape==(120,2)
    np.testing.assert_allclose(actual.composition_target,d.composition_target,atol=1e-6)
    assert 'types' not in actual.batch(actual.rows[0][:4]) and not hasattr(actual,'priors')
    assert set(actual.audit['encoder_genes'])<=set(d.genes)


def test_notebook_has_valid_run_all_entry():
    from pathlib import Path
    import nbformat,ast
    root=Path(__file__).resolve().parents[1];n=nbformat.read(root/'XeniumPrime5K_GNN_niche_analysis.ipynb',as_version=4)
    # Existing real-data notebooks may contain user execution records. Check
    # their runnable source without requiring deletion of previous results.
    nbformat.validate(n)
    texts=[]
    for cell in n.cells:
        if cell.cell_type=='code':
            ast.parse('\n'.join(s for s in cell.source.splitlines() if not s.startswith('%')));texts.append(cell.source)
    assert 'ensure_results(CONFIG)' in '\n'.join(texts)
    assert 'composition_dictionary' in '\n'.join(texts) and 'biological_heatmaps' not in '\n'.join(texts)


def test_reuse_rejects_different_configuration(tmp_path):
    from phenoniche.gnn.pipeline import REQUIRED
    cfg=replace(GNNConfig(),output_dir=str(tmp_path))
    for name in REQUIRED:
        f=tmp_path/name;f.parent.mkdir(exist_ok=True,parents=True);f.touch()
    (tmp_path/'config.json').write_text(json.dumps(cfg.to_dict()));(tmp_path/'summary.json').write_text(json.dumps({'architecture':cfg.architecture}))
    assert results_complete(cfg)
    with pytest.raises(ValueError):results_complete(replace(cfg,lambda_comp=.2))
