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


def test_projected_dictionary_revives_small_loadings_and_masks():
    legacy=MicroenvironmentGNN(4,2,3,programs=4,hidden=8,feature_mask=[True,True,False])
    direct=MicroenvironmentGNN(4,2,3,programs=4,hidden=8,feature_mask=[True,True,False],dictionary_parameterization='projected')
    with torch.no_grad():
        legacy.raw_dictionary[:,0]=0.;legacy.raw_dictionary[:,1]=-16.
        direct.raw_dictionary.copy_(legacy.dictionary)
    torch.testing.assert_close(direct.dictionary,legacy.dictionary)
    legacy.dictionary[:,1].sum().backward();direct.dictionary[:,1].sum().backward()
    assert direct.raw_dictionary.grad[:,1].abs().min()>100000*legacy.raw_dictionary.grad[:,1].abs().max()
    with torch.no_grad():direct.raw_dictionary.fill_(-1)
    direct.project_dictionary()
    assert (direct.dictionary>=0).all() and (direct.dictionary[:,2]==0).all()
    torch.testing.assert_close(direct.dictionary.sum(1),torch.ones(4))


def test_projected_dictionary_fit_checkpoint(synthetic,tmp_path):
    cfg=replace(GNNConfig(),programs=4,hidden=12,batch_size=16,max_epochs=2,patience=2,
        device='cpu',output_dir=str(tmp_path),dictionary_parameterization='projected')
    model=fit(synthetic,cfg)
    from phenoniche.gnn.model import model_from_config
    payload=torch.load(tmp_path/'checkpoint.pt',weights_only=False)
    loaded=model_from_config(synthetic,GNNConfig(**payload['config']))
    loaded.load_state_dict(payload['model_state'])
    batch=synthetic.batch(synthetic.rows[1][:8]);model.eval();loaded.eval()
    torch.testing.assert_close(synthetic.forward(model,batch)['reconstruction'],synthetic.forward(loaded,batch)['reconstruction'])
    assert (loaded.dictionary>=0).all()
    torch.testing.assert_close(loaded.dictionary.sum(1),torch.ones(4))


def test_scaled_dictionary_preserves_native_reconstruction():
    native=MicroenvironmentGNN(4,2,3,programs=4,hidden=8,feature_mask=[True,True,False],dictionary_parameterization='projected')
    scaled=MicroenvironmentGNN(4,2,3,programs=4,hidden=8,feature_mask=[True,True,False],dictionary_parameterization='projected',dictionary_scale=[.001,20.,1.])
    with torch.no_grad():
        d=native.dictionary/scaled.dictionary_scale;d/=d.sum(1,keepdim=True);scaled.raw_dictionary.copy_(d)
    torch.testing.assert_close(native.dictionary,scaled.dictionary)
    scaled.dictionary[:,0].sum().backward()
    assert torch.isfinite(scaled.raw_dictionary.grad).all()
    scaled.project_dictionary();torch.testing.assert_close(scaled.dictionary.sum(1),torch.ones(4))


def test_robust_initialization_excludes_masked_features(synthetic):
    from phenoniche.gnn.training import initialize
    d=synthetic;d.feature_mask[-1]=False
    cfg=replace(GNNConfig(),programs=4,hidden=12,dictionary_initialization='log1p_active_kmeans')
    first=initialize(d,cfg)
    d.targets.matrix[:,-1]=np.arange(len(d.cells))*100000.
    second=initialize(d,cfg)
    torch.testing.assert_close(first.dictionary,second.dictionary,rtol=0,atol=0)
    assert first.initialization_audit['prototype_features']==7
    assert sum(first.initialization_audit['cluster_counts'])==len(d.rows[0])


def test_nmf_initialization_training_only_mask_and_reproducibility(synthetic):
    from phenoniche.gnn.training import initialize
    d=synthetic;d.feature_mask[-1]=False
    cfg=replace(GNNConfig(),programs=4,hidden=12,dictionary_initialization='nmf',
        nmf_initialization_sample_size=32,nmf_initialization_max_iter=500)
    first=initialize(d,cfg)
    assert first.initialization_audit['method']=='nmf'
    assert first.initialization_audit['training_centers']==32
    assert first.initialization_audit['prototype_features']==7
    assert first.initialization_audit['encoder_pretraining'] is False
    assert torch.isfinite(first.dictionary).all() and (first.dictionary>=0).all()
    assert (first.dictionary[:,-1]==0).all()
    torch.testing.assert_close(first.dictionary.sum(1),torch.ones(4))
    # Held-out CCC and excluded features cannot alter the training initializer.
    d.targets.matrix[np.concatenate(d.rows[1:])]=100000.
    d.targets.matrix[:,-1]=np.arange(len(d.cells))*100000.
    second=initialize(d,cfg)
    for key,value in first.state_dict().items():
        torch.testing.assert_close(value,second.state_dict()[key],rtol=0,atol=0)


def test_nmf_dictionary_native_scale_and_joint_training(synthetic,tmp_path):
    from phenoniche.gnn.training import initialize
    from sklearn.decomposition import NMF
    d=synthetic;cfg=replace(GNNConfig(),programs=4,hidden=12,batch_size=16,max_epochs=2,patience=2,
        device='cpu',output_dir=str(tmp_path),dictionary_initialization='nmf',nmf_initialization_max_iter=500)
    initial=initialize(d,cfg)
    sample=np.random.default_rng(cfg.seed).choice(d.rows[0],len(d.rows[0]),replace=False)
    target=d.targets[sample][:,d.feature_mask]/d.feature_scale[d.feature_mask]
    expected=NMF(n_components=4,init='nndsvda',random_state=cfg.seed,max_iter=500,tol=1e-4)
    z=expected.fit_transform(target);h=expected.components_*d.feature_scale[None]
    mass=h.sum(1);h/=mass[:,None];z*=mass[None]
    torch.testing.assert_close(initial.dictionary,torch.from_numpy(h),rtol=1e-4,atol=1e-6)
    p=z.mean(0);p/=p.sum()
    torch.testing.assert_close(initial.composition_head.bias.softmax(0),torch.from_numpy(p),rtol=1e-4,atol=1e-6)
    torch.testing.assert_close(torch.nn.functional.softplus(initial.activity_head.bias),torch.tensor([np.median(z.sum(1))],dtype=torch.float32),rtol=1e-4,atol=1e-6)
    trained=fit(d,cfg)
    assert (tmp_path/'initial_H.npy').is_file()
    assert json.loads((tmp_path/'initialization_audit.json').read_text())['method']=='nmf'
    assert not torch.allclose(trained.dictionary,initial.dictionary)
    batch=d.batch(d.rows[1][:8]);result=d.forward(trained,batch)
    assert torch.isfinite(result['reconstruction']).all() and (result['Z']>=0).all()


def test_pair_pooling_warm_start_alignment_and_gradients(synthetic):
    from phenoniche.gnn.model import model_from_config
    d=synthetic;cfg=replace(GNNConfig(),programs=4,hidden=12)
    old=model_from_config(d,cfg);new=model_from_config(d,replace(cfg,program_pair_pooling=True))
    state=new.state_dict();state.update(old.state_dict());new.load_state_dict(state)
    b=d.batch(d.rows[0][:8]);old.eval();new.eval()
    torch.testing.assert_close(d.forward(old,b)['reconstruction'],d.forward(new,b)['reconstruction'],rtol=0,atol=0)
    with torch.no_grad():new.pair_mixture.weight.copy_(torch.eye(4));new.pair_activity.weight.fill_(.1)
    out=d.forward(new,b);ccc_loss(out['reconstruction'],b['target'],d.feature_scale,d.feature_mask).backward()
    for name in ('pair_sender_role','pair_receiver_role','pair_sender_signal','pair_receiver_signal'):
        assert torch.isfinite(getattr(new,name).weight.grad).all() and getattr(new,name).weight.grad.abs().sum()>0
    order=torch.randperm(len(b['expression']));inverse=torch.argsort(order)
    shuffled=dict(b,expression=b['expression'][order],local_index=inverse[b['local_index']])
    torch.testing.assert_close(out['reconstruction'],d.forward(new,shuffled)['reconstruction'])


def test_sampled_preload_preserves_cell_alignment(synthetic):
    d=synthetic;rows=d.rows[0][:8];before=d.batch(rows)
    model=make_model(d);model.eval();prediction=d.forward(model,before)['reconstruction']
    d.prepare_subset_device(rows,'cpu');after=d.batch(rows)
    for name in ('expression','local_index','coordinates','anchor_weights','target'):
        torch.testing.assert_close(before[name],after[name])
    torch.testing.assert_close(prediction,d.forward(model,after)['reconstruction'])
    with pytest.raises(ValueError,match='not preloaded'):d.batch(d.rows[1][:2])


def test_log_counts_and_library_standardization(synthetic):
    from phenoniche.gnn.data import expression_block
    d=synthetic;rows=d.rows[0][:8]
    actual=expression_block(d.expression,rows,d.encoder_indices,d.totals,'log_counts')
    np.testing.assert_allclose(actual,np.log1p(d.expression[rows][:,d.encoder_indices].toarray()))
    d.expression_normalization='log_counts';d.library_mean=2.;d.library_scale=.5
    before=d.batch(rows);d.prepare_subset_device(rows,'cpu');after=d.batch(rows)
    torch.testing.assert_close(before['expression'],after['expression'])
    physical=np.unique(d.members[rows]);expected=(np.log1p(d.totals[physical])-2.)/.5
    np.testing.assert_allclose(before['expression'][:,-1].numpy(),expected)


def test_independent_activities_and_frozen_dictionary(synthetic,tmp_path):
    from phenoniche.gnn.model import model_from_config
    from phenoniche.gnn.losses import regularization
    d=synthetic;cfg=replace(GNNConfig(),programs=4,hidden=12,activity_parameterization='independent',
        max_epochs=2,patience=2,batch_size=16,device='cpu',output_dir=str(tmp_path))
    model=model_from_config(d,cfg);b=d.batch(d.rows[0][:8]);out=d.forward(model,b)
    torch.testing.assert_close(out['Z'],out['activity'][:,None]*out['P'])
    torch.testing.assert_close(out['P'],out['Z'].log().softmax(1))
    assert (out['Z']>0).all() and out['Z'].shape==(8,4)
    model.raw_dictionary.requires_grad_(False)
    guarded_backward(ccc_loss(out['reconstruction'],b['target'],d.feature_scale,d.feature_mask),.005*regularization(out),model.parameters())
    assert model.program_head.weight.grad.abs().sum()>0
    assert model.raw_dictionary.grad is None
    fitted=fit(d,cfg,rescale_activity=True)
    checkpoint=torch.load(tmp_path/'checkpoint.pt',weights_only=False)
    loaded=model_from_config(d,cfg);loaded.load_state_dict(checkpoint['model_state']);loaded.eval();fitted.eval()
    torch.testing.assert_close(d.forward(fitted,b)['Z'],d.forward(loaded,b)['Z'],rtol=0,atol=0)


def test_lr_pair_pooling_real_expression_alignment_and_gradient(synthetic):
    from phenoniche.gnn.model import model_from_config
    d=synthetic;cfg=replace(GNNConfig(),programs=4,hidden=12,program_pair_pooling=True,lr_pair_pooling=True)
    model=model_from_config(d,cfg);rows=d.rows[0][:8];before=d.batch(rows)
    assert 'types' not in before and before['raw_lr_expression'].shape[1]==4
    with torch.no_grad():model.pair_mixture.weight.copy_(torch.eye(4));model.pair_activity.weight.fill_(.1)
    model.eval();out=d.forward(model,before)
    ccc_loss(out['reconstruction'],before['target'],d.feature_scale,d.feature_mask).backward()
    assert model.pair_sender_role.weight.grad.abs().sum()>0 and torch.isfinite(model.raw_dictionary.grad).all()
    d.prepare_subset_device(rows,'cpu');after=d.batch(rows)
    torch.testing.assert_close(before['raw_lr_expression'],after['raw_lr_expression'])
    torch.testing.assert_close(out['reconstruction'],d.forward(model,after)['reconstruction'])
    d.types=1-d.types
    torch.testing.assert_close(out['reconstruction'],d.forward(model,d.batch(rows))['reconstruction'])


def test_disable_dropout_preserves_warm_start_state(synthetic):
    from phenoniche.gnn.model import model_from_config
    d=synthetic;cfg=replace(GNNConfig(),programs=4,hidden=12,architecture='expression_gnn_residual_v2',encoder_depth=2,message_layers=2,dropout=.1)
    old=model_from_config(d,cfg);new=model_from_config(d,replace(cfg,dropout=0.))
    new.load_state_dict(old.state_dict());old.eval();new.eval();b=d.batch(d.rows[0][:8])
    torch.testing.assert_close(d.forward(old,b)['reconstruction'],d.forward(new,b)['reconstruction'],rtol=0,atol=0)


def test_rna_cell_states_are_label_blind_and_train_only(synthetic,tmp_path):
    import copy
    from phenoniche.gnn.cell_state import prepare_cell_states
    from phenoniche.gnn.model import model_from_config
    d=synthetic;rng=np.random.default_rng(111)
    extra=sp.csr_matrix(rng.poisson(2,(len(d.cells),70)).astype(np.float32))
    d.expression=sp.hstack((d.expression,extra),format='csr');d.genes=tuple(d.genes)+tuple(f'G{i}' for i in range(10,80))
    d.totals=np.asarray(d.expression.sum(1)).ravel()
    cfg=replace(GNNConfig(),programs=4,hidden=12,cell_state_components=32)
    prepare_cell_states(d,cfg,tmp_path/'first')
    duplicate=copy.copy(d);duplicate.types=1-d.types
    duplicate.targets=Targets(np.zeros(d.targets.shape,np.float32),d.targets.columns)
    prepare_cell_states(duplicate,cfg,tmp_path/'second')
    np.testing.assert_allclose(d.cell_state_scores,duplicate.cell_state_scores,rtol=0,atol=0)
    fitted=d.cell_state_preprocessing['fit_cell_ids']
    assert not np.intersect1d(fitted,np.unique(d.members[d.rows[1]])).size
    batch=d.batch(d.rows[0][:8]);model=model_from_config(d,cfg)
    assert batch['expression'].shape[1]==len(d.encoder_indices)+32+1
    out=d.forward(model,batch);ccc_loss(out['reconstruction'],batch['target'],d.feature_scale,d.feature_mask).backward()
    assert model.cell_encoder[0].weight.grad.abs().sum()>0


def test_ccc_observation_weights_are_explicit(synthetic):
    d=synthetic;batch=d.batch(d.rows[0][:8]);out=d.forward(make_model(d),batch)
    plain=ccc_loss(out['reconstruction'],batch['target'],d.feature_scale,d.feature_mask)
    ones=ccc_loss(out['reconstruction'],batch['target'],d.feature_scale,d.feature_mask,entry_weights=torch.ones_like(batch['target']))
    torch.testing.assert_close(plain,ones,rtol=0,atol=0)
    with pytest.raises(ValueError,match='observation weights'):
        ccc_loss(out['reconstruction'],batch['target'],d.feature_scale,d.feature_mask,entry_weights=-torch.ones_like(batch['target']))


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
    cfg=replace(GNNConfig(),output_dir=str(tmp_path),lambda_comp=.05)
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


def test_no_composition_loss_when_disabled(synthetic,monkeypatch):
    from phenoniche.gnn import training
    def forbidden(*args,**kwargs):raise AssertionError('Composition loss was called')
    monkeypatch.setattr(training,'composition_loss',forbidden)
    d=synthetic;d.composition_target[:]=np.nan
    cfg=replace(GNNConfig(),programs=4,hidden=12,device='cpu',lambda_comp=0.)
    result=training.epoch(make_model(d),d,d.rows[0][:8],cfg)
    assert 'composition' not in result and 'regularization' in result
    assert result['total']==pytest.approx(result['ccc']+cfg.lambda_reg*result['regularization'])


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
    history=pd.read_csv(tmp_path/'training.csv');assert ('train_composition' in history)==(lambda_comp>0) and np.isfinite(history.select_dtypes('number')).all().all()
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
    assert 'composition_dictionary' not in '\n'.join(texts) and 'composition_reconstruction' not in '\n'.join(texts)
    assert 'biological_heatmaps' not in '\n'.join(texts)


def test_reuse_rejects_different_configuration(tmp_path):
    from phenoniche.gnn.pipeline import REQUIRED
    cfg=replace(GNNConfig(),output_dir=str(tmp_path))
    for name in REQUIRED:
        f=tmp_path/name;f.parent.mkdir(exist_ok=True,parents=True);f.touch()
    (tmp_path/'config.json').write_text(json.dumps(cfg.to_dict()));(tmp_path/'summary.json').write_text(json.dumps({'architecture':cfg.architecture}))
    assert results_complete(cfg)
    with pytest.raises(ValueError):results_complete(replace(cfg,lambda_comp=.2))


def test_lr_complete_selection_keeps_low_variance_complex_members():
    from phenoniche.gnn.genes import select_encoder_genes
    genes=('L0','R0','R1','HLA-DRA','G4','G5','G6','G7')
    features=pd.DataFrame({'ligand':['L0'],'receptor':['R0&R1&HLA-DRA']})
    variance=np.array([.001,.001,.001,.001,100,90,80,70])
    indices,audit=select_encoder_genes(genes,features,variance,6)
    assert set(indices)=={0,1,2,3,4,5} and audit['CCC_features_fully_covered']==1
    assert audit['LR_members_in_encoder']==4 and not audit['missing_encoder_members']
    with pytest.raises(ValueError,match='budget'):select_encoder_genes(genes,features,variance,3)
    absent=features.copy();absent['ligand']='MISSING'
    with pytest.raises(ValueError,match='absent'):select_encoder_genes(genes,absent,variance,6)


def test_nonzero_weighting_recovers_sparse_signal_gradient_and_mask():
    target=torch.zeros(100,2);target[0,0]=10;target[:,1]=1000
    parameter=torch.tensor(.05,requires_grad=True);prediction=parameter.expand_as(target)
    ordinary=ccc_loss(prediction,target,np.ones(2),np.array([True,False]))
    ordinary_gradient=torch.autograd.grad(ordinary,parameter,retain_graph=True)[0]
    weighted=ccc_loss(prediction,target,np.ones(2),np.array([True,False]),nonzero_fraction=np.array([.01,1.]),nonzero_weight_max=8)
    weighted_gradient=torch.autograd.grad(weighted,parameter)[0]
    assert ordinary_gradient>0 and weighted_gradient<0
    changed=target.clone();changed[:,1]=-1  # Invalid values are still audited even outside loss mask.
    with pytest.raises(ValueError):ccc_loss(prediction,changed,np.ones(2),np.array([True,False]))
    with pytest.raises(ValueError,match='training feature support'):ccc_loss(prediction,target,np.ones(2),np.ones(2,bool),nonzero_weight_max=8)
    torch.testing.assert_close(ordinary,ccc_loss(prediction,target,np.ones(2),np.array([True,False]),nonzero_fraction=np.array([.01,1.]),nonzero_weight_max=1))


def test_residual_large_expression_input_has_finite_gradients():
    torch.manual_seed(17)
    model=MicroenvironmentGNN(2049,10,2374,programs=32,hidden=128,message_layers=2,encoder_depth=2,dropout=.1)
    model.eval();expression=torch.randn(45,2049);ids=torch.stack((torch.arange(30),torch.arange(15,45)))
    xy=torch.randn(2,30,2)*15;out=model(expression,ids,xy,torch.ones(2,30),ids)
    assert not hasattr(model,'type_embedding') and out['Z'].shape==(2,32)
    assert torch.isfinite(out['Z']).all() and (out['Z']>=0).all()
    torch.testing.assert_close(out['reconstruction'],out['Z']@out['H'])
    target=torch.zeros(2,2374);target[:,:8]=5
    ccc_loss(out['reconstruction'],target,np.ones(2374),np.ones(2374,bool)).backward()
    for layer in model.message_blocks:
        assert layer.sender.weight.grad.abs().sum()>0 and torch.isfinite(layer.sender.weight.grad).all()
    clone=MicroenvironmentGNN(2049,10,2374,32,128,message_layers=2,encoder_depth=2,dropout=.1);clone.load_state_dict(model.state_dict());clone.eval()
    torch.testing.assert_close(out['Z'],clone(expression,ids,xy,torch.ones(2,30),ids)['Z'])


def test_residual_weighted_training_export_and_checkpoint_reuse(synthetic,tmp_path):
    from phenoniche.gnn.pipeline import load_checkpoint
    d=synthetic
    cfg=replace(GNNConfig(),architecture='expression_gnn_residual_v2',encoder_gene_selection='lr_plus_variance',programs=4,hidden=16,message_layers=2,encoder_depth=2,dropout=.1,ccc_nonzero_weight_max=8,batch_size=16,max_epochs=2,patience=2,device='cpu',output_dir=str(tmp_path))
    fit(d,cfg);restored,payload=load_checkpoint(d,cfg);arrays=infer(restored,d,tmp_path,16)
    history=pd.read_csv(tmp_path/'training.csv')
    assert {'validation_ccc_unweighted','validation_ccc_positive_huber','validation_ccc_gain_over_zero','validation_ccc_gain_over_mean'}<=set(history)
    assert np.isfinite(history.select_dtypes('number')).all().all() and 'train_composition' not in history
    export_reconstruction(d,arrays,np.load(tmp_path/'H.npy'),np.load(tmp_path/'B.npy'),cfg)
    metrics=json.loads((tmp_path/'reconstruction_metrics.json').read_text())['test']
    assert 'gain_over_train_mean' in metrics and metrics['target_nonzero_fraction']==1
    assert np.isfinite(metrics['native_relative_RMSE'])
    np.testing.assert_array_equal(payload['feature_nonzero_fraction'],d.feature_nonzero_fraction)
