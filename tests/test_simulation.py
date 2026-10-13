"""Only 180-cell tests; no full simulation/Notebook execution."""
from dataclasses import replace
from pathlib import Path
import ast,json,tempfile,hashlib
import numpy as np,pandas as pd,pytest,torch,nbformat
from phenoniche.storage import STORAGE_ROOT
from phenoniche.simulation.config import SimulationConfig
from phenoniche.simulation.workflow import dataset,run_experiment,protect
from phenoniche.simulation.adapter import adapt
from phenoniche.simulation.injection import inject
from phenoniche.simulation.generation import make_lr,calibrate_type_means,seed_reference,marker_audit,TYPES
from phenoniche.simulation.dependencies import prepare_environment
from phenoniche.simulation.plotting import SimulationReport
from phenoniche.simulation.evaluation import label_mapping,mapped,partition_scores
from phenoniche.simulation.baselines import dictionary_program_projection
from phenoniche.gnn.model import MicroenvironmentGNN
from phenoniche.gnn.losses import ccc_loss,composition_loss

@pytest.fixture(scope='module')
def tiny():
    torch.set_num_threads(2)
    root=STORAGE_ROOT/'tmp';root.mkdir(exist_ok=True,parents=True)
    with tempfile.TemporaryDirectory(prefix='tumor_edge_test_',dir=root) as path:
        cfg=replace(SimulationConfig(),n_cells=180,n_genes=128,n_lr=50,replicates=3,core_radius_um=28.,edge_width_um=24.,outer_radius_um=80.,block_um=30.,reference_cells_per_type=6,sc_cube_epochs=1,sc_cube_hidden=8,gnn_epochs=2,ae_stage1_epochs=2,ae_stage2_epochs=2,batch_size=16,hidden=12,k_max=6,device='cpu',figure_dpi=100,output_dir=path)
        prepare_environment();adata,lr=dataset(cfg);data=adapt(adata,lr,cfg,Path(path))
        yield cfg,adata,lr,data
        # NFS retains deleted mmap files as .nfs entries until every map closes.
        # Close the fixture's reusable graph/target maps before directory cleanup.
        for value in [*vars(data).values(),data.targets.matrix]:
            if isinstance(value,np.memmap):value._mmap.close()
        import gc
        gc.collect()


def test_cube_environment_and_real_LR_expression_injection(tiny):
    cfg,a,lr,d=tiny
    import anndata as ad
    before=ad.read_h5ad(Path(cfg.output_dir)/'before_injection.h5ad')
    assert a.shape==(180,128) and len(lr)==50 and (lr.mi==1).sum()==(lr.mi==2).sum()==6
    assert np.any(a.X!=before.X) and np.array_equal(a.obsm['spatial'],before.obsm['spatial'])
    non_lr=[i for i,g in enumerate(a.var_names) if g not in set(lr.ligand)|set(lr.receptor)]
    np.testing.assert_array_equal(a.X[:,non_lr],before.X[:,non_lr])
    assert 'scCube_commit' in a.uns['simulation_provenance']


def test_composition_and_physical_cell_splits(tiny):
    cfg,a,lr,d=tiny;n=len(a)//2
    np.testing.assert_allclose(d.composition_target[:n],d.composition_target[n:],atol=2e-6)
    np.testing.assert_allclose(d.composition_target.sum(1),1,atol=1e-6)
    used=[np.unique(d.members[r]) for r in d.rows]
    for i in range(3):
        for j in range(i):assert not np.intersect1d(used[i],used[j]).size
    assert d.targets.shape==(180,1250) and np.isfinite(d.targets[:]).all()


def test_current_gnn_forward_backward_no_truth_or_type_encoder(tiny):
    cfg,a,lr,d=tiny
    batch=d.batch(d.rows[0][:8]);model=MicroenvironmentGNN(len(d.encoder_indices)+1,5,len(d.features),32,12,20.,d.feature_mask)
    out=d.forward(model,batch);assert out['Z'].shape==(8,32) and out['B'].shape==(32,5)
    assert 'types' not in batch and not hasattr(model,'type_embedding')
    assert set(d.cells)=={'cell_id','cell_type'} and 'true_niche' not in batch
    loss=ccc_loss(out['reconstruction'],batch['target'],d.feature_scale,d.feature_mask)+.05*composition_loss(out['composition'],batch['composition_target']);loss.backward()
    assert model.cell_encoder[0].weight.grad.abs().sum()>0 and model.raw_composition.grad.abs().sum()>0


def test_optional_pairing_only_preserves_conditional_marginals(tiny):
    cfg,a,lr,d=tiny
    import anndata as ad
    before=ad.read_h5ad(Path(cfg.output_dir)/'before_injection.h5ad');strict=replace(cfg,mode='spatial_pairing_only',radial_bins=1)
    result=inject(before,lr,strict);truth=result.obs.true_niche.to_numpy()
    for sample in result.obs.sample_id.unique():
        for celltype in result.obs.cell_type.unique():
            keep=(result.obs.sample_id.to_numpy()==sample)&(result.obs.cell_type.to_numpy()==celltype)
            left=result.X[keep&(truth==2)];right=result.X[keep&(truth==3)]
            assert sorted(row.tobytes() for row in left)==sorted(row.tobytes() for row in right)
    assert np.array_equal(result.obsm['spatial'],before.obsm['spatial'])


def test_edge_mapping_cannot_assign_core_labels():
    truth=np.array([0,1,2,3,2,3]);pred=np.array([3,2,0,0,1,1]);rows=np.arange(2,6)
    m=label_mapping(pred,truth,rows)
    assert set(m.values())<={2,3};assert (mapped(pred[rows],m)==truth[rows]).mean()==.5


def test_configurable_dictionary_and_config_roundtrip(tmp_path):
    cfg=replace(SimulationConfig(),sender_type='Macrophage',receiver_type='T cell',mi_loadings=[[1,2,3,4,5,6],[6,5,4,3,2,1],[1,2,3,4,5,6],[6,5,4,3,2,1]])
    lr=make_lr(cfg)
    for mi in (1,2):
        weights=np.asarray(cfg.mi_loadings[mi-1],float)
        np.testing.assert_allclose(lr.loc[lr.mi==mi,'loading'],weights/weights.sum())
    assert set(lr.loc[lr.mi.isin([1,2]),'sender_role'])=={'Macrophage'} and set(lr.loc[lr.mi.isin([1,2]),'receiver_role'])=={'T cell'}
    assert set(lr.loc[lr.mi==3,'sender_role'])=={'Tumor'} and set(lr.loc[lr.mi==4,'receiver_role'])=={'Normal epithelial'}
    path=tmp_path/'config.json';path.write_text(json.dumps(cfg.to_dict()))
    assert SimulationConfig.from_json(path).signature()==cfg.signature()


def test_flat_vae_calibration_restores_types_without_niche_labels():
    cfg=replace(SimulationConfig(),reference_cells_per_type=6);ref=seed_reference(cfg,make_lr(cfg))
    labels=np.repeat(TYPES,12);flat=np.ones((len(labels),len(ref.var_names)))
    result=calibrate_type_means(flat,labels,ref)
    for name in TYPES:
        np.testing.assert_allclose(result[labels==name].mean(0),ref.X[ref.obs.Cell_type.to_numpy()==name].mean(0),atol=1e-6)
    records=marker_audit(result,labels,ref.var_names)
    assert all(r['own_marker_mean']>10*r['other_marker_mean'] for r in records)


def test_sccube_preprocessing_does_not_mutate_calibration_reference():
    from scCube import scCube
    cfg=replace(SimulationConfig(),reference_cells_per_type=6);ref=seed_reference(cfg,make_lr(cfg));before=ref.X.copy()
    normalized=scCube().pre_process(pd.DataFrame(ref.X.T.copy(),index=ref.var_names,columns=ref.obs.Cell),ref.obs.copy(),is_normalized=False)
    np.testing.assert_array_equal(ref.X,before)
    assert not np.array_equal(normalized.X,before)


def test_degenerate_test_partition_is_not_reported_as_failed_recovery():
    truth=np.array([2,3,1,1]);pred=np.array([0,1,0,1]);rows=np.array([2,3])
    scores=partition_scores(truth,pred,pred,rows,{0:2,1:3},np.array([0,1]))
    assert not scores['overall_evaluable'] and not scores['edge_detection_evaluable']
    assert np.isnan(scores['overall_ARI']) and np.isnan(scores['edge_F1'])


def test_dictionary_metric_respects_reconstruction_and_redundancy():
    h=np.array([[.7,.3],[.7,.3],[.1,.9]],np.float32);scale=np.array([.5,2.],np.float32)
    projection=dictionary_program_projection(h,scale)
    z=np.array([[1,0,0],[0,1,0],[0,0,1]],np.float32)
    delta=z[0]-z[2]
    np.testing.assert_allclose(np.sum((delta@projection)**2),np.sum((delta@h/scale)**2),rtol=1e-5)
    np.testing.assert_allclose(z[0]@projection,z[1]@projection,atol=1e-7)


def test_truth_labels_do_not_change_model_data(tiny):
    cfg,a,lr,d=tiny
    changed=a.copy();changed.obs['true_niche']=np.random.default_rng(71).permutation(a.obs.true_niche)
    output=Path(cfg.output_dir)/'truth_independence'
    other=adapt(changed,lr,cfg,output)
    try:
        for name in ('encoder_indices','gene_mean','feature_mask','folds','composition_target'):
            np.testing.assert_array_equal(getattr(d,name),getattr(other,name))
        np.testing.assert_array_equal(d.targets[:],other.targets[:])
    finally:
        for value in [*vars(other).values(),other.targets.matrix]:
            if isinstance(value,np.memmap):value._mmap.close()


def test_mini_end_to_end_and_checkpoint_reuse(tiny):
    cfg,a,lr,d=tiny;output=run_experiment(cfg)
    metrics=pd.read_csv(output/'metrics.csv');assert set(metrics.method)=={'Cell Composition','Expression PCA','CCC-AE','CCC_Phe GNN (fixed K)','CCC_Phe GNN (selected K)'}
    diag=json.loads((output/'diagnostics.json').read_text())
    assert diag['composition_only_grouped_CV_AUC']==pytest.approx(.5,abs=.02)
    comp=metrics[metrics.method=='Cell Composition'];assert np.allclose(comp.edge_AB_accuracy,.5)
    assert np.isfinite(metrics.overall_ARI).all() and np.isfinite(diag['gnn_test_CCC_Huber'])
    model=output/'gnn/checkpoint.pt';before=hashlib.sha256(model.read_bytes()).hexdigest();mtime=model.stat().st_mtime_ns
    assert run_experiment(cfg)==output and model.stat().st_mtime_ns==mtime and hashlib.sha256(model.read_bytes()).hexdigest()==before
    report=SimulationReport(output)
    import matplotlib.pyplot as plt
    for name in ('ground_truth','main_figure','recovered_programs','dictionary','composition_audit','adaptive_niches','reconstruction','summary_figure'):
        fig=getattr(report,name)();assert fig.axes;plt.close(fig)
    assert len(list((output/'figures').glob('*.png')))==8 and len(list((output/'figures').glob('*.pdf')))==8
    from phenoniche.simulation.summary_figure import regional_profiles
    fractions,means,selected=regional_profiles(report)
    assert fractions.shape==(4,5) and means.shape==(4,d.targets.shape[1])
    np.testing.assert_allclose(fractions.sum(1),1,atol=1e-5)
    assert len(selected)<=12 and np.isfinite(means).all()
    metadata=json.loads((output/'report_tables/summary_figure_metadata.json').read_text())
    assert metadata['n_planted_MIs']==4 and metadata['n_regions']==4 and metadata['removed_panels']==['b','d']
    with pytest.raises(ValueError):protect(replace(cfg,signal_strength=21.))


def test_notebook_static_order_and_valid_outputs():
    root=Path(__file__).resolve().parents[1];n=nbformat.read(root/'TumorEdge_Communication_Simulation.ipynb',as_version=4)
    sources=[]
    for c in n.cells:
        if c.cell_type=='code':
            assert not any(o.output_type=='error' for o in c.outputs)
            ast.parse('\n'.join(v for v in c.source.splitlines() if not v.startswith('%')));sources.append(c.source)
    text='\n'.join(sources)
    assert text.index('dataset(config)')<text.index('adapt(adata')<text.index('run_experiment(config)')
    assert 'load_results=False' in text and 'prepare_environment()' in text
    cfg=SimulationConfig.from_json(root/'config/tumor_edge_simulation.json');assert cfg.n_cells==8000 and cfg.mode=='expression'


def test_lr_background_is_type_specific_without_region_labels():
    cfg=replace(SimulationConfig(),reference_cells_per_type=8);lr=make_lr(cfg);ref=seed_reference(cfg,lr)
    mi_genes=set(lr.loc[lr.mi>0,'ligand'])|set(lr.loc[lr.mi>0,'receptor'])
    basal=lr[lr.basal_type!=''];assert len(basal)==5*cfg.basal_lr_per_type
    assert not mi_genes.intersection(set(basal.ligand)|set(basal.receptor))
    for name in TYPES:
        genes=sorted(set(basal.loc[basal.basal_type==name,'ligand'])|set(basal.loc[basal.basal_type==name,'receptor']))
        ids=[ref.var_names.get_loc(g) for g in genes];own=ref.obs.Cell_type.to_numpy()==name
        assert ref.X[own][:,ids].mean()>20*ref.X[~own][:,ids].mean()
    assert 'true_niche' not in ref.obs


def test_protected_real_data_outputs():
    with pytest.raises(ValueError):protect(replace(SimulationConfig(),output_dir=str(STORAGE_ROOT/'outputs/xenium_prime_5k')))


def test_four_program_truth_and_self_type_physical_pairs(tiny):
    cfg,a,lr,d=tiny;out=Path(cfg.output_dir)
    truth=np.load(out/'ground_truth/MI_dictionary.npy');activity=np.load(out/'ground_truth/MI_activity.npy')
    assert cfg.mi_count==4 and truth.shape==(4,d.targets.shape[1]) and activity.shape==(180,4)
    np.testing.assert_allclose(truth.sum(1),1)
    assert np.all((truth>0).sum(1)==cfg.mi_lr_count) and np.isfinite(activity).all()
    for mi,role in [(3,'Tumor'),(4,'Normal epithelial')]:
        features=d.features.iloc[np.flatnonzero(truth[mi-1])]
        assert set(features.sender)==set(features.receiver)=={role}
        assert activity[:,mi-1].max()>0
    np.testing.assert_array_equal(np.load(out/'ground_truth/MI_activity_before.npy'),np.zeros((180,4)))
    # Self-type communication still requires different physical cells.
    from phenoniche.gnn.model import physical_weights
    from phenoniche.simulation.programs import program_specs
    row=int(np.argmax(activity[:,2]));ids=np.asarray(d.members[row]);pair=physical_weights(torch.from_numpy(d.coordinates[ids][None]),torch.from_numpy(np.asarray(d.weights[row][None]).copy()),20.,torch.from_numpy(ids[None]))[0].numpy()
    assert np.diag(pair).sum()==0
    types=a.obs.cell_type.to_numpy()[ids];sender=a.obsm['injected_sender_activity'][ids,2]*(types=='Tumor');receiver=a.obsm['injected_receiver_activity'][ids,2]*(types=='Tumor')
    exposure=(np.sqrt(sender[:,None]*receiver[None,:])*pair).sum();opportunity=(pair*(types[:,None]=='Tumor')*(types[None,:]=='Tumor')).sum()
    assert activity[row,2]==pytest.approx(exposure/(opportunity+1e-4),rel=1e-5)
    assert len(set(lr.ligand)|set(lr.receptor))==2*cfg.n_lr
