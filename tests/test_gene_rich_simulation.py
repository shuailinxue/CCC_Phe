"""Gene-rich simulation tests; no full 8,000-cell run or Notebook execution."""
from dataclasses import replace
from pathlib import Path
import ast,json,tempfile
import numpy as np
import pandas as pd
import pytest
import torch
import nbformat
from phenoniche.storage import STORAGE_ROOT
from phenoniche.simulation.config import SimulationConfig,ROOT
from phenoniche.simulation.generation import make_lr
from phenoniche.simulation.rich_background import rich_reference,inject_moderate_mi,prepare_rich_background,simulation_audit
from phenoniche.simulation.response_simulation import ExpressionExperimentConfig,checksum
from phenoniche.simulation.response_experiment import run_expression_experiment


def profile():
    value=json.loads((ROOT/'config/tumor_edge_gene_rich_background.json').read_text())
    return value.get('source_background_profile',value)


def test_3000_gene_reference_structured_background():
    cfg=SimulationConfig.from_json(ROOT/'config/tumor_edge_gene_rich_simulation.json');lr=make_lr(cfg)
    reference=rich_reference(cfg,profile(),lr);repeat=rich_reference(cfg,profile(),lr)
    np.testing.assert_array_equal(reference.X,repeat.X)
    assert reference.n_vars==3000 and reference.var.kind.value_counts().to_dict()=={'type_program':1200,'housekeeping':825,'shared_state':800,'LR':150,'marker':25}
    assert set(lr.ligand)|set(lr.receptor)<=set(reference.var_names)
    a=reference.obs.Cell_type.to_numpy()=='Tumor';g=(reference.var.kind.to_numpy()=='type_program')&(reference.var.module.to_numpy()==0)
    assert reference.X[a][:,g].mean()>3*reference.X[~a][:,g].mean()
    assert reference.X[:,reference.var.kind.to_numpy()=='LR'].sum()/reference.X.sum()<.04
    assert (reference.X>=0).all() and np.equal(reference.X,np.floor(reference.X)).all()


@pytest.fixture(scope='module')
def mini():
    torch.set_num_threads(2)
    with tempfile.TemporaryDirectory(prefix='gene_rich_smoke_',dir=STORAGE_ROOT/'tmp') as path:
        root=Path(path);p=profile()
        cfg=replace(SimulationConfig.from_json(ROOT/'config/tumor_edge_gene_rich_simulation.json'),n_cells=180,n_lr=50,replicates=3,core_radius_um=28.,edge_width_um=24.,outer_radius_um=80.,reference_cells_per_type=6,sc_cube_epochs=1,sc_cube_hidden=8,gnn_epochs=2,ae_stage1_epochs=2,ae_stage2_epochs=2,batch_size=16,hidden=12,k_max=6,device='cpu',figure_dpi=80,output_dir=str(root/'source'))
        source_config=root/'simulation.json';source_config.write_text(json.dumps(cfg.to_dict()));pp=root/'background.json';pp.write_text(json.dumps(p))
        yield root,cfg,p,source_config,pp


def test_moderate_injection_and_preparation_cache(mini):
    import anndata as ad
    root,cfg,p,source_config,pp=mini
    source=prepare_rich_background(source_config,pp);before=ad.read_h5ad(source/'before_injection.h5ad');after=ad.read_h5ad(source/'injected.h5ad');lr=pd.read_csv(source/'lr.csv')
    assert after.shape==(180,3000)
    non_lr=after.var.kind.to_numpy()!='LR';n=len(after)//2
    np.testing.assert_array_equal(after.X[:,non_lr],before.X[:,non_lr]);np.testing.assert_array_equal(after.X[:n,non_lr],after.X[n:,non_lr])
    np.testing.assert_array_equal(after.obsm['spatial'],before.obsm['spatial'])
    assert np.any(after.X!=before.X) and np.equal(after.X,np.floor(after.X)).all()
    repeated=inject_moderate_mi(before,lr,cfg,p);np.testing.assert_array_equal(repeated.X,after.X)
    audit=simulation_audit(before,after,lr,p);assert audit['edge_MI_genes']==24 and audit['non_LR_background_exactly_mirrored']
    assert audit['all_LR_count_fraction']<.3
    stamp=(source/'injected.h5ad').stat().st_mtime_ns
    assert prepare_rich_background(source_config,pp)==source and (source/'injected.h5ad').stat().st_mtime_ns==stamp
    changed=dict(p,mi_fold_change=2.1);other=root/'changed.json';other.write_text(json.dumps(changed))
    with pytest.raises(ValueError):prepare_rich_background(source_config,other)


def test_gene_rich_end_to_end_existing_models_and_baselines(mini):
    root,base,p,source_config,pp=mini;prepare_rich_background(source_config,pp)
    from phenoniche.simulation.coherent_background import prepare_coherent_background
    coherent=replace(base,output_dir=str(root/'coherent'),expression_coverage=.1)
    source_config=root/'coherent.json';source_config.write_text(json.dumps(coherent.to_dict()))
    value=json.loads((ROOT/'config/tumor_edge_gene_rich_background.json').read_text());value['background_source_dir']=base.output_dir
    value.pop('core_exterior_activity_floor',None);value.pop('boundary_source_dir',None)
    cpp=root/'coherent_profile.json';cpp.write_text(json.dumps(value))
    source=prepare_coherent_background(source_config,cpp)
    stamp=(source/'injected.h5ad').stat().st_mtime_ns
    assert prepare_coherent_background(source_config,cpp)==source and (source/'injected.h5ad').stat().st_mtime_ns==stamp
    # The current notebook's constant core/exterior scenario gets a separate
    # expression dataset and fresh models/baselines, preserving the ramped one.
    from phenoniche.simulation.boundary_background import prepare_boundary_background
    boundary=replace(coherent,output_dir=str(root/'boundary'))
    boundary_config=root/'boundary.json';boundary_config.write_text(json.dumps(boundary.to_dict()))
    value.update(core_exterior_activity_floor=1.,boundary_source_dir=str(source))
    bpp=root/'boundary_profile.json';bpp.write_text(json.dumps(value))
    old_hash=checksum(source/'injected.h5ad')
    new_source=prepare_boundary_background(boundary_config,bpp)
    assert prepare_boundary_background(boundary_config,bpp)==new_source
    assert checksum(source/'injected.h5ad')==old_hash
    source_config=boundary_config
    cfg=replace(ExpressionExperimentConfig(),base_simulation_config=str(source_config),output_dir=str(root/'joint'),frozen_control_dir=str(root/'frozen'),device='cpu',expression_epochs=2,joint_epochs=2,patience=2,batch_size=16,training_mode='joint',figure_dpi=80)
    out=run_expression_experiment(cfg)
    assert np.load(out/'gnn/Z.npy').shape==(180,32)
    saved=torch.load(out/'gnn/checkpoint.pt',map_location='cpu',weights_only=False)
    assert len(saved['encoder_indices'])==3000 and saved['config']['encoder_genes']==3000
    assert np.load(out/'expression/prediction.npy').shape==(180,5,3036)
    summary=json.loads((out/'summary.json').read_text());assert summary['base_genes']==3000 and summary['response_genes_excluded_from_encoder']
    original=checksum(out/'gnn/checkpoint.pt')
    from baselines.benchmark import run_comparison
    bc=json.loads((ROOT/'baselines/config.json').read_text());bc.update(device='cpu',graphst_epochs=2,graphst_hidden=8,pca_components=4,gmm_epochs=5,scniche_epochs=2,scniche_batch_size=32,scniche_graph_k=5,scniche_mik=2,nichecompass_epochs=2,nichecompass_addon_programs=4,nichecompass_hidden=12,nichecompass_node_batch=16,spaniche_topics=4,spaniche_maxiter=4,spaniche_initial_maxiter=4,spaniche_initial_restarts=1,stagate_epochs=2,stagate_hidden_dims=[12,4],banksy_k_geom=3)
    bp=root/'baseline.json';bp.write_text(json.dumps(bc));comparison=run_comparison(out,bp)
    assert {'GraphST','CellCharter (PCA)','scNiche','NicheCompass','SpaNiche'}.issubset(set(pd.read_csv(comparison/'metrics.csv').method))
    assert checksum(out/'gnn/checkpoint.pt')==original
    from phenoniche.simulation.joint_reporting import JointExpressionReport
    from baselines.reporting import BaselineReport
    import matplotlib.pyplot as plt
    report=JointExpressionReport(out);benchmark=BaselineReport(out,comparison)
    for name in ('spatial_comparison','communication_recovery','expression_recovery','niche_cell_types','curves','summary_figure'):
        fig=getattr(report,name)();assert fig.axes;plt.close(fig)
    for fig in (benchmark.spatial(),benchmark.metric_plot()):assert fig.axes;plt.close(fig)
    from phenoniche.simulation.niche_refinement import refine_saved_niches
    refined=refine_saved_niches(out,root/'refined')
    assert refine_saved_niches(out,root/'refined')==refined
    assert checksum(out/'gnn/checkpoint.pt')==original==checksum(refined/'gnn/checkpoint.pt')
    assert checksum(refined/'expression/prediction.npy')==checksum(out/'expression/prediction.npy')
    refinement=json.loads((refined/'summary.json').read_text());assert refinement['niche_refinement']['local_mixing']==.2
    q=np.load(refined/'gnn/selected_model/Q.npy');contribution=pd.read_csv(refined/'gnn/selected_model/program_contribution.csv',index_col=0).to_numpy()
    tr=np.flatnonzero(np.load(refined/'anchor_inputs/input/folds.npy')==0)
    z=np.load(refined/'gnn/Z.npy');expected=q[tr].astype(float).T@z[tr];expected/=q[tr].sum(0)[:,None]
    np.testing.assert_allclose(contribution,expected,rtol=1e-5,atol=1e-5)
    refined_comparison=run_comparison(refined,bp)
    metrics=pd.read_csv(refined_comparison/'metrics.csv');assert 'CCC-only GNN (unrefined)' in set(metrics.method)
    updated=JointExpressionReport(refined);assert np.array_equal(updated.old_labels,pd.read_csv(refined/'ccc_only/raw_clusters.csv')['CCC_Phe GNN (fixed K)'])
    for name in ('spatial_comparison','niche_cell_types','summary_figure'):
        fig=getattr(updated,name)();assert fig.axes;plt.close(fig)
    for holder in (report,benchmark.reference):
        for value in vars(holder).values():
            if isinstance(value,np.memmap):value._mmap.close()


def test_new_notebook_static_and_old_notebook_preserved(tmp_path):
    import importlib.util
    original=ROOT/'TumorEdge_CCC_Expression_Reconstruction.ipynb';before=checksum(original)
    spec=importlib.util.spec_from_file_location('rich_builder',ROOT/'scripts/build_tumor_edge_gene_rich_notebook.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);import shutil;shutil.copy2(original,tmp_path/original.name);module.ROOT=tmp_path;path=module.build()
    assert checksum(original)==before and path.name=='TumorEdge_GeneRich_CCC_Expression_Reconstruction.ipynb'
    notebook=nbformat.read(path,as_version=4);nbformat.validate(notebook);source=''
    for cell in notebook.cells:
        if cell.cell_type=='code':
            ast.parse('\n'.join(line for line in cell.source.splitlines() if not line.startswith('%')));assert not cell.outputs;source+=cell.source
    assert source.index('prepare_boundary_background(config.base_simulation_config')<source.index('run_expression_experiment(config)')<source.index('run_comparison(OUTPUT)')
    a=ExpressionExperimentConfig.from_json(ROOT/'config/tumor_edge_gene_rich_expression.json');b=ExpressionExperimentConfig.from_json(ROOT/'config/tumor_edge_expression.json')
    assert a.output_dir!=b.output_dir and a.frozen_control_dir!=b.frozen_control_dir and a.device=='cuda:2'
