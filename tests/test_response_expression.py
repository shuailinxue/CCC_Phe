"""Lightweight 180-cell response/decoder tests; never run full 8,000-cell Notebook."""
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import ast,json,tempfile
import anndata as ad
import numpy as np
import pandas as pd
import pytest
import torch
import nbformat
from phenoniche.storage import STORAGE_ROOT
from phenoniche.simulation.config import SimulationConfig
from phenoniche.simulation.workflow import dataset,run_experiment
from phenoniche.simulation.adapter import adapt
from phenoniche.simulation.response_simulation import ExpressionExperimentConfig,inject_responses,checksum
from phenoniche.simulation.response_experiment import run_expression_experiment
from phenoniche.simulation.expression_decoder import weighted_expression_targets,FrozenProgramExpressionDecoder,preprocessing


def test_weighted_expression_targets_and_absent_types():
    a=ad.AnnData(np.array([[1,3],[4,1],[8,2]],np.float32),obs=pd.DataFrame({'cell_id':['a','b','c']},index=['a','b','c']))
    a.obsm['spatial']=np.zeros((3,2),np.float32)
    data=SimpleNamespace(cells=a.obs.copy(),coordinates=a.obsm['spatial'],categories=('A','B','C'),types=np.array([0,1,0]),members=np.tile(np.arange(3),(3,1)),weights=np.ones((3,3),np.float32))
    target,present,norm=weighted_expression_targets(a,data,1)
    np.testing.assert_allclose(target[:,0,1],np.log1p((30000+2500)/2),rtol=1e-6)
    np.testing.assert_allclose(target[:,1,1],np.log1p(2500),rtol=1e-6)
    np.testing.assert_allclose(norm[:,0],5625);assert present[:,:2].all() and not present[:,2].any()
    assert (target[:,2]==0).all()
    with pytest.raises(ValueError):weighted_expression_targets(a[[2,1,0]],data,1)


def test_expression_gradient_cannot_change_programs():
    model=FrozenProgramExpressionDecoder(np.zeros(4,np.float32),np.ones(4,np.float32),np.ones((2,3),np.float32),np.ones((2,3),np.float32),np.ones((2,3),bool))
    z=torch.ones(5,4,requires_grad=True);norm=torch.ones(5,2)
    target=model(z,norm).detach()+.5;loss,count=model.loss(z,target,torch.ones(5,2,dtype=torch.bool),norm);loss.backward()
    assert count==30 and z.grad is None and model.dictionary.grad.abs().sum()>0
    assert all(name=='dictionary' for name,_ in model.named_parameters())


def test_expression_preprocessing_training_only():
    rng=np.random.default_rng(12);z=rng.uniform(size=(10,4)).astype(np.float32);target=rng.uniform(size=(10,2,3)).astype(np.float32)
    present=np.ones((10,2),bool);norm=np.ones((10,2),np.float32);train=np.arange(5)
    first=preprocessing(z,target,present,norm,train);target[5:]=1000;z[5:]=1000
    second=preprocessing(z,target,present,norm,train)
    for a,b in zip(first,second):np.testing.assert_array_equal(a,b)


@pytest.fixture(scope='module')
def mini():
    torch.set_num_threads(2)
    with tempfile.TemporaryDirectory(prefix='ccc_expression_test_',dir=STORAGE_ROOT/'tmp') as path:
        root=Path(path)
        base=replace(SimulationConfig(),n_cells=180,n_genes=128,n_lr=50,replicates=3,core_radius_um=28.,edge_width_um=24.,outer_radius_um=80.,reference_cells_per_type=6,sc_cube_epochs=1,sc_cube_hidden=8,gnn_epochs=2,ae_stage1_epochs=2,ae_stage2_epochs=2,batch_size=16,hidden=12,k_max=6,device='cpu',figure_dpi=80,dictionary_initialization='nmf',output_dir=str(root/'base'))
        a,lr=dataset(base);data=adapt(a,lr,base,root/'test_adapter')
        expression=replace(ExpressionExperimentConfig(),output_dir=str(root/'expression'),device='cpu',expression_epochs=3,patience=2,batch_size=16,figure_dpi=80)
        yield root,base,expression,a,lr,data
        for value in [data.targets.matrix,*vars(data).values()]:
            if isinstance(value,np.memmap):value._mmap.close()
        import gc
        gc.collect()


def test_synthetic_response_design_no_truth_or_LR_modification(mini):
    root,base,cfg,a,lr,data=mini
    enriched,table,exposure,before=inject_responses(a,lr,data,base,cfg)
    assert enriched.shape==(180,164) and exposure.shape==(180,4)
    np.testing.assert_array_equal(enriched.X[:,:128],a.X)
    np.testing.assert_array_equal(enriched.obsm['spatial'],a.obsm['spatial'])
    null=np.flatnonzero(table.kind.to_numpy()=='negative_control')
    np.testing.assert_array_equal(enriched.X[:,128+null],before[:,null])
    assert (table.direction==1).sum()==(table.direction==-1).sum()==12
    shuffled=a.copy();shuffled.obs['true_niche']=np.random.default_rng(18).permutation(a.obs.true_niche)
    alternative,_,alternative_exposure,_=inject_responses(shuffled,lr,data,base,cfg)
    np.testing.assert_array_equal(alternative.X,enriched.X);np.testing.assert_array_equal(alternative_exposure,exposure)


def test_response_experiment_end_to_end_and_frozen_anchor(mini):
    root,base,cfg,a,lr,data=mini
    output=run_expression_experiment(cfg,base)
    summary=json.loads((output/'summary.json').read_text())
    assert summary['CCC_weights_Z_H_niches_unchanged'] and summary['response_genes_excluded_from_encoder']
    assert summary['base_genes']==128 and summary['expression_genes']==164
    before=checksum(Path(base.output_dir)/'gnn/checkpoint.pt');model=output/'expression/checkpoint.pt';stamp=model.stat().st_mtime_ns
    assert run_expression_experiment(cfg,base)==output
    assert model.stat().st_mtime_ns==stamp and checksum(Path(base.output_dir)/'gnn/checkpoint.pt')==before
    from phenoniche.simulation.expression_reporting import ExpressionReport
    import matplotlib.pyplot as plt
    report=ExpressionReport(output)
    for name in ('architecture','curves','recovery','controls','dictionary','response_maps'):
        figure=getattr(report,name)();assert figure.axes;plt.close(figure)
    sensitivity=report.program_ablation();assert len(sensitivity)==5*164
    assert len(list((output/'figures').glob('*.png')))==6 and len(list((output/'figures').glob('*.pdf')))==6
    assert checksum(Path(base.output_dir)/'gnn/checkpoint.pt')==before
    # Close report's NFS maps before deleting the mini-test directory.
    for value in vars(report).values():
        if isinstance(value,np.memmap):value._mmap.close()


def test_new_expression_notebook_static():
    root=Path(__file__).resolve().parents[1]
    notebook=nbformat.read(root/'TumorEdge_CCC_Expression_Reconstruction.ipynb',as_version=4);nbformat.validate(notebook)
    text=''
    for cell in notebook.cells:
        if cell.cell_type=='code':
            ast.parse('\n'.join(line for line in cell.source.splitlines() if not line.startswith('%')))
            assert not any(o.output_type=='error' for o in cell.outputs);text+=cell.source
    assert text.index('run_expression_experiment(config)')<text.index('report=JointExpressionReport(OUTPUT)')
    cfg=ExpressionExperimentConfig.from_json(root/'config/tumor_edge_expression.json')
    assert cfg.device=='cuda:2' and cfg.output_dir.startswith(str(STORAGE_ROOT))


def test_joint_decoder_updates_Z_and_protects_H():
    from phenoniche.simulation.joint_expression import protected_gradients
    encoder=torch.nn.Parameter(torch.tensor([1.,2.]));h=torch.nn.Parameter(torch.tensor([.5]))
    primary=(encoder.square()).sum()+h.square().sum()
    auxiliary=-100*encoder.sum()+100*h.sum()
    guard=protected_gradients(primary,auxiliary,[encoder],[h],.1)
    p=2*encoder.detach();a=encoder.grad-p
    assert torch.dot(p,a)>=-1e-6 and a.norm()<=.1*p.norm()+1e-6
    torch.testing.assert_close(h.grad,2*h.detach())
    assert guard['auxiliary_conflict']==1.
    model=FrozenProgramExpressionDecoder(np.zeros(4,np.float32),np.ones(4,np.float32),np.ones((2,3),np.float32),np.ones((2,3),np.float32),np.ones((2,3),bool),detach_programs=False)
    with torch.no_grad():model.dictionary.fill_(.1)
    z=torch.ones(5,4,requires_grad=True);norm=torch.ones(5,2)
    loss,_=model.loss(z,model(z,norm).detach()+.5,torch.ones(5,2,dtype=torch.bool),norm);loss.backward()
    assert z.grad is not None and z.grad.abs().sum()>0


def test_joint_end_to_end_and_compact_figures(mini):
    from phenoniche.simulation.joint_expression import run_joint_experiment
    from phenoniche.simulation.joint_reporting import JointExpressionReport
    import matplotlib.pyplot as plt
    root,base,cfg,a,lr,data=mini
    control=run_expression_experiment(cfg,base)
    joint=replace(cfg,training_mode='joint',output_dir=str(root/'joint'),frozen_control_dir=str(control),joint_epochs=2,patience=2)
    before=checksum(Path(base.output_dir)/'gnn/checkpoint.pt')
    output=run_joint_experiment(joint,base)
    assert run_joint_experiment(joint,base)==output
    report=JointExpressionReport(output)
    assert report.z.shape==(180,32) and np.isfinite(report.prediction).all()
    assert (report.z>=0).all() and (report.h>=0).all()
    assert report.summary['validation_CCC_after']<=report.summary['validation_CCC_before']*1.005+1e-6
    assert len(report.top_ccc)==10*report.summary['selected_K']
    for name in ('spatial_comparison','communication_recovery','expression_recovery','niche_cell_types','curves','summary_figure'):
        fig=getattr(report,name)();assert fig.axes;plt.close(fig)
    assert len(list(report.figures.glob('*.png')))==6
    assert checksum(Path(base.output_dir)/'gnn/checkpoint.pt')==before
    for value in vars(report).values():
        if isinstance(value,np.memmap):value._mmap.close()
