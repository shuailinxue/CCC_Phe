"""Lightweight actual upstream GraphST/CellCharter execution, not dummy labels."""
from dataclasses import replace
from pathlib import Path
import ast,json,tempfile
import numpy as np
import pandas as pd
import pytest
import torch
import nbformat
from phenoniche.storage import STORAGE_ROOT
from phenoniche.simulation.config import SimulationConfig
from phenoniche.simulation.response_simulation import ExpressionExperimentConfig,checksum
from phenoniche.simulation.response_experiment import run_expression_experiment
from baselines.adapters import clean_input,graph_from_members
from baselines.benchmark import run_comparison,common_test_rows


def test_baseline_annotations_and_graph_identity():
    a=clean_input(np.ones((4,3)),['a','b','c'],np.arange(8).reshape(4,2),['one']*4)
    assert list(a.obs)==['sample_id'] and not a.uns
    m=np.array([[0,1],[1,0],[2,3],[3,2]])
    g=graph_from_members(m)
    assert g.diagonal().sum()==0 and g.nnz==4
    assert set(g[0].indices)=={1}


def test_common_dependency_purge_drops_cross_fold_edges():
    members=np.array([[0,1],[1,0],[2,3],[3,2],[4,5],[5,4]])
    folds=np.array([0,0,1,1,2,2]);g=graph_from_members(members)
    np.testing.assert_array_equal(common_test_rows(members,g,folds),[4,5])
    g=g.tolil();g[4,0]=1
    with pytest.raises(ValueError,match='purge'):common_test_rows(members,g.tocsr(),folds)


def test_actual_upstream_baseline_smoke():
    torch.set_num_threads(2)
    with tempfile.TemporaryDirectory(prefix='baseline_smoke_',dir=STORAGE_ROOT/'tmp') as path:
        root=Path(path)
        base=replace(SimulationConfig(),n_cells=180,n_genes=128,n_lr=50,replicates=3,core_radius_um=28.,edge_width_um=24.,outer_radius_um=80.,reference_cells_per_type=6,sc_cube_epochs=1,sc_cube_hidden=8,gnn_epochs=2,ae_stage1_epochs=2,ae_stage2_epochs=2,batch_size=16,hidden=12,k_max=6,device='cpu',figure_dpi=80,dictionary_initialization='nmf',output_dir=str(root/'source'))
        frozen=replace(ExpressionExperimentConfig(),output_dir=str(root/'frozen'),device='cpu',expression_epochs=2,patience=2,batch_size=16,figure_dpi=80)
        run_expression_experiment(frozen,base)
        joint=replace(frozen,training_mode='joint',output_dir=str(root/'joint'),frozen_control_dir=str(root/'frozen'),joint_epochs=2)
        experiment=run_expression_experiment(joint,base)
        cfg=json.loads((Path(__file__).resolve().parents[1]/'baselines/config.json').read_text());cfg.update(device='cpu',graphst_epochs=2,graphst_hidden=8,pca_components=4,gmm_epochs=5,figure_dpi=80)
        config=root/'config.json';config.write_text(json.dumps(cfg))
        original=checksum(experiment/'gnn/checkpoint.pt');out=run_comparison(experiment,config)
        table=pd.read_csv(out/'metrics.csv');assert {'GraphST','CellCharter (PCA)'}.issubset(set(table.method))
        assert table.loc[table.scope=='common_spatial_test','n'].nunique()==1
        labels=pd.read_csv(out/'labels.csv');assert len(labels)==180 and labels['GraphST'].nunique()==4
        for p in (out/'graphst/embedding.npy',out/'cellcharter/embedding.npy'):assert np.isfinite(np.load(p)).all()
        assert checksum(experiment/'gnn/checkpoint.pt')==original
        assert run_comparison(experiment,config)==out
        from baselines.reporting import BaselineReport
        import matplotlib.pyplot as plt
        report=BaselineReport(experiment,out)
        for fig in (report.spatial(),report.spatial(True),report.metric_plot()):plt.close(fig)
        assert len(list(report.figures.glob('*.png')))==3
        for value in vars(report.reference).values():
            if isinstance(value,np.memmap):value._mmap.close()


def test_notebook_tail_preserves_execution_and_valid_order():
    path=Path(__file__).resolve().parents[1]/'TumorEdge_CCC_Expression_Reconstruction.ipynb'
    notebook=nbformat.read(path,as_version=4);nbformat.validate(notebook)
    source=''
    for cell in notebook.cells:
        if cell.cell_type=='code':
            ast.parse('\n'.join(line for line in cell.source.splitlines() if not line.startswith('%')));source+=cell.source
    assert source.index('run_comparison(OUTPUT)')<source.index('baseline_report.spatial')<source.index('baseline_report.metric_table')
