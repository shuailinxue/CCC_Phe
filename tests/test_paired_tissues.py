"""Paired simulation: causal injection, null, held-out CCC/GNN and report."""
from dataclasses import replace
from pathlib import Path
import ast,json,tempfile
import anndata as ad
import numpy as np
import pandas as pd
import nbformat
import torch
from phenoniche.storage import STORAGE_ROOT
from phenoniche.simulation.paired_tissues import PairedConfig,simulate
from phenoniche.simulation.paired_workflow import run
from phenoniche.simulation.paired_workflow import encoder_panel


def scaffold():
    genes=tuple([f'L{i}' for i in range(75)]+[f'R{i}' for i in range(75)]+[f'SYN_BG{i}' for i in range(50)])
    means=np.ones((5,200),np.float32)
    for t in range(5):means[t,150+t*10:160+t*10]=6
    lr=pd.DataFrame({'lr_id':[f'UNIT_LR_{i}' for i in range(75)],'ligand':genes[:75],
                     'receptor':genes[75:150],'source':'Synthetic unit-test identities, not biological priors'})
    return genes,means,lr


def mini_config(**kwargs):
    return PairedConfig(cells_per_slice=100,radius_um=65,epochs=2,patience=2,
                        batch_size=32,hidden=12,nmf_iterations=20,k_max=6,device='cpu',figure_dpi=80,**kwargs)


def test_paired_expression_reproducibility_and_matching():
    genes,means,lr=scaffold();cfg=mini_config()
    a,_,_=simulate(cfg,genes,means,lr);b,_,_=simulate(cfg,genes,means,lr)
    np.testing.assert_array_equal(a.X,b.X)
    np.testing.assert_array_equal(a.X[:,150:],a.layers['before_injection'][:,150:])
    for pair in range(cfg.pairs):
        h=a[(a.obs.pair_id==pair)&(a.obs.condition=='Healthy')]
        c=a[(a.obs.pair_id==pair)&(a.obs.condition=='Cancer')]
        np.testing.assert_array_equal(h.obsm['spatial'],c.obsm['spatial'])
        np.testing.assert_array_equal(h.obs.cell_type,c.obs.cell_type)
    assert (a.X!=a.layers['before_injection']).any()
    assert not (a.obsm['true_program_activity'][a.obs.condition=='Healthy',5:]>0).any()
    assert (a.obsm['true_program_activity'][a.obs.condition=='Cancer',5:]>0).any()


def test_null_has_no_planted_disease_niche():
    a,_,_=simulate(replace(mini_config(),disease_strength=0),*scaffold())
    assert a.obs.true_niche.max()<=3
    assert not a.obsm['true_program_activity'][:,5:].any()


def test_added_training_pairs_preserve_original_slices():
    cfg=mini_config();old,_,_=simulate(cfg,*scaffold())
    new,_,_=simulate(replace(cfg,extra_training_pairs=2,diverse_training_geometry=True),*scaffold())
    np.testing.assert_array_equal(new.X[:len(old)],old.X)
    np.testing.assert_array_equal(new.obsm['spatial'][:len(old)],old.obsm['spatial'])
    np.testing.assert_array_equal(new.obs.fold.iloc[:len(old)],old.obs.fold)
    assert (new.obs.fold.iloc[len(old):]==0).all()
    assert new.obs.sample_id.nunique()==12


def test_encoder_selection_is_training_only_and_keeps_every_LR():
    cfg=replace(mini_config(),encoder_gene_selection='lr_markers')
    a,lr,_=simulate(cfg,*scaffold());genes=list(a.var_names)
    genes[150:175]=[f'SYN_MARKER_{i}' for i in range(25)];a.var_names=genes
    train=np.flatnonzero(a.obs.fold.to_numpy()==0)
    chosen,audit=encoder_panel(a,lr,train,cfg)
    assert len(chosen)==175 and audit['LR_genes_retained']==150
    a.X[a.obs.fold.to_numpy()!=0]=0
    a.obs['condition']='permuted';a.obs['true_niche']=-1;a.obs['cell_type']='ignored'
    other,_=encoder_panel(a,lr,train,cfg)
    np.testing.assert_array_equal(chosen,other)


def test_mini_joint_training_and_report():
    torch.set_num_threads(2)
    with tempfile.TemporaryDirectory(prefix='paired_smoke_',dir=STORAGE_ROOT/'tmp') as folder:
        root=Path(folder);source=root/'background';source.mkdir()
        genes,means,lr=scaffold()
        # Reuse adapter donor-type names only; synthetic unit scaffolds.
        types=['Normal epithelial','Fibroblast','Macrophage','T cell','Tumor']
        a=ad.AnnData(np.repeat(means,3,axis=0),obs=pd.DataFrame({'cell_type':np.repeat(types,3)}),var=pd.DataFrame(index=genes))
        a.write_h5ad(source/'before_injection.h5ad');lr.to_csv(source/'lr.csv',index=False)
        cfg=mini_config(background_dir=str(source),output_dir=str(root/'results'),
                        encoder_gene_selection='lr_markers',extra_training_pairs=2,
                        diverse_training_geometry=True,training_centers_per_slice=25)
        out=run(cfg);z=np.load(out/'gnn/Z.npy');h=np.load(out/'gnn/H.npy')
        assert z.shape==(1200,32) and h.shape==(32,1875)
        assert len(np.load(out/'input/training_rows.npy'))==200
        assert len(np.load(out/'input/preprocessing.npz')['encoder_indices'])==150
        assert np.isfinite(z).all() and (z>=0).all() and (h>=0).all()
        np.testing.assert_allclose(h.sum(1),1,atol=1e-5)
        np.testing.assert_allclose(np.load(out/'reconstruction.npy'),z@h,atol=1e-5)
        members=np.load(out/'input/members.npy');folds=np.load(out/'input/folds.npy')
        assert (folds[members]==folds[:,None]).all()
        a=ad.read_h5ad(out/'tissues.h5ad');sample=a.obs.sample_id.to_numpy()
        assert (sample[members]==sample[:,None]).all()
        audit=pd.read_csv(out/'pair_audit.csv');assert audit.max_weighted_composition_difference.max()<1e-6
        assert run(cfg)==out
        from phenoniche.simulation.paired_reporting import PairedReport
        import matplotlib.pyplot as plt
        r=PairedReport(out)
        for method in (r.cell_types,r.spatial,r.groups,r.programs,r.training):plt.close(method())
        assert len(list((out/'figures').glob('*.png')))==5
        del r
        # GraphData/Torch cycles can keep target memmaps open until collection;
        # collect before NFS TemporaryDirectory cleanup removes those files.
        import gc
        gc.collect()


def test_notebook_static():
    root=Path(__file__).resolve().parents[1]
    n=nbformat.read(root/'HealthyCancer_Communication_Niches.ipynb',4);nbformat.validate(n)
    for c in n.cells:
        if c.cell_type=='code':ast.parse('\n'.join(line for line in c.source.splitlines() if not line.startswith('%')))
    source='\n'.join(c.source for c in n.cells if c.cell_type=='code')
    assert source.index('run(config)')<source.index('report=PairedReport')<source.index('show(report.spatial())')
