"""Biological coverage, mass conservation, physical alignment and exact halos."""
import os
os.environ['PYTHONDONTWRITEBYTECODE']='1'
import numpy as np,pandas as pd,torch
from phenoniche.phenomap_adapter.genes import select_panel
from phenoniche.phenomap_adapter.model import zero_offset_scale,graph,halo_ids,make_model
from phenoniche.phenomap_adapter.annotation import decisions
from phenoniche.phenomap_adapter.features import instance_centers


def test_complete_LR_complexes_and_markers_never_cut_for_target_size():
    atlas=pd.DataFrame({'ligand':['L','MISSING'],'receptor':['R1&R2','R1']})
    genes,audit=select_panel([['L','R1','R2','CD3D','TRAC']],atlas,target=2,maximum=20)
    assert set(genes)=={'L','R1','R2','CD3D','TRAC'} and audit['complete_LR_pairs']==1


def test_marker_missing_in_one_sample_is_not_deleted_from_entire_cohort():
    atlas=pd.DataFrame({'ligand':['L','PRIVATE'],'receptor':['R','R']})
    genes,audit=select_panel([['L','R','PRIVATE','MS4A1'],['L','R','CD19']],atlas)
    assert set(genes)=={'L','R','MS4A1','CD19'}
    assert audit['complete_LR_pairs']==1
    assert audit['marker_source_coverage']=={'CD19':1,'MS4A1':1}
    assert audit['markers_not_shared_by_all']==['CD19','MS4A1']
    assert 'MS4A1' not in audit['missing_markers']['B_cells']


def test_source_absence_is_explicit_and_column_case_is_preserved(tmp_path):
    from phenoniche.phenomap_adapter.pipeline import read_counts
    path=tmp_path/'cnts.csv'
    pd.DataFrame({'spots':['s1','s2'],'Cd19':[2,0],'L':[1,3]}).to_csv(path,index=False)
    counts=read_counts(path,['CD19','MS4A1','L'])
    assert counts.attrs['source_gene_present']==[True,False,True]
    np.testing.assert_array_equal(counts.to_numpy(),[[2,0,1],[0,0,3]])


def test_zero_offset_export_is_nonnegative_and_mass_conserving():
    raw=np.array([[20.,5.,0.],[50.,10.,0.]],np.float32);scaled,scale,active=zero_offset_scale(raw)
    cells=np.array([[.1,.2,0],[.3,.3,0]],np.float32)
    np.testing.assert_allclose((cells*scale).sum(0),cells.sum(0)*scale)
    np.testing.assert_allclose(scaled*scale,raw)
    assert (scaled>=0).all() and active.tolist()==[True,True,False]


def test_confident_annotation_does_not_require_two_methods_to_agree():
    p=np.array([[.9,.1],[.51,.49],[.8,.2]])
    labels,confidence,margin,support=decisions(p,['T_cells','Myeloid'],np.array([[2,0],[2,0],[0,0]]),np.array([1.,1.]))
    assert labels.tolist()==['T_cells','Unknown','Unknown']


def test_segmentation_centroid_ownership_does_not_split_instances():
    mask=np.zeros((20,20),np.int32);mask[6:14,6:14]=1
    a,_=instance_centers(mask,(0,20,0,20,0,10,0,20),minimum=3)
    b,_=instance_centers(mask,(0,20,0,20,10,20,0,20),minimum=3)
    assert len(a)+len(b)==1 and (a+b)[0][3]==8


def test_halo_inference_equals_whole_graph_and_has_gradients():
    torch.manual_seed(7)
    c={'phenomap_repo':'/home/xueshuailin/PhenoMap','learning_rate':.001,'hidden_dim':16,'layers':2,'heads':2,'dropout':0}
    xy=np.c_[np.arange(40)*5.,np.zeros(40)].astype(np.float32)
    rng=np.random.default_rng(7);local=rng.normal(size=(40,8)).astype(np.float32);ctx=rng.normal(size=(40,8)).astype(np.float32)
    model=make_model(c,8,5);model.eval();full=graph(local,ctx,xy,11.,4);y=model(full.x,full.x_context,full.edge_index)
    core=np.array([19,20]);halo=halo_ids(xy,core,11.,2);g=graph(local[halo],ctx[halo],xy[halo],11.,4)
    pred=model(g.x,g.x_context,g.edge_index)
    torch.testing.assert_close(y[core],pred[np.searchsorted(halo,core)],atol=1e-7,rtol=1e-5)
    assert y.shape==(40,5) and (y>=0).all();y.sum().backward()
    assert any(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum()>0 for p in model.parameters())
