import numpy as np,pandas as pd,scipy.sparse as sp
from phenoniche.cohorts.kidney_response import (
    target_weights,response_score,sender_exposure,spatial_association,partial_rank,ccc_genes,is_immunoglobulin,
)


def test_prior_projection_respects_panel_detection_and_LR_exclusions():
    prior=pd.Series([.9,.8,.7,.6,.5],index=['ABSENT','LIG','LOW','T1','T2'])
    result=target_weights(prior,['LIG','LOW','T1','T2'],[.9,.01,.3,.7],{'LIG'},top=2)
    assert result.index.tolist()==['T1','T2']
    np.testing.assert_allclose(result,[.6,.5])
    features=pd.DataFrame({'ligand':['HLA-DRA'],'receptor':['CD74&CXCR4']})
    assert ccc_genes(features)=={'HLA-DRA','CD74','CXCR4'}
    assert all(is_immunoglobulin(g) for g in ['IGHG1','IGHA2','IGHM','IGKC','IGHV3-23'])
    assert not is_immunoglobulin('IGHMBP2') and not is_immunoglobulin('IRF1')


def test_response_gene_order_alignment_and_exact_baseline_centering():
    x=sp.csr_matrix(np.arange(35,dtype=float).reshape(7,5)/20)
    genes=np.array(['A','B','C','D','E']);targets=pd.Series([1,2,3,4,5.],index=genes)
    baseline=np.array([0,1,2]);score=response_score(x,genes,baseline,targets)
    order=np.array([4,2,0,3,1]);other=response_score(x[:,order],genes[order],baseline,targets)
    np.testing.assert_allclose(score,other,atol=1e-7)
    assert abs(score[baseline].mean())<1e-7 and score[-1]>score[0]
    assert response_score(x,genes,baseline,targets.iloc[:4]) is None


def test_sender_exposure_uses_sender_RNA_and_original_physical_weights():
    types=np.array(['TAL','TAL','CD8+','Macro'])
    u=np.array([0,1,0]);v=np.array([2,2,3]);w=np.array([1.,3.,2.])
    x=np.array([[2.],[6.],[999.],[999.]])
    exposure,opportunity=sender_exposure(u,v,w,types,np.array([2,3]),x)
    np.testing.assert_allclose(exposure[[2,3],0],[5.,2.])
    np.testing.assert_allclose(opportunity[[2,3]],[4.,2.])
    x[2:]=-100000
    updated,_=sender_exposure(u,v,w,types,np.array([2,3]),x)
    np.testing.assert_array_equal(exposure,updated)


def test_spatial_association_adjusts_confounding_and_repeats_seed():
    rng=np.random.default_rng(52173);n=800;c=rng.normal(size=n)
    x=5*c+rng.normal(size=n);y=5*c+rng.normal(size=n)
    rx,ry=partial_rank(x,y,c[:,None])
    assert abs(np.corrcoef(rx,ry)[0,1])<.15
    xy=np.c_[np.repeat(np.arange(40),20)*200,np.tile(np.arange(20),40)]
    a,*_=spatial_association(x,y,xy,c[:,None],bootstraps=49)
    b,*_=spatial_association(x,y,xy,c[:,None],bootstraps=49)
    assert a==b and a['blocks']==40 and a['cells']==n
