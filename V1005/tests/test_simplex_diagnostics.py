import numpy as np
from scipy import sparse
from phenoniche.v1005.simplex_training import split_mask,normalized_input
from phenoniche.v1005.ccc_tensor import entry_holdout
from phenoniche.v1005.diagnostics import latent_audit,graph_audit

def test_validation_test_disjoint_and_zero_input_safe():
    r=np.arange(80)[:,None];c=np.arange(90)[None];val=split_mask(r,c,True);test=split_mask(r,c,False)
    assert not (val&test).any();np.testing.assert_array_equal(val|test,entry_holdout(r,c))
    a,m=normalized_input(sparse.csr_matrix([[0.,0.],[1.,2.]]))
    assert m[0]==0 and np.isfinite(a.data).all()

def test_constant_simplex_and_fragmented_graph_are_reported():
    z=np.full((50,4),.25);d=latent_audit(z)
    assert d['near_constant_fraction']==1
    assert abs(d['effective_active_programs']-4)<1e-10
    g=sparse.csr_matrix([[0,1,0],[1,0,0],[0,0,0]])
    d=graph_audit(g,np.array([0,0,1]))
    assert d['graph_connected_components']==2 and d['fraction_cells_niches_lt20']==1
    assert d['min_niche_size']==1
