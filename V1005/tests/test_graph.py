import numpy as np
import pytest
from scipy import sparse
from phenoniche.v1005.graph import fuse_graphs,normalize_edge_mass,fuse_three_graphs
from phenoniche.v1005.clustering import leiden
from phenoniche.v1005.evaluation import match_programs,spatial_coherence

def test_graph_fusion_and_clustering():
    a=sparse.csr_matrix([[0.,2,0,0],[2,0,0,0],[0,0,0,1],[0,0,1,0]])
    b=sparse.csr_matrix(np.ones((4,4))-np.eye(4))
    g=fuse_graphs(a,b,.2)
    np.testing.assert_allclose(g.toarray(),a.toarray()/1.5+.2*b.toarray()/3)
    np.testing.assert_allclose(fuse_graphs(a,b,0).toarray(),a.toarray()/1.5)
    assert len(leiden(g,40700,1))==4
    with pytest.raises(ValueError):fuse_graphs(a,b,1)

def test_program_matching_and_coherence():
    h=np.eye(3);s,i,j=match_programs(h,h[[2,0,1]])
    np.testing.assert_allclose(s[i,j],1)
    np.testing.assert_array_equal(j,[1,2,0])
    assert spatial_coherence(np.array([0,0,1]),np.array([[0,1],[1,0],[2,0]]))==pytest.approx(2/3)

def test_three_graph_fusion_normalizes_each_input_mass():
    a=sparse.csr_matrix([[0,2,0],[2,0,1],[0,1,0]],dtype=float)
    b=a*10;c=a*.1
    for graph in [a,b,c]:assert np.isclose(normalize_edge_mass(graph).sum()/3,1)
    fused=fuse_three_graphs(a,b,c,alpha=.2,beta=.1)
    assert np.isclose(fused.sum()/3,1.3)
