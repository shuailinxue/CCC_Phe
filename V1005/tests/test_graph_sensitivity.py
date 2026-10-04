import numpy as np
from scipy import sparse
from phenoniche.v1005.graph_sensitivity import graph_overlap,reweight,edge_similarity

def test_graph_overlap_partitions_union_edges():
    a=sparse.csr_matrix([[0,1,1],[1,0,0],[1,0,0]],dtype=float)
    b=sparse.csr_matrix([[0,1,0],[1,0,1],[0,1,0]],dtype=float)
    c=sparse.csr_matrix([[0,0,1],[0,0,1],[1,1,0]],dtype=float)
    pair,categories=graph_overlap([a,b,c]);row=categories.iloc[0]
    assert len(pair)==3 and row.union_edges==3
    assert np.isclose(row.shared_edge_fraction,1)
    assert np.isclose(row.only_ccc_edge_fraction+row.only_comp_edge_fraction+row.only_spatial_edge_fraction,0)

def test_reweight_preserves_edge_support():
    base=sparse.csr_matrix([[0,1,1],[1,0,0],[1,0,0]],dtype=float)
    reference=sparse.csr_matrix([[0,1,0],[1,0,1],[0,1,0]],dtype=float)
    embedding=np.array([[0.,0.],[1,0],[3,0]],dtype=np.float32)
    similarity,scale=edge_similarity(base,embedding,reference,row_chunk=2)
    result=reweight(base,[similarity],[.1])
    np.testing.assert_array_equal(result.indices,base.indices)
    np.testing.assert_array_equal(result.indptr,base.indptr)
    assert scale>0 and np.isclose(result.sum()/len(embedding),1)
