"""Physical-fold isolation and immutable clustering-only transformations."""
import numpy as np
from phenoniche.gnn.spatial_refinement import physical_fold_weights,spatial_cluster_features


def test_pooling_does_not_use_cross_fold_donors_or_modify_Z():
    members=np.array([[0,1,2],[0,1,2],[1,2,3],[2,3,4],[4,5,6],[4,5,6],[5,6,7],[5,6,7]])
    rows=[np.array([0]),np.array([],dtype=int),np.array([4])]
    weights=np.ones(members.shape);x=np.arange(16,dtype=float).reshape(8,2);before=x.copy()
    w,valid=physical_fold_weights(members,weights,rows)
    assert w[0,0]==0 and w[0,1]==1 and w[0,2]==0
    assert not valid[4,0] and valid[4,1] and not valid[4,2]
    pooled=spatial_cluster_features(x,members,weights,rows,min_donors=1)
    np.testing.assert_allclose(pooled[0],.8*x[0]+.2*x[1])
    np.testing.assert_array_equal(x,before)
    altered=x.copy();altered[2:]+=10000
    other=spatial_cluster_features(altered,members,weights,rows,min_donors=1)
    np.testing.assert_array_equal(pooled[0],other[0])
    np.testing.assert_array_equal(spatial_cluster_features(x,members,weights,rows,mixing=0),x)
