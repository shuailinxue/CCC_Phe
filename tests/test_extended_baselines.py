"""Fairness contracts for official baseline adapters, without anatomical truth."""
import numpy as np
import scipy.sparse as sp
from baselines.protocol import physical_owners,isolate_graph,panel_lr_prior
from baselines.spaniche_adapter import radius_views


def test_owner_and_views_do_not_cross_folds():
    members=np.array([[0,1],[1,0],[2,3],[3,2],[4,5],[5,4]])
    folds=np.array([0,0,1,1,2,2]);owners=physical_owners(members,folds)
    g=isolate_graph(sp.csr_matrix(np.ones((6,6))),owners)
    row,col=g.nonzero();assert np.all(owners[row]==owners[col])
    xy=np.column_stack((np.arange(6),np.zeros(6)))
    for view in radius_views(xy,owners,3):
        row,col=view.nonzero();assert np.all(owners[row]==owners[col])
        np.testing.assert_allclose(np.asarray(view.sum(1)).ravel(),1.)


def test_public_LR_projection_requires_only_gene_panel():
    table=panel_lr_prior(['A2M','LRP1','SYN_HOUSE_1'])
    assert list(table.ligand)==['A2M'] and list(table.receptor)==['LRP1']
    assert 'mi' not in table and 'loading' not in table and 'cell_type' not in table


def test_spaniche_native_annulus_includes_center():
    near,far,annulus=radius_views(np.array([[0.,0],[1,0],[2.5,0]]),np.zeros(3),1.5)
    assert near[0,1]>0 and near[0,2]==0
    assert far[0,2]>0 and annulus[0,2]>0
    assert annulus[0,1]==0 and annulus[0,0]>0
