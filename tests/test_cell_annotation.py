"""Reference transfer must preserve missingness, ambiguity and held-out evidence."""
import numpy as np
import pytest
from phenoniche.cell_annotation import SingleRAnnotator, marker_ranks, validate_matrix


def test_inferred_positive_values_are_not_used_as_marker_detection():
    genes=['CD79A','CD22','EPCAM','KRT8']
    x=np.array([[1,2,10,20],[10,20,1,2]],dtype=float)
    scores=marker_ranks(x,genes,['B_cells','Cancer_epithelial'])
    assert (x>0).all()
    assert scores.B_cells.iloc[0]<scores.B_cells.iloc[1]
    assert scores.Cancer_epithelial.iloc[0]>scores.Cancer_epithelial.iloc[1]


def test_single_marker_cannot_support_lineage():
    result=marker_ranks(np.array([[10,1]]),['CD79A','EPCAM'],['B_cells'])
    assert np.isnan(result.B_cells.iloc[0])
    with pytest.raises(ValueError):validate_matrix(np.ones((1,2)),['CD79A','CD79A'])


def test_official_singler_transfer_and_rejection_with_disjoint_cells():
    pytest.importorskip('singler')
    rng=np.random.default_rng(7)
    genes=['CD79A','CD22','MS4A1','PECAM1','VWF','KDR','X','Y']
    means=np.array([[8,7,9,.1,.1,.1,1,2],[.1,.1,.1,8,7,9,1,2]])
    labels=np.repeat(['B_cells','Endothelial'],40)
    x=np.vstack([rng.lognormal(np.log(m),.15,size=(40,8)) for m in means])
    train=np.r_[0:20,40:60];cal=np.r_[20:30,60:70];test=np.r_[30:40,70:80]
    model=SingleRAnnotator(x[train],labels[train],genes,threads=1)
    assert (model.predict(x[test]).singleR_label.to_numpy()==labels[test]).all()
    model.calibrate(x[cal],labels[cal])
    assert model.predict(x[test]).cell_type.ne('Unknown').sum()>=15
    assert model.predict(np.zeros((1,8))).cell_type.iloc[0]=='Unknown'
