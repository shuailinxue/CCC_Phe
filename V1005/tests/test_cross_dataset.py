import numpy as np
import pandas as pd
from phenoniche.v1005 import cross_dataset

def test_matching_uses_exact_shared_direction_and_reports_coverage(tmp_path,monkeypatch):
    monkeypatch.setattr(cross_dataset,'ROOT',tmp_path)
    a=tmp_path/'outputs/hbc1';b=tmp_path/'outputs/xenium5k';a.mkdir(parents=True);b.mkdir(parents=True)
    pd.DataFrame({'sender':['A','B','A'],'receiver':['B','A','A'],'ligand':['L1','L2','L3'],'receptor':['R1','R2','R3']}).to_csv(a/'features.csv',index=False)
    pd.DataFrame({'sender':['B','A','B'],'receiver':['A','B','B'],'ligand':['L2','L1','L4'],'receptor':['R2','R1','R4']}).to_csv(b/'features.csv',index=False)
    np.save(a/'seed_40700_tensor_H.npy',np.array([[1.,0.,9.],[0.,1.,9.]]))
    np.save(b/'seed_40700_tensor_H.npy',np.array([[1.,0.,2.],[0.,1.,2.]]))
    t=cross_dataset.match_datasets();assert (t.n_shared_retained_CCC==2).all()
    np.testing.assert_allclose(t.cosine_similarity,1)
    assert len(pd.read_csv(tmp_path/'outputs/cross_dataset/program_coverage.csv'))==4
