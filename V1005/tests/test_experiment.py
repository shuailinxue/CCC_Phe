import numpy as np
import pandas as pd
from scipy import sparse
from phenoniche.v1005.experiment import train_model,DEFAULT,heldout_error,randomized_pca
from phenoniche.v1005.ccc_tensor import training_only

def test_small_training_both_ablation_and_prior(tmp_path):
    rng=np.random.default_rng(9);x=sparse.csr_matrix(rng.random((20,12)).astype('float32'));train=training_only(x)
    context={'features':pd.DataFrame(list(np.ndindex(2,2,3)),columns=['sender_id','receiver_id','lr_index']),'shape':(2,2,3)}
    g=sparse.csr_matrix(np.ones((20,20),np.float32)-np.eye(20,dtype='float32'))
    config={**DEFAULT,'device':'cpu','K':3,'warmup_epochs':1,'graph_epochs':2,'graph_ramp_epochs':2,'patience':2,'batch_size':8,'feature_chunk':5}
    for prior in [False,True]:
        m,z,h=train_model(x,train,context,g,config,40700,prior,tmp_path)
        assert z.shape==(20,3) and h.shape==(3,12)
        assert np.isfinite(z).all() and (z>=0).all() and (h>=0).all()
        np.testing.assert_allclose(z.sum(1),1,atol=1e-6)
        np.testing.assert_allclose(h.sum(1),1,atol=1e-6)
        _,zr,hr=train_model(x,train,context,g,config,40700,prior,tmp_path)
        np.testing.assert_array_equal(z,zr);np.testing.assert_array_equal(h,hr)
    z,v,mean=randomized_pca(train,k=3)
    err=heldout_error(x,train,lambda a:np.asarray(a@v.T)-mean@v.T,lambda z,l,r:z@v[:,l:r]+mean[l:r],chunk=5)
    assert err['heldout_entries']>0 and np.isfinite(err['heldout_MSE'])

def test_output_niche_ids_and_type_labels_are_consistent(tmp_path,monkeypatch):
    import json
    from phenoniche.v1005 import data
    from phenoniche.v1005.evaluation import finalize_table_labels
    monkeypatch.setattr(data,'ROOT',tmp_path);folder=tmp_path/'outputs/hbc1';folder.mkdir(parents=True)
    (folder/'tensor_audit.json').write_text(json.dumps({'types':['A','B']}))
    pd.DataFrame({'niche':[0,1],'0':[.8,.2],'1':[.2,.8]}).to_csv(folder/'niche_composition.csv',index=False)
    pd.DataFrame({'niche':[0,1],'0':[1.,2.],'1':[2.,1.]}).to_csv(folder/'niche_program_activity.csv',index=False)
    pd.DataFrame({'niche':[1,2],'weight':[1.,1.]}).to_csv(folder/'niche_top_edges.csv',index=False)
    finalize_table_labels('hbc1');finalize_table_labels('hbc1')
    assert list(pd.read_csv(folder/'niche_composition.csv').columns)==['niche','A','B']
    assert list(pd.read_csv(folder/'niche_program_activity.csv').columns)==['niche','Program 1','Program 2']
    assert pd.read_csv(folder/'niche_top_edges.csv').niche.tolist()==[0,1]

def test_constant_latent_diagnostic_does_not_invent_variance(tmp_path,monkeypatch):
    from phenoniche.v1005 import data
    from phenoniche.v1005.evaluation import representation_diagnostics
    monkeypatch.setattr(data,'ROOT',tmp_path);folder=tmp_path/'outputs/hbc1';folder.mkdir(parents=True)
    np.save(folder/'seed_40700_tensor_H.npy',np.eye(2,dtype=np.float32))
    np.save(folder/'seed_40700_tensor_Z.npy',np.full((1000,2),1e-8,dtype=np.float32))
    result=representation_diagnostics('hbc1').iloc[0]
    assert result.maximum_latent_column_std==0
    assert result.centered_Z_effective_rank==0
    assert np.isnan(result.centered_Z_first_component_variance_fraction)
