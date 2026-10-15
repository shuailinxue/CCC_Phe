"""Native spatial baselines: dimensions, gradients and training-fold isolation."""
import json
import joblib
import numpy as np
import scipy.sparse as sp
import torch
from baselines.adapters import clean_input,graph_from_members
from baselines.banksy_adapter import banksy
from baselines.stagate_adapter import stagate
from baselines.vendor.banksy.main import concatenate_all
from baselines.vendor.stagate.STAGATE import STAGATE


def config():
    return {'device':'cpu','seed':17,'cpu_threads':2,'pca_components':4,
            'banksy_lambda':.8,'banksy_k_geom':3,'banksy_weight_decay':'scaled_gaussian',
            'stagate_hidden_dims':[8,4],'stagate_epochs':2,'stagate_lr':.001,
            'stagate_gradient_clipping':5.,'stagate_weight_decay':.0001}


def fixture():
    rng=np.random.default_rng(17);xy=rng.normal(size=(40,2));x=rng.poisson(2.,(40,12))
    a=clean_input(x,[f'g{i}' for i in range(12)],xy,['sample']*40)
    return a,np.arange(20),np.repeat([0,2],20)


def test_banksy_native_scaling_uses_training_rows():
    rng=np.random.default_rng(4);blocks=[rng.normal(size=(12,5)) for _ in range(3)];train=np.arange(6)
    actual=concatenate_all(blocks,.8,reference_rows=train)
    expected=np.column_stack([(v-v[train].mean(0))/v[train].std(0)*w for v,w in zip(blocks,np.sqrt([.2,.8*2/3,.8/3]))])
    np.testing.assert_allclose(actual,expected,atol=1e-7)


def test_banksy_zero_training_variance_does_not_explode_on_test():
    blocks=[np.r_[np.zeros((4,2)),np.ones((4,2))*100] for _ in range(3)]
    matrix=concatenate_all(blocks,.8,reference_rows=np.arange(4))
    assert np.isfinite(matrix).all() and not matrix.any()


def test_banksy_no_heldout_expression_in_fitted_features(tmp_path):
    a,train,owner=fixture();cfg=config();first=tmp_path/'one';second=tmp_path/'two';first.mkdir();second.mkdir()
    z,t=banksy(a,train,owner,cfg,first)
    changed=a.copy();changed.X=changed.X.toarray();changed.X[20:]=np.arange(240).reshape(20,12)+1;changed.X=sp.csr_matrix(changed.X)
    zz,tt=banksy(changed,train,owner,cfg,second)
    assert z.shape==(40,4) and np.isfinite(z).all()
    np.testing.assert_allclose(t,tt,atol=1e-6)
    np.testing.assert_allclose(joblib.load(first/'pca.joblib').components_,joblib.load(second/'pca.joblib').components_,atol=1e-6)


def test_native_stagate_gradients_and_frozen_inference(tmp_path):
    torch.set_num_threads(2);torch.manual_seed(17)
    model=STAGATE([12,8,4]);x=torch.rand(6,12);e=torch.cartesian_prod(torch.arange(6),torch.arange(6)).T
    z,y=model(x,e);loss=(x-y).square().mean();loss.backward()
    assert z.shape==(6,4) and torch.isfinite(loss)
    assert any(p.grad is not None and p.grad.abs().sum()>0 for p in model.parameters())
    a,train,owner=fixture();members=np.column_stack((np.arange(40),np.roll(np.arange(40),1)))
    first=tmp_path/'one';second=tmp_path/'two';first.mkdir();second.mkdir();cfg=config()
    z,t=stagate(a,graph_from_members(members),train,cfg,first)
    changed=a.copy();changed.X=changed.X.toarray();changed.X[20:]+=50;changed.X=sp.csr_matrix(changed.X)
    zz,tt=stagate(changed,graph_from_members(members),train,cfg,second)
    old=torch.load(first/'checkpoint.pt',weights_only=False);new=torch.load(second/'checkpoint.pt',weights_only=False)
    for key in old['model_state']:torch.testing.assert_close(old['model_state'][key],new['model_state'][key])
    np.testing.assert_allclose(t,tt)
    assert z.shape==(40,4) and np.isfinite(z).all()


def test_reuse_cache_refuses_changed_data_and_reuses_matching_method(tmp_path):
    import pytest
    from baselines.cache import reuse_completed
    old=tmp_path/'old';new=tmp_path/'new';old.mkdir();new.mkdir();method=old/'graphst';method.mkdir()
    manifest={'inputs':{'counts':'same'},'versions':{'torch':'test'},
              'config':{'device':'cpu','seed':17,'fixed_k':4,'cpu_threads':2,'graphst_epochs':2,'graphst_hidden':8},
              'implementation':{'adapters.py':'identical'}}
    (old/'config.json').write_text(json.dumps(manifest));(old/'complete.json').write_text('{}');(method/'complete.json').write_text('{}')
    np.save(method/'embedding.npy',np.ones((4,2)));np.save(method/'labels.npy',np.arange(4))
    assert reuse_completed(old,new,manifest)==['graphst']
    assert (new/'graphst/reused_from.json').exists()
    with pytest.raises(ValueError,match='different expression'):
        reuse_completed(old,new,dict(manifest,inputs={'counts':'changed'}))
