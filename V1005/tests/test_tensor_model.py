import numpy as np
import torch
from scipy import sparse
from phenoniche.v1005.tensor_model import DeepTensorCCC,sparse_tensor

def test_tensor_projection_matches_dense_and_gradients():
    torch.manual_seed(1);modes=np.array(list(np.ndindex(2,2,3)));m=DeepTensorCCC(2,3,modes,K=4)
    x=np.random.default_rng(1).random((5,12)).astype('float32');x[x<.3]=0
    z=m(sparse_tensor(sparse.csr_matrix(x),'cpu'))
    g=torch.einsum('nsrl,sa,rb,lc->nabc',torch.tensor(x.reshape(5,2,2,3)),m.A_s,m.A_r,m.A_l)
    torch.testing.assert_close(z,m.encoder(g.flatten(1)),rtol=1e-5,atol=1e-6)
    assert z.shape==(5,4) and (z>=0).all() and (m.H>=0).all()
    torch.testing.assert_close(m.decode(z),z@m.dense_dictionary().reshape(4,12))
    m.decode(z).sum().backward()
    for p in m.parameters():assert p.grad is not None and torch.isfinite(p.grad).all()

def test_guard_and_direction():
    modes=np.array(list(np.ndindex(2,2,1)));m=DeepTensorCCC(2,1,modes,K=2)
    import pytest
    with pytest.raises(MemoryError):m.dense_dictionary(max_elements=1)
    a=sparse.csr_matrix([[0.,1.,0.,0.]]);b=sparse.csr_matrix([[0.,0.,1.,0.]])
    assert not torch.allclose(m(sparse_tensor(a,'cpu')),m(sparse_tensor(b,'cpu')))

def test_simplex_dictionary_is_used_in_every_decode():
    modes=np.array(list(np.ndindex(2,2,3)));m=DeepTensorCCC(2,3,modes,K=4)
    x=sparse.csr_matrix(np.ones((3,12),dtype=np.float32));z=m(sparse_tensor(x,'cpu'))
    torch.testing.assert_close(z.sum(1),torch.ones(3))
    torch.testing.assert_close(m.H.sum(1),torch.ones(4))
    torch.testing.assert_close(m.decode(z).sum(1),torch.ones(3))
    torch.testing.assert_close(torch.cat([m.decode(z,0,5),m.decode(z,5,12)],dim=1),m.decode(z))
