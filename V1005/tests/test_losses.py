import torch
from phenoniche.v1005.losses import nonconvex_lowrank,geometry_loss,group_sparsity

def test_loss_values_and_gradients():
    z=torch.tensor([[1.,2.],[3.,5.],[2.,1.]],requires_grad=True)
    p=nonconvex_lowrank(z)
    torch.testing.assert_close(p,torch.log1p(torch.linalg.svdvals(z)/.001).sum())
    p.backward();assert torch.isfinite(z.grad).all()
    assert geometry_loss(z,z,torch.ones(3))==0
    h=torch.tensor([[3.,4.,0.,12.]],requires_grad=True)
    g=group_sparsity(h,torch.tensor([0,0,1,1]),2)
    torch.testing.assert_close(g,torch.tensor(17.),atol=3e-6,rtol=1e-6)
    g.backward();assert torch.isfinite(h.grad).all()
