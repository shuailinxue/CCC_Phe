import numpy as np
import torch

from phenoniche.v1.network_workflow import explain
from phenoniche.v1.probabilistic import Config, Dictionary, Niches, batch_data, infer_activity, pooled
from phenoniche.v1.refit import spatial_split
from phenoniche.v1.simulated import simulate


def test_support_mask_and_exact_score(tmp_path):
    d=simulate(tmp_path,"sparse")
    f,v=spatial_split(d)
    assert (d.xyz[v,0].min()-d.xyz[f,0].max())>100
    assert not np.any((f|v)&d.test)
    cfg=Config(rank=2,k=2)
    m=Dictionary(*pooled(d,np.flatnonzero(f),"cpu"),cfg.rank)
    b=batch_data(d,np.array([0,1]),"cpu")
    a=torch.tensor([[.5,.5],[1.5,1.5]])
    h,_=m.predict(a)
    assert (h[0]<m.base[0]).all() and (h[1]>m.base[0]).all()
    mask=torch.arange(b.s.shape[1])%2==0
    weights=m.features(b,mask)
    for x in weights:
        assert torch.count_nonzero(x[:,~mask])==0
    scores=m.score(weights,a)
    losses=m.local_loss(weights,a,m.weights())
    for score,loss in zip(scores,losses,strict=True):
        torch.testing.assert_close(score.diag(),-loss,atol=1e-5,rtol=1e-5)
    u=infer_activity(m,weights,cfg,steps=80)
    assert (u>=0).all()
    before=sum(m.local_loss(weights,torch.ones_like(u),m.weights()))
    after=sum(m.local_loss(weights,u,m.weights()))+.5*cfg.local_ridge*(u-1).square().sum(1)
    assert torch.all(after <= before+1e-4)


def test_attribution_matches_total_margin(tmp_path):
    d=simulate(tmp_path,"both")
    m=Dictionary(*pooled(d,np.flatnonzero(d.train),"cpu"),2)
    n=Niches(2,d.nt,2,"joint",.5)
    with torch.no_grad():
        n.activity.copy_(torch.tensor([[0.,.5],[2.,2.]]))
        n.pi.copy_(torch.tensor([[0.,1.],[1.,0.]]))
    info,table=explain(d,0,m,n,np.ones(2,dtype=np.float32),"cpu")
    np.testing.assert_allclose(table.occurrence_margin.sum()+table.intensity_margin.sum(),
                               info["communication_margin"],atol=1e-5)
    np.testing.assert_allclose(info["composition_margin"]+info["prior_margin"]+info["communication_margin"],
                               info["total_margin"],atol=1e-5)
