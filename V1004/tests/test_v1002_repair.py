import numpy as np
import torch
from phenoniche.v1002.repair_primary import audit, passed, evaluate
from phenoniche.v1002.simulation_model import fit_multiview


def test_view_gradient_audit_matches_autograd():
    torch.manual_seed(4)
    w=torch.rand(7,6,dtype=torch.float64,requires_grad=True)
    h={k:torch.rand(6,f,dtype=torch.float64) for k,f in [('HC',8),('HI',21),('HO',24)]}
    x={k:torch.rand(7,v.shape[1],dtype=torch.float64) for k,v in h.items()}
    result=audit(w,h,x)
    for k in h:
        gradient=torch.autograd.grad(((w@h[k]-x[k])**2).mean(),w)[0]
        assert np.isclose(result[k]['gradient_norm'],float(gradient.norm()),rtol=1e-12)


def test_audit_callback_does_not_change_factorization():
    x={'HC':np.ones((12,8),dtype=np.float32),'HI':np.eye(12,dtype=np.float32)}
    checkpoints=[]
    a=fit_multiview(x,niches=6,iterations=10,device='cpu')
    b=fit_multiview(x,niches=6,iterations=10,device='cpu',audit_callback=lambda epoch,w,h:checkpoints.append(epoch))
    assert torch.equal(a.W,b.W)
    assert checkpoints==[0,10]


def test_background_is_a_predicted_factor():
    from types import SimpleNamespace
    labels=np.repeat(np.arange(6),3)
    w=np.eye(6)[labels]
    h={'HC':np.eye(6),'HI':np.eye(6)}
    report=evaluate(w,h,h,SimpleNamespace(labels=labels),w)
    assert report['predicted']==labels.tolist()
    assert passed([report]*3)
    report['per_factor'][0]['sensitivity']=.79
    assert not passed([report]*3)
