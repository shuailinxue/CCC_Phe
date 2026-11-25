"""Fixed-rank PCA control, separate from V1 GNN."""
import torch
from torch import nn


class JointPCABaseline(nn.Module):
    """Fixed-rank randomized PCA plus a diagnostic linear ridge decoder."""
    def __init__(self,center,components,decoder,scale,mask,types):
        super().__init__()
        for key,value in dict(center=center,components=components,decoder=decoder,
                              feature_scale=scale,feature_mask=mask).items():
            self.register_buffer(key,torch.as_tensor(value))
        self.types=int(types)

    def encode(self,metric_input):return (metric_input-self.center)@self.components

    def forward(self,metric_input):
        h=self.encode(metric_input)
        y=torch.cat((h,torch.ones((len(h),1),device=h.device)),1)@self.decoder
        f=len(self.feature_scale)
        p=y[:,f:].clamp_min(1e-8);p=p/p.sum(1,keepdim=True)
        return dict(shared=h,reconstruction=y[:,:f].clamp_min(0)*self.feature_scale*self.feature_mask,
                    composition=p)
