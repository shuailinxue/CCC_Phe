"""Mode projections on a sparse directed tensor, no raw-tensor flatten MLP."""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

class DeepTensorCCC(nn.Module):
    def __init__(self,C,L,feature_modes,K=16,r_l=32):
        super().__init__();self.C=C;self.L=L;self.K=K
        s,r,l=np.asarray(feature_modes).T;self.register_buffer('s',torch.tensor(s,dtype=torch.long));self.register_buffer('r',torch.tensor(r,dtype=torch.long));self.register_buffer('l',torch.tensor(l,dtype=torch.long))
        self.A_s=nn.Parameter(torch.randn(C,min(C,8))/C**.5)
        self.A_r=nn.Parameter(torch.randn(C,min(C,8))/C**.5)
        self.A_l=nn.Parameter(torch.randn(L,r_l)/max(L,1)**.5)
        dim=min(C,8)**2*r_l
        self.encoder=nn.Sequential(nn.LayerNorm(dim),nn.Linear(dim,64),nn.GELU(),nn.Linear(64,K),nn.Softmax(dim=-1))
        self.h_raw=nn.Parameter(torch.full((K,len(s)),-4.)+torch.randn(K,len(s))*.1)
    @property
    def H(self):
        h=F.softplus(self.h_raw)
        return h/h.sum(dim=1,keepdim=True)
    def encode_sparse(self,x):
        x=x.coalesce();rows,cols=x.indices();pair=self.s[cols]*self.C+self.r[cols]
        unfolded=torch.sparse_coo_tensor(torch.stack([rows*self.C**2+pair,self.l[cols]]),x.values(),
                     (x.shape[0]*self.C**2,self.L),device=x.device).coalesce()
        g=torch.sparse.mm(unfolded,self.A_l).reshape(x.shape[0],self.C,self.C,-1)
        g=torch.einsum('nsrl,sa,rb->nabl',g,self.A_s,self.A_r)
        return self.encoder(g.flatten(1))
    def forward(self,x):return self.encode_sparse(x)
    def decode(self,z,start=0,stop=None):return z @ self.H[:,start:stop]
    def dense_dictionary(self,max_elements=20_000_000):
        if self.K*self.C*self.C*self.L>max_elements:raise MemoryError('Guarded dictionary materialization')
        h=torch.zeros((self.K,self.C,self.C,self.L),device=self.h_raw.device)
        h[:,self.s,self.r,self.l]=self.H
        return h

def sparse_tensor(x,device):
    c=x.tocoo()
    return torch.sparse_coo_tensor(np.stack([c.row,c.col]),torch.as_tensor(c.data,dtype=torch.float32),c.shape,device=device).coalesce()
