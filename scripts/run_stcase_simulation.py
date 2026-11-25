"""Whole-tissue, sparse STCase adaptation. See result metadata for deviations."""
import json,time
from pathlib import Path
import numpy as np,pandas as pd,anndata as ad,scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score

torch.set_num_threads(4);torch.manual_seed(2025);np.random.seed(2025)
root=Path('/data1/xueshuailin/CCC_Phe/workspace');out=root/'outputs/tumor_immune_interface/v2_baseline_k5/stcase';out.mkdir(parents=True,exist_ok=True)
a=ad.read_h5ad(root/'data/tumor_immune_interface_v2_epithelial_majority/tissue.h5ad')
x=a.X.toarray() if sp.issparse(a.X) else np.asarray(a.X)
x=np.asarray(x,dtype=np.float32);x=np.log1p(x/(x.sum(1,keepdims=True)+1e-8)*1e4)
prior=pd.read_csv('/home/xueshuailin/CCC_Phe/data/commuspace_human_lr_atlas.tsv',sep='\t');prior=prior.loc[prior.ligand.isin(a.var_names)&prior.receptor.isin(a.var_names),['ligand','receptor']].drop_duplicates()
xyz=np.asarray(a.obsm['spatial']);neighbors=NearestNeighbors(n_neighbors=31).fit(xyz).kneighbors(return_distance=False)[:,:30]
src=np.repeat(np.arange(len(x)),30);dst=neighbors.ravel();gene={g:i for i,g in enumerate(a.var_names)}
li=np.array([gene[g] for g in prior.ligand]);ri=np.array([gene[g] for g in prior.receptor]);
# STCase supports ranking LR by edge-signal variation and setting lr_cut. Estimate
# on a fixed random sample of physical edges, without simulation labels.
sample=np.random.default_rng(2025).choice(len(src),min(50000,len(src)),replace=False)
variance=np.var(x[src[sample,None],li]*x[dst[sample,None],ri],axis=0)
take=np.argsort(-variance)[:min(256,len(li))];li,ri=li[take],ri[take]
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
xt=torch.as_tensor(x,device=device);s=torch.as_tensor(src,device=device);d=torch.as_tensor(dst,device=device)
print('cells',len(x),'edges',len(src),'panel LR',len(prior),'selected LR',len(li),'device',device,flush=True)
# Per-edge, per-LR values replace upstream dense N-by-N LR matrices. Product
# weights are the declared adapter because this simulation has no LR_cell_weight.
edge=(xt[s[:,None],torch.as_tensor(li,device=device)]*xt[d[:,None],torch.as_tensor(ri,device=device)]).half()
class STCaseAllCells(nn.Module):
 def __init__(self):
  super().__init__();self.lr_logits=nn.Parameter(torch.zeros(len(li)));self.proj=nn.Linear(2*x.shape[1],64,bias=False);self.bn=nn.BatchNorm1d(64)
 def forward(self):
  w=torch.softmax(self.lr_logits,0);weights=(edge.float()@w);proj0=F.linear(xt,self.proj.weight[:,:x.shape[1]]);proj1=F.linear(xt,self.proj.weight[:,x.shape[1]:]);agg=torch.zeros_like(proj1);agg.index_add_(0,d,proj1[s]*weights[:,None]);return self.bn(proj0+agg)
model=STCaseAllCells().to(device);opt=torch.optim.Adam(model.parameters(),lr=.001);start=time.time()
for epoch in range(100):
 model.train();z=model();loss=3*F.softplus(-(z[s]*z[d]).sum(1)).mean();opt.zero_grad();loss.backward();opt.step()
 if epoch%10==0 or epoch==99:print('epoch',epoch+1,'loss',float(loss.detach()),flush=True)
model.eval()
with torch.no_grad():emb=model().cpu().numpy()
np.save(out/'embedding.npy',emb);labels=KMeans(n_clusters=5,n_init=10,random_state=40700).fit_predict(emb);np.save(out/'labels.npy',labels)
truth=np.array([0,1,2,3,4,4])[a.obs.design_region_id.to_numpy()]
info={'method':'STCase whole-tissue sparse adaptation','source_commit':'33f62709da8265c1abc774cc3b0624f3c6c9d12a','lr_complete':len(prior),'lr_selected_unsupervised':len(li),'epochs':100,'adaptations':['single all-cell encoder instead of separate cell-type encoders','edge LR weights from normalized expression products; input does not supply upstream LR_cell_weight','sparse physical graph and LR edge weights','KMeans K=5 instead of per-cell-type Leiden'],'truth_used_in_fit':False,'global_ARI':float(adjusted_rand_score(truth,labels)),'tumor_ARI':float(adjusted_rand_score(truth[truth!=0],labels[truth!=0])),'seconds':time.time()-start}
(out/'complete.json').write_text(json.dumps(info,indent=2));print(json.dumps(info,indent=2),flush=True)
