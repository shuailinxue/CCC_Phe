"""Type-specific STCase-style sparse training and same-cell evaluation.

Computational adapter: sparse LR matrices, fixed six-neighbor graph, and
resolution=1 preclustering. See metadata; this is not the untouched pipeline.
"""
import json,time,sys
from pathlib import Path
import numpy as np,pandas as pd,anndata as ad,scanpy as sc,scipy.sparse as sp,torch
import torch.nn as nn,torch.nn.functional as F
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score

root=Path('/data1/xueshuailin/CCC_Phe/workspace');out=root/'outputs/tumor_immune_interface/v2_baseline_k5/stcase_fair';data=root/'data/tumor_immune_interface_v2_epithelial_majority';v2=root/'outputs/tumor_immune_interface/v2_single_tissue_epithelial_majority'
torch.set_num_threads(4);torch.manual_seed(0);np.random.seed(0);device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if not (out/'ccc_complete.json').exists():raise RuntimeError('CCC background calculations have not finished')
a=ad.read_h5ad(data/'tissue.h5ad');ct=a.obs.cell_type.astype(str).to_numpy();xy=np.asarray(a.obsm['spatial']);raw=a.copy();sc.pp.highly_variable_genes(raw,flavor='seurat_v3',n_top_genes=3000);sc.pp.normalize_total(raw,target_sum=1e4);sc.pp.log1p(raw);sc.pp.scale(raw,zero_center=False,max_value=10);X=raw[:,raw.var.highly_variable].X;X=X.toarray() if sp.issparse(X) else np.asarray(X);X=np.asarray(X,dtype='float32');x=torch.as_tensor(X,device=device)
n=len(ct);near=NearestNeighbors(n_neighbors=7).fit(xy).kneighbors(return_distance=False)[:,:6];src=np.repeat(np.arange(n),6);dst=near.ravel();adj=sp.csr_matrix((np.ones(len(src),dtype=np.float32),(src,dst)),shape=(n,n));adj=adj.maximum(adj.T).tocsr();df=pd.read_csv(out/'panel_covered_official_lr.csv')
cache=out/'ccc_sparse';metrics=[];result={};start=time.time()

class Encoder(nn.Module):
 def __init__(self,k):
  super().__init__();self.logits=nn.Parameter(torch.zeros(k,device=device));self.proj=nn.Linear(2*X.shape[1],10,bias=False,device=device);self.bn=nn.BatchNorm1d(10,device=device)
 def forward(self,mats):
  p=F.linear(x,self.proj.weight[:,:X.shape[1]]);q=F.linear(x,self.proj.weight[:,X.shape[1]:]);l=torch.softmax(self.logits,0);m=torch.zeros_like(q)
  for i,A in enumerate(mats):m=m+l[i]*torch.sparse.mm(A,q)
  return self.bn(p+m)

for name in sorted(set(ct)):
 ids=np.flatnonzero(ct==name);target=np.zeros(n,bool);target[ids]=True;sub=out/('type_'+name.replace(' ','_'));sub.mkdir(exist_ok=True)
 if (sub/'labels.npy').exists():print('reuse',name,flush=True);continue
 coo=adj.tocoo();ok=target[coo.row]|target[coo.col];type_adj=sp.csr_matrix((coo.data[ok],(coo.row[ok],coo.col[ok])),shape=(n,n));coo=type_adj.tocoo();ps=torch.as_tensor(coo.row,device=device);pdst=torch.as_tensor(coo.col,device=device)
 # Upstream type-aware graph: expression PCA / neighbors / Louvain, fixed
 # resolution is the public optional parameter (avoids 100k-point search).
 mini=ad.AnnData(X[ids]);sc.tl.pca(mini,svd_solver='arpack',random_state=0);sc.pp.neighbors(mini,random_state=0);sc.tl.louvain(mini,resolution=1.,random_state=0);groups=np.full(n,-1,dtype=np.int32);groups[ids]=pd.Categorical(mini.obs.louvain).codes;prune=(groups[coo.row]==groups[coo.col]);qs=torch.as_tensor(coo.row[prune],device=device);qd=torch.as_tensor(coo.col[prune],device=device)
 mats=[];lr_ids=[]
 for j in range(len(df)):
  mat=sp.load_npz(cache/f'{j:04d}.npz').tocsr()
  if mat.nnz==0 or mat.multiply(type_adj).nnz==0:continue
  C=mat.tocoo();v=C.data.copy();swap=target[C.row]&~target[C.col]
  if swap.any():v[swap]=np.asarray(mat[C.col[swap],C.row[swap]]).ravel()
  nz=v!=0;indices=torch.as_tensor(np.vstack((C.col[nz],C.row[nz])),device=device,dtype=torch.long);values=torch.as_tensor(v[nz],device=device,dtype=torch.float32);mats.append(torch.sparse_coo_tensor(indices,values,(n,n),device=device).coalesce());lr_ids.append(j)
 if not mats:print('no usable LR',name,flush=True);continue
 print('train',name,'cells',len(ids),'LR',len(mats),'edges',len(ps),'type edges',len(qs),flush=True)
 model=Encoder(len(mats));opt=torch.optim.Adam(model.parameters(),lr=.001);best=float('inf');beststate=None
 for epoch in range(100):
  model.train();z=model(mats);loss=.75*3*F.softplus(-(z[ps]*z[pdst]).sum(1)).mean()+.25*3*F.softplus(-(z[qs]*z[qd]).sum(1)).mean();opt.zero_grad();loss.backward();opt.step();val=float(loss.detach())
  if val<best:best=val;beststate={k:v.detach().clone() for k,v in model.state_dict().items()}
  if (epoch+1)%10==0:print(name,'epoch',epoch+1,'loss',round(val,5),flush=True)
 model.load_state_dict(beststate);model.eval()
 with torch.no_grad():emb=model(mats)[ids].cpu().numpy()
 # Upstream searches Leiden resolution for a fixed number of subtypes.
 b=ad.AnnData(emb);sc.pp.neighbors(b,n_neighbors=20,n_pcs=0,random_state=0);chosen=None
 for resolution in np.arange(4.98,.0,-.02):
  sc.tl.leiden(b,resolution=float(resolution),random_state=0)
  labs=pd.Categorical(b.obs.leiden).codes
  if len(np.unique(labs))==5:chosen=float(resolution);break
 if chosen is None:raise RuntimeError(f'{name}: native Leiden scan did not yield exactly five clusters')
 np.save(sub/'ids.npy',ids);np.save(sub/'labels.npy',labs);np.save(sub/'embedding.npy',emb);(sub/'fit.json').write_text(json.dumps({'lr_count':len(mats),'epochs':100,'alpha':.25,'hidden':10,'leiden_resolution':chosen,'minimum_loss':best,'target_type':name},indent=2));del mats,model,opt;torch.cuda.empty_cache()

# Ground truth enters only after all type models are trained.
folder=Path(json.loads((data/'target_cache.json').read_text())['folder']);composition=np.load(folder/'composition.npy');coarse=KMeans(n_clusters=4,n_init=10,random_state=40700).fit_predict(np.sqrt(np.maximum(composition,0)));means=np.stack([composition[coarse==k].mean(0) for k in range(4)]);choice=np.flatnonzero((means[:,0]>.3)&(means[:,1]>.1)&(means[:,3]>.10));saved=np.load(v2/'single_tissue_predictions.npz');assert len(choice)==1 and np.array_equal(saved['ids'],np.flatnonzero(coarse==choice[0]));fine=coarse.copy();fine[saved['ids']]=4+saved['network'];truth=np.array([0,1,2,3,4,4])[a.obs.design_region_id.to_numpy()]
for name in sorted(set(ct)):
 sub=out/('type_'+name.replace(' ','_'));ids=np.load(sub/'ids.npy');labels=np.load(sub/'labels.npy');counts=np.bincount(truth[ids],minlength=5);metrics.append({'cell_type':name,'n_cells':len(ids),'ground_truth_classes':int((counts>0).sum()),'STCase_ARI':adjusted_rand_score(truth[ids],labels),'our_method_ARI':adjusted_rand_score(truth[ids],fine[ids]),'K_STCase':len(np.unique(labels))})
pd.DataFrame(metrics).to_csv(out/'per_cell_type_metrics.csv',index=False);(out/'training_complete.json').write_text(json.dumps({'method':'type-specific sparse STCase adaptation','same_cell_scope':True,'truth_used_in_fit':False,'normalization':'CP10k log1p; official model HVG scaling','computational_adaptations':['sparse LR matrices','symmetric six-nearest graph','precluster resolution=1'],'seconds':time.time()-start},indent=2));print(pd.DataFrame(metrics).to_string(index=False),flush=True)
