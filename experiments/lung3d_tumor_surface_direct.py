"""Direct LR vectors, pooled in 2D or 3D without CP."""
import os,pickle
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse import load_npz
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
from phenoniche.spatial import Settings
from phenoniche.spatial.channels import Channels
from phenoniche.spatial.cell_programs import CellTensor
import lung3d_v2 as study
out=Path('/data1/xueshuailin/CCC_Phe/workspace/outputs/3D_lung_CosMx/v2_bidirectional/tumor_surface_slice2')
a,xyz=study.read_input()
z=np.unique(xyz[:,2])[1]
ids=np.flatnonzero((xyz[:,2]==z)&a.obs.niches_3D.eq('Tumor surface').to_numpy())
graph=load_npz(study.OUT/'contacts.npz')
_,types=np.unique(a.obs.celltypes.astype(str),return_inverse=True)
channels=Channels(a.layers['counts'],a.var_names,pd.read_csv(study.ROOT/'data/commuspace_human_lr_atlas.tsv',sep='\t'))
cfg=Settings();tensor=CellTensor(graph,types,channels,cfg.sigma_um);allowed=np.ones(a.n_obs,dtype=bool)
near=np.unique(np.r_[ids,graph[ids].indices]);print(f'{len(ids)} target; {len(near)} cells; {len(channels.names)} LR pairs',flush=True)
raw=np.zeros((len(near),2*len(channels.names)),dtype=np.float32)
for j,i in enumerate(near):
 raw[j]=tensor.fiber(i,allowed)[2].sum(axis=0)
 if (j+1)%5000==0:print(f'cell vectors {j+1}/{len(near)}',flush=True)
# Sparse physical distance weights, including the center. Same vectors are used for both comparisons.
local=graph[ids][:,near].tocsr();local.data=np.exp(-.5*(local.data/cfg.sigma_um)**2)
pos=np.searchsorted(near,ids)
local=local+sparse.csr_matrix((np.ones(len(ids)),(np.arange(len(ids)),pos)),shape=local.shape)
results={'cell_id':np.asarray(a.obs_names[ids]),'z':z,'lr_count':len(channels.names)}
for mode in ('3D','2D'):
 weights=local.copy()
 if mode=='2D':
  weights=weights.multiply((xyz[near,2]==z)[None,:]).tocsr()
 pooled=weights@raw
 pooled/=np.maximum(np.asarray(weights.sum(axis=1)),1e-12)
 # Compress count scale while preserving outgoing/incoming channel identities.
 pooled=np.log1p(pooled)
 pooled=normalize(pooled,norm='l2',axis=1)
 for k in (2,3,4):
  results[f'{mode}_{k}']=KMeans(n_clusters=k,n_init=5,random_state=73).fit_predict(pooled)
  print(mode,k,np.bincount(results[f'{mode}_{k}']).tolist(),flush=True)
tmp=out/'direct.partial.pkl'
with tmp.open('wb') as f:pickle.dump(results,f,protocol=5)
os.replace(tmp,out/'direct_2d_3d.pkl')
print('Finished',flush=True)
