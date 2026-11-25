"""Class-averaged direct LR signal by ordered cell types."""
import os,pickle
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse import load_npz
from phenoniche.spatial import Settings
from phenoniche.spatial.channels import Channels
from phenoniche.spatial.cell_programs import CellTensor
import lung3d_v2 as study
out=Path('/data1/xueshuailin/CCC_Phe/workspace/outputs/3D_lung_CosMx/v2_bidirectional/tumor_surface_slice2')
a,xyz=study.read_input()
with (out/'direct_2d_3d.pkl').open('rb') as f:r=pickle.load(f)
ids=a.obs_names.get_indexer(r['cell_id']); groups=r['3D_3']; graph=load_npz(study.OUT/'contacts.npz')
names,types=np.unique(a.obs.celltypes.astype(str),return_inverse=True)
channels=Channels(a.layers['counts'],a.var_names,pd.read_csv(study.ROOT/'data/commuspace_human_lr_atlas.tsv',sep='\t'))
cfg=Settings();tensor=CellTensor(graph,types,channels,cfg.sigma_um);allowed=np.ones(a.n_obs,dtype=bool)
w=graph[ids].copy();w.data=np.exp(-.5*(w.data/cfg.sigma_um)**2)
w+=sparse.csr_matrix((np.ones(len(ids)),(np.arange(len(ids)),ids)),shape=w.shape)
w=sparse.diags(1/np.maximum(np.asarray(w.sum(axis=1)).ravel(),1e-12))@w
weights=np.vstack([np.asarray(w[groups==k].mean(axis=0)).ravel() for k in range(3)])
near=np.flatnonzero(weights.any(axis=0));n_lr=len(channels.names)
scores=np.zeros((3,len(names),len(names),n_lr),dtype=np.float64)
print(f'Processing {len(near)} neighborhood cells',flush=True)
for n,i in enumerate(near):
 cats,vals=tensor.fiber(i,allowed)[1:3]
 for k in np.flatnonzero(weights[:,i]):
  weight=weights[k,i]
  for j,neighbor in enumerate(cats):
   scores[k,types[i],neighbor,:]+=weight*vals[j,:n_lr]
   scores[k,neighbor,types[i],:]+=weight*vals[j,n_lr:]
 if (n+1)%5000==0: print(f'{n+1}/{len(near)}',flush=True)
tmp=out/'direct_interactions.partial.pkl'
with tmp.open('wb') as f:pickle.dump({'scores':scores,'types':names,'lr':channels.names},f,protocol=5)
os.replace(tmp,out/'direct_interactions.pkl');print('Finished',flush=True)
