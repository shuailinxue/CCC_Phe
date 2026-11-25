"""Use the fitted CP model; pool same-section neighbor programs for 2D comparison."""
import os,pickle
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.sparse import load_npz
from sklearn.cluster import KMeans
from phenoniche.spatial import Settings
from phenoniche.spatial.channels import Channels
from phenoniche.spatial.cell_programs import CellTensor,pool_programs
from phenoniche.spatial.programs import project
import lung3d_v2 as study
out=Path('/data1/xueshuailin/CCC_Phe/workspace/outputs/3D_lung_CosMx/v2_bidirectional/tumor_surface_slice2')
a,xyz=study.read_input()
with (out/'result.pkl').open('rb') as f: prior=pickle.load(f)
ids=a.obs_names.get_indexer(prior['results']['cell_id']);z=prior['results']['z']
graph=load_npz(study.OUT/'contacts.npz');_,types=np.unique(a.obs.celltypes.astype(str),return_inverse=True)
channels=Channels(a.layers['counts'],a.var_names,pd.read_csv(study.ROOT/'data/commuspace_human_lr_atlas.tsv',sep='\t'))
cfg=Settings();tensor=CellTensor(graph,types,channels,cfg.sigma_um)
all_allowed=np.ones(a.n_obs,dtype=bool);same_layer=xyz[:,2]==z
near=np.unique(np.r_[ids,graph[ids].indices]);near=near[same_layer[near]]
scores=np.zeros((a.n_obs,cfg.rank),dtype=np.float32);scores[ids]=prior['scores']
remaining=np.setdiff1d(near,ids)
print(f'{len(ids)} targets, {len(near)} same-slice cells',flush=True)
for j,i in enumerate(remaining):
 scores[i]=np.nan_to_num(project(tensor.fiber(i,all_allowed),prior['model']))
 if (j+1)%5000==0:print(f'{j+1}/{len(remaining)}',flush=True)
features=pool_programs(ids,same_layer,graph,types,scores,cfg.sigma_um)
scale=np.maximum(np.quantile(features,.9,axis=0),1e-7);x=np.log1p(features/scale)
result=dict(prior['results'])
for k in (2,3,4):
 result[f'k{k}']=KMeans(n_clusters=k,n_init=10,random_state=73).fit_predict(x)
 print(k,np.bincount(result[f'k{k}']).tolist(),flush=True)
tmp=out/'microenvironment_2d.partial.pkl'
with tmp.open('wb') as f:pickle.dump(dict(results=result,features=features),f,protocol=5)
os.replace(tmp,out/'microenvironment_2d.pkl')
print('Finished',flush=True)
