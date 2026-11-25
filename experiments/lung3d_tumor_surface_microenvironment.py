"""Pool CP scores from every neighboring cell before clustering target cells."""
import os,pickle
from pathlib import Path
import numpy as np
from scipy.sparse import load_npz
from sklearn.cluster import KMeans
from phenoniche.spatial import Settings
from phenoniche.spatial.channels import Channels
from phenoniche.spatial.cell_programs import CellTensor, pool_programs
from phenoniche.spatial.programs import project
import pandas as pd
import lung3d_v2 as study
out=Path('/data1/xueshuailin/CCC_Phe/workspace/outputs/3D_lung_CosMx/v2_bidirectional/tumor_surface_slice2')
a,xyz=study.read_input()
with (out/'result.pkl').open('rb') as f: prior=pickle.load(f)
ids=a.obs_names.get_indexer(prior['results']['cell_id'])
assert (ids>=0).all()
graph=load_npz(study.OUT/'contacts.npz')
_,types=np.unique(a.obs.celltypes.astype(str),return_inverse=True)
channels=Channels(a.layers['counts'],a.var_names,pd.read_csv(study.ROOT/'data/commuspace_human_lr_atlas.tsv',sep='\t'))
cfg=Settings(); tensor=CellTensor(graph,types,channels,cfg.sigma_um)
allowed=np.ones(a.n_obs,dtype=bool)
neighbors=np.unique(np.concatenate([ids,graph[ids].indices]))
print(f'Target {len(ids)} cells; projecting {len(neighbors)} cells in 3D neighborhoods',flush=True)
scores=np.zeros((a.n_obs,cfg.rank),dtype=np.float32)
scores[ids]=prior['scores']
remaining=np.setdiff1d(neighbors,ids)
for j,i in enumerate(remaining):
    scores[i]=np.nan_to_num(project(tensor.fiber(i,allowed),prior['model']))
    if (j+1)%5000==0:print(f'Projected neighbors {j+1}/{len(remaining)}',flush=True)
features=pool_programs(ids,allowed,graph,types,scores,cfg.sigma_um)
scale=np.maximum(np.quantile(features,.9,axis=0),1e-7)
x=np.log1p(features/scale)
result=dict(prior['results'])
for k in (2,3,4):
 result[f'k{k}']=KMeans(n_clusters=k,n_init=10,random_state=73).fit_predict(x)
 print(k,np.bincount(result[f'k{k}'],minlength=k).tolist(),flush=True)
tmp=out/'microenvironment.partial.pkl'
with tmp.open('wb') as f:pickle.dump(dict(results=result,model=prior['model'],features=features),f,protocol=5)
os.replace(tmp,out/'microenvironment.pkl')
print('Finished',flush=True)
