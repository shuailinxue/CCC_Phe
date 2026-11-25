"""Composition-matched communication diagnostic; tissue labels used only in evaluation."""
from pathlib import Path
import sys,json
import numpy as np,pandas as pd,anndata as ad
from scipy.spatial import cKDTree
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
W=Path('/data1/xueshuailin/CCC_Phe/workspace')
O=Path('/home/nas3/biod/xueshuailin/CCC_Phe/workspace/outputs/brain_v4_communication_diagnosis');O.mkdir(parents=True,exist_ok=True)
IN=W/'outputs/harmonics_fig2/merfish_scaled/fit_min25/v3_inputs.npz';a=np.load(IN);p=np.sqrt(np.maximum(a['composition'],0));xy=a['coordinates'];n=len(p)
h=np.load('/home/nas3/biod/xueshuailin/CCC_Phe/workspace/outputs/brain_v4_weighted_exploration/merfish/seed_112/predictions.npz')['h']
folds=KFold(n_splits=5,shuffle=True,random_state=12);resid=np.empty_like(h)
for tr,te in folds.split(p):
 model=Ridge(alpha=1).fit(p[tr],h[tr]);resid[te]=h[te]-model.predict(p[te])
total=((h-h.mean(0))**2).sum();frac=1-(resid**2).sum()/max(total,1e-12)
rng=np.random.default_rng(173);ids=np.sort(rng.choice(n,min(n,5000),replace=False));near=NearestNeighbors(n_neighbors=101,n_jobs=4).fit(p).kneighbors(p[ids],return_distance=False)[:,1:]
radius=np.median(cKDTree(xy).query(xy,k=30,workers=4)[0][:,-1]);sep=np.linalg.norm(xy[near]-xy[ids,None],axis=2)>2*radius
valid=sep.sum(1)>=10;ids=ids[valid];near=near[valid];sep=sep[valid]
def select(x):
 out=[]
 for start in range(0,len(ids),64):
  sl=slice(start,start+64); d=np.sum((x[near[sl]]-x[ids[sl],None])**2,axis=2);d[~sep[sl]]=np.inf
  out.append(np.take_along_axis(near[sl],np.argsort(d,axis=1)[:,:10],axis=1))
 return np.concatenate(out)
selected={'composition_candidates':near,'communication_h':select(h),'composition_residual_h':select(resid)}
for i in range(3):selected[f'view_{i}']=select(a[f'x{i}'])
np.savez_compressed(O/'frozen_neighborhoods.npz',ids=ids,eligible=sep,**selected)
# Post-fit evaluation only. Conditional comparison is descriptive, not a significance test.
raw=ad.read_h5ad(W/'data/harmonics_fig2/merfish_scaled/donor12_slice0_analysis_input.h5ad',backed='r');truth=raw.obs.benchmark_region.astype(str).to_numpy();raw.file.close()
eligible_same=(truth[near]==truth[ids,None])&sep
result={'n_cells':n,'n_anchors':len(ids),'spatial_exclusion_radius_coordinate_units':float(2*radius),'composition_explains_h_fraction_cv':float(frac),'candidate_same_region_fraction':float(np.mean(eligible_same.sum(1)/sep.sum(1))),'methods':{}}
for name,idx in selected.items():
 if name=='composition_candidates':continue
 result['methods'][name]=float(np.mean(truth[idx]==truth[ids,None]))
base=Path('/home/nas3/biod/xueshuailin/CCC_Phe/workspace/outputs/brain_v3_communication_audit/merfish/composition_only_seed112/result.npz'); joint=Path('/home/nas3/biod/xueshuailin/CCC_Phe/workspace/outputs/brain_v4_weighted_exploration/merfish/seed_112/predictions.npz')
y=pd.Categorical(truth).codes;labs=[]
for path in (base,joint):
 z=np.load(path)['labels'];cm=confusion_matrix(y,z,labels=np.arange(7));r,c=linear_sum_assignment(-cm);mp=dict(zip(c,r));labs.append(np.array([mp[i] for i in z]))
b,v=labs;changed=b!=v
result['assignment']={'n_changed':int(changed.sum()),'corrected':int(np.sum(changed&(b!=y)&(v==y))),'spoiled':int(np.sum(changed&(b==y)&(v!=y))),'wrong_to_wrong':int(np.sum(changed&(b!=y)&(v!=y)))}
(O/'report.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2),flush=True)
