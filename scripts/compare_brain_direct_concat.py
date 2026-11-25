"""Direct KMeans on sqrt composition + existing V3 h, ten saved seeds."""
import os,json
os.environ.setdefault('OMP_NUM_THREADS','4');os.environ.setdefault('OPENBLAS_NUM_THREADS','4');os.environ.setdefault('MKL_NUM_THREADS','4')
from pathlib import Path
import numpy as np,pandas as pd,anndata as ad
from scipy.spatial import cKDTree
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score,confusion_matrix,f1_score
from threadpoolctl import threadpool_limits
W=Path('/data1/xueshuailin/CCC_Phe/workspace');B=Path('/home/nas3/biod/xueshuailin/CCC_Phe/workspace/outputs/brain_10run_benchmark');O=W/'outputs/brain_direct_concat_10run';O.mkdir(parents=True,exist_ok=True)
def norm(x):
 x=np.asarray(x,dtype=np.float32);x=x-x.mean(0);return x/max(float(np.sqrt(np.mean(np.sum(x*x,axis=1)))),1e-8)
rows=[]
for dataset in ('starmap','merfish'):
 x=np.load(W/f'outputs/harmonics_fig2/{dataset}_scaled/fit_min25/v3_inputs.npz');p=norm(np.sqrt(np.maximum(x['composition'],0)));data=W/'data/harmonics_fig2'/('starmap/analysis_input.h5ad' if dataset=='starmap' else 'merfish_scaled/donor12_slice0_analysis_input.h5ad');a=ad.read_h5ad(data,backed='r');xy=np.asarray(a.obsm['spatial']);n=len(a)
 if dataset=='starmap':
  ref=ad.read_h5ad(W/'data/harmonics_fig2/starmap/stDyer_mouse_cortex.h5ad');dd,ix=cKDTree(np.c_[ref.obs.X,-ref.obs.Y]).query(xy);valid=dd<.01;y=pd.Categorical(ref.obs.label.iloc[ix].astype(str).str.upper()).codes
 else:valid=np.ones(n,bool);y=pd.Categorical(a.obs.benchmark_region.astype(str)).codes
 a.file.close();assert len(p)==n and valid.sum()>0
 with threadpool_limits(limits=4):
  for seed in range(112,122):
   hp=B/dataset/'v3'/f'seed_{seed}'/'h.npy';assert hp.exists(),f'missing {hp}';h=np.load(hp);assert len(h)==n;features=np.column_stack((p,norm(h)));pred=KMeans(n_clusters=7,n_init=10,random_state=seed+40588).fit_predict(features);np.save(O/f'{dataset}_seed_{seed}.npy',pred);cm=confusion_matrix(y[valid],pred[valid],labels=np.arange(7));r,c=linear_sum_assignment(-cm);mapping=dict(zip(c,r));matched=np.array([mapping.get(v,-1) for v in pred[valid]]);row={'dataset':dataset,'seed':seed,'kmeans_seed':seed+40588,'ARI':float(adjusted_rand_score(y[valid],pred[valid])),'macro_F1':float(f1_score(y[valid],matched,average='macro')),'n_evaluated':int(valid.sum())};rows.append(row);print(dataset,seed,'ARI',round(row['ARI'],4),flush=True);pd.DataFrame(rows).to_csv(O/'metrics.csv',index=False)
(O/'protocol.json').write_text(json.dumps({'inputs':'existing historical V3 benchmark h 112-121 and physical target composition, sigma=20um','K':7,'clustering':'sqrt composition and h independently centered and RMS-scaled; equal weights; KMeans n_init10 seed=representation_seed+40588; no spectral graph or post-cluster iterations','annotation':'evaluation and figure colors only','caveat':'historical mouse-to-human ortholog prior; development data; no re-training'},indent=2));print('COMPLETE',flush=True)
