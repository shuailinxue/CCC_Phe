"""Fixed-K single-view KMeans on saved brain benchmark inputs and h."""
import os,json
os.environ.setdefault('OMP_NUM_THREADS','4');os.environ.setdefault('OPENBLAS_NUM_THREADS','4');os.environ.setdefault('MKL_NUM_THREADS','4')
from pathlib import Path
import numpy as np,pandas as pd,anndata as ad
from scipy.spatial import cKDTree
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score,confusion_matrix,f1_score
from threadpoolctl import threadpool_limits
W=Path('/data1/xueshuailin/CCC_Phe/workspace');B=Path('/home/nas3/biod/xueshuailin/CCC_Phe/workspace/outputs/brain_10run_benchmark');O=W/'outputs/brain_single_view_10run';O.mkdir(parents=True,exist_ok=True);rows=[]
for dataset in ('starmap','merfish'):
 z=np.load(W/f'outputs/harmonics_fig2/{dataset}_scaled/fit_min25/v3_inputs.npz');p=np.sqrt(np.maximum(z['composition'],0));data=W/'data/harmonics_fig2'/('starmap/analysis_input.h5ad' if dataset=='starmap' else 'merfish_scaled/donor12_slice0_analysis_input.h5ad');a=ad.read_h5ad(data,backed='r');xy=np.asarray(a.obsm['spatial']);n=len(a)
 if dataset=='starmap':
  ref=ad.read_h5ad(W/'data/harmonics_fig2/starmap/stDyer_mouse_cortex.h5ad');dist,ix=cKDTree(np.c_[ref.obs.X,-ref.obs.Y]).query(xy);valid=dist<.01;y=pd.Categorical(ref.obs.label.iloc[ix].astype(str).str.upper()).codes
 else:valid=np.ones(n,bool);y=pd.Categorical(a.obs.benchmark_region.astype(str)).codes
 a.file.close();assert len(p)==n and valid.any()
 with threadpool_limits(limits=4):
  for seed in range(112,122):
   h=np.load(B/dataset/'v3'/f'seed_{seed}'/'h.npy');assert len(h)==n
   for name,features in [('composition_only',p),('h_only',h)]:
    labels=KMeans(n_clusters=7,n_init=10,random_state=seed+40588).fit_predict(features);np.save(O/f'{dataset}_{name}_seed_{seed}.npy',labels);cm=confusion_matrix(y[valid],labels[valid],labels=np.arange(7));r,c=linear_sum_assignment(-cm);mapping=dict(zip(c,r));matched=np.array([mapping.get(v,-1) for v in labels[valid]]);row={'dataset':dataset,'method':name,'seed':seed,'ARI':float(adjusted_rand_score(y[valid],labels[valid])),'macro_F1':float(f1_score(y[valid],matched,average='macro'))};rows.append(row);print(dataset,name,seed,round(row['ARI'],4),flush=True)
 pd.DataFrame(rows).to_csv(O/'metrics.csv',index=False)
(O/'protocol.json').write_text(json.dumps({'input':'same existing sigma20um historical benchmark composition and ten V3 h vectors','algorithm':'KMeans K7 n_init10 seed 40700–40709; sqrt composition or raw h; no spectrum/iteration','truth':'evaluation only','independence':'composition fixed across 10 KMeans seeds; h has 10 separately trained encoders','caveat':'historical mouse-to-human ortholog inputs; development datasets'},indent=2));print('COMPLETE',flush=True)
