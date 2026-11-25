"""Composition-only and pretrained V3 communication-h-only KMeans controls."""
import os
os.environ.setdefault('OMP_NUM_THREADS','4');os.environ.setdefault('OPENBLAS_NUM_THREADS','4');os.environ.setdefault('MKL_NUM_THREADS','4')
import json
from pathlib import Path
import numpy as np,pandas as pd,anndata as ad
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score,confusion_matrix,f1_score
from scipy.optimize import linear_sum_assignment
from threadpoolctl import threadpool_limits
W=Path('/data1/xueshuailin/CCC_Phe/workspace');B=Path('/home/nas3/biod/xueshuailin/CCC_Phe/workspace/outputs/nichecompass_sim_10run_benchmark');O=W/'outputs/nichecompass_simulation/single_view_kmeans_10run';O.mkdir(parents=True,exist_ok=True)
src=W/'data/nichecompass_simulation/sim1_1105genes_10000locs_strongincrements.h5ad';a=ad.read_h5ad(src,backed='r');truth=pd.Categorical(a.obs['niche_types'],categories=[f'Niche {i}' for i in range(1,9)]).codes;a.file.close();assert (truth>=0).all()
target=Path(json.loads((W/'outputs/nichecompass_simulation/v2/target_cache.json').read_text())['folder']);comp=np.load(target/'composition.npy',mmap_mode='r');assert len(comp)==len(truth);x=np.sqrt(np.maximum(comp,0));rows=[]
with threadpool_limits(limits=4):
 for seed in range(112,122):
  h=np.load(B/'v3'/f'seed_{seed}'/'h.npy');assert h.ndim==2 and len(h)==len(truth)
  for name,z in [('composition_only',x),('communication_h_only',h)]:
   labels=KMeans(n_clusters=8,n_init=10,random_state=seed+40588).fit_predict(z);score=float(adjusted_rand_score(truth,labels));cm=confusion_matrix(truth,labels,labels=np.arange(8));r,c=linear_sum_assignment(-cm);aligned=np.array([dict(zip(c,r))[i] for i in labels]);f1=float(f1_score(truth,aligned,labels=np.arange(8),average='macro'));rows.append({'method':name,'seed':seed,'kmeans_seed':seed+40588,'ARI':score,'macro_F1':f1});np.savez_compressed(O/f'{name}_seed_{seed}.npz',labels=labels);print(name,seed,round(score,4),flush=True)
pd.DataFrame(rows).to_csv(O/'metrics.csv',index=False);(O/'protocol.json').write_text(json.dumps({'source':str(src),'composition':str(target/'composition.npy'),'h':'saved V3 seed-specific pretrained h; no retraining','K':8,'method':'direct KMeans, n_init=10; sqrt composition, raw h; no graph/postprocessing','truth':'evaluation and display only','caveat':'simulation used during development, not independent test'},indent=2));print('COMPLETE',flush=True)
