"""K=8 composition/h KMeans on the existing KNN30 equal-weight V3 fit."""
import os,json
os.environ.setdefault('OMP_NUM_THREADS','4');os.environ.setdefault('OPENBLAS_NUM_THREADS','4')
from pathlib import Path
import numpy as np,pandas as pd,anndata as ad
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score,f1_score,confusion_matrix
from scipy.optimize import linear_sum_assignment
from threadpoolctl import threadpool_limits
W=Path('/data1/xueshuailin/CCC_Phe/workspace');B=W/'outputs/nichecompass_simulation/knn30_equal_v3_compare';O=W/'outputs/nichecompass_simulation/single_view_knn30_equal';O.mkdir(parents=True,exist_ok=True)
f=Path(json.loads((B/'target_cache.json').read_text())['folder']);p=np.load(f/'composition.npy');h=np.load(B/'h.npy');a=ad.read_h5ad(W/'data/nichecompass_simulation/sim1_1105genes_10000locs_strongincrements.h5ad',backed='r');y=pd.Categorical(a.obs.niche_types,categories=[f'Niche {i}' for i in range(1,9)]).codes;a.file.close();assert len(p)==len(h)==len(y) and (y>=0).all();results=[]
with threadpool_limits(limits=4):
 for name,x in [('composition_only',np.sqrt(np.maximum(p,0))),('communication_h_only',h)]:
  label=KMeans(n_clusters=8,n_init=10,random_state=40700).fit_predict(x);r,c=linear_sum_assignment(-confusion_matrix(y,label,labels=np.arange(8)));mapped=np.array([dict(zip(c,r))[i] for i in label]);results.append({'method':name,'ARI':float(adjusted_rand_score(y,label)),'macro_F1':float(f1_score(y,mapped,labels=np.arange(8),average='macro'))});np.save(O/(name+'.npy'),label)
pd.DataFrame(results).to_csv(O/'metrics.csv',index=False);(O/'protocol.json').write_text(json.dumps({'source':str(B),'K':8,'fit':'saved equal-weight V3 h, seed 112, single trained model','kmeans':'n_init=10 seed=40700; sqrt composition and raw h','reference':'same KNN30, equal physical-pair weights','truth':'evaluation only','no_new_training':True},indent=2));print(pd.DataFrame(results).to_string(index=False))
