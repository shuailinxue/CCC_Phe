"""Direct KMeans on concatenated globally scaled composition and h, no graph."""
import os,json
os.environ.setdefault('OMP_NUM_THREADS','4');os.environ.setdefault('OPENBLAS_NUM_THREADS','4')
from pathlib import Path
import numpy as np,pandas as pd,anndata as ad
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score,f1_score,confusion_matrix
from scipy.optimize import linear_sum_assignment
from threadpoolctl import threadpool_limits
W=Path('/data1/xueshuailin/CCC_Phe/workspace');B=W/'outputs/nichecompass_simulation/knn30_equal_v3_compare';O=W/'outputs/nichecompass_simulation/direct_concat_knn30_equal';O.mkdir(parents=True,exist_ok=True);f=Path(json.loads((B/'target_cache.json').read_text())['folder']);p=np.sqrt(np.maximum(np.load(f/'composition.npy'),0));h=np.load(B/'h.npy');a=ad.read_h5ad(W/'data/nichecompass_simulation/sim1_1105genes_10000locs_strongincrements.h5ad',backed='r');y=pd.Categorical(a.obs.niche_types,categories=[f'Niche {i}' for i in range(1,9)]).codes;a.file.close();assert len(y)==len(p)==len(h) and (y>=0).all()
def scale(x):
 x=x-x.mean(0);return x/max(float(np.sqrt(np.mean(np.sum(x*x,axis=1)))),1e-8)
x=np.column_stack((scale(p),scale(h)))
with threadpool_limits(limits=4):labels=KMeans(n_clusters=8,n_init=10,random_state=40700).fit_predict(x)
r,c=linear_sum_assignment(-confusion_matrix(y,labels,labels=np.arange(8)));mapped=np.array([dict(zip(c,r))[i] for i in labels]);result={'ARI':float(adjusted_rand_score(y,labels)),'macro_F1':float(f1_score(y,mapped,labels=np.arange(8),average='macro')),'K':8,'method':'KMeans n_init=10 seed40700 on concatenated blocks','blocks':'sqrt composition and communication h, each globally centered and RMS-scaled, equal block weights','graph_or_iterative_updates':False,'h':'equal-weight pre-trained encoder seed112','truth':'evaluation only'};np.save(O/'labels.npy',labels);(O/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
