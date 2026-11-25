"""Fuse equal-weight local composition with saved NicheCompass latent state."""
import os,json
os.environ.setdefault('OMP_NUM_THREADS','4');os.environ.setdefault('OPENBLAS_NUM_THREADS','4')
from pathlib import Path
import numpy as np,pandas as pd,anndata as ad
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score,confusion_matrix,f1_score
from scipy.optimize import linear_sum_assignment
from threadpoolctl import threadpool_limits
W=Path('/data1/xueshuailin/CCC_Phe/workspace');B=W/'outputs/nichecompass_simulation/knn30_equal_v3_compare';N=Path('/home/nas3/biod/xueshuailin/CCC_Phe/workspace/outputs/nichecompass_sim_10run_benchmark/nichecompass/seed_112');O=W/'outputs/nichecompass_simulation/equal_composition_nichecompass_fusion';O.mkdir(parents=True,exist_ok=True);p=np.sqrt(np.maximum(np.load(Path(json.loads((B/'target_cache.json').read_text())['folder'])/'composition.npy'),0));h=np.load(N/'h.npy');a=ad.read_h5ad(W/'data/nichecompass_simulation/sim1_1105genes_10000locs_strongincrements.h5ad',backed='r');y=pd.Categorical(a.obs.niche_types,categories=[f'Niche {i}' for i in range(1,9)]).codes;a.file.close();assert len(y)==len(p)==len(h) and (y>=0).all()
def norm(x):
 x=x-x.mean(0);return x/max(float(np.sqrt(np.mean(np.sum(x*x,axis=1)))),1e-8)
p=norm(p);h=norm(h);scores=[]
with threadpool_limits(limits=4):
 for name,x in [('nichecompass_latent_kmeans',h),('composition_plus_nichecompass',np.column_stack((p,h)))]:
  lab=KMeans(n_clusters=8,n_init=10,random_state=40700).fit_predict(x);r,c=linear_sum_assignment(-confusion_matrix(y,lab,labels=np.arange(8)));mapped=np.array([dict(zip(c,r))[i] for i in lab]);scores.append({'method':name,'ARI':float(adjusted_rand_score(y,lab)),'macro_F1':float(f1_score(y,mapped,labels=np.arange(8),average='macro'))});np.save(O/(name+'.npy'),lab)
pd.DataFrame(scores).to_csv(O/'metrics.csv',index=False);(O/'protocol.json').write_text(json.dumps({'composition':'our KNN30 equal physical pair weights','latent':'saved NicheCompass embedding seed112, trained with its own original graph; not V3 communication h','method':'sqrt composition and latent individually centered and RMS-scaled; equal block weights; KMeans K8 n_init10 seed40700','comparison':'NicheCompass latent KMeans and NicheCompass original Leiden are distinct','truth':'evaluation and display only'},indent=2));print(pd.DataFrame(scores).to_string(index=False))
