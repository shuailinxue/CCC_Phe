"""Independent fixed-K joint prototype experiment; truth used only after fitting."""
import os,json
from pathlib import Path
os.environ.setdefault('OMP_NUM_THREADS','3');os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import numpy as np,pandas as pd,anndata as ad
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score,confusion_matrix
from scipy.optimize import linear_sum_assignment
W=Path('/data1/xueshuailin/CCC_Phe/workspace');B=W/'outputs/nichecompass_simulation';O=B/'simple_joint_centroid_knn30_equal';O.mkdir(parents=True,exist_ok=True);src=B/'knn30_equal_v3_compare';t=Path(json.loads((src/'target_cache.json').read_text())['folder']);p=np.sqrt(np.maximum(np.load(t/'composition.npy'),0)).astype('float32');h=np.load(src/'h.npy').astype('float32');n=len(p);K=8
# Fixed rule follows prior top-four joint-distance experiment; no truth used.
def fit(seed):
 initial=KMeans(n_clusters=K,n_init=10,random_state=seed).fit_predict(p);labels=initial.copy();history=[];saved={}
 for it in range(30):
  if len(np.unique(labels))!=K:break
  cp=np.stack([p[labels==j].mean(0) for j in range(K)]);ch=np.stack([h[labels==j].mean(0) for j in range(K)]);dp=np.linalg.norm(p[:,None,:]-cp[None,:,:],axis=2);dh=np.linalg.norm(h[:,None,:]-ch[None,:,:],axis=2);i=np.arange(n);sp=max(float(np.median(dp[i,labels])),1e-8);sh=max(float(np.median(dh[i,labels])),1e-8);score=dp/sp+2*dh/sh;candidates=np.argsort(dp,axis=1)[:,:4];masked=np.full_like(score,np.inf);masked[i[:,None],candidates]=score[i[:,None],candidates];new=masked.argmin(1);changed=int((new!=labels).sum());history.append(changed)
  if it==0:saved['once']=new.copy()
  if len(np.unique(new))!=K:print('empty cluster; preserving last nonempty',flush=True);break
  labels=new
  if changed==0:break
 return initial,saved.get('once',labels),labels,history
# Make predictions first, then open true labels for evaluation.
results={seed:fit(seed) for seed in [40700,40701,40702]}
a=ad.read_h5ad(W/'data/nichecompass_simulation/sim1_1105genes_10000locs_strongincrements.h5ad',backed='r');y=pd.Categorical(a.obs.niche_types,categories=[f'Niche {i}' for i in range(1,9)]).codes;xy=np.asarray(a.obsm['spatial']);a.file.close();rows=[]
for seed,(initial,once,final,hist) in results.items():
 for name,pred in [('composition',initial),('one_update',once),('iterated',final)]:rows.append({'seed':seed,'method':name,'ARI':adjusted_rand_score(y,pred),'iterations':len(hist),'last_changed':hist[-1]})
 print(seed,'history',hist,'ARIs',[(r['method'],round(r['ARI'],5)) for r in rows[-3:]],flush=True)
pd.DataFrame(rows).to_csv(O/'metrics.csv',index=False);x=results[40700];np.savez_compressed(O/'labels.npz',truth=y,composition=x[0],once=x[1],iterated=x[2]);(O/'protocol.json').write_text(json.dumps({'K':8,'initial':'KMeans sqrt local composition','input':'KNN30 equal weighting; existing communication h seed112','candidate_classes':4,'distance':'composition/median within-cluster distance + 2 * h/median within-cluster distance','update':'recompute centers and 4 candidates per iteration; max30','truth':'evaluation only','note':'3 Kmeans seeds, fixed communication model; development tissue'},indent=2));print('COMPLETE',flush=True)
