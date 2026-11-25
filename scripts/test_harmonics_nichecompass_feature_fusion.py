"""Harmonics composition + pretrained NicheCompass embedding, posthoc experiment.
All embeddings and 10-seed baselines are existing; K and equal block RMS fixed.
Ground truth loaded only after predictions; no architecture or benchmark mutation.
"""
import os,sys,json
os.environ.setdefault('OPENBLAS_NUM_THREADS','1');os.environ.setdefault('OMP_NUM_THREADS','3')
from pathlib import Path
import numpy as np,pandas as pd,anndata as ad
from scipy import sparse
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score,confusion_matrix
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0,'/home/nas3/biod/xueshuailin/CCC_Phe/workspace/artifacts/baselines/Harmonics')
from Harmonics import Harmonics_Model
W=Path('/data1/xueshuailin/CCC_Phe/workspace');B=Path('/home/nas3/biod/xueshuailin/CCC_Phe/workspace/outputs/nichecompass_sim_10run_benchmark');O=W/'outputs/nichecompass_simulation/harmonics_nichecompass_fusion';O.mkdir(parents=True,exist_ok=True)
a=ad.read_h5ad(W/'data/nichecompass_simulation/sim1_1105genes_10000locs_strongincrements.h5ad');truth_raw=a.obs.pop('niche_types').copy();a.uns.pop('gp_data',None);a.X=sparse.csr_matrix(a.layers['counts'],dtype=np.float32);a.obs['celltype']=a.obs.cell_types.astype(str);xy=np.asarray(a.obsm['spatial']);m=Harmonics_Model([a],['slice'],seed=112,parallel=False,verbose=False,light_mode=True);m.preprocess(ct_key='celltype',spatial_key='spatial',method='joint',n_step=3,n_neighbors=20);p=m.micro_dist_basic.toarray() if sparse.issparse(m.micro_dist_basic) else np.asarray(m.micro_dist_basic);n=len(p)
def norm(x):
 x=np.asarray(x,dtype='float32');x=x-x.mean(0);return x/max(float(np.sqrt(np.mean(np.sum(x*x,axis=1)))),1e-8)
p=norm(np.sqrt(np.maximum(p,0)));preds={}
for seed in range(112,122):
 h=np.load(B/f'nichecompass/seed_{seed}/h.npy');assert len(h)==n;x=np.column_stack((p,norm(h)));preds[seed]=KMeans(n_clusters=8,n_init=10,random_state=seed).fit_predict(x);print('fused seed',seed,flush=True)
# Ground truth enters only here, after all fusions are completed.
y=pd.Categorical(truth_raw,categories=[f'Niche {i}' for i in range(1,9)]).codes;assert (y>=0).all();rows=[]
for seed,labels in preds.items():
 rows.append({'method':'Harmonics composition + NicheCompass h','seed':seed,'ARI':float(adjusted_rand_score(y,labels))})
 for method in ['harmonics','nichecompass','v3']:
  val=json.loads((B/method/f'seed_{seed}/metrics.json').read_text());rows.append({'method':method,'seed':seed,'ARI':val['ARI']})
df=pd.DataFrame(rows);df.to_csv(O/'metrics.csv',index=False);np.savez_compressed(O/'fusion_labels.npz',**{str(k):v for k,v in preds.items()});print(df.groupby('method').ARI.agg(['mean','std','min','max']).to_string(),flush=True)
# Median-run figure; only colours matched to true labels for display.
ordered=['harmonics','nichecompass','v3','Harmonics composition + NicheCompass h'];titles=['Harmonics','NicheCompass','V3','Composition + NicheCompass h'];pal=plt.get_cmap('tab10')(np.arange(8));fig,axes=plt.subplots(1,5,figsize=(10,4.2),layout='constrained')
for j in range(5):
 ax=axes[j]
 if j==0:labels=y;title='Ground truth'
 else:
  method=ordered[j-1];sub=df[df.method==method];r=sub.iloc[np.argmin(abs(sub.ARI-sub.ARI.median()).to_numpy())];seed=int(r.seed);labels=preds[seed] if method.startswith('Harmonics composition') else np.load(B/method/f'seed_{seed}/result.npz')['labels'];rr,cc=linear_sum_assignment(-confusion_matrix(y,labels,labels=range(8)));mapping=dict(zip(cc,rr));labels=np.array([mapping[x] for x in labels]);title=f'{titles[j-1]}\nARI={r.ARI:.3f}'
 ax.scatter(xy[:,0],xy[:,1],c=pal[labels],s=2.3,lw=0,rasterized=True);ax.set_aspect('equal');ax.axis('off');ax.set_title(title,fontsize=9)
fig.savefig(O/'comparison.png',dpi=100,bbox_inches='tight');plt.close(fig);(O/'protocol.json').write_text(json.dumps({'composition':'Harmonics three-step Delaunay minimum 20, sqrt, centered global RMS','communication':'pretrained NicheCompass h seed112-121, centered global RMS','fusion':'concatenated equally normalized blocks then KMeans, K=8','not':'not full Harmonics algorithm nor NicheCompass decoder; this is a direct feature-fusion baseline','random_seed':list(range(112,122)),'truth':'evaluation and color-matching only','limitations':'single composition-aligned simulated tissue, no independent proof of communication-specific contribution'},indent=2));print('COMPLETE',flush=True)
