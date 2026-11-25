"""Exploratory confidence-adaptive graph on NicheCompass; no truth in fitting."""
from pathlib import Path
import json,time
import numpy as np,pandas as pd,anndata as ad
from scipy import sparse
from scipy.optimize import linear_sum_assignment
from scipy.sparse.csgraph import connected_components
from sklearn.neighbors import NearestNeighbors
from sklearn.manifold import spectral_embedding
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score,confusion_matrix
W=Path('/data1/xueshuailin/CCC_Phe/workspace');BASE=W/'outputs/nichecompass_simulation/physical_knn50';OUT=BASE/'adaptive_similarity_K8';OUT.mkdir(parents=True,exist_ok=True)
def log(x):print(time.strftime('%F %T'),x,flush=True)
target=Path(json.loads((BASE/'target_cache.json').read_text())['folder']);comp=np.sqrt(np.maximum(np.load(target/'composition.npy'),0)).astype('float32');h=np.load(BASE/'global_communication_K8/h.npy').astype('float32');h=(h-h.mean(0))/max(float(np.sqrt(np.mean((h-h.mean(0))**2))),1e-8)
# Confidence is unsupervised distance margin between two nearest prototypes, per view.
# A margin measures internal geometric separation, not biological correctness.
def confidence(X):
 km=KMeans(n_clusters=8,n_init=10,random_state=40700).fit(X);d=km.transform(X);d.sort(axis=1);return np.clip((d[:,1]-d[:,0])/np.maximum(d[:,1],1e-8),0,1).astype('float32')
log('Fitting unsupervised confidence prototypes')
cc=confidence(comp);ch=confidence(h)
# 0.25 baseline odds, adapted by confidence ratio; bounds prevent either view disappearing.
a=np.clip(ch/(ch+3*cc+1e-6),.10,.60).astype('float32')
log('Building independent 30-NN graphs')
edges=[];k=30
for X in (comp,h):
 _,ind=NearestNeighbors(n_neighbors=k+1,n_jobs=4).fit(X).kneighbors(X);i=np.repeat(np.arange(len(X)),k);j=ind[:,1:].reshape(-1);edges.append(sparse.coo_matrix((np.ones(len(i),np.float32),(i,j)),shape=(len(X),len(X))).tocsr())
mask=(edges[0]+edges[1]).maximum((edges[0]+edges[1]).T).tocoo();i,j=mask.row,mask.col;dc=((comp[i]-comp[j])**2).sum(1);dh=((h[i]-h[j])**2).sum(1);cs=max(float(np.median(dc[dc>0])),1e-8);hs=max(float(np.median(dh[dh>0])),1e-8);weight=(a[i]+a[j])/2;v=(1-weight)*np.exp(-dc/cs)+weight*np.exp(-dh/hs);g=sparse.csr_matrix((v,(i,j)),shape=(len(comp),len(comp)));g=g.maximum(g.T);g.setdiag(0);g.eliminate_zeros();log(f'spectral embedding, components={connected_components(g,directed=False,return_labels=False)}')
z=spectral_embedding(g,n_components=8,drop_first=False,random_state=40700,eigen_solver='arpack',eigen_tol=1e-4);labels=KMeans(n_clusters=8,n_init=10,random_state=40700).fit_predict(z);np.save(OUT/'labels_K8.npy',labels);np.savez_compressed(OUT/'confidence.npz',composition=cc,communication=ch,communication_weight=a)
ad=ad.read_h5ad(W/'data/nichecompass_simulation/sim1_1105genes_10000locs_strongincrements.h5ad',backed='r');truth=pd.Categorical(ad.obs.niche_types,categories=[f'Niche {i}' for i in range(1,9)]).codes;ad.file.close();ari=adjusted_rand_score(truth,labels);rows,cols=linear_sum_assignment(-confusion_matrix(truth,labels,labels=range(8)));mapped=np.array([dict(zip(cols,rows))[x] for x in labels]);recall={f'Niche {j+1}':float(np.mean(mapped[truth==j]==j)) for j in range(8)};(OUT/'result.json').write_text(json.dumps({'ARI':float(ari),'recall_by_truth':recall,'communication_weight_quantiles':np.quantile(a,[0,.1,.5,.9,1]).tolist(),'truth_only_for_evaluation':True,'caveat':'prototype distance margin is geometric confidence, not proven reliability; fixed 0.1–0.6 bounds'},indent=2));log('ARI '+str(ari)+' recalls '+str(recall))
import nbformat
from nbclient import NotebookClient
p=Path('/home/xueshuailin/CCC_Phe/NicheCompass_Simulation_V2.ipynb');nb=nbformat.read(p,4)
source=f'''# 自适应组成与通讯融合图，K=8（探索）
from pathlib import Path
import numpy as np, pandas as pd, anndata as ad, matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.metrics import adjusted_rand_score,confusion_matrix
from scipy.optimize import linear_sum_assignment
run=Path('{OUT}');a=ad.read_h5ad('{W}/data/nichecompass_simulation/sim1_1105genes_10000locs_strongincrements.h5ad',backed='r');xy=np.asarray(a.obsm['spatial']);truth=pd.Categorical(a.obs['niche_types'],categories=[f'Niche {{i}}' for i in range(1,9)]).codes;a.file.close()
pred=np.load(run/'labels_K8.npy');ari=adjusted_rand_score(truth,pred);rows,cols=linear_sum_assignment(-confusion_matrix(truth,pred,labels=np.arange(8)));mapping=dict(zip(cols,rows));matched=np.array([mapping[k] for k in pred]);colors=plt.get_cmap('tab10')(np.arange(8));order=np.random.default_rng(2026).permutation(len(truth));fig,axes=plt.subplots(1,2,figsize=(6.8,7),layout='constrained')
for ax,lab,title in [(axes[0],truth,'Ground truth · 8 niches'),(axes[1],matched,f'Adaptive joint graph · K=8\\nARI = {{ari:.3f}}')]:
 ax.scatter(xy[order,0],xy[order,1],c=colors[lab[order]],s=4,lw=0,rasterized=True);ax.set_aspect('equal');ax.axis('off');ax.set_title(title,fontsize=11)
fig.legend(handles=[Line2D([],[],marker='o',lw=0,color=colors[i],label=f'Niche {{i+1}}',markersize=5) for i in range(8)],loc='outside lower center',ncol=4,frameon=False,fontsize=8);plt.show()
print('微环境 KNN50；组成和通讯分别估计无监督距离间隔，按微环境自适应融合。真值仅用于评估和配色。')
'''
one=nbformat.v4.new_notebook(cells=[nbformat.v4.new_code_cell(source)],metadata=nb.metadata);NotebookClient(one,timeout=300,kernel_name='cccphe').execute(cwd=str(p.parent));latest=nbformat.read(p,4)
if any(c.cell_type=='code' and c.source.startswith('# 自适应组成与通讯融合图，K=8（探索）') for c in latest.cells):raise RuntimeError('cell exists')
latest.cells.append(one.cells[0]);nbformat.write(latest,p);log('Notebook updated')
