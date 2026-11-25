"""Exploratory global composition + communication affinity graph; truth used only for evaluation."""
from pathlib import Path
import json, time
import numpy as np, pandas as pd
from scipy import sparse
from scipy.sparse.csgraph import connected_components
from sklearn.neighbors import NearestNeighbors
from sklearn.manifold import spectral_embedding
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
import anndata as ad
W=Path('/data1/xueshuailin/CCC_Phe/workspace'); BASE=W/'outputs/nichecompass_simulation/v2'; OUT=BASE/'joint_similarity_K8'; OUT.mkdir(parents=True,exist_ok=True)
def log(s): print(time.strftime('%F %T'),s,flush=True)
target=Path(json.loads((BASE/'target_cache.json').read_text())['folder'])
comp=np.sqrt(np.maximum(np.load(target/'composition.npy'),0)).astype('float32')
h=np.load(BASE/'global_communication_K8/h.npy').astype('float32')
# One global scale per block; no supervision or per-feature inflation.
h=(h-h.mean(0))/max(float(np.sqrt(np.mean((h-h.mean(0))**2))),1e-8)
k=30
log('Building independent composition and communication kNN graphs')
edges=[]
for X in (comp,h):
 nn=NearestNeighbors(n_neighbors=k+1,n_jobs=4).fit(X); _,ind=nn.kneighbors(X); rows=np.repeat(np.arange(len(X)),k); cols=ind[:,1:].reshape(-1); edges.append(sparse.coo_matrix((np.ones(len(rows),np.float32),(rows,cols)),shape=(len(X),len(X))).tocsr())
mask=(edges[0]+edges[1]).maximum((edges[0]+edges[1]).T).tocoo(); i,j=mask.row,mask.col
cd=np.sum((comp[i]-comp[j])**2,axis=1); hd=np.sum((h[i]-h[j])**2,axis=1)
# Median over graph edges calibrates each view without labels.
cs=max(float(np.median(cd[cd>0])),1e-8); hs=max(float(np.median(hd[hd>0])),1e-8)
alpha=.25
v=(1-alpha)*np.exp(-cd/cs)+alpha*np.exp(-hd/hs)
graph=sparse.csr_matrix((v,(i,j)),shape=(len(comp),len(comp))); graph=graph.maximum(graph.T); graph.setdiag(0);graph.eliminate_zeros()
ncomp=connected_components(graph,directed=False,return_labels=False);log(f'Graph edges {graph.nnz}, connected components {ncomp}; spectral embedding')
z=spectral_embedding(graph,n_components=8,drop_first=False,random_state=40700,eigen_solver='arpack',eigen_tol=1e-4)
labels=KMeans(n_clusters=8,n_init=10,random_state=40700).fit_predict(z)
np.save(OUT/'labels_K8.npy',labels);np.save(OUT/'spectral_embedding.npy',z)
a=ad.read_h5ad(W/'data/nichecompass_simulation/sim1_1105genes_10000locs_strongincrements.h5ad',backed='r');truth=pd.Categorical(a.obs.niche_types,categories=[f'Niche {i}' for i in range(1,9)]).codes;a.file.close();ari=adjusted_rand_score(truth,labels)
(OUT/'result.json').write_text(json.dumps({'ARI':float(ari),'k':8,'neighbors_per_view':k,'communication_weight':alpha,'composition_median_squared_distance':cs,'communication_median_squared_distance':hs,'components':int(ncomp),'truth_used_only_for_evaluation':True,'reliability_adaptation':False,'communication_representation':'previously trained global three-encoder h64'},indent=2));log(f'ARI {ari:.6f}')
import nbformat
from nbclient import NotebookClient
p=Path('/home/xueshuailin/CCC_Phe/NicheCompass_Simulation_V2.ipynb'); nb=nbformat.read(p,4)
source=f'''# 联合相似性图：组成 + 通讯，K=8
from pathlib import Path
import numpy as np, pandas as pd, anndata as ad, matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.metrics import adjusted_rand_score, confusion_matrix
from scipy.optimize import linear_sum_assignment
run=Path('{OUT}')
a=ad.read_h5ad('{W}/data/nichecompass_simulation/sim1_1105genes_10000locs_strongincrements.h5ad',backed='r')
xy=np.asarray(a.obsm['spatial']); truth=pd.Categorical(a.obs['niche_types'],categories=[f'Niche {{i}}' for i in range(1,9)]).codes; a.file.close()
pred=np.load(run/'labels_K8.npy'); ari=adjusted_rand_score(truth,pred)
rows,cols=linear_sum_assignment(-confusion_matrix(truth,pred,labels=np.arange(8))); mapping=dict(zip(cols,rows)); matched=np.array([mapping[k] for k in pred])
colors=plt.get_cmap('tab10')(np.arange(8));order=np.random.default_rng(2026).permutation(len(truth));fig,axes=plt.subplots(1,2,figsize=(6.8,7),layout='constrained')
for ax,labels,title in [(axes[0],truth,'Ground truth · 8 niches'),(axes[1],matched,f'Joint similarity graph · K=8\\nARI = {{ari:.3f}}')]:
 ax.scatter(xy[order,0],xy[order,1],c=colors[labels[order]],s=4,lw=0,rasterized=True);ax.set_aspect('equal');ax.axis('off');ax.set_title(title,fontsize=11)
fig.legend(handles=[Line2D([],[],marker='o',lw=0,color=colors[i],label=f'Niche {{i+1}}',markersize=5) for i in range(8)],loc='outside lower center',ncol=4,frameon=False,fontsize=8);plt.show()
print('组成和三路通讯 h64 分别构建近邻，经距离校准后联合构图；通讯权重固定为 0.25。真值仅用于评估与配色。')
'''
one=nbformat.v4.new_notebook(cells=[nbformat.v4.new_code_cell(source)],metadata=nb.metadata);NotebookClient(one,timeout=300,kernel_name='cccphe').execute(cwd=str(p.parent));latest=nbformat.read(p,4)
if any(c.cell_type=='code' and c.source.startswith('# 联合相似性图：组成 + 通讯，K=8') for c in latest.cells): raise RuntimeError('Cell exists; do not overwrite')
latest.cells.append(one.cells[0]);nbformat.write(latest,p);log('Notebook updated')
