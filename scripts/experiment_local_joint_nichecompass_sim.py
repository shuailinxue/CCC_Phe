"""Unsupervised local union graph of composition and communication neighbors.

Exploratory control, not an update to mainline V3. Ground truth is read after
clustering and never used in graph construction, selection or parameters.
"""
import os,json
os.environ.setdefault('OMP_NUM_THREADS','4');os.environ.setdefault('OPENBLAS_NUM_THREADS','4')
from pathlib import Path
import numpy as np,pandas as pd,anndata as ad
from scipy import sparse
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import eigsh
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import adjusted_rand_score,confusion_matrix,f1_score
from scipy.optimize import linear_sum_assignment
from threadpoolctl import threadpool_limits
W=Path('/data1/xueshuailin/CCC_Phe/workspace');B=W/'outputs/nichecompass_simulation/knn30_equal_v3_compare';O=W/'outputs/nichecompass_simulation/local_joint_equal_experiment';O.mkdir(parents=True,exist_ok=True);folder=Path(json.loads((B/'target_cache.json').read_text())['folder']);p=np.sqrt(np.maximum(np.load(folder/'composition.npy'),0)).astype('float32');h=np.load(B/'h.npy').astype('float32');n=len(p);assert len(h)==n and np.isfinite(p).all() and np.isfinite(h).all()
# Global block scale makes distances reproducible; local scales correct variable density.
def norm(x):
 x=x-x.mean(0);return x/max(float(np.sqrt(np.mean(np.sum(x*x,axis=1)))),1e-8)
p=norm(p);h=norm(h);kp,kh,local=30,30,30
with threadpool_limits(limits=4):
 print('Finding composition and communication neighbors',flush=True)
 dp,ip=NearestNeighbors(n_neighbors=kp+1,algorithm='brute',n_jobs=4).fit(p).kneighbors(p);dh,ih=NearestNeighbors(n_neighbors=kh+1,algorithm='brute',n_jobs=4).fit(h).kneighbors(h)
 dp,ip,dh,ih=dp[:,1:],ip[:,1:],dh[:,1:],ih[:,1:];ps=np.maximum(dp[:,-1],1e-5);hs=np.maximum(dh[:,-1],1e-5)
 # Each candidate may arise from either block. Gaussian affinities have soft tails,
 # so communicating cells can connect even with different compositions.
 i=np.repeat(np.arange(n),kp+kh);j=np.column_stack((ip,ih)).ravel();dpc=np.linalg.norm(p[i]-p[j],axis=1);dhc=np.linalg.norm(h[i]-h[j],axis=1);wp=np.exp(-.5*dpc**2/np.maximum(ps[i]*ps[j],1e-10));wh=np.exp(-.5*dhc**2/np.maximum(hs[i]*hs[j],1e-10));weight=wp*wh
 graph=sparse.coo_matrix((weight,(i,j)),shape=(n,n)).tocsr();graph.sum_duplicates();graph.data=np.minimum(graph.data,1);graph=graph.maximum(graph.T);graph.eliminate_zeros();degree=np.asarray(graph.sum(1)).ravel();inv=1/np.sqrt(np.maximum(degree,1e-12));aff=sparse.diags(inv)@graph@sparse.diags(inv);components=int(connected_components(graph,directed=False,return_labels=False));print('Graph built components',components,'edges',graph.nnz,flush=True)
 ev,vec=eigsh(aff,k=10,which='LA',tol=1e-4,maxiter=4000,v0=np.random.default_rng(42).normal(size=n));spectral=vec[:,np.argsort(ev)[::-1][:8]];spectral/=np.maximum(np.linalg.norm(spectral,axis=1,keepdims=True),1e-12);labels=KMeans(n_clusters=8,n_init=10,random_state=40700).fit_predict(spectral)
np.save(O/'labels.npy',labels);sparse.save_npz(O/'graph.npz',graph)
# Evaluation only after graph and labels are frozen.
a=ad.read_h5ad(W/'data/nichecompass_simulation/sim1_1105genes_10000locs_strongincrements.h5ad',backed='r');y=pd.Categorical(a.obs.niche_types,categories=[f'Niche {i}' for i in range(1,9)]).codes;a.file.close();assert len(y)==n and (y>=0).all();rr,cc=linear_sum_assignment(-confusion_matrix(y,labels,labels=range(8)));match=dict(zip(cc,rr));aligned=np.array([match[x] for x in labels]);res={'ARI':float(adjusted_rand_score(y,labels)),'macro_F1':float(f1_score(y,aligned,labels=range(8),average='macro')),'K':8,'graph_components':components,'graph_nnz':int(graph.nnz),'composition_neighbors':kp,'communication_neighbors':kh,'seed':40700,'representation_seed':112,'protocol':'union of composition 30-NN and h 30-NN, product of local-scaled Gaussian affinities, symmetric graph, normalized spectral embedding K=8 and KMeans n_init10; no spatial label smoothing','truth':'post-fit evaluation only','caveat':'one fixed h, development simulation, adaptive local scales exploratory; no proof of improved communication-specific states'};(O/'result.json').write_text(json.dumps(res,indent=2));print(json.dumps(res,indent=2),flush=True)
