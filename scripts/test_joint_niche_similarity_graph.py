"""Exploratory joint composition/communication graph on existing tumor simulation.

No labels enter graph construction, clustering or parameter selection. The two
pre-specified graphs, a composition-only control and prior V3 results are all
reported; truth is loaded after the labels have been saved.
"""
import os,sys,json,time
os.environ.setdefault('OMP_NUM_THREADS','4');os.environ.setdefault('OPENBLAS_NUM_THREADS','4')
from pathlib import Path
import numpy as np,pandas as pd,torch,anndata as ad
from scipy import sparse
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import eigsh
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score,confusion_matrix
from scipy.optimize import linear_sum_assignment
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
W=Path('/data1/xueshuailin/CCC_Phe/workspace');BASE=W/'outputs/tumor_immune_interface';OUT=BASE/'joint_similarity_graph_experiment';OUT.mkdir(parents=True,exist_ok=True)
D=W/'data/tumor_immune_interface_v2_epithelial_majority';target=Path(json.loads((D/'target_cache.json').read_text())['folder']);P=np.sqrt(np.maximum(np.load(target/'composition.npy'),0)).astype('float32');H=np.load(BASE/'v3_full_tissue_trained/h.npy').astype('float32');n=len(P);assert n==len(H)==22000;torch.set_num_threads(4);device='cuda:2';neighbors=30
print('forming composition and communication nearest neighbors',flush=True)
def knn(x,k):
 x=torch.tensor(x,device=device);norm=(x*x).sum(1);ix=[];ds=[]
 for i in range(0,len(x),512):
  d=(norm[i:i+512,None]+norm[None,:]-2*x[i:i+512]@x.T).clamp_min_(0);rows=torch.arange(i,min(i+512,len(x)),device=device);d[torch.arange(len(rows),device=device),rows]=float('inf');v,j=torch.topk(d,k,dim=1,largest=False,sorted=True);ix.append(j.cpu().numpy());ds.append(v.sqrt().cpu().numpy())
 return np.concatenate(ix),np.concatenate(ds)
pi,pdistances=knn(P,neighbors);hi,hd=knn(H,neighbors);ps=np.maximum(pdistances[:,-1],1e-5);hs=np.maximum(hd[:,-1],1e-5);src=np.repeat(np.arange(n),2*neighbors);dst=np.column_stack([pi,hi]).reshape(-1);valid=src!=dst;src,dst=src[valid],dst[valid]
# Duplicates represent the same pair, not extra evidence.
pairs=np.unique(src.astype('int64')*n+dst);src=pairs//n;dst=pairs%n
pdist=np.linalg.norm(P[src]-P[dst],axis=1);hdist=np.linalg.norm(H[src]-H[dst],axis=1);dp=pdist/np.sqrt(ps[src]*ps[dst]);dh=hdist/np.sqrt(hs[src]*hs[dst]);print('graph candidates',len(src),'median local scales',np.median(ps),np.median(hs),flush=True)
# Both views are pre-calibrated via local neighbor radii. Conditional weighting
# decreases communication influence smoothly as composition distance increases.
weights={'balanced':np.exp(-.5*(dp*dp+dh*dh)), 'composition_conditional':np.exp(-(dp*dp+np.exp(-.5*dp*dp)*dh*dh)/2)}
results={};graphs={}
for name,weight in weights.items():
 g=sparse.coo_matrix((weight,(src,dst)),shape=(n,n)).tocsr();g=(g+g.T)*.5;g.eliminate_zeros();graphs[name]=g
 components,_=connected_components(g,directed=False);degree=np.asarray(g.sum(1)).ravel();inv=1/np.sqrt(np.maximum(degree,1e-12));norm=sparse.diags(inv)@g@sparse.diags(inv);print(name,'components',components,'spectral eigenvectors...',flush=True)
 vals,vec=eigsh(norm,k=7,which='LA',tol=1e-4,maxiter=3000,v0=np.random.default_rng(112).normal(size=n));order=np.argsort(vals)[::-1];vec=vec[:,order[:5]];vec/=np.maximum(np.linalg.norm(vec,axis=1,keepdims=True),1e-12);labels=KMeans(n_clusters=5,n_init=20,random_state=40700).fit_predict(vec);results[name]=labels;np.save(OUT/(name+'_labels.npy'),labels);np.save(OUT/(name+'_embedding.npy'),vec);sparse.save_npz(OUT/(name+'_graph.npz'),g);print(name,'clusters',np.bincount(labels),'eigenvalues',vals[order],flush=True)
np.save(OUT/'composition_only_labels.npy',KMeans(n_clusters=5,n_init=10,random_state=40700).fit_predict(P));
# All algorithm decisions are complete before evaluation truth is read.
a=ad.read_h5ad(D/'tissue.h5ad',backed='r');truth=np.array([0,1,2,3,4,4])[a.obs.design_region_id.to_numpy()];xy=np.asarray(a.obsm['spatial']);a.file.close();old=np.load(W/'outputs/v3_fixed_total_k_two_simulations/multifocal.npz')['labels'];comp=np.load(OUT/'composition_only_labels.npy');runs={'composition_only':comp,'previous_v3':old,**results};names=['Background','Shared core','Interface A','Interface B','Infiltrated tumor'];colors=np.array(['#d5d9de','#e78291','#d59a38','#946db3','#46939a']);ab=np.isin(truth,[2,3]);fg=truth!=0;metrics={}
for name,labels in runs.items():metrics[name]={'global_ARI':float(adjusted_rand_score(truth,labels)),'tumor_ARI':float(adjusted_rand_score(truth[fg],labels[fg])),'AB_ARI':float(adjusted_rand_score(truth[ab],labels[ab])),'truth_by_cluster':pd.crosstab(truth,labels).to_numpy().tolist()}
# Weighted graph fractions by simulation state diagnose which relations dominate.
for name,g in graphs.items():
 table=np.zeros((5,5));
 for i in range(5):
  rows=np.flatnonzero(truth==i);sg=g[rows];den=float(sg.sum());
  for j in range(5):table[i,j]=float(sg[:,truth==j].sum())/max(den,1e-12)
 metrics[name]['graph_weighted_neighbor_truth_fractions']=table.tolist()
(OUT/'metrics.json').write_text(json.dumps(metrics,indent=2));print('metrics',json.dumps(metrics),flush=True)
fig,axs=plt.subplots(1,5,figsize=(15,3.7),layout='constrained');order=np.random.default_rng(1).permutation(n)
for ax,(name,lab) in zip(axs,[('Ground truth',truth),*runs.items()]):
 if name!='Ground truth':
  cm=confusion_matrix(truth,lab,labels=range(5));r,c=linear_sum_assignment(-cm);mapping=dict(zip(c,r));lab=np.array([mapping.get(v,v) for v in lab]);title=f"{name.replace('_',' ')}\nARI={metrics[name]['global_ARI']:.3f} · A/B={metrics[name]['AB_ARI']:.3f}"
 else:title=name
 ax.scatter(xy[order,0],xy[order,1],c=colors[lab[order]],s=2.2,lw=0,rasterized=True);ax.set_aspect('equal');ax.axis('off');ax.set_title(title,fontsize=9)
fig.legend(handles=[Line2D([],[],marker='o',lw=0,color=c,label=s,markersize=5) for c,s in zip(colors,names)],loc='outside lower center',ncol=5,frameon=False,fontsize=8);fig.savefig(OUT/'joint_graph_comparison.png',dpi=110,bbox_inches='tight');plt.close(fig);print('DONE',OUT,flush=True)
