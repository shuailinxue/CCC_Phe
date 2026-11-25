"""Exploratory joint composition and communication manifold hierarchy.

Architecture and numerical settings are fixed before evaluating the other
three datasets. Tumor simulation was inspected during development; no claim of
independent generalization. No annotations enter fitting or merging.
"""
import os,sys,json,time
os.environ.setdefault('NUMBA_NUM_THREADS','4');os.environ.setdefault('OMP_NUM_THREADS','4');os.environ.setdefault('OPENBLAS_NUM_THREADS','4')
from pathlib import Path
import numpy as np,pandas as pd,anndata as ad
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score,confusion_matrix
from scipy.spatial.distance import cdist
from scipy.spatial import cKDTree
from scipy.optimize import linear_sum_assignment
import umap,matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
W=Path('/data1/xueshuailin/CCC_Phe/workspace');OUT=W/'outputs/v3_joint_manifold_tree_four';OUT.mkdir(parents=True,exist_ok=True);k0=15;weight=.5;seed=40700
M=W/'data/tumor_immune_interface_v2_epithelial_majority';N=W/'data/nichecompass_simulation/sim1_1105genes_10000locs_strongincrements.h5ad';S=W/'data/harmonics_fig2/starmap/analysis_input.h5ad';O=W/'outputs/harmonics_fig2';R=W/'data/harmonics_fig2/merfish_scaled/donor12_slice0_analysis_input.h5ad';B=W/'outputs/tumor_immune_interface';npath=Path(json.loads((W/'outputs/nichecompass_simulation/v2/target_cache.json').read_text())['folder']);mpath=Path(json.loads((M/'target_cache.json').read_text())['folder']);spath=Path(json.loads((O/'starmap_scaled/target_cache.json').read_text())['folder']);rpath=Path(json.loads((O/'merfish_scaled/target_cache.json').read_text())['folder']);
cases={'tumor':(mpath,B/'v3_full_tissue_trained/h.npy',M/'tissue.h5ad',5),'nichecompass':(npath,W/'outputs/nichecompass_simulation/v3_joint_hierarchy/h.npy',N,8),'starmap':(spath,O/'starmap_scaled/fit_min25/result.npz',S,7),'merfish':(rpath,O/'merfish_scaled/fit_min25/result.npz',R,7)}
for name,(target,hpath,apath,k) in cases.items():
 dest=OUT/name;dest.mkdir(exist_ok=True);p=np.sqrt(np.maximum(np.load(target/'composition.npy'),0));h0=np.load(hpath);h=h0['h'] if isinstance(h0,np.lib.npyio.NpzFile) else h0;p=(p-p.mean(0))/max(np.sqrt(np.mean(np.sum((p-p.mean(0))**2,axis=1))),1e-8);u=dest/'communication_manifold.npy'
 if not u.exists():
  if name=='tumor':z=np.load(B/'v3_global_vs_coarse_h_umap/global_umap.npy');print(name,'using verified existing UMAP',flush=True)
  else:print(name,'fit communication UMAP',len(h),flush=True);z=umap.UMAP(n_neighbors=30,min_dist=.2,metric='euclidean',random_state=42,n_jobs=1).fit_transform(h)
  np.save(u,z)
 z=np.load(u);z=(z-z.mean(0))/max(np.sqrt(np.mean(np.sum((z-z.mean(0))**2,axis=1))),1e-8);X=np.column_stack([p,weight*z]).astype('float32');print(name,'initial joint centers',k0,flush=True);km=KMeans(n_clusters=k0,n_init=8,random_state=seed).fit(X);cent=km.cluster_centers_;trace=[]
 while len(cent)>k:
  dis=cdist(cent,cent,'sqeuclidean');np.fill_diagonal(dis,np.inf);i,j=np.unravel_index(dis.argmin(),dis.shape);score=float(dis[i,j]);cent[i]=(cent[i]+cent[j])/2;cent=np.delete(cent,j,axis=0);km=KMeans(n_clusters=len(cent),init=cent,n_init=1,random_state=seed,max_iter=100,tol=1e-4).fit(X);cent=km.cluster_centers_;trace.append({'to_k':len(cent),'merged_centers':[int(i),int(j)],'cost':score,'iterations':int(km.n_iter_)});print(name,'K',len(cent),flush=True)
 np.save(dest/'labels.npy',km.labels_);(dest/'trace.json').write_text(json.dumps(trace,indent=2))
# Evaluate after all four methods have finished.
report={};plots=[]
for name,(target,hpath,apath,k) in cases.items():
 a=ad.read_h5ad(apath,backed='r');xy=np.asarray(a.obsm['spatial']);labels=np.load(OUT/name/'labels.npy');mask=np.ones(len(labels),bool)
 if name=='tumor':truth=np.array([0,1,2,3,4,4])[a.obs.design_region_id.to_numpy()]
 elif name=='nichecompass':truth=pd.Categorical(a.obs.niche_types,categories=[f'Niche {i}' for i in range(1,9)]).codes
 elif name=='merfish':truth=pd.Categorical(a.obs.benchmark_region).codes
 else:
  original=ad.read_h5ad(W/'data/harmonics_fig2/starmap/stDyer_mouse_cortex.h5ad');distance,idx=cKDTree(np.c_[original.obs.X,-original.obs.Y]).query(xy);mask=distance<.01;truth=pd.Categorical(original.obs.label.iloc[idx].astype(str).str.upper()).codes
 a.file.close();scores={'ARI':float(adjusted_rand_score(truth[mask],labels[mask])),'n':int(mask.sum()),'K':k}
 if name=='tumor':
  ab=np.isin(truth,[2,3]);fg=truth!=0;scores.update(AB_ARI=float(adjusted_rand_score(truth[ab],labels[ab])),tumor_ARI=float(adjusted_rand_score(truth[fg],labels[fg])))
 report[name]=scores;plots.append((name,xy,truth,labels,mask,k));print('RESULT',name,json.dumps(scores),flush=True)
(OUT/'metrics.json').write_text(json.dumps({'parameter_note':'M=15, communication multiplier .5 fixed after tumor exploration; other datasets preliminary out-of-tumor checks','truth_used_for_fitting':False,'results':report},indent=2))
fig,axes=plt.subplots(4,2,figsize=(7.5,13),layout='constrained')
for row,(name,xy,t,l,mask,k) in enumerate(plots):
 cm=confusion_matrix(t[mask],l[mask],labels=range(k));i,j=linear_sum_assignment(-cm);mapping=dict(zip(j,i));aligned=np.array([mapping.get(v,0) for v in l]);cmap=plt.get_cmap('tab10');order=np.random.default_rng(2).permutation(len(t));order=order[mask[order]]
 for ax,v,title in [(axes[row,0],t,name+' · annotation'),(axes[row,1],aligned,'joint hierarchy · ARI %.3f'%report[name]['ARI'])]:ax.scatter(xy[order,0],xy[order,1],c=v[order],vmin=0,vmax=9,cmap=cmap,s=2,lw=0,rasterized=True);ax.set_aspect('equal');ax.axis('off');ax.set_title(title,fontsize=9)
fig.savefig(OUT/'comparison.png',dpi=100,bbox_inches='tight');plt.close(fig);print('DONE',flush=True)
