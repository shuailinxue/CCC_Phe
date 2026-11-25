"""Composition-only K=8 diagnostic for the published simulation."""
from pathlib import Path
import json
import numpy as np,pandas as pd,anndata as ad,matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score,normalized_mutual_info_score,confusion_matrix
from scipy.optimize import linear_sum_assignment
W=Path('/data1/xueshuailin/CCC_Phe/workspace');p=W/'outputs/nichecompass_simulation/v2';target=Path(json.loads((p/'target_cache.json').read_text())['folder']);comp=np.load(target/'composition.npy');a=ad.read_h5ad(W/'data/nichecompass_simulation/sim1_1105genes_10000locs_strongincrements.h5ad',backed='r');xy=np.asarray(a.obsm['spatial']);truth=pd.Categorical(a.obs.niche_types,categories=[f'Niche {i}' for i in range(1,9)]).codes;a.file.close();assert len(comp)==len(truth) and (truth>=0).all();labels=KMeans(n_clusters=8,n_init=10,random_state=40700).fit_predict(np.sqrt(np.maximum(comp,0)));ari=adjusted_rand_score(truth,labels);nmi=normalized_mutual_info_score(truth,labels);cont=confusion_matrix(truth,labels,labels=np.arange(8));i,j=linear_sum_assignment(-cont);matched=np.array([dict(zip(j,i))[k] for k in labels]);np.save(p/'composition_only_K8.npy',labels);pd.DataFrame(cont,index=[f'Niche {k+1}' for k in range(8)],columns=[f'C{k+1}' for k in range(8)]).to_csv(p/'composition_only_K8_contingency.csv');result={'K':8,'features':'sqrt local cell composition, same 30-member distance-weighted physical neighborhood as V2','algorithm':'KMeans, n_init=10, seed=40700','truth_used':'evaluation and display color matching only','ARI':float(ari),'NMI':float(nmi)};(p/'composition_only_K8_metrics.json').write_text(json.dumps(result,indent=2)+'\n')
colors=plt.get_cmap('tab10')(np.arange(8));fig,axes=plt.subplots(1,2,figsize=(7.4,7),layout='constrained');order=np.random.default_rng(2026).permutation(len(truth))
for ax,v,title in [(axes[0],truth,'Ground truth · 8 niches'),(axes[1],matched,f'Composition only · K=8\nARI={ari:.3f} · NMI={nmi:.3f}')]:
 ax.scatter(xy[order,0],xy[order,1],c=colors[v[order]],s=4,lw=0,rasterized=True);ax.set_aspect('equal');ax.axis('off');ax.set_title(title,fontsize=11)
fig.legend(handles=[Line2D([],[],marker='o',lw=0,color=colors[i],label=f'Niche {i+1}',markersize=5) for i in range(8)],loc='outside lower center',ncol=4,frameon=False,fontsize=8);fig.savefig(p/'composition_only_K8.png',dpi=170,bbox_inches='tight');plt.close(fig);print(json.dumps(result,indent=2));print(pd.DataFrame(cont).to_string(index=False,header=False))
