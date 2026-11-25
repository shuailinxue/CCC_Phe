"""Compare STCase and our predictions on identical center-cell types."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import anndata as ad
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score

ROOT=Path('/data1/xueshuailin/CCC_Phe/workspace')
DATA=ROOT/'data/tumor_immune_interface_v2_epithelial_majority'
V2=ROOT/'outputs/tumor_immune_interface/v2_cellphonedb_v5'
OUT=ROOT/'outputs/tumor_immune_interface/v2_baseline_k5/stcase_fair'
NAMES=['Background','Shared tumor core','Interface A','Interface B','Infiltrated tumor']
COLORS=['#dcdfe3','#e98088','#dc9c3b','#8e69b1','#398d93']

def matched_colors(reference, labels):
 """Use truth solely for post-fit display color matching; preserve assignments."""
 values=np.unique(labels);table=np.array([[np.sum((reference==i)&(labels==j)) for j in values] for i in range(5)])
 rows,cols=linear_sum_assignment(-table);lookup={values[c]:int(r) for r,c in zip(rows,cols)}
 return np.asarray([lookup[z] for z in labels])

def make_plot(types=('Cancer epithelial','T cell')):
 a=ad.read_h5ad(DATA/'tissue.h5ad',backed='r');xy=np.asarray(a.obsm['spatial']);cts=a.obs.cell_type.astype(str).to_numpy();truth=np.array([0,1,2,3,4,4])[a.obs.design_region_id.to_numpy()];a.file.close()
 folder=Path(json.loads((DATA/'target_cache.json').read_text())['folder']);composition=np.load(folder/'composition.npy');coarse=KMeans(n_clusters=4,n_init=10,random_state=40700).fit_predict(np.sqrt(np.maximum(composition,0)));means=np.stack([composition[coarse==k].mean(0) for k in range(4)]);chosen=np.flatnonzero((means[:,0]>.3)&(means[:,1]>.1)&(means[:,3]>.10));saved=np.load(V2/'single_tissue_predictions.npz');assert len(chosen)==1 and np.array_equal(saved['ids'],np.flatnonzero(coarse==chosen[0]));ours=coarse.copy();ours[saved['ids']]=4+saved['network']
 scores=pd.read_csv(OUT/'per_cell_type_metrics.csv').set_index('cell_type');fig,axes=plt.subplots(len(types),3,figsize=(12,4.2*len(types)),squeeze=False,layout='constrained')
 for r,name in enumerate(types):
  sub=OUT/('type_'+name.replace(' ','_'));ids=np.load(sub/'ids.npy');pred=np.load(sub/'labels.npy');assert np.array_equal(ids,np.flatnonzero(cts==name));labels=[truth[ids],matched_colors(truth[ids],ours[ids]),matched_colors(truth[ids],pred)];ari=[None,adjusted_rand_score(truth[ids],ours[ids]),adjusted_rand_score(truth[ids],pred)];assert np.isclose(ari[2],scores.loc[name,'STCase_ARI'])
  for col,ax in enumerate(axes[r]):
   ax.scatter(xy[:,0],xy[:,1],s=.55,c='#eceef0',linewidths=0,rasterized=True)
   for k,color in enumerate(COLORS):
    keep=ids[labels[col]==k];ax.scatter(xy[keep,0],xy[keep,1],s=7,c=color,linewidths=0,rasterized=True)
   title=['Ground truth','Our method','STCase · sparse adaptation'][col];ax.set_title(title if col==0 else f'{title} · ARI={ari[col]:.3f}',fontsize=11);ax.set_aspect('equal');ax.axis('off')
  axes[r,0].text(-.04,.5,f'{name}\n{len(ids):,} cells',transform=axes[r,0].transAxes,rotation=90,va='center',ha='right',fontsize=11,fontweight='bold')
 fig.legend(handles=[Line2D([],[],marker='o',lw=0,color=c,markersize=6,label=n) for n,c in zip(NAMES,COLORS)],loc='outside lower center',ncol=5,frameon=False,fontsize=9)
 fig.suptitle('Comparison on identical cells · K=5 for STCase\nPredicted colors matched to truth for display only',fontsize=13)
 return fig

if __name__=='__main__':
 dest=OUT/'per_type_spatial_comparison.png';fig=make_plot();fig.savefig(dest,dpi=150,bbox_inches='tight');print(dest)
