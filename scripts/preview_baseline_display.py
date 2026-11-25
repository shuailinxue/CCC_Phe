"""Display-only baseline preview: fixed randomized point order, all cells shown."""
from pathlib import Path
import numpy as np,anndata as ad
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import adjusted_rand_score
W=Path('/data1/xueshuailin/CCC_Phe/workspace');S=W/'data/tumor_immune_interface_v2_epithelial_majority';O=W/'outputs/tumor_immune_interface/v2_baseline_k5'
a=ad.read_h5ad(S/'tissue.h5ad',backed='r');xy=np.asarray(a.obsm['spatial']);truth=np.array([0,1,2,3,4,4])[a.obs.design_region_id.to_numpy()];a.file.close();order=np.random.default_rng(2026).permutation(len(truth));fg=truth!=0
methods=[('GraphST','graphst'),('STAGATE','stagate'),('CellCharter','cellcharter'),('scNiche','scniche'),('NicheCompass','nichecompass'),('SpaNiche','spaniche'),('BANKSY','banksy')]
fig,axes=plt.subplots(2,4,figsize=(19,10),layout='constrained');fig.suptitle('Baseline comparison · identical tissue · K = 5',fontsize=18)
colors=np.array(['#d9dce0','#e47781','#d99532','#8963b1','#398f94']);palette=np.array(['#2780b8','#f09039','#49a662','#9b70b9','#cf7583'])
for ax in axes.flat:ax.axis('off');ax.set_aspect('equal')
axes.flat[0].scatter(xy[order,0],xy[order,1],s=4.3,c=colors[truth[order]],lw=0,rasterized=True);axes.flat[0].set_title('Ground truth',fontsize=13)
for ax,(name,key) in zip(axes.flat[1:],methods):
 pred=np.load(O/key/'labels.npy');ari=adjusted_rand_score(truth,pred);tari=adjusted_rand_score(truth[fg],pred[fg]);ax.scatter(xy[order,0],xy[order,1],s=4.3,c=palette[pred[order]],lw=0,rasterized=True);ax.set_title(f'{name}   ARI {ari:.3f}  ·  tumor {tari:.3f}',fontsize=12)
fig.savefig(O/'comparison_display_preview.png',dpi=160,bbox_inches='tight');print(O/'comparison_display_preview.png')
