"""Exploratory comparison on two saved PF specimens; no network retraining.

The reference procedure is an original fixed-reference distance baseline, not
Harmonics or a formal disease-specific niche test. Threshold is the 95th
percentile of held-out control distances and is not calibrated across donors.
"""
import json
import os
from pathlib import Path
os.environ.setdefault('OMP_NUM_THREADS', '4')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '4')
import anndata as ad
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import nbformat
import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import pairwise_distances_argmin_min
from sklearn.model_selection import train_test_split

WORK = Path('/home/nas3/biod/xueshuailin/CCC_Phe/workspace')
SRC = WORK / 'outputs/harmonics_pf/joint_two_v3'
OUT = Path('/data1/xueshuailin/CCC_Phe/workspace/outputs/harmonics_pf/reference_comparison')
OUT.mkdir(parents=True, exist_ok=True)
NB = Path('/home/xueshuailin/CCC_Phe/Harmonics_Pulmonary_Fibrosis_V3.ipynb')
P = Path(json.loads((SRC/'target_cache.json').read_text())['folder'])
adata = ad.read_h5ad(WORK/'data/harmonics_pf/joint_two_native_v3_input.h5ad')
pred = np.load(SRC/'predictions.npz')
assert np.array_equal(adata.obs_names.astype(str), pred['cell_ids'].astype(str))
sample = pred['sample_id']; joint = pred['labels']; xy = adata.obsm['spatial']
p = np.sqrt(np.load(P/'composition.npy', mmap_mode='r').astype('float32'))
h = np.load(SRC/'h.npy', mmap_mode='r').astype('float32')
assert len(p) == len(h) == len(adata) and np.isfinite(p).all() and np.isfinite(h).all()
control, disease = np.flatnonzero(sample==0), np.flatnonzero(sample==1)
train, hold = train_test_split(control, test_size=.25, random_state=42)
def scale(x):
    center = x[train].mean(0); size = np.sqrt(np.mean(np.sum((x[train]-center)**2, axis=1)))
    return (x-center)/max(float(size),1e-8)
x = np.concatenate((scale(p),scale(h)), axis=1).astype('float32')
model = MiniBatchKMeans(n_clusters=4, n_init=10, batch_size=2048, random_state=42).fit(x[train])
ref, dist = pairwise_distances_argmin_min(x,model.cluster_centers_)
cutoff = float(np.quantile(dist[hold], .95))
novel = (sample==1) & (dist>cutoff)
if novel.sum() < 100:
    raise ValueError('Too few disease cells beyond control threshold for exploratory subclustering')
novel_model = MiniBatchKMeans(n_clusters=4, n_init=10,batch_size=2048,random_state=42).fit(x[novel])
reference = ref.copy(); reference[novel] = 4+novel_model.predict(x[novel])
np.savez_compressed(OUT/'results.npz', cell_ids=pred['cell_ids'], sample_id=sample, joint=joint, reference=reference, novel_distance=dist, cutoff=cutoff)
info = {'samples':['THD0008','VUILD106MA'],'representation':'existing jointly trained V3 h and composition; held-out cells are excluded only from reference prototypes','k_reference':4,'novel_subclusters':4,'threshold':'95th percentile of held-out control cell distances','holdout_fraction':.25,'disease_beyond_threshold_fraction':float(novel[disease].mean()),'control_holdout_beyond_threshold_fraction':float((dist[hold]>cutoff).mean()),'inference':'exploratory; one specimen per condition; unknown sample/depth effects; no donor-level inference'}
(OUT/'protocol.json').write_text(json.dumps(info,indent=2))
tab = pd.crosstab([pd.Series(sample,name='sample'),pd.Series(reference,name='reference_niche')],adata.obs.cell_type.to_numpy(),normalize='index')
tab.to_csv(OUT/'cell_type_fractions.csv')
print('Reference novelty fraction', info['disease_beyond_threshold_fraction'],flush=True)

plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,'axes.spines.right':False,'figure.facecolor':'white'})
palette = np.array(['#407e9b','#e49c63','#68a899','#a488b7','#b86068','#d2b04e','#668c60','#8e7081'])
fig,axs=plt.subplots(2,2,figsize=(10.4,9),layout='constrained')
for row,(s,title) in enumerate(((0,'Control · THD0008'),(1,'Pulmonary fibrosis · VUILD106MA'))):
    ids=np.flatnonzero(sample==s)
    for col,(lab,name) in enumerate(((joint,'Joint V3 · K=8'),(reference,'Control-reference approach'))):
        ax=axs[row,col]; ax.scatter(xy[ids,0],xy[ids,1],c=palette[lab[ids]],s=.28,linewidths=0,rasterized=True);ax.set_aspect('equal');ax.invert_yaxis();ax.axis('off');ax.set_title(f'{title}\n{name}',fontsize=10,pad=6)
handles=[Line2D([],[],marker='o',linestyle='',color=palette[i],markersize=5,label=(f'Ref {i+1}' if i<4 else f'Candidate {i-3}')) for i in range(8)]
fig.legend(handles=handles,loc='outside lower center',ncol=8,frameon=False,fontsize=8,title='Reference approach only; joint colors have independent IDs',title_fontsize=8)
fig.savefig(OUT/'spatial_comparison.png',dpi=140,bbox_inches='tight');plt.close(fig)

nb=nbformat.read(NB,as_version=4)
source='''# Existing joint V3 vs control-reference exploration; reads cached results.\nfrom IPython.display import display, Image\nimport json\nfrom pathlib import Path\np = Path('/data1/xueshuailin/CCC_Phe/workspace/outputs/harmonics_pf/reference_comparison')\ndisplay(Image(filename=str(p / 'spatial_comparison.png'), width=1000))\nprint(json.loads((p / 'protocol.json').read_text()))\n'''
tag='pf_reference_comparison_20261116'
if any(tag in c.get('metadata',{}).get('tags',[]) for c in nb.cells):
    raise RuntimeError('Comparison cell already exists; refusing to overwrite user edits')
cell=nbformat.v4.new_code_cell(source,metadata={'tags':[tag]})
with (OUT/'spatial_comparison.png').open('rb') as f:
    import base64
    png=base64.b64encode(f.read()).decode()
cell.outputs=[nbformat.v4.new_output('display_data',data={'image/png':png},metadata={}),nbformat.v4.new_output('stream',name='stdout',text=json.dumps(info,ensure_ascii=False,indent=2)+'\n')]
nb.cells.append(cell); nbformat.write(nb,NB)
print('Notebook updated:',NB,flush=True)
