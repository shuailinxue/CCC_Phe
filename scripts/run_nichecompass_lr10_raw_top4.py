"""LR10, raw CCC, unfiltered response, pre-existing top4 joint-distance rule."""
from pathlib import Path
import sys,json,traceback
import numpy as np,pandas as pd,anndata as ad
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import confusion_matrix,adjusted_rand_score
R=Path('/home/xueshuailin/CCC_Phe');sys.path[:0]=[str(R/'src'),str(R/'scripts')]
import run_nichecompass_feature_sensitivity as s
O=s.ROOT/'lr10_raw_unfiltered_response';W=s.W
try:
 s.train_one('lr10_raw_unfiltered_response',s.NEW,False,True)
 h=np.load(O/'h.npy');coarse=np.load(O/'coarse.npy');cc=np.load(O/'coarse_centers.npy');t=Path(json.loads((s.NEW/'target_cache.json').read_text())['folder']);x=np.sqrt(np.maximum(np.load(t/'composition.npy'),0));dc=np.linalg.norm(x[:,None,:]-cc,axis=2);hc=np.stack([h[coarse==j].mean(0) for j in range(8)]);dh=np.linalg.norm(h[:,None,:]-hc,axis=2);ids=np.arange(len(x));dc0=dc/np.median(dc[ids,coarse]);dh0=dh/np.median(dh[ids,coarse]);order=np.argsort(dc,axis=1);d=np.full_like(dc,np.inf);d[ids[:,None],order[:,:4]]=dc0[ids[:,None],order[:,:4]]+2*dh0[ids[:,None],order[:,:4]];pred=np.argmin(d,axis=1);np.save(O/'joint_distance_top4.npy',pred)
 a=ad.read_h5ad(W/'data/nichecompass_simulation/sim1_1105genes_10000locs_strongincrements.h5ad',backed='r');truth=pd.Categorical(a.obs.niche_types,categories=[f'Niche {i}' for i in range(1,9)]).codes;a.file.close();ari=float(adjusted_rand_score(truth,pred));(O/'top4_result.json').write_text(json.dumps({'ARI':ari,'lr_detection_threshold':.10,'CCC':'raw unmapped','response_coverage_filter':None,'k':8,'candidate_classes':4,'distance_weight_communication':2,'truth_used_only_evaluation':True},indent=2));print('top4 ARI',ari,flush=True)
 import nbformat
 from nbclient import NotebookClient
 p=R/'NicheCompass_Simulation_V2.ipynb';nb=nbformat.read(p,4);source=f'''# LR 10% + 原始 CCC：4 类候选联合修正，K=8
%matplotlib inline
from pathlib import Path
import numpy as np,pandas as pd,anndata as ad,matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.metrics import adjusted_rand_score,confusion_matrix
from scipy.optimize import linear_sum_assignment
out=Path('{O}');a=ad.read_h5ad('{W}/data/nichecompass_simulation/sim1_1105genes_10000locs_strongincrements.h5ad',backed='r');xy=np.asarray(a.obsm['spatial']);truth=pd.Categorical(a.obs.niche_types,categories=[f'Niche {{i}}' for i in range(1,9)]).codes;a.file.close();coarse=np.load(out/'coarse.npy');pred=np.load(out/'joint_distance_top4.npy');r,c=linear_sum_assignment(-confusion_matrix(truth,coarse,labels=range(8)));mapping=dict(zip(c,r));colors=plt.get_cmap('tab10')(np.arange(8));order=np.random.default_rng(2026).permutation(len(truth));fig,axes=plt.subplots(1,3,figsize=(10.5,7),layout='constrained')
for ax,labels,title in [(axes[0],truth,'Ground truth'),(axes[1],np.array([mapping[i] for i in coarse]),f'Composition · K=8\\nARI = {{adjusted_rand_score(truth,coarse):.3f}}'),(axes[2],np.array([mapping[i] for i in pred]),f'LR 10% + raw CCC · K=8\\nARI = {{adjusted_rand_score(truth,pred):.3f}}')]:
 ax.scatter(xy[order,0],xy[order,1],c=colors[labels[order]],s=4,lw=0,rasterized=True);ax.set_aspect('equal');ax.axis('off');ax.set_title(title,fontsize=11)
fig.legend(handles=[Line2D([],[],marker='o',lw=0,color=colors[i],label=f'Niche {{i+1}}',markersize=5) for i in range(8)],loc='outside lower center',ncol=4,frameon=False,fontsize=8);plt.show()
''';one=nbformat.v4.new_notebook(cells=[nbformat.v4.new_code_cell(source)],metadata=nb.metadata);NotebookClient(one,timeout=300,kernel_name='cccphe').execute(cwd=str(R));assert any('image/png' in o.get('data',{}) for o in one.cells[0].outputs);latest=nbformat.read(p,4);idx=[i for i,c in enumerate(latest.cells) if c.cell_type=='code' and c.source.startswith('# CCC 与响应特征筛选、原始 CCC 对照')];
 if len(idx)!=1:raise RuntimeError('Target cell missing or changed; preserving notebook')
 latest.cells[idx[0]]=one.cells[0];nbformat.write(latest,p);print('notebook cell updated',flush=True)
except BaseException:
 O.mkdir(parents=True,exist_ok=True);(O/'top4_failure.txt').write_text(traceback.format_exc());traceback.print_exc();raise
