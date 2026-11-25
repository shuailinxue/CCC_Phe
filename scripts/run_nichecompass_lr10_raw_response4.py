"""LR10 raw CCC sensitivity: retain response signatures with >=4% weight coverage."""
from pathlib import Path
import os,sys,json,subprocess,traceback
import numpy as np,pandas as pd,anndata as ad
from sklearn.metrics import adjusted_rand_score
R=Path('/home/xueshuailin/CCC_Phe');sys.path[:0]=[str(R/'src'),str(R/'scripts')]
import run_nichecompass_feature_sensitivity as s
from run_v2_global_communication_ablation import standardize
W=s.W;O=s.ROOT/'lr10_raw_response4';O.mkdir(parents=True,exist_ok=True);src=s.ROOT/'lr10_raw_unfiltered_response'
try:
 if not (O/'results.json').exists():
  z=np.load(src/'inputs.npz');pack={k:z[k] for k in z.files};E=np.load(s.NEW/'response.npy');M=np.load(s.NEW/'response_mask.npy');_,_,keepidx=standardize(E,M);info=pd.read_csv(s.NEW/'response_support.csv').iloc[keepidx];coverage=pd.read_csv(s.NEW/'response_coverage.csv').set_index('program');keep=info.program.map(coverage.absolute_weight_fraction).to_numpy()>=.04
  for key in ['x1','m1']:pack[key]=pack[key][:,keep]
  pack['p1']=pack['p1'][keep];np.savez_compressed(O/'inputs.npz',**pack);print('response retained',int(keep.sum()),'of',len(keep),'programs',sorted(info.program[keep].unique().tolist()),flush=True)
  runner=(src/'runner.py').read_text().replace(str(src),str(O));(O/'runner.py').write_text(runner);subprocess.run([sys.executable,'-u',str(O/'runner.py')],check=True,env={**os.environ,'OMP_NUM_THREADS':'3','OPENBLAS_NUM_THREADS':'1'})
 h=np.load(O/'h.npy');coarse=np.load(O/'coarse.npy');cc=np.load(O/'coarse_centers.npy');t=Path(json.loads((s.NEW/'target_cache.json').read_text())['folder']);x=np.sqrt(np.maximum(np.load(t/'composition.npy'),0));dc=np.linalg.norm(x[:,None,:]-cc,axis=2);hc=np.stack([h[coarse==j].mean(0) for j in range(8)]);dh=np.linalg.norm(h[:,None,:]-hc,axis=2);ids=np.arange(len(x));dc0=dc/np.median(dc[ids,coarse]);dh0=dh/np.median(dh[ids,coarse]);order=np.argsort(dc,axis=1);d=np.full_like(dc,np.inf);d[ids[:,None],order[:,:4]]=dc0[ids[:,None],order[:,:4]]+2*dh0[ids[:,None],order[:,:4]];pred=np.argmin(d,axis=1);np.save(O/'joint_distance_top4.npy',pred)
 a=ad.read_h5ad(W/'data/nichecompass_simulation/sim1_1105genes_10000locs_strongincrements.h5ad',backed='r');truth=pd.Categorical(a.obs.niche_types,categories=[f'Niche {i}' for i in range(1,9)]).codes;a.file.close();ari=float(adjusted_rand_score(truth,pred));(O/'top4_result.json').write_text(json.dumps({'ARI':ari,'lr_threshold':.10,'response_absolute_weight_coverage_threshold':.04,'candidate_classes':4,'truth_only_evaluation':True},indent=2));print('top4 ARI',ari,flush=True)
 import nbformat
 from nbclient import NotebookClient
 p=R/'NicheCompass_Simulation_V2.ipynb';nb=nbformat.read(p,4);source=f'''# 响应签名覆盖率 ≥4%：LR 10%、原始 CCC、K=8
%matplotlib inline
from pathlib import Path
import numpy as np,pandas as pd,anndata as ad,matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.metrics import adjusted_rand_score,confusion_matrix
from scipy.optimize import linear_sum_assignment
out=Path('{O}');base=Path('{src}');a=ad.read_h5ad('{W}/data/nichecompass_simulation/sim1_1105genes_10000locs_strongincrements.h5ad',backed='r');xy=np.asarray(a.obsm['spatial']);truth=pd.Categorical(a.obs.niche_types,categories=[f'Niche {{i}}' for i in range(1,9)]).codes;a.file.close();coarse=np.load(out/'coarse.npy');pred=np.load(out/'joint_distance_top4.npy');basepred=np.load(base/'joint_distance_top4.npy');r,c=linear_sum_assignment(-confusion_matrix(truth,coarse,labels=range(8)));mapping=dict(zip(c,r));colors=plt.get_cmap('tab10')(np.arange(8));order=np.random.default_rng(2026).permutation(len(truth));fig,axes=plt.subplots(1,3,figsize=(10.5,7),layout='constrained')
for ax,labels,title in [(axes[0],truth,'Ground truth'),(axes[1],np.array([mapping[i] for i in basepred]),f'All responses\\nARI = {{adjusted_rand_score(truth,basepred):.3f}}'),(axes[2],np.array([mapping[i] for i in pred]),f'Response coverage ≥4%\\nARI = {{adjusted_rand_score(truth,pred):.3f}}')]:
 ax.scatter(xy[order,0],xy[order,1],c=colors[labels[order]],s=4,lw=0,rasterized=True);ax.set_aspect('equal');ax.axis('off');ax.set_title(title,fontsize=11)
fig.legend(handles=[Line2D([],[],marker='o',lw=0,color=colors[i],label=f'Niche {{i+1}}',markersize=5) for i in range(8)],loc='outside lower center',ncol=4,frameon=False,fontsize=8);plt.show()
''';one=nbformat.v4.new_notebook(cells=[nbformat.v4.new_code_cell(source)],metadata=nb.metadata);NotebookClient(one,timeout=300,kernel_name='cccphe').execute(cwd=str(R));assert any('image/png' in o.get('data',{}) for o in one.cells[0].outputs);latest=nbformat.read(p,4)
 if not any(c.cell_type=='code' and c.source.startswith('# 响应签名覆盖率 ≥4%') for c in latest.cells):latest.cells.append(one.cells[0]);nbformat.write(latest,p)
 print('notebook updated',flush=True)
except BaseException:(O/'failure.txt').write_text(traceback.format_exc());traceback.print_exc();raise
