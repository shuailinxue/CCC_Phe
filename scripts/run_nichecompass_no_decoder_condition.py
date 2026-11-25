"""Ablation: same 0.931 protocol without decoder class condition."""
from pathlib import Path
import sys,os,json,subprocess,traceback
import numpy as np,pandas as pd,anndata as ad
from sklearn.metrics import adjusted_rand_score
R=Path('/home/xueshuailin/CCC_Phe');W=Path('/data1/xueshuailin/CCC_Phe/workspace');B=W/'outputs/nichecompass_simulation/feature_sensitivity';src=B/'lr10_raw_unfiltered_response';O=B/'lr10_raw_no_decoder_condition';O.mkdir(parents=True,exist_ok=True)
try:
 if not (O/'h.npy').exists():
  script=(src/'runner.py').read_text().replace(str(src),str(O)).replace("data=np.load('"+str(O)+"/inputs.npz')","data=np.load('"+str(src)+"/inputs.npz')").replace('torch.nn.Linear(72,128)','torch.nn.Linear(64,128)').replace('hc=torch.cat([h,torch.nn.functional.one_hot(cls,8).float()],1);return h,[f(hc) for f in self.dec]','return h,[f(h) for f in self.dec]');(O/'runner.py').write_text(script);subprocess.run([sys.executable,'-u',str(O/'runner.py')],check=True,env={**os.environ,'OMP_NUM_THREADS':'3','OPENBLAS_NUM_THREADS':'1'})
 h=np.load(O/'h.npy');coarse=np.load(O/'coarse.npy');cc=np.load(O/'coarse_centers.npy');target=Path(json.loads((B/'lr10_features/target_cache.json').read_text())['folder']);x=np.sqrt(np.maximum(np.load(target/'composition.npy'),0));dc=np.linalg.norm(x[:,None,:]-cc,axis=2);cent=np.stack([h[coarse==j].mean(0) for j in range(8)]);dh=np.linalg.norm(h[:,None,:]-cent,axis=2);i=np.arange(len(x));score=dc/np.median(dc[i,coarse])+2*dh/np.median(dh[i,coarse]);order=np.argsort(dc,axis=1);masked=np.full_like(score,np.inf);masked[i[:,None],order[:,:4]]=score[i[:,None],order[:,:4]];pred=np.argmin(masked,axis=1);np.save(O/'joint_distance_top4.npy',pred);a=ad.read_h5ad(W/'data/nichecompass_simulation/sim1_1105genes_10000locs_strongincrements.h5ad',backed='r');truth=pd.Categorical(a.obs.niche_types,categories=[f'Niche {i}' for i in range(1,9)]).codes;a.file.close();ari=float(adjusted_rand_score(truth,pred));(O/'top4_result.json').write_text(json.dumps({'ARI':ari,'decoder_condition':False,'within_initial_composition_class_geometry':True,'truth_used_only_for_evaluation':True},indent=2));print('no decoder condition ARI',ari,flush=True)
 import nbformat
 from nbclient import NotebookClient
 p=R/'NicheCompass_Simulation_V2.ipynb';nb=nbformat.read(p,4);source=f'''# 解码器组成条件消融：相同输入与修正规则
%matplotlib inline
from pathlib import Path
import numpy as np,pandas as pd,anndata as ad,matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.metrics import confusion_matrix,adjusted_rand_score
from scipy.optimize import linear_sum_assignment
out=Path('{O}');base=Path('{src}');a=ad.read_h5ad('{W}/data/nichecompass_simulation/sim1_1105genes_10000locs_strongincrements.h5ad',backed='r');xy=np.asarray(a.obsm['spatial']);truth=pd.Categorical(a.obs.niche_types,categories=[f'Niche {{i}}' for i in range(1,9)]).codes;a.file.close();coarse=np.load(out/'coarse.npy');r,c=linear_sum_assignment(-confusion_matrix(truth,coarse,labels=range(8)));mapping=dict(zip(c,r));p1=np.load(base/'joint_distance_top4.npy');p2=np.load(out/'joint_distance_top4.npy');colors=plt.get_cmap('tab10')(np.arange(8));order=np.random.default_rng(2026).permutation(len(truth));fig,axes=plt.subplots(1,3,figsize=(10.5,7),layout='constrained')
for ax,labels,title in [(axes[0],truth,'Ground truth'),(axes[1],np.array([mapping[i] for i in p1]),f'With condition\\nARI = {{adjusted_rand_score(truth,p1):.3f}}'),(axes[2],np.array([mapping[i] for i in p2]),f'Without condition\\nARI = {{adjusted_rand_score(truth,p2):.3f}}')]:
 ax.scatter(xy[order,0],xy[order,1],c=colors[labels[order]],s=4,lw=0,rasterized=True);ax.set_aspect('equal');ax.axis('off');ax.set_title(title,fontsize=11)
fig.legend(handles=[Line2D([],[],marker='o',lw=0,color=colors[i],label=f'Niche {{i+1}}',markersize=5) for i in range(8)],loc='outside lower center',ncol=4,frameon=False,fontsize=8);plt.show()
print('仅移除解码器的粗类条件；训练中的类内距离约束和聚类修正规则保持一致。')
''';one=nbformat.v4.new_notebook(cells=[nbformat.v4.new_code_cell(source)],metadata=nb.metadata);NotebookClient(one,timeout=300,kernel_name='cccphe').execute(cwd=str(R));assert any('image/png' in o.get('data',{}) for o in one.cells[0].outputs);nb=nbformat.read(p,4)
 if not any(c.cell_type=='code' and c.source.startswith('# 解码器组成条件消融：') for c in nb.cells):nb.cells.append(one.cells[0]);nbformat.write(nb,p)
 print('notebook updated',flush=True)
except BaseException:(O/'failure.txt').write_text(traceback.format_exc());traceback.print_exc();raise
