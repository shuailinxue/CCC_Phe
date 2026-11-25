"""BANKSY and GraphST baseline on NicheCompass simulation, full cells, fixed K8."""
from pathlib import Path
import os,sys,json,time,subprocess,traceback
os.environ.setdefault('OMP_NUM_THREADS','4');os.environ.setdefault('OPENBLAS_NUM_THREADS','4');os.environ.setdefault('MPLBACKEND','Agg')
R=Path('/home/xueshuailin/CCC_Phe');sys.path[:0]=[str(R),str(R/'src')]
W=Path('/data1/xueshuailin/CCC_Phe/workspace');O=W/'outputs/nichecompass_simulation/baselines_K8';O.mkdir(parents=True,exist_ok=True)
def worker(method):
 import numpy as np,pandas as pd,anndata as ad,torch
 from baselines.adapters import clean_input,graph_from_members
 from baselines.scalable import fit
 from sklearn.cluster import KMeans
 from sklearn.preprocessing import StandardScaler
 from sklearn.metrics import adjusted_rand_score
 out=O/method.lower();out.mkdir(exist_ok=True)
 if (out/'complete.json').exists():print(method,'already complete',flush=True);return
 cfg=json.loads((R/'baselines/config.json').read_text());cfg.update(json.loads((R/'config/tumor_interface_baselines.json').read_text()));cfg.update(device='cuda:1',fixed_k=8,validation_fraction=0.,cpu_threads=4)
 torch.set_num_threads(4);a=ad.read_h5ad(W/'data/nichecompass_simulation/human_ortholog_analysis_input.h5ad');target=Path(json.loads((W/'outputs/nichecompass_simulation/v2/target_cache.json').read_text())['folder']);members=np.load(target/'members.npy');graph=graph_from_members(members);clean=clean_input(a.X,tuple(a.var_names),a.obsm['spatial'],np.repeat('simulation',len(a)));clean.obs_names=a.obs_names.copy();types=a.obs.cell_type.cat.codes.to_numpy();cats=list(a.obs.cell_type.cat.categories);print(method,'start',len(a),'cells',a.n_vars,'genes','K8',flush=True)
 t=time.monotonic();emb=np.asarray(fit(method,clean,graph,members,types,cats,cfg,out),dtype=np.float32);np.save(out/'embedding.npy',emb);X=emb if method=='GraphST' else StandardScaler().fit_transform(emb);labels=KMeans(n_clusters=8,n_init=10,random_state=40700).fit_predict(X);np.save(out/'labels.npy',labels);truth=pd.Categorical(a.obs.niche_types,categories=[f'Niche {i}' for i in range(1,9)]).codes;ari=adjusted_rand_score(truth,labels);(out/'complete.json').write_text(json.dumps({'method':method,'ARI':ari,'K':8,'cells':len(a),'genes':a.n_vars,'seconds':time.monotonic()-t,'label_truth_only_evaluation':True,'graphst_epochs':cfg['graphst_epochs'] if method=='GraphST' else None,'banksy_lambda':cfg['banksy_lambda'] if method=='BANKSY' else None,'banksy_neighbors':cfg['banksy_k_geom'] if method=='BANKSY' else None},indent=2));print(method,'completed ARI',ari,flush=True)
def report():
 import nbformat
 from nbclient import NotebookClient
 p=R/'NicheCompass_Simulation_V2.ipynb';nb=nbformat.read(p,4);source=f'''# BANKSY 与 GraphST：同一模拟切片，K=8
from pathlib import Path
import json,numpy as np,pandas as pd,anndata as ad,matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.metrics import adjusted_rand_score,confusion_matrix
from scipy.optimize import linear_sum_assignment
out=Path('{O}');a=ad.read_h5ad('{W}/data/nichecompass_simulation/sim1_1105genes_10000locs_strongincrements.h5ad',backed='r');xy=np.asarray(a.obsm['spatial']);truth=pd.Categorical(a.obs.niche_types,categories=[f'Niche {{i}}' for i in range(1,9)]).codes;a.file.close();colors=plt.get_cmap('tab10')(np.arange(8));order=np.random.default_rng(2026).permutation(len(truth));panels=[(truth,'Ground truth')]
for method in ['banksy','graphst']:
 lab=np.load(out/method/'labels.npy');r,c=linear_sum_assignment(-confusion_matrix(truth,lab,labels=range(8)));mapping=dict(zip(c,r));matched=np.array([mapping[i] for i in lab]);panels.append((matched,f'{{method.upper()}} · K=8\\nARI = {{adjusted_rand_score(truth,lab):.3f}}'))
fig,axes=plt.subplots(1,3,figsize=(10,7),layout='constrained')
for ax,(labels,title) in zip(axes,panels):
 ax.scatter(xy[order,0],xy[order,1],c=colors[labels[order]],s=4,lw=0,rasterized=True);ax.set_aspect('equal');ax.axis('off');ax.set_title(title,fontsize=11)
fig.legend(handles=[Line2D([],[],marker='o',lw=0,color=colors[i],label=f'Niche {{i+1}}',markersize=5) for i in range(8)],loc='outside lower center',ncol=4,frameon=False,fontsize=8);plt.show()
''';one=nbformat.v4.new_notebook(cells=[nbformat.v4.new_code_cell(source)],metadata=nb.metadata);NotebookClient(one,timeout=300,kernel_name='cccphe').execute(cwd=str(R));latest=nbformat.read(p,4)
 if not any(c.cell_type=='code' and c.source.startswith('# BANKSY 与 GraphST：同一模拟切片，K=8') for c in latest.cells):latest.cells.append(one.cells[0]);nbformat.write(latest,p)
 print('Notebook updated',flush=True)
if __name__=='__main__':
 if len(sys.argv)>1:worker(sys.argv[1])
 else:
  try:
   jobs=[]
   for method in ['BANKSY','GraphST']:
    log=(O/(method.lower()+'.log')).open('a');jobs.append((method,subprocess.Popen([sys.executable,'-u',__file__,method],stdout=log,stderr=subprocess.STDOUT,env=os.environ.copy()),log))
   failures=[]
   for method,proc,log in jobs:
    code=proc.wait();log.close();print(method,'worker exit',code,flush=True)
    if code:failures.append(method)
   if failures:raise RuntimeError('Failed workers: '+str(failures))
   report()
  except BaseException:(O/'failure.txt').write_text(traceback.format_exc());traceback.print_exc();raise
