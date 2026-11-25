"""Refresh only the single-run NicheCompass result cell."""
import nbformat
from pathlib import Path
from nbclient import NotebookClient
p=Path('/home/xueshuailin/CCC_Phe/Brain_V3_scNiche_Harmonics_Benchmark.ipynb');n=nbformat.read(p,4)
s='''# NicheCompass：STARmap / MERFISH 各一次（独立追加）
from pathlib import Path
import json,numpy as np,pandas as pd,anndata as ad,matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import confusion_matrix
from IPython.display import display
W=Path('/data1/xueshuailin/CCC_Phe/workspace');O=W/'outputs/brain_nichecompass_single_run';B=Path('/home/nas3/biod/xueshuailin/CCC_Phe/workspace/outputs/brain_10run_benchmark');rows=[];results={}
for dataset in ['starmap','merfish']:
 folder=O/dataset;f=folder/'complete.json'
 if not f.exists():
  failure=folder/'failure.txt';print(dataset.upper(), '运行失败：'+failure.read_text()[-1000:] if failure.exists() else '尚未完成；日志见 '+str(folder/'run.log'));continue
 res=json.loads(f.read_text());results[dataset]=res
 for method in ['KMeans','Leiden']:
  value=res[method];rows.append({'dataset':dataset,'method':'NicheCompass '+method,'training':res['training'],'epoch_budget':res['epochs_budget'],'ARI':np.nan if value is None else value['ARI'],'macro_F1':np.nan if value is None else value['macro_F1'],'Leiden_resolution':res['leiden_resolution'] if method=='Leiden' else None})
for dataset in results:
 f=B/dataset/'v3/seed_112/metrics.json'
 if f.exists():
  r=json.loads(f.read_text());rows.append({'dataset':dataset,'method':'V3 joint · seed 112','training':'existing fit','epoch_budget':100,'ARI':r['ARI'],'macro_F1':r['macro_F1'],'Leiden_resolution':None})
if rows:display(pd.DataFrame(rows).set_index(['dataset','method']).round(4))
if results:
 fig,axes=plt.subplots(len(results),3,figsize=(9,3.6*len(results)),layout='constrained',squeeze=False)
 for row,(dataset,res) in enumerate(results.items()):
  path=W/'data/harmonics_fig2'/('starmap/analysis_input.h5ad' if dataset=='starmap' else 'merfish_scaled/donor12_slice0_analysis_input.h5ad');a=ad.read_h5ad(path,backed='r');xy=np.asarray(a.obsm['spatial']);valid=np.ones(len(a),bool);q=np.load(O/dataset/'predictions.npz');assert np.array_equal(a.obs_names.to_numpy(dtype=str),q['cell_ids'])
  if dataset=='starmap':
   ref=ad.read_h5ad(W/'data/harmonics_fig2/starmap/stDyer_mouse_cortex.h5ad');dist,ix=cKDTree(np.c_[ref.obs.X,-ref.obs.Y]).query(xy);valid=dist<.01;truth=pd.Categorical(ref.obs.label.iloc[ix].astype(str).str.upper()).codes
  else:truth=pd.Categorical(a.obs.benchmark_region.astype(str)).codes
  a.file.close();order=np.random.default_rng(5).permutation(np.flatnonzero(valid))
  for col,(title,pred) in enumerate([('Annotation',truth),('NicheCompass · KMeans',q['kmeans']),('NicheCompass · Leiden',q['leiden'])]):
   ax=axes[row,col];ax.axis('off');ax.set_aspect('equal')
   if len(pred)==0:ax.set_title(title+' · no K=7');continue
   if col:
    rr,cc=linear_sum_assignment(-confusion_matrix(truth[valid],pred[valid],labels=np.arange(7)));mp=dict(zip(cc,rr));lab=np.array([mp.get(v,0) for v in pred]);score=res['KMeans' if col==1 else 'Leiden']['ARI'];title+=f' · ARI {score:.3f}'
   else:lab=pred
   ax.scatter(xy[order,0],xy[order,1],c=lab[order],vmin=0,vmax=9,cmap='tab10',s=2,lw=0,rasterized=True);ax.set_title(dataset.upper()+' · '+title,fontsize=9)
 plt.show()
print('各数据集 1 次，seed=112；STARmap 复用已完成的 400 轮模型，MERFISH 新训练预算 100 轮，预算不一致且不是正式十次 benchmark。原生 NicheCompass 使用 mouse gene programs 与自身 KNN4 物理图；V3 历史结果使用小鼠到人同源映射输入。Leiden 从预设网格选首个 K=7，不根据真实标签选择；若找不到则报告缺失。KMeans 便于统一表示比较。标注只用于评价和配色。')
'''
one=nbformat.v4.new_notebook(cells=[nbformat.v4.new_code_cell(s)],metadata=n.metadata);NotebookClient(one,timeout=600,kernel_name='cccphe').execute(cwd=str(p.parent));latest=nbformat.read(p,4);match=[i for i,c in enumerate(latest.cells) if c.cell_type=='code' and c.source.startswith('# NicheCompass：STARmap / MERFISH 各一次')]
if len(match)>1:raise ValueError('duplicate cell marker')
if match:latest.cells[match[0]]=one.cells[0]
else:latest.cells.append(one.cells[0])
nbformat.write(latest,p);print('Notebook NicheCompass cell refreshed',flush=True)
