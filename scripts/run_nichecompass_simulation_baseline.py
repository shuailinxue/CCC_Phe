"""NicheCompass comparison on NicheCompass simulation; public GP, no planted GP."""
from pathlib import Path
import os,sys,json,time,traceback
os.environ['PHENONICHE_NICHECOMPASS_LRT_GP']='/data1/xueshuailin/CCC_Phe/workspace/data/priors/nichenet_v2/nichecompass_lrt_programs.json'
R=Path('/home/xueshuailin/CCC_Phe');sys.path[:0]=[str(R),str(R/'src')]
import numpy as np,pandas as pd,anndata as ad,torch
from baselines.adapters import clean_input,graph_from_members
from baselines.nichecompass_adapter import nichecompass
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
W=Path('/data1/xueshuailin/CCC_Phe/workspace');BASE=W/'outputs/nichecompass_simulation/v2';O=W/'outputs/nichecompass_simulation/baselines_K8/nichecompass';O.mkdir(parents=True,exist_ok=True)
try:
 if not (O/'complete.json').exists():
  cfg=json.loads((R/'baselines/config.json').read_text());cfg.update(json.loads((R/'config/tumor_interface_baselines.json').read_text()));cfg.update(device='cuda:1',fixed_k=8,all_cells=True,cpu_threads=4);torch.set_num_threads(4);a=ad.read_h5ad(W/'data/nichecompass_simulation/human_ortholog_analysis_input.h5ad');t=Path(json.loads((BASE/'target_cache.json').read_text())['folder']);members=np.load(t/'members.npy');clean=clean_input(a.X,tuple(a.var_names),a.obsm['spatial'],np.repeat('simulation',len(a)));clean.obs_names=a.obs_names.copy();graph=graph_from_members(members);print('NicheCompass fitting: 10,000 cells, 100 epochs; public LR + NicheNet ligand-target priors; GPU1',flush=True);start=time.monotonic();h,_=nichecompass(clean,graph,np.arange(len(clean)),cfg,O);np.save(O/'embedding.npy',h);labels=KMeans(n_clusters=8,n_init=10,random_state=40700).fit_predict(StandardScaler().fit_transform(h));np.save(O/'labels.npy',labels);truth=pd.Categorical(a.obs.niche_types,categories=[f'Niche {i}' for i in range(1,9)]).codes;ari=float(adjusted_rand_score(truth,labels));(O/'complete.json').write_text(json.dumps({'ARI':ari,'K':8,'epochs':100,'seconds':time.monotonic()-start,'prior':'public LR + NicheNet v2 ligand-target + addon; excludes simulation planted GP','truth_used_only_for_evaluation':True},indent=2));print('NicheCompass ARI',ari,flush=True)
 import nbformat
 from nbclient import NotebookClient
 p=R/'NicheCompass_Simulation_V2.ipynb';nb=nbformat.read(p,4);idx=[i for i,c in enumerate(nb.cells) if c.cell_type=='code' and c.source.startswith('# BANKSY 与 GraphST：同一模拟切片，K=8')];
 if len(idx)!=1:raise RuntimeError('Baseline cell missing or edited; preserve notebook')
 cell=nb.cells[idx[0]];src=cell.source.replace('# BANKSY 与 GraphST：同一模拟切片，K=8','# BANKSY、GraphST 与 NicheCompass：同一模拟切片，K=8').replace("['banksy','graphst']","['banksy','graphst','nichecompass']").replace('plt.subplots(1,3,figsize=(10,7)', 'plt.subplots(1,4,figsize=(13,7)');one=nbformat.v4.new_notebook(cells=[nbformat.v4.new_code_cell(src)],metadata=nb.metadata);NotebookClient(one,timeout=300,kernel_name='cccphe').execute(cwd=str(R));assert any('image/png' in o.get('data',{}) for o in one.cells[0].outputs);latest=nbformat.read(p,4);idx=[i for i,c in enumerate(latest.cells) if c.cell_type=='code' and c.source.startswith('# BANKSY 与 GraphST：同一模拟切片，K=8')];
 if len(idx)!=1:raise RuntimeError('Baseline cell changed; preserve notebook')
 latest.cells[idx[0]]=one.cells[0];nbformat.write(latest,p);print('notebook updated',flush=True)
except BaseException:(O/'failure.txt').write_text(traceback.format_exc());traceback.print_exc();raise
