"""CommuSpace public implementation on the current simulation; K=5 adaptation."""
import json, os, sys, time, subprocess
from pathlib import Path
import numpy as np
import pandas as pd
import anndata as ad
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score

root=Path('/data1/xueshuailin/CCC_Phe/workspace')
sys.path[:0]=[str(root/'artifacts/third_party/python'),str(root/'artifacts/third_party/CommuSpace')]
import torch
import CommuSpace as cs
out=root/'outputs/tumor_immune_interface/v2_baseline_k5/commuspace'
out.mkdir(parents=True,exist_ok=True)
data=root/'data/tumor_immune_interface_v2_epithelial_majority/tissue.h5ad'
a=ad.read_h5ad(data)
prior=pd.read_csv('/home/xueshuailin/CCC_Phe/data/commuspace_human_lr_atlas.tsv',sep='\t')
prior=prior.loc[prior.ligand.isin(a.var_names)&prior.receptor.isin(a.var_names),['ligand','receptor']].drop_duplicates()
prior.columns=['from','to']
print('cells',a.n_obs,'genes',a.n_vars,'complete single-gene LR pairs',len(prior),flush=True)
if len(prior)==0: raise RuntimeError('No matching LR pairs')
cs.utils.normalize_then_clip(a,95)
# Public BC demo: same-type smoothing, distance-normalized 100-neighbor sender tensor.
tensor,*_=cs.utils.cal_cell_L_R(a,prior,cell_type_col='cell_type',k=100,smoothing_ct=True,normalization_dist=True)
print('tensor',tensor.shape,flush=True)
# Upstream CP helper hardcodes CPU; changing device only does not change objective.
cs.cp._resolve_device=lambda:torch.device('cuda' if torch.cuda.is_available() else 'cpu')
start=time.time()
A,B,C,lam,total,rec,fit,zeta=cs.cp.train_cp_decomposition_large_cells(tensor,rank=16,sample_ratio=.1,subset_iters=200,hals_inner=4,epochs=20,block_rows=50000,A_inner=3,A_pgd_iters=12,random_state=2025)
embedding=A.cpu().numpy();np.save(out/'embedding.npy',embedding)
info={'method':'CommuSpace','source_commit':'0f37cdec60b3cbe5b30e675d56a8f8b9c16dd2e7','n_lr':len(prior),'rank':16,'seconds':time.time()-start,'truth_used_in_fit':False,'fit':fit}
(out/'complete.json').write_text(json.dumps(info,indent=2))
subprocess.run([sys.executable,str(Path(__file__).with_name('run_commuspace_consensus.py'))],check=True)
