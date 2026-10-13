#!/usr/bin/env python
"""Real-data exact CCC sample and inference/storage benchmark; no model fitting."""
from pathlib import Path
import sys,json,time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np,pandas as pd,anndata as ad,torch
from scipy import sparse
from phenoniche.storage import STORAGE_ROOT
from phenoniche.ccc.lr_atlas import load_lr_atlas
from phenoniche.gnn.losses import pair_ccc
from phenoniche.gnn.model import physical_weights,MicroenvironmentGNN

torch.set_num_threads(4);torch.manual_seed(40700)
root=STORAGE_ROOT;old=root/'outputs/microenvironment_gnn';cache=old/'cache'
x=ad.read_h5ad(root/'data/processed/xenium_prime_5k_breast.h5ad');lookup={str(g).upper():i for i,g in enumerate(x.var_names)}
atlas=load_lr_atlas(Path(__file__).resolve().parents[1]/'data/commuspace_human_lr_atlas.tsv')
lr=[a for a in atlas.interactions if set(a.ligand_components+a.receptor_components)<=lookup.keys()]
categories=sorted(x.obs.cell_type_coarse.astype(str).unique());types=pd.Categorical(x.obs.cell_type_coarse.astype(str),categories=categories).codes
sets=sorted({a.ligand_components for a in lr}|{a.receptor_components for a in lr});si={s:i for i,s in enumerate(sets)};cov=np.zeros((10,len(sets)))
for t in range(10):
 b=x.X[types==t].toarray()>0
 for i,s in enumerate(sets):cov[t,i]=b[:,[lookup[g] for g in s]].all(1).mean()
lcov=cov[:,[si[a.ligand_components] for a in lr]];rcov=cov[:,[si[a.receptor_components] for a in lr]]
mask=((lcov[:,None,:]>=.05)&(rcov[None,:,:]>=.05)).reshape(100,-1)
# Previous exact physical-opportunity audit verified all 100 directions pass;
# labels, coordinates, neighbor definition and spatial weights are unchanged.
prior=json.loads((old/'ccc_filter_count_audit.json').read_text());assert prior['supported_type_pairs']==100
flat=np.flatnonzero(mask.ravel());pair,idx=np.divmod(flat,len(lr));s,r=np.divmod(pair,10)
features=pd.DataFrame({'sender':np.asarray(categories)[s],'receiver':np.asarray(categories)[r]})
print('Actual 5% candidate features:',len(features),flush=True)
rng=np.random.default_rng(40700);rows=np.sort(rng.choice(len(x),4096,replace=False))
members=np.load(cache/'members.npy',mmap_mode='r');weights=np.load(cache/'anchor_weights.npy',mmap_mode='r');xy=np.asarray(x.obsm['spatial'],dtype=np.float32)
local=np.asarray(members[rows]);unique,inv=np.unique(local,return_inverse=True);raw=x.X[unique].toarray();inv=inv.reshape(local.shape)
sides=[]
for field in ('ligand_components','receptor_components'):
 values=np.column_stack([np.prod(raw[:,[lookup[g] for g in getattr(lr[i],field)]],axis=1)**(1/len(getattr(lr[i],field))) for i in idx]).astype(np.float32)
 sides.append(values)
print('Computing exact directed CCC for 4,096 sampled real microenvironments',flush=True)
blocks=[]
with torch.no_grad():
 for start in range(0,len(rows),256):
  select=slice(start,start+256);loc=local[select]
  args=[torch.as_tensor(a[inv[select]],device='cuda:0') for a in sides]
  w=physical_weights(torch.as_tensor(xy[loc],device='cuda:0'),torch.as_tensor(np.asarray(weights[rows[select]]).copy(),device='cuda:0'),20.,torch.as_tensor(loc,device='cuda:0'))
  blocks.append(pair_ccc(*args,torch.as_tensor(types[loc],device='cuda:0'),features,categories,w).cpu().numpy())
c=np.concatenate(blocks);csr=sparse.csr_matrix(c);csr.eliminate_zeros()
n,f=len(x),len(features);density=csr.nnz/c.size
report={'threshold':.05,'cells':n,'candidate_features':f,'CCC_sample_microenvironments':len(rows),'sample_seed':40700,'sample_exact_nonzero_fraction':float(density),'sample_mean_nonzero_features':float(csr.nnz/len(rows)),'dense_float32_GB':n*f*4/1e9,'estimated_CSR_float32_int32_GB':(n*f*density*8+(n+1)*4)/1e9,'Z_float32_MB':n*32*4/1e6,'H_float32_MB':32*f*4/1e6,'batch_512_CCC_MB':512*f*4/1e6,'CSR_sample_bytes':int(csr.data.nbytes+csr.indices.nbytes+csr.indptr.nbytes),'dense_sample_bytes':int(c.nbytes),'scope':'CCC values from real expression and exact existing physical pair formula; CSR whole-matrix size is a sample extrapolation. Neural network timings use untrained weights and fixed real inputs, not biological analysis results.'}
# Compare only neural forward cost on identical real graph/expression inputs.
rows=rows[:512];loc=np.asarray(members[rows]);unique,inv=np.unique(loc,return_inverse=True)
enc=np.load(cache/'encoder_features.npy',mmap_mode='r')
args=[torch.as_tensor(np.asarray(enc[unique]).copy(),device='cuda:0'),torch.as_tensor(types[unique].astype(np.int64),device='cuda:0'),torch.as_tensor(inv.reshape(loc.shape),device='cuda:0'),torch.as_tensor(xy[loc],device='cuda:0'),torch.as_tensor(np.asarray(weights[rows]).copy(),device='cuda:0'),torch.as_tensor(loc,device='cuda:0')]
results={}
with torch.no_grad():
 for dim in (450,f):
  model=MicroenvironmentGNN(enc.shape[1],10,dim,32,64).cuda(0).eval()
  for reconstruction in (False,True):
   for _ in range(10):model(*args,reconstruct=reconstruction)
   times=[]
   for repeat in range(5):
    torch.cuda.synchronize();start=time.perf_counter()
    for _ in range(50):model(*args,reconstruct=reconstruction)
    torch.cuda.synchronize();times.append((time.perf_counter()-start)/50*1000)
   results[f'F={dim},reconstruct={reconstruction}']={'median_ms_per_512_cells':float(np.median(times)),'range_ms':[float(min(times)),float(max(times))]}
report['GPU_forward_benchmark']=results
p=root/'artifacts/ccc_expr05_storage_assessment.json';p.parent.mkdir(exist_ok=True,parents=True);p.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2),flush=True)
