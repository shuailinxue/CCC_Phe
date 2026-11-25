"""Memory-bounded STCase CCC preparation, using upstream scoring equations.

No niche labels enter this process. One significance-filtered sparse LR matrix
is saved at a time. Reference: STCase ccci.get_true_weight_matirx and
calculate_fake_weight; normal scoring, scope=6, min_exp=.1, 100 backgrounds.
"""
import json,random,time
from pathlib import Path
import numpy as np,pandas as pd,anndata as ad,scipy.sparse as sp,torch
from scipy.spatial import cKDTree

root=Path('/data1/xueshuailin/CCC_Phe/workspace');out=root/'outputs/tumor_immune_interface/v2_baseline_k5/stcase_fair';cache=out/'ccc_sparse';cache.mkdir(parents=True,exist_ok=True)
torch.set_num_threads(4);device=torch.device('cuda' if torch.cuda.is_available() else 'cpu');rng=np.random.default_rng(2025);random.seed(2025)
a=ad.read_h5ad(root/'data/tumor_immune_interface_v2_epithelial_majority/tissue.h5ad');x=a.X.toarray() if sp.issparse(a.X) else np.asarray(a.X);x=np.asarray(x,dtype=np.float32)
# Per-cell CP10k followed by log1p, predeclared for this dataset.
x=np.log1p(x/np.maximum(x.sum(1,keepdims=True),1e-8)*1e4).astype('float32');names=np.asarray(a.var_names,dtype=str);gene_idx={g:i for i,g in enumerate(names)}
df=pd.read_csv(out/'panel_covered_official_lr.csv');xy=np.asarray(a.obsm['spatial'],dtype=np.float32);dist,_=cKDTree(xy).query(xy,k=7,workers=4);v=dist[:,1:].ravel();good=np.abs((v-v.mean())/max(v.std(),1e-8))<=3;radius=float(v[good].max());wr=1.5*radius
excluded=set();
for s in list(df.ligand_subunits)+list(df.receptor_subunits):excluded.update(str(s).split('|'))
background=np.array([i for i,g in enumerate(names) if g not in excluded and not g.startswith(('MT-','MT_'))],dtype=int)
if len(background)<100:raise RuntimeError('Fewer than 100 matched background genes')
means=x.mean(0);bio={}
for side in ('ligand','receptor'):
 for key,sub in zip(df[side],df[side+'_subunits']):
  if key not in bio:
   ids=[gene_idx[g] for g in str(sub).split('|')];z=x[:,ids]
   if len(ids)==1:val=z[:,0]
   else:val=(np.quantile(z,.25,axis=1)*.25+np.quantile(z,.5,axis=1)*.5+np.quantile(z,.75,axis=1)*.25).astype('float32')
   bg=background[np.argsort(np.abs(means[background]-float(val.mean())),kind='stable')[:100]].copy();rng.shuffle(bg);bio[key]=(val,bg)
xt=torch.as_tensor(x,device=device);xy_tree=cKDTree(xy);start=time.time();manifest={'method':'STCase CCC sparse preparation','scoring':'normal','scope':6,'radius':radius,'min_exp':.1,'cutoff':.05,'background_number':100,'device':str(device),'seed':2025,'pairs':len(df),'truth_used':False,'note':'Exact sparse evaluation of upstream score formulas; per-cell expression CP10k log1p; 100 shuffled mean-matched backgrounds; no LR preselection.'};(out/'ccc_protocol.json').write_text(json.dumps(manifest,indent=2))
print(json.dumps(manifest),flush=True)
for j,row in df.iterrows():
 path=cache/f'{j:04d}.npz'
 if path.exists():continue
 L,bl=bio[row.ligand];R,br=bio[row.receptor];secret=str(row.annotation).lower()=='secreted'
 maxlig=float(L.max());maxdist=wr*np.sqrt(max(0.,np.log(maxlig/.1))) if secret and maxlig>.1 else radius
 maxdist=max(maxdist,0.)
 if maxdist==0:sp.save_npz(path,sp.csr_matrix((len(x),len(x)),dtype=np.float32));continue
 # Evaluate only true nonzero candidates. Other entries are necessarily zero.
 neighbors=xy_tree.query_ball_point(xy,maxdist,workers=4);src=np.repeat(np.arange(len(x)),[len(v) for v in neighbors]);dst=np.concatenate(neighbors).astype('int32');del neighbors
 ds=np.linalg.norm(xy[src]-xy[dst],axis=1);decay=np.exp(-np.square(ds/wr));ok=(L[src]*decay>=.1)&(R[dst]>0)
 if not secret:ok&=ds<radius
 src=src[ok];dst=dst[ok];decay=decay[ok].astype('float32');value=(L[src]*R[dst]*decay).astype('float32');keep=[]
 for lo in range(0,len(src),16384):
  hi=min(lo+16384,len(src));ss=torch.as_tensor(src[lo:hi],device=device);dd=torch.as_tensor(dst[lo:hi],device=device);dw=torch.as_tensor(decay[lo:hi],device=device);true=torch.as_tensor(value[lo:hi],device=device)
  lb=xt[ss[:,None],torch.as_tensor(bl,device=device)];rb=xt[dd[:,None],torch.as_tensor(br,device=device)];fake=lb*rb*dw[:,None];fake=torch.where(lb*dw[:,None]>=.1,fake,0);p=1-(fake<true[:,None]).float().mean(1);keep.append((p<.05).cpu().numpy())
 valid=np.concatenate(keep) if keep else np.zeros(0,dtype=bool);sp.save_npz(path,sp.csr_matrix((value[valid],(src[valid],dst[valid])),shape=(len(x),len(x)),dtype=np.float32))
 if (j+1)%10==0 or j==0:print('CCC',j+1,'/',len(df),'candidate',len(src),'retained',int(valid.sum()),'elapsed',round(time.time()-start,1),flush=True)
(out/'ccc_complete.json').write_text(json.dumps({'n_pairs':len(df),'seconds':time.time()-start,'protocol':manifest},indent=2));print('CCC preparation complete',flush=True)
