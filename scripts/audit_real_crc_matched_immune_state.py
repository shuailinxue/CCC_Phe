"""Descriptive, same-specimen composition-matched local T-state variation.
No niche predictions, clinical outcomes or truth labels enter selection. Grouping
on GZMB is descriptive; PRF1 and CCL5 are separate corroborating RNA markers.
Overlapping K30 windows are not independent biological replicates.
"""
from pathlib import Path
import json
import numpy as np,pandas as pd,anndata as ad
from scipy import sparse
P=Path('/home/nas3/biod/xueshuailin/CCC_Phe/workspace');ROOT=P/'data/crc_neutrophil_xenium';M=ROOT/'matrices/1a212859f85252e9';O=P/'outputs/tumor_immune_interface/v2_directed_pair_encoder/real_crc_matched';O.mkdir(parents=True,exist_ok=True)
a=ad.read_h5ad(ROOT/'processed/CRC_selected_five.h5ad');cats=json.loads((M/'complete.json').read_text())['categories']; comp=np.load(M/'composition.npy',mmap_mode='r');members=np.load(M/'members.npy',mmap_mode='r');weights=np.load(M/'anchor_weights.npy',mmap_mode='r');typ=a.obs.cell_type.astype(str).to_numpy();sample=a.obs.specimen_id.astype(str).to_numpy();raw=sparse.csr_matrix(a.X);tot=np.asarray(raw.sum(1)).ravel();gene=['GZMB','PRF1','CCL5'];ix=a.var_names.get_indexer(gene);assert np.all(ix>=0);value=raw[:,ix].toarray().astype(np.float32)*10000/np.maximum(tot[:,None],1);results=[]
for spec in ('crc_a_c','crc_a_m','crc_b_m'):
 cancer=np.asarray(comp[:,cats.index('Cancer cell')]);T=np.asarray(comp[:,cats.index('T cell')]);fib=np.asarray(comp[:,cats.index('Fibroblast')]);eligible=(sample==spec)&(typ=='Cancer cell')&(cancer>=.4)&(T>=.05);inds=np.flatnonzero(eligible)
 # A fixed 10%-wide bin is selected using the composition counts only.
 b=np.c_[np.floor(cancer[inds]*10),np.floor(T[inds]*10),np.floor(fib[inds]*10)].astype(int);table=pd.DataFrame(b,columns=['cancer','T','fib']);counts=table.value_counts();
 if counts.empty:continue
 winner=counts.index[0];chosen=(b==np.array(winner)).all(1);ids=inds[chosen];scores=np.zeros((len(ids),3),np.float32);support=np.zeros(len(ids),np.float32)
 for off in range(0,len(ids),512):
  sl=slice(off,min(off+512,len(ids)));m=members[ids[sl]];w=weights[ids[sl]]*(typ[m]=='T cell');den=w.sum(1);scores[sl]=np.einsum('bk,bkg->bg',w,value[m])/np.maximum(den[:,None],1e-8);support[sl]=den
 lo,hi=np.quantile(scores[:,0],[.25,.75]);low=scores[:,0]<=lo;high=scores[:,0]>=hi
 if hi==lo:print(spec,'no GZMB spread',len(ids));continue
 def quality(group):
  physical=np.unique(members[ids[group]].ravel());physical=physical[typ[physical]=='T cell'];return dict(unique_T_cells=int(len(physical)),median_T_panel_counts=float(np.median(tot[physical])) if len(physical) else None)
 item=dict(specimen=spec,selection='cancer-centered K30, cancer >=40%, T >=5%; largest 10%-wide cancer/T/fibroblast bin; within-bin GZMB bottom/top quartiles',n=int(len(ids)),n_low=int(low.sum()),n_high=int(high.sum()),bin=dict(zip(['cancer','T','fibroblast'],[int(v) for v in winner])),GZMB_q25=float(lo),GZMB_q75=float(hi),low_composition=np.asarray(comp[ids[low]]).mean(0).round(5).tolist(),high_composition=np.asarray(comp[ids[high]]).mean(0).round(5).tolist(),categories=cats,low_mean=dict(zip(gene,scores[low].mean(0).astype(float))),high_mean=dict(zip(gene,scores[high].mean(0).astype(float))),low_quality=quality(low),high_quality=quality(high),caveat='Descriptive within one specimen; groups chosen by GZMB, overlapping spatial windows; no causality or independent sample test.')
 q0=np.unique(members[ids[low]].ravel());q1=np.unique(members[ids[high]].ravel());item['overlap_unique_T_cells']=int(np.sum(typ[np.intersect1d(q0,q1)]=='T cell'))
 q0=np.setdiff1d(q0,q1);q1=np.setdiff1d(q1,np.unique(members[ids[low]].ravel()));q0=q0[typ[q0]=='T cell'];q1=q1[typ[q1]=='T cell'];universe=np.r_[q0,q1];edges=np.unique(np.quantile(tot[universe],np.linspace(0,1,11)));rng=np.random.default_rng(491);picks=[[],[]]
 for j in range(len(edges)-1):
  groups=[z[(tot[z]>=edges[j])&((tot[z]<=edges[j+1]) if j==len(edges)-2 else (tot[z]<edges[j+1]))] for z in (q0,q1)];n=min(map(len,groups))
  if n:
   for k in (0,1):picks[k].extend(rng.choice(groups[k],size=n,replace=False).tolist())
 if len(picks[0]):
  item['nonoverlap_depth_matched_T']=dict(n_per_group=len(picks[0]),low_PRF1=float(value[picks[0],1].mean()),high_PRF1=float(value[picks[1],1].mean()),low_median_depth=float(np.median(tot[picks[0]])),high_median_depth=float(np.median(tot[picks[1]])))
 results.append(item);print(json.dumps(item),flush=True)
(O/'summary.json').write_text(json.dumps(results,indent=2)+'\n')
