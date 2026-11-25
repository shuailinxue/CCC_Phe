"""Map ST1/ST5 to fixed ST1 V3 training labels using leave-one-out kNN.
Calibrate per-label composition/communication distances on held-out ST1 blocks;
show spatial enrichment separately without changing cell labels.
"""
import os,sys,json,traceback,time
from pathlib import Path
os.environ.setdefault('OMP_NUM_THREADS','4');os.environ.setdefault('OPENBLAS_NUM_THREADS','4')
import numpy as np,pandas as pd,torch,anndata as ad
W=Path('/data1/xueshuailin/CCC_Phe/workspace'); B=W/'outputs/nsclc_zhencaiwei/ST1_ST5/v3_joint'; O=W/'outputs/nsclc_zhencaiwei/ST1_ST5/v3_st1_reference_k10';D=O/'mapping_v2';D.mkdir(exist_ok=True);ROOT=Path('/home/xueshuailin/CCC_Phe');DEVICE='cuda:0';torch.set_num_threads(4)
def log(*x):print(time.strftime('%F %T'),*x,flush=True)
def main():
 q=np.load(O/'mapping.npz');h=np.load(O/'h.npy',mmap_mode='r');sid=q['sample_id'];train=q['train_indices'];hold=q['calibration_indices'];true=q['training_labels'];n=len(sid);P=Path(json.loads((B/'target_cache.json').read_text())['folder']);comp=np.load(P/'composition.npy',mmap_mode='r');members=np.load(P/'members.npy',mmap_mode='r')
 p=np.sqrt(np.maximum(np.asarray(comp),0));pm=p[train].mean(0);ps=max(float(np.sqrt(np.mean(np.sum((p[train]-pm)**2,axis=1)))),1e-8);hm=h[train].mean(0);hs=max(float(np.sqrt(np.mean(np.sum((h[train]-hm)**2,axis=1)))),1e-8);pn=((p-pm)/ps).astype('float32');hn=((h-hm)/hs).astype('float32');x=np.column_stack((pn,hn)).astype('float32');device=torch.device(DEVICE);ref=torch.as_tensor(x[train],device=device);ref_norm=(ref*ref).sum(1);y=torch.as_tensor(true,dtype=torch.long,device=device);labels=np.empty(n,np.int16);cd=np.empty(n,np.float32);hd=np.empty(n,np.float32);confidence=np.empty(n,np.float32);train_position=np.full(n,-1,np.int32);train_position[train]=np.arange(len(train));k=15
 log('Start kNN mapping all',n,'ref',len(train),'GPU',DEVICE)
 for start in range(0,n,256):
  end=min(start+256,n);z=torch.as_tensor(x[start:end],device=device);dist=((z*z).sum(1)[:,None]+ref_norm[None,:]-2*z@ref.T).clamp_min_(0);pos=train_position[start:end];mask=pos>=0
  if mask.any():dist[torch.as_tensor(np.flatnonzero(mask),device=device),torch.as_tensor(pos[mask],device=device)]=float('inf')
  d,ii=torch.topk(dist,k,dim=1,largest=False);yy=y[ii];votes=torch.nn.functional.one_hot(yy,num_classes=10).sum(1);pred=votes.argmax(1);sel=(yy==pred[:,None]);den=sel.sum(1).clamp_min(1);neighbor=ref[ii];refp=(neighbor[:,:,:pn.shape[1]]*sel[:,:,None]).sum(1)/den[:,None];refh=(neighbor[:,:,pn.shape[1]:]*sel[:,:,None]).sum(1)/den[:,None];labels[start:end]=pred.cpu().numpy();confidence[start:end]=(den/k).cpu().numpy();cd[start:end]=torch.linalg.vector_norm(z[:,:pn.shape[1]]-refp,dim=1).cpu().numpy();hd[start:end]=torch.linalg.vector_norm(z[:,pn.shape[1]:]-refh,dim=1).cpu().numpy()
  if start%25600==0:log('Mapped',end,'/',n)
 acc=float(np.mean(labels[train]==true));log('Training leave-one-out agreement',acc)
 tc=[];th=[]
 for c in range(10):
  j=hold[labels[hold]==c]
  if len(j)<100:raise ValueError(f'Insufficient ST1 calibration for class {c}: {len(j)}')
  tc.append(float(np.quantile(cd[j],.95)));th.append(float(np.quantile(hd[j],.95)))
 tc=np.array(tc);th=np.array(th);state=np.where(cd>tc[labels],2,np.where(hd>th[labels],1,0)).astype('int8')
 # Spatial neighborhood enrichment is a descriptive overlay, never relabeling.
 # Calibrate threshold on heldout ST1 only; strict > avoids flagging the mass at quantile.
 density=np.empty((n,2),np.float32)
 for start in range(0,n,2048):
  ids=np.asarray(members[start:start+2048]);density[start:start+2048,0]=(state[ids]==1).mean(1);density[start:start+2048,1]=(state[ids]==2).mean(1)
 cut=np.quantile(density[hold],.99,axis=0);regions=np.zeros(n,np.int8);regions[density[:,0]>cut[0]]=1;regions[density[:,1]>cut[1]]=2
 np.savez_compressed(D/'mapping.npz',cell_ids=q['cell_ids'],sample_id=sid,reference_label=labels,status=state,regions=regions,density=density,composition_distance=cd,communication_distance=hd,confidence=confidence,train_indices=train,calibration_indices=hold,training_labels=true)
 meta={'method':'15-nearest ST1 training reference centers in jointly RMS-scaled sqrt composition and h; majority vote; leave-self-out for ST1 training centers','training_leave_one_out_agreement':acc,'per_class_calibration':'separate 95th percentiles of composition and communication distances in held-out ST1 blocks','calibration_counts':[int(np.sum(labels[hold]==c)) for c in range(10)],'comp_thresholds':tc.tolist(),'comm_thresholds':th.tolist(),'spatial_overlay':'within-slice original 30-member neighborhood candidate fraction > ST1 holdout 99th percentile; descriptive only; memberships unchanged','spatial_thresholds':cut.tolist(),'st1_holdout_status_counts':np.bincount(state[hold],minlength=3).tolist(),'st5_status_counts':np.bincount(state[sid==1],minlength=3).tolist(),'st1_holdout_region_counts':np.bincount(regions[hold],minlength=3).tolist(),'st5_region_counts':np.bincount(regions[sid==1],minlength=3).tolist(),'reference_converged':False,'warning':'Global thresholds are not tests of biological novelty; reference fit had not converged after 200 iterations; spatially overlapping windows and single sample per slice limit inference.'}
 (D/'complete.json').write_text(json.dumps(meta,indent=2));log('Mapping complete',json.dumps(meta));update_notebook()
def update_notebook():
 import nbformat
 from nbclient import NotebookClient
 p=ROOT/'Cellist_ST1_ST5_Joint_V3.ipynb';n=nbformat.read(p,4)
 source='''# ST1 K=10 参考：ST5 映射与空间候选区域（描述性）
from pathlib import Path
import json,numpy as np,pandas as pd,anndata as ad,matplotlib.pyplot as plt
from matplotlib.lines import Line2D
W=Path('/data1/xueshuailin/CCC_Phe/workspace');D=W/'outputs/nsclc_zhencaiwei/ST1_ST5/v3_st1_reference_k10/mapping_v2';q=np.load(D/'mapping.npz');meta=json.loads((D/'complete.json').read_text());a=ad.read_h5ad(W/'data/nsclc_zhencaiwei/ST1_ST5_joint_v3_input.h5ad',backed='r');assert np.array_equal(a.obs_names.to_numpy(dtype=str),q['cell_ids']);xy=np.asarray(a.obsm['spatial']);a.file.close();sid=q['sample_id'];lab=q['reference_label'];state=q['status'];region=q['regions'];pal=plt.get_cmap('tab10')(np.arange(10));colors=np.array(['#d4d8d8','#e69f00','#be4451']);fig,axes=plt.subplots(1,3,figsize=(15,5),layout='constrained')
for j in (0,1):
 ix=np.flatnonzero(sid==j);axes[j].scatter(xy[ix,0],xy[ix,1],c=pal[lab[ix]],s=.45,lw=0,rasterized=True);axes[j].axis('off');axes[j].set_aspect('equal');axes[j].set_title(['ST1 · K=10 reference mapping','ST5 · ST1 reference mapping'][j])
ix=np.flatnonzero(sid==1);axes[2].scatter(xy[ix,0],xy[ix,1],c=colors[region[ix]],s=.5,lw=0,rasterized=True);axes[2].axis('off');axes[2].set_aspect('equal');axes[2].set_title('ST5 · spatial candidate enrichment');axes[2].legend(handles=[Line2D([],[],marker='o',lw=0,color=colors[i],label=x,markersize=5) for i,x in enumerate(['below threshold','composition similar; communication differs','composition differs'])],loc='lower center',bbox_to_anchor=(.5,-.12),fontsize=7,frameon=False);plt.show()
print('ST1 参考训练细胞的留一法映射一致率：%.1f%%。'% (100*meta['training_leave_one_out_agreement']))
print('ST5 逐细胞候选数（参考内 / 组成相似通讯不同 / 组成不同）：',meta['st5_status_counts']);print('ST5 空间富集叠加的细胞数（未超阈值 / 通讯候选富集 / 组成候选富集）：',meta['st5_region_counts'])
print('空间图：原始 30 细胞邻域中候选比例超过 ST1 空间留出区的 99% 分位；这是展示用邻域富集，不改变逐细胞标签。留出区按各参考类分别校准两种距离的 95% 阈值。')
print('注意：参考 K=10 在 200 次更新后未满足收敛阈值；空间邻域重叠，ST1/ST5 各一张切片，结果仅为探索性候选。')
display(pd.crosstab(pd.Series(lab[sid==1]+1,name='ST1 reference niche'),pd.Series(state[sid==1],name='ST5 cell-level result')).rename(columns={0:'within reference',1:'communication shift',2:'composition shift'}))
'''
 one=nbformat.v4.new_notebook(cells=[nbformat.v4.new_code_cell(source)],metadata=n.metadata);NotebookClient(one,timeout=600,kernel_name='cccphe',allow_errors=False).execute(cwd=str(ROOT));latest=nbformat.read(p,4);latest.cells[4]=one.cells[0];latest.cells[4].execution_count=5
 for out in latest.cells[4].outputs:
  if 'execution_count' in out:out['execution_count']=5
 nbformat.write(latest,p);log('Notebook cell 5 updated')
if __name__=='__main__':
 try:main()
 except Exception:(D/'failure.txt').write_text(traceback.format_exc());traceback.print_exc();raise
