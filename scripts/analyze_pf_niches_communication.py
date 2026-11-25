"""Descriptive two-section PF analysis: cell, niche and communication evidence.

No differential hypothesis testing: observations come from one donor per group.
Communication summaries use measured, opportunity-supported inputs and fixed
stratified sampling; feature selection is descriptive and post hoc.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS','4')
os.environ.setdefault('OPENBLAS_NUM_THREADS','4')
from pathlib import Path
import json
import base64
import numpy as np
import pandas as pd
import anndata as ad
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D
import nbformat

SRC=Path('/home/nas3/biod/xueshuailin/CCC_Phe/workspace/outputs/harmonics_pf/joint_two_v3')
DATA=Path('/home/nas3/biod/xueshuailin/CCC_Phe/workspace/data/harmonics_pf/joint_two_native_v3_input.h5ad')
OUT=Path('/data1/xueshuailin/CCC_Phe/workspace/outputs/harmonics_pf/deep_analysis');OUT.mkdir(parents=True,exist_ok=True)
NB=Path('/home/xueshuailin/CCC_Phe/Harmonics_Pulmonary_Fibrosis_V3.ipynb')
a=ad.read_h5ad(DATA);z=np.load(SRC/'predictions.npz');assert np.array_equal(a.obs_names.astype(str),z['cell_ids'].astype(str))
sample=z['sample_id'];lab=z['labels'];xy=a.obsm['spatial'];ct=a.obs.cell_type.astype(str).to_numpy()
print('Loaded',a.shape,'and verified cell alignment',flush=True)
counts=pd.crosstab(pd.Series(sample,name='sample'),pd.Series(lab,name='niche')).reindex(index=[0,1],columns=range(8),fill_value=0)
frac=counts.div(counts.sum(axis=1),axis=0);frac.to_csv(OUT/'niche_proportions.csv')
comp=pd.crosstab(pd.Series(lab,name='niche'),pd.Series(ct,name='cell_type'),normalize='index').reindex(range(8)).fillna(0)
comp.to_csv(OUT/'niche_cell_fractions.csv')
groups=[]
for s in (0,1):
    for k in np.argsort(counts.loc[s].to_numpy())[::-1][:4]:groups.append((s,int(k)))
rng=np.random.default_rng(42); selected={}
for s,k in groups:
    ids=np.flatnonzero((sample==s)&(lab==k));selected[s,k]=np.sort(rng.choice(ids,size=min(2000,len(ids)),replace=False))
group_names=[('Ctrl' if s==0 else 'PF')+f' · N{k+1}' for s,k in groups]
print('Group samples',[(group_names[i],len(selected[g])) for i,g in enumerate(groups)],flush=True)

def pooled(view,mask,support,category):
    a=np.load(SRC/(view+'.npy'),mmap_mode='r');m=np.load(SRC/(mask+'.npy'),mmap_mode='r');meta=pd.read_csv(SRC/(support+'.csv')); assert a.shape[1]==len(meta)
    cats=meta[category].astype(str).to_numpy(); names=sorted(set(cats));mat=np.full((len(groups),len(names)),np.nan);support_counts=np.zeros_like(mat)
    for i,g in enumerate(groups):
        rows=selected[g]; vals=np.asarray(a[rows],dtype=np.float32);ok=np.asarray(m[rows],dtype=bool)
        for j,n in enumerate(names):
            v=(cats==n);valid=ok[:,v];support_counts[i,j]=valid.sum()
            if valid.sum()>=100:mat[i,j]=float(np.where(valid,vals[:,v],0).sum()/valid.sum())
    return names,mat,support_counts
pn,pm,ps=pooled('exposure','exposure_mask','exposure_support','pathway');rn,rm,rs=pooled('response','response_mask','response_support','program')
pd.DataFrame(pm,index=group_names,columns=pn).to_csv(OUT/'pathways_support_pooled.csv');pd.DataFrame(ps,index=group_names,columns=pn).to_csv(OUT/'pathways_support_counts.csv')
pd.DataFrame(rm,index=group_names,columns=rn).to_csv(OUT/'response_support_pooled.csv');pd.DataFrame(rs,index=group_names,columns=rn).to_csv(OUT/'response_support_counts.csv')
print('Computed support-pooled pathways and responses',flush=True)

def select_z(names,mat,n=14):
    # Ranking is post-hoc visualization only, not significance or feature selection for fitting.
    ok=np.sum(np.isfinite(mat),axis=0)>=max(3,len(groups)//2);names=np.asarray(names)[ok];v=mat[:,ok]
    mean=np.nanmean(v,axis=0);sd=np.nanstd(v,axis=0);good=sd>1e-8;v=v[:,good];mean=mean[good];sd=sd[good];names=names[good]
    zz=(v-mean)/sd;rank=np.argsort(-np.nanmax(np.abs(zz),axis=0))[:n];rank=sorted(rank,key=lambda j:(np.nanargmax(zz[:,j]),-np.nanmax(zz[:,j])))
    return names[rank],np.clip(zz[:,rank],-2,2)
pn2,pz=select_z(pn,pm,n=8);rn2,rz=select_z(rn,rm,n=8)

# Inspect supported directed LR opportunities for a prespecified set of lung
# injury-related ligands. Keep sender/receiver labels; no expression-significance claim.
folder=Path(json.loads((SRC/'target_cache.json').read_text())['folder']);features=pd.read_csv(folder/'features.csv')
keep=features.ligand.isin(['TGFB1','TGFB2','SPP1','CXCL9','CXCL10','CXCL12','CCL2','WNT5A'])
fi=np.flatnonzero(keep.to_numpy()); ff=features.iloc[fi].reset_index(drop=True)
ff.to_csv(OUT/'candidate_directed_lr_features.csv',index=False)
cats=pd.Categorical(a.obs.cell_type.astype(str)).categories;lookup={v:i for i,v in enumerate(cats)}
pair=np.array([lookup[s]*len(cats)+lookup[r] for s,r in zip(ff.sender,ff.receiver)])
ccc=np.load(folder/'ccc.npy',mmap_mode='r');opp=np.load(folder/'opportunity.npy',mmap_mode='r')
lr=np.full((len(groups),len(fi)),np.nan)
for i,g in enumerate(groups):
    rows=selected[g][::max(1,len(selected[g])//400)][:400];v=np.asarray(ccc[rows])[:,fi];mask=np.asarray(opp[rows])[:,pair]>0
    n=mask.sum(axis=0);lr[i]=np.divide(np.where(mask,v,0).sum(axis=0),n,out=np.full(len(fi),np.nan),where=n>=20)
names=(ff.sender+' → '+ff.receiver+' · '+ff.ligand+'–'+ff.receptor).to_numpy()
valid=(np.sum(lr>0,axis=0)>=6)&(np.sum(np.isfinite(lr),axis=0)>=6)
scores=np.nanmax(lr[:,valid],axis=0)-np.nanmin(lr[:,valid],axis=0)
top=np.argsort(-scores)[:8];ln=names[valid][top];lz=np.log1p(lr[:,valid][:,top])
pd.DataFrame(lr,index=group_names,columns=names).to_csv(OUT/'directed_lr_supported_means.csv')
print('Computed directed LR supported means',flush=True)

plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'axes.spines.top':False,'axes.spines.right':False,'figure.facecolor':'white'})
palette=np.array(['#407e9b','#e49c63','#68a899','#a488b7','#b86068','#d2b04e','#668c60','#8e7081'])
fig=plt.figure(figsize=(11.5,21),layout='constrained');gs=fig.add_gridspec(5,2,height_ratios=[1.2,.62,.85,.85,1.05])
for s,title in [(0,'Control · THD0008'),(1,'Pulmonary fibrosis · VUILD106MA')]:
 ax=fig.add_subplot(gs[0,s]);ids=np.flatnonzero(sample==s);ax.scatter(xy[ids,0],xy[ids,1],c=palette[lab[ids]],s=.25,lw=0,rasterized=True);ax.set_aspect('equal');ax.invert_yaxis();ax.axis('off');ax.set_title(title+' · joint V3 niches',fontsize=11)
fig.legend(handles=[Line2D([],[],marker='o',lw=0,color=palette[i],label=f'N{i+1}',markersize=5) for i in range(8)],loc='outside upper center',ncol=8,frameon=False,fontsize=8)
ax=fig.add_subplot(gs[1,0]);left=np.zeros(2)
for k in range(8):
 vals=frac[k].to_numpy();ax.barh([0,1],vals,left=left,color=palette[k],height=.55);left+=vals
ax.set_yticks([0,1],['Control','Fibrosis']);ax.invert_yaxis();ax.set_xlim(0,1);ax.set_xlabel('Fraction of cells');ax.set_title('Niche distribution',loc='left',fontweight='bold')
ax=fig.add_subplot(gs[1,1]);preferred=['Activated Fibrotic FBs','Alveolar FBs','Transitional AT2','AT2','CD4+ T-cells','CD8+ T-cells','B cells','Plasma','Multiciliated','Basal']
cols=[x for x in preferred if x in comp];vals=comp[cols].T.to_numpy();im=ax.imshow(vals,aspect='auto',cmap='YlGnBu',vmin=0,vmax=max(.25,float(vals.max())));ax.set_yticks(range(len(cols)),cols,fontsize=7);ax.set_xticks(range(8),[f'N{k+1}' for k in range(8)]);ax.set_title('Cell types within each niche',loc='left',fontweight='bold');fig.colorbar(im,ax=ax,fraction=.035,pad=.02,label='Fraction')
for grid,title,names,values in [(gs[2,:],'Communication pathways',pn2,pz),(gs[3,:],'Receiver response',rn2,rz),(gs[4,:],'Directed LR · sender → receiver',ln,lz)]:
 import textwrap
 ax=fig.add_subplot(grid);is_lr=grid==gs[4,:]
 cmap=plt.get_cmap('YlGnBu' if is_lr else 'RdBu_r').copy();cmap.set_bad('#e1e5e7')
 im=ax.imshow(np.ma.masked_invalid(values.T),aspect='auto',cmap=cmap,norm=None if is_lr else TwoSlopeNorm(vmin=-2,vcenter=0,vmax=2),interpolation='none')
 labels=[str(x).replace('Signaling by ','').replace(' · ','\n') for x in names]
 labels=['\n'.join(textwrap.wrap(x,width=38,break_long_words=False,break_on_hyphens=False)) if '\n' not in x else x for x in labels]
 ax.set_yticks(range(len(labels)),labels,fontsize=8);ax.set_xticks(range(len(groups)),[x.replace(' · ','\n') for x in group_names],fontsize=8);ax.xaxis.tick_top();ax.tick_params(axis='both',length=0,pad=5)
 ax.axvline(3.5,color='white',lw=2.5);ax.set_title(title,loc='left',fontweight='bold',pad=35)
 fig.colorbar(im,ax=ax,fraction=.016,pad=.015,label='log(1 + mean CCC)' if is_lr else 'Within-feature z score')
fig.savefig(OUT/'overview.png',dpi=140,bbox_inches='tight');plt.close(fig)
print('Figure saved',flush=True)

info={'samples':['THD0008','VUILD106MA'],'view':'joint V3 K=8','groups':group_names,'sampled_cells_per_group':{group_names[i]:len(selected[g]) for i,g in enumerate(groups)},'pooling':'mean of supported cell-type-directed features; each category aggregates available sender/receiver opportunities; LR uses up to 400 centers per group','heatmap':'post-hoc high-variation examples; pathways and response column z-scores; LR log1p means; no significance','inference':'one specimen per condition; descriptive only; LR is a modeled proxy, not direct evidence of signaling'}
(OUT/'analysis_manifest.json').write_text(json.dumps(info,indent=2))
source="""# Exploratory composition and communication analysis on the two existing sections\nfrom IPython.display import display, Image\nimport json\nfrom pathlib import Path\np = Path('/data1/xueshuailin/CCC_Phe/workspace/outputs/harmonics_pf/deep_analysis')\ndisplay(Image(filename=str(p/'overview.png'), width=1100))\nprint(json.loads((p/'analysis_manifest.json').read_text())['inference'])\n"""
n=nbformat.read(NB,4);tag='pf_deep_analysis_20261116'
matches=[c for c in n.cells if tag in c.get('metadata',{}).get('tags',[])]
if matches:
 c=matches[0]
 if c.source!=source:raise RuntimeError('Existing user cell changed; refusing to overwrite')
else:
 c=nbformat.v4.new_code_cell(source,metadata={'tags':[tag]});n.cells.append(c)
png=base64.b64encode((OUT/'overview.png').read_bytes()).decode();c.outputs=[nbformat.v4.new_output('display_data',data={'image/png':png},metadata={}),nbformat.v4.new_output('stream',name='stdout',text=info['inference']+'\n')];nbformat.write(n,NB)
print('Completed',NB,flush=True)
