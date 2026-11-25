"""ST1/ST5 composition consensus and composition-matched communications.

No phenotypes/regions/communications enter K selection. ROI definitions are
composition-based proxies and all two-section differences are descriptive.
"""
import os
for x in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ.setdefault(x,'4')
import json,sys,base64,time
from pathlib import Path
import numpy as np,pandas as pd,anndata as ad
from sklearn.cluster import KMeans,AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score
from sklearn.neighbors import NearestNeighbors
from scipy.optimize import linear_sum_assignment
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import nbformat
ROOT=Path('/home/xueshuailin/CCC_Phe');W=Path('/data1/xueshuailin/CCC_Phe/workspace');B=W/'outputs/nsclc_zhencaiwei/ST1_ST5/v3_joint';OUT=W/'outputs/nsclc_zhencaiwei/ST1_ST5/composition_consensus_analysis';OUT.mkdir(parents=True,exist_ok=True)
def log(*x):print(time.strftime('%F %T'),*x,flush=True)
target=Path(json.loads((B/'target_cache.json').read_text())['folder']);p=np.load(target/'composition.npy',mmap_mode='r');h=np.load(B/'h.npy',mmap_mode='r');pred=np.load(B/'predictions.npz');sid=pred['sample_id'];ids=pred['cell_ids']
a=ad.read_h5ad(W/'data/nsclc_zhencaiwei/ST1_ST5_joint_v3_input.h5ad',backed='r');assert np.array_equal(a.obs_names.to_numpy(dtype=str),ids.astype(str));xy=np.asarray(a.obsm['spatial']);ct=a.obs.cell_type.astype(str).to_numpy();a.file.close()
n=len(ids);assert len(p)==len(h)==len(sid)==n
names=sorted(np.unique(ct));assert len(names)==p.shape[1]
valid=(np.abs(p.sum(1)-1)<1e-3)&np.isfinite(h).all(1);print('N',n,'valid',int(valid.sum()),'type columns',names,flush=True)
X=np.sqrt(np.maximum(np.asarray(p,dtype=np.float32),0));rng=np.random.default_rng(20261116)
# Fixed heldout anchor windows are sampled equally across sections and excluded
# from every model fit; label comparison is purely composition-based.
anchor=np.sort(np.concatenate([rng.choice(np.flatnonzero((sid==s)&valid),size=1200,replace=False) for s in (0,1)]));pool=[]
for s in (0,1):
 index=np.flatnonzero((sid==s)&valid);pool.append(np.setdiff1d(index,anchor,assume_unique=True))
ks=[6,8,10,12];reps=10;results=[];assignments={};consensus_labels={}
for k in ks:
 labels=[]
 for r in range(reps):
  gen=np.random.default_rng(1000+r);fitix=np.concatenate([gen.choice(pool[s],min(16000,len(pool[s])),replace=False) for s in (0,1)])
  model=KMeans(n_clusters=k,n_init=5,random_state=4000+r,max_iter=200).fit(X[fitix]);labels.append(model.predict(X[anchor]));log('consensus K',k,'run',r+1,'/',reps)
 labels=np.stack(labels);co=np.zeros((len(anchor),len(anchor)),np.float32)
 for y in labels:co+=(y[:,None]==y[None,:])/reps
 upper=co[np.triu_indices(len(anchor),1)];pac=float(((upper>.1)&(upper<.9)).mean());avgari=float(np.mean([adjusted_rand_score(labels[i],labels[j]) for i in range(reps) for j in range(i)]))
 consensus=AgglomerativeClustering(n_clusters=k,metric='precomputed',linkage='average').fit_predict(np.clip(1-co,0,1));counts=np.bincount(consensus,minlength=k);within=np.mean([co[np.ix_(consensus==j,consensus==j)][np.triu_indices(counts[j],1)].mean() for j in range(k) if counts[j]>1]); minfrac=float(counts.min()/len(anchor))
 results.append({'K':k,'PAC':pac,'mean_pairwise_ARI':avgari,'mean_within_consensus':float(within),'min_anchor_fraction':minfrac,'anchor_cells':len(anchor),'runs':reps})
 assignments[k]=labels;consensus_labels[k]=consensus
 pd.DataFrame(results).to_csv(OUT/'K_consensus_stability.csv',index=False);log('K done',results[-1])
# Predeclared: among Ks whose every consensus cluster >=0.5%, choose minimum
# PAC; if within 0.01 of the minimum choose smaller K. Slice IDs are excluded.
tab=pd.DataFrame(results);eligible=tab[tab.min_anchor_fraction>=.005]
if eligible.empty:raise ValueError('No candidate K meets 0.5% anchor support')
cut=float(eligible.PAC.min()+.01);K=int(eligible.loc[eligible.PAC<=cut,'K'].min());log('selected K',K,'PAC minimum tolerance',cut)
cons=consensus_labels[K];centers=np.stack([X[anchor[cons==j]].mean(0) for j in range(K)])
# Fit all windows jointly. Equal slice weights prevent larger slice dominating;
# clustering input is only sqrt composition. No h/region/slice identity feature.
weight=np.zeros(n,np.float32)
for s in (0,1):weight[(sid==s)&valid]=.5/np.sum((sid==s)&valid)
final=KMeans(n_clusters=K,init=centers,n_init=1,random_state=42,max_iter=300).fit(X[valid],sample_weight=weight[valid]);lab=np.full(n,-1,np.int16);lab[valid]=final.labels_
np.savez_compressed(OUT/'labels.npz',cell_ids=ids,sample_id=sid,labels=lab,anchor=anchor,anchor_consensus=cons,centers=final.cluster_centers_)
fractions=pd.DataFrame(p[valid]).groupby(lab[valid]).mean();fractions.columns=names;fractions.index.name='niche';fractions.to_csv(OUT/'composition_centers.csv')
# Region proxies defined once from observed physical composition, after K choice.
mal=p[:,names.index('Malignant')]; immune=p[:,[names.index(x) for x in ('Lymphocyte','Plasma','Myeloid','Mast')]].sum(1)
roi=np.full(n,'other',dtype='U16');roi[(mal>=.7)&valid]='tumor_core';roi[(mal>=.25)&(mal<.7)&(immune>=.15)&valid]='tumor_edge';roi[(mal<.25)&(immune>=.5)&valid]='immune';roi[~valid]='invalid'
regioncounts=pd.crosstab([pd.Series(sid,name='slice'),pd.Series(roi,name='region')],pd.Series(lab,name='niche'));regioncounts.to_csv(OUT/'region_cluster_counts.csv')
# Strict pair matching: same composition class, proxy region and central type;
# one-to-one without replacement, 0.2 sqrt-composition distance maximum.
# 4 nearest candidates + greedy matching; sampled max 3000/side/group.
pairs=[];summaries=[]
for region in ('tumor_core','tumor_edge','immune'):
 for k in range(K):
  paira=[];pairb=[];paird=[]
  for typ in names:
   left=np.flatnonzero(valid&(sid==0)&(lab==k)&(roi==region)&(ct==typ));right=np.flatnonzero(valid&(sid==1)&(lab==k)&(roi==region)&(ct==typ))
   if len(left)<3 or len(right)<3:continue
   gen=np.random.default_rng(60000+k*100+names.index(typ));left=gen.choice(left,min(3000,len(left)),replace=False);right=gen.choice(right,min(3000,len(right)),replace=False)
   distances,idx=NearestNeighbors(n_neighbors=min(10,len(right)),n_jobs=4).fit(X[right]).kneighbors(X[left]);used=set()
   for i in gen.permutation(len(left)):
    for d,j in zip(distances[i],idx[i]):
     if d>.2:break
     j=int(j)
     if j not in used:used.add(j);paira.append(int(left[i]));pairb.append(int(right[j]));paird.append(float(d));break
  if paira:
   aa=np.asarray(paira);bb=np.asarray(pairb);dd=np.asarray(paird);delta=np.linalg.norm(h[aa]-h[bb],axis=1);compdelta=np.abs(p[aa]-p[bb]).mean(axis=0)
   # Match the within-slice comparator to each cross-slice pair's composition
   # distance (and center type); compare h on the same subset of query pairs.
   # Reuse within-slice partners is permitted, so this is descriptive only.
   within=[]; cross_subset=[]; base_comp=[]; cross_comp=[]
   for rows in (aa,bb):
    for typ in names:
     pos=np.flatnonzero(ct[rows]==typ)
     if len(pos)<3:continue
     sub=rows[pos]; ds,js=NearestNeighbors(n_neighbors=min(64,len(sub)),n_jobs=4).fit(X[sub]).kneighbors(X[sub]);ds[:,0]=np.inf
     best=np.argmin(np.abs(ds-dd[pos,None]),axis=1);dref=ds[np.arange(len(pos)),best];ok=(dref<=.2)&(np.abs(dref-dd[pos])<=.025)
     if ok.any():
      within.extend(np.linalg.norm(h[sub[ok]]-h[sub[js[np.flatnonzero(ok),best[ok]]]],axis=1).tolist());cross_subset.extend(delta[pos[ok]].tolist());base_comp.extend(dref[ok].tolist());cross_comp.extend(dd[pos[ok]].tolist())
   enough=len(within)>=30 and len(within)>=.2*len(aa)
   summaries.append({'region':region,'niche':k+1,'pairs':len(aa),'comp_distance_median':float(np.median(dd)),'comp_distance_p95':float(np.quantile(dd,.95)),'composition_abs_mean_max':float(compdelta.max()),'cross_h_median':float(np.median(delta)),'baseline_pairs':len(within),'baseline_comp_median':float(np.median(base_comp)) if within else None,'cross_comp_for_baseline_median':float(np.median(cross_comp)) if within else None,'baseline_cross_h_median':float(np.median(cross_subset)) if enough else None,'within_h_median':float(np.median(within)) if enough else None,'h_ratio_cross_within':float(np.median(cross_subset)/max(np.median(within),1e-8)) if enough else None})
   pairs.extend((int(a),int(b),region,k,float(d)) for a,b,d in zip(aa,bb,dd))
log('matched pairs',len(pairs),'groups',len(summaries));pairdf=pd.DataFrame(pairs,columns=['ST1_index','ST5_index','region','niche','comp_distance']);pairdf.to_csv(OUT/'matched_pairs.csv',index=False);summary=pd.DataFrame(summaries);summary.to_csv(OUT/'matched_groups.csv',index=False)
# Exposure is pathway x sender x receiver; require direct opportunity support in
# both matched windows. Ranking uses standardized descriptive paired effects.
exp=np.load(B/'exposure.npy',mmap_mode='r');mask=np.load(B/'exposure_mask.npy',mmap_mode='r');meta=pd.read_csv(B/'exposure_support.csv');assert exp.shape[1]==len(meta)
records=[]
for r in summaries:
 subset=pairdf[(pairdf.region==r['region'])&(pairdf.niche==r['niche']-1)]
 if len(subset)<30:continue
 if len(subset)>1200:subset=subset.sample(1200,random_state=42)
 aa=subset.ST1_index.to_numpy(int);bb=subset.ST5_index.to_numpy(int)
 A=np.asarray(exp[aa],np.float32);C=np.asarray(exp[bb],np.float32);ma=np.asarray(mask[aa],bool);mb=np.asarray(mask[bb],bool);ok=ma&mb&np.isfinite(A)&np.isfinite(C);cnt=ok.sum(0);dif=np.where(ok,A-C,0);mean=dif.sum(0)/np.maximum(cnt,1);pooled=np.where(ok,(A+C)/2,0).sum(0)/np.maximum(cnt,1);variance=np.where(ok,((A-pooled)**2+(C-pooled)**2)/2,0).sum(0)/np.maximum(cnt,1);effect=mean/np.maximum(np.sqrt(variance),.01)
 # Larger effects must also have >=30 supporting matched pairs.
 for j in np.flatnonzero(cnt>=30):records.append({'region':r['region'],'niche':r['niche'],'pathway':meta.pathway.iloc[j],'sender':meta.sender.iloc[j],'receiver':meta.receiver.iloc[j],'matched_pairs_supported':int(cnt[j]),'mean_ST1':float(np.where(ok[:,j],A[:,j],0).sum()/cnt[j]),'mean_ST5':float(np.where(ok[:,j],C[:,j],0).sum()/cnt[j]),'ST1_minus_ST5_pooled_SD':float(effect[j])})
 del A,C,ma,mb,ok,dif
res=pd.DataFrame(records);res.to_csv(OUT/'matched_communication_pathways.csv',index=False)
if len(res):
 best=pd.concat([g.sort_values('ST1_minus_ST5_pooled_SD').head(4).assign(higher='ST5') for _,g in res.groupby(['region','niche'])]+[g.sort_values('ST1_minus_ST5_pooled_SD',ascending=False).head(4).assign(higher='ST1') for _,g in res.groupby(['region','niche'])]).sort_values(['region','niche','higher']);best.to_csv(OUT/'top_pathways.csv',index=False)
else:best=pd.DataFrame()
log('pathway records',len(res))
info={'K_selection':'10 repeated KMeans runs on balanced 32,000 composition windows; 2,400 independent balanced fixed anchor windows predicted in all runs; pair consensus PAC; among K6/8/10/12 with min consensus anchor class >=0.5%, smallest K within 0.01 of lowest PAC; class identities permuted without matching','K':K,'all_cells_clustering':'sqrt composition; consensus initial centers, weighted KMeans using all valid windows, each slice gets total weight 0.5; labels mapped to all windows','consensus_K_metrics':results,'ROI_proxies':'tumor core malignant >=0.70; tumor edge malignant 0.25 to <0.70 and immune >=0.15; immune malignant <0.25 and immune >=0.50; immune = lymphocyte+plasma+myeloid+mast; other windows excluded from ROI comparisons','matching':'within same joint class, ROI, and central type; each slice up to 3000 windows per type, greedy one-to-one, nearest 10 and sqrt-composition distance <=0.20; not all windows match','communication':'jointly pretrained h; within-slice baseline same central type and within 0.025 sqrt-composition distance of each cross-slice pair, requires >=30 and >=20% coverage; opportunity-supported matched paired pathway x sender x receiver exposure features, 30 or more supported pairs; ranking descriptive','warnings':['single slice per condition; neighboring windows overlap, matching does not establish statistical independence','ROI proxies are not pathologist annotated regions','h and pathway features come from the same shared expression data; not direct proof of signaling','pathway ranking post hoc, possible abundance, depth, batch and morphology confounding','coordinate scaling 0.5 um per native unit is unverified'],'excluded_invalid':int((~valid).sum()),'matched_groups':summaries}
(OUT/'protocol.json').write_text(json.dumps(info,ensure_ascii=False,indent=2));
# result figures: K stability, joint niche map, region proxies, matched h effect
fig,axs=plt.subplots(2,3,figsize=(15,9),layout='constrained');axs[0,0].plot(tab.K,tab.PAC,'o-',color='#446b91',label='PAC (lower better)');axs[0,0].set(xticks=ks,xlabel='K',ylabel='PAC',title='Composition consensus');ax2=axs[0,0].twinx();ax2.plot(tab.K,tab.mean_pairwise_ARI,'s--',color='#b06a4f');ax2.set_ylabel('Pairwise ARI');axs[0,0].axvline(K,color='gray',ls=':')
pal=plt.get_cmap('tab20')(np.arange(K)%20);proxies={'tumor_core':'#bd3949','tumor_edge':'#e0a538','immune':'#4279aa','other':'#d1d1d1','invalid':'#222222'}
for s in (0,1):
 ix=np.flatnonzero(sid==s);order=np.random.default_rng(42).permutation(ix);coords=xy[order];col=np.array([pal[j] if j>=0 else [.2,.2,.2,1] for j in lab[order]])
 axs[0,s+1].scatter(coords[:,0],coords[:,1],c=col,s=.35,lw=0,rasterized=True);axs[0,s+1].axis('off');axs[0,s+1].set_aspect('equal');axs[0,s+1].set_title(f'{["ST1","ST5"][s]} · composition K={K}')
 axs[1,s].scatter(coords[:,0],coords[:,1],c=[proxies[v] for v in roi[order]],s=.35,lw=0,rasterized=True);axs[1,s].axis('off');axs[1,s].set_aspect('equal');axs[1,s].set_title(f'{["ST1","ST5"][s]} · region proxies')
axs[1,0].legend(handles=[Line2D([],[],marker='o',ls='',color=c,label=t,markersize=5) for t,c in proxies.items() if t!='invalid'],loc='lower left',fontsize=7)
if len(summary):
 xx=np.arange(len(summary));axs[1,2].bar(xx,summary.h_ratio_cross_within,color='#7799a6');axs[1,2].axhline(1,color='gray',ls=':');axs[1,2].set_xticks(xx,[f"{r.region}\nN{r.niche}\nn={r.pairs}" for r in summary.itertuples()],rotation=45,ha='right',fontsize=7);axs[1,2].set(ylabel='Median cross-slice / within-slice h distance',title='Matched h difference · descriptive')
else:axs[1,2].text(.5,.5,'No sufficiently matched groups',ha='center',va='center')
fig.savefig(OUT/'overview.png',dpi=140);plt.close(fig)
# Append only after calculations complete; never replace the user's first cell.
nbpath=ROOT/'Cellist_ST1_ST5_Joint_V3.ipynb';nb=nbformat.read(nbpath,4);tag='st1_st5_composition_consensus_matched_20261116'
source='''# ST1 + ST5：组成一致聚类与组成相似微环境的通讯比较\nfrom pathlib import Path\nimport json, pandas as pd\nfrom IPython.display import display, Image\np=Path('/data1/xueshuailin/CCC_Phe/workspace/outputs/nsclc_zhencaiwei/ST1_ST5/composition_consensus_analysis')\ninfo=json.loads((p/'protocol.json').read_text())\ndisplay(Image(filename=str(p/'overview.png'),width=1100))\nprint('一致聚类选中 K =',info['K'],'；PAC 越低越稳定。')\ndisplay(pd.read_csv(p/'K_consensus_stability.csv').round(3))\nprint('区域以邻域组成近似定义，不是病理标注：',info['ROI_proxies'])\nprint('同一组成类、区域代理和中心细胞类型内，一对一匹配；组成距离不超过 0.20。')\ndisplay(pd.read_csv(p/'matched_groups.csv').round(3))\nprint('下面为支持匹配的通路 × 发送细胞 × 接收细胞差异，排序仅供探索。')\nif (p/'top_pathways.csv').exists(): display(pd.read_csv(p/'top_pathways.csv').round(3))\nprint('注意：每组只有一张切片；邻域有重叠，h 和通讯推断共用表达数据，不能视为独立重复或直接信号证据。')\n'''
cell=nbformat.v4.new_code_cell(source,metadata={'tags':[tag]});txt='一致聚类 K = '+str(K)+'\n'+tab.to_string(index=False)+'\n\n匹配组\n'+summary.to_string(index=False)+'\n\n通路候选（按组及方向各取4项）\n'+best.to_string(index=False)+'\n\n区域是邻域组成代理；单切片比较、空间重叠和同源表达输入均限制结论。\n';cell.outputs=[nbformat.v4.new_output('display_data',data={'image/png':base64.b64encode((OUT/'overview.png').read_bytes()).decode()},metadata={}),nbformat.v4.new_output('stream',name='stdout',text=txt)]
found=[i for i,c in enumerate(nb.cells) if tag in c.get('metadata',{}).get('tags',[])]
if found:nb.cells[found[0]]=cell
else:nb.cells.append(cell)
nbformat.write(nb,nbpath);log('COMPLETE notebook updated',nbpath)
