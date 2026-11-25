"""Joint h UMAP for tumor-core N4 and tumor-edge N9 in ST1 and ST5."""
import os
for var in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS'): os.environ.setdefault(var,'4')
import time,json,base64
from pathlib import Path
import anndata as ad,numpy as np,pandas as pd
from sklearn.neighbors import NearestNeighbors
import umap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import nbformat
W=Path('/data1/xueshuailin/CCC_Phe/workspace');B=W/'outputs/nsclc_zhencaiwei/ST1_ST5/v3_joint';R=W/'outputs/nsclc_zhencaiwei/ST1_ST5/composition_consensus_analysis';O=W/'outputs/nsclc_zhencaiwei/ST1_ST5/core_edge_h_umap';O.mkdir(parents=True,exist_ok=True)
z=np.load(R/'labels.npz'); ids=z['cell_ids'];sid=z['sample_id'];labels=z['labels'];h=np.load(B/'h.npy',mmap_mode='r');target=Path(json.loads((B/'target_cache.json').read_text())['folder']);p=np.load(target/'composition.npy',mmap_mode='r');pairs=pd.read_csv(R/'matched_pairs.csv');a=ad.read_h5ad(W/'data/nsclc_zhencaiwei/ST1_ST5_joint_v3_input.h5ad',backed='r');assert np.array_equal(a.obs_names.to_numpy(dtype=str),ids.astype(str));xy=np.asarray(a.obsm['spatial']);a.file.close()
types=['Epithelial','Epithelial-Airway','Fibroblast','Lymphocyte','Malignant','Mast','Myeloid','Plasma','SMC'];mal=p[:,types.index('Malignant')];immune=p[:,[types.index(x) for x in ['Lymphocyte','Plasma','Myeloid','Mast']]].sum(1)
regions=[('tumor_core','Tumor core · N4',3,(mal>=.70)),('tumor_edge','Tumor edge · N9',8,(mal>=.25)&(mal<.70)&(immune>=.15))]
colors=np.array(['#3a78b0','#e4773c']);fig,axs=plt.subplots(2,4,figsize=(17,8),layout='constrained');summary=[];rng=np.random.default_rng(20261116)
for r,(region,title,k,roi) in enumerate(regions):
 mask=(labels==k)&roi; pair=pairs[(pairs.region==region)&(pairs.niche==k)].copy();pair=pair.sample(n=min(len(pair),1200),random_state=42);aidx=pair.ST1_index.to_numpy(int);bidx=pair.ST5_index.to_numpy(int);assert np.all(mask[aidx]) and np.all(mask[bidx])
 selected=[];matched=[];totals=[]
 for s,mand in [(0,aidx),(1,bidx)]:
  candidates=np.flatnonzero((sid==s)&mask);totals.append(len(candidates));rest=np.setdiff1d(candidates,mand,assume_unique=False);extra=rng.choice(rest,size=min(len(rest),max(0,3000-len(mand))),replace=False);selected.append(np.concatenate([mand,extra]));matched.append(np.r_[np.ones(len(mand),bool),np.zeros(len(extra),bool)])
 ix=np.concatenate(selected);which=sid[ix];is_matched=np.concatenate(matched);v=np.asarray(h[ix],dtype='float32');print(time.strftime('%F %T'),title,'sampled',len(v),'matched pairs',len(aidx),flush=True)
 model=umap.UMAP(n_neighbors=30,min_dist=.3,metric='euclidean',n_components=2,random_state=42,n_jobs=1);emb=model.fit_transform(v)
 np.savez_compressed(O/(region+'_embedding.npz'),cell_ids=ids[ix],sample_id=which,embedding=emb,matched=is_matched,global_indices=ix)
 # Quantify local sample separation in original h, not in UMAP coordinates.
 def same_slice_fraction(sub):
  x=v[sub];y=which[sub];knn=NearestNeighbors(n_neighbors=31,n_jobs=4).fit(x).kneighbors(x,return_distance=False)[:,1:];return float((y[knn]==y[:,None]).mean())
 all_mix=same_slice_fraction(np.ones(len(ix),bool));matched_mix=same_slice_fraction(is_matched)
 d=np.asarray(pair.comp_distance);summary.append({'region':region,'joint_composition_niche':k+1,'ST1_available':totals[0],'ST5_available':totals[1],'ST1_sampled':len(selected[0]),'ST5_sampled':len(selected[1]),'matched_pairs_in_plot':len(pair),'matched_composition_distance_median':float(np.median(d)),'matched_composition_distance_p95':float(np.quantile(d,.95)),'h_30NN_same_slice_fraction_all':all_mix,'h_30NN_same_slice_fraction_matched':matched_mix})
 for s in (0,1):
  ax=axs[r,s];bg=np.flatnonzero(sid==s);target_ix=np.flatnonzero((sid==s)&mask);ax.scatter(xy[bg,0],xy[bg,1],c='#e5e5e5',s=.13,lw=0,rasterized=True);ax.scatter(xy[target_ix,0],xy[target_ix,1],c=('#ffbb78' if k==3 else '#9467bd'),s=.5,lw=0,rasterized=True);ax.set_aspect('equal');ax.axis('off');ax.set_title(f'{["ST1","ST5"][s]} · {title}\nn={totals[s]:,}')
 for col,sub,label in [(2,np.ones(len(ix),bool),'All sampled'),(3,is_matched,'Composition-matched pairs')]:
  ax=axs[r,col];order=rng.permutation(np.flatnonzero(sub));ax.scatter(emb[order,0],emb[order,1],c=colors[which[order]],s=2,alpha=.7,lw=0,rasterized=True);ax.set_xticks([]);ax.set_yticks([]);ax.set_title(f'{title} · h UMAP\n{label}')
  ax.legend(handles=[Line2D([],[],marker='o',ls='',c=colors[s],label=['ST1','ST5'][s],markersize=5) for s in (0,1)],loc='best',fontsize=7,frameon=False)
fig.savefig(O/'core_edge_h_umap.png',dpi=150);plt.close(fig);pd.DataFrame(summary).to_csv(O/'summary.csv',index=False)
info={'representation':'saved 64-dimensional h trained jointly on both sections; UMAP fitted separately for core and edge on raw h only, no composition, section label or tissue coordinates','parameters':{'n_neighbors':30,'min_dist':.3,'seed':42,'max_per_section':3000,'matched_pairs_max':1200},'region_definitions':'Core: joint composition N4 and local malignant proportion >=0.70. Edge: joint composition N9 and malignant proportion >=0.25 and <0.70, immune proportion >=0.15. Regions inferred by composition, not pathology annotation.','matched':'1:1 pairs selected from previous matching within same composition cluster, region and center type; sqrt-composition distance <=0.20; up to 1200 pairs per region included; other h-UMAP sample balanced across sections.','knn':'same-slice fraction among 30 nearest neighbors in original h, balanced sections, descriptive only; UMAP is visualization, not a statistical test.','limitations':'two individual sections, overlapping neighborhoods, unverified coordinate scaling; cannot separate biological and batch effects or establish statistical significance','results':summary}
(O/'protocol.json').write_text(json.dumps(info,ensure_ascii=False,indent=2))
nbpath=Path('/home/xueshuailin/CCC_Phe/Cellist_ST1_ST5_Joint_V3.ipynb');n=nbformat.read(nbpath,4);tag='cellist_core_edge_h_umap';source='''# 肿瘤核心和边缘：ST1/ST5 的 h UMAP\nfrom pathlib import Path\nimport json, pandas as pd\nfrom IPython.display import display, Image\np=Path('/data1/xueshuailin/CCC_Phe/workspace/outputs/nsclc_zhencaiwei/ST1_ST5/core_edge_h_umap')\ndisplay(Image(filename=str(p/'core_edge_h_umap.png'),width=1200))\ninfo=json.loads((p/'protocol.json').read_text())\ndisplay(pd.read_csv(p/'summary.csv').round(3))\nprint('左两列：选中的肿瘤核心 N4、边缘 N9 在两张切片的位置；右两列：仅用原始 64 维 h 做联合 UMAP，分别展示均衡抽样和组成匹配的配对。')\nprint('组成匹配：相同组成类、区域代理与中心细胞类型，sqrt 组成距离 <= 0.20，一对一；h 30 近邻同切片比例在原始 h 中计算，仅为描述。')\nprint('核心与边缘按邻域组成近似定义，并非病理标注；每组仅一张切片，不能排除批次或样本差异。')\n''';cell=nbformat.v4.new_code_cell(source,metadata={'tags':[tag]});cell.outputs=[nbformat.v4.new_output('display_data',data={'image/png':base64.b64encode((O/'core_edge_h_umap.png').read_bytes()).decode()},metadata={}),nbformat.v4.new_output('stream',name='stdout',text=pd.DataFrame(summary).round(4).to_string(index=False)+'\n\n仅用 h 构图；组成为切片匹配和区域定义，未进入 UMAP。组间区分并不能证明原因。\n')];found=[i for i,c in enumerate(n.cells) if tag in c.get('metadata',{}).get('tags',[])];
if found:n.cells[found[0]]=cell
else:n.cells.append(cell)
nbformat.write(n,nbpath);print('COMPLETE',O,flush=True)
