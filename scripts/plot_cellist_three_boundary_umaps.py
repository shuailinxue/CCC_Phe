"""UMAP composition and existing h for the three author-boundary groups."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS'):os.environ.setdefault(k,'4')
from pathlib import Path
import json,base64,time
import numpy as np,pandas as pd,anndata as ad,umap,nbformat
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import fontManager,FontProperties
from matplotlib.lines import Line2D
W=Path('/data1/xueshuailin/CCC_Phe/workspace');B=W/'outputs/nsclc_zhencaiwei/ST1_ST5/v3_joint';R=W/'outputs/nsclc_zhencaiwei/ST1_ST5/v3_joint_K12';O=W/'outputs/nsclc_zhencaiwei/ST1_ST5/three_boundary_umaps';O.mkdir(parents=True,exist_ok=True)
q=np.load(R/'predictions.npz');ids=q['cell_ids'].astype(str);sid=q['sample_id'];lab=q['labels'];h=np.load(B/'h.npy',mmap_mode='r');target=Path(json.loads((B/'target_cache.json').read_text())['folder']);comp=np.load(target/'composition.npy',mmap_mode='r');bounds=[]
for s in ('ST1','ST5'):
 a=ad.read_h5ad(W/f'data/nsclc_zhencaiwei/processed/{s}/{s}_counts.h5ad',backed='r');assert np.array_equal(np.char.add(s+'::',a.obs_names.to_numpy(dtype=str)),ids[sid==({'ST1':0,'ST5':1}[s])]);bounds.append(a.obs['Tumor_boundary'].astype(str).to_numpy());a.file.close()
mask=np.isin(np.concatenate(bounds),['Inner 1','Inner 2','Outer 1','Outer 2']);valid=(np.abs(comp.sum(1)-1)<1e-3)&np.isfinite(h).all(1)
groups=[('ST1 边界 N8',(sid==0)&mask&(lab==7)&valid),('ST1 边界其他类',(sid==0)&mask&(lab!=7)&valid),('ST5 全部边界',(sid==1)&mask&valid)];rng=np.random.default_rng(42);samples=[];full=[]
for label,sel in groups:
 ix=np.flatnonzero(sel);full.append(len(ix));samples.append(np.sort(rng.choice(ix,min(len(ix),2000),replace=False)))
idx=np.concatenate(samples);g=np.concatenate([np.full(len(x),i) for i,x in enumerate(samples)]);print('full groups',full,'sample sizes',[len(x) for x in samples],flush=True)
x=np.sqrt(np.maximum(comp[idx],0));v=np.asarray(h[idx],np.float32);emb=[]
for name,data in [('composition',x),('h',v)]:
 print(time.strftime('%F %T'),name,'UMAP',len(data),flush=True);coords=umap.UMAP(n_neighbors=30,min_dist=.3,metric='euclidean',random_state=42,n_jobs=1).fit_transform(data);np.save(O/f'umap_{name}.npy',coords);emb.append(coords)
np.savez_compressed(O/'sampling.npz',indices=idx,cell_ids=ids[idx],group=g)
fontManager.addfont('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc');font=FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc');plt.rcParams['font.family']=font.get_name();colors=np.array(['#d85150','#5682b0','#e7a746']);fig,axs=plt.subplots(1,2,figsize=(12.5,5.5),layout='constrained');order=rng.permutation(len(idx))
for ax,coords,title in zip(axs,emb,['组成向量 UMAP','通讯 h UMAP']):
 ax.scatter(coords[order,0],coords[order,1],c=colors[g[order]],s=6,alpha=.7,lw=0,rasterized=True);ax.set_title(title,fontproperties=font,fontsize=16);ax.set_xticks([]);ax.set_yticks([])
fig.legend(handles=[Line2D([],[],marker='o',ls='',color=colors[i],label=f'{groups[i][0]}（抽样 {len(samples[i]):,}）',markersize=7) for i in range(3)],loc='outside lower center',ncol=3,frameon=False,prop=font);fig.savefig(O/'boundary_umaps.png',dpi=160);plt.close(fig)
meta={'groups':[{ 'label':name,'available':full[i],'sampled':len(samples[i])} for i,(name,_) in enumerate(groups)],'boundary':'author Inner1/2 Outer1/2','representation':'sqrt composition vs saved jointly trained h, independently fitted UMAPs','parameters':'seed42 n_neighbors30 min_dist0.3; balanced sampling up to 2000 per group','warning':'sample sizes equalized for visibility; do not infer tissue abundance from points; groups used for color only, not fitting; h trained jointly on slices'};(O/'protocol.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2))
nbpath=Path('/home/xueshuailin/CCC_Phe/Cellist_ST1_ST5_Joint_V3.ipynb');n=nbformat.read(nbpath,4);tag='cellist_three_boundary_umaps';source='''# 三组肿瘤边界：组成向量和通讯 h 各自的 UMAP\nfrom pathlib import Path\nimport json, pandas as pd\nfrom IPython.display import display, Image\np=Path('/data1/xueshuailin/CCC_Phe/workspace/outputs/nsclc_zhencaiwei/ST1_ST5/three_boundary_umaps')\ndisplay(Image(filename=str(p/'boundary_umaps.png'),width=1100))\ninfo=json.loads((p/'protocol.json').read_text());display(pd.DataFrame(info['groups']))\nprint('两图使用同一批微环境，组别只用于着色；左图仅使用组成向量，右图仅使用已有联合 h。')\nprint('每组等量抽样，图上点数不表示原始区域占比；边界是作者 Inner 1–2 和 Outer 1–2。')\n''';cell=nbformat.v4.new_code_cell(source,metadata={'tags':[tag]});cell.outputs=[nbformat.v4.new_output('display_data',data={'image/png':base64.b64encode((O/'boundary_umaps.png').read_bytes()).decode()},metadata={}),nbformat.v4.new_output('stream',name='stdout',text=pd.DataFrame(meta['groups']).to_string(index=False)+'\n两图各自拟合，组别仅用于着色；每组最多抽样 2000 个。\n')];found=[i for i,c in enumerate(n.cells) if tag in c.get('metadata',{}).get('tags',[])];
if found:n.cells[found[0]]=cell
else:n.cells.append(cell)
nbformat.write(n,nbpath);print('COMPLETE notebook updated',flush=True)
