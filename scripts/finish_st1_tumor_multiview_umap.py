"""Render saved ST1 tumor model and sampled UMAP without retraining."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS'):os.environ.setdefault(key,'4')
import base64,json,time
from pathlib import Path
import numpy as np,pandas as pd,nbformat,umap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
W=Path('/data1/xueshuailin/CCC_Phe/workspace');B=W/'outputs/nsclc_zhencaiwei/ST1_ST5/v3_joint';O=W/'outputs/nsclc_zhencaiwei/ST1_ST5/st1_tumor_three_vectors_fresh_ae';NB=Path('/home/xueshuailin/CCC_Phe/Cellist_ST1_ST5_Joint_V3.ipynb');T=Path(json.loads((B/'target_cache.json').read_text())['folder']);c=np.load(T/'composition.npy',mmap_mode='r');ids=np.load(O/'selected_global_indices.npy');h=np.load(O/'h.npy');assert len(ids)==len(h)==26948 and np.isfinite(h).all()
idx=np.sort(np.random.default_rng(42).choice(len(ids),6000,replace=False));np.save(O/'umap_sample_indices.npy',idx);data=[np.sqrt(np.maximum(c[ids[idx]],0)),h[idx]];emb=[]
for name,v in zip(['composition','h'],data):
 print(time.strftime('%F %T'),name,'UMAP',len(v),flush=True)
 x=umap.UMAP(n_neighbors=30,min_dist=.3,metric='euclidean',random_state=42,n_jobs=1).fit_transform(v);np.save(O/f'umap_{name}_sampled.npy',x);emb.append(x)
mal=c[ids[idx],4];order=np.random.default_rng(42).permutation(len(idx));fig,axs=plt.subplots(1,2,figsize=(12,5.3),layout='constrained')
for ax,x,title in zip(axs,emb,['ST1 tumor · composition UMAP','ST1 tumor · new h UMAP']):
 sc=ax.scatter(x[order,0],x[order,1],c=mal[order],cmap='viridis',vmin=.25,vmax=1,s=3,lw=0,rasterized=True);ax.set_title(title+'\n6,000 sampled windows');ax.set_xticks([]);ax.set_yticks([])
fig.colorbar(sc,ax=axs,label='Malignant fraction in microenvironment',shrink=.85);fig.savefig(O/'composition_and_h_umap.png',dpi=160);plt.close(fig)
meta={'n_microenvironments':len(ids),'epochs':100,'umap_sample':len(idx),'selection':'ST1 microenvironments with malignant fraction >=0.25','input':'three stored communication vectors for these ST1 rows; no ST5 training rows','caution':'some input feature preprocessing and prior calculations used ST1/ST5 jointly','UMAP':'same randomly selected 6000 windows in both panels; raw 64D h and sqrt composition separately; n_neighbors30 min_dist0.3 seed42','plotting_reason':'all-26948 composition UMAP exceeded one hour; uses reproducible subset instead of restarting training'};(O/'complete.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2))
n=nbformat.read(NB,4);tag='cellist_st1_tumor_three_vectors_fresh_ae';source='''# ST1 肿瘤区域独立训练：组成与新 h 的 UMAP\nfrom pathlib import Path\nimport json\nfrom IPython.display import display, Image\np=Path('/data1/xueshuailin/CCC_Phe/workspace/outputs/nsclc_zhencaiwei/ST1_ST5/st1_tumor_three_vectors_fresh_ae')\ninfo=json.loads((p/'complete.json').read_text())\ndisplay(Image(filename=str(p/'composition_and_h_umap.png'),width=1100))\nprint('训练使用 ST1 肿瘤区域',info['n_microenvironments'],'个微环境；两图展示同一批随机抽样的',info['umap_sample'],'个。')\nprint('颜色：邻域肿瘤细胞比例；输入为这批细胞已有的三个通讯向量，模型从头训练 100 轮。')\nprint('原始向量的部分上游处理曾使用两张切片。肿瘤区域使用邻域恶性细胞比例 ≥25% 近似，不是病理区域标注。')\n''';cell=nbformat.v4.new_code_cell(source,metadata={'tags':[tag]});cell.outputs=[nbformat.v4.new_output('display_data',data={'image/png':base64.b64encode((O/'composition_and_h_umap.png').read_bytes()).decode()},metadata={}),nbformat.v4.new_output('stream',name='stdout',text=f"训练使用 26,948 个 ST1 肿瘤微环境，各图展示同一批随机抽样的 6,000 个；模型从头训练 100 轮。颜色表示邻域肿瘤细胞比例。\n")];idxs=[i for i,x in enumerate(n.cells) if tag in x.get('metadata',{}).get('tags',[])];
if idxs:n.cells[idxs[0]]=cell
else:n.cells.append(cell)
nbformat.write(n,NB);print('COMPLETE notebook updated',flush=True)
