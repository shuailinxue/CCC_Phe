"""Independent within-section h UMAPs of composition-defined tumor core N4."""
import os
for var in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS'):os.environ.setdefault(var,'4')
from pathlib import Path
import json,base64,time
import numpy as np,pandas as pd,anndata as ad,umap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import nbformat
W=Path('/data1/xueshuailin/CCC_Phe/workspace');B=W/'outputs/nsclc_zhencaiwei/ST1_ST5/v3_joint';R=W/'outputs/nsclc_zhencaiwei/ST1_ST5/composition_consensus_analysis';O=W/'outputs/nsclc_zhencaiwei/ST1_ST5/individual_core_h_umap';O.mkdir(parents=True,exist_ok=True)
z=np.load(R/'labels.npz');ids=z['cell_ids'];sid=z['sample_id'];labels=z['labels'];h=np.load(B/'h.npy',mmap_mode='r');target=Path(json.loads((B/'target_cache.json').read_text())['folder']);p=np.load(target/'composition.npy',mmap_mode='r');a=ad.read_h5ad(W/'data/nsclc_zhencaiwei/ST1_ST5_joint_v3_input.h5ad',backed='r');assert np.array_equal(a.obs_names.to_numpy(dtype=str),ids.astype(str));xy=np.asarray(a.obsm['spatial']);typ=a.obs.cell_type.astype(str).to_numpy();a.file.close()
valid=(labels==3)&(p[:,4]>=.7);fig,axs=plt.subplots(2,3,figsize=(13,9),layout='constrained');rows=[];cmap=plt.get_cmap('viridis')
for s in (0,1):
 index=np.flatnonzero((sid==s)&valid);rng=np.random.default_rng(42);sel=np.sort(rng.choice(index,min(len(index),5000),replace=False));v=np.asarray(h[sel],dtype='float32');print(time.strftime('%F %T'),['ST1','ST5'][s],len(index),'sampled',len(sel),flush=True)
 emb=umap.UMAP(n_neighbors=30,min_dist=.3,metric='euclidean',random_state=42,n_jobs=1).fit_transform(v);np.savez_compressed(O/f'ST{[1,5][s]}_core_embedding.npz',cell_ids=ids[sel],global_indices=sel,embedding=emb)
 axs[s,0].scatter(emb[:,0],emb[:,1],s=2,c=['#3a78b0','#e4773c'][s],lw=0,alpha=.7,rasterized=True);axs[s,0].set_title(f'{["ST1","ST5"][s]} · core h UMAP');axs[s,0].set_xticks([]);axs[s,0].set_yticks([])
 sc=axs[s,1].scatter(emb[:,0],emb[:,1],c=p[sel,4],cmap=cmap,vmin=.7,vmax=1,s=2,lw=0,rasterized=True);axs[s,1].set_title('Same h UMAP · malignant fraction');axs[s,1].set_xticks([]);axs[s,1].set_yticks([]);fig.colorbar(sc,ax=axs[s,1],fraction=.046,pad=.02)
 # Spatial coordinate colors locate the continuous patterns in the section.
 order=np.random.default_rng(42).permutation(len(sel));cols=(xy[sel,0]-xy[sel,0].min())/max(float(np.ptp(xy[sel,0])),1);ax=axs[s,2];ax.scatter(emb[order,0],emb[order,1],c=cols[order],cmap='coolwarm',vmin=0,vmax=1,s=2,lw=0,rasterized=True);ax.set_title('Same h UMAP · spatial x');ax.set_xticks([]);ax.set_yticks([])
 rows.append({'slice':['ST1','ST5'][s],'core_n':len(index),'UMAP_sample_n':len(sel),'malignant_fraction_median':float(np.median(p[sel,4]))})
fig.savefig(O/'single_slice_core_h_umap.png',dpi=150);plt.close(fig);info={'selection':'composition N4, malignant proportion >=0.70','representation':'jointly trained 64-dimensional h; independent UMAP per section; other section not used to fit each UMAP','UMAP':'n_neighbors=30, min_dist=0.3, seed42, random sample up to 5000 per section','caution':'The two UMAPs have independent arbitrary coordinate systems. UMAP alone cannot establish absence or presence of biological subtypes. Composition-based core is not pathologist-annotated.','results':rows};(O/'protocol.json').write_text(json.dumps(info,indent=2,ensure_ascii=False))
nbpath=Path('/home/xueshuailin/CCC_Phe/Cellist_ST1_ST5_Joint_V3.ipynb');n=nbformat.read(nbpath,4);tag='cellist_single_slice_core_h_umap';src='''# 单切片肿瘤核心：分别只用 h 画 UMAP\nfrom pathlib import Path\nfrom IPython.display import display, Image\np=Path('/data1/xueshuailin/CCC_Phe/workspace/outputs/nsclc_zhencaiwei/ST1_ST5/individual_core_h_umap')\ndisplay(Image(filename=str(p/'single_slice_core_h_umap.png'),width=1100))\nprint('每行分别对 ST1 或 ST5 的肿瘤核心单独拟合 h UMAP；同一行三图是同一个 UMAP，分别显示全部点、恶性细胞比例和切片内空间 x 坐标。')\nprint('两张切片的 UMAP 坐标分别拟合，不能直接比较两图的位置和距离；孤岛也不能自动等同于生物学子类。')\n''';cell=nbformat.v4.new_code_cell(src,metadata={'tags':[tag]});cell.outputs=[nbformat.v4.new_output('display_data',data={'image/png':base64.b64encode((O/'single_slice_core_h_umap.png').read_bytes()).decode()},metadata={}),nbformat.v4.new_output('stream',name='stdout',text='ST1 和 ST5 各自单独拟合，仅输入 h；同一行的三个图为相同坐标。\n'+pd.DataFrame(rows).to_string(index=False)+'\n')];found=[i for i,c in enumerate(n.cells) if tag in c.get('metadata',{}).get('tags',[])];
if found:n.cells[found[0]]=cell
else:n.cells.append(cell)
nbformat.write(n,nbpath);print('COMPLETE',flush=True)
