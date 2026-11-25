"""Create the requested ST5 notebook once; subsequent runs keep user edits."""
from pathlib import Path
import nbformat

ROOT=Path('/home/xueshuailin/CCC_Phe')
NB=ROOT/'Cellist_ST5_V2_Communication_Niches.ipynb'
if NB.exists():raise SystemExit(f'Notebook already exists; preserving it: {NB}')
prefix="""from pathlib import Path
import json, numpy as np, pandas as pd, anndata as ad
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from IPython.display import display
RUN=Path('/data1/xueshuailin/CCC_Phe/workspace/outputs/nsclc_zhencaiwei/ST5/v2')
DATA=Path('/data1/xueshuailin/CCC_Phe/workspace/data/nsclc_zhencaiwei/processed/ST5/ST5_counts.h5ad')
"""
cells=[nbformat.v4.new_markdown_cell('''# Cellist ST5 · V2 组成与通讯生态位

组成粗类固定为 **K=10**。以恶性细胞局部比例识别肿瘤富集粗类，再按物理邻接选择周边粗类；两者分别进行 V2 三视角训练和细分。自动选择的阈值与细分 K 为探索性设置，具体见下方表格。

ST5 空间坐标目前沿用 ST1 的 **0.5 μm/原始单位假设**，尚未得到独立证实；作者细胞类型用于组成和通讯目标，不以作者肿瘤边界或病理标签训练。''')]
cells.append(nbformat.v4.new_code_cell('''# 1. ST5 作者细胞类型的空间分布
%matplotlib inline
'''+prefix+'''if not DATA.exists():
 print('原始数据转换尚在运行。')
else:
 a=ad.read_h5ad(DATA,backed='r');xy=np.asarray(a.obsm['spatial_native']);types=a.obs['assign.copyKAT'].astype(str).to_numpy();a.file.close()
 levels=sorted(np.unique(types));colors=plt.get_cmap('tab10').colors;order=np.random.default_rng(2026).permutation(len(xy));idx=pd.Categorical(types,categories=levels).codes
 fig,ax=plt.subplots(figsize=(9,8),layout='constrained');ax.scatter(xy[order,1],-xy[order,0],c=np.asarray(colors)[idx[order]],s=.8,lw=0,rasterized=True);ax.set_aspect('equal');ax.axis('off');ax.set_title(f'ST5 · author cell types · {len(xy):,} cells')
 ax.legend(handles=[Line2D([],[],marker='o',lw=0,color=colors[i],label=f'{v} (n={(idx==i).sum():,})',markersize=6) for i,v in enumerate(levels)],bbox_to_anchor=(1.02,1),loc='upper left',frameon=False);plt.show()
'''))
cells.append(nbformat.v4.new_code_cell('''# 2. 当前准备与训练进度（各粗类日志包含每轮 loss）
'''+prefix+'''p=RUN/'status.json'
print(json.dumps(json.loads(p.read_text()),ensure_ascii=False,indent=2) if p.exists() else '等待运行')
if (RUN/'selected_coarse_classes.json').exists():
 display(pd.read_csv(RUN/'coarse_selection_summary.csv'))
 for group in json.loads((RUN/'selected_coarse_classes.json').read_text())['selected_zero_based']:
  fit=RUN/f'coarse_{group}_fit'
  if (fit/'loss.csv').exists():
   log=pd.read_csv(fit/'loss.csv'); print(f'C{group+1}: {len(log)} epochs'); display(log.tail(1))
  else:
   for log in sorted(RUN.glob(f'coarse_{group}_gpu*.log')):
    lines=log.read_text().splitlines(); print(f'C{group+1}:', '\\n'.join(lines[-2:]))
'''))
cells.append(nbformat.v4.new_code_cell('''# 3. 组成粗类 K=10，以及肿瘤与周边粗类的选择
%matplotlib inline
'''+prefix+'''labels_file=RUN/'coarse_K10.npy'; selection=RUN/'coarse_selection_summary.csv'
if not DATA.exists() or not labels_file.exists() or not selection.exists():
 print('组成粗类尚未完成。')
else:
 a=ad.read_h5ad(DATA,backed='r');xy=np.asarray(a.obsm['spatial_native']);a.file.close();lab=np.load(labels_file);tab=pd.read_csv(selection);display(tab.round(3))
 if len(lab)!=len(xy): raise ValueError('细胞顺序不一致')
 order=np.random.default_rng(2026).permutation(len(xy));colors=plt.get_cmap('tab10').colors
 fig,ax=plt.subplots(figsize=(9,8),layout='constrained');ax.scatter(xy[order,1],-xy[order,0],c=np.asarray(colors)[lab[order]],s=.8,lw=0,rasterized=True);ax.set_aspect('equal');ax.axis('off');ax.set_title('ST5 · composition-defined coarse niches · K=10')
 ax.legend(handles=[Line2D([],[],marker='o',lw=0,color=colors[i],label=f"C{i+1} · {tab.iloc[i]['role']}",markersize=6) for i in range(10)],bbox_to_anchor=(1.02,1),loc='upper left',frameon=False);plt.show()
'''))
cells.append(nbformat.v4.new_code_cell('''# 4. 肿瘤及周边粗类分别细分：位置、细胞数及候选 K 的诊断
%matplotlib inline
'''+prefix+'''if not (RUN/'pipeline_complete.json').exists():
 print('V2 类内训练仍在进行；完成后会自动更新此单元格。')
else:
 info=json.loads((RUN/'pipeline_complete.json').read_text());a=ad.read_h5ad(DATA,backed='r');xy=np.asarray(a.obsm['spatial_native']);a.file.close();coarse=np.load(RUN/'coarse_K10.npy');order=np.random.default_rng(2026).permutation(len(xy))
 groups=info['selected']['selected_zero_based'];cols=min(3,len(groups));rows=(len(groups)+cols-1)//cols;fig,axes=plt.subplots(rows,cols,figsize=(6*cols,5*rows),squeeze=False,layout='constrained');diag=[]
 for ax,g in zip(axes.flat,groups):
  fit=RUN/f'coarse_{g}_fit';ids=np.flatnonzero(coarse==g);k=info['fine_k'][str(g+1)];labels=np.zeros(len(ids),int) if k==1 else np.load(fit/f'fine_K{k}.npy')
  if len(ids)!=len(labels):raise ValueError(f'C{g+1} 标签和细胞未对齐')
  all_lab=np.full(len(xy),-1);all_lab[ids]=labels;bg=order[all_lab[order]<0];fg=order[all_lab[order]>=0];palette=plt.get_cmap('tab10').colors
  ax.scatter(xy[bg,1],-xy[bg,0],c='#e5e8eb',s=.12,lw=0,rasterized=True);ax.scatter(xy[fg,1],-xy[fg,0],c=np.asarray(palette)[all_lab[fg]],s=.9,lw=0,rasterized=True);ax.set_aspect('equal');ax.axis('off');role='tumor' if g in info['selected']['tumor_zero_based'] else 'adjacent';ax.set_title(f'C{g+1} · {role} · fine K={k}')
  ax.legend(handles=[Line2D([],[],marker='o',lw=0,color=palette[j],label=f'{j+1} (n={(labels==j).sum():,})',markersize=5) for j in range(k)],loc='lower center',bbox_to_anchor=(.5,-.06),ncol=max(1,k),frameon=False,fontsize=8)
  q=pd.read_csv(fit/'fine_k_diagnostics.csv');q.insert(0,'coarse_class',g+1);diag.append(q)
 for ax in axes.flat[len(groups):]:ax.axis('off')
 plt.show();display(pd.concat(diag,ignore_index=True).round(3))
 print('候选 K 使用同一 h 的 K-means 初始值稳定性、最小类比例和抽样 silhouette 选择；这些是探索性指标，不能单独证明功能亚类。')
'''))
nb=nbformat.v4.new_notebook(cells=cells,metadata={'kernelspec':{'display_name':'cccphe','language':'python','name':'cccphe'}})
nbformat.write(nb,NB);print('Created',NB)
