from pathlib import Path
import nbformat as n
p=Path('/home/xueshuailin/CCC_Phe/Brain_V3_scNiche_Harmonics_Benchmark.ipynb')
if p.exists():raise FileExistsError('Preserving existing notebook')
nb=n.v4.new_notebook();nb.metadata={'kernelspec':{'display_name':'Python (cccphe)','language':'python','name':'cccphe'}}
nb.cells=[n.v4.new_markdown_cell('## STARmap / MERFISH：V3、scNiche 与 Harmonics\n\n每种方法独立训练 10 次，K=7。STARmap 使用 stDyer 标注；MERFISH 使用区域标注。V3 使用既有小鼠到人类同源基因映射的通讯特征。STARmap scNiche 使用官方完整训练；MERFISH 使用官方批次模型及适配器。Harmonics 使用作者代码。'),n.v4.new_code_cell('''from pathlib import Path
import json, numpy as np, pandas as pd
from IPython.display import display
base=Path('/home/nas3/biod/xueshuailin/CCC_Phe/workspace/outputs/brain_10run_benchmark')
records=[json.loads(p.read_text()) for p in sorted(base.glob('*/*/seed_*/metrics.json'))]
df=pd.DataFrame(records)
if df.empty:
    print('尚无完成结果；后台仍在运行，请稍后重运行本单元格。')
else:
    display(df.groupby(['dataset','method']).agg(runs=('ARI','size'),ARI_mean=('ARI','mean'),ARI_std=('ARI','std'),ARI_median=('ARI','median'),macro_F1_mean=('macro_F1','mean')).round(3))
'''),n.v4.new_code_cell('''import matplotlib.pyplot as plt
if not df.empty:
    fig,axs=plt.subplots(1,2,figsize=(10,4),layout='constrained')
    for ax,dataset in zip(axs,['starmap','merfish']):
        sub=df[df.dataset==dataset]; methods=['v3','scniche','harmonics']; vals=[sub.loc[sub.method==m,'ARI'].values for m in methods]
        for i,v in enumerate(vals):
            if len(v):
                ax.boxplot(v,positions=[i],widths=.5,showfliers=False);ax.scatter(np.full(len(v),i)+np.random.default_rng(7).normal(0,.04,len(v)),v,s=15,alpha=.8)
        ax.set_xticks(range(3),['V3','scNiche','Harmonics']);ax.set_ylabel('ARI');ax.set_title(dataset.upper());ax.grid(axis='y',alpha=.2)
    plt.show()
'''),n.v4.new_markdown_cell('空间图展示各方法已完成运行中 ARI 最接近其中位数的一次；真实标签仅用于评价、选择代表图和匹配显示颜色。'),n.v4.new_code_cell('''import anndata as ad
from scipy.spatial import cKDTree
from sklearn.metrics import confusion_matrix
from scipy.optimize import linear_sum_assignment
if not df.empty:
    W=Path('/data1/xueshuailin/CCC_Phe/workspace/data/harmonics_fig2')
    fig,axes=plt.subplots(2,4,figsize=(12,7),layout='constrained')
    for row,dataset in enumerate(['starmap','merfish']):
        data=W/('starmap/analysis_input.h5ad' if dataset=='starmap' else 'merfish_scaled/donor12_slice0_analysis_input.h5ad')
        a=ad.read_h5ad(data,backed='r');xy=np.asarray(a.obsm['spatial']);valid=np.ones(len(a),bool)
        if dataset=='starmap':
            ref=ad.read_h5ad(W/'starmap/stDyer_mouse_cortex.h5ad');dist,ix=cKDTree(np.c_[ref.obs.X,-ref.obs.Y]).query(xy);valid=dist<.01;truth=pd.Categorical(ref.obs.label.iloc[ix].astype(str).str.upper()).codes
        else:truth=pd.Categorical(a.obs.benchmark_region.astype(str)).codes
        a.file.close();order=np.random.default_rng(5).permutation(np.flatnonzero(valid));palette=plt.get_cmap('tab10')
        for col,method in enumerate(['truth','v3','scniche','harmonics']):
            ax=axes[row,col];ax.set_aspect('equal');ax.axis('off')
            if method=='truth':lab=truth;title='Annotation'
            else:
                sub=df[(df.dataset==dataset)&(df.method==method)]
                if sub.empty:ax.set_title(method+' · pending');continue
                chosen=sub.iloc[(sub.ARI-sub.ARI.median()).abs().argmin()];pred=np.load(base/dataset/method/f"seed_{int(chosen.seed)}"/'result.npz')['labels'];cm=confusion_matrix(truth[valid],pred[valid],labels=np.arange(7));r,c=linear_sum_assignment(-cm);mp=dict(zip(c,r));lab=np.array([mp.get(x,0) for x in pred]);title=f"{method} · ARI {chosen.ARI:.3f}"
            ax.scatter(xy[order,0],xy[order,1],c=lab[order],vmin=0,vmax=9,cmap=palette,s=2,lw=0,rasterized=True);ax.set_title(title,fontsize=10)
    plt.show()
''')];n.write(nb,p);print(p)
