"""Update the existing ST1 cell with two curved regions and niche fractions."""
from pathlib import Path
import json
import nbformat
from nbclient import NotebookClient

ROOT=Path('/home/xueshuailin/CCC_Phe')
NB=ROOT/'Cellist_ST1_V2_Communication_Niches.ipynb'
MARKER='# ST1 C4 上下肿块边缘对照'
CODE=r'''# ST1 C4 上下肿块边缘对照
%matplotlib inline
from pathlib import Path
import numpy as np, pandas as pd, anndata as ad
import matplotlib.pyplot as plt
from matplotlib.patches import PathPatch, Patch
from matplotlib.path import Path as MplPath
from IPython.display import display

RUN=Path('/data1/xueshuailin/CCC_Phe/workspace/outputs/nsclc_zhencaiwei/ST1/v2')
DATA=Path('/data1/xueshuailin/CCC_Phe/workspace/data/nsclc_zhencaiwei/processed/ST1/ST1_counts.h5ad')
adata=ad.read_h5ad(DATA,backed='r')
xy=np.asarray(adata.obsm['spatial_native']);boundary=adata.obs['Tumor_boundary'].astype(str).to_numpy()
adata.file.close()
coarse=np.load(RUN/'coarse_K4.npy');ids=np.flatnonzero(coarse==3)
niche=np.load(RUN/'granularity_exploration/coarse_3_fine_K6.npy')
if len(ids)!=len(niche) or len(xy)!=len(coarse): raise ValueError('细胞与标签不一致')
x=xy[ids,1]; y=xy[ids,0]
edge=np.isin(boundary[ids],['Inner 1','Outer 1'])

# 用闭合贝塞尔曲线分别沿肿块走向圈定，圈选与统计使用同一条路径。
# 点用绘图坐标 (原始 x, -原始 y) 表示，数据本身没有进行坐标转换。
def curved_region(points):
    vertices=[(px,-py) for px,py in points]
    return MplPath(vertices,[MplPath.MOVETO]+[MplPath.CURVE4]*(len(vertices)-2)+[MplPath.CLOSEPOLY])
regions={
 'Upper':curved_region([(17400,8200),(16900,7200),(17700,6300),(18600,6000),
     (19700,5500),(21100,5700),(21400,6800),
     (21500,7600),(20700,8700),(20100,9000),
     (19000,9300),(17800,9200),(17400,8200),(17400,8200)]),
 'Lower':curved_region([(18300,10300),(18300,9700),(19200,9300),(20100,9250),
     (20700,9200),(20900,8350),(21600,8500),
     (22500,8650),(22700,10400),(22400,11900),
     (22200,13000),(20300,13200),(19200,12600),
     (18300,12300),(17700,11100),(18300,10300),(18300,10300)])}
selected={name:edge & shape.contains_points(np.column_stack((x,-y))) for name,shape in regions.items()}
if np.any(selected['Upper'] & selected['Lower']):raise ValueError('两个圈选区域重叠，请调整轮廓')
if min(map(np.sum,selected.values()))<50:raise ValueError('圈选的边缘细胞过少')

# 分母为该椭圆内所有 C4 且作者标注为 Inner 1 或 Outer 1 的边缘细胞；
# C4.3 仍包含在分母中，避免把五个展示类别错误地归一化为 100%。
shown=[0,1,3,4,5]
fractions={name:np.bincount(niche[mask],minlength=6)/mask.sum() for name,mask in selected.items()}
table=pd.DataFrame({'Niche':[f'C4.{i+1}' for i in shown],
                    'Upper (%)':[100*fractions['Upper'][i] for i in shown],
                    'Lower (%)':[100*fractions['Lower'][i] for i in shown]})

niche_colors=np.array(['#34999b','#df9269','#9c7bb4','#d6aa4c','#5b80b4','#7baa69'])
region_colors={'Upper':'#288b89','Lower':'#c97350'}
fig,(ax,bar)=plt.subplots(1,2,figsize=(16,7.5),gridspec_kw={'width_ratios':[1.05,1]},layout='constrained')
order=np.random.default_rng(2026).permutation(len(ids))
ax.scatter(x[order],-y[order],c=niche_colors[niche[order]],s=1.1,lw=0,rasterized=True)
for name,shape in regions.items():
    ax.add_patch(PathPatch(shape,fill=False,lw=2.4,edgecolor=region_colors[name]))
    ax.text(17800 if name=='Upper' else 18400,-5900 if name=='Upper' else -9350,name,color=region_colors[name],fontweight='bold',fontsize=13)
ax.set_xlim(15800,23100);ax.set_ylim(-13600,-5100);ax.set_aspect('equal');ax.axis('off')
ax.set_title('a   Tumor regions',loc='left',fontsize=15,fontweight='bold')
ax.legend(handles=[Patch(color=niche_colors[i],label=f'C4.{i+1}') for i in range(6)],loc='lower center',bbox_to_anchor=(.5,-.06),ncol=3,frameon=False,fontsize=9)
z=np.arange(len(shown));width=.34
for offset,name in [(-width/2,'Upper'),(width/2,'Lower')]:
    values=100*fractions[name][shown]
    bars=bar.bar(z+offset,values,width,color=region_colors[name],label=f'{name} (n={selected[name].sum():,})')
    bar.bar_label(bars,fmt='%.1f',padding=3,fontsize=9)
bar.set_xticks(z,[f'C4.{i+1}' for i in shown]);bar.set_ylabel('Fraction of edge cells (%)')
bar.set_title('b   Niche distribution at tumor borders',loc='left',fontsize=15,fontweight='bold')
bar.spines[['top','right']].set_visible(False);bar.set_ylim(0,max(100*max(fractions['Upper'][shown].max(),fractions['Lower'][shown].max())*1.2,5));bar.legend(frameon=False)
plt.show()
display(table.round(1))
print(f"分母：上方 {selected['Upper'].sum():,} 个、下方 {selected['Lower'].sum():,} 个 C4 边缘细胞；按作者 Inner 1 / Outer 1 标注筛选。")
print('C4.3 计入总边缘细胞数，但不单独显示在柱图中。圈选是基于空间位置的探索性区域，并非独立样本。')

# 对上述同一批边缘微环境比较有向通讯程序。只对该程序有接触机会的
# 微环境计算均值；各侧支持至少 50 个微环境且不低于该区域的 20%。
V=np.load(RUN/'exposure.npy',mmap_mode='r'); M=np.load(RUN/'exposure_mask.npy',mmap_mode='r')
meta=pd.read_csv(RUN/'exposure_support.csv')
if V.shape!=M.shape or V.shape[1]!=len(meta):raise ValueError('通讯程序矩阵与注释不匹配')
means=[]; nums=[]; rates=[]; varparts=[]
for name in ('Upper','Lower'):
    rows=ids[selected[name]]; v=np.asarray(V[rows]); mask=np.asarray(M[rows],bool)&np.isfinite(v)
    count=mask.sum(0); mean=np.where(mask,v,0).sum(0)/np.maximum(count,1)
    variance=np.where(mask,(v-mean[None,:])**2,0).sum(0)/np.maximum(count-1,1)
    means.append(mean);nums.append(count);rates.append(count/len(rows));varparts.append(variance)
means=np.stack(means);nums=np.stack(nums);rates=np.stack(rates);varparts=np.stack(varparts)
valid=(nums>=50).all(0)&(rates>=.2).all(0)
pooled_sd=np.sqrt(np.maximum((varparts[0]+varparts[1])/2,0))
valid&=pooled_sd>1e-6
effect=(means[0]-means[1])/np.maximum(pooled_sd,1e-6)
upper_ids=np.flatnonzero(valid & (effect>0));upper_ids=upper_ids[np.argsort(effect[upper_ids])[::-1][:6]]
lower_ids=np.flatnonzero(valid & (effect<0));lower_ids=lower_ids[np.argsort(effect[lower_ids])[:6]]
top=np.r_[upper_ids,lower_ids]
names=(meta.pathway.astype(str).str.replace('Signaling by ','',regex=False)+' · '+meta.sender.astype(str)+' → '+meta.receiver.astype(str))
results=meta.iloc[top][['pathway','sender','receiver','mapped_LR']].copy().reset_index(drop=True)
results['Upper mean']=means[0,top];results['Lower mean']=means[1,top]
results['Upper support']=rates[0,top];results['Lower support']=rates[1,top];results['Standardized difference']=effect[top]

fig2, ax2=plt.subplots(figsize=(12.5,6.5),layout='constrained')
order2=top[np.argsort(effect[top])];color2=np.where(effect[order2]>0,region_colors['Upper'],region_colors['Lower'])
ax2.barh(np.arange(len(order2)),effect[order2],color=color2,height=.72)
ax2.set_yticks(np.arange(len(order2)),names.iloc[order2],fontsize=9)
ax2.axvline(0,color='#87949d',lw=.9)
ax2.spines[['top','right']].set_visible(False)
ax2.set_xlabel('Standardized difference (Upper − Lower)')
ax2.set_title('c   Directed communication pathways at tumor borders',loc='left',fontsize=15,fontweight='bold')
ax2.legend(handles=[Patch(color=region_colors['Upper'],label='Higher in upper border'),Patch(color=region_colors['Lower'],label='Higher in lower border')],loc='lower right',frameon=False,fontsize=9)
plt.show()
print('通讯通路按上方减下方的标准化均值差排序；每一项同时注明发送与接收细胞类型。仅展示两边均有足够接触机会的特征。')
display(results.round(3))
if len(upper_ids) and len(lower_ids):
    print(f"观察：上方相对较高的包括 {names.iloc[upper_ids[0]]}；下方相对较高的包括 {names.iloc[lower_ids[0]]}。")
print('两处来自同一张切片，且细胞组成可能不同：此处是探索性通讯差异，不能据此认定独立重复、统计显著性或因果关系。')
'''

nb=nbformat.read(NB,4)
old=[i for i,c in enumerate(nb.cells) if c.cell_type=='code' and c.source.startswith(MARKER)]
if not old:raise RuntimeError('原对照单元格不存在；尊重手动删除，不自动恢复')
one=nbformat.v4.new_notebook(cells=[nbformat.v4.new_code_cell(CODE)],metadata=nb.metadata)
NotebookClient(one,timeout=1200,kernel_name='cccphe',allow_errors=False).execute(cwd=str(ROOT))
one.cells[0].execution_count=None
nb.cells[old[0]]=one.cells[0]
nbformat.write(nb,NB)
print('Updated:',NB,flush=True)
