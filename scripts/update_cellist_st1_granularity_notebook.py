"""Add saved ST1 granularity diagnostics to its notebook without retraining."""
from pathlib import Path
import nbformat
from nbclient import NotebookClient

ROOT = Path('/home/xueshuailin/CCC_Phe')
NB = ROOT / 'Cellist_ST1_V2_Communication_Niches.ipynb'
MARKER = '# ST1 V2 粒度评估'

coarse = r'''# ST1 V2 粒度评估 · 组成粗类 K
%matplotlib inline
from pathlib import Path
import numpy as np, pandas as pd, anndata as ad
import matplotlib.pyplot as plt
from IPython.display import display

RUN = Path('/data1/xueshuailin/CCC_Phe/workspace/outputs/nsclc_zhencaiwei/ST1/v2')
G = RUN / 'granularity_exploration'
DATA = Path('/data1/xueshuailin/CCC_Phe/workspace/data/nsclc_zhencaiwei/processed/ST1/ST1_counts.h5ad')
coarse_stats = pd.read_csv(G / 'coarse_candidates.csv')
adata_g = ad.read_h5ad(DATA, backed='r')
xy_g = np.asarray(adata_g.obsm['spatial_native']); adata_g.file.close()
order_g = np.random.default_rng(2026).permutation(len(xy_g))
fig, axes = plt.subplots(2, 4, figsize=(17, 8.1), layout='constrained')
axes[0,0].plot(coarse_stats.k, coarse_stats.stability_ari_mean, 'o-', color='#338e8b', label='Mean stability ARI')
axes[0,0].plot(coarse_stats.k, coarse_stats.stability_ari_min, 'o--', color='#b6678c', label='Minimum stability ARI')
axes[0,0].plot(coarse_stats.k, coarse_stats.heldout_silhouette_mean, 'o-', color='#d29c50', label='Held-out silhouette')
axes[0,0].set(xlabel='Coarse K', ylabel='Score', title='Coarse K diagnostics', ylim=(0,1.02)); axes[0,0].legend(frameon=False, fontsize=8)
axes[0,1].bar(coarse_stats.k, 100*coarse_stats.min_heldout_cluster_fraction, color='#86bdb5')
axes[0,1].set(xlabel='Coarse K', ylabel='Cells (%)', title='Smallest held-out class')
for ax,k in zip(axes.flat[2:], range(3,9)):
    labels = np.load(G / f'coarse_K{k}.npy')
    if len(labels) != len(xy_g): raise ValueError('细胞坐标和聚类标签数不一致')
    ax.scatter(xy_g[order_g,1], -xy_g[order_g,0], c=labels[order_g], cmap='tab10', s=.48, lw=0, rasterized=True, vmin=0, vmax=9)
    ax.set_aspect('equal'); ax.axis('off'); ax.set_title(f'Composition · K={k}', fontsize=11)
plt.show()
display(coarse_stats.round(3).rename(columns={'k':'K','stability_ari_mean':'平均稳定性 ARI','stability_ari_min':'最低稳定性 ARI','heldout_silhouette_mean':'留出轮廓系数','min_heldout_cluster_fraction':'最小类占比'}))
print('这些是候选粒度的诊断，不会自动宣布最优 K；颜色在不同 K 之间未做类别对应。')
'''

fine = r'''# ST1 V2 粒度评估 · 粗类内细分候选
%matplotlib inline
from pathlib import Path
import numpy as np, pandas as pd, anndata as ad
import matplotlib.pyplot as plt
from IPython.display import display
RUN = Path('/data1/xueshuailin/CCC_Phe/workspace/outputs/nsclc_zhencaiwei/ST1/v2')
G = RUN / 'granularity_exploration'
DATA = Path('/data1/xueshuailin/CCC_Phe/workspace/data/nsclc_zhencaiwei/processed/ST1/ST1_counts.h5ad')
fine_stats = pd.read_csv(G / 'fine_candidates_all.csv')
shown = fine_stats[['coarse_class','k','stability_ari_mean','heldout_gain_mean','gaussian_null_excess','min_heldout_cluster_fraction','size_guard_pass']].copy()
shown['coarse_class'] = shown['coarse_class'] + 1
display(shown.round(3).rename(columns={'coarse_class':'原粗类','k':'细分 K','stability_ari_mean':'稳定性 ARI','heldout_gain_mean':'留出改善','gaussian_null_excess':'超出高斯对照','min_heldout_cluster_fraction':'最小类占比','size_guard_pass':'达到 2% 下限'}))
adata_g = ad.read_h5ad(DATA, backed='r'); xy_g = np.asarray(adata_g.obsm['spatial_native']); adata_g.file.close()
coarse_original = np.load(RUN / 'coarse_K4.npy')
order_g = np.random.default_rng(2026).permutation(len(xy_g))
fine_colors = np.array(['#d48663','#4c9e9e','#9b76aa','#dbb653'])
fig, axes = plt.subplots(4, 4, figsize=(15, 15), layout='constrained')
for g in range(4):
    ids = np.flatnonzero(coarse_original == g)
    for col,k in enumerate((1,2,3,4)):
        ax = axes[g,col]
        local = np.zeros(len(ids), dtype=int) if k == 1 else np.load(G / f'coarse_{g}_fine_K{k}.npy')
        if len(local) != len(ids): raise ValueError(f'粗类 {g+1} 标签与细胞未对齐')
        labels = np.full(len(xy_g), -1, dtype=int); labels[ids] = local
        background = order_g[labels[order_g] < 0]; selected = order_g[labels[order_g] >= 0]
        ax.scatter(xy_g[background,1],-xy_g[background,0], c='#e5e8eb', s=.12, lw=0, rasterized=True)
        ax.scatter(xy_g[selected,1],-xy_g[selected,0], c=fine_colors[labels[selected]], s=.55, lw=0, rasterized=True)
        ax.set_aspect('equal'); ax.axis('off'); ax.set_title(f'Coarse C{g+1} · K={k}', fontsize=10)
plt.show()
print('每行使用原 K=4 中一个粗类的 h；同一行内颜色未跨 K 匹配。K=1 表示保留该粗类，不强制细分。')
print('高斯对照仅用于探索：尚未控制空间相关性、细胞组成残余和测序深度，不能据此断言功能亚类。')
print('空间分块使用 0.5 µm/坐标单位的暂定换算；h 在整张切片训练，留出分块并非端到端独立验证。')
'''

selection = r'''# ST1 V2 粒度评估 · 用户选择并展示
%matplotlib inline
# 基于上一单元格的稳定性、空间分布和后续生物学解释，自行调整每类 K；1 表示不细分。
# 当前 4 个 h 均属于原始粗类 K=4；改粗类 K 后需要重新计算对应视图并训练，不能套用这些标签。
FINE_K = {1: 1, 2: 1, 3: 1, 4: 1}
from pathlib import Path
import numpy as np, pandas as pd, anndata as ad
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
RUN = Path('/data1/xueshuailin/CCC_Phe/workspace/outputs/nsclc_zhencaiwei/ST1/v2')
G = RUN / 'granularity_exploration'
DATA = Path('/data1/xueshuailin/CCC_Phe/workspace/data/nsclc_zhencaiwei/processed/ST1/ST1_counts.h5ad')
if set(FINE_K) != {1,2,3,4} or any(k not in (1,2,3,4) for k in FINE_K.values()): raise ValueError('FINE_K 必须为 4 个粗类分别指定 1、2、3 或 4')
adata_g = ad.read_h5ad(DATA, backed='r'); xy_g = np.asarray(adata_g.obsm['spatial_native']); adata_g.file.close()
coarse_original = np.load(RUN / 'coarse_K4.npy'); order_g = np.random.default_rng(2026).permutation(len(xy_g))
joint = np.full(len(xy_g), -1, dtype=int); names = []; palette = plt.get_cmap('tab20').colors
for g in range(4):
    ids = np.flatnonzero(coarse_original == g); k = FINE_K[g+1]
    local = np.zeros(len(ids), dtype=int) if k == 1 else np.load(G / f'coarse_{g}_fine_K{k}.npy')
    if len(local) != len(ids): raise ValueError('标签与细胞数不一致')
    for s in range(k):
        joint[ids[local == s]] = len(names); names.append(f'C{g+1}' if k == 1 else f'C{g+1}.{s+1}')
if (joint < 0).any(): raise ValueError('存在未分配细胞')
fig, ax = plt.subplots(figsize=(9,8), layout='constrained')
ax.scatter(xy_g[order_g,1], -xy_g[order_g,0], c=np.asarray(palette)[joint[order_g]], s=.6, lw=0, rasterized=True)
ax.set_aspect('equal'); ax.axis('off'); ax.set_title('ST1 · chosen coarse and functional granularity')
ax.legend(handles=[Line2D([],[],marker='o',lw=0,color=palette[i],label=f'{name} (n={(joint==i).sum():,})',markersize=6) for i,name in enumerate(names)],bbox_to_anchor=(1.01,1),loc='upper left',frameon=False)
plt.show()
print('细分 K 是当前用户设置，不代表已自动证明存在功能亚类；需进一步核对通讯、响应、组成及测序深度。')
'''

nb = nbformat.read(NB, 4)
nb.cells = [c for c in nb.cells if not (c.cell_type == 'code' and c.source.startswith(MARKER))]
new = nbformat.v4.new_notebook(cells=[nbformat.v4.new_code_cell(s) for s in (coarse, fine, selection)])
new.metadata = nb.metadata
NotebookClient(new, timeout=1200, kernel_name='cccphe', allow_errors=False).execute(cwd=str(ROOT))
for cell in new.cells:
    cell.execution_count = None
nb.cells.extend(new.cells)
nbformat.write(nb, NB)
print(f'Added and executed {len(new.cells)} cells: {NB}')
