"""Add a self-contained, executed descriptive C4 analysis to the ST1 notebook."""
from pathlib import Path
import nbformat
from nbclient import NotebookClient

ROOT = Path('/home/xueshuailin/CCC_Phe')
NB = ROOT / 'Cellist_ST1_V2_Communication_Niches.ipynb'
MARKER = '# ST1 V2 C4 通讯细分分析'
CODE = r'''# ST1 V2 C4 通讯细分分析
%matplotlib inline
from pathlib import Path
import json
import numpy as np, pandas as pd, anndata as ad
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from IPython.display import display

# C4 指原始 K=4 粗分类中的第 4 类；可选择已计算的细分 K。
C4_FINE_K = 6  # 可改成 2、3、4、6 或 8
FEATURES_PER_CLASS = 3
if C4_FINE_K not in (2, 3, 4, 6, 8): raise ValueError('C4_FINE_K 仅可为 2、3、4、6 或 8')
RUN = Path('/data1/xueshuailin/CCC_Phe/workspace/outputs/nsclc_zhencaiwei/ST1/v2')
DATA = Path('/data1/xueshuailin/CCC_Phe/workspace/data/nsclc_zhencaiwei/processed/ST1/ST1_counts.h5ad')
G = RUN / 'granularity_exploration'
target = Path(json.loads((RUN / 'target_cache_01.json').read_text())['folder'])
coarse = np.load(RUN / 'coarse_K4.npy'); ids = np.flatnonzero(coarse == 3)
labels = np.load(G / f'coarse_3_fine_K{C4_FINE_K}.npy')
if len(ids) != len(labels): raise ValueError('C4 细胞与标签长度不一致')
adata = ad.read_h5ad(DATA, backed='r')
xy = np.asarray(adata.obsm['spatial_native'])
obs = adata.obs.iloc[ids][['assign.copyKAT', 'nCount_RNA', 'nFeature_RNA']].copy()
adata.file.close()
if len(xy) != len(coarse): raise ValueError('细胞坐标与粗类标签长度不一致')
types = sorted(obs['assign.copyKAT'].astype(str).unique())
composition = np.load(target / 'composition.npy', mmap_mode='r')[ids]
if composition.shape[1] != len(types): raise ValueError('组成列数与细胞类型数不一致')

counts = np.bincount(labels, minlength=C4_FINE_K)
summary = pd.DataFrame({'细类':[f'C4.{i+1}' for i in range(C4_FINE_K)],
    '细胞数':counts, 'C4内占比':counts / len(labels),
    '恶性中心细胞占比':[np.mean(obs['assign.copyKAT'].astype(str).to_numpy()[labels==i]=='Malignant') for i in range(C4_FINE_K)],
    'nCount_RNA 中位数':[np.median(pd.to_numeric(obs['nCount_RNA']).to_numpy()[labels==i]) for i in range(C4_FINE_K)],
    'nFeature_RNA 中位数':[np.median(pd.to_numeric(obs['nFeature_RNA']).to_numpy()[labels==i]) for i in range(C4_FINE_K)]})
display(summary.round(3))

def compare_view(name, mask_name, support_file, view):
    values = np.load(RUN / name, mmap_mode='r')[ids]
    mask = np.load(RUN / mask_name, mmap_mode='r')[ids].astype(bool)
    meta = pd.read_csv(RUN / support_file)
    if values.shape != mask.shape or values.shape[1] != len(meta): raise ValueError(f'{view} 列和元数据不一致')
    mask &= np.isfinite(values)
    all_n = mask.sum(axis=0)
    safe = np.where(mask, values, 0.)
    all_mean = safe.sum(axis=0) / np.maximum(all_n, 1)
    variance = np.where(mask, (values - all_mean[None,:])**2, 0.).sum(axis=0) / np.maximum(all_n-1, 1)
    sd = np.sqrt(np.maximum(variance, 0))
    means, supports, fractions = [], [], []
    for i in range(C4_FINE_K):
        select = labels == i; n = mask[select].sum(axis=0)
        means.append(safe[select].sum(axis=0) / np.maximum(n, 1))
        supports.append(n); fractions.append(n / max(int(select.sum()), 1))
    means, supports, fractions = np.array(means), np.array(supports), np.array(fractions)
    valid = (supports >= 20).all(axis=0) & (fractions >= .2).all(axis=0) & (sd > 1e-6)
    effects = (means - all_mean[None,:]) / np.maximum(sd[None,:], 1e-6)
    # Each block contains features whose largest observed subtype mean is that
    # subtype's; retain only positive separation from the next-highest group.
    # This is a display selection, never a supervised training or significance test.
    winner = np.argmax(effects, axis=0)
    ordered = np.sort(effects, axis=0)
    specificity = ordered[-1] - ordered[-2]
    blocks = []
    for i in range(C4_FINE_K):
        eligible = np.flatnonzero(valid & (winner == i) & (specificity > 0))
        eligible = eligible[np.argsort(specificity[eligible])[::-1][:FEATURES_PER_CLASS]]
        blocks.extend(eligible.tolist())
    ranked = np.asarray(blocks, dtype=int)
    if view == 'pathway':
        names = meta['pathway'].astype(str) + ' · ' + meta['sender'].astype(str) + ' → ' + meta['receiver'].astype(str)
    else:
        names = meta['program'].astype(str) + ' · ' + meta['receiver'].astype(str)
    table = meta.iloc[ranked].reset_index(drop=True).copy()
    table['相对最高的细类'] = [f'C4.{winner[j]+1}' for j in ranked]
    table['领先次高细类的标准化差距'] = specificity[ranked]
    for i in range(C4_FINE_K):
        table[f'C4.{i+1} 均值'] = means[i,ranked]
        table[f'C4.{i+1} 支持比例'] = fractions[i,ranked]
    return effects[:,ranked], names.iloc[ranked].tolist(), table, int(valid.sum())

ep, ep_names, ep_table, ep_n = compare_view('exposure.npy', 'exposure_mask.npy', 'exposure_support.csv', 'pathway')
rp, rp_names, rp_table, rp_n = compare_view('response.npy', 'response_mask.npy', 'response_support.csv', 'response')
colors = np.array(['#308f91','#d98964','#9273ad','#d3ad50','#5e80b5','#82ab67','#c87091','#887b70'])
fig = plt.figure(figsize=(17, 14), layout='constrained')
gs = fig.add_gridspec(3, 2, height_ratios=[2.4,.72,.72], width_ratios=[1.8,1]); ax0=fig.add_subplot(gs[0,0]); ax1=fig.add_subplot(gs[0,1]); ax2=fig.add_subplot(gs[1,:]); ax3=fig.add_subplot(gs[2,:])
order = np.random.default_rng(2026).permutation(len(xy)); is_c4 = np.zeros(len(xy), bool); is_c4[ids] = True
bg = order[~is_c4[order]]; pos = np.random.default_rng(2026).permutation(len(ids))
ax0.scatter(xy[bg,1], -xy[bg,0], c='#e8eaed', s=.12, lw=0, rasterized=True)
ax0.scatter(xy[ids[pos],1], -xy[ids[pos],0], c=colors[labels[pos]], s=2.0, lw=0, rasterized=True)
ax0.set_aspect('equal'); ax0.axis('off'); ax0.set_title(f'C4 spatial subtypes · K={C4_FINE_K} · main region')
# Crop to robust C4 coordinate quantiles to make the main tumor region legible;
# the inset retains the full section and shows outlying C4 cells as well.
lo, hi = np.percentile(xy[ids], [2,98], axis=0)
pad = .08 * (hi-lo)
ax0.set_xlim(lo[1]-pad[1], hi[1]+pad[1]); ax0.set_ylim(-hi[0]-pad[0], -lo[0]+pad[0])
inset = ax0.inset_axes([.015,.015,.20,.26])
inset.scatter(xy[bg,1],-xy[bg,0],c='#e5e8eb',s=.025,lw=0,rasterized=True)
inset.scatter(xy[ids[pos],1],-xy[ids[pos],0],c=colors[labels[pos]],s=.11,lw=0,rasterized=True)
inset.set_aspect('equal'); inset.set_xticks([]); inset.set_yticks([]); inset.set_title('Whole section',fontsize=7)
ax0.legend(handles=[Line2D([],[],marker='o',lw=0,color=colors[i],label=f'C4.{i+1} (n={counts[i]:,})',markersize=6) for i in range(C4_FINE_K)],loc='lower center',bbox_to_anchor=(.5,-.08),ncol=C4_FINE_K,frameon=False)
bottom = np.zeros(C4_FINE_K); bars = np.array([composition[labels==i].mean(axis=0) for i in range(C4_FINE_K)])
for j,t in enumerate(types):
    ax1.bar(np.arange(C4_FINE_K), bars[:,j], bottom=bottom, color=plt.get_cmap('tab10')(j), label=t); bottom += bars[:,j]
ax1.set_xticks(np.arange(C4_FINE_K), [f'C4.{i+1}' for i in range(C4_FINE_K)]); ax1.set_ylabel('Mean local cell fraction'); ax1.set_title('Local cell composition'); ax1.legend(bbox_to_anchor=(1.02,1),loc='upper left',frameon=False,fontsize=8)
for ax, v, names, title in [(ax2,ep,ep_names,'Communication pathways'),(ax3,rp,rp_names,'Receiver response')]:
    if len(names):
        limit=max(float(np.abs(v).max()), .1); im=ax.imshow(v, aspect='auto', cmap='RdBu_r', vmin=-limit, vmax=limit, interpolation='nearest')
        ax.set_xticks(range(len(names)), names, rotation=45, ha='right', rotation_mode='anchor', fontsize=7)
        ax.set_yticks(range(C4_FINE_K), [f'C4.{i+1}' for i in range(C4_FINE_K)])
        fig.colorbar(im, ax=ax, shrink=.75, pad=.01, label='(Subtype mean − C4 mean) / C4 SD')
    else: ax.text(.5,.5,'No features pass support filter',ha='center',va='center',transform=ax.transAxes)
    ax.set_title(title)
plt.show()
print(f'细类间每类均有 ≥20 个且 ≥20% 有支持的特征：通讯程序 {ep_n} 个；接收响应 {rp_n} 个。')
print(f'热图每行一个细类，特征按相对最高的细类分组，每类最多展示 {FEATURES_PER_CLASS} 项，按领先第二高类的幅度排序；这是为展示选择的特征。')
print('热图为各细类在有支持细胞中的原始视图均值，相对 C4 总均值按特征标准差缩放；每列颜色不能直接比较绝对强度。')
print('表格给出原始视图均值和各类支持比例。排序是探索性组间差异，并非统计显著性检验或模型因果归因。')
candidate_path = G / ('coarse_3_fine_candidates.csv' if C4_FINE_K <= 4 else 'coarse_3_fine_candidates_extra.csv')
candidate = pd.read_csv(candidate_path).set_index('k').loc[C4_FINE_K]
print(f'当前 K={C4_FINE_K} 的稳定性 ARI={candidate.stability_ari_mean:.2f}，最小细类占比={candidate.min_heldout_cluster_fraction:.1%}；须结合空间重复、组成和测序深度核验。')
print('坐标按数据原始单位展示；病理标签未用于此分析。')
print('主空间图按 C4 坐标 2%–98% 分位数加边距裁剪以突出主要区域，左下角插图展示完整切片。')
print('\n通讯程序：')
display(ep_table.round(3))
print('接收响应：')
display(rp_table.round(3))
'''

nb = nbformat.read(NB, 4)
old = [i for i,c in enumerate(nb.cells) if c.cell_type == 'code' and c.source.startswith(MARKER)]
new = nbformat.v4.new_notebook(cells=[nbformat.v4.new_code_cell(CODE)], metadata=nb.metadata)
NotebookClient(new, timeout=1200, kernel_name='cccphe', allow_errors=False).execute(cwd=str(ROOT))
new.cells[0].execution_count = None
if old:
    nb.cells[old[0]] = new.cells[0]
else:
    nb.cells.append(new.cells[0])
nbformat.write(nb, NB)
print('Completed C4 analysis:', NB)
