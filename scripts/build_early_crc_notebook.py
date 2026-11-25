"""Prepare, but do not execute, the eleven-specimen CRC result notebook."""
from pathlib import Path
import nbformat

ROOT=Path(__file__).resolve().parents[1]
BOOK=ROOT/'CosMx_EarlyCRC_Communication_Niches.ipynb'


def build():
    intro='''# CosMx early CRC · Communication niches

11位患者、2张载玻片上的11份独立组织标本，约126万作者QC细胞、6,175个基因。每位患者单独建立空间邻域，全部标本联合训练；没有人为划定参考/查询集。

固定使用项目主线：原30成员局部网络的有向CCC与细胞组成 → 有向细胞–边GNN → h96 → 两个直接预测MLP；多尺度联合距离约束h，以固定全局尺度h做K-means。患者分期、原作者niche及原UMAP均不参与训练。早停依据训练总loss，这是探索性拟合，并非独立验证。

[Nature原论文](https://doi.org/10.1038/s41586-026-10344-7) · [数据](https://zenodo.org/records/20136513)。下列单元格按顺序执行；各阶段独立缓存，K和绘图设置不触发重训。'''
    sources=[
'''# 1. 参数、读取数据、全部11份标本的细胞类型空间图
import os, sys
from pathlib import Path
from dataclasses import replace
from IPython.display import display, Image
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS'):
    os.environ.setdefault(key,'6')
PROJECT = Path('/home/xueshuailin/CCC_Phe')
if str(PROJECT / 'src') not in sys.path:
    sys.path.insert(0, str(PROJECT / 'src'))
from phenoniche.cohorts.cosmx_early_crc import EarlyCRCConfig, load_early_crc, prepare_crc_matrices, fit_crc
from phenoniche.cohorts.section_dual_workflow import cluster_niches
from phenoniche.cohorts.nsclc_analysis import summarize_niches
from phenoniche.cohorts.early_crc_reporting import (
    cell_atlas, spatial_atlas, crc_umap, embedding_atlas, multiscale_maps,
    niche_characterization, local_maps,
)

config = EarlyCRCConfig(
    device='cuda:2',
    epochs=30,                       # 上限；每轮打印loss；可续训
    minimum_training_epochs=20,       # 修改epochs时，此值也应<=epochs
    convergence_patience=7,
    encoder_genes=2048,
    batch_size=128,
    k=12, k_coarse=3, k_medium=6,     # 细分K可改；粗/中为同一h上的Ward合并
    lambda_geometry=1.0,
    point_size=1.2,
    local_point_size=8.,
    display_samples=('T1_NANO_021', 'T1_NANO_027'),  # 只决定后面的局部展示
    umap_neighbors=30, umap_min_dist=0.3, umap_epochs=250,
)
COUNTS = (config.k_coarse, config.k_medium, config.k)

def show(path, width=1450):
    if path is not None:
        display(Image(filename=str(path), width=width))

adata = load_early_crc(config)  # 首次仅转换原生RNA计数；以后直接读取H5AD
show(cell_atlas(adata, config))
''',
'''# 2. 构建有向CCC与组成矩阵：已有兼容缓存则读取，未完成可续算
# 同一30成员、sigma20µm距离衰减；图严格限制在同一患者标本内。
# GPU默认卡2；显存不足会自动等待，保留已有任务。
# 保留作者24类注释作为CCC目标；临床信息/原作者niche不用于建图和筛选。
# 不运行归一化表达插补；CCC用实测原始计数和完整覆盖的LR。
matrices = prepare_crc_matrices(adata, config)
''',
'''# 3. 训练当前GNN：两个直接预测MLP + h的近/中/远距离保持
# 每轮打印CCC、组成KL、距离及total；按训练total恢复最佳checkpoint。
# 已训练完成则直接读取；改K、展示患者或点大小不改变模型缓存。
fit = fit_crc(adata, matrices, config)
''',
'''# 4. Loss变化与CCC相对均值预测器的改善
import matplotlib.pyplot as plt
from phenoniche.cohorts.section_loss_curves import loss_curves
figure = loss_curves(fit, config)
plt.close(figure)
show(Path(fit) / 'figures/loss_curves.png')
''',
'''# 5. 固定全局尺度h96的K-means，以及全部11份标本的niche空间分布
# K是探索性颗粒度，不是按作者分期/已发表niche指定的真实类别数。
clusters = cluster_niches(fit, config)
show(spatial_atlas(adata, clusters, config))
''',
'''# 6. 联合图谱：niche、细胞类型、患者着色的同一套UMAP
# 输入就是聚类用的h，不用原作者UMAP；所有细胞参与，坐标缓存独立于K。
coordinates = crc_umap(adata, clusters, config)
show(embedding_atlas(adata, clusters, coordinates, config))
''',
'''# 7. 同一表示的粗/中/细层级：UMAP + 参数中两位患者的空间图
# 默认3/6/12；改变K后，从第5格重新执行即可，不重训。
show(multiscale_maps(adata, clusters, coordinates, config, counts=COUNTS))
''',
'''# 8. niche组成与有向CCC：原始观测支持 + 模型预测富集
# 分块汇总非线性预测，不写出完整细胞×CCC预测矩阵。
# 作者临床分期和已有niche均未参与这些分组。富集是描述性结果，不代表因果。
summary = summarize_niches(adata, matrices, fit, clusters, config)
show(niche_characterization(adata, summary, config))
''',
'''# 9. 局部上皮–间质界面：细胞类型 + 3/6/12类niche
# 参数指定展示患者；窗口只用细胞类型和几何选取，不用niche或分期挑图。
# 这些是空间案例，不预先命名为免疫浸润/排斥病理真值。
show(local_maps(adata, clusters, config, counts=COUNTS))
''']
    cells=[nbformat.v4.new_markdown_cell(intro),*[nbformat.v4.new_code_cell(s) for s in sources]]
    book=nbformat.v4.new_notebook(cells=cells,metadata=dict(
        kernelspec=dict(display_name='cccphe (Python3.11)',language='python',name='cccphe'),
        language_info=dict(name='python',version='3.11'),workflow='11-specimen CRC current directed-edge dual-MLP model',
        model_architecture='directed_cell_edge_direct_mlp'))
    if BOOK.exists():
        old=nbformat.read(BOOK,4)
        if any(c.get('outputs') for c in old.cells):
            raise RuntimeError('Notebook already contains results; preserve them and edit deliberately')
    for cell in book.cells:
        if cell.cell_type=='code':compile(cell.source,str(BOOK),'exec')
    nbformat.validate(book);nbformat.write(book,BOOK)
    print(BOOK)
    return BOOK


if __name__=='__main__':build()
