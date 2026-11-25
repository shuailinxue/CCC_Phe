"""One result notebook for the full eight-section NSCLC analysis."""
from pathlib import Path
import nbformat

ROOT=Path(__file__).resolve().parents[1]
BOOK=ROOT/'CosMx_NSCLC_Communication_Niches.ipynb'


def build():
    sources=[
'''# 1. 全部8张切片：细胞类型与空间组织
import sys
from pathlib import Path
from IPython.display import display, Image

PROJECT = Path('/home/xueshuailin/CCC_Phe')
if str(PROJECT / 'src') not in sys.path:
    sys.path.insert(0, str(PROJECT / 'src'))
from phenoniche.cohorts.cosmx_nsclc import NSCLCConfig, load_nsclc, prepare_nsclc_matrices, fit_nsclc, wait_for_gpu
from phenoniche.cohorts.nsclc_analysis import cluster_nsclc, nsclc_umap, summarize_niches, region_candidates, audit_shared_geometry
from phenoniche.cohorts.nsclc_reporting import (
    cell_atlas, embedding_atlas, spatial_niche_atlas, query_mapping,
    niche_characterization, tumor_interfaces, immune_structures, matched_composition_example,
)

config = NSCLCConfig(
    device='cuda:2',
    epochs=30,                      # 上限；收敛后提前停止，恢复总loss最低的checkpoint
    minimum_training_epochs=20,
    convergence_patience=7,
    reference_k=12,                 # 参考图谱
    k=16,                           # 冻结推理后，全部8切片的探索性联合分群
    lambda_geometry=1.0,
    geometry_pairs_per_scale=8,
    geometry_distance_floor=0.05,
)

def show(path):
    if path is not None:
        display(Image(filename=str(path), width=1450))

adata = load_nsclc(config)
show(cell_atlas(adata, config))
''',
'''# 2. 当前模型：双MLP直接预测 + h的近/中/远距离保持
# 仅6张参考切片参与特征标定和训练；LUAD-5 R3、LUSC-6作为未训练查询。
# 最多30轮；允许收敛早停和降低学习率。按参考总loss保留最佳checkpoint。
# 不使用已发表niche标签/嵌入，不添加批次校正网络，不用查询目标选择checkpoint。
# 每轮CCC、组成、距离和总loss均保存在此输出及NAS训练记录中。
wait_for_gpu(config)
matrices = prepare_nsclc_matrices(adata, config)
fit = fit_nsclc(adata, matrices, config)
''',
'''# 3. 固定全局尺度h96的K-means，以及全切片niche分布
# 参考K-means只拟合参考细胞；联合K-means为冻结h上的下游探索。
clusters = cluster_nsclc(adata, fit, config)
show(spatial_niche_atlas(adata, clusters, config))
audit_shared_geometry(adata, matrices, fit, clusters, config)
''',
'''# 4. 图谱总览：K=12的niche、细胞类型、供者（复用原UMAP坐标）
# 仅本格比较K=12；其他单元格仍使用原分区。模型与UMAP不重算。
from dataclasses import replace
from phenoniche.cohorts.nsclc_niche_view import joint_partition_for_display

UMAP_K = 12
coordinates = nsclc_umap(adata, clusters, config)
umap_clusters = joint_partition_for_display(adata, clusters, config, k=UMAP_K)
show(embedding_atlas(adata, umap_clusters, coordinates, replace(config, k=UMAP_K)))
''',
'''# 5. 冻结参考映射：新重复切片与新供者
# 亲和度是到参考中心的相对距离分数，不是校准后的生物学概率。
show(query_mapping(adata, clusters, coordinates, config))
''',
'''# 6. niche的组成与有方向CCC解释
# 表达数据支持 + 训练后精确解码汇总；不会导出整张上万维预测矩阵。
summary = summarize_niches(adata, matrices, fit, clusters, config)
show(niche_characterization(adata, summary, config))
''',
'''# 7. 肿瘤界面与免疫空间结构
# 区域依据作者细胞类型和几何选取，不依据我们的niche标签挑图。
candidates = region_candidates(adata, matrices, clusters, config)
show(tumor_interfaces(adata, matrices, clusters, candidates, config))
show(immune_structures(adata, matrices, clusters, candidates, config))
''',
'''# 8. 进一步探索：同一切片内，组成相近而通讯不同的区域
# 只有满足非重叠、组成总变差<=0.10、单类型差<=0.06等条件才展示。
# 这是训练后探索案例，不作为独立验证或超过其它方法的证据。
show(matched_composition_example(adata, matrices, fit, clusters, candidates, config))
''',
'''# 9. 收敛概况（逐轮详细记录在第2格）
import matplotlib.pyplot as plt
from phenoniche.cohorts.section_loss_curves import loss_curves
figure = loss_curves(fit, config)
plt.close(figure)
show(Path(fit) / 'figures/loss_curves.png')
''']
    cells=[nbformat.v4.new_markdown_cell('''# CosMx NSCLC · Communication niches

完整8切片、5位供者。6张切片建立参考图谱，2张切片做冻结映射；模型保持有向细胞–边GNN、两个直接预测MLP及多尺度h距离约束。展示借鉴NicheCompass Fig. 5，所有结果均由本次模型计算。

原始研究：He et al., *Nature Biotechnology* (2022)。[数据](https://doi.org/10.6084/m9.figshare.25976224.v2)；[NicheCompass论文](https://doi.org/10.1038/s41588-025-02120-6)。''')]
    cells += [nbformat.v4.new_code_cell(source) for source in sources]
    cells[2].metadata['jupyter']={'outputs_hidden':True}
    book=nbformat.v4.new_notebook(cells=cells,metadata=dict(kernelspec=dict(display_name='cccphe (Python3.11)',language='python',name='cccphe'),
        language_info=dict(name='python',version='3.11'),workflow='Reference-only current GNN, frozen queries, full NSCLC result atlas'))
    if BOOK.exists():
        old=nbformat.read(BOOK,as_version=4)
        if len(old.cells)==len(book.cells):
            for a,b in zip(old.cells,book.cells):
                if b.cell_type=='code' and a.source==b.source:
                    b.outputs,b.execution_count=a.outputs,a.execution_count
    nbformat.validate(book);nbformat.write(book,BOOK)
    print(BOOK)
    return BOOK


if __name__=='__main__':build()
