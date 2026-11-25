"""Build the cached single-section COAD workflow and local immune-pattern view."""
from pathlib import Path
import nbformat as nbf

ROOT=Path(__file__).resolve().parents[1]
NOTEBOOK=ROOT/'SPATCH_COAD_Communication_Niches.ipynb'


def build():
    sources=[
'''# 1. 参数、读取数据，以及细胞类型的空间分布
import sys
from pathlib import Path
from dataclasses import replace
from IPython.display import display
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

PROJECT = Path('/home/xueshuailin/CCC_Phe')
if str(PROJECT / 'src') not in sys.path:
    sys.path.insert(0, str(PROJECT / 'src'))
from phenoniche.cohorts.section_dual_workflow import (
    SectionConfig, spatial_cells, cell_umap, prepare_matrices,
    train_model, cluster_niches, niche_map, granularity_maps,
)
from phenoniche.cohorts.spatch_coad import load_section

config = SectionConfig(
    data_root='/home/nas3/biod/xueshuailin/CCC_Phe/workspace/data/spatch_coad_xenium5k',
    output_root='/home/nas3/biod/xueshuailin/CCC_Phe/workspace/outputs/spatch_coad_xenium5k',
    device='cuda:2',              # 物理 GPU 2
    epochs=10,
    batch_size=128,
    learning_rate=0.001,
    hidden=96,
    lambda_geometry=1.0,         # 多尺度距离保持loss权重；CCC/组成权重均为1
    geometry_pairs_per_scale=8,  # 每个锚点在近/中/远各取8对；不是物理邻域大小
    geometry_distance_floor=0.05,# 相对误差的稳定下限，单位为典型联合平方距离的比例
    encoder_genes=2048,
    expression_coverage=0.10,
    umap_neighbors=30,
    umap_min_dist=0.3,
    umap_epochs=200,
    point_size=0.5,
)

def show(fig):
    display(fig)
    plt.close(fig)

# 沿用作者 high_quality=1 且 codex_common=1 的范围，与刚才预览的切片一致。
# 按原始细胞ID对齐表达/坐标/类型；网页行号不作为细胞ID。
# Epithelial 是作者上皮总类，尚未区分癌上皮和正常上皮。
# 下载、表达、矩阵、模型、图片及缓存都在NAS；相同配置直接读取。
adata = load_section(config)
show(spatial_cells(adata, config))
''',
'''# 2. 全部纳入细胞的表达 UMAP，按作者细胞类型着色
# 全部细胞参与PCA/UMAP，不做细胞下采样；兼容的UMAP坐标直接读取。
# 此UMAP不作为GNN输入，也不参与niche聚类。
show(cell_umap(adata, config))
''',
'''# 3. 构建或读取 CCC 矩阵X、细胞组成矩阵C
# 固定30成员邻域（包含中心），sigma=20µm距离衰减。
# CCC沿用有方向、不同物理细胞对及机会归一化；C使用相同权重的发出边度比例。
# 使用原始计数计算CCC；细胞类型只用于监督，作者spatial_cluster不参与学习。
matrices = prepare_matrices(adata, config)
''',
'''# 4. 训练或读取 GNN + 两个预测 MLP + h的多尺度距离保持loss
# LR按实际配对进入有向边；两轮细胞/边联合更新 -> h96。
# 两个MLP直接预测CCC和组成；无ZH/PB、程序字典或NMF初始化。
# 总loss = CCC Huber + 组成KL + lambda_geometry × 近/中/远距离相对误差Huber。
# 参考距离只来自计算出的CCC/组成，不使用指定T细胞、坐标、病理、niche或结局。
# 每轮打印CCC、组成、距离loss（含近/中/远分项）、总loss；第0轮为未训练起点。
# h在初始训练样本上确定一个固定全局尺度；训练/聚类共用，不逐细胞单位化。
# 导出h96和组成；不再导出整张上万维CCC预测矩阵。中断可继续。
# 本次为单切片全细胞探索性拟合，不是独立验证集实验。
fit = train_model(adata, matrices, config)
''',
'''# 5. 每轮loss：CCC、组成、多尺度距离保持与总loss
from phenoniche.cohorts.section_loss_curves import loss_curves

# 同一批细胞、同一矩阵、相同Huber/KL；旧模型曲线从第1轮开始。
# 旧参照只比较CCC/组成；没有多尺度距离loss，因此不比较总loss。
PREVIOUS_FIT = '/home/nas3/biod/xueshuailin/CCC_Phe/workspace/outputs/spatch_coad_xenium5k/fits/652369278bc33047'
show(loss_curves(fit, config, reference_fit=PREVIOUS_FIT))
''',
'''# 6. 修改K，仅重新聚类和出图；不重算CCC、不重训GNN
K = 12
config = replace(config, k=K)
# 直接对受多尺度距离loss约束的h96做固定全局尺度校准，再做K-means。
# 不展开CCC预测、不拼接分支输出、不训练额外映射、不做空间平滑。
clusters = cluster_niches(fit, config)
show(niche_map(adata, fit, clusters, config))
''',
'''# 7. 同一套表示的粗、中、细三个层次（均可修改）
COARSE_K = 3
MEDIUM_K = 6
config = replace(config, k_coarse=COARSE_K, k_medium=MEDIUM_K)
# 要求 2 <= 粗 <= 中 <= 上一格K。
# 细层沿用K-means；在同一联合空间按细胞数加权的Ward准则逐级合并。
# 保证每个细类只有一个中类/粗类归属，模型和表示不变。
show(granularity_maps(adata, fit, clusters, config))
''',
'''# 8. 局部对照：严格按两个红框边界，右框第一行、左框第二行
from phenoniche.cohorts.spatch_coad_regions import block_edge_local_maps

# 截图横纵缩放比例不同；用完整仿射配准和绘图坐标逆变换定位。
# 每行使用各自红框的实际边界，不统一扩成更大的方框。单位µm。
# 只显示细胞类型和既有3、6、12类niche，标签/颜色不变。
REGION_BOUNDS = (
    (4002.1602539695127, 4131.982408682338, 4629.1918302126, 4774.120272958289),  # 右框：第一行（xmin, ymin, xmax, ymax）
    (2334.412943314517, 4864.141719167308, 2828.0331758369125, 5377.852420976276),  # 左框：第二行
)
show(block_edge_local_maps(adata, fit, clusters, config,
                          region_bounds=REGION_BOUNDS, point_size=10.))

# 仅下游检查用户指出的同一颗T细胞，不进入训练、距离标定或K-means。
from phenoniche.cohorts.section_geometry_audit import nominated_cell_audit
display(nominated_cell_audit(adata, matrices, fit, clusters,
                             cell_id='cdnfjjbp-1', box_bounds=(4320.,4510.,4420.,4610.), counts=(3, 6, 12)))
''',
'''# 9. 聚类所用h96的 UMAP：3、6、12 类 niche
from phenoniche.cohorts.spatch_coad_niche_umap import niche_umap_maps

# 使用聚类时保存的固定全局尺度h96，不使用基因表达或预测输出拼接向量。
# 全部细胞共用一套UMAP坐标；3/6类沿用12类的层级合并，标签/颜色与空间图一致。
# 不重新训练；UMAP只用于展示，相同配置直接读取缓存。
show(niche_umap_maps(adata, clusters, config, counts=(3, 6, 12), point_size=1.))
''',
]
    nb=nbf.v4.new_notebook(cells=[nbf.v4.new_code_cell(s) for s in sources])
    nb.metadata.update(kernelspec=dict(display_name='cccphe (Python3.11)',language='python',name='cccphe'),
        language_info=dict(name='python',version='3.11'),
        workflow='Directed cell-edge encoder; direct CCC/composition MLPs; multiscale distance loss on shared h; fixed-global h KMeans; per-epoch losses')
    if NOTEBOOK.exists():
        old=nbf.read(NOTEBOOK,as_version=4)
        changed_training=(old.cells[0].source!=nb.cells[0].source or old.cells[3].source!=nb.cells[3].source)
        for index,(a,b) in enumerate(zip(old.cells,nb.cells)):
            if a.source==b.source and (not changed_training or index<3):
                b.outputs,b.execution_count=a.outputs,a.execution_count
    nbf.validate(nb);nbf.write(nb,NOTEBOOK);print(NOTEBOOK)
    return NOTEBOOK


if __name__=='__main__':build()
