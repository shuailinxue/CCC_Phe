"""Separate richer simulation notebook, reusing the existing reporting cells."""
from pathlib import Path
import nbformat
ROOT=Path(__file__).resolve().parents[1]


def build():
    template=ROOT/'TumorEdge_CCC_Expression_Reconstruction.ipynb'
    notebook=nbformat.read(template,as_version=4)
    notebook.cells[0].source=(
        '# Tumor-edge：3,000 基因背景与稀疏通讯变化\n\n'
        '选择 **cccphe**，执行 **Restart Kernel → Run All**。这是独立实验，原模拟和原 Notebook 保留。'
        '8,000 个细胞、五种类型、四个真实 MI、32 维 Z；沿用原 KNN30、sigma=20 µm 和 CCC 定义。\n\n'
        '**背景：**3,000 个编码器基因，其中 150 个 LR 基因、25 个标记基因、1,200 个类型模块基因、800 个共享状态基因和 825 个管家基因。'
        '背景通过 scCube 生成并进行显式类型均值校准，再加入共享状态、文库大小和负二项变异；这些是公开记录的模拟假设。'
        '两个半环的类型、径向位置和原始非 LR 表达逐对匹配。\n\n'
        '**区域通讯活性：**Tumor core 的 MI-3、Normal exterior 的 MI-4 在各自区域内为 1，取消原来额外的 50 µm 边界衰减。两个边缘 MI 不变；KNN 混合仍产生真实的边界过渡。旧数据与旧结果保留，新场景重新计算 CCC 并训练所有方法。\n\n'
        '**通讯注入：**两个边缘 MI 涉及 24 个基因（面板的 0.8%）；sender ligand/receiver receptor 由独立潜在表达率注入（弱基础率 0.15＋峰值增量 3），未分配 LR 使用类型相关背景；'
        '使用一次 Gamma–Poisson 计数抽样和独立 MI 增量，允许零表达细胞被激活。训练细胞类型内表达覆盖率 ≥10%，先确定有效监督 mask 再拟合 CCC 尺度。只修改真实表达，不修改 CCC 矩阵。'
        '文库归一化可能带来小幅间接背景差异，实际 LR 计数比例等审计写入 NAS。此实验不是严格的 spatial-pairing-only 模拟，也不保证某种方法获胜。\n\n'
        '**模型：**NMF 初始化后的 CCC-only 阶段只有 CCC＋正则影响 Z，基因 decoder 拟合时冻结 Z。另行记录受保护的联合尝试（表达梯度上限 10%，验证 CCC 波动上限 0.5%），只有 CCC 保护与基因验证改善同时通过才采纳，否则保留冻结起点。'
        '额外 24 个模拟响应基因和 12 个阴性对照只作表达标签，不进入编码器。'
        '**Niche 分配：**聚类表示保留 80% 自身、加入 20% 现有距离权重下的邻居表示，再用完整 KMeans；训练/验证/测试的物理细胞隔离保持。只调整聚类，不改变模型、Z、H、CCC 或基因重构。自动 K=4–15 使用同一调整后的表示。'
        '图表保持原 Notebook 的顺序，末尾比较 GraphST、CellCharter(scVI)、scNiche、NicheCompass、SpaNiche、BANKSY 和 STAGATE。完整数据、模型和图片均写入新的 NAS 输出目录，已完成且一致的结果可复用。'
    )
    for cell in notebook.cells:
        if cell.cell_type=='code':
            cell.outputs=[];cell.execution_count=None
            cell.metadata.pop('execution',None)
            cell.source=cell.source.replace("ROOT/'config/tumor_edge_expression.json'","ROOT/'config/tumor_edge_gene_rich_expression.json'")
            if 'run_expression_experiment(config)' in cell.source:
                cell.source=cell.source.replace('from phenoniche.training import set_seed','from phenoniche.training import set_seed\nfrom phenoniche.simulation.boundary_background import prepare_boundary_background')
                cell.source=cell.source.replace('set_seed(config.seed)\n','set_seed(config.seed)\nif torch.cuda.is_available():torch.cuda.set_device(2)\n')
                cell.source=cell.source.replace('    prepare_environment()\n    run_expression_experiment(config)',"    prepare_environment()\n    prepare_boundary_background(config.base_simulation_config,ROOT/'config/tumor_edge_gene_rich_background.json')\n    run_expression_experiment(config)")
                cell.source=cell.source.replace('import sys,contextlib','import sys,contextlib,json')
                cell.source=cell.source.replace('from phenoniche.simulation.boundary_background import prepare_boundary_background','from phenoniche.simulation.boundary_background import prepare_boundary_background\nfrom phenoniche.simulation.niche_refinement import refine_saved_niches')
                cell.source=cell.source.replace('OUTPUT=Path(config.output_dir)',"MODEL_OUTPUT=Path(config.output_dir)\nrefinement=json.loads((ROOT/'config/tumor_edge_gene_rich_niche_refinement.json').read_text())\nOUTPUT=Path(refinement['output_dir'])")
                cell.source=cell.source.replace('    run_expression_experiment(config)','    run_expression_experiment(config)\n    refine_saved_niches(MODEL_OUTPUT,OUTPUT,refinement[\'mixing\'])')
        else:
            cell.source=cell.source.replace('原始 256 个基因','原始 3,000 个基因')
            cell.source=cell.source.replace('现有 GNN checkpoint 不重训、不改参数','新场景先训练 GNN；baseline 比较阶段不修改已完成的 GNN checkpoint')
            if cell.metadata.get('ccc_baseline_comparison') and '## 8.' in cell.source:
                cell.source=(
                    '## 8. 与空间聚类 baseline 比较\n\n'
                    '比较 [GraphST](https://github.com/JinmiaoChenLab/GraphST)、[CellCharter](https://github.com/CSOgroup/cellcharter)（官方 scVI 流程）、'
                    '[scNiche](https://github.com/ZJUFanLab/scNiche)、[NicheCompass](https://github.com/Lotfollahi-lab/nichecompass)、'
                    '[SpaNiche](https://github.com/SiyuanHuang1/SpaNiche)、[BANKSY](https://github.com/prabhakarlab/Banksy_py)、'
                    '[STAGATE](https://github.com/RucDongLab/STAGATE_pyG)。使用固定版本的官方模型/训练核心；适配代码与来源记录在 `baselines/`。\n\n'
                    '**公平协议：**同一份新场景的 8,000 个细胞、3,000 个原始表达基因；新增响应基因不进入任何 baseline。'
                    '表示模型、PCA、scVI、特征筛选及归一化只拟合原 GNN 训练物理细胞，聚类只拟合训练中心。所有主比较固定 K=4，使用同一批严格空间测试中心；不按真实区域标签调参。新表达场景重新训练各方法，旧数据上的分数不混入。\n\n'
                    '**保留各方法定义：**GraphST 为官方稀疏模型；CellCharter 为原始 counts 的 scVI（ZINB、10维）＋官方 Delaunay 图（三层聚合）＋官方 GMM；scVI 在独立兼容环境运行，所有方法统一测试集额外审计其三层邻域；scNiche 为中心表达、邻域表达、细胞组成的三视图 MGAE/GFN/MMIM。'
                    'NicheCompass 为官方空间 VGAE，以整个公开 LR 库在真实基因面板中的交集构造 LR 程序，并保留 de novo 程序，使用官方 active-program 表示聚类；这是 LR-only prior 变体，不声称有合成基因的真实 TF/target 先验。'
                    'SpaNiche 为官方 IntNMF 初始化＋图正则联合 NMF；单细胞类型替代 spot 去卷积，视图半径由训练 KNN30 尺度决定，Gaussian sigma=20 µm。其测试推理固定训练 H，再按原目标估计 W；官方 topic-argmax 原始结果也保存。\n\n'
                    'BANKSY 保留官方邻域均值与 centered AGF 表达梯度，采用组织分区参数 λ=0.8、K15/K30；训练邻居按物理 fold 隔离，标准化与 PCA 只拟合训练细胞；冻结后在完整组织图推理，统一测试集额外检查它实际使用的邻居。'
                    'STAGATE 保留官方表达 MSE、图注意力与 decoder 权重共享，hidden=512、latent=30、1,000 epochs；仅训练空间区域内的诱导图用于优化，推理冻结权重。两者均使用完整的 3,000 基因，不使用细胞类型或 CCC。\n\n'
                    '**信息条件不同：**GraphST/CellCharter/BANKSY/STAGATE 使用表达与坐标；scNiche/SpaNiche 另使用观测细胞类型；NicheCompass/SpaNiche 另使用公开 LR 先验；CCC-AE 输入预计算 CCC，我们的 GNN 以 CCC 监督。'
                    '真实 niche 标签、MI 活性及 MI loading 只用于拟合结束后的评价和配色。参数预算、依赖版本、适配差异与原始 cluster ID 均保存，不保证任何方法获胜。'
                )
            # Coverage is only knowable after the new spatial simulation runs.
            cell.source=cell.source.replace('严格两跳划分后没有肿瘤核心测试中心，因此不能宣称核心区基因恢复已获独立验证','严格两跳划分可能排除部分区域；下方测试覆盖图给出实际覆盖，不对未覆盖区域宣称独立验证')
    notebook.metadata.pop('widgets',None)
    path=ROOT/'TumorEdge_GeneRich_CCC_Expression_Reconstruction.ipynb';nbformat.validate(notebook);nbformat.write(notebook,path)
    return path

if __name__=='__main__':print(build())
