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
        '**通讯注入：**两个边缘 MI 涉及 24 个基因（面板的 0.8%）；sender ligand/receiver receptor 在激活位置的条件期望最多增加到原来的 2 倍，'
        '使用 Poisson 计数增量，保留零表达与噪声。只修改真实表达，不修改 CCC 矩阵。'
        '文库归一化可能带来小幅间接背景差异，实际 LR 计数比例等审计写入 NAS。此实验不是严格的 spatial-pairing-only 模拟，也不保证某种方法获胜。\n\n'
        '**模型：**复用 NMF 初始化、CCC 预训练、受保护的 CCC＋表达联合训练；表达梯度上限 10%，验证 CCC 波动上限 0.5%。'
        '额外 24 个模拟响应基因和 12 个阴性对照只作表达标签，不进入编码器。'
        '图表保持原 Notebook 的顺序，末尾保留 GraphST 和 CellCharter(PCA) 比较。完整数据、模型和图片均写入新的 NAS 输出目录，已完成且一致的结果可复用。'
    )
    for cell in notebook.cells:
        if cell.cell_type=='code':
            cell.outputs=[];cell.execution_count=None
            cell.metadata.pop('execution',None)
            cell.source=cell.source.replace("ROOT/'config/tumor_edge_expression.json'","ROOT/'config/tumor_edge_gene_rich_expression.json'")
            if 'run_expression_experiment(config)' in cell.source:
                cell.source=cell.source.replace('from phenoniche.training import set_seed','from phenoniche.training import set_seed\nfrom phenoniche.simulation.rich_background import prepare_rich_background')
                cell.source=cell.source.replace('    prepare_environment()\n    run_expression_experiment(config)',"    prepare_environment()\n    prepare_rich_background(config.base_simulation_config,ROOT/'config/tumor_edge_gene_rich_background.json')\n    run_expression_experiment(config)")
        else:
            cell.source=cell.source.replace('原始 256 个基因','原始 3,000 个基因')
            # Coverage is only knowable after the new spatial simulation runs.
            cell.source=cell.source.replace('严格两跳划分后没有肿瘤核心测试中心，因此不能宣称核心区基因恢复已获独立验证','严格两跳划分可能排除部分区域；下方测试覆盖图给出实际覆盖，不对未覆盖区域宣称独立验证')
    notebook.metadata.pop('widgets',None)
    path=ROOT/'TumorEdge_GeneRich_CCC_Expression_Reconstruction.ipynb';nbformat.validate(notebook);nbformat.write(notebook,path)
    return path

if __name__=='__main__':print(build())
