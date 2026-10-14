"""Build the concise CCC-priority joint-expression simulation notebook."""
from pathlib import Path
import nbformat
ROOT=Path(__file__).resolve().parents[1]

def build():
    notebook=nbformat.v4.new_notebook();cells=[]
    def md(text):cells.append(nbformat.v4.new_markdown_cell(text))
    def code(text):cells.append(nbformat.v4.new_code_cell(text))
    md('# Tumor-edge：CCC 主任务与基因表达联合重构\n\n选择 **cccphe**，执行 **Restart Kernel → Run All**。复用同一份 8,000-cell 模拟数据和 NMF 初始化的 CCC 预训练模型；联合阶段让基因表达损失参与更新 GNN/Z，H 只由 CCC 更新。表达梯度冲突投影且范数不超过 CCC 梯度的 10%；验证 CCC 相对预训练最多允许 0.5% 波动，再选择表达验证损失最小的合格 checkpoint。若没有合格改进，保留基线并如实标注。结果独立存入 NAS，不覆盖旧模型。\n\n表达标签是**微环境内各细胞类型的空间加权表达**，缺失类型不监督；包含 24 个由实际 LR 信号驱动的模拟响应基因与 12 个阴性对照，均不进入编码器。它们是模拟基因，不称为真实上游调控因子或通路。')
    code("%matplotlib inline\nfrom pathlib import Path\nimport sys,contextlib\nimport matplotlib.pyplot as plt\nimport torch\nfrom IPython.display import display\nROOT=Path('/home/xueshuailin/CCC_Phe')\nsys.path.insert(0,str(ROOT/'src'))\nfrom phenoniche.simulation.dependencies import prepare_environment\nfrom phenoniche.simulation.response_simulation import ExpressionExperimentConfig\nfrom phenoniche.simulation.response_experiment import run_expression_experiment\nfrom phenoniche.simulation.joint_reporting import JointExpressionReport\nfrom phenoniche.training import set_seed\nconfig=ExpressionExperimentConfig.from_json(ROOT/'config/tumor_edge_expression.json')\nOUTPUT=Path(config.output_dir)\nOUTPUT.mkdir(parents=True,exist_ok=True)\ntorch.set_num_threads(6)\nset_seed(config.seed)\nif config.device.startswith('cuda:') and not torch.cuda.is_available():\n    raise RuntimeError('请在服务器 cccphe GPU 环境中运行')\nwith (OUTPUT/'notebook_execution.log').open('a',buffering=1) as log,contextlib.redirect_stdout(log),contextlib.redirect_stderr(log):\n    prepare_environment()\n    run_expression_experiment(config)\nreport=JointExpressionReport(OUTPUT)\ndef show(fig):\n    display(fig)\n    plt.close(fig)")
    md('## 1. 模拟组织与模型 niche\n\n显示原始聚类编号，不用真实标签替换模型输出。CCC-only 与联合模型使用相同数据，联合模型 K=4–15 的选择只用训练/验证集，不读取真实 niche。')
    code('show(report.spatial_comparison())')
    md('## 2. 四个真实通讯程序是否恢复\n\n上下两行使用相同数值尺度；下方由 **Z @ H** 沿预先定义的 MI 字典投影，汇总全部 32 个程序的贡献，不强制一个 MI 对应一个 Z 维度。真实字典只用于训练后的评估。')
    code('show(report.communication_recovery())')
    md('## 3. 基因表达是否对应\n\n同一列是同一个模拟响应基因；每个 niche 内按该基因的 receiver 类型汇总空间加权表达，再减去该类型中这个基因的观测均值。上下两幅共用色标和参考均值，蓝色较低、红色较高。热图展示**全组织（含训练和验证中心）的描述性结果**。下方另用空间测试集对比基线、冻结版和联合版的 RMSE，并显示测试覆盖；严格两跳划分后没有肿瘤核心测试中心，因此不能宣称核心区基因恢复已获独立验证。')
    code('show(report.expression_recovery())')
    md('## 4. Niche 与细胞类型的关系\n\n同一物理细胞对权重计算的空间加权组成。此图仅用于下游解释，不增加细胞组成重建损失。')
    code('show(report.niche_cell_types())')
    md('## 5. 每个 niche 的 top10 CCC\n\n对 niche 内 Z 取均值后乘 H：**mean(Z | niche) @ H**。表中列出 sender→receiver→LR、平均重构强度及贡献最大的程序；niche 可由多个程序共同形成。排序按原始 CCC 单位计算，不代表临床显著性或因果作用。')
    code("display(report.top_ccc.rename(columns={'niche':'Niche','rank':'Rank','sender':'Sender','receiver':'Receiver','LR':'Ligand–Receptor','mean_reconstructed_CCC':'Mean CCC','main_program':'Main program'}).drop(columns=['LR_id']).style.format({'Mean CCC':'{:.3f}'}).hide(axis='index'))")
    md('## 6. 训练 loss\n\n分别显示 CCC 预训练和联合阶段的两个任务，数值尺度不同。虚线标出选中的 joint epoch，点线表示验证 CCC 的接受上限；epoch 0 是冻结版起点。')
    code('show(report.curves())')
    md('## 7. 模拟数据与模型结果总图\n\n真实四区域、细胞类型、CCC-only 和联合 niche 并排显示；下方比较四个真实 MI 与恢复的通讯强度。**仅这张总图**使用训练中心拟合的标签颜色映射，便于直观对照；第一图和表中仍保留原始编号。底部指标由 held-out test 实际计算，标签匹配未进入训练或 K 选择。高分是本模拟、当前随机种子的结果，不是临床验证。全部 PNG/PDF/SVG、指标和模型数组在 NAS 输出目录。')
    code('show(report.summary_figure())')
    import sys
    sys.path.insert(0,str(ROOT))
    from baselines.notebook_cells import cells as baseline_cells
    cells.extend(baseline_cells())
    notebook.cells=cells;notebook.metadata={'kernelspec':{'display_name':'cccphe','name':'cccphe','language':'python'},'language_info':{'name':'python'}}
    path=ROOT/'TumorEdge_CCC_Expression_Reconstruction.ipynb';nbformat.write(notebook,path);return path
if __name__=='__main__':print(build())
