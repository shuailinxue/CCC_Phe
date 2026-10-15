from pathlib import Path
import nbformat as nbf
import json

ROOT=Path(__file__).resolve().parents[1]


def build():
    md=nbf.v4.new_markdown_cell;code=nbf.v4.new_code_cell
    cfg=json.loads((ROOT/'config/healthy_cancer_communication.json').read_text())
    pairs=cfg['pairs']+cfg.get('extra_training_pairs',0)
    train_pairs=pairs-2
    n=2*pairs*cfg['cells_per_slice']
    cells=[md(f'# Healthy–Cancer communication niches\n\n选择 **cccphe** → **Restart Kernel and Run All**。{pairs}对独立切片（共{n:,}细胞），{train_pairs}对训练、1对验证、1对测试。联合学习共享 GNN 和 H，每张切片独立建图。只注入表达，不修改 CCC；真实区域与分组不进入模型或 K 选择。\n\n**本次对照：**完整保留150个LR基因，另选通过训练覆盖/方差审计的预定义类型标记；背景状态没有通讯生成依据，不强行纳入编码器。新增独立训练切片的边界形态、通讯斑块方位与背景状态相位均有变化；每轮监督中心仍为8,000，原验证/测试切片完全不变。末尾提供原模型、只改输入、输入＋多样性三项比较，CCC误差统一用原训练尺度计算。\n\n共享组织结构、免疫斑块和弯曲界面；癌症部分界面切换两个 LR 程序并叠加免疫通讯响应。局部类型/坐标在配对切片间严格匹配，表达独立抽样。两个条件均使用 Epithelial 标签，避免靠 Tumor 标签辨别疾病。\n\n这是从既有 scCube 背景统计独立抽样的**受控通讯状态仿真**，不是完整癌症生物学；8个设计程序、6类真值不等于32维 Z，也不用于指定 K。训练只使用 CCC Huber＋正则，保留当前 NMF 初始化和20%局部 niche 聚合；不训练额外组成或疾病损失。'),
    code("%matplotlib inline\nfrom pathlib import Path\nimport sys,contextlib,json\nimport matplotlib.pyplot as plt\nfrom matplotlib_inline.backend_inline import set_matplotlib_formats\nset_matplotlib_formats('retina')\nfrom IPython.display import display\nROOT=Path('/home/xueshuailin/CCC_Phe')\nsys.path.insert(0,str(ROOT/'src'))\nfrom phenoniche.simulation.paired_tissues import PairedConfig\nfrom scripts.compare_paired_training import main as compare_training\nfrom phenoniche.simulation.paired_reporting import PairedReport\nconfig=PairedConfig.from_json(ROOT/'config/healthy_cancer_communication.json')\nOUTPUT=Path(config.output_dir);OUTPUT.mkdir(parents=True,exist_ok=True)\nwith (OUTPUT/'notebook_execution.log').open('a') as log,contextlib.redirect_stdout(log),contextlib.redirect_stderr(log):\n    compare_training()\nreport=PairedReport(OUTPUT)\ndef show(fig):\n    display(fig);plt.close(fig)\nprint(f\"{report.summary['slices']} slices · {report.summary['cells']:,} cells · {report.summary['expression_genes']} encoder genes · K={report.summary['selected_K']}\")"),
    md('## 1. 未见过的健康/癌症切片：细胞类型、真实 niche 与预测\n\n评估配色只在训练细胞上匹配；灰色表示未匹配 cluster。细胞类型图单独展示，下方2×2主图直接对比真值和预测；图例放在图外。原始 cluster ID 另存 NAS，不合并为理想真值。'),
    code('show(report.cell_types())\nshow(report.spatial())\ndisplay(report.metrics.query("scope == \'heldout_slices\'").round(3))'),
    md('## 2. 哪些 niche 在患病组富集？是否只是细胞组成？\n\n候选仅按训练切片频率定义：健康 <2%、癌症 >3%、平滑频率比 >3。配对点图中每条线连接一对切片，菱形为留出测试切片；浅色底标识训练中定义的疾病候选，不表示显著性。组成热图标注百分数。下表独立显示测试切片频率，不用测试结果筛选候选。重复数较少，不做患者级推断或伪造显著性。'),
    code('show(report.groups())\ndisplay(report.enrichment.round(3))'),
    md('## 3. 恢复了什么通讯程序？\n\n真值活性按原物理细胞对和距离权重汇总。通过 H 与真实 LR loading 的相似性匹配程序，再展示未见癌症切片上的 Z；允许多个真实程序对应同一学习维度，如实展示恢复不足。每行使用统一色标；真值与Z的数值单位不同，分别标尺。ρ基于原始活性计算，完整数据未改变。'),
    code('show(report.programs())\ndisplay(report.recovery.iloc[5:].round(3))\ncommunications=report.candidate_communications()\nif len(communications):\n    display(communications.round(3).head(30))\nelse:\n    print("No niche satisfies the predeclared disease-specific criterion; inspect enrichment and recovery above.")'),
    md('## 4. 训练是否正常，K 如何选择？\n\nCCC 误差为训练集特征尺度下的 Huber 损失；竖虚点线标记实际选用的最佳 epoch。K=4–15 使用验证切片的轮廓系数、稳定性和簇规模选择，不使用分组或真实 niche。所有完整数据、矩阵、模型、原始标签及 PNG/PDF/SVG 均在 NAS。'),
    code('show(report.training())\ncomparison=__import__("pandas").read_csv(OUTPUT/"experiment_comparison.csv")\ndisplay(comparison[["experiment","encoder_genes","training_slices","training_centers","best_epoch","common_validation_Huber","common_test_Huber","heldout_ARI","heldout_disease_F1"]].round(3))\naudit=__import__("pandas").read_csv(OUTPUT/"pair_audit.csv")\nassert audit.matched_types.all() and audit.matched_coordinates.all()\nassert audit.max_weighted_composition_difference.max()<1e-6\nprint("Matched-composition audit passed. Results:",OUTPUT)')]
    n=nbf.v4.new_notebook(cells=cells,metadata={'kernelspec':{'display_name':'cccphe','language':'python','name':'cccphe'},'language_info':{'name':'python'}})
    nbf.validate(n);path=ROOT/'HealthyCancer_Communication_Niches.ipynb';nbf.write(n,path);return path

if __name__=='__main__':print(build())
