from pathlib import Path
import nbformat as nbf
ROOT=Path(__file__).resolve().parents[1]

def build():
 m=nbf.v4.new_markdown_cell;c=nbf.v4.new_code_cell
 cells=[m('# Xenium 5K colitis — communication niches\n\n选择 **cccphe**，执行 **Restart Kernel → Run All**。读取已下载的健康（HC）、溃疡性结肠炎（UC）和免疫治疗相关结肠炎（ICI）数据；本次复用第32轮最佳权重，仅执行推理、聚类与出图，不追加训练；完整结果存在时直接复用。所有矩阵、模型及高清 PNG/PDF/SVG 保存到 NAS。\n\n用原始计数构建 CCC，逐组织片段建图；GNN 只输入表达和空间关系，学习32个非负程序。目标为 **CCC 重构＋0.005×程序正则**，NMF 初始化；不训练组成或基因表达分支。全部9位患者、35个保留组织片段的所有微环境共同参与训练，不留出独立测试集。固定训练子集只用于观察训练内重构曲线和无监督K选择，不称为独立验证。疾病分组不进入编码器、损失或K选择。Niche 基于相对程序组成，并保留较低权重的总体通讯强度，以减轻检测深度差异的影响。'),
 c("%matplotlib inline\nfrom pathlib import Path\nimport sys,contextlib,json\nimport matplotlib.pyplot as plt\nfrom matplotlib_inline.backend_inline import set_matplotlib_formats\nfrom IPython.display import display\nset_matplotlib_formats('retina')\nROOT=Path('/home/xueshuailin/CCC_Phe')\nsys.path.insert(0,str(ROOT/'src'))\nfrom phenoniche.cohorts.colitis import run\nfrom phenoniche.cohorts.colitis_reporting import ColitisReport\nCONFIG=ROOT/'config/colitis_5k.json'\nOUTPUT=Path(json.loads(CONFIG.read_text())['output_dir'])\nOUTPUT.mkdir(parents=True,exist_ok=True)\nwith (OUTPUT/'notebook_execution.log').open('a',buffering=1) as log,contextlib.redirect_stdout(log),contextlib.redirect_stderr(log):\n    run(CONFIG,postprocess_only=True)\nreport=ColitisReport(OUTPUT)\ndef show(fig):\n    display(fig);plt.close(fig)\ndisplay(report.overview())"),
 m('## 1. 空间组织与 communication niches\n\n依次展示 HC、UC、ICI 三组的全部9位患者。每张图三列分别对应该组的3位患者，展示每位患者的全部保留片段，各不相连片段仅在图上平移紧凑排列，并标注F编号；保留每个片段内部的原始形状、距离和微米尺度，模型使用的坐标不变。上排是作者细胞类型，下排是相同细胞的 niche，形状与坐标严格一致；标题列出细胞数与片段数。'),
 c("for condition in ('HC','UC','ICI'):\n    show(report.spatial(condition))"),
 m('## 2. 哪些 niche 在疾病中富集？\n\n热图显示全部9位患者；每个点是一位患者，横线为组均值。九位患者的全部组织片段均参与联合发现，所有空间图已在上一节完整展示。丰度以患者为单位，三组各3位；这里展示组间模式，不做细胞数放大的显著性检验。'),
 c('show(report.groups())'),
 m('## 3. Niche 的通讯程序与细胞类型组成\n\n左侧是 niche × program 活性，按列标准化以便比较；右侧用同一物理细胞对的空间权重汇总细胞组成，仅作解释，不作为模型辅助损失。'),
 c('show(report.niche_heatmaps())'),
 m('## 4. 32个通讯程序学到了什么？\n\n左侧显示程序平均活性占比，右侧展示非负字典 H 的代表性 sender→receiver→LR。H 的颜色按各程序最大 loading 缩放；原始 H 和完整排名另存 NAS。'),
 c('show(report.dictionary())'),
 m('## 5. 疾病相关程序的空间位置与代表性 CCC\n\n选择患者平均活性相对 HC 增加最多的 UC、ICI 程序用于训练后解释；每组展示细胞数最多患者的全部片段，标题标注患者；同一行三组共用颜色尺度。下表显示丰度至少3%、相对 HC 最富集的 UC/ICI niche 的 top10 CCC；所有 niche 的完整 top10 表在 NAS。富集表示描述性关联，不能解释为实验验证的通讯因果。'),
 c('show(report.program_spatial())\ndisplay(report.representative_ccc())\nprint("All niche top10 CCC:",OUTPUT/"niche_top10_CCC.csv")'),
 m('## 6. 训练与重构\n\n左：全体微环境的训练 CCC loss 与固定训练子集的监测 loss（评估模式、无梯度），虚线为选中 checkpoint；监测子集也参与训练。中：K=4–15 的队列内无监督选择；右：全队列抽样微环境的观测与重构 CCC。当前目标是联合发现，不输出独立测试性能；K选择不使用疾病标签。'),
 c('show(report.training())\nprint("All model results and figures:",OUTPUT)')]
 n=nbf.v4.new_notebook(cells=cells,metadata={'kernelspec':{'display_name':'cccphe','language':'python','name':'cccphe'},'language_info':{'name':'python'}})
 nbf.validate(n);p=ROOT/'Colitis_Xenium5K_Communication_Niches.ipynb';nbf.write(n,p);return p
if __name__=='__main__':print(build())
