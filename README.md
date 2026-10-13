# CCC_Phe

项目保留既有 CCC-vector AE，同时提供独立下游 microenvironment GNN；两者共享原始 directed CCC 定义，独立保存模型和结果，临床 CCC/Cox 主线不变。

## 使用

在项目根目录打开 `XeniumPrime5K_Breast_niche_walkthrough.ipynb`。Notebook 直接读取 NAS 上的现有结果，不会因文件缺失自动启动训练。图片导出到 `/home/nas3/biod/xueshuailin/CCC_Phe/workspace/outputs/xenium_prime_5k/figures/`。

新 GNN 的真实数据报告：[XeniumPrime5K_GNN_niche_analysis.ipynb](/home/xueshuailin/CCC_Phe/XeniumPrime5K_GNN_niche_analysis.ipynb)。使用 cccphe kernel，Run All 将构建/审计输入、训练新版 GNN（完整结果存在时复用）并展示结果；图片和模型矩阵均在 NAS 独立输出目录。

已有环境：`conda activate cccphe`。需要安装时执行 `python -m pip install -e '.[dev]'`；运行测试：`python -m pytest`。

`python scripts/run_prime5k.py` 是计算入口；运行真实数据分析前，必须完成 `AGENTS.md` 要求的 ST anchor、BayesPrism 输出、临床终点、feature mask 和样本匹配审计。

## 目录

- `src/phenoniche/`：当前模型、训练、结果读取及报告模块。
- `src/phenoniche/ccc/`：独立的空间邻域、LR 图谱读取、directed cell-pair CCC 聚合与输入构建模块。
- `config/`：模型配置及 CCC 输入配置；大型文件使用 NAS 路径，小型 LR 图谱路径相对于项目根目录解析。
- `data/`：小型 LR 图谱及其来源审计。
- `scripts/`：计算入口和 notebook 生成脚本。
- `tests/`：合成数据测试与 notebook 语法检查。

## 大型文件存储

统一存储目录：`/home/nas3/biod/xueshuailin/CCC_Phe/workspace/`。代码和配置直接使用 NAS 地址，不使用符号链接。

- `outputs/xenium_prime_5k/`：模型权重、结果矩阵、模型选择表和图片。
- `data/ccc/cache/`：CCC 输入矩阵及元数据。
- `data/processed/`：processed AnnData 等大型输入。
- `artifacts/`：其它大型文件。
- `tmp/`：大型临时文件。

现有大型文件已迁入上述目录；今后的新增输出也放在这里。默认存储路径集中定义在 `src/phenoniche/storage.py`，运行配置中的路径在 `config/` 中设置。

## 分析边界

niche 分析属于独立下游模块。项目的 phenotype-associated CCC 主线仍遵循 `AGENTS.md`：

`current-study ST anchor -> BayesPrism bulk -> patient exact directed CCC -> graph-smoothed sparse Cox`

下游 niche 结果不能进入 ST-anchor 构建、Cox screening 或超参数选择。

## 微环境 GNN

当前 Encoder 只输入细胞表达与空间图，删除细胞类型 embedding 和所有上下游调控训练分支。每个微环境输出 `Z = activity × softmax(program logits)`；CCC 重构为 `Z @ H`，H 非负、逐行和为 1。当前不训练或展示细胞组成重构。

`L = masked scaled CCC Huber + lambda_reg × regularization`，默认 `lambda_reg=0.005`，`lambda_comp=0`。组成重构损失不计算、不参与梯度和模型选择；细胞类型仅用于有向 CCC 标签及下游描述性分析，不进入 Encoder。无 TF/ligand-target prior 训练依赖。

保持 5% LR 表达覆盖率、KNN=30、sigma=20 µm、原始 CCC 数值公式及训练集有效特征 mask。表达预处理、特征尺度和 mask 只在训练空间块拟合，邻域跨 fold 的中心被剔除，K=4–15 在训练块拟合、验证块选择。组成诊断及旧 AE 比较属于描述性分析，不证明 phenotype 或通讯因果。

打开根目录 `XeniumPrime5K_GNN_niche_analysis.ipynb`，选择 `cccphe` kernel，Run All。Notebook 调用项目模块完成 CCC 缓存构建、真实标签/特征/CCC 审计、训练、推理、niche 选择和可视化。完整且匹配配置的结果直接复用；已有不同配置/架构 checkpoint 拒绝覆盖。当前 CCC＋正则模型写入 NAS `outputs/microenvironment_gnn_ccc_only_expr05/`；已有组成约束模型的输出与权重保留；旧 AE 和旧 GNN 结果保留。

命令行等价入口：`python scripts/run_gnn.py ensure`；仅在审计已完成时训练用 `train`，推理用 `infer`。独立下游单组织分析明确审计 BayesPrism、临床终点与患者匹配不适用；结果不进入临床 CCC/Cox 主线。此次代码改造不执行真实训练或 Notebook。

## Tumor-edge 模拟实验

打开根目录 `TumorEdge_Communication_Simulation.ipynb`，选择 `cccphe` kernel，Run All。默认 8,000 cells、5 种类型、75 个真实 LR、两个各 6-LR 的 edge MI。调用已核验的 scCube API；合成种子除了类型 marker，还为每种细胞设置 3 个类型特异基础 LR。LR 身份来自现有数据库，类型表达分配是明确的合成假设，不冒充实验先验。计数均值校准保留原始 count reference；MI 仅注入 sender ligand / receiver receptor 表达，之后调用原 CCC 公式。

训练目标为 CCC 重构 + 0.005 × program regularization；无 Composition loss、Cell Type Embedding 或旧调控损失。CCC 使用训练集部分 RMS 归一化和 Huber error；native activity 参数在优化时重参数化，保存前合并回原 Linear，GNN 推理架构不变。niche 使用由 H 推导的 Z 距离，避免重复程序维度被标准化放大。K=4–15 选择不使用真实标签，另报固定 K=4 对照。

配置：`config/tumor_edge_simulation.json`。当前完整结果写入 NAS `outputs/tumor_edge_simulation_type_specific_lr/`；旧模拟和 Xenium 结果保留。镜像半环匹配径向/局部组成；单组织按镜像径向扇区划分并剔除共享细胞，真值仅用于生成和评估。Notebook 已执行并保存实际结果；匹配的完整结果复用。CLI：`python scripts/run_tumor_edge_simulation.py --run-full`。数值设置诊断入口 `scripts/tune_tumor_edge_simulation.py` 记录在较早的均一 LR 模拟上选择的参数；新旧模拟数据不同，不能声称分数提升全来自模型。
