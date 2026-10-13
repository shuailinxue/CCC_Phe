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

当前 Encoder 只输入细胞表达与空间图，删除细胞类型 embedding 和所有上下游调控训练分支。每个微环境输出 `Z = activity × softmax(program logits)`；CCC 重构为 `Z @ H`，空间加权细胞组成为 `P @ B`。H 非负、逐行和为 1；B 非负、逐行和为 1，分别学习并共享同一组通讯程序。

`L = masked scaled CCC Huber + lambda_comp × KL(pi || P @ B) + lambda_reg × regularization`，默认 `lambda_comp=0.005`。CCC 是主任务，冲突的辅助梯度仍按 CCC 梯度投影。pi 按原 CCC 的有向物理细胞对权重汇总 sender degree 后归一化；细胞类型只用于 CCC/组成标签及下游分析，不进入 Encoder。无 TF/ligand-target prior 训练依赖。

保持 5% LR 表达覆盖率、KNN=30、sigma=20 µm、原始 CCC 数值公式及训练集有效特征 mask。表达预处理、特征尺度和 mask 只在训练空间块拟合，邻域跨 fold 的中心被剔除，K=4–15 在训练块拟合、验证块选择。组成诊断及旧 AE 比较属于描述性分析，不证明 phenotype 或通讯因果。

打开根目录 `XeniumPrime5K_GNN_niche_analysis.ipynb`，选择 `cccphe` kernel，Run All。Notebook 调用项目模块完成 CCC 缓存构建、真实标签/特征/CCC 审计、训练、推理、niche 选择和可视化。完整且匹配配置的结果直接复用；已有不同配置/架构 checkpoint 拒绝覆盖。当前弱组成约束结果写入 NAS `outputs/microenvironment_gnn_composition_weak_expr05/`，已有 `outputs/microenvironment_gnn_composition_expr05/` 结果保留；旧 AE 和旧 GNN 结果保留。

命令行等价入口：`python scripts/run_gnn.py ensure`；仅在审计已完成时训练用 `train`，推理用 `infer`。独立下游单组织分析明确审计 BayesPrism、临床终点与患者匹配不适用；结果不进入临床 CCC/Cox 主线。此次代码改造不执行真实训练或 Notebook。

## Tumor-edge 模拟实验

打开根目录 `TumorEdge_Communication_Simulation.ipynb`，选择 `cccphe` kernel，Run All。默认 8,000 cells、5 种类型、75 个真实 LR、两个各 6-LR 的设计 MI。调用已核验的 scCube VAE/Poisson 空间 API，使用自包含的合成种子表达，随后仅修改 sender ligand / receiver receptor 表达，再调用原 CCC 公式及最新版 expression-only GNN。镜像类型/空间布局匹配两半环的径向与局部组成；真值只在生成和评估模块中使用，不进入训练或 K 选择。可选 `spatial_pairing_only` 默认关闭。

配置：`config/tumor_edge_simulation.json`。所有数据、模型、指标和 PNG/PDF 写入 NAS `outputs/tumor_edge_simulation_calibrated/`（已有未校准实验 `outputs/tumor_edge_simulation/` 保留）。主图显示未对齐的原始固定 K=4 聚类；GNN 自适应 K=4–15、训练标签拟合的评估匹配及 held-out/transductive 指标分别保存。对照信息条件不同：组成、无类型的邻域表达 PCA、原 CCC-vector AE、表达/空间 GNN。失败结果照实展示。默认单组织使用镜像配对的完整径向扇区并剔除跨 fold 邻域；微型测试使用三个样本级划分的独立小组织。

scCube source commit/模块校验记录在 `data/scCube_source_audit.json`，环境准备不会安装其旧 requirements 中的 NumPy/PyTorch 降级依赖。已有安装时 Run All 无需下载；缺失时安装固定源码及缺失依赖。CLI 完整运行必须显式 `python scripts/run_tumor_edge_simulation.py --run-full`；轻量验证用 `--smoke`。Notebook 生成器仅生成文件，不执行任何单元格。

模拟计数显式校准至合成种子的类型基因均值，保留 scCube 的类型内相对变化，并保存未校准计数及 marker audit；不使用 niche/MI 标签校准。B 可用训练集 CCC 原型对应的观测组成初始化，不改变 GNN 架构。H 初始化相似度与训练后相似度分别报告，避免把初始化成功当作 Encoder 学到 MI。

同架构 CCC-only 对照：`config/gnn_ccc_only.json` 和 `config/tumor_edge_simulation_ccc_only.json`。在相应 Notebook 首个配置单元格切换配置文件，然后 Restart Kernel / Run All；各实验输出目录独立。
