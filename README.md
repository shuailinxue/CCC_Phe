# CCC_Phe

项目保留既有 CCC-vector AE，同时提供独立下游 microenvironment GNN；两者共享原始 directed CCC 定义，独立保存模型和结果，临床 CCC/Cox 主线不变。

## 使用

在项目根目录打开 `XeniumPrime5K_Breast_niche_walkthrough.ipynb`。Notebook 直接读取 NAS 上的现有结果，不会因文件缺失自动启动训练。图片导出到 `/home/nas3/biod/xueshuailin/CCC_Phe/workspace/outputs/xenium_prime_5k/figures/`。

新 GNN 的真实数据报告：[XeniumPrime5K_GNN_niche_analysis.ipynb](/home/xueshuailin/CCC_Phe/XeniumPrime5K_GNN_niche_analysis.ipynb)。已执行并读取单独保存的结果，打开时不会重新训练；图片和模型矩阵均在 NAS 输出目录。

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

`细胞表达 + 类型 + 空间关系 -> directed attention/message passing -> 微环境 Z >= 0 -> C_hat = Z @ H`。
每个微环境一个向量，默认 32 个程序。批内重复细胞只编码一次；只缓存 30 个邻域成员和权重，不存储全部子图。H 在原始 CCC 单位下逐行和为 1，因此 `sum(Z_i) = sum(C_hat_i)`，强度与组成同时保留。CCC 只作监督，绝不进入 GNN forward。

损失：`L = Huber(C_hat/s, C/s)[有效特征] + λ_bio(α L_up + β L_down) + λ_reg R`，可选 `+ λ_contrast L_contrast`。s、有效特征 mask、表达预处理、基线校正和生物学尺度全部仅由训练块拟合。R 包含温和 dictionary entropy、相似程序惩罚及很弱的最小使用率约束；辅助梯度与 CCC 主梯度冲突时投影，生物学权重逐步增加。

上游活动来自有符号 TF-target regulon，下游活动来自 ligand-target 模块；均排除 LR 基因和自身 regulator。以同一有向物理细胞对及空间权重聚合 sender/receiver 活动。固定先验矩阵 A 将 H 连接到调控模块，没有自由基因 decoder。训练集 Ridge 去除细胞类型、局部组成、中心及按细胞类型/物理配对权重汇总的参与细胞表达状态、library size 基线；固定的训练集校准将 `(C_hat - D B_C) @ A` 与观测增量活动比较。它是**增量关联约束，不证明因果**。默认把打分 target genes 从 encoder、其归一化分母及强度输入留出；四种消融应使用相同先验及留出基因集合。

空间块/样本划分后，剔除邻域跨 fold 的中心，训练/验证/测试不共享物理细胞。K=4–15 仅在训练集拟合、验证集选择，测试用于最终诊断。导出 niche CCC、程序贡献、调控来源、空间图及 composition-only 对照。原 AE 的随机重叠划分结果只能做描述性比较；公平比较需在相同空间 fold 中重新训练原 AE。

```bash
python scripts/run_gnn.py audit
python scripts/run_gnn.py train --mode both --output /home/nas3/biod/xueshuailin/CCC_Phe/workspace/outputs/gnn_both
python scripts/run_gnn.py infer --output /home/nas3/biod/xueshuailin/CCC_Phe/workspace/outputs/gnn_both
python scripts/run_gnn.py baseline --output /home/nas3/biod/xueshuailin/CCC_Phe/workspace/outputs/gnn_both
python scripts/run_gnn.py compare --output /home/nas3/biod/xueshuailin/CCC_Phe/workspace/outputs/gnn_both
```

`config/gnn.json` 中配置路径、训练及消融。`split_seed` 固定空间划分和预处理，`seed` 改变模型初始化，可在相同 fold 上比较稳定性。mode 为 `ccc_only / upstream / downstream / both`；启用生物学分支时可使用负对照 `--prior-control shuffle_links / shuffle_activity`，必须使用独立实验输出；最终生物学诊断仍对照真实标签及原始先验。`--contrastive` 默认关闭：同类型 receiver 重排空间配对；仅在 CCC 差异可检测且 opportunity 总量漂移 <10% 时施加约束。所有真实输出在 NAS `workspace/outputs/`；已有 checkpoint 禁止覆盖。

调控输入为 NAS `data/priors/upstream.tsv`（`tf,target,weight,source,evidence`）和 `downstream.tsv`（`ligand,target,weight,source,evidence`），tab 分隔，每条关系须有来源及证据。`manifest.json` 为 upstream/downstream 分别记录 `database,version,url,species,sha256`。建议核验 [CollecTRI](https://github.com/saezlab/CollecTRI) 的有符号 regulons 和 [NicheNet](https://github.com/saeyslab/nichenetr) 的 ligand-target prior；程序不自行猜测调控关系，也不把 TF mRNA 当 TF 活性。只有 ligand-target prior 时明确标注“受体特异性未证实”。覆盖、未覆盖基因/CCC、目标信息重叠及来源写入 NAS audit。

`audit_file` 指向实际完成的研究审计 JSON：`st_anchor,bayesprism,clinical_endpoint,feature_mask,sample_matching,cell_labels` 分别提供 `status=verified,evidence_file,sha256`。独立下游、无临床输入时，可以明确设置 `scope=independent_downstream_niche`，将不适用的 BayesPrism/clinical endpoint/sample matching 记录为 `status=not_applicable,reason`；不得自动伪造通过审计。缺失有效先验时只能明确使用 CCC-only；请求启用的调控损失会报错。Xenium Prime 5K 的真实审计由 `scripts/audit_prime5k_gnn.py` 生成，CollecTRI/NicheNet 先验由 `scripts/prepare_gnn_priors.py` 下载并校验来源。`scripts/report_prime5k_gnn.py` 读取完成的 GNN 结果并执行报告 Notebook；不会在 Notebook 中训练模型。
