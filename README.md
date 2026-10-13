# CCC_Phe

当前项目以 Xenium Prime 5K Breast 的 directed CCC 输入、已训练的双解码器模型及其下游 niche 结果为唯一工作版本。

## 使用

在项目根目录打开 `XeniumPrime5K_Breast_niche_walkthrough.ipynb`。Notebook 直接读取 NAS 上的现有结果，不会因文件缺失自动启动训练。图片导出到 `/home/nas3/biod/xueshuailin/CCC_Phe/workspace/outputs/xenium_prime_5k/figures/`。

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

本次整理保留已有下游 niche 工作内容，未重新运行真实数据分析。项目的 phenotype-associated CCC 主线仍遵循 `AGENTS.md`：

`current-study ST anchor -> BayesPrism bulk -> patient exact directed CCC -> graph-smoothed sparse Cox`

下游 niche 结果不能进入 ST-anchor 构建、Cox screening 或超参数选择。
