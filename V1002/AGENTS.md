# V1002 simulation benchmark rules

- 唯一项目根目录为 `/home/xueshuailin/CCC_Phe/V1002`；不得修改 V999、V1000、V1001 或其他旧版本。
- 主模型固定为 flat matrix factorization：`CS≈WS HC`、`IS≈WS HI`，K_total 固定为 6（Background + 5 niches），ST discovery phenotype-free。
- simulation 必须从 cells、coordinates、cell types 和 LR-related expression 生成，再通过 neighborhood pipeline 计算 CS/IS。
- CCC mask 只能使用 assay measurability、sender/receiver-specific 10% expression coverage 和 spatial pair opportunity。
- ST 与 bulk 必须使用相同 `(sender, receiver, LR)` feature identity；bulk 不得使用坐标、spatial opportunity 或 abundance multiplier。
- 固定 robustness grid 为 purity 3 levels × niche size 3 levels × noise 3 levels × 3 seeds。
- 本轮 v2 primary 对照固定为 C-only、I-only、C+I。
- 禁止 tensor、CP、deep learning、Top-N、phenotype-guided filtering、K sweep、truth/test 调参及直接模拟最终 CS/IS。
- 论文图只使用 Python/matplotlib，导出 vector PDF 和 300 dpi PNG，不隐藏失败结果。

- `outputs/v1002_repair/` 保留上一轮 CCC-anchored staged 历史诊断；本轮 v2 主方案为固定规则的 RMS + relative-loss + 初始化梯度平衡 joint factorization。
- 本轮 v2 在独立模块中定义 N1/N2 的 CCC twin 和 N3/N4 的 composition twin，并采用 Background-reference ALR survival truth/Cox；旧 simulation_cells.py 与旧输出保留。
- v2 primary 三 seed 未达到预先固定的 ST 门槛时，记录失败并停止 bulk/Cox/robustness；不得根据结果调 truth、beta、噪声或训练权重。
- 最新 identifiability 轮次先做独立 tissue held-out 的 CS/IS oracle；只有 oracle 通过才冻结观测数据并运行 C-only、I-only、C+I。`pipeline.ipynb` 的 core-neighborhood oracle 是单独诊断口径，不能与脚本的 all-anchor oracle 混报。
