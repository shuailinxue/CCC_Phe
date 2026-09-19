# 复用与适配记录

- 复用来源：`/home/xueshuailin/CCC_Phe/src/cccphe/core.py` 的 `cox_breslow_loss`（原第 16–29 行）。
- 原文件 SHA-256：`1098e1836f2de658afc0c88ff51aa82f7404d6d145cebfa8e2ac6a4674f8aa75`。
- 目标：`src/phenoniche/losses/survival.py`。保留排序、logcumsumexp、并列时间风险集及事件数平均逻辑；添加输入校验，去除注释与旧项目依赖，明确事件类型转换。
- 已检查旧 `network_cox.py`：其稀疏回归/FISTA 与本版共享生态位模型不同，因此未搬入旧 trainer。
- 其他数学核心、模拟、评估、测试均在 V999 内新建；没有导入外部项目模块，没有复用旧分析配置或数据。
- 本轮实现依据用户 pasted-text.txt 中的明确请求；替代此前 V999 的旧版 exact CCC Cox 主线，外部 AGENTS.md 未修改。
- 测试使用已安装的 `/home/xueshuailin/miniconda3/envs/cccphe/bin/python` 及第三方库，不向该环境安装或改写文件；V999 自行声明包依赖，可另建环境安装。

## 困难模拟与留出评估增量

- 遵照本轮用户请求，仅新增推断、模拟、评估、测试与文档；没有改变现有模型数学关系。
- 原有 30 个 Python 文件 SHA-256 全部保持一致；记录位于 outputs/challenging_synthetic/implementation_audit.json。
- 所有实验生成器参数、种子、训练预算和候选 λ 在训练前固定，并保存 protocol.json。两场景各 30 次候选拟合、5 次配对置换，共 70 次。
- 无监督模型另外拟合训练集冻结字典 Cox 评估头，原分解 gamma 保持零；没有添加因子或修改字典来增强基线。
- NNLS 使用已有 SciPy；未新增依赖，未引入真实数据或外部旧版本运行时依赖。
- 完整结果包含监督未获益的同组成不同 CCC 场景，没有据此反向调整生成器。

## Inferred-WB 增量

- 用户本轮明确授权移除自由 WB，改用统一可微推断；模型、trainer、推断及对应测试据此更新。
- 旧 NNLS 提取为 inference/nnls_reference.py，仅供数值参考；训练和所有患者评分使用 inference/bulk.py 的同一展开投影梯度实现。
- 数据生成器和患者拆分实现哈希未变，旧真值与拆分逐项核对；旧 outputs/challenging_synthetic 结果未改写。
- capacity 30 次、same_composition 完整模型 30 次、CCC-only 20 次，共 80 次正式拟合；没有新增正则或真实数据模块。
- 原始扫描未全部满足预设数值阈值，保留 tolerance_met=false，最终模型求解误差另行完整审计。
- 本轮 smoke 示例只保留运行日志，删除临时重复 tensor 产物。

## Anti-collapse 单一增量

- 仅新增三视图 cosine 乘积的 pair mean 惩罚，零权重保持原目标；训练器新增梯度与最终损失记录。
- 复用 V999 既有 inferred-WB、模型、生成器、拆分及评估函数；没有新增外部代码依赖。既有 Python 文件仅三处变化，详见 outputs/anti_collapse/implementation_audit.json。
- 75 次固定方案拟合完成，82 项测试通过；十次正式零权重结果与旧结果的审计指标差异为0，小型修改前 fixture 的参数和旧损失精确相同。
- 两场景 validation rule 均选 λ=0.1；same_composition 真值恢复未改善，事实判断 C。完整结果见 outputs/anti_collapse/RESULTS.md。

## Dual-bank residual 增量

- 新增代码全部位于 V999；复用现有 inferred-WB background solver、Cox loss、生成器、split、scaling 和基础指标，没有运行时导入外部工程。
- 既有 Python 文件哈希全部保持上一轮值。新模型、推断、训练、matching、benchmark、入口和五类测试均以新文件加入。
- single-bank baseline 直接复用已审计的 lambda_ph=0.1、lambda_collapse=0 结果。总容量、模型 seeds、数据和 split 相同。
- 共得到30个 final model，五次置换精确复用同一 post-Stage-1 background state；全部正式模型 lambda_collapse=0。
- Primary Residual-Frozen 未达到 same-composition bank role、HI、WS 和 held-out C-index 成功标准；事实判断 E。完整结果见 outputs/dual_bank/RESULTS.md。
