# 困难模拟与患者留出评估（上一轮 free-WB 协议）

本文记录 `outputs/challenging_synthetic` 中保留的上一轮结果。当前代码已移除自由 WB，新的统一推断协议与诊断见 INFERRED_WB.md；不得将旧版训练机制说明当作当前实现。

## 固定实验方案

运行 `python scripts/run_challenging_synthetic.py`，或使用本机已有解释器 `/home/xueshuailin/miniconda3/envs/cccphe/bin/python`。默认独立运行 capacity 与 same_composition 两个场景，输出位于 `outputs/challenging_synthetic/<scenario>/`。`--device cuda` 可切换训练设备；使用不同设备时须指定新的输出目录。非负最小二乘在 CPU 上以 float64 求解，结果恢复到输入张量的设备及类型。

每个场景 720 个 bulk 患者、300 个独立空间 anchor，K_true=6、K_fit=4，拆分为 432/144/144。数据 seed=20260917，拆分 seed=90210。模型种子 31–35；每个种子完整扫描 λ={0,0.001,0.005,0.01,0.05,0.1}；每次相同初始化、800 epoch 预热和 800 epoch 联合训练，无早停。每个场景另做 5 次训练表型配对置换，种子 101–105，固定模型 seed=31 和已选 λ。总计 70 次拟合，不根据输出调整模拟生成机制。

所有数据配置与训练参数在拟合前写入 protocol.json。所有 λ 的模型拟合完成后，仅按五个种子的 validation C-index 均值选取最佳正 λ，平分取较小 λ。selection.json 写入后才计算测试评分。完整曲线包含所有候选的测试结果，但这些结果不能反馈到选择或训练。统计来自同一模拟队列的不同初始化，不代表五个独立队列，也不作为置信区间。

## 两个场景

capacity：niche 0 风险、1 保护，gamma 分别为 +1.8、−1.8；2/3 为高方差 nuisance；4/5 中性。患者及空间活性各列独立生成，nuisance 活性采用更高方差分布，三个字典的 nuisance 行幅度也更强。其余 gamma 严格为零。存活时间由固定基线风险 0.04 和真实 eta 生成，删失采用独立随机流。

same_composition：沿用同样的数据容量和噪声机制，令 HC[0]=HC[1]，保留互异、有方向的 HI[0]/HI[1]；只让 niche 0 影响生存，niche 1 中性。因此该场景没有 protective niche，结果以 null 表示而非伪造保护作用。

输入是非负线性混合特征，CB 未被强制归一化为行和 1。噪声为各数据块信号 RMS 的 2.5%，观测截断至非负。组成、CCC、拓扑的真实字典与活性均随 ground_truth.npz 保存。空间没有 phenotype label。未实现可选的 topology-only 场景。

## 无泄漏协议

训练 CohortData 仅含 train 患者及独立空间数据。BlockScaler 仅拟合这些数据，validation/test 使用固定尺度。推断函数只接收 CB、IB、HC、HI、重建权重和求解预算；通过加权非负最小二乘求解固定字典问题，不接收 time/event/gamma，不回传字典梯度，也不改变模型。各患者独立求解。

原有 train、模型、loss、scaler、简单 synthetic 和旧测试均保持不变。新评估流程通过现有接口组合，不使用真实 niche 标签参与模型训练、正则化、初始化、早停或选参。

无监督分解本身的 gamma 保持零。为提供有意义的预测对照，额外对其固定字典推断得到的训练活性拟合一个 L2 Cox 评估头，ridge 预设为 0.01，不改 WB/WS/HC/HI/HO。这与“先发现生态位、再拟合 Cox”基线对应，不能将该评估头误认为联合监督。监督模型直接使用联合训练得到的 gamma，不额外重拟合。两者同样为 K_fit 个风险系数，分解 epoch 和容量相同。结果同时保存原始 factorization_gamma 和 evaluation_gamma。

train_c_index 用固定字典重新推断训练活性后评分，和 validation/test 采用相同推断路径；train_fitted_c_index 额外报告训练自由 WB 的拟合表现。置换控制仅联合打乱 train 的 time/event 对，测试与验证仍使用原始结局；不为每次置换重新选 λ。

## 恢复与结果解释

按 HC/HI 拼接后余弦相似度为每个真实 niche 找最佳 learned niche，报告 HC/HI/HO cosine、训练与测试 WB correlation、WS correlation 及 gamma。允许多个真实 niche 找到同一最佳因子，但明确报告 best_match_collision；另做矩形 Hungarian 匹配，未分配真值按零计入 overall_dictionary_recovery。K_fit/K_true=4/6，因此该整体指标上限为 2/3，不能与前一轮等 K 的整体 cosine 直接比较。

summary.json 汇总五种子均值、样本标准差、符号一致率、配对真值相似度以及真实字典推断的 oracle 诊断。lambda_sweep.json 和 seed_stability.json 保留逐次记录。phenotype_permutation.json 保留五次置换和汇总。只保存首个预设模型种子的监督/无监督因素，避免根据测试分数选择最好看的模型。undefined correlation 明确以 null 保存并统计数量。

监督可能改善、保持或降低恢复和测试预测；置换也可能因有限样本而偏离 0.5。代码不通过断言强制监督获胜或强制置换获得随机分数，正式输出按实际结果报告。
