# 跨视图 anti-collapse 单一增量实验

## 唯一新增目标

对每个字典 HC/HO/HI 的行，使用 `H[k] / (norm(H[k]) + eps)`，eps=1e-8；对所有 k<l 计算各视图 cosine 的乘积，并取 pair mean。训练目标只新增 `LossWeights.lambda_collapse * L_collapse`，默认权重为零。

没有单视图 orthogonality，没有 threshold、entropy、sparsity、JSD 或其他新增机制。三视图任意一个能区分 pair 时，乘积可降低；HC 相同而 HI 或 HO 正交时，惩罚为零。加性 eps 使极小范数附近不具备精确尺度不变性，正常非零范数范围在数值容差内保持尺度不变。

K=1 时没有不同 niche pair，返回可微的零。数学函数只计算相似度与 loss，记录、选参和结果判断全部位于 evaluation。模型、inferred-WB、Cox、重建与空间损失实现不变。新损失在 warmup 与 joint 两个阶段使用同一个固定权重；lambda_ph 的原预热方案不变。

## 零权重退化与稳定性

在修改前，已从上一轮 inferred-WB 实现捕获一个小型完整训练结果，保存在 tests/fixtures/inferred_wb_zero_collapse.json。回归测试比较修改后零权重训练的所有因素与原有损失。另对两个正式场景各五个零权重结果，逐 seed 比较上一轮同 K、同 lambda_ph 的推断训练、验证、测试评分、字典恢复与重建。

训练器仍使用原 Adam、学习率、L2、梯度裁剪和阶段预算。新增记录裁剪前梯度范数、是否触发裁剪、总裁剪次数及最终各损失；不依据结果调整优化器。每个成功运行保存 final total/bulk/spatial/Cox/collapse/L2 数值。非有限目标或梯度继续明确抛错；数值失败单独写入 numerical_failures.json，不能作为成功运行计入均值，候选规则排除不完整的五种子组合。

## 固定实验方案

运行 `python scripts/run_anti_collapse.py`，默认 CPU、四个进程、各自单线程。输出目录为 outputs/anti_collapse；切换设备请用新的输出目录。所有推断参数保持 100 steps、相对 lr=1.0。

- same_composition：K=6，lambda_ph=0.1。
- capacity：K=4，lambda_ph=0.1。
- 两个场景统一 lambda_collapse={0,0.0001,0.001,0.005,0.01,0.05,0.1}，模型 seeds=31–35。
- 每次仍为 800 epoch warmup + 800 epoch joint，其他参数及缩放均不变。
- same_composition 另做五次 time/event 联合置换，置换 seeds=101–105，模型 seed=31，使用预先规定 validation rule 选出的 collapse 权重，lambda_ph 仍为 0.1。

生成器、噪声、effect size、nuisance 强度、生存生成以及 train/validation/test split 均保持不变；启动时核对旧真值和 split。总计 35+35+5=75 次正式拟合。五模型种子的标准差仅反映固定队列上的初始化波动，不是独立数据队列的置信区间。

## 候选选择与测试隔离

每个场景所有模型拟合后，先按 λ 汇总五种子的 validation C-index。只保留不低于最佳均值减 0.02 的候选，从中选择训练字典 mean_pair_collapse 最低者。完全相同的分数用较小 λ 确定性打破平局。零权重可以被选中。

选择函数只接收 validation 分数与训练字典冗余度，不接收 truth 或 test。选择 JSON 写入之后才做 test 评分和真值恢复。完整七个 λ 的结果均保留，不因最终测试或真值恢复结果替换候选。

## 诊断解释

A/B 主匹配沿用 HC+HI，另列 HI-only collision。为每个匹配 niche 报告 HC/HO/HI cosine、WB/WS correlation、gamma 和总体字典恢复。每次运行保存全部 learned niche pairs 的 sC、sO、sI、乘积及各视图 mean/median/min/max；还保存匹配 A/B 的三视图相似度和联合 score。

若 A/B 匹配到同一个 factor，其匹配相似度接近 1；但 L_collapse 只对两个不同 learned factor 计算，不能把真值匹配冲突与直接受罚的 learned pair 混为一谈。

Collision=0 不单独判定成功。必须同时查看 learned A/B HI 相似度、A/B HI 与 WS 恢复、留出预测和重建。检查所有视图的 pair 分布，不能把无恢复收益的机械分离称为有效。由于不同视图允许替代区分，单看 HI pair mean 也不足以判断损失是否正确或是否发生人为正交化。

输出 same_composition_sweep.json、capacity_sweep.json、same_composition_permutation.json、pairwise_similarity_diagnostics.json 和 summary.json；不保存大量模型或中间 tensor。旧结果保持原样。
