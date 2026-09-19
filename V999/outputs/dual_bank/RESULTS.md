# Dual-bank residual learning 结果

五个模型种子的结果报告为均值 ± 样本标准差。所有30个 final model 均使用 lambda_ph=0.1、lambda_collapse=0；生成器、survival、split 和总容量未变化。结论以 Residual-Frozen 为 primary。

## Same-composition 主比较

| 模型 | Test C-index | A best bank | A phenotype consistency | A HI | A WS | B best bank | B background consistency | B HI | B WS | A/B different-bank | Background Test C | Phenotype Test C |
|---|---:|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|
| Single-bank | 0.6744 ± 0.0090 | single | N/A | 0.4793 ± 0.0889 | 0.4926 ± 0.1315 | single | N/A | 0.5572 ± 0.0454 | 0.7163 ± 0.1111 | N/A | N/A | 0.6744 ± 0.0090 |
| Dual-JointScratch | 0.6621 ± 0.0184 | phenotype | 60% | 0.3891 ± 0.0449 | 0.3481 ± 0.1220 | phenotype | 40% | 0.4272 ± 0.0780 | 0.3309 ± 0.1505 | 0% | 0.6121 ± 0.0225 | 0.6621 ± 0.0184 |
| Dual-Residual-Frozen | 0.6479 ± 0.0054 | phenotype | 60% | 0.5007 ± 0.1665 | 0.2582 ± 0.0512 | background | 100% | 0.3560 ± 0.0124 | 0.4214 ± 0.0255 | 60% | 0.5944 ± 0.0053 | 0.6479 ± 0.0054 |
| Dual-Residual-JointFT | 0.6925 ± 0.0076 | background | 40% | 0.3789 ± 0.1054 | 0.3534 ± 0.1167 | background | 80% | 0.4024 ± 0.0901 | 0.3283 ± 0.1911 | 20% | 0.6111 ± 0.0247 | 0.6925 ± 0.0076 |

Residual-Frozen 将 B 放入 background bank 5/5，但 A 只进入 phenotype bank 3/5，A/B 分 bank也只有3/5。A HI 从 single-bank 的0.4793小幅变为0.5007，没有达到0.65；B HI 从0.5572降至0.3560。A WS 从0.4926降至0.2582，B WS 从0.7163降至0.4214。Test C-index 下降0.0265，超过预定0.02容忍范围。JointFT 提高预测但重新混合 bank role；JointScratch 没有分开 A/B role。

Residual-Frozen 的逐 seed assignment：

| Seed | A bank/factor | A margin | B bank/factor | B margin |
|---:|---|---:|---|---:|
| 31 | phenotype/1 | 0.002 | background/1 | -0.049 |
| 32 | phenotype/0 | 0.052 | background/3 | -0.063 |
| 33 | background/0 | -0.013 | background/0 | -0.022 |
| 34 | background/0 | -0.031 | background/0 | -0.145 |
| 35 | phenotype/1 | 0.088 | background/3 | -0.023 |

## Capacity 主比较

| 模型 | Test C-index | Risk phenotype consistency | Protective phenotype consistency | Risk HI | Protective HI | Risk WS | Protective WS | Nuisance background consistency |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Single-bank | 0.7270 ± 0.0039 | N/A | N/A | 0.4516 ± 0.1466 | 0.3880 ± 0.1241 | 0.3149 ± 0.1789 | 0.3768 ± 0.1222 | N/A |
| Dual-Residual-Frozen | 0.7308 ± 0.0037 | 100% | 100% | 0.3367 ± 0.0485 | 0.2553 ± 0.0498 | 0.3907 ± 0.0963 | 0.4466 ± 0.0352 | 100% |
| Dual-Residual-JointFT | 0.7378 ± 0.0024 | 100% | 100% | 0.3153 ± 0.0226 | 0.2184 ± 0.0245 | 0.3952 ± 0.1791 | 0.3773 ± 0.0528 | 100% |

Capacity 中 bank role 分离稳定且预测保持，但 risk/protective HI 都低于 single-bank。Residual-Frozen 的 risk/protective WB correlation 为0.8994±0.0195和0.8605±0.0160，说明预测活动可以分离；对应 CCC 字典真值恢复不足。risk/protective gamma sign consistency 均为100%。

## Phenotype permutation

下表 A HI 和 A WS 使用每次置换中“最相似 phenotype-bank factor”的恢复，而不是固定 background bank 的最佳总体匹配。

| Permutation seed | Test C-index | A phenotype-bank | A margin | A phenotype HI | A phenotype WS | A phenotype gamma | Gamma sign correct |
|---:|---:|---:|---:|---:|---:|---:|---|
| 101 | 0.5220 | background | -0.0875 | 0.3392 | 0.0989 | -0.0458 | false |
| 102 | 0.4687 | background | -0.0695 | 0.3648 | 0.1022 | -0.1241 | false |
| 103 | 0.4055 | background | -0.0096 | 0.4239 | 0.1239 | -0.3264 | false |
| 104 | 0.4251 | background | -0.0421 | 0.3784 | 0.1184 | -0.1634 | false |
| 105 | 0.5172 | background | -0.0916 | 0.3394 | 0.0987 | -0.1304 | false |
| Mean ± SD | 0.4677 ± 0.0526 | 0/5 phenotype | -0.0601 ± 0.0343 | 0.3691 ± 0.0350 | 0.1084 ± 0.0119 | — | 0% |

五次置换精确复用同一个 background warm-up state。Test C-index 回到随机附近，A phenotype-bank rate 从真实 phenotype 的60%降至0%，A margin 由0.0198降至-0.0601，phenotype-bank WS recovery 只有0.1084。真实 phenotype supervision 确实改变 bank assignment，但这种改变没有带来充分的 HI/WS 真值恢复。

## 重建与稳定性

Residual-Frozen 的 same-composition bulk/spatial MSE 为0.001983/0.002965，single-bank 为0.001426/0.002372；capacity 为0.004378/0.005639，single-bank 为0.002623/0.003588。每个 bulk 和 spatial block 都满足 background contribution + phenotype contribution = total reconstruction，最大绝对守恒误差为0，shape 正确且全部有限。

30/30 final models 无 NaN/Inf，未触发 gradient clipping。全量测试92项通过。所有 checkpoint 均包含 final state_dict、config、seed、stage history 和诊断；未保存中间 epoch tensor。

## 事实判断

**E. dual-bank 更差，应放弃当前 NMF formulation。**

最可能的一个核心 failure mode 是：无表型的 Stage 1 background warm-up 已吸收 shared-composition A/B 结构和 A 的部分信号；冻结后 Stage 2 只能在剩余残差中获得可预测的 WBp，却无法恢复正确的 CCC 字典与空间 WS。因而模型可以形成 phenotype-associated activity，但该 activity 不是纯净的真实空间生态位。
