# Structured ST-first 结果

## Simulation specification

| P | N | Grid | C | Q | F | E | K | Active CCC edges/niche |
|---:|---:|---|---:|---:|---:|---:|---:|---:|
| 320 | 1200 | 30×40 | 8 | 12 | 768 | 24 | 6 | 48 |

A/B truth：HC cosine=1.0000，HI cosine=1.68×10⁻⁸，WB correlation=0.0329，WS correlation=0.1084。真实 survival C-index=0.6829，event fraction=0.5375。CCC 由 `[6,8,8,12]` tensor 构建，包含 niche-specific、shared 和弱 background edges；模型输入经过固定 tau=0.0001 的 opportunity normalization，未使用 survival 信息。

## ST-only recovery

| Niche | HC | HI | HO | WS | Top-50 precision | Top-50 recall |
|---|---:|---:|---:|---:|---:|---:|
| A risk | 0.8652 | 0.7244 | 0.7486 | 0.8653 | 0.660 | 0.688 |
| B neutral | 0.9980 | 0.9366 | 0.9750 | 0.9957 | 0.960 | 1.000 |
| C protective | 0.9739 | 0.8903 | 0.9306 | 0.9981 | 0.920 | 0.958 |
| D neutral | 0.9409 | 0.8468 | 0.8953 | 0.4869 | 0.720 | 0.750 |
| E nuisance | 0.9990 | 0.9825 | 0.9993 | 0.9954 | 0.960 | 1.000 |
| F spatial | 0.9988 | 0.9808 | 0.9994 | 0.9935 | 0.960 | 1.000 |

HC-only 对 A/B 发生 collision，这是 `HC_A=HC_B` 的预期结果；HI-only 和 HC+HI 都没有 collision。5-seed pairwise mean stability 为 HC=0.9838、HI=0.9462、HO=0.9349。Primary discovery gate 通过。

## Frozen vs Iterative

| Method | Validation C | Test C | Risk WB | Protective WB | A gamma | B gamma | C gamma | HC anchor | HI anchor | WS anchor |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ST-Frozen | 0.6083 | 0.6884 | 0.9275 | 0.9944 | 1.9995 | 0.0018 | -1.2608 | 1.0000 | 1.0000 | 1.0000 |
| ST-Iterative | 0.6083 | 0.6884 | 0.9275 | 0.9944 | 1.9995 | 0.0018 | -1.2608 | 1.0000 | 1.0000 | 1.0000 |

Frozen 正确恢复 A risk 和 C protective 的 gamma 符号，B gamma 接近0。Iterative 与 Frozen 完全相同，因为全部真实表型 candidate 都未通过 preservation gate，最终按协议回滚。

## Five outer iterations

| Iteration | Accepted | Validation C | HC anchor | HI anchor | ST reconstruction change | Max abs delta |
|---:|---|---:|---:|---:|---:|---:|
| 1 | No | 0.6043 | 0.999995 | 0.999673 | +30.86% | 0.1308 |
| 2 | No | 0.6043 | 0.999995 | 0.999673 | +30.86% | 0.1308 |
| 3 | No | 0.6043 | 0.999995 | 0.999673 | +30.86% | 0.1308 |
| 4 | No | 0.6043 | 0.999995 | 0.999673 | +30.86% | 0.1308 |
| 5 | No | 0.6043 | 0.999995 | 0.999673 | +30.86% | 0.1308 |

Cosine gate 本身通过，但 ST reconstruction degradation 超过固定5%上限，因此每轮均正确 rollback。最大 delta 低于 `log(1.25)=0.2231`。无 gradient clipping 或非有限值。

## CCC dimension sensitivity

| F | ST A HI | ST B HI | A/B collision | A WS | B WS | Frozen risk WB | Frozen Test C | Iterative risk WB | Iterative Test C | HC anchor | HI anchor | WS anchor |
|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 128 | 0.7800 | 0.9750 | No | 0.9280 | 0.9964 | 0.9476 | 0.6861 | 0.9476 | 0.6861 | 1.0000 | 1.0000 | 1.0000 |
| 256 | 0.7588 | 0.9795 | No | 0.8341 | 0.9977 | 0.9360 | 0.7013 | 0.9360 | 0.7013 | 1.0000 | 1.0000 | 1.0000 |
| 512 | 0.7632 | 0.9356 | No | 0.9079 | 0.9973 | 0.9418 | 0.6698 | 0.9418 | 0.6698 | 1.0000 | 1.0000 | 1.0000 |
| 768 | 0.7244 | 0.9366 | No | 0.8653 | 0.9957 | 0.9275 | 0.6884 | 0.9275 | 0.6884 | 1.0000 | 1.0000 | 1.0000 |

F 增大后 A HI 有缓慢下降，但所有维度都维持 A/B 分离、较高 WB recovery 和相近的 held-out performance。Iterative 没有在任何维度稳定优于 Frozen。

## Phenotype permutation

| Seed | Iterative Test C | A gamma | B gamma | C gamma | Accepted iterations |
|---:|---:|---:|---:|---:|---:|
| 101 | 0.5718 | 1.952 | 3.511 | 1.575 | 5 |
| 102 | 0.6651 | -0.587 | -2.020 | -2.301 | 5 |
| 103 | 0.5368 | -4.121 | -3.785 | -4.125 | 5 |
| 104 | 0.3617 | -0.069 | -0.263 | 0.343 | 5 |
| 105 | 0.4481 | -0.350 | 0.485 | 0.312 | 5 |
| Mean ± SD | 0.5167 ± 0.1164 | inconsistent | inconsistent | inconsistent | — |

五次置换使用完全相同的 ST anchor。Test C-index 回到随机附近且波动很大，A/C gamma 方向不稳定，没有稳定的 phenotype improvement，未发现 test leakage。

## Numerical audit

inner steps 10/25/50/100/200 相对 NNLS objective excess 分别为0.5226、0.01464、6.74×10⁻⁵、6.08×10⁻⁶和0。按预注册规则选择50步。HC/HI gradient norm 为0.06277/0.01652，全部有限。

## Five factual answers

1. ST-only 能在高维结构化 CCC 下可靠区分 same-composition A/B；HI-only 与 HC+HI 均无 collision。
2. 固定 ST dictionary 后能在 bulk cohort 恢复 phenotype-associated activity；risk/protective WB recovery 为0.9275/0.9944。
3. Iterative phenotype calibration 不优于 Frozen；真实表型下没有 candidate 通过 preservation gate。
4. 没有获得可接受的性能提升。最终 identity 完全保持是 rollback 的结果。
5. F=128–768 的 ST discovery 与 bulk projection 总体稳定，A HI 随维度增大轻度下降但未发生 collision。

失败层：**D. iterative refinement**。
