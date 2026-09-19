# Current mainline: V0

1. 当前运行以 GSE327192 的 8 张 HCC MERFISH 为唯一 ST anchor。
2. GEO 未提供细胞类型标签，因此仅用 MERFISH 500-gene panel 和既有 GSE189903
   肝癌 scRNA 参考进行 6 大类 outcome-blind 标签转移；不使用 TLS 信息。
3. 从 panel 完整覆盖的 CellChat LR、MERFISH 细胞类型表达和单细胞空间 kNN
   邻接汇总 `W_ST[sender,receiver,LR]` 与 `M_ST[sender,receiver,LR]`。
4. 对 TCGA-LIHC 使用既有 BayesPrism 结果计算 `M_RNA`。
5. 构造 `C = W_ST * (M_RNA + eps) / (M_ST + eps)`。
6. 将每条 `(sender → receiver, LR)` CCC 直接作为一个患者级预测变量。
7. 结局盲 QC 后拟合 CCC 图正则稀疏 Cox；正 beta 为风险方向，负 beta 为保护方向。
8. 固定已调好的 `lambda_fraction=0.075`、`graph_lambda=0.05`，采用 5-fold
   外部评估及 50 次分层半样本重拟合评估泛化和方向稳定性。

本阶段不做 CP/NMF、communication program、communication ecosystem 或 PR-high/low 空间区域比较。
TLS annotation 只能在未来无监督 niche 完成后用于最终验证。
