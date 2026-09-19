# CCC–Phenotype V0

本项目只保留一条主线：

`当前肝癌 ST（RCTD + COMMOT） → TCGA-LIHC BayesPrism → 患者级有向 exact CCC → CCC 图正则稀疏 Cox`

六张肝癌 ST 本身就是本次分析唯一的空间锚点；不存在“参考 ST”和“验证 ST”的区分。

患者级 CCC 定义为：

`C[s,a,b,p] = W_ST[a,b,p] * (M_RNA[s,a,b,p] + eps) / (M_ST[a,b,p] + eps)`

随后将 `(sender, receiver, LR)` 直接展开为 Cox 特征。当前发现步骤不进行 CP/NMF、communication programs、空间表型区域映射、H&E 转移或 Xenium 验证。

入口：`scripts/run_network_cox.py`。

逐格运行的完整流程见：`notebooks/V0_LIHC_phenotype_CCC.ipynb`。GSE189903 肝癌单细胞参考以及现有 LIHC/ST 路径均已预填。
