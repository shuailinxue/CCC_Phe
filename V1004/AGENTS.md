# V1004 active HBC1 workflow

- 只在 `/home/xueshuailin/CCC_Phe/V1004` 内工作，不修改 V1002 或其他历史版本。
- HBC1 的 C 与 pairwise directed CCC I 分别进行独立 NMF；K=8、seed=40700、neighborhood、sigma、CCC filtering 和 iterations 固定。
- 使用本地 `src/` 内的 pairwise CCC 和 canonicalized NMF；Notebook 不从原始 V1002 目录导入。
- 仅用 hard-assignment overlap 的 Hungarian permutation 对齐标签；不训练 shared W，不平均 W，不进行第二次聚类。
- 只为实际出现的 aligned `(C, I)` joint states 编号。匹配质量如实报告，不根据结果调参或合并小状态。
- 复制来的其他 Notebook、脚本和输出为历史材料；当前主流程为 `HBC1_V1004_niche_walkthrough.ipynb`。
