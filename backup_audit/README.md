# CCC_Phe 可用分析数据容量盘点（2026-09-16）

按用户最新要求：只保留可继续分析的现成数据及结果，不统计整个外部数据集，不备份原始下载包、原始测序文件、STAR counts 或 MERFISH 原始转录本文件。

状态：10.254.29.187:2049 仍连接超时。NAS 总容量未知，未执行分析或复制 NAS 数据。

nas3_backup_roots.tsv 是本地代码支持的候选范围：
- CCC_Phe/data/processed/：已处理输入、BayesPrism、RCTD、ST anchor、细胞标签、CCC 张量等。
- CCC_Phe/results/：已有分析结果。
- TCGA_LIHC 的 clinical_aligned_plus.tsv 和 expr_aligned_plus.tsv 两个文件。
- GSE189903_scRNA/processed/GSE189903_liver_reference_raw_counts.h5ad：整理后的参考矩阵，名称中的 raw_counts 指原始计数值，并非原始测序文件。
- GSE327192_HCC_MERFISH/processed/：整理后的 panel、坐标和清单，待核实实际必要文件。

此清单没有重叠根目录，但还不是最小备份集合：NAS 恢复后须读取运行记录、产物清单，确认当前有效版本、排除废弃试跑及可重建缓存，审查目录外软链接。不能仅根据本地代码断言候选文件都实际用过。去除原始文件意味着这里只保留继续已有分析所需的候选产物，不保证可以从头重建所有步骤；依赖原始转录本的下游步骤能否继续，要看处理缓存是否完整。

本地代码、配置、notebook 和导出材料 /home/xueshuailin/CCC_Phe 也应保留，本次盘点前为约 55 MiB。

恢复连通后执行只读容量盘点：

```bash
python /home/xueshuailin/CCC_Phe/backup_audit/measure_nas3.py
```

输出 nas3_sizes.json，包括逻辑/已分配字节数、文件数、错误和软链接。硬链接跨目录去重，不跟随软链接；有读取错误时只报告下限。NFS hard 挂载中途断网可能使工作进程阻塞；脚本在等待上限后尝试终止。容量统计只代表候选清单；最终精简容量须待 NAS 恢复后进一步核实。
