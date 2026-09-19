# Phenoniche V1000

V1000 是 `/home/xueshuailin/CCC_Phe` 下的新独立版本。初始代码基线复制自 V999 的已验证 `src/`、`tests/` 和 `scripts/`，便于后续直接复用稳定模块。

V1000 不在运行时导入 V999，也不读取 V999 的配置、数据或结果。后续代码、测试、数据和输出全部保存在本目录。当前尚未依据下一轮 prompt 选择新的模型或实验主线。

当前新增实验为 [Structured ST-first](/home/xueshuailin/CCC_Phe/V1000/STRUCTURED_ST_FIRST.md)：使用结构化 sender×receiver×program CCC，先从 ST 独立发现 stable niche，再做固定 bulk projection 和受 preservation gate 约束的 phenotype calibration。

复制来源与边界见 [PROVENANCE.md](/home/xueshuailin/CCC_Phe/V1000/PROVENANCE.md)，初始文件清单见 [FILES.md](/home/xueshuailin/CCC_Phe/V1000/FILES.md)。
