# Structured ST-first experiment

本实验在 V1000 内独立新增，不修改 inherited single-bank、anti-collapse、dual-bank 或旧 synthetic generator。流程为 ST-only shared nonnegative factorization、五种子 consensus anchor、固定字典 bulk projection、training-only Cox，以及带硬边界和 preservation gate 的五轮 phenotype adapter。

Primary 使用 P=320、30×40 spatial grid、C=8、Q=12、F=768、E=24、K=6。CCC feature 保留 sender、receiver、program 映射，模型输入经过 composition-opportunity normalization。所有 reconstruction objective 使用 per-element mean；bulk inference 使用等价的 `1/C` 和 `1/F` block weights。

完整协议保存在 [protocol.json](/home/xueshuailin/CCC_Phe/V1000/outputs/structured_st_first/protocol.json)，结果见 [RESULTS.md](/home/xueshuailin/CCC_Phe/V1000/outputs/structured_st_first/RESULTS.md)。
