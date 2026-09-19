# Phenoniche V1001

V1001 是 `/home/xueshuailin/CCC_Phe` 下的独立版本，以当前 V1000 structured ST-first 工程为唯一基线。

V1001 不新增模型。唯一核心变化是将 `IS[i,a,b,q]` 定义为 anchor i 的同一 local neighborhood 内所有 `a→b` ordered non-self cell pairs 的 aggregate communication activity，并用完全同步的 pair opportunity 校正。

Composition、CCC 和 topology 仍使用 V1000 线性共享 `WS` 分解。ST discovery 完全 phenotype-free，然后冻结 HC/HI，原样复用 50-step nonnegative bulk projection 和 Cox。

完整结果位于 [outputs/v1001](/home/xueshuailin/CCC_Phe/V1001/outputs/v1001)。复制来源与边界见 [PROVENANCE.md](/home/xueshuailin/CCC_Phe/V1001/PROVENANCE.md)，文件清单见 [FILES.md](/home/xueshuailin/CCC_Phe/V1001/FILES.md)。
