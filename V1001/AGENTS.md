# V1001 独立版本规则

- 唯一项目根目录为 `/home/xueshuailin/CCC_Phe/V1001`；不得修改 V999、V1000 或其他外部旧版本。
- 用户已授权检查、复制适用的外部模块；复制后必须适配本版本并记录来源。运行时禁止导入旧项目或读取其配置、数据、结果。
- 本版本的源代码、依赖声明、配置、数据、日志、缓存和输出独立管理。可使用系统已安装的 Python 与第三方库，不得修改共享环境。
- 不通过外部链接、路径回退或 PYTHONPATH 引入旧版本。
- 初始代码基线从 V1000 复制；V1000 保持只读，后续实现以用户针对 V1001 的最新 prompt 为准。
- V1001 只将 spatial CCC 改为 anchor-centered neighborhood 内所有 ordered non-self cell pairs 的聚合；线性 ST-first、bulk projection 和 Cox 保持不变。
- 禁止 deep learning、GNN、iterative phenotype refinement、新 loss、新 survival generator 和基于 truth/test 的调参。
- 不写代码注释、冗余 docstring、占位代码。WS/H 用 softplus，WB 由可微投影梯度推断，gamma 无符号约束。
- 继承的 inferred-WB、single-bank、anti-collapse 与 dual-bank 模块仅作为可复用基线；不得因其存在而默认选用其中任一方案。
- 共享 W 下使用共同 niche 尺度保持所有重建与风险不变；禁止将三个字典独立归一化后声称重建不变。
- TLS 标签、区域、基因列表与下游 niche 结果不得作为外部监督泄漏到训练或选参。
