# 新增／修改文件与职责

| 文件 | 状态 | 职责 |
|---|---|---|
| [AGENTS.md](/home/xueshuailin/CCC_Phe/V999/AGENTS.md) | 修改 | 更新 V999 独立边界、复用授权与本次明确采用的新数学主线。 |
| [README.md](/home/xueshuailin/CCC_Phe/V999/README.md) | 修改 | 说明运行方式、数据契约、模型 API 和数学及评估边界。 |
| [PROVENANCE.md](/home/xueshuailin/CCC_Phe/V999/PROVENANCE.md) | 新增 | 记录旧 Cox 代码的来源、哈希、适配内容及解释器使用方式。 |
| [pyproject.toml](/home/xueshuailin/CCC_Phe/V999/pyproject.toml) | 新增 | 声明独立 Python 包、第三方依赖和 pytest 配置。 |
| [.gitignore](/home/xueshuailin/CCC_Phe/V999/.gitignore) | 新增 | 排除缓存、本地环境和运行产物。 |
| [src/phenoniche/data/schema.py](/home/xueshuailin/CCC_Phe/V999/src/phenoniche/data/schema.py) | 新增 | 校验 bulk、空间、生存和可选图输入并提供设备转换。 |
| [src/phenoniche/data/scaling.py](/home/xueshuailin/CCC_Phe/V999/src/phenoniche/data/scaling.py) | 新增 | 显式拟合共享特征兼容的数据块缩放系数并支持逆变换。 |
| [src/phenoniche/data/synthetic.py](/home/xueshuailin/CCC_Phe/V999/src/phenoniche/data/synthetic.py) | 新增 | 生成有风险、保护与中性生态位的非负模拟数据及完整真值。 |
| [src/phenoniche/model/niche_factorization.py](/home/xueshuailin/CCC_Phe/V999/src/phenoniche/model/niche_factorization.py) | 新增 | 实现共享 HC/HI、空间 HO 和 softplus 非负 WB/WS 模型。 |
| [src/phenoniche/model/phenotype_head.py](/home/xueshuailin/CCC_Phe/V999/src/phenoniche/model/phenotype_head.py) | 新增 | 用无符号限制的 gamma 计算患者风险分数。 |
| [src/phenoniche/model/normalization.py](/home/xueshuailin/CCC_Phe/V999/src/phenoniche/model/normalization.py) | 新增 | 以共同 niche 尺度归一化因素并保持五项重建及风险不变。 |
| [src/phenoniche/losses/reconstruction.py](/home/xueshuailin/CCC_Phe/V999/src/phenoniche/losses/reconstruction.py) | 新增 | 计算带显式形状校验的平方 Frobenius 重建损失。 |
| [src/phenoniche/losses/survival.py](/home/xueshuailin/CCC_Phe/V999/src/phenoniche/losses/survival.py) | 新增 | 实现数值稳定、Breslow 并列处理和事件数平均的 Cox 损失。 |
| [src/phenoniche/losses/spatial.py](/home/xueshuailin/CCC_Phe/V999/src/phenoniche/losses/spatial.py) | 新增 | 校验组合 Laplacian 并计算 dense/sparse 空间平滑能量。 |
| [src/phenoniche/training/config.py](/home/xueshuailin/CCC_Phe/V999/src/phenoniche/training/config.py) | 新增 | 校验模型训练参数和各损失权重并提供显式每元素权重。 |
| [src/phenoniche/training/trainer.py](/home/xueshuailin/CCC_Phe/V999/src/phenoniche/training/trainer.py) | 新增 | 执行无监督预热和联合训练、梯度裁剪、阶段内早停及日志记录。 |
| [src/phenoniche/evaluation/metrics.py](/home/xueshuailin/CCC_Phe/V999/src/phenoniche/evaluation/metrics.py) | 新增 | 计算余弦相似度、Pearson 相关和考虑删失的 C-index。 |
| [src/phenoniche/evaluation/matching.py](/home/xueshuailin/CCC_Phe/V999/src/phenoniche/evaluation/matching.py) | 新增 | 按 HC/HI 拼接字典做 Hungarian 匹配并报告因素恢复指标。 |
| [src/phenoniche/utils/seed.py](/home/xueshuailin/CCC_Phe/V999/src/phenoniche/utils/seed.py) | 新增 | 设置 Python、NumPy 和 PyTorch 的随机种子。 |
| [src/phenoniche/utils/validation.py](/home/xueshuailin/CCC_Phe/V999/src/phenoniche/utils/validation.py) | 新增 | 提供矩阵、设备类型和生存向量的公共入口校验。 |
| [scripts/run_synthetic.py](/home/xueshuailin/CCC_Phe/V999/scripts/run_synthetic.py) | 新增 | 运行同种子有无监督对照并保存指标、因素、模型状态和真值。 |
| [tests/conftest.py](/home/xueshuailin/CCC_Phe/V999/tests/conftest.py) | 新增 | 提供测试数据、单线程设置及完整恢复测试的共享训练结果。 |
| [tests/test_shapes.py](/home/xueshuailin/CCC_Phe/V999/tests/test_shapes.py) | 新增 | 检查形状、非负性、forward 纯度、非法输入及 CUDA 训练。 |
| [tests/test_cox_loss.py](/home/xueshuailin/CCC_Phe/V999/tests/test_cox_loss.py) | 新增 | 核验手算 ties、显式风险集、数值稳定性和有限差分梯度。 |
| [tests/test_gradients.py](/home/xueshuailin/CCC_Phe/V999/tests/test_gradients.py) | 新增 | 核验直接及联合梯度、监督对空间投影的影响和无监督结局独立性。 |
| [tests/test_synthetic_recovery.py](/home/xueshuailin/CCC_Phe/V999/tests/test_synthetic_recovery.py) | 新增 | 核验生成真值、置换与尺度匹配、五项恢复及风险保护方向。 |
| [tests/test_normalization_scaling.py](/home/xueshuailin/CCC_Phe/V999/tests/test_normalization_scaling.py) | 新增 | 核验归一化守恒、缩放往返、共享单位与零块处理。 |
| [tests/test_spatial_training.py](/home/xueshuailin/CCC_Phe/V999/tests/test_spatial_training.py) | 新增 | 核验图能量及梯度、完整加权目标、早停、重现性和 C-index 方向。 |
| [src/phenoniche/__init__.py](/home/xueshuailin/CCC_Phe/V999/src/phenoniche/__init__.py) | 新增 | 声明 Python 包边界，不执行初始化副作用。 |
| [src/phenoniche/data/__init__.py](/home/xueshuailin/CCC_Phe/V999/src/phenoniche/data/__init__.py) | 新增 | 声明 Python 包边界，不执行初始化副作用。 |
| [src/phenoniche/evaluation/__init__.py](/home/xueshuailin/CCC_Phe/V999/src/phenoniche/evaluation/__init__.py) | 新增 | 声明 Python 包边界，不执行初始化副作用。 |
| [src/phenoniche/losses/__init__.py](/home/xueshuailin/CCC_Phe/V999/src/phenoniche/losses/__init__.py) | 新增 | 声明 Python 包边界，不执行初始化副作用。 |
| [src/phenoniche/model/__init__.py](/home/xueshuailin/CCC_Phe/V999/src/phenoniche/model/__init__.py) | 新增 | 声明 Python 包边界，不执行初始化副作用。 |
| [src/phenoniche/training/__init__.py](/home/xueshuailin/CCC_Phe/V999/src/phenoniche/training/__init__.py) | 新增 | 声明 Python 包边界，不执行初始化副作用。 |
| [src/phenoniche/utils/__init__.py](/home/xueshuailin/CCC_Phe/V999/src/phenoniche/utils/__init__.py) | 新增 | 声明 Python 包边界，不执行初始化副作用。 |
| [FILES.md](/home/xueshuailin/CCC_Phe/V999/FILES.md) | 新增 | 提供本逐文件交付清单。 |

## 运行产物

- [outputs/synthetic/ground_truth.npz](/home/xueshuailin/CCC_Phe/V999/outputs/synthetic/ground_truth.npz)：保存未缩放的所有模拟真值因素。
- [outputs/synthetic/summary.json](/home/xueshuailin/CCC_Phe/V999/outputs/synthetic/summary.json)：保存配置、缩放、恢复指标、gamma、训练内 C-index 及两组差值。
- [outputs/synthetic/supervised_factors.npz](/home/xueshuailin/CCC_Phe/V999/outputs/synthetic/supervised_factors.npz)：保存共同尺度归一化后的模型因素。
- [outputs/synthetic/supervised_history.json](/home/xueshuailin/CCC_Phe/V999/outputs/synthetic/supervised_history.json)：保存该训练的逐 epoch 结构化日志。
- [outputs/synthetic/supervised_state.pt](/home/xueshuailin/CCC_Phe/V999/outputs/synthetic/supervised_state.pt)：保存原始可加载模型 state_dict。
- [outputs/synthetic/unsupervised_factors.npz](/home/xueshuailin/CCC_Phe/V999/outputs/synthetic/unsupervised_factors.npz)：保存共同尺度归一化后的模型因素。
- [outputs/synthetic/unsupervised_history.json](/home/xueshuailin/CCC_Phe/V999/outputs/synthetic/unsupervised_history.json)：保存该训练的逐 epoch 结构化日志。
- [outputs/synthetic/unsupervised_state.pt](/home/xueshuailin/CCC_Phe/V999/outputs/synthetic/unsupervised_state.pt)：保存原始可加载模型 state_dict。
- [outputs/synthetic_cuda/ground_truth.npz](/home/xueshuailin/CCC_Phe/V999/outputs/synthetic_cuda/ground_truth.npz)：保存未缩放的所有模拟真值因素。
- [outputs/synthetic_cuda/summary.json](/home/xueshuailin/CCC_Phe/V999/outputs/synthetic_cuda/summary.json)：保存配置、缩放、恢复指标、gamma、训练内 C-index 及两组差值。
- [outputs/synthetic_cuda/supervised_factors.npz](/home/xueshuailin/CCC_Phe/V999/outputs/synthetic_cuda/supervised_factors.npz)：保存共同尺度归一化后的模型因素。
- [outputs/synthetic_cuda/supervised_history.json](/home/xueshuailin/CCC_Phe/V999/outputs/synthetic_cuda/supervised_history.json)：保存该训练的逐 epoch 结构化日志。
- [outputs/synthetic_cuda/supervised_state.pt](/home/xueshuailin/CCC_Phe/V999/outputs/synthetic_cuda/supervised_state.pt)：保存原始可加载模型 state_dict。
- [outputs/synthetic_cuda/unsupervised_factors.npz](/home/xueshuailin/CCC_Phe/V999/outputs/synthetic_cuda/unsupervised_factors.npz)：保存共同尺度归一化后的模型因素。
- [outputs/synthetic_cuda/unsupervised_history.json](/home/xueshuailin/CCC_Phe/V999/outputs/synthetic_cuda/unsupervised_history.json)：保存该训练的逐 epoch 结构化日志。
- [outputs/synthetic_cuda/unsupervised_state.pt](/home/xueshuailin/CCC_Phe/V999/outputs/synthetic_cuda/unsupervised_state.pt)：保存原始可加载模型 state_dict。
- [outputs/pytest.log](/home/xueshuailin/CCC_Phe/V999/outputs/pytest.log)：保存对应测试或示例运行的完整日志。
- [outputs/cpu_run.log](/home/xueshuailin/CCC_Phe/V999/outputs/cpu_run.log)：保存对应测试或示例运行的完整日志。
- [outputs/cuda_run.log](/home/xueshuailin/CCC_Phe/V999/outputs/cuda_run.log)：保存对应测试或示例运行的完整日志。

## 困难模拟与留出评估增量

本轮完整新增／修改文件及产物见 [CHALLENGING_FILES.md](/home/xueshuailin/CCC_Phe/V999/CHALLENGING_FILES.md)。

## Inferred-WB 与诊断增量

本轮文件与产物见 [INFERRED_FILES.md](/home/xueshuailin/CCC_Phe/V999/INFERRED_FILES.md)。

## Anti-collapse 单一增量

本轮文件与产物见 [ANTI_COLLAPSE_FILES.md](/home/xueshuailin/CCC_Phe/V999/ANTI_COLLAPSE_FILES.md)。

## Dual-bank residual 增量

本轮文件与产物见 [DUAL_BANK_FILES.md](/home/xueshuailin/CCC_Phe/V999/DUAL_BANK_FILES.md)。
