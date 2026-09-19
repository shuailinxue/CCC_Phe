# 本轮文件清单

| 新增文件 | 职责 |
|---|---|
| [src/phenoniche/inference/__init__.py](/home/xueshuailin/CCC_Phe/V999/src/phenoniche/inference/__init__.py) | 声明独立推断包。 |
| [src/phenoniche/inference/bulk.py](/home/xueshuailin/CCC_Phe/V999/src/phenoniche/inference/bulk.py) | 仅根据固定字典和 bulk 特征进行加权非负最小二乘推断。 |
| [src/phenoniche/data/challenging_synthetic.py](/home/xueshuailin/CCC_Phe/V999/src/phenoniche/data/challenging_synthetic.py) | 实现容量受限及同组成不同 CCC 两种固定真值模拟。 |
| [src/phenoniche/evaluation/patient_protocol.py](/home/xueshuailin/CCC_Phe/V999/src/phenoniche/evaluation/patient_protocol.py) | 实现独立患者划分、训练表型配对置换、冻结 Cox 头和验证集选参。 |
| [src/phenoniche/evaluation/phenotype_recovery.py](/home/xueshuailin/CCC_Phe/V999/src/phenoniche/evaluation/phenotype_recovery.py) | 计算 phenotype-specific 恢复、矩形匹配及最佳匹配冲突。 |
| [src/phenoniche/evaluation/challenging_benchmark.py](/home/xueshuailin/CCC_Phe/V999/src/phenoniche/evaluation/challenging_benchmark.py) | 执行无泄漏候选训练、封存选参、测试评分、五种子及置换实验。 |
| [scripts/run_challenging_synthetic.py](/home/xueshuailin/CCC_Phe/V999/scripts/run_challenging_synthetic.py) | 提供两个场景的完整可复现运行入口。 |
| [tests/test_heldout_inference.py](/home/xueshuailin/CCC_Phe/V999/tests/test_heldout_inference.py) | 检验推断最优性、标签独立性、缩放边界、冻结头和 CUDA 返回类型。 |
| [tests/test_challenging_synthetic.py](/home/xueshuailin/CCC_Phe/V999/tests/test_challenging_synthetic.py) | 检验风险生成、nuisance 独立性、容量限制、CCC 差异和真值可观测性。 |
| [tests/test_phenotype_permutation.py](/home/xueshuailin/CCC_Phe/V999/tests/test_phenotype_permutation.py) | 检验配对置换和完整评估产物及选择顺序。 |
| [CHALLENGING_BENCHMARK.md](/home/xueshuailin/CCC_Phe/V999/CHALLENGING_BENCHMARK.md) | 记录预设实验、指标、冻结头和测试隔离协议。 |
| [CHALLENGING_FILES.md](/home/xueshuailin/CCC_Phe/V999/CHALLENGING_FILES.md) | 列出本轮新增和修改文件。 |

| 修改文件 | 修改内容 |
|---|---|
| [README.md](/home/xueshuailin/CCC_Phe/V999/README.md) | 更新留出推断状态并增加困难实验入口。 |
| [FILES.md](/home/xueshuailin/CCC_Phe/V999/FILES.md) | 增加本轮交付清单链接。 |
| [PROVENANCE.md](/home/xueshuailin/CCC_Phe/V999/PROVENANCE.md) | 记录本轮无核心重构、原有源码校验与实验规模。 |

原有模型、trainer、loss、scaler、简单 synthetic、旧测试和旧示例均未改动，30 个原有 Python 文件哈希全部一致。

## 新增实验产物

- [outputs/challenging_synthetic/RESULTS.md](/home/xueshuailin/CCC_Phe/V999/outputs/challenging_synthetic/RESULTS.md)
- [outputs/challenging_synthetic/capacity/ground_truth.npz](/home/xueshuailin/CCC_Phe/V999/outputs/challenging_synthetic/capacity/ground_truth.npz)
- [outputs/challenging_synthetic/capacity/ground_truth_metadata.json](/home/xueshuailin/CCC_Phe/V999/outputs/challenging_synthetic/capacity/ground_truth_metadata.json)
- [outputs/challenging_synthetic/capacity/lambda_sweep.json](/home/xueshuailin/CCC_Phe/V999/outputs/challenging_synthetic/capacity/lambda_sweep.json)
- [outputs/challenging_synthetic/capacity/phenotype_permutation.json](/home/xueshuailin/CCC_Phe/V999/outputs/challenging_synthetic/capacity/phenotype_permutation.json)
- [outputs/challenging_synthetic/capacity/protocol.json](/home/xueshuailin/CCC_Phe/V999/outputs/challenging_synthetic/capacity/protocol.json)
- [outputs/challenging_synthetic/capacity/seed_stability.json](/home/xueshuailin/CCC_Phe/V999/outputs/challenging_synthetic/capacity/seed_stability.json)
- [outputs/challenging_synthetic/capacity/selection.json](/home/xueshuailin/CCC_Phe/V999/outputs/challenging_synthetic/capacity/selection.json)
- [outputs/challenging_synthetic/capacity/splits.json](/home/xueshuailin/CCC_Phe/V999/outputs/challenging_synthetic/capacity/splits.json)
- [outputs/challenging_synthetic/capacity/summary.json](/home/xueshuailin/CCC_Phe/V999/outputs/challenging_synthetic/capacity/summary.json)
- [outputs/challenging_synthetic/capacity/supervised_factors.npz](/home/xueshuailin/CCC_Phe/V999/outputs/challenging_synthetic/capacity/supervised_factors.npz)
- [outputs/challenging_synthetic/capacity/unsupervised_factors.npz](/home/xueshuailin/CCC_Phe/V999/outputs/challenging_synthetic/capacity/unsupervised_factors.npz)
- [outputs/challenging_synthetic/capacity_run.log](/home/xueshuailin/CCC_Phe/V999/outputs/challenging_synthetic/capacity_run.log)
- [outputs/challenging_synthetic/core_hashes_before.json](/home/xueshuailin/CCC_Phe/V999/outputs/challenging_synthetic/core_hashes_before.json)
- [outputs/challenging_synthetic/implementation_audit.json](/home/xueshuailin/CCC_Phe/V999/outputs/challenging_synthetic/implementation_audit.json)
- [outputs/challenging_synthetic/pytest.log](/home/xueshuailin/CCC_Phe/V999/outputs/challenging_synthetic/pytest.log)
- [outputs/challenging_synthetic/same_composition/ground_truth.npz](/home/xueshuailin/CCC_Phe/V999/outputs/challenging_synthetic/same_composition/ground_truth.npz)
- [outputs/challenging_synthetic/same_composition/ground_truth_metadata.json](/home/xueshuailin/CCC_Phe/V999/outputs/challenging_synthetic/same_composition/ground_truth_metadata.json)
- [outputs/challenging_synthetic/same_composition/lambda_sweep.json](/home/xueshuailin/CCC_Phe/V999/outputs/challenging_synthetic/same_composition/lambda_sweep.json)
- [outputs/challenging_synthetic/same_composition/phenotype_permutation.json](/home/xueshuailin/CCC_Phe/V999/outputs/challenging_synthetic/same_composition/phenotype_permutation.json)
- [outputs/challenging_synthetic/same_composition/protocol.json](/home/xueshuailin/CCC_Phe/V999/outputs/challenging_synthetic/same_composition/protocol.json)
- [outputs/challenging_synthetic/same_composition/seed_stability.json](/home/xueshuailin/CCC_Phe/V999/outputs/challenging_synthetic/same_composition/seed_stability.json)
- [outputs/challenging_synthetic/same_composition/selection.json](/home/xueshuailin/CCC_Phe/V999/outputs/challenging_synthetic/same_composition/selection.json)
- [outputs/challenging_synthetic/same_composition/splits.json](/home/xueshuailin/CCC_Phe/V999/outputs/challenging_synthetic/same_composition/splits.json)
- [outputs/challenging_synthetic/same_composition/summary.json](/home/xueshuailin/CCC_Phe/V999/outputs/challenging_synthetic/same_composition/summary.json)
- [outputs/challenging_synthetic/same_composition/supervised_factors.npz](/home/xueshuailin/CCC_Phe/V999/outputs/challenging_synthetic/same_composition/supervised_factors.npz)
- [outputs/challenging_synthetic/same_composition/unsupervised_factors.npz](/home/xueshuailin/CCC_Phe/V999/outputs/challenging_synthetic/same_composition/unsupervised_factors.npz)
- [outputs/challenging_synthetic/same_composition_run.log](/home/xueshuailin/CCC_Phe/V999/outputs/challenging_synthetic/same_composition_run.log)
