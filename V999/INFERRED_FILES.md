# 本轮新增/修改文件

| 状态 | 文件 | 职责 |
|---|---|---|
| 新增 | [scripts/run_inferred_wb.py](/home/xueshuailin/CCC_Phe/V999/scripts/run_inferred_wb.py) | 运行本轮 80 次固定配置实验。 |
| 新增 | [src/phenoniche/evaluation/inference_diagnostics.py](/home/xueshuailin/CCC_Phe/V999/src/phenoniche/evaluation/inference_diagnostics.py) | 扫描 inner 求解参数并以 NNLS 参考解审计数值误差。 |
| 新增 | [src/phenoniche/evaluation/inferred_benchmark.py](/home/xueshuailin/CCC_Phe/V999/src/phenoniche/evaluation/inferred_benchmark.py) | 组织 capacity、K 扫描、CCC-only 和旧结果对比。 |
| 新增 | [src/phenoniche/evaluation/pair_diagnostics.py](/home/xueshuailin/CCC_Phe/V999/src/phenoniche/evaluation/pair_diagnostics.py) | 计算 A/B 最佳匹配、冲突、恢复和真值活性相关性。 |
| 新增 | [src/phenoniche/inference/nnls_reference.py](/home/xueshuailin/CCC_Phe/V999/src/phenoniche/inference/nnls_reference.py) | 仅供数值诊断的 NNLS 参考求解器。 |
| 新增 | [tests/test_inferred_wb.py](/home/xueshuailin/CCC_Phe/V999/tests/test_inferred_wb.py) | 检验参数移除、直接 Cox 梯度、统一推断及诊断模式。 |
| 修改 | [scripts/run_challenging_synthetic.py](/home/xueshuailin/CCC_Phe/V999/scripts/run_challenging_synthetic.py) | 调整新版默认输出目录，保留旧实验结果。 |
| 修改 | [scripts/run_synthetic.py](/home/xueshuailin/CCC_Phe/V999/scripts/run_synthetic.py) | 适配 inferred-WB 说明与默认输出目录。 |
| 修改 | [src/phenoniche/evaluation/challenging_benchmark.py](/home/xueshuailin/CCC_Phe/V999/src/phenoniche/evaluation/challenging_benchmark.py) | 统一训练/留出推断配置并移除自由 WB 评分。 |
| 修改 | [src/phenoniche/inference/bulk.py](/home/xueshuailin/CCC_Phe/V999/src/phenoniche/inference/bulk.py) | 实现保持计算图的固定步数加速投影梯度推断。 |
| 修改 | [src/phenoniche/model/niche_factorization.py](/home/xueshuailin/CCC_Phe/V999/src/phenoniche/model/niche_factorization.py) | 删除 raw_WB，以观测数据和当前字典推断 bulk 活性。 |
| 修改 | [src/phenoniche/training/config.py](/home/xueshuailin/CCC_Phe/V999/src/phenoniche/training/config.py) | 增加 inner_steps 与 inner_lr 的显式配置和校验。 |
| 修改 | [src/phenoniche/training/trainer.py](/home/xueshuailin/CCC_Phe/V999/src/phenoniche/training/trainer.py) | 将全部 bulk 损失接到推断 WB，并在 CCC-only 排除未使用字典的原有 L2。 |
| 修改 | [tests/test_gradients.py](/home/xueshuailin/CCC_Phe/V999/tests/test_gradients.py) | 将旧间接梯度断言更新为直接 Cox 字典梯度断言。 |
| 修改 | [tests/test_heldout_inference.py](/home/xueshuailin/CCC_Phe/V999/tests/test_heldout_inference.py) | 更新可微推断的计算图预期，保留无泄漏和最优性检查。 |
| 修改 | [tests/test_normalization_scaling.py](/home/xueshuailin/CCC_Phe/V999/tests/test_normalization_scaling.py) | 为已无自由 WB 的模型提供显式观测输入。 |
| 修改 | [tests/test_shapes.py](/home/xueshuailin/CCC_Phe/V999/tests/test_shapes.py) | 校验推断 WB 的形状、非负性和 forward 纯度。 |
| 修改 | [tests/test_spatial_training.py](/home/xueshuailin/CCC_Phe/V999/tests/test_spatial_training.py) | 让数学目标测试使用显式 bulk 输入。 |
| 新增 | [INFERRED_WB.md](/home/xueshuailin/CCC_Phe/V999/INFERRED_WB.md) | 记录统一推断、数值误差、固定实验和诊断边界。 |
| 新增 | [INFERRED_FILES.md](/home/xueshuailin/CCC_Phe/V999/INFERRED_FILES.md) | 提供本轮文件清单。 |
| 修改 | [AGENTS.md](/home/xueshuailin/CCC_Phe/V999/AGENTS.md) | 更新为用户本轮明确要求的 inferred-WB 主线。 |
| 修改 | [README.md](/home/xueshuailin/CCC_Phe/V999/README.md) | 更新现行 API、模型参数与运行入口。 |
| 修改 | [CHALLENGING_BENCHMARK.md](/home/xueshuailin/CCC_Phe/V999/CHALLENGING_BENCHMARK.md) | 将旧 free-WB 实验协议标记为历史记录。 |
| 修改 | [FILES.md](/home/xueshuailin/CCC_Phe/V999/FILES.md) | 增加本轮清单链接。 |
| 修改 | [PROVENANCE.md](/home/xueshuailin/CCC_Phe/V999/PROVENANCE.md) | 记录本轮来源、旧结果保留与数据冻结审计。 |

## 新增产物

- [outputs/inferred_wb/RESULTS.md](/home/xueshuailin/CCC_Phe/V999/outputs/inferred_wb/RESULTS.md)
- [outputs/inferred_wb/capacity_lambda_sweep.json](/home/xueshuailin/CCC_Phe/V999/outputs/inferred_wb/capacity_lambda_sweep.json)
- [outputs/inferred_wb/capacity_selection.json](/home/xueshuailin/CCC_Phe/V999/outputs/inferred_wb/capacity_selection.json)
- [outputs/inferred_wb/capacity_summary.json](/home/xueshuailin/CCC_Phe/V999/outputs/inferred_wb/capacity_summary.json)
- [outputs/inferred_wb/example_smoke.log](/home/xueshuailin/CCC_Phe/V999/outputs/inferred_wb/example_smoke.log)
- [outputs/inferred_wb/frozen_data_audit.json](/home/xueshuailin/CCC_Phe/V999/outputs/inferred_wb/frozen_data_audit.json)
- [outputs/inferred_wb/gradient_audit.json](/home/xueshuailin/CCC_Phe/V999/outputs/inferred_wb/gradient_audit.json)
- [outputs/inferred_wb/implementation_audit.json](/home/xueshuailin/CCC_Phe/V999/outputs/inferred_wb/implementation_audit.json)
- [outputs/inferred_wb/inner_inference_diagnostics.json](/home/xueshuailin/CCC_Phe/V999/outputs/inferred_wb/inner_inference_diagnostics.json)
- [outputs/inferred_wb/protocol.json](/home/xueshuailin/CCC_Phe/V999/outputs/inferred_wb/protocol.json)
- [outputs/inferred_wb/pytest.log](/home/xueshuailin/CCC_Phe/V999/outputs/inferred_wb/pytest.log)
- [outputs/inferred_wb/run.log](/home/xueshuailin/CCC_Phe/V999/outputs/inferred_wb/run.log)
- [outputs/inferred_wb/same_composition_ccc_only.json](/home/xueshuailin/CCC_Phe/V999/outputs/inferred_wb/same_composition_ccc_only.json)
- [outputs/inferred_wb/same_composition_k_sweep.json](/home/xueshuailin/CCC_Phe/V999/outputs/inferred_wb/same_composition_k_sweep.json)
- [outputs/inferred_wb/same_composition_truth_diagnostics.json](/home/xueshuailin/CCC_Phe/V999/outputs/inferred_wb/same_composition_truth_diagnostics.json)
- [outputs/inferred_wb/source_hashes_before.json](/home/xueshuailin/CCC_Phe/V999/outputs/inferred_wb/source_hashes_before.json)
