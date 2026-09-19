# Dual-bank 新增与修改文件

## 新增代码

- [dual_bank.py](/home/xueshuailin/CCC_Phe/V999/src/phenoniche/inference/dual_bank.py)：统一 background、phenotype residual 和 dual-bank activity inference。
- [dual_bank_factorization.py](/home/xueshuailin/CCC_Phe/V999/src/phenoniche/model/dual_bank_factorization.py)：双 bank 非负模型与 phenotype-only Cox head。
- [dual_bank_trainer.py](/home/xueshuailin/CCC_Phe/V999/src/phenoniche/training/dual_bank_trainer.py)：Stage 1、Stage 2、JointFT 和 JointScratch 训练。
- [bank_matching.py](/home/xueshuailin/CCC_Phe/V999/src/phenoniche/evaluation/bank_matching.py)：bank-aware matching、margin、contamination 和分 bank 恢复诊断。
- [dual_bank_benchmark.py](/home/xueshuailin/CCC_Phe/V999/src/phenoniche/evaluation/dual_bank_benchmark.py)：固定实验、probe、置换、重建守恒、checkpoint 与汇总。
- [run_dual_bank.py](/home/xueshuailin/CCC_Phe/V999/scripts/run_dual_bank.py)：正式实验入口。

## 新增测试

- [test_dual_bank_shapes.py](/home/xueshuailin/CCC_Phe/V999/tests/test_dual_bank_shapes.py)
- [test_dual_bank_inference.py](/home/xueshuailin/CCC_Phe/V999/tests/test_dual_bank_inference.py)
- [test_dual_bank_gradients.py](/home/xueshuailin/CCC_Phe/V999/tests/test_dual_bank_gradients.py)
- [test_dual_bank_training.py](/home/xueshuailin/CCC_Phe/V999/tests/test_dual_bank_training.py)
- [test_bank_matching.py](/home/xueshuailin/CCC_Phe/V999/tests/test_bank_matching.py)

## 文档与产物

- [DUAL_BANK.md](/home/xueshuailin/CCC_Phe/V999/DUAL_BANK.md)：数学、训练和评估协议。
- [RESULTS.md](/home/xueshuailin/CCC_Phe/V999/outputs/dual_bank/RESULTS.md)：完整事实报告。
- [summary.json](/home/xueshuailin/CCC_Phe/V999/outputs/dual_bank/summary.json)：主汇总。
- [bank_matching_diagnostics.json](/home/xueshuailin/CCC_Phe/V999/outputs/dual_bank/bank_matching_diagnostics.json)：全部 bank matching 诊断。
- [background_probe.json](/home/xueshuailin/CCC_Phe/V999/outputs/dual_bank/background_probe.json)：background-only 与 phenotype-bank probe。
- [pytest.log](/home/xueshuailin/CCC_Phe/V999/outputs/dual_bank/pytest.log)：全量测试日志。
- [implementation_audit.json](/home/xueshuailin/CCC_Phe/V999/outputs/dual_bank/implementation_audit.json)：源码与冻结边界审计。
- `outputs/dual_bank/states/`：30 个 final checkpoint；每个包含 final state_dict、config、seed、stage history 和训练诊断。

现有 single-bank、inferred-WB、anti-collapse、synthetic generator、Cox、split 和基础评估 Python 文件均未修改。README、FILES 和 PROVENANCE 仅追加本轮索引与来源记录。
