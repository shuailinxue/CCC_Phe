# V1000 初始文件

- `src/phenoniche/`：从 V999 复制的核心数据、模型、推断、loss、训练与评估模块，包括 inferred-WB、single-bank、anti-collapse 和 dual-bank 实现。
- `tests/`：从 V999 复制的92项测试与零权重回归 fixture。
- `scripts/`：从 V999 复制的 synthetic、inferred-WB、anti-collapse 和 dual-bank 运行入口。
- `outputs/`：V1000 独立空输出目录；未复制 V999 历史实验产物。
- `pyproject.toml`：V1000 独立包与测试配置。
- `AGENTS.md`：V1000 独立边界和复用规则。
- `README.md`：版本入口说明。
- `PROVENANCE.md`：复制来源和适配记录。

## Structured ST-first 增量

- `src/phenoniche/simulation/structured_ccc.py`：结构化高维 CCC controlled simulation。
- `src/phenoniche/training/st_first.py`：ST-only factorization、block scaling 和空间 activity inference。
- `src/phenoniche/model/adapters.py`：有界 multiplicative dictionary adapter。
- `src/phenoniche/training/iterative_refinement.py`：preservation-gated iterative calibration。
- `src/phenoniche/evaluation/niche_stability.py`：consensus、Hungarian recovery、collision 和 edge recovery。
- `src/phenoniche/evaluation/structured_st_first_benchmark.py`：完整固定实验与输出。
- `scripts/run_structured_st_first.py`：正式运行入口。
- `tests/test_structured_simulation.py`、`tests/test_st_first.py`、`tests/test_iterative_refinement.py`：simulation、leakage、gradient、boundary 和 rollback 测试。
- `outputs/structured_st_first/`：正式结果、stable anchor、adapter state、审计与报告。
