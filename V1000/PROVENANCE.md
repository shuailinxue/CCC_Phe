# V1000 初始化来源

- 初始化日期：2026-09-18。
- 来源版本：`/home/xueshuailin/CCC_Phe/V999`。
- 复制内容：`src/`、`tests/`、`scripts/`、`pyproject.toml`、`.gitignore` 和 `AGENTS.md`。
- 未复制内容：V999 的 `outputs/`、pytest cache、Python bytecode 和历史结果文档。
- 适配内容：项目根目录改为 V1000；包名改为 `phenoniche-v1000`；版本号重置为 `0.0.0`；V999 特定的活动主线规则改为等待 V1000 最新 prompt。
- 运行时禁止通过路径回退、软链接或 PYTHONPATH 引入 V999。需要复用的新代码应复制到 V1000 后适配，并在本文件继续登记。
- 初始化复制后，V999 未被修改。

## Structured ST-first 增量

- 全部新代码和产物仅写入 V1000；未修改 inherited 旧模型或生成器。
- 新模拟固定 seed=20260918；模型 seeds=31–35；permutation seeds=101–105；patient split seed=90210。
- Primary ST-only gate 通过后才运行 Frozen、Iterative、permutation 和维度敏感性。
- inner solver 只按相对 NNLS reconstruction excess 选择50步，未使用 validation/test phenotype 或 truth recovery。
- 五轮真实 phenotype adapter candidate 均因 ST reconstruction degradation 超过5%被拒绝；没有根据结果修改 loss、gate、epoch、delta bound、beta 或噪声。
