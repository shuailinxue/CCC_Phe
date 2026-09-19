# Provenance

- LR vocabulary：本地 CommuSpace human union atlas，canonical direction-preserving dedup 后 8,408 LR。
- Neighborhood、Gaussian distance kernel、ordered non-self pair aggregation和 opportunity correction复用 V1001 语义。
- 新 simulation：`simulation_cells.py`，从坐标、domain、cell type、zero-inflated NB-like counts 和 complex-aware LR expression生成观测。
- 新 benchmark：`simulation_benchmark.py` 和 `simulation_model.py`，固定 27×3 grid 与四种 ablation。
- 图形：`simulation_figures.py`，Python/matplotlib，PDF 可编辑文字、PNG 300 dpi。
- 旧 selected-CCC 和 tensor 结果不参与当前主 benchmark；历史输出保存在 `outputs/v1002/`。
