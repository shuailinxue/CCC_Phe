# V1005 — TensorCCC + CompGraph

Isolated experiment explicitly requested by the user. It does not change V1000–V1004 or replace the repository's separate ST-anchor/BayesPrism/Cox mainline. No bulk, Cox, phenotype, TLS or clinical fields enter this experiment.

**Current scope:** cross-dataset analysis is paused by user instruction. Default runners and the notebook report only independent within-dataset results. Previously generated matching artifacts remain historical and are not part of the current conclusions.

## Inputs and fixed choices

- HBC1: 166,363 cells, 313 genes, existing labels. Prime 5K: 699,110 cells, 5,101 genes, **existing provisional marker-based labels, not validated cell-type truth**.
- Original labels are preserved in `outputs/<dataset>/cells.csv`. `data.TAXONOMY` fixes the coarse alignment before fitting. HBC1 DCIS/invasive epithelial subtypes are grouped as Epithelial; ambiguous hybrid labels are Unknown. This grouping is a limitation, not evidence of biological equivalence.
- Read-only import of V1002 `build_neighborhoods`, `_side_expression`, LR atlas and `aggregate_pairwise_ccc`. Gaussian 30-nearest-cell neighborhoods, sigma 20 µm; ordered distinct physical pairs; same-type pairs allowed; geometric-mean complex expression; opportunity denominator tau=1e-4. Coverage ≥0.10; pair supported in ≥max(20,ceil(0.01 N)) neighborhoods; all components measurable. No Top-N feature selection.
- COO/CSR retained features have explicit sender/receiver/LR indices. The encoder contracts the LR mode, then sender/receiver modes, **before** flattening the small projected core. H stores only retained coordinates; other conceptual tensor entries are excluded, not scored as zeros.
- K=16; ranks min(C,8), min(C,8), 32; independent composition NMF Kc=8. Seeds 40700–40704. Adam lr .001, 5 epochs (no convergence guarantee), maximum batch 256 with fixed memory bound, feature chunk 2048.
- Loss weights lowrank .001, geometry .01, H L1 .001, group .001; singular-value epsilon .001. Huber reconstruction and sparsity penalties use dimension-normalized averages. Geometry is a sampled local-neighbor weighted average. The no-prior ablation removes lowrank and geometry only.
- Feature-independent training-only RMS rescales CCC internally; saved H is returned to raw CCC units. The differentiable nonconvex latent singular-value penalty is log1p, not rank truncation.
- Graph KNN=15, Leiden resolution=1, two Leiden iterations. CCC-derived centered randomized PCA fixes G0 without outcomes or spatial-only adjacency. Composition and CCC graph total edge mass are normalized to N before Gccc+alpha Gcomp; primary alpha .2, sensitivity 0/.1/.2/.3. No shared W or latent averaging.
- Fixed entry hash holds out 10% of retained entries including zeros. Encoder and PCA input mask held-out entries; reconstruction training excludes them. Evaluation uses a fixed random 1,024-anchor subset for tractability, never for tuning. Filtering and cell labels are transductive whole-dataset inputs; this is entry reconstruction, not independent-patient validation.
- Spatial agreement uses all 30 neighborhood members excluding self. CCC-flat uses one fixed PCA representation/G0 and five Leiden seeds; other branches also vary their representation seeds. Report this difference when interpreting seed stability.
- Program matching uses only shared retained directed features, a conservative subset of shared measurable LR and types. Missing assay features cannot establish dataset-specific biology. High between-program similarity may also indicate collapse. No niche accuracy is claimed.

Inspiration only: [Fu, Hu & Wang, ICML 2026](https://proceedings.mlr.press/v306/fu26g.html). The implementation borrows nonconvex low-rank regularization, a local geometry prior and robust reconstruction; it does not reproduce their multiview self-expression formulation.

## Execution

Use the existing `/home/xueshuailin/miniconda3/envs/cccphe/bin/python`. Set `PYTHONDONTWRITEBYTECODE=1`, `PYTHONPATH=V1005/src`, `NUMBA_CACHE_DIR=V1005/outputs/cache/numba`, and BLAS/OMP thread counts to 4. No package/environment mutation is required.

```bash
python -m pytest V1005/tests -q -o cache_dir=V1005/outputs/pytest_cache --basetemp=V1005/outputs/pytest_tmp
python V1005/scripts/run_hbc1.py
python V1005/scripts/run_xenium5k.py
python V1005/scripts/finalize.py
```

The first two runners may run independently on GPU 1 and GPU 2. GPU 0 or CPU is used if fewer devices exist. Resume preserves completed CCC blocks and model checkpoints with identical configuration. Partial epochs rerun from the fixed seed; partial losses are not a completed model. `run_v1005.py` is a sequential convenience wrapper. Finalization requires complete results, executes the presentation notebook without cross-dataset matching. Tests must pass before notebook execution.

## Outputs

`outputs/{hbc1,xenium5k}`: immutable-per-run config; tensor/annotation audit; explicit feature mapping; sparse CCC cache; seed/model Z,H,weights,losses; five-baseline and seed-stability tables; primary niche composition/program/edge tables; alpha sensitivity. `outputs/cross_dataset`: shared features and matched programs. `V1005_breast_niche_walkthrough.ipynb`: 15 sections loading saved outputs and module functions, including full-cell spatial maps, fixed-subsample UMAP (display only), program heatmaps and top edges. Large numeric/figure artifacts are ignored by git. Runtime failures remain in logs and must be reported.

`outputs/protected_versions_before.json` records preexisting V1000–V1004 file sizes/mtimes for an end-of-task read-only check; preexisting V1002 user changes are not reverted.
