# V1005 — simplex TensorCCC + CompGraph

The current post-training niche definition uses the already trained seed-40700 CCC latent representation and normalized graph fusion `G_ccc + 0.2 G_comp + 0.1 G_spatial`. It performs a deterministic Leiden resolution sweep and does not retrain DeepTensorCCC. Results are written under `outputs/<dataset>/graph_selection/`.

Only V1005 is modified. V1000–V1004 remain read-only. This is the user-authorized isolated spatial experiment, not a change to the repository's bulk/Cox mainline.

HBC1 and Prime 5K are analyzed independently. Their panels, annotations and retained CCC spaces differ substantially. There is **no cross-dataset/cross-slice program matching** in the runners or notebook. Shared validation is deferred until consecutive sections or technically comparable data are available. Old results, including historical matching, are preserved under `outputs/archive_pre_simplex/` and are not current evidence.

## Unchanged inputs

Reuse the V1002 physical-pair aggregation, geometric-mean complex expression and CommuSpace atlas through read-only imports. Directed distinct physical pairs include same-type pairs, use Gaussian sigma=20 µm, 30 neighbors and opportunity tau=1e-4. Measurability, expression coverage ≥0.10 and opportunity support ≥max(20,ceil(.01 N)) are unchanged. No phenotype inputs or Top-N input filtering.

HBC1 has 166,363 cells; Prime 5K has 699,110. Existing coarse taxonomy is unchanged; original labels remain in `cells.csv`. Prime 5K labels are provisional, and its zero-count cells and sparse gene coverage remain explicit QC limitations. Existing CCC/PCA/G0 caches are reused. No N×C×C×L dense tensor is materialized.

## Scale-constrained reconstruction

- Encoder: sender/receiver/LR mode projections, LayerNorm/MLP, **Softmax**. Every Z row sums to one.
- Decoder: positive H via Softplus, divided by its program-wise sum **on every decode**. Saved H rows also sum to one.
- With both constraints, ZH represents a distribution, not arbitrary CCC magnitude. Fixed, non-learned anchor mass is `m = sum(training-visible CCC)/.9`. Raw reconstruction is `m * (ZH)`. The mass uses no validation/test values and cannot restore a trainable Z/H scaling degeneracy.
- Encoder inputs and Huber targets use CCC density (`F * CCC/m`). Huber compares `F * ZH` against this density, avoiding vanishing loss simply because F is large. Zero observed-mass rows have zero input and zero reconstructed raw mass; zero denominators are handled explicitly.
- H L1 is constant under sum-to-one normalization. The sparsity term is therefore mean normalized H entropy; minimizing it favors concentration. Sender–receiver group L2 sparsity remains. These are reported, not assumed to produce distinct programs.
- **No low-rank penalty is called by training.** The old standalone function is retained for reference/tests only.

## Fixed training protocol

K=16, Kc=8, mode ranks min(C,8)/min(C,8)/32, Adam lr=.001, batch maximum256, feature chunks2048, lambda_graph=.01, lambda_sparse=.001, lambda_group=.001. Seeds40700–40704. NMF max_iter300 and its convergence warnings are retained.

Stage1: 15 reconstruction+sparsity/group epochs, no geometry. Stage2: up to25 further epochs, with graph weight ramping from0 to.01 over10 epochs for CCC-tensor. Early stopping patience6 with minimum improvement1e-5 starts in stage2 and cannot stop before the ramp budget is traversed. Save the best qualifying validation checkpoint, not the last epoch. If the best checkpoint is from warm-up, its graph weight is explicitly reported as0.

The no-prior ablation uses the same budget and no geometry. The two branches reuse the same seed's complete warm-up model, optimizer and RNG state; this avoids duplicate calculation and starts the graph comparison from the same representation.

The original fixed 10% entry holdout is divided deterministically into disjoint validation/test subsets. The encoder and G0 see neither subset. Validation on fixed1,024 anchors selects checkpoints; separate test entries on fixed1,024 anchors provide final raw-CCC reconstruction metrics. Whole-dataset feature filtering remains transductive, not independent-patient validation.

G0 remains the fixed raw CCC-flat PCA KNN. The sampled geometry loss uses degree importance correction to estimate mean weighted G0 edge loss. Each epoch audits a fixed sample of up to4,096 anchors: neighbor/random squared distances, entropy, global/per-cell effective program counts, per-dimension variance, sampled full-Z max/min singular values, and fraction of dimensions with variance<1e-8. High entropy or high ARI alone is not evidence against collapse.

## Baselines and evaluation

Composition-only; CCC-flat; simplex CCC-autoencoder-no-prior; simplex CCC-tensor with graph prior; Proposed = CCC-tensor graph + .2 composition graph. Graphs are normalized to the same total edge mass before fusion. KNN15, Leiden resolution1 and two Leiden iterations are unchanged. Alpha sensitivity0/.1/.2/.3 is descriptive, not parameter selection.

For every method/seed, report niche count, median/min size, cell fraction in niches smaller than20, connected components, mean degree and degree quantiles, spatial agreement and held-out errors where a CCC decoder exists. Within-dataset five-seed ARI/NMI and program stability remain. CCC-flat retains its fixed PCA basis and varies Leiden seed only.

Program reports include relative top sender, receiver, sender→receiver pair, LR and directed edges, plus entropy/sparsity. Niche activity is mean simplex Z; niche top-edge weights are the corresponding relative mixture. Niche IDs are zero-based and program IDs are one-based throughout saved tables/figures.

## Run

Use `/home/xueshuailin/miniconda3/envs/cccphe/bin/python` with `PYTHONDONTWRITEBYTECODE=1`, BLAS/OMP threads4 and `NUMBA_CACHE_DIR=V1005/outputs/cache/numba`.

```bash
python -m pytest V1005/tests -q -o cache_dir=V1005/outputs/pytest_cache --basetemp=V1005/outputs/pytest_tmp
python V1005/scripts/run_fixed_experiments.py
```

The runner uses four bounded local seed processes, saves separate seed logs/metrics, aggregates both independent analyses, reruns tests and executes the notebook. Completed seed outputs can be resumed. `run_hbc1.py`, `run_xenium5k.py`, and `run_v1005.py` also support sequential execution. `finalize.py` only audits and renders completed results; it does not retrain or match datasets.

The notebook calls formal source functions only. It displays training/collapse curves, fragmentation tables, spatial maps, program interpretations, stability and alpha sensitivity. Failed, collapsed or inferior results are reported without tuning resolution, seeds, filters or model settings after inspecting outcomes.
