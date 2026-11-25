Current tumor-interface simulation (2026-10-23)
=============================================

`python scripts/run_interface_baselines.py` runs the seven established native
baseline adapters on the current frozen 22,000-cell/3,000-gene simulation.
Physical GPU0 is isolated with CUDA_VISIBLE_DEVICES=0; a single GPU queue and
a CPU queue handle GraphST, STAGATE, CellCharter(scVI), scNiche, NicheCompass,
SpaNiche and BANKSY. Configuration:config/tumor_interface_baselines.json.
All cells fit each method, without validation/test splits. No current GNN
retraining, simulation generation or CCC recomputation occurs.

Primary comparison uses full KMeans K6, seed40700,n_init10 on each method's
representation (native GraphST normalized distance preserved). CellCharter's
native same-K GMM assignments are also saved separately. Cell composition and
physical pair-degree pooled expression/PCA30 are simple controls. Known K is a
benchmark input; generating state labels enter post-fit metrics/display only.
The simulation and method choices were developed on this tissue; this is a
descriptive comparison, not independent validation.

Restartable outputs and per-method logs are under NAS
`outputs/tumor_immune_interface/baselines/`. Comparison PNG/SVG/PDF and metrics
refresh after each method; the third cell of Multifocal_Communication_Simulation
displays cached results only and never starts a training job. The first two
cells/current dual-MLP result are retained. Existing unrelated GPU jobs are
left untouched. Per-method completion markers require matching input/config
and implementation hashes; the batch and worker locks prevent duplicates.

Four-notebook all-cell comparison (2026-10-17)
============================================

`python scripts/run_notebook_baselines.py` runs the existing HealthyCancer,
Kidney_DKD, Xenium Lung5K and Xenium Prime5K breast datasets. It does not
regenerate data, select new samples or use any held-out model weights.
HealthyCancer uses all 40,000 existing cells; kidney uses all 62,018 cells in
the two existing ROIs, including the existing physical halo; lung/breast use
257,782/699,110 cells. Every fitted method uses all current cells, no splits.

Seven methods: BANKSY, GraphST, STAGATE, CellCharter(scVI), scNiche,
NicheCompass and SpaNiche. Cell composition and physical pair-degree pooled
logCP10k/PCA30 are simple controls, following the supplied TumorEdge figure.
The current two GNNs have composition weight 0/1. Healthy/kidney receive
20-epoch whole-cell fits with shared initialization; current lung/breast
models are reused. Existing unrelated/TNBC/breast workers are left alone.

Comparison maps cluster frozen representations with common K: six predeclared
simulation niches, or the current dual model's automatic K for real data.
CellCharter retains its native same-K GMM. Original automatic-K model labels
are not changed. Hungarian color alignment uses truth only after fitting in
the simulation, and the dual map as a visual reference in real data.
The existing lung models have different training budgets (40/20 epochs), so
their comparison is descriptive, not an equal-budget superiority experiment.

HealthyCancer's comparison and eight descriptive metric panels use all 20
simulated slices. The map shows the fixed first healthy/cancer pair and says
so in its caption. Only this simulation receives recovery metrics. Each real
notebook gets exactly one comparison figure in its final code cell; existing
body figures are retained. Completed panels are embedded as PNGs, and pending
methods remain clearly marked until finished. Cells display saved results;
opening a notebook never silently launches a second training job.

Large-data execution is explicit in `scalable.py`: GraphST uses the native
full-graph objective with algebraically equivalent sparse readout/chunked
MSE gradients (numerically checked against native gradients); its compact
embedding is an isometry of normalized native reconstructed expression.
STAGATE follows the official Tutorial8 spatial-batch recipe, visits every
cell every epoch and infers with graph halos. BANKSY spatial weights never
cross actual samples. `spaniche_large.py` and its R runner use native
IntNMF/SpaNiche objectives, sparse Gaussian arithmetic (8-sigma cutoff),
truncated SVD and partition-based stability rather than quadratic consensus
arrays; these numerical execution adaptations are recorded in method audits.
No method is replaced with a proxy clustering algorithm.

Resource repairs on 2026-10-18: the four-cohort NicheCompass runs use its
official neighbor sampler (10 neighbors per hop, node/edge minibatches32/128)
after unbounded two-hop induced subgraphs exhausted GPU memory. All cells
still participate, the original graph/model/loss/100 epochs are retained,
and this execution setting is included in audits. GraphST's bilinear scores
are evaluated as equivalent matrix products, avoiding the native backward
N-by-64-by-64 temporary; native loss/gradient equivalence still passes.
SpaNiche's matrix element counts are promoted to double to prevent R's
32-bit integer overflow above2^31 entries; its weight formula is unchanged.
`--action repair` retries missing results only and preserves completed ones.

One GPU queue on physical GPU2 and one CPU queue are managed by a locked
controller. Logs/status are in NAS `outputs/notebook_baseline_batch/`;
per-method caches and plots are in each dataset's `baselines/` directory.
Healthy/kidney's new full-cell fits are in their semantic `all_cells/`
subdirectories; previous split-based model files remain historical only.
Completed checkpoints are reused on continuation; inspect live logs before
relaunching. A failed method is recorded, not counted as completed.

Current full-tissue comparison
============================

`python baselines/all_cells.py` fits all 8,000 cells for every official method,
with no training/validation/test split. CellCharter scVI explicitly uses train_size=1,
validation_size=0 and no early stopping; NicheCompass uses zero node/edge validation
ratios. All normalization, model fitting and common K=4 clustering use all cells.
The same `TumorEdge_CCC_Composition_Decoder_Comparison.ipynb` receives spatial and
multi-metric plots plus CCC/composition interpretation. NAS outputs are under
`outputs/tumor_edge_ccc_composition_comparison/all_cells/baselines/`.

Older split-based protocol (historical results only)
===================================================

Baseline adapters for the independent tumor-edge simulation, never clinical Cox.

- GraphST: audited upstream commit in `vendor/graphst/provenance.json`, upstream
  AGPL license retained. Sparse encoder/reconstruction/contrastive losses unmodified.
  Train on original GNN training physical cells; instantiate its graph-independent
  parameters on the full inference graph. Published normalized `emb` + KMeans.
- CellCharter 0.3.7: official transcriptomics tutorial workflow: raw counts,
  sample-conditioned scVI with ZINB and default 10-dimensional latent space,
  per-sample Delaunay graph (percentile=99), three-layer official
  `aggregate_neighbors` and `Cluster` (full-covariance TorchGMM), fixed K=4.
  scVI runs in the compatible NAS environment specified in config.json;
  the existing cccphe packages are not modified. scVI uses at most 400 epochs
  with native early stopping and training physical cells only. Its frozen
  encoder produces all-cell latent means. The common spatial test also
  excludes centers whose actual three-hop Delaunay donors cross physical folds.
  Previous PCA results remain as an explicitly separate control on NAS.
  Prepare the isolated scVI environment with
  `python scripts/prepare_cellcharter_scvi.py` from cccphe.
  Prediction calls the public fitted TorchGMM module's log responsibilities
  and argmax, avoiding an incompatible Lightning 1.9 DataLoader wrapper;
  the original EM fit and Gaussian posterior calculation are unchanged.
- scNiche: pinned [official source](https://github.com/ZJUFanLab/scNiche),
  three-view MGAE/GFN/MMIM and native `Runner_batch`. Views are observed
  type composition, center expression PCA and neighboring expression PCA.
  PCAs fit training cells only; inference batches have the same size as
  training and never mix physical folds. DGL adjacency size and NumPy sequence
  compatibility fixes are documented in `vendor/scniche/provenance.json`.
- NicheCompass 0.3.3: [official package](https://github.com/Lotfollahi-lab/nichecompass),
  native graph/expression losses and default **active-program** posterior for
  clustering, with 100 de novo programs. Prior programs use the entire public
  LR atlas intersected with measured real genes, never the planted MI list.
  This is an explicitly reported LR-only prior variant: synthetic genes have
  no invented TF/target annotations. No cell types enter this model.
- SpaNiche: pinned [official R source](https://github.com/SiyuanHuang1/SpaNiche)
  with official IntNMF NNDSVD/NNLS initialization and unchanged graph-regularized
  joint NMF. Observed cell types replace spot deconvolution. Abundance has three
  spatial views and LR products have four directional/range views. Radii derive
  from training KNN30 reach; graph bandwidth is 20 micrometres. The missing
  held-out interface freezes training H/modality weights and estimates W using
  the same NNLS and graph objective within each held-out physical fold. Native
  topic-argmax assignments are saved alongside the common K=4 clustering.
- BANKSY: pinned [official Python features](https://github.com/prabhakarlab/Banksy_py),
  scaled-Gaussian neighborhood mean and centered m=1 azimuthal Gabor filter.
  Predeclared tissue-domain lambda=0.8, native K15/K30. Training graph donors
  are restricted to their physical fold and radial weights renormalized;
  frozen inference uses the full native spatial graph, with actual donor
  footprint checked on the common test set. The official z-score and
  concatenation have an optional training-reference hook. PCA fits training
  cells only. No cell types or LR priors enter this method.
- STAGATE: pinned [official PyG implementation](https://github.com/RucDongLab/STAGATE_pyG),
  native tied graph-attention autoencoder, expression MSE, Adam and clipping.
  Native hidden 512 / latent 30 / 1,000 epochs; only the induced training-cell
  graph enters optimization. A return-model hook enables saving/frozen inference,
  and loss logging stores detached scalars; no model/loss changes. Common KNN30
  symmetrization and self-loops are recorded. No cell types/CCC labels are used.

Fixed K=4, one seed, identical base gene panel/coordinates and source training
centers. No appended response genes, anatomical labels or MI dictionaries enter
baseline fitting. Cell type/CCC supervision differs across methods and is reported.
Fitted normalizers and models exclude held-out physical cells. Common evaluation
uses two-hop dependency purge, including GraphST's symmetric neighbors, for all
methods, plus the native CellCharter three-hop Delaunay donor check when
using scVI. All-center metrics/maps include training cells and are descriptive.
Matching for F1/A-B accuracy/colors is fitted on training centers after clustering.
Raw labels, package versions, source hashes, runtimes and every configuration are
saved. Additional common-KMeans CellCharter labels expose clustering-backend effects.

Run the appended Notebook cells in `cccphe` (GPU 2); completed matching results
are reused. `config.json` controls budgets. No model defaults are tuned using truth.
Figures/arrays/checkpoints are stored in NAS `baseline_comparison/`, not here.
The current native-program report uses `baseline_comparison_official/`; the
earlier all-program NicheCompass control remains intact. Reused model files
are identical trained weights, not a fresh training or a truth-selected model.
SpaNiche needs `/usr/bin/Rscript` and MASS; its core R functions and IntNMF
dependencies are vendored, so Seurat is not needed for matrix I/O. DGL/PyG
wheels match cccphe's PyTorch 2.3 / CUDA 12.1. Full seed/budget configuration
and adapter limitations are included in the Notebook and each result audit.
The expanded BANKSY/STAGATE report uses `baseline_comparison_banksy_stagate/`.
The current scVI CellCharter report uses `baseline_comparison_cellcharter_scvi/`.
The isolated-inference BANKSY interface control remains in
`baseline_comparison_spatial_controls/` and is not selected using its score.
Unchanged completed methods are copied only after matching input hashes,
method-specific settings, implementation hashes and package versions; the
earlier five-method report and all its trained models remain intact.
