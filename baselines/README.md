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
