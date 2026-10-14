Baseline adapters for the independent tumor-edge simulation, never clinical Cox.

- GraphST: audited upstream commit in `vendor/graphst/provenance.json`, upstream
  AGPL license retained. Sparse encoder/reconstruction/contrastive losses unmodified.
  Train on original GNN training physical cells; instantiate its graph-independent
  parameters on the full inference graph. Published normalized `emb` + KMeans.
- CellCharter 0.3.7: explicitly the **PCA** representation variant, official
  `aggregate_neighbors` and `Cluster` (full-covariance TorchGMM). Original graph
  has 29 physical neighbors + center; one hop maintains the local spatial scale.
  We do not claim the scVI tutorial recipe. scVI 0.20.3 is incompatible with the
  current AnnData; no shared core package is downgraded to conceal this.
  Prediction calls the public fitted TorchGMM module's log responsibilities
  and argmax, avoiding an incompatible Lightning 1.9 DataLoader wrapper;
  the original EM fit and Gaussian posterior calculation are unchanged.

Fixed K=4, one seed, identical base gene panel/coordinates and source training
centers. No appended response genes, anatomical labels or MI dictionaries enter
baseline fitting. Cell type/CCC supervision differs across methods and is reported.
Fitted normalizers and models exclude held-out physical cells. Common evaluation
uses two-hop dependency purge, including GraphST's symmetric neighbors, for all
methods. All-center metrics/maps include training cells and are descriptive.
Matching for F1/A-B accuracy/colors is fitted on training centers after clustering.
Raw labels, package versions, source hashes, runtimes and every configuration are
saved. Additional common-KMeans CellCharter labels expose clustering-backend effects.

Run the appended Notebook cells in `cccphe` (GPU 2); completed matching results
are reused. `config.json` controls budgets. No model defaults are tuned using truth.
Figures/arrays/checkpoints are stored in NAS `baseline_comparison/`, not here.
