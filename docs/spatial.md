# Spatial niches from composition and directed communication

Use `SpatialNiches.coarse(...)` to preview coarse classes and their adjacency
enrichment. Use `SpatialNiches.fit(...)` to fit each coarse class independently,
or pass `interface=(a, b)` to fit a single manually selected interface. Use the
same input, cell order and settings for the preview and the fit. The fit
recomputes the coarse labels. Each interface center remains a member of its
original coarse class and may appear in the output of both analyses.

Inputs are aligned cell-by-gene raw counts, unique cell IDs, cell types,
specimen IDs, registered 2D or 3D coordinates in micrometres, and a
species-appropriate TSV with `ligand` and `receptor` columns. Supply raw counts
in the `counts` layer by default, or use `--layer X`. Coordinate tables use
`cell_id`, `x_um`, `y_um` and optionally `z_um`. The CosMx lung option
`--lung-stim-dir` checks alignment before converting to micrometres.

## Workflow

1. **Coarse classes.** Gaussian-weighted cell-type proportions include the
   center and all cells within `window_um`. The composition similarity graph
   and its Leiden partition are unchanged.
2. **Spatial enrichment.** Triangulate independently in each specimen.
   Use the affine span for coplanar/collinear coordinates, and use one vertex
   per coincident location (the first input cell at that coordinate). Filter
   edges strictly longer than `delaunay_max_um`, which defaults to the
   effective `contact_um`. Count each undirected edge once, pool counts across
   specimens, and estimate the null mean and population SD from 1,000 default
   permutations that shuffle labels separately within specimens. The score is
   (observed − null mean) / null SD. Zero null SD gives `NA` in the table and
   grey in the heatmap. Diagonal counts are within-class edges. Contact counts
   are unique endpoint cells on each side.
3. **Manual interface.** Inspect the heatmap, then select two coarse IDs.
   Within each specimen, use the midpoints of retained edges linking the two
   classes. Keep cells of either class within `interface_um` of their nearest
   midpoint. Specimens with no connecting edges contribute no interface centers.
4. **CCC tensor.** Neighbors are within the signaling-type radius and the same
   specimen, regardless of coarse class. Bidirectional LR scores are averaged
   over eligible neighbors of each type, with no distance decay. A present
   neighbor type with zero expression is observed zero; absence is masked.
5. **CP and clustering.** Each repeat fits A, B and C on all centers. The
   default H1 representation Gaussian-averages A over the current object's
   `window_um` neighborhood, including the center. Consensus Leiden is unchanged.

Expression defaults to `expression_preprocessing="commuspace"`: CP10K, log1p,
then clip and scale each gene by its nonzero 95th percentile. The original
CP10K scoring is available as `"cp10k"`; complexes use the geometric mean.
`tensor_log1p=True` optionally transforms the resulting tensor. Its default is
false to avoid an additional compression after the default expression transform.

`representation` selects `"A"`, `"H1"` (default), `"H2"`, or `"H3"`.
H2 concatenates per-cell-type weighted means; H3 uses the total neighborhood
weight as denominator for every cell type, so its type contributions sum to H1.
Only cells within the same analysis object provide A for pooling; CCC neighbors
can still be outside it. All four representations are saved for each repeat.

The dense features passed to the existing Leiden graph use zero placeholders
for absent types. The exported support mask distinguishes these placeholders
from observed zero activity. Use it when interpreting or summarizing features.
The current graph distance itself does not mask those dimensions.

## CP objective and execution

Reconstruction is mean squared error over observed center–neighbor-type–LR
entries, including observed zeros, using the mask for the LR signaling type. The data are divided by
their observed RMS for numerical conditioning. Missing pairs contribute no
reconstruction loss. An object with no nonzero CCC signal receives no CP
programs. Centers without any observed pairs have zero activity.

Each epoch streams every center in computational batches. The loss uses sparse
matrix multiplication and the LR factor Gram matrix to account for predicted
values at observed zeros without materializing a dense reconstruction.
After summing reconstruction gradients across batches, compute the BC joint
orthogonality gradient and take one Adam step followed by nonnegative projection.
There is no A orthogonality penalty. The BC penalty uses the elementwise product
of the column-normalized B and C Gram matrices. Its coefficient `cp_beta` is
calibrated to the initial reconstruction and penalty magnitudes, with zero
penalty if the initial penalty is zero. Best-epoch factors and loss history are
saved. B and C are column-normalized on export, with their scales absorbed in A.

The default batch cap is 128 centers. CUDA reduces this cap based on current
free memory and a conservative estimate that includes observed fibers, factors
and optimizer state. Factors and optimizer state reside on the selected
device. Sparse observed batches are prepared once and retained in host memory
for reuse across epochs; their contents are transferred in batches. This avoids
repeated preprocessing but requires host memory for the prepared sparse tensor.
A single center's observed fibers and all
factor matrices must fit on the device. CUDA runs one context worker per device.
The CPU supports multiple workers in separate processes.

For communication-feature clustering, each feature is scaled by the 90th
percentile of its positive values before `log1p`. Missing and zero values remain
zero. This prevents rare features whose overall 90th percentile is zero from
being amplified by an arbitrary near-zero denominator.

## Parameters

`window_um=50`, `contact_um=50`, `interface_um=25`, `sigma_um=20`,
`expression_preprocessing="commuspace"`, `expression_quantile=0.95`,
`tensor_log1p=False`, and `representation="H1"`.
Contact and ECM radii are 20 μm; Secreted is 100 μm; Unknown and Ambiguous use
`contact_um=50`. Counts and effective weights are equal because communication
neighbors are equally weighted. Gaussian weights remain in local A aggregation.

CP defaults are `rank_mode="manual"`, `rank=12`, `iterations=500`, `repeats=5`,
`cp_beta=0.02`, `cp_learning_rate=0.03`, `cp_tol=1e-4`, `cp_patience=5`.
`rank_mode="auto"` evaluates `rank_candidates=(4,8,12,16,24)` using all
centers and repeated initializations. For each candidate, mean reconstruction
error and pairwise Hungarian-matched cosine similarity of complete BC patterns
are recorded. The interior rank maximizing normalized chord-elbow improvement
multiplied by consistency is selected. An invalid candidate range or a curve
without an interior positive elbow raises an error; set rank manually in that case.

For example: `--rank-mode auto --rank-candidates 4 8 12 16 24`, or
`--rank 12 --representation H3 --expression-preprocessing cp10k --tensor-log1p`.
All settings may also be supplied through Python `Settings(...)` or a JSON file.

`cp_device="auto"` prefers CUDA device 2 when at least three CUDA devices
are visible, otherwise the first visible CUDA device, with CPU fallback if
CUDA is unavailable. Override with `--cp-device cpu` or `--cp-device cuda:2`.
`--cp-batch-size` overrides the computational batch cap. Set other parameters
in a JSON file passed to `--settings`. Omit the retired `block_um` and
`fit_windows` keys from older settings files.

## Output

- `coarse_cells.tsv`, `composition.tsv.gz`, `enrichment.tsv` and
  `enrichment.png` describe the coarse classes and spatial enrichment.
- `cells.tsv` contains only centers fitted in the requested analysis.
  `channels.tsv` records fitted Program loadings rather than causal evidence.
- `neighbor_support.npz` contains aligned cell IDs, type names, contact counts
  and effective equal weights for the entire physical contact neighborhood.
  `program_support_*.npz` holds in-object pooling weights and type presence.
- `tensor_*.npz` and `tensor_mask_*.npz` contain sparse values, aligned cells,
  signaling-specific structural masks, radii and channel indices.
- `rank_evaluation_*.json` records reconstruction, consistency and elbow scores.
- `programs_*_repeat*.npz` contains B, C and all A/H1/H2/H3 features per repeat.
- `cp_losses_*.tsv` and `cp_losses_*.png` record loss trajectories.
  `cp_metrics_*.tsv` records calibrated weights, best epoch, maximum off-diagonal
  cosine for A and BC, counts of centers and observed entries, and runtime.
- `settings.json` records the effective settings. Checkpoints depend on the
  input, settings and model source, so old checkpoint files remain separate.

The existing lung analysis notebook is on NAS under
`/home/nas3/biod/xueshuailin/CCC_Phe/notebooks/`.

For joint control–case analysis with shared communication programs and
leave-one-donor calibration, see [comparative mode](comparative.md).

## PhenoMap V1

The independent V1 entry point is `scripts/phenomap_v1.py`. Its network cache,
model, inference, and Pt73 configuration are described in [PhenoMap V1](phenomap_v1.md).

The default direct-network refit is documented in [PhenoMap V1 network mixture](phenomap_network.md).

For the sequential LR-prior, rank, evidence-scale and niche-count comparisons,
see [PhenoMap V1 polishing](phenomap_polish.md).
