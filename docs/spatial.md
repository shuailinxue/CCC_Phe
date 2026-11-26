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
4. **CCC tensor.** A center's neighbors are other cells of its specimen within
   the radius of each LR signaling type, independent of coarse class or interface.
   Compute center-sender and center-receiver LR scores using panel-total CP10k,
   complete complexes, the geometric-mean expression score and Gaussian
   distance weights. A neighbor type with zero expression is an observed
   zero. A type absent within that LR's radius contributes no CP loss.
5. **CP and clustering.** Each repeat starts from a different nonnegative
   random initialization and jointly fits A (center activity), B (neighbor
   type loading), and C (directed LR loading) using every center in the analysis
   object. Pool the fitted A within `window_um` using only that object's cells,
   including the center. Consensus Leiden and adjusted Rand stability use
   these pooled features as before.

| `aggregation` | CCC weights | Pooled Program denominator |
|---|---|---|
| `sum-sum` | Gaussian-weighted sum for each neighbor type | Gaussian weight over all in-object cells in the window |
| `mean-sum` | Gaussian-weighted mean within each neighbor type | Same original denominator |
| `mean-mean` (default) | Gaussian-weighted mean within each neighbor type | Gaussian weight within each in-object cell type |

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
After summing reconstruction gradients across batches, compute the two
orthogonality gradients and take one Adam step followed by projection onto the
nonnegative orthant. Computational batches share B and C.

Normalize each factor column to unit L2 norm for the penalties. The A term
is the sum of squared off-diagonal entries of the full normalized A Gram
matrix. The BC term uses the elementwise product of the normalized B and C Gram
matrices, which is the Gram of the complete joint patterns. Both terms are
differentiated during training. Each weight is its configured multiplier times
initial reconstruction loss divided by the corresponding initial penalty.
The effective coefficient is zero when that penalty is numerically zero
(at most 1e-8). Track relative improvement in the total objective and retain
the factors from the best epoch. On export, B and C are column-normalized and
their scale and the data RMS are absorbed into A.

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
`enrichment_permutations=1000`, `delaunay_max_um=None` (use `contact_um`),
`iterations=500` (maximum; early stopping remains active), and
`aggregation="mean-mean"`.

CCC neighborhoods are shared by signaling type: `contact_radius_um=20`,
`secreted_radius_um=100`, and `ecm_radius_um=20`. The corresponding Gaussian
scales are `contact_sigma_um=20`, `secreted_sigma_um=40`, and
`ecm_sigma_um=20`. Unknown and ambiguous pairs retain `contact_um` and
`sigma_um`. Set any of these fields in the JSON supplied to `--settings`, or
pass them to Python `Settings(...)`. Neighbor counts and weight sums by type
are stored in `neighbor_support.npz`; masks for reconstruction are specific to
each signaling type, and an observed zero remains distinct from a missing type.

CP defaults are `rank=12`, `iterations=40` maximum optimizer steps,
`repeats=5`, `cp_learning_rate=0.03`, `cp_alpha=0.02`, `cp_beta=0.02`,
`cp_tol=1e-4` and `cp_patience=5`. Set either multiplier to zero to disable
its term independently. Set both to zero for ordinary masked nonnegative CP.

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
  and effective Gaussian weights for the entire physical contact neighborhood.
  `program_support_*.npz` holds in-object pooling weights and type presence.
- `programs_*_repeat*.npz` retains the cell rows and IDs, B and C loadings,
  focal activity A and pooled activity for each repeat.
- `cp_losses_*.tsv` and `cp_losses_*.png` record loss trajectories.
  `cp_metrics_*.tsv` records calibrated weights, best epoch, maximum off-diagonal
  cosine for A and BC, counts of centers and observed entries, and runtime.
- `settings.json` records the effective settings. Checkpoints depend on the
  input, settings and model source, so old checkpoint files remain separate.

The existing lung analysis notebook is on NAS under
`/home/nas3/biod/xueshuailin/CCC_Phe/notebooks/`.

For joint control–case analysis with shared communication programs and
leave-one-donor calibration, see [comparative mode](comparative.md).
