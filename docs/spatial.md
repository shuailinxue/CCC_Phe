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
4. **CCC tensor.** A center's neighbors are all other cells of its specimen
   within `contact_um`, independent of their coarse class or interface.
   Compute center-sender and center-receiver LR scores using panel-total CP10k,
   complete complexes, the geometric-mean expression score and Gaussian
   distance weights. A neighbor type with zero expression is an observed
   zero. An absent type has no tensor row and contributes no CP loss.
5. **CP and clustering.** Fit nonnegative CP independently for each analysis
   object using repeated subsamples. Block coordinates are floor(coordinate /
   `block_um`), with separate block IDs across specimens. For each repeat,
   randomly select approximately 80% of occupied blocks and cycle through the
   selected blocks in random order, drawing one shuffled cell per block per
   round until reaching `fit_windows` or exhausting those blocks. A single
   block uses ordinary sampling without replacement. When the entire object
   has at most `fit_windows` cells, use all of them. Project each center onto
   its object's CP factors, then pool within `window_um` using only that
   object's cells, including the center. The existing Consensus Leiden uses
   the pooled features, and repeat agreement is reported with adjusted Rand
   scores.

| `aggregation` | CCC weights | Pooled Program denominator |
|---|---|---|
| `sum-sum` | Gaussian-weighted sum for each neighbor type | Gaussian weight over all in-object cells in the window |
| `mean-sum` | Gaussian-weighted mean within each neighbor type | Same original denominator |
| `mean-mean` (default) | Gaussian-weighted mean within each neighbor type | Gaussian weight within each in-object cell type |

The dense features passed to the existing Leiden graph use zero placeholders
for absent types. The exported support mask distinguishes these placeholders
from observed zero activity. Use it when interpreting or summarizing features.
The current graph distance itself does not mask those dimensions.

## Parameters

`window_um=50`, `contact_um=50`, `interface_um=25`, `sigma_um=20`,
`block_um=100`, `fit_windows=256`, `enrichment_permutations=1000`,
`delaunay_max_um=None` (use `contact_um`), and `aggregation="mean-mean"`.
Set overrides in a JSON file passed to `--settings`. The new settings are
also available as command-line flags. Leave the same coarse partition settings
in use between preview and fit. Remove `interface_fraction` and `min_cells`
from old settings files. Both were thresholds for the retired interface filter.

## Output

- `coarse_cells.tsv`, `composition.tsv.gz`, `enrichment.tsv` and
  `enrichment.png` describe the coarse classes and spatial enrichment.
- `cells.tsv` contains only centers fitted in the requested analysis.
  `channels.tsv` records fitted Program loadings rather than causal evidence.
- `neighbor_support.npz` contains aligned cell IDs, type names, contact counts
  and effective Gaussian weights for the entire physical contact neighborhood.
  `program_support_*.npz` holds in-object pooling weights and type presence.
- `sampling_*.tsv` records each attempted repeat's sampled cell IDs, block IDs
  and fit status. `spatial_blocks.tsv` maps block IDs to specimen and block
  coordinates. `programs_*_repeat*.npz` stores successful repeats using the
  same zero-based repeat IDs and the corresponding cells, factors and scores.
- `settings.json` records the effective settings. Checkpoints depend on the
  input, settings and model source, so old checkpoint files remain separate.

The existing lung analysis notebook is on NAS under
`/home/nas3/biod/xueshuailin/CCC_Phe/notebooks/`.
