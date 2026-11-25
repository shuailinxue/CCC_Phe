# Spatial niches: composition and bidirectional communication

Entry point is `phenoniche.spatial.SpatialNiches.fit`, or
`scripts/run.py`. Input is raw cell-by-gene counts, micrometer coordinates,
cell types, specimen identifiers and a ligand–receptor table. For the lung data
the CLI accepts `--lung-stim-dir`, which verifies registration before converting
coordinates to micrometers. Use `python scripts/run.py --help` for the
input options. A coordinates TSV requires `cell_id`, `x_um`, `y_um` and
optionally `z_um`. An LR TSV requires `ligand` and `receptor` columns. All
subunits of a receptor or ligand complex must be present in the gene panel.

Pass optional parameter overrides as a JSON object via `--settings`. For
example, `{"window_um": 40, "contact_um": 40, "seed": 73}` overrides the two
radii and sets the seed. The output records effective settings. Context
checkpoints are keyed by input, settings and source code, so changes to any of
these can require a new fit. Archived checkpoints and outputs retain their
original formats.

The lung analysis notebook is stored on NAS at `/home/nas3/biod/xueshuailin/CCC_Phe/notebooks/CosMx_3D_Lung_V2.ipynb`.

1. Construct an independent physical radius graph within each specimen.
   Composition uses the center plus all cells within 50 µm with Gaussian
   distance weights. Leiden partitions a nearest-neighbor graph of composition
   vectors. The physical radius and Leiden settings are configurable.
2. Determine co-located coarse classes by physical contacts. Retain an
   interface only when both flanks satisfy the configured contact support.
   Give centers touching multiple interfaces to the interface with the largest
   cross-interface neighbor fraction. Other centers belong to their coarse
   interior. Ties follow sorted coarse identifiers.
3. Within each interior or interface, for every focal cell form a three-way
   tensor of focal cell × neighbor cell type × directed LR. The last dimension
   contains sender LR followed by receiver LR. Only neighbors assigned to the
   same context enter its tensor. Focal sender values use focal ligand and
   neighbor receptor; receiver values use neighbor ligand and focal receptor.
   Each directed cell-pair expression score is a geometric mean, with Gaussian
   distance weighting, and scores are summed across each neighbor type.
   Complete LR complexes are required. Raw counts use panel-total CP10k.
   Missing neighbor types and observed zero expression are represented
   separately in the CP loss.
4. Fit nonnegative CP independently in each context. Its cell, neighbor-type
   and directed-LR loadings are saved. Each repeated fit uses a spatial block
   subsample. Rank, iterations and sampling are explicit configuration.
5. For each center, aggregate its same-context neighborhood's cell activity by
   focal-cell type. The aggregation uses the same physical radius and distance
   weights, normalized by total neighborhood weight. Build a communication
   similarity graph, run Leiden at multiple resolutions and random seeds, and
   obtain a consensus using co-clustering frequencies on similarity edges.
   Repeat the procedure for independently fitted CP programs and report
   adjusted-Rand agreement. Every cell has one final niche label.

Outputs include `cells.tsv`, `composition.tsv.gz`, `channels.tsv`,
`lr.tsv`, `settings.json`, and CP factors, focal-cell activities and
neighborhood features for each context and repeat. In `channels.tsv`,
`direction` specifies whether the focal cell is sender or receiver,
`focal_type` and `neighbor_type` identify the populations, and the two
loadings describe the corresponding program. Program loadings describe the
learned pattern; a high loading alone is not proof of causation or a unique
contribution to a particular clustering decision.

The implementation reports stability rather than assuming a consensus
partition is reproducible. Parameters and resource requirements need
validation on full specimens. The lung notebook can be opened with the optional
`notebook` dependencies, installed with `python -m pip install -e ".[notebook]"`.
