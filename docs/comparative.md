# Control–case comparative mode

`phenoniche.comparative.fit_control_case` fits control and case cells jointly.
Inputs share one gene panel, cell-type vocabulary and LR table; physical graphs
are built within `sample_id`. Metadata provides `sample_id`, `donor_id`,
`condition` (`control` / `case`), and `batch` for every cell. Batch is recorded,
not corrected. A sample belongs to one donor, condition and batch. Data are
row-aligned and coordinates use micrometers.

```python
from phenoniche.spatial import Settings
from phenoniche.comparative import ComparativeSettings, fit_control_case

result = fit_control_case(
    counts=counts, genes=genes, cell_ids=cell_ids,
    coordinates_um=coordinates_um, cell_types=cell_types,
    metadata=metadata, lr=lr_table,
    settings=Settings(),
    comparative=ComparativeSettings(k=10, balance="donor"),
    cache_dir="/home/nas3/biod/xueshuailin/CCC_Phe/results/my_dataset/ours/comparative_cache",
)
print(result.cells[["coarse", "niche", "reference_status", "abnormality"]])
print(result.donor_proportions)
```

The default is `k=10`, Euclidean distance in the **original composition**,
Gaussian matching sigma `0.2`, two donor-balanced 0.95 quantiles, and unweighted
CP (`balance="none"`). `balance="sample"` or `"donor"` averages the masked
reconstruction loss separately over each sample's or donor's observed tensor
entries, then equally over groups. Cosine distance can be selected with
`matching_metric="cosine"`. The existing `Settings` controls radii, Gaussian
weights, CP, repeats and clustering. Complete LR complexes and shared channels,
nonnegativity, masks, expression scaling and program pooling follow discovery.

Within each coarse niche, all control and case centers jointly fit the CP model.
Each repeat has its own factors; downstream comparisons use the first repeat's
shared basis. Leave-one-control-donor matching provides the reference
composition-distance distribution, with equal total weight per donor. Its
quantile is a maximum permitted Kth-neighbor distance. The abnormality quantile
is calculated independently from control residual norms among composition-valid
leave-one-donor-out matches, also with equal donor weight. Each requires at least
two contributing control donors. If distance calibration is unavailable,
computable case residuals are retained and marked `uncalibrated`; if no K
controls exist or a calibrated distance is exceeded, the case is `no_reference`
and has no expected vector or residual. If abnormality calibration is unavailable,
no abnormality claim is made. The continuous score is the Euclidean norm of
`H - H_expected` and is reported alongside the signed residual.

`result.contexts[coarse]` contains aligned `cell_ids`, shared CP fits, pooled
`features`, pooling support, `expected`, `delta`, distances, scores, cell-type
and LR labels, signaling types, and both
calibration thresholds. `result.matches` provides reference cell IDs, distances,
and weights, including candidates later rejected by the distance threshold.
`result.cells` retains metadata, reference status, abnormality status and niche
labels. Signed residuals are encoded as their positive and negative parts before
the existing consensus Leiden routine; abnormality thresholds do not select
features or assign clusters. `R` niche labels refer to residual-based clusters;
`U` labels use the existing actual-communication clustering when a residual
cannot be computed. These namespaces are not directly comparable. The donor
summary reports counts and fractions within each donor, condition and coarse
niche, including zero counts for observed niches. It is evidence for follow-up,
not a claim of a new or expanded disease state.

The mandatory `cache_dir` contains a fingerprinted directory with the input
manifest, cell IDs, parameters, software/source versions, composition, sparse
communication tensors and observation masks, repeated CP fits, and final result.
Identical reruns load cached results; input, parameter, software or source changes
produce a new cache. Existing cache directories are not overwritten.
