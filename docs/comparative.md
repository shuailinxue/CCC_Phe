# Control–case comparative mode

`phenoniche.comparative.fit_control_case` jointly analyzes control and case
cells. Inputs use a common gene panel, cell-type vocabulary and LR table.
Physical graphs are built independently within `sample_id`. Row-aligned metadata
must contain `sample_id`, `donor_id`, `condition` (`control` or `case`), and
`batch`. Batch is recorded, not corrected. Coordinates are in micrometers.

```python
from phenoniche.spatial import Settings
from phenoniche.comparative import ComparativeSettings, fit_control_case

result = fit_control_case(
    counts, genes, cell_ids, coordinates_um, cell_types, metadata, lr_table,
    settings=Settings(),
    comparative=ComparativeSettings(k=10),
    cache_dir="/home/nas3/biod/xueshuailin/CCC_Phe/results/my_dataset/ours/comparative_cache",
)
print(result.cells[["coarse", "functional_niche", "remodeling_niche", "abnormality"]])
print(result.functional_donor_proportions)
print(result.remodeling_donor_proportions)
```

The defaults are donor-balanced CP, `k=10` neighbors **per reference donor**,
Euclidean composition distance, Gaussian matching sigma `0.2`, and donor-weighted
0.95 quantiles for both composition and communication calibration. Select
`balance="sample"` or `"none"`, or `matching_metric="cosine"`, when appropriate.
Existing `Settings` governs physical radii, expression and CP settings. Groups
without observed tensor entries are excluded from balancing and listed in each
context's `excluded_balance_groups`. Observed zero-valued entries still count.

For each query, K neighbors are selected independently within each eligible
control donor. A donor passes when its Kth-neighbor composition distance is at
most the threshold calibrated from control leave-one-donor-out matches. The
calibration uses the nearest eligible donor distance per query and gives each
query donor equal total weight. Accepted donors contribute equally to expected
communication; within each donor, its K neighbors receive Gaussian composition
weights. Calibration excludes the query's donor; case matching uses the same
rules against all control donors. If a threshold cannot be calibrated, available
matches and signed residuals remain `uncalibrated`; no anomaly claim is made.
If no donor has K cells or passes a calibrated threshold, the result is
`no_reference` and no residual is computed.

`delta` remains the signed difference between actual H and the donor-balanced
expected H. The anomaly score is the Euclidean distance from actual H to its
nearest composition-matched control neighbor's H; it does not measure distance
to the mean, so distinct healthy communication modes can each be normal. Its
threshold is the donor-weighted quantile of control leave-one-donor-out scores,
computed using the same matching and donor-filtering rules. A threshold requires
at least two contributing control donors. Raw scores and thresholds are
retained; they are descriptive, not evidence that a disease state is new.

Each coarse class has one joint CP fit per repeat. Repeats have different
initializations and independently calibrated regularization weights. To compare
objectives consistently, selection uses the median effective alpha and beta
across repeats and evaluates each retained solution with these shared weights;
the lowest score is the main result. `contexts` records the selected repeat,
weights, scores and adjusted Rand indices comparing the functional and
remodeling labels across repeats. A remodeling comparison reports the number of
cells that have reference matches in both repeats.

The functional labels cluster actual H using the existing consensus algorithm.
Remodeling labels independently cluster the positive and negative parts of
`delta`, using the same algorithm. Unmatched cells retain functional labels and
are marked `no_reference` for remodeling. Abnormality labels do not enter either
clustering. The two donor summaries provide counts and within-donor/condition/
coarse fractions, including zeros for niches observed in that coarse class.
`donor_proportions` is the remodeling summary. `contexts` retains cell IDs,
features, expected values, signed residuals, factors, support, and thresholds;
`matches` retains reference cell IDs, distances, and weights.

The required `cache_dir` stores fingerprinted input/parameter/software/source
metadata, composition, sparse communication tensors and observation masks, and
CP fits. Matching settings have a separate result key: changing K, distance,
sigma or calibration quantiles reuses the same composition, tensor and CP fits.
Changing CP balancing, inputs or implementation creates a distinct upstream
cache. No matching result is used to claim condition-specific emergence or
expansion without downstream donor-level analysis.
