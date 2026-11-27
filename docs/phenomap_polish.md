# PhenoMap V1 polishing

The observation cache, hurdle dictionary, centered activities and direct
network mixture are described in [the base model](phenomap_network.md).
The A–E workflow lives in `scripts/polish_network_v1.py`. The original
V1 and the direct-network A checkpoints remain independently runnable.

## Added objectives

The structured dictionary adds two optional regularizers to the original
dictionary penalty.

* **LR graph.** Resolve complete human CellChat ligand/receptor complexes to
  sorted subunit sets. Retain every matching CellChat pathway as an annotation.
  A shared exact ligand contributes 1 and a shared exact receptor contributes
  0.75, divided by that arm's family size minus one. Associations also require
  agreement on the cached signaling type. Known disjoint pathways block the
  association. A shared annotated pathway increases the arm contribution by
  35%. Pathway co-membership without a shared arm carries no similarity
  constraint, since pathway membership alone has no signed interpretation.
  No downstream response relation is inferred. Apply symmetric degree scaling
  to the resulting sparse adjacency. Each directed sender–receiver pair has
  its own supported subgraph. Add normalized squared loading differences for
  both heads independently. Unlinked interactions retain independent weights.
* **High-cosine penalty.** Concatenate both heads for a program and apply
  mean squared hinge above cosine 0.80 to distinct program pairs. Loadings
  remain nonnegative and scaled as in A. The penalty allows shared edges.
* **Support sensitivity.** The D candidate uses square-root shrinkage on
  S_train/(S_train+32) and C_train/(C_train+12), and narrows the LR frequency
  multiplier's clipping interval. Occurrence trials on the same edge retain
  equal weights for zeros and positives. This changes the composite training
  criterion, while prediction is always measured on unweighted sender trials.

For E, node evidence still uses the original count cap 16. The scalar edge
temperature multiplies both observed hurdle log-gain terms. This preserves
the exact edge-wise margin decomposition of the base model. Score diagnostics
split the winner-versus-runner-up margin into node, edge and mixture terms.
A separate descriptive discount divides absolute edge margin by shared
ligand, receptor or pathway group size to measure concentration of evidence
in related LRs. It has no training effect.

## Comparison and model selection

The A baseline has two frozen dictionary / mixture checkpoints. In B and C,
train the proposed dictionaries with the same rank-independent local solver
and spatial sampler. Compare via equally weighted random-edge and grouped
validation protocols. In grouped holdout, choose target LRs and hide every
LR sharing any target ligand or receptor subunit, in either role, for all
typed directions. Score target LRs only. This tests prediction from other
genes within new windows. The dictionary already encountered the target LRs
in training windows. The protocol does not claim generalization to never
measured genes.

Validation utility combines raw trial-level detection log loss (0.35), positive
detection log loss (0.15), and positive log intensity MSE (0.50), each relative
to the pooled baseline. B and C retain an extension only when prediction and
matched-program stability meet the recorded tolerances. D requires a minimum
0.003 utility improvement with allowable 0.04 matched-cosine variation.
Stage C varies rank with the same diversity penalty. The R=12 comparison
isolates its effect at the original rank. Lower-rank comparisons are conditional
on that penalty. Compare temperatures 0.25, 0.5 and 1 at K=4, then K=4,6,8,12 at
the selected temperature. Penalize disagreement between seeds, tiny validation
components, and model complexity by the recorded criterion. Prototype separation
and occupancy are exported. The final choice is persisted before test scoring
and the third seed is fitted with those settings.

AUPRC uses 512 fixed log-probability bins to avoid materializing every
sender opportunity. Reliability reports observed and expected positives
and trial count per probability bin. Metric groups use LR detection frequency
estimated exclusively from training windows and local sender support divided
into 1–4, 5–16 and >16 trials. Zero-span and unsupported observations do not
contribute trial mass. Grouped and random protocols may use different sets of
LRs and should be compared to their own baseline.

## Execution and interpretation

```bash
python scripts/polish_network_v1.py --network NETWORK --null PAIRED_NULL \
    --reference EXISTING_SEED73_FIT EXISTING_SEED74_FIT \
    --resources HUMAN_CELLCHAT_TABLE_DIRECTORY --root NEW_RESULT_DIRECTORY
```

The result records candidate checkpoints, CPU-readable CSR LR graph, pathway
annotations, validation choices, stagewise diagnostics and unweighted test
metrics. Where a regularizer is rejected, the final toggle fits its addition.
The fit never reads author outcome labels. Run
`experiments/report_polish.py` after selection to align labels, compare
composition-matched subgroups in the same 500-unit spatial blocks, and produce
the spatial and LR-direction figures. Paired outcome differences use block
sign randomization and within-family FDR. The report selects the closest
occupied prototypes with common composition support, then evaluates outcomes.
A subtype or pathology association is not inferred from LR similarity alone.

For full sections use `experiments/prepare_full_polish.py` to build a
compatible observation cache with the sampled training gene scales.
`scripts/predict_polished_v1.py` verifies exact panel, type, radius and
expression-scale compatibility before fixed-global inference.

Use `scripts/explain_polished_v1.py --selected SELECTION --network NETWORK
--cell CELL --candidate K1 --against K2 --output NEW_DIRECTORY` for either
candidate pair. The export rows identify full ligand/receptor complexes and
preserve the measured mean, observation support, denoised prediction, typed
edge score margin and each program's signed additive contribution.

## Observed choices on the sampled sections

| Dictionary candidate | Pt73 validation utility | Ovarian validation utility |
| --- | ---: | ---: |
| A, R=12 | 0.87631 | 0.87808 |
| B, R=12 | 0.88015 | 0.88137 |
| C, R=6 | 0.89792 | 0.90871 |
| C, R=8 | 0.90021 | 0.88481 |
| C, R=12 | 0.87660 | 0.87895 |
| D, R=12 | 0.87410 | 0.87657 |

C and D build on the retained prior stage for each dataset. In Pt73, D uses
the A dictionary penalty. In the ovarian section, D includes the retained
C diversity penalty. Both default choices use the first-stage Hurdle weight
formula. The LR graph is retained as metadata and an ablation, with zero
regularizer weight in the defaults.

The Pt73 default uses the A dictionary with a fixed communication temperature
of 1 and K=4. The ovarian default uses R=12, a diversity coefficient of 4,
temperature 1 and K=8. The two seed comparisons selected these settings on
internal data. The third seed evaluates the frozen settings. Ablation
checkpoints use seed 73. The held-out evaluation includes fit and dictionary
prediction separately, since prototypes discard some of the continuous
local-activity information.

The full-slice predictor persists sequential center blocks before merging.
This permits retry after process interruption without recomputing completed
activity and assignment blocks. Full-slice inference uses frozen sampled
expression scales and fitted globals, with no re-selection from full tissue.
