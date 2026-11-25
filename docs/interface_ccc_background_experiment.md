# Data-only CCC background/signature pilot

User explicitly requested testing the data layer before any network. This pilot
does not alter the active V1, its physical microenvironments or any notebook.
The protocol is fixed in config/interface_ccc_background.json before evaluation.

Inputs are native RNA, gene IDs, author/simulated cell type, physical coordinates,
original30members/weights, observed pair opportunities and composition. Original
4902 directed CCC candidate columns stay fixed. Planted region/state/template
annotations and the biological perturbation list never enter fitting.

1. Normalize complete-panel RNA toCP10k and calculate complex expressions with
   the existing geometric mean. Use exact weighted sqrt(L*R)/(opportunity+1e-4).
   The batched bilinear implementation is checked against the original kernel,
   including repeated physical IDs and whole-profile permutations. An actual
  512-center matrix comparison also authenticates numerical equivalence.
2. Permute complete RNA profiles jointly within cell-type and10depth-quantile
   strata. Positions, types, physical members and opportunities stay fixed;
   ligand/receptor/gene covariance within a donor profile is preserved.
  32permutations estimate per-window,per-feature background means/SD. The SD
   floor is0.1of each feature's pooled null SD, with numerical minimum1e-5.
3. Standardize observed-minus-null background and center each feature over
   physically supported windows. This screen asks for varying communication
   states, not a uniform whole-section shift. Raw CCC is kept as comparator.
4. Test coherence on a predeclared statistical center graph. Linked windows have
   nonoverlapping enclosing disks, so directly compared windows never share a
   physical member. Eight nearby neighbors within180µm, bandwidth80µm, weighted
   additionally bysqrt-composition similarity (bandwidth0.2). This graph is NOT
   a replacement of the physical30-member microenvironment. Full permutations
   recompute all original overlapping physical observations.
5.64independent permutations calibrate full/fold-specific coherence. A predictive
   t approximation andBH q<=.05, plus standardized excess>=2in both160µm
   checkerboard folds, select features. This is an EXPLORATORY approximation for
   quadratic spatial statistics, not a proven exact FDR-controlled test. The
   unperturbed-baseline and a held-out permutation report miscalibration/false
   signals. Fold gates check spatial stability; folds are not claimed to be
   independent biological replicates.
6. Group selected features with complete-linkage positive correlations>=.7of
   their local score maps. Use coherence-excess weights normalized within each
   group for native-window signature scores. Negative directions stay separate;
   isolated features are retained. No factorization or reconstruction is used.
7. DirectKMeans fixedK6,seed40700,10initializations, full22,000centers. Each data
   block is globally RMS-calibrated; no row unit normalization. Compare raw CCC,
   native-window signatures, composition-joint versions and a SEPARATELY LABELLED
   disjoint-neighbor score-averaged version. The latter adds statistical spatial
   averaging and must never be presented as native-window feature selection alone.

Observed and reference_baseline-layer branches each fit their own background
and selection; only then does evaluation read A/B/state labels. Baseline controls
use generator counterfactual RNA solely as a negative control, never to fit the
observed background. No parameters/features/K are chosen using label scores.
If no signatures survive, report that failure and a labelled composition-only
fallback; never claim such classes are communication states.

Moments and compact score/null tables are saved, not64full permuted CCC matrices.
All outputs under NAS outputs/tumor_immune_interface/ccc_background_signatures.
The worker is resumable with a source/config/code-hashed protocol and lock.
Native/signature/negative-control results are development evidence, not biological
validation or evidence that adding a future network will necessarily succeed.
