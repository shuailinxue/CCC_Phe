# Communication-defined functional subniches: evidence and proposed plan

This document is a proposal and does not change the locked V1 framework or any
cohort notebook. The requested NMF experiment and an external-signature coverage
audit were completed on 2026-10-29.

## What was tested

- Existing simulated RNA-derived, directed nonnegative CCC, not ground-truth
  programs and not a real patient CCC matrix. One composition K3 class had 4,275
  centers and 4,902 CCC columns, including 914 A and 914 B interface centers.
  K3 and this class were inspected using simulation truth in earlier development,
  so this is an exploratory test, not an independent blind benchmark.
- Frobenius NMF, rank32, 500 multiplicative updates, seed40700. Native CCC and
  original per-feature RMS-scaled CCC were each fitted. Each loading row was
  L2-normalized, compensating its activity coefficient. K-means with K=2/3/4/6
  used activity alone. Truth was not used in the NMF or full-class K-means.
- The unperturbed simulation layer was independently decomposed as a negative
  control. A separate A/B-only K2 diagnostic uses the known truth to select A/B
  after NMF; it is oracle evaluation, not a valid unsupervised clinical workflow.
- RMS-scaled rank32 NMF reduced MSE by 27.46% relative to the per-feature
  constant-mean predictor; native CCC reduction was 50.08%. RMS-scaled K2 put
  all corresponding A/B centers in the same class. Native K2 kept 96.50% in
  the same class. The A/B-only oracle also failed. These outcomes cannot prove
  every NMF variant or every low-dimensional representation will fail.
- Post-fit mean IFNG family ratio A/B was 2.263 in the scaled CCC inputs and
  1.00094 in the NMF reconstruction. The TGFβ family ratio B/A was about 2.02
  in the inputs and 1.054 in the reconstruction. Program ranks 16 and 64,
  two seeds each, and K=2/3/4/6 also failed in post-hoc exploratory checks;
  rank64 reduced global MSE to about 0.756 from rank32 0.929 yet its IFNG
  reconstructed A/B ratio was about 0.99. Mean ratios are descriptive and can
  be unstable at low expression; no causal claim follows.
- The simulation averages IFNG family A/B≈2.26; TGFβ B/A≈2.02; CAF→T
  CXCL12–CXCR4 B/A≈1.64; CXCL9/10/11–CXCR3 A/B≈1.04. Magnitudes vary by
  family and individual pair, and depend on local detection, cell opportunity,
  normalization, the square-root CCC scoring rule and data generation. A/B
  share physical templates, donors and random seeds and are not biological
  replicates. The generator directly alters gene probabilities by state; it
  does not simulate or validate a causal ligand→receptor→TF cascade.

Outputs: `outputs/tumor_immune_interface/ccc_nmf_program_pilot` on NAS,
`scripts/experiment_interface_ccc_nmf.py`, its rank/seed sensitivity script,
`postfit_mechanism_recovery.csv`, `evaluation.csv`, `nmf_spatial_comparison.png`.

## What actual literature does and does not establish

- An HNSCC spatial study displayed differentially expressed ligands using
  `log2FC≥1` (at least twofold) in FRC-like or iCAF-associated spots; this is a
  **selection threshold for ligand expression**, not a universal communication
  effect size and not an exact composition-matched A/B CCC comparison.
  https://pmc.ncbi.nlm.nih.gov/articles/PMC11702232/
- A malignant rhabdoid tumor spatial study used a 1.5-fold threshold to display
  differentially expressed genes between hot and cold tumor regions. This is a
  gene-expression threshold, and the regions need not have the same cellular
  mixture; it cannot be treated as a CCC signal effect-size distribution.
  https://pubmed.ncbi.nlm.nih.gov/40894019/
- A pancreatic spatial study found altered CAF–malignant interactions after
  therapy and supported candidate IL-6 family signaling with an ex vivo
  tumoroid coculture; it does not yield a transferable, same-composition CCC
  fold-change distribution. https://pubmed.ncbi.nlm.nih.gov/39227743/
- We did not find a universal empirical distribution of same-composition
  directed CCC ratios with the exact current score. Multiple effect sizes,
  within-state variability and independent patients are required for validation.

## What CellCall and CytoSig contribute

- CellCall constructs prior LR→TF links using KEGG and TF→target evidence,
  integrating sender/receiver expression with downstream receiver TF activity.
  https://doi.org/10.1093/nar/gkab638
- CytoSig estimates a receiver's cytokine-response activity from a signed,
  reference-weighted target-gene signature using ridge regression. It is not a
  directional sender-specific LR score. The original validated reference has
  43 cytokine signatures. https://www.nature.com/articles/s41592-021-01274-5
- Actual simulation-panel coverage audit uses the official CytoSig centroid
  file at commit ce53cc7f496f996dc0f759652e21afcd0f20bb8e: 1,062/4,881
  signature genes are present among the 3,004 simulated genes. IFNG top-100
  has 49 genes covered and 30.3% of total absolute signature weight; TGFB1
  has 54 top-100 genes and 27.8% weight. Presence is not evidence of sufficient
  expression or reliable activity inference. See `scripts/audit_simulation_cytosig_coverage.py`
  and `cytosig_simulation_panel_coverage.csv` on NAS. Coverage for real panels
  has not been measured here and needs separate verification.
- Of the 1,045 effective LR pairs in the current simulation, 182 have the
  existing combined-atlas `cellchat_v2` source flag (694/4,902 directed typed
  features). This flag is not a complete independent pathway-mapping audit.
  Dropping all other LR pairs without reporting mapping and coverage would
  change the scientific scope.

## Proposed framework (not yet implemented or validated)

1. Keep the existing physical microenvironment and composition vectors; form
   broad composition classes. Choose granularity without clinical labels and
   avoid equating center-defined truth categories with composition modes.
2. Within each broad class, use predefined CellChat/CellCall-compatible prior
   mappings to aggregate reliable directed LR evidence into pathway exposures.
   Record sender/receiver/individual-LR support and contact opportunity. Avoid
   summing an arbitrary number of LR pairs; report incomplete/unmapped axes and
   retain specific informative pairs when pathway averaging would erase them.
3. Independently score receiver transcriptional response with appropriate
   signed CytoSig/TF-target signatures, considering receiver type, measurable
   gene coverage, expressed cells, overlap, baseline and uncertainty. Missing
   response or absent receptor/cell type is missing support, not biological zero.
   Keep pathway exposure and response as separate measured views; a negative
   response score is not automatically absence, and shared downstream genes do
   not identify a unique cytokine. They need not always agree.
4. First test standardized informative program-exposure/response scores and
   their masks directly within the broad class against native CCC/NMF controls.
   If evidence is adequate, an encoder may take these compact multiview scores,
   yielding h; direct heads predict program exposure and receiver response,
   with a per-program, coverage-aware objective. Do not add back a requirement
   to reconstruct thousands of typed CCC features. Preserve separate components
   or an explicit skip path so rare signature differences are not silently lost.
   Concrete architecture and loss are hypotheses pending quantitative ablation.
5. Report each niche's prior program, underlying candidate LR, directional
   sender/receiver evidence and receptor/TF/target support. A TF footprint can
   arise from intrinsic cell state or other sources; correlated expression is
   not proof of cell-to-cell causation. Validate response on held-out genes or
   orthogonal assays, and independent samples; reusing response genes used to
   fit clusters as independent evidence is circular.
6. Test independent simulation replicates, realistic dropout and multiple
   effect/noise levels, composition-matched different-CCC, different-composition
   unchanged-CCC, no-change controls and negative/missing controls. Include
   priors-mismatched generators rather than only prior-aligned synthetic data.
   Evaluate detection, false positive rates, niche stability, boundary leakage
   and program recovery on spatial blocks and independent patients.

CellCall/CytoSig are related work, not the same objective as spatial functional
subniche discovery; do not claim that LR-to-response prior linking itself is new.
