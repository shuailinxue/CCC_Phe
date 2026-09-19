# Non-negotiable project rule

This repository has one analysis mainline only:

`current-study ST anchor -> BayesPrism bulk -> patient exact directed CCC -> graph-smoothed sparse Cox`

- The spatial transcriptomics dataset being studied is itself the ST anchor that defines
  `sender -> receiver -> LR` direction and `W_ST`.
- Never introduce separate concepts called "reference ST" and "validation ST".
- For another cancer or spatial dataset, that dataset becomes the ST anchor for that run.
- For spot-based ST, use RCTD + COMMOT. For single-cell imaging ST with usable cell
  labels, use the observed cell labels and spatial-neighbor graph directly; if labels
  must be transferred, the transfer must be outcome-blind and fully audited.
- Construct patient CCC propensity as
  `C = W_ST * (M_RNA + epsilon) / (M_ST + epsilon)`.
- Flatten each `(sender, receiver, LR)` entry into one Cox predictor.
- Identify phenotype-associated CCC directly with graph-smoothed sparse Cox.
- Do not insert phenotype-to-spatial-region mapping, CP/NMF programs, communication
  ecosystems, H&E transfer, Xenium validation, or program validation into this mainline.
- Do not run an analysis until its current-study ST anchor, BayesPrism outputs, clinical
  endpoint, feature mask, and sample matching have been audited.
- TLS labels, TLS regions, TLS gene lists, and downstream niche results must never enter
  ST-anchor construction, Cox screening, or hyperparameter selection.
