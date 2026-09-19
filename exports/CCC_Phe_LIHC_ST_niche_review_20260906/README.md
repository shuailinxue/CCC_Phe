# TCGA-LIHC phenotype-associated CCC → ST functional niche review package

This package is intended for method/result review. It does **not** contain the original Visium matrices, images, RCTD objects, or TCGA data.

## Included material

- `code/V0_LIHC_phenotype_CCC.ipynb`: complete V0 notebook; the ST niche extension is appended after the completed bulk/Cox workflow.
- `code/st_niche.py`: independent implementation of protective-CCC spatial activity, normalization, repeat-init NMF, spatial hotspot calling, plotting, and strictly post-discovery TLS validation.
- `code/pyproject.toml`: Python package dependencies.
- `inputs/stable_ccc.tsv`: stable CCC table from the selected Cox model (`lambda_fraction=0.075`, `graph_lambda=0.05`).
- `results/st_functional_niche/`: complete functional-niche output directory.
- `tls_annotation/`: cHC-1L held-out TLS labels from the official SpaPheno package demo and source notes.

## Result map

- NMF K selection: `results/st_functional_niche/nmf_k_diagnostics.tsv`
- CCC loadings: `results/st_functional_niche/program_ccc_loadings.tsv`
- Program composition with sender/receiver/LR: `results/st_functional_niche/program_composition.tsv`
- Spot-level program activity: `results/st_functional_niche/program_activity.tsv`
- Smoothed spot scores and hotspot status: `results/st_functional_niche/program_spatial_scores.tsv`
- Candidate niche summary: `results/st_functional_niche/candidate_niches.tsv`
- Candidate niche spot membership: `results/st_functional_niche/candidate_niche_membership.tsv`
- Program-level TLS statistics: `results/st_functional_niche/tls_program_validation.tsv`
- Niche-level TLS overlap/random-region statistics: `results/st_functional_niche/tls_niche_validation.tsv`
- Discovery audit: `results/st_functional_niche/discovery_manifest.json`
- TLS validation audit: `results/st_functional_niche/tls_validation_manifest.json`

## Current headline results

- 23 stable protective CCCs were selected solely by `sign_selection_probability >= 0.70` and `full_coefficient < 0`; no TLS/cell-type prefilter was used.
- 25,419 RCTD-matched Visium spots across six slices were analyzed.
- Repeat-init NMF selected K=6; K=6 component stability was 0.9017.
- Spatial hotspot calling used the within-slice 90th percentile and a minimum connected-component size of 10 spots, yielding 370 program-specific candidate regions across 6 slices × 6 programs.
- The public held-out TLS labels cover cHC-1L: 212 TLS spots and 4,304 non-TLS spots; 4,500 spots matched the RCTD-filtered discovery matrix.
- P2 was the strongest TLS-associated program: ROC AUC 0.7009, standardized mean difference 0.8271, Mann–Whitney FDR 5.44e-23.
- P2 was dominated by CDH1–ITGAE_ITGB7 routes among endothelial and cancer epithelial sender/receiver combinations.
- The strongest individual candidate niche contained 7 TLS spots among 10 spots and had connected-random-region nominal P=0.0080, but it was not significant after testing all candidate niches (FDR=0.305). Thus the program-level TLS association is supported, while individual niche-level evidence remains preliminary.

## Method boundary

TLS labels were loaded only after `discovery_manifest.json` had recorded `label_free_discovery_complete` and `tls_labels_used=false`. They did not influence protective CCC selection, normalization, NMF K/programs, or hotspot definition.
