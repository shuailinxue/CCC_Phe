# V1002: composition and directed CCC

The current spatial model uses two observed views: `C` (local cell-type composition) and `I` (local directed CCC). Its flat factorization is `C ≈ W_ST H_C` and `I ≈ W_ST H_I`, with six factors (Background plus five niches). The comparison is fixed to `C-only`, `I-only`, and `C+I`. Neighborhood construction, directed CCC feature identity, filtering, seeds, simulation truth, and bulk/Cox protocols remain unchanged.

Run `scripts/run_v1002.py` from this directory with the `cccphe` Python environment. The end-to-end simulation walkthrough is `V1002_end_to_end_walkthrough.ipynb`; the Xenium HBC1 walkthrough is `HBC1_V1002_niche_walkthrough.ipynb`. `pipeline.ipynb` contains the older identifiability diagnostics, now using only C and I views. Previously generated output directories are historical results and should not be interpreted as results from the current two-view code.
