# V1004: independent composition and CCC niches

The active Xenium HBC1 analysis is `HBC1_V1004_niche_walkthrough.ipynb`. It uses the local `src/` copy of the pairwise CCC implementation and canonicalized NMF. Composition and directed CCC are fitted independently with K=8, seed=40700, and the existing neighborhood, filtering, and iteration settings. A Hungarian permutation aligns I labels to C labels using cell overlap. Every observed aligned `(C, I)` assignment receives a deterministic Final Niche ID; no shared-W fit or additional clustering is used.

`src/phenoniche/v1004/late_fusion.py` contains the alignment and observed joint-state numbering. Other copied V1002 notebooks and scripts are historical and are not part of the active V1004 HBC1 workflow.
