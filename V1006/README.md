# V1006

Dual-decoder directed-CCC analysis for Xenium Prime 5K Breast. A nonlinear autoencoder first learns a 32-dimensional embedding. A magnitude head and hierarchical simplex head then produce 32 nonnegative exposures arranged as eight niches with four subprograms each. The model-learned, row-normalized nonnegative H32 dictionary explicitly reconstructs CCC through `Z32 @ H32`, while the nonlinear decoder preserves richer representation capacity. Composition, spatial graphs, Leiden, bulk outcomes and cross-dataset information are excluded from training.

Open `XeniumPrime5K_Breast_V1006_niche_walkthrough.ipynb` and use **Run All**. Existing valid V1003 CCC caches and complete V1006 results are reused automatically.
