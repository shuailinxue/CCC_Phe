# V1003

Prime 5K breast CCC-only interpretable autoencoder. The scientific input is the
read-only V1002 pairwise directed CCC implementation and its processed Prime 5K
AnnData. V1003 adds no graph, Leiden, tensor model, bulk, or Cox analysis.

Run `XeniumPrime5K_Breast_V1003_niche_walkthrough.ipynb` from top to bottom.
The first code cell builds the V1002-compatible CCC cache and trains the model
only when complete V1003 results are absent. Large caches and results remain
under `outputs/` and are ignored by git.

