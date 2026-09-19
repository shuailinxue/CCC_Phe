# SpaPheno TLS spot annotation

- Source: `Duan-Lab1/SpaPheno`, `inst/extdata/HCC_survival.RData`
- Source URL: https://github.com/Duan-Lab1/SpaPheno/blob/main/inst/extdata/HCC_survival.RData
- Extracted object: named vector `sample_information_region`
- Slice identity: `cHC-1L`, established by an exact 4,516/4,516 barcode match to the downloaded Space Ranger matrix
- Labels: 212 `TLS`, 4,304 `nonTLS`
- Export: `tls_spot_annotations.tsv` with columns `sample_id`, `spot_id`, `tls`

The public SpaPheno demo file supplies labels for cHC-1L only. It must be used only after label-free niche discovery; no TLS score or label is used to select CCCs, NMF K, programs, or hotspots.
