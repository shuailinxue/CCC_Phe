# CRC neutrophil spatial atlas: source audit and selected specimens

2026-10-26: user selected this dataset and explicitly authorized downloading it
and displaying its author cell types in a notebook. No CCC construction or model
training is authorized by this request. Preserve existing project defaults and
notebooks.

## Authorized download and spatial overview

- One-cell notebook: `Xenium_CRC_Neutrophil_Atlas.ipynb`.
- Downloader/controller: `scripts/download_crc_neutrophil_atlas.py`.
- Reader/display: `src/phenoniche/cohorts/crc_neutrophil_atlas.py`.
- NAS input: `/home/nas3/biod/xueshuailin/CCC_Phe/workspace/data/crc_neutrophil_xenium`.
- NAS figures: `/home/nas3/biod/xueshuailin/CCC_Phe/workspace/outputs/crc_neutrophil_xenium`.

The controller downloads only the 27.3-GB author H5AD, plus small study metadata;
the redundant 39.2-GB SpatialData archive and raw images are not necessary.
Eight HTTP range workers write disjoint 64-MB segments to the NAS partial file.
Progress is measured from completed segments, **not** the sparse partial file's
apparent length. Per-segment SHA256 receipts support resume and verification;
on completion the whole file receives a SHA256 and required HDF5 fields are
checked before the partial file is renamed. The published listing provides size,
not an independent published checksum; do not call the local SHA a source MD5.

In parallel, a bounded HTTP-range HDF5 reader extracts only author cell types,
patient/specimen/region IDs and row-aligned physical coordinates to a compact
Parquet display cache. It never reads the expression matrix for the preview.
This allows the notebook to be executed before the full H5AD download finishes.
The preview cache's audit records whether full expression was available at that
time; authoritative transfer completion is `metadata/download_complete.json`.

Each plot panel is an independent original specimen, not a merged patient or
slide. The four-column gallery is paginated at 12 specimens per figure, within
one notebook cell. All author cells are displayed with shared cell-type colors;
the author core/margin/normal sampling labels remain visible. No inferred cell
labels, new niches, functional-state labels or spatial smoothing are introduced.

## Primary publication

Marteau et al. **Single-cell integration and multi-modal profiling reveals
phenotypes and spatial organization of neutrophils in colorectal cancer**.
Cancer Cell 44(1):146–165.e14, 12 January 2026.
DOI: https://doi.org/10.1016/j.ccell.2025.12.003
PubMed: https://pubmed.ncbi.nlm.nih.gov/41529665/

Human Xenium spatial profiling: 15 patients; tumor cores, invasive margins and
adjacent normal tissue sections. The publication describes 3.7 million spatially
profiled cells; do not confuse these with the separate 4.27-million-cell scRNA
reference atlas, or the approximately 0.7-million-cell protein imaging cohort.
The biological study includes neutrophil functional programs and IL-1-based
neutrophil–fibroblast interaction, supported by organoid/experimental work.
This is independent biological support, not yet proof of a nominated pair of
large tumor–immune boundaries with the requested communication contrast.

## Verified public processed data and labels

- BioImage Archive: https://doi.org/10.6019/S-BIAD2208
- Project website: https://crc.icbi.at/
- Author analysis: https://github.com/icbi-lab/crc-atlas/tree/main/analyses/60_Xenium_spatial
- Study API: `https://www.ebi.ac.uk/biostudies/api/v1/studies/S-BIAD2208`
- File list: `https://www.ebi.ac.uk/biostudies/files/S-BIAD2208/xenium.json`

The API file list was read, not merely inferred from the paper:

| Path relative to BioStudies files endpoint | Bytes |
| --- | ---: |
| `xenium/processed/crca_xenium.h5ad` | 27,337,886,004 |
| `xenium/processed/crca_xenium.zarr.zip` | 39,155,780,344 |
| `xenium/processed/xenium_explorer_annotations.csv` | 86,367,695 |

An HTTP Range read of the annotation CSV confirmed the actual header
`cell_id,group` and author labels including Cancer cell, Epithelial cell,
Macrophage, Fibroblast, T cell, B cell, Plasma cell, Neutrophil, Dendritic cell,
Endothelial cell, Smooth muscle cell, Schwann cell and Mast cell.

The executed author `05_results.ipynb` explicitly loads `crca_xenium.h5ad` and
uses `obs['celltype']`, `obs['patient_id']`, `obs['tissue_region']`, physical
`obsm['spatial']`, `layers['counts']` and `layers['norm']`.
Thus usable native counts and author annotations have an identified public
processed-object entry. The H5AD metadata and native count layer have now been
inspected by validated HTTP ranges; whole-object transfer runs separately.
Prefer the object's existing cell/sample IDs for alignment; the standalone CSV
does not contain an explicit sample column in its header, so verify barcode
uniqueness before joining it to separate instrument outputs.

`tissue_region` distinguishes core/margin/normal sampled tissues. They are
separate specimens/regions, not a claim of one contiguous whole-organ section.
Author code merges specimens into a display layout and shifts coordinates;
future physical neighborhoods MUST stay within the original specimen. Never
connect across display tiles or use published Niche labels/clinical outcomes in
training. Existing author Niche labels may support downstream evaluation only.

Author Fig. 5 code examines neutrophils and fibroblasts across niche contexts,
including the panel genes CXCL1, CXCL3, CXCL5, IL1A, IL1B and IL1R1. Inspect the
original margin maps before nominating a boundary example or downloading all
processed data. Do not claim the exact requested example is already established.

## Additional verified but lower-priority alternative

SIMVI CosMx melanoma: https://doi.org/10.1038/s41467-025-58089-7
https://zenodo.org/records/14708000
`Melanoma_5612.h5ad`: 478,277,043 bytes. In-memory HTTP Range/HDF5 metadata
inspection (~3 MB; no retained dataset) confirmed `obs['celltype']`, 17 author
types, physical `obsm['spatial']`, 25 patient IDs, immune-checkpoint treatment,
response and survival fields. Published analysis: 56,761 cells x 960 genes.
It uses 0.6-mm TMA cores, which limits suitability for the user's large,
continuous interface examples. The parent TMA cohort of 60 patients is not the
25-patient spatial analysis. Clinical outcomes and author niche labels must not
enter model fitting. No bulk dataset was downloaded.
