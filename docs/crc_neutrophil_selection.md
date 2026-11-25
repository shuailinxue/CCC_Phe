# Five specimens for exploratory joint niche analysis

## Notebook joint training authorized subsequently

Later user override: stop the ongoing fit and produce results immediately.
The complete saved history is epochs0–5; epoch6 was interrupted during evaluation
and is not reported as completed. Best checkpoint is epoch5 (CCC0.210006,
composition KL0.020899, geometry0.023507, total0.254412). Both best/last checkpoints
are preserved. `scripts/finish_crc_neutrophil_joint.py` validates the exact input
audit/configuration, performs **zero optimizer updates**, exports the saved
checkpoint and resumes the existing atlas runner. Maximum budget10 remains in
the original configuration to retain cache/checkpoint compatibility; actual5
completed epochs and explicit user stop are recorded in completion metadata.
No future run should silently resume to10 after this user stop. Inspect the live
result-generation PID in controller_pid.json/worker.pid and status before acting.

The user subsequently requested adding four code cells to
`Xenium_CRC_Neutrophil_Atlas.ipynb`: selected-five cell maps; joint training for
exactly10 epochs; editable-K KMeans and spatial maps; one same-coordinate UMAP
colored by niche, original cell type and specimen. The original37-specimen atlas
cell and its outputs are retained. Implementation is
`cohorts/crc_neutrophil_joint.py`; it calls the existing locked GNN, direct heads,
target kernels, shared-h geometry, full-cell10-start KMeans and all-cell h UMAP.
No model architecture or mainline training objective was changed.

Physical grouping is **specimen**, not patient: A core and A margin stay separate.
All755638 cells train one model; no invented reference/query split or batch
correction is added. Every epoch logs CCC/composition/geometry/near-middle-far/total
loss, including the unchanged epoch-zero baseline. Exactly10 update epochs run;
the existing trainer restores the lowest training-total checkpoint. This is
exploratory fitting, not independent evaluation or a promise of convergence.

Cell4 contains `K = 12` as an editable default. K changes require only cells4–5;
model and matrix cache signatures exclude K/plot settings. UMAP uses the exact
globally calibrated h96 stored for KMeans, with no row normalization, PCA or
label-guided layout; its content-addressed cache is independent of K. All three
UMAP panels use identical coordinates and draw order. Specimen means actual
specimen, not patient, so the two A specimens have different colors.

Background runner: `scripts/run_crc_neutrophil_joint.py`, physical GPU2. Check
NAS `outputs/crc_neutrophil_xenium/joint_five/{worker.pid,worker.log,status.json}`
and its live worker before restarting. Only the four appended cells execute;
the original37-specimen display is not rerun. The runner persists streams into
the notebook during execution and preserves source edits. It reads the latest
K/UMAP cell source immediately before executing those stages, so a K edit made
while training runs takes effect. Matrix construction is resumable and guarded
against duplicate stages. Completed compatible fits are read without retraining.
All input matrices remain in NAS data; fits/clustering/UMAP/figures in NAS outputs.

Focused integration check: five deliberately overlapping synthetic specimens
never share physical members; an actual2-epoch current-model fit exports compact
h/mixture only; changing K reuses the fit; clustering equals fixed-global
calibrated h; the UMAP adapter receives that same array; all display paths render.

Selection requested2026-10-26. No GNN training, model change, niche relabeling,
or notebook edit is part of this selection. The existing one-cell atlas remains
intact. Configuration: `config/crc_neutrophil_selection.json`.

| Author specimen | Intended role | Cells | Median native counts | Median detected genes |
| --- | --- | ---: | ---: | ---: |
| crc_a_c | Immune-infiltrated core |215437|117|55|
| crc_l_c | Immune-poor core |124331|288|79|
| crc_a_m | Margin A |183618|93|45|
| crc_b_m | Margin B |173109|143|61|
| crc_j_n | Normal mucosa |59143|88|41|

Total755638 cells, four patients, five independent author specimens.
`a_c` and `a_m` are different physical specimens from the same patient.
Always construct physical neighborhoods within specimen_id, including during
joint model fitting. All13 author cell labels and380 measured genes are retained.

## Source and integrity

Author `layers/counts`, `obs/celltype`, `obs/name`, `obs/patient_id`,
`obs/tissue_region`, `obsm/spatial`; source S-BIAD2208, Cancer Cell
DOI10.1016/j.ccell.2025.12.003. Every count is a nonnegative integer. Count-row
sums match author transcript_counts **exactly for all3706544 cells**, providing
an additional alignment check. No cell labels or counts were inferred.

During the full-object download, the native CSR layer was read independently
through checked HTTP206 ranges and SHA-validated completed local segments.
Unfilled parts of the sparse partial file were never interpreted as data.
Full-object completion remains governed by metadata/download_complete.json;
extracting a layer does not certify that the27.3GB download is complete.

## Core contrast

A has8593 T cells, including4138 whose surrounding30 other cells contain at
least50% cancer cells, and2651 with at least80% cancer neighbors. L has1243
T cells, with122 and50 meeting those respective criteria. Mean T fractions
around cancer cells are2.981% and0.171%. These are descriptive screening
statistics, not a change to the30-member model microenvironment.

Depth-normalized T-cell GZMB means are63.41 versus27.80 per10000 panel counts;
PRF1 means are56.59 versus31.67. A was preferred over the initial M candidate
after considering intratumoral localization and these functional measurements.
Neither specimen is assigned a formal author-defined cold/hot diagnosis.

Normal J has preserved gland-like organization and compact tissue coverage,
no zero-count cells, and median88 native counts. Its90th-percentile30-neighbor
radius is33.0um; the small, fragmented normal L has146.5um and was not selected.
J is an author normal sampling region. It still has1.74% cells labeled Cancer
cell by the author; retain and disclose these labels rather than silently
relabeling them or claiming a perfectly cancer-free reference.

## Margin comparison

Both margins have extended coherent tumor boundaries adjoining T cells,
macrophages, fibroblasts, plasma cells and endothelial cells. Among boundary
centers, respective A/B fractions are: cancer57.9/51.9%, T6.3/3.0%,
fibroblast12.5/18.2%, macrophage12.6/16.9%. Matching cell identities, rather than
requiring identical fractions, follows the user's requested comparison.

All214 LR pairs with complete measured ligand/receptor subunits were examined.
Native expression coverage is at least10% in the relevant sending and receiving
cell classes in **both** specimens for reported features. The screen uses the
unchanged ordered-pair CCC kernel:30 members, sigma20um, anchor and pair-distance
decay, no self pairs, sqrt(L*R), normalization by opportunity+1e-4. Tile sampling
does not alter this kernel and does not create a full wide training matrix.

Boundary centers have20–80% cancer neighbors; one geometrically selected center
per100um tile is assessed,1103 tiles in A and1562 in B. For each directed class
pair, report only centers with opportunity>1. The comparison includes zeros;
an absence in one specimen must not be discarded by requiring both to be active.

| Directed measured signal | Native B/A mean ratio | Panel-depth-normalized B/A |
| --- | ---: | ---: |
| Fibroblast→T, CXCL12–CXCR4 |15.30|8.42|
| Cancer→T, CXCL16–CXCR6 |3.04|1.48|
| Fibroblast→Cancer, FN1–CD44 |8.93|3.72|

Both checkerboard spatial halves have the same effect direction, and native and
depth-normalized directions agree. These are exploratory between-patient effect
sizes, **not** biological replicates, significance tests, causal signaling
measurements, or proof of immune activation versus suppression. A broad
composition match does not rule out finer cell-state/subtype differences.

Large900um windows are selected by tumor/non-tumor spatial separation, adequate
cell numbers and presence of T/fibroblast/macrophage classes; LR values do not
choose the windows. They contain6465 and6187 cells. Their local signal audit is
saved separately; the displayed fold-change chart summarizes distributed
boundaries across the specimens, not just these two windows. Published Niche
labels, clinical stage, genotype, response and outcome do not select the
specimens or enter a model. Author core/margin/normal sampling identities are
used as explicitly requested by the user.

## Files and reproduction

NAS directory:
`/home/nas3/biod/xueshuailin/CCC_Phe/workspace/outputs/crc_neutrophil_xenium/selection`.

Selected specimen counts are sparse NPZ files under
`workspace/data/crc_neutrophil_xenium/processed/selection_counts`, with row order
given by display_cells.parquet filtered by specimen_id. Physical screening
arrays are under the sibling selection_neighborhoods directory. Genes are in
the selection output's genes.json.
Quantitative source data and selection.json accompany all figures.

```bash
/home/xueshuailin/miniconda3/envs/cccphe/bin/python scripts/select_crc_atlas_specimens.py --spatial
/home/xueshuailin/miniconda3/envs/cccphe/bin/python scripts/select_crc_atlas_specimens.py
/home/xueshuailin/miniconda3/envs/cccphe/bin/python scripts/select_crc_atlas_specimens.py --boundaries --specimens crc_a_m crc_b_m
/home/xueshuailin/miniconda3/envs/cccphe/bin/python scripts/select_crc_atlas_specimens.py --finalize
```

The first two steps produced audit tables across all37 specimens. Unselected
per-cell screening intermediates and temporary remote blocks can be discarded;
the full author object remains the source for any subsequent reselection.
Existing niche results were never used to select this case. This deliberately
selected development example is not an unbiased performance benchmark.
