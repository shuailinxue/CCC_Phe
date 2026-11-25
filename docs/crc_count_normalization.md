# Consistent LR count normalization for the CRC joint atlas

The user authorized this change and a complete rerun on2026-10-26. The prior
notebook, including all embedded outputs, was copied to NAS before any edits:
`workspace/artifacts/backups/crc_before_count_normalization_20261026_233327/`.
Its manifest verifies a byte-identical SHA256 and records the original fit and
KMeans/Leiden directories. Existing native results remain available there and
in their original content-addressed NAS caches.

For each physical cell, divide the measured380-gene count vector by its full
panel total and multiply by10000. This unlogged CP10k vector defines BOTH the
ordered LR edge input and directed CCC supervision. The denominator is never
computed over the smaller LR-gene subset. Geometric means for complexes and
sqrt ligand/receptor products are applied after scaling. Zero-count rows remain
zero. No sample-specific z-score or batch-integration transform is introduced.

The node branch still uses log1p CP10k and one common per-gene mean/std. The
appended log total-count feature is removed, so the gene input has380 dimensions
instead of381. This does not change h96 or either direct MLP. The raw source
AnnData and native count layer are never overwritten. `raw_lr_expression` is a
legacy internal batch key; in the normalized run its VALUES are CP10k RNA.

Physical30-member graphs, Gaussian distance weights, ordered sender/receiver
identities, no-self-edge rule, opportunity denominator and composition targets
are unchanged. Existing physical arrays are copied with matching SHA256 checks
when source identity and physical-kernel provenance agree. The CCC matrix,
feature moments/RMS, geometry calibration, neural fit, h calibration, KMeans,
UMAP and Leiden all receive new cache identities. The previously requested
Leiden comparison remains a comparison; no other cohort default is switched.

The seven existing notebook cells run in order: original author-cell maps;
selected five maps; normalized matrices and10epoch joint fit; KMeans spatial
maps; exact-h UMAP; count-matched Leiden control; Leiden spatial maps. Epoch
logs are persisted immediately, including CCC/composition/geometry and the
CCC MSE reduction relative to its own mean-prediction baseline. No new analysis
panels are appended. Previous native loss values must not be compared in
absolute units to normalized loss values.

Tests check the full-panel denominator, zero rows, complex LR and directed CCC
invariance to arbitrary positive count multipliers per cell, identical physical
members, normalized graph encoder invariance in streamed and resident paths,
rejection of mixed target/input units, and a full five-specimen synthetic fit.

CP10k is an explicit size-factor convention, not proof of perfect technical
correction. It does not eliminate dropout or gene-specific probe effects and
can remove real panel-total RNA variation. Loss improvement is an empirical
question; no biological niche or plateau improvement is assumed in advance.

The full rerun was launched at23:47 on2026-10-26 (Asia/Shanghai), controller
PID2652570. Before any restart, inspect NAS `outputs/crc_neutrophil_xenium/joint_five/`
files `count_normalization_controller.json`, `active_run.json`, `status.json`
and `count_normalization_run.log`, and verify the live controller/kernel.
The worker holds `worker.lock`, executes every code cell, and persists epoch
streams into the current notebook. Do not launch a duplicate worker.
