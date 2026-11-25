# Current project direction (2026-11-25)

The only current model is the former spatial V2, now `phenoniche.spatial`, with CLI `scripts/run.py` and documentation `docs/spatial.md`. Do not create versioned model interfaces. Former V1 and compatibility code are archived on NAS at `/home/nas3/biod/xueshuailin/CCC_Phe/Backup/retired_version_interfaces/`. Preserve existing notebooks and historical results; old paths/imports may require migration. NAS is organized under data, results, Backup, baselines, env and LR_base. Avoid generating new files or experiments unless needed or requested.

# Historical project instructions (superseded where conflicting)

# Project mainline: V1 composition and functional communication (formerly V3)

Latest explicit user decision, 2026-11-16: rename the previous V3 to V1 and
archive the previous V1, V2 and V4 implementations on NAS at
`/home/nas3/biod/xueshuailin/CCC_Phe/workspace/artifacts/retired_models_pre_renumbering/`.
Current V1: `phenoniche.v1`, `scripts/run_v1.py`, `config/models/v1.json`,
`docs/v1_mainline.md`. Clustering modes: composition, h, concat, spectral and
joint; joint is the unchanged former V3 fixed-K graph/iterative algorithm.
Legacy `phenoniche.v3` is a compatibility alias; gnn/v4 and historical v2
modules load archived historical source. The current spatial V2 is independent of V1: `phenoniche.v2.spatial`,
`scripts/run_v2.py`, `docs/v2_spatial.md`. Keep historical callers working.
The previous experimental V2 is
`phenoniche.v2.ReferenceNiches`; see `docs/v2_mainline.md`. Its control-reference
matching is distinct from historical V2 and does not replace V1. Preserve existing notebooks, inputs,
checkpoints, historical results, baseline outputs and their historical names.
No split/merge or communication-center update is part of current V1.
Its adaptive constants have not been independently validated.

## Scientific scope and historical V1 details

- Study functional niches jointly defined by local cell composition and directed
  communication, including similar mixtures with different communication states.
  Disease/clinical phenotype differences are downstream niche associations.
- The abandoned ST-anchor/BayesPrism/bulk-CCC/Cox workflow is not the mainline.
- Historical V1 uses the directed cell-edge joint encoder in
  `src/phenoniche/gnn/dual_heads.py`: persistent ordered LR edge states, separate
  incoming/outgoing messages, shared h96 and two direct MLP prediction heads.
  Predict native directed CCC and composition probabilities. No ZH/PB, NMF
  initialization, learned loading, parallel post-pooling LR summary or cell-state
  shortcut. Cell types define targets and interpretation, never encoder input.
- For V1 and the current simulation preserve the audited physical definition: center plus nearest cells, 30 total
  members, sigma20um distance weights, distinct directed physical cell pairs and
  opportunity-normalized CCC. Every message stays within its supplied local graph.
- In historical V1 retain CCC Huber, composition KL and multiscale joint-distance Huber on shared h.
  Reference distances use observed scaled relative CCC plus weak native total
  (.25), sqrt composition and separately RMS-calibrated blocks. Calibration and
  batch references use training rows only. Near/middle/far bands, eight pairs per
  anchor per band, equal band weights, denominator floor .05. Default geometry
  weight1. Calibrate one fixed global h affine scale on8192 initial training cells;
  never normalize h rows to unit length.
- In historical V1 use full-cell KMeans directly on calibrated h, ten initializations. No SVD, output
  sketch, posthoc representation mapper, automatic Leiden substitution or spatial
  label smoothing. K and training budgets remain explicit per experiment.
- For historical V1 print CCC/composition/distance/near/mid/far/total every epoch; persist notebook
  streams during long jobs. Export compact h and composition, not a wide CCC
  prediction matrix. Interpret selected CCC predictions on demand with
  `gnn/predictions.py`; this reader is not a clustering algorithm.
- LR normalization is dataset-specific and must match targets and graph-edge
  inputs. CRC/simulation use panel-total CP10k and exclude native library size.
  Preserve explicitly documented historical COAD native-count settings/results
  until the user authorizes their change. Do not silently rebuild cohort caches.
- Known niches, simulation truth and clinical outcomes enter only downstream
  evaluation/association. They must not select features, tune distances or supply
  training positives. The nominated single-T example is downstream evaluation.

## Entrances and saved results

- Later explicit user request (2026-10-29) authorizes a DATA-ONLY CCC experiment:
  conditional whole-RNA-profile background, stable feature screening, signatures
  and fixedK6 direct clustering, without a neural network. Entry
  `scripts/run_interface_ccc_background.py`, config/interface_ccc_background.json,
  NAS outputs/tumor_immune_interface/ccc_background_signatures. This is an
  isolated historical statistical experiment, NOT a replacement of current V1.
  Never use A/B, planted states or hand-selected IFNG/TGFB axes during fitting.
  Evaluate after freezing both observed and unperturbed-baseline control fits.
  Preserve all V1/cohort notebooks/results. Check worker/progress/protocol files
  before restarting. GPU2. Native-window signatures are primary; disjoint-neighbor
  score averaging is a separately labelled diagnostic, never hidden smoothing.
- Current V1 reusable API: `phenoniche.v1`; CLI `scripts/run_v1.py`; config `config/models/v1.json`.
  Former V3, V2, V4 and original V1 imports are compatibility aliases; historical
  source/config lives in the NAS archive described above. Historical outputs keep their original names.
- The user requested biological resimulation on 2026-10-29: exact existing22,000
  cell layout/types/physical graphs retained. `Multifocal_Communication_Simulation.ipynb`
  now displays the new empirical-reference RNA simulation and composition/CCC QC.
  The935 paired A/B edge centers share composition, contact geometry and library
  sizes, with prespecified IFNG-associated versus TGFB/CXCL12-associated states.
  Use the FULL project LR atlas, complete subunits and current real-data filtering,
  not the former capped/disjoint75-candidate,24-injected-axis generator. No fixed
  communication programs or learned loadings. See docs/interface_biological_simulation.md.
  The later explicit request "重训 V1 搞起来" authorizes the V1 run below;
  baseline reruns remain unauthorized.
- Simulation V1 entrance: `scripts/run_interface_dual_mlp.py`; unchangedK6,
  seed40700,50epochs, current target cache in data/tumor_immune_interface/target_cache.json.
  The authorized50-epoch GPU2 run automatically exports compact h, clustersK6
  and updates notebook cell2; losses stream into this cell every epoch.
  New fits use outputs/tumor_immune_interface/v1_gnn. Old-expression V1 and
  baselines are preserved at their original NAS paths but are source-incompatible;
  do not replay them on the new RNA. Readers enforce source checksums.
- `SPATCH_COAD_Communication_Niches.ipynb` retains its original nine cells/results.
  Do not relabel or rerun existing cohort notebooks just because code is cleaned.
- `Xenium_CRC_Neutrophil_Atlas.ipynb` reads the completed five-specimen V1 fit
  `outputs/crc_neutrophil_xenium/joint_five/fits/978e32b08d3e3525`: five completed
  epochs, original checkpoint budget10; no automatic retraining. K remains explicit.
- Retain CosMx NSCLC, CosMx EarlyCRC, breast5K, lung5K, HealthyCancer and gene-rich
  simulation notebooks. Migrated historical notebooks have RUN_TRAINING=False,
  with clearly marked historical displays. Historical plots are not new V1 fits.
- EarlyCRC: use11 specimens independently for physical graphs, not slide names.
  Retain24 author cell labels; x_slide_mm/y_slide_mm multiply1000 for micrometers.
  Published neighborhoods/embeddings and clinical stages never enter fitting.
- External baselines and NMF controls are retained. Controls live in `baselines/`
  and `src/phenoniche/controls/`, separate from the current V1 mainline. Do not delete
  completed baseline runs or restart them. The interface seven-method benchmark
  is physicalGPU0, fixedK6; inspect its controller/status before any restart.
- `config/v1_cache_compatibility.json` contains exact whole-protocol aliases for
  pure source cleanup. It preserves compatible saved matrices/fits after deleting
  unused classes. It is not permission to bypass changed data/config/numerics.

## Editing, storage and execution

- Edit current project files directly. The user manages Git. No version trees,
  duplicate working copies or routine backups. Back up only when explicitly asked.
- User storage update (2026-10-30): while NAS is full, write new generated
  outputs/checkpoints/matrices/processed datasets/large temporary files under
  `/data1/xueshuailin/CCC_Phe/workspace/`, mirroring NAS subpaths `outputs/`,
  `data/`, `artifacts/`, and `tmp/` for later migration. Existing inputs and
  saved results remain on NAS and should be read from their original paths.
  No symlinks. Code/notebooks/config/small reference tables stay in this repository.
- Preserve source data, physical matrices, current checkpoints and user-requested
  backups. Deleting alternative code does not authorize bulk data/result deletion.
  Do not recreate deleted notebooks through historical report generators.
- For removed historical datasets consult
  `workspace/artifacts/workspace_cleanup_20261025_report.json`; do not redownload
  or regenerate without a new request. Breast caches `data/ccc_expr10/cache` and
  `outputs/microenvironment_gnn_nmf_expr10_5epochs/cache` remain dependencies.
- Other GPU work defaults to physicalGPU2 unless specified. Never stop unrelated
  jobs, launch duplicates or retrain merely after a handoff. For an authorized long
  job verify startup, give a concise Chinese update and end the turn.

## Supporting TNBC preparation

- Read PROJECT_HANDOFF.md before TNBC work. PhenoMap prepares later research and
  is not a completed cohort association analysis. `/home/xueshuailin/PhenoMap/`
  is read-only; implement adapters only in this repository.
- 94 patient IDs,280 tissue directories,93 selected patients with one sample each;
  patient34 deferred. No outcome-based sample selection. Evaluate native H&E,
  nuclei/segmentation, spot/LR quality, coverage and within-patient representativeness.
- Existing TNBC preprocessing authorization is GPUs1/2, one worker per GPU,
  disjoint samples. Preserve compatible checkpoints and production selection.
