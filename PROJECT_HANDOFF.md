# Current handoff: V2 mainline (2026-10-31)

The user selected the CellPhoneDB v5 three-view V2 as the mainline. The stable
model API/config and the exact scope of the completed simulation are documented
in `docs/v2_mainline.md` and `config/models/v2.json`. The V1 account below is
historical, not the current default. Preserve V1 artifacts and baseline results.

The Cellist ST1 RDS has been converted with raw counts checked against nCount_RNA;
one cell without coordinates is excluded. `Cellist_ST1_V2_Communication_Niches.ipynb`
shows annotation, boundaries and coverage; V2 has NOT been trained on ST1.
KNN30 in native coordinates is cached, but no distance weights were applied:
verify the RDS coordinate-to-micrometer conversion before building CCC. Sources
are in `/data1/xueshuailin/CCC_Phe/workspace/data/nsclc_zhencaiwei/raw/`.

---

# Historical handoff: V1 GNN (2026-10-29)

The user explicitly selected V1 GNN and ordered deletion of all other candidate
scheme code. See AGENTS.md for the full locked architecture/training/clustering
contract. Do not revive old schemes from historical conversation or Git.

- Public entry: phenoniche.gnn; config/models/gnn.json.
- Latest user explicitly authorized bypassing networks to test CCC backgrounds
  and signatures directly. Implemented scripts/run_interface_ccc_background.py,
  src/phenoniche/ccc/background.py, config/interface_ccc_background.json.
  Output: outputs/tumor_immune_interface/ccc_background_signatures; inspect
  worker.json/progress.json/run.log/protocol.json before restarting. GPU2,
 32moment+64coherence permutations per observed/baseline-control branch, fixedK6.
  Complete profiles permuted within type/depth bins, same physical30-member CCC.
  Statistical graph connects only windows with disjoint enclosing disks, with
  composition-aware weights. Approximate predictive-t/BH screening and two-block
  stability gate; null-control behavior is essential, no exact FDR claim.
  Correlated positive signatures use complete-linkage, no NMF/deep network.
  Primary scores have no neighbor averaging; an explicitly labelled secondary
  view averages disjoint-neighbor scores. Native CCC-only/joint controls retained.
  Fit never receives known state/template/mechanism lists. Evaluate both frozen
  branches before reading state labels; no truth-based retuning. Reports render
  spatial comparison and signature heatmap. Existing notebooks/V1 unchanged.
- DATA-ONLY PILOT COMPLETED, DID NOT PASS (2026-10-29): observed20/4902 selected
  columns→15signatures; unperturbed baseline14→11signatures. Held-out shuffled
  controls spuriously selected18(observed calibration) and20(baseline calibration)
  columns. Thus approximate statistical-screen calibration is inadequate and
  selected signatures cannot be claimed reliable. A/B ARI≈.00057for CCC-only
  signatures,≈.00051with composition,≈.00032after explicit neighbor averaging.
  Native-window joint view assigns96.26%of paired A/B centers to the same class.
  Only2/15prespecified mechanism columns survived (T→Cancer IFNG–IFNGR2 and
  CAF→T TGFB3–TGFBR2); the latter was also selected in baseline. No neural fit,
  model change, notebook update or truth-driven retuning was performed.
  Preserve the failed pilot/results. User's status query does not authorize a
  silent parameter sweep; next scientific issue is null-calibration validity.
- Simulation was regenerated at the user's request on2026-10-29, retaining the
  exact22,000-cell layout/types/30-member physical graphs. New RNA has3004genes;
  empirical reference profiles and fixed library sizes; paired A/B boundaries935
  centers each, IFNG-associated A versus TGFB/CXCL12-associated B. No fixed
  communication programs. Full atlas8408LR, complete panel1410LR; observed
  detection/contact filtering retains1045LR and4902directed features.
  Data: workspace/data/tumor_immune_interface; physical provenance layout.npz
  and layout_origin.json. Targets: target_cache.json. QC and plots:
  outputs/tumor_immune_interface/biological_validation. See docs/interface_biological_simulation.md.
- Later user instruction "重训 V1 搞起来" explicitly authorizes50-epoch V1 training
  on physicalGPU2, seed40700,K6, followed by compact export,10-initialization
  KMeans and notebook cell2 update. Entry:
  `scripts/run_interface_dual_mlp.py --train --notebook`. Its live losses are
  persisted each epoch. Inspect outputs/tumor_immune_interface/v1_gnn/
  notebook_status.json,progress.json,run.log,launcher.json before any restart.
  Baselines remain read-only; do not launch them. The existing old-expression fit at
  outputs/tumor_immune_interface/gnn_geometry_exploration/v1_baseline and old
  baseline runs describe OLD RNA, are preserved and must not be replayed as new
  results. Source-checked entrance scripts/run_interface_dual_mlp.py uses
  outputs/tumor_immune_interface/v1_gnn only after explicit --train. Locked V1,
 50epochs, seed40700,K6 unchanged. Notebook first cell shows new design/QC;
  cell2 is now the authorized training/result entrance, cell3 retains the old-source
  baseline warning until a separate benchmark rerun is requested.
- The new V1 run completed50/50 epochs; best epoch50, CCC Huber .268398→.130703,
  scaled MSE reduction vs constant mean55.13%. Current K6 mixes A/B boundaries.
  Read-only audit `scripts/audit_interface_signal_geometry.py` authenticates this
  source/target/checkpoint and writes v1_gnn/signal_geometry_audit, without training,
  reclustering or notebook changes. Observed A/B joint squared centroid gap .012589,
  core-versus-A gap1.050659 (~83x); h gaps .007292 and1.096337 (~150x).
  Both observed CCC and composition blocks have global variance1. A/B-related
  perturbations involve237/4902 typed CCC columns (4.83%). Frozen head shrinks
  IFNG A/B2.264→1.055, TGFB B/A2.016→1.191, CXCL12 B/A1.643→1.033.
  Thus key-axis QC passed but full-space separation was weak and learned recovery
  further attenuated it. This does not establish the cause of real-cohort failures.
  Current user asks for diagnosis only: do not retune/retrain or change the locked
  model, K, simulation or real notebooks without a new request.
- Requested all-CCC heatmap: `scripts/plot_interface_all_ccc_heatmap.py`; output
  v1_gnn/signal_geometry_audit/all_CCC_heatmap. All4902 columns shown in7 wrapped
  strips (Amean,Bmean,A−B,pooled within-SD), same training-RMS units and color
  scale, no column z-score/clipping;15 prespecified axes enlarged. Source CSV
  retains native and RMS units plus all feature IDs. Median absolute mean shift
  .02499 versus median pooled within-SD .99877;55/4902 shifts exceed0.5within-SD,
  only2 exceed1SD. This describes within-region heterogeneity, not evidence that
  A/B began with different background donor profiles. No notebook/model changes.
- COAD: original nine-cell notebook and V1 outputs preserved.
- CRC: Xenium_CRC_Neutrophil_Atlas.ipynb restored to saved V1 five-epoch results,
  NAS outputs/crc_neutrophil_xenium/joint_five/fits/978e32b08d3e3525.
- AE/MultiViewAE notebooks and their implementation/config/test/launcher code are
  removed. External baselines/NMF controls, source data, current matrix caches and
  weights remain. No cohort was retrained as part of cleanup.
- Exact cache aliases live in config/v1_cache_compatibility.json; no dataset,
  config or numerical-model mismatch is waived.
- All generated artifacts under /home/nas3/biod/xueshuailin/CCC_Phe/workspace.
  Preserve user-requested backups and completed baselines. No routine copies.

## Supporting TNBC preprocessing

PhenoMap upstream is read-only. Production selection is93 patients (one sample
per patient) out of94 patient IDs/280 tissue directories; patient34 deferred.
Selection is quality-based, never outcome-based. Existing workers are authorized
on physicalGPUs1/2, disjoint samples, one worker per GPU. Inspect live processes
and NAS production manifests before resuming; do not restart or reselect blindly.
The original full project backup/cleanup manifests remain under NAS
artifacts/backups and artifacts/workspace_cleanup_20261025_report.json.

## TNBC preparation retained

TNBC PhenoMap is supporting preparation, not the overall niche project and not a completed clinical association study. Upstream `/home/xueshuailin/PhenoMap/` is read-only. Adapter lives in CCC_Phe/src/phenoniche/phenomap_adapter. Input root is `/home/nas3/biod/xueshuailin/New_Projects/data/HBC/TNBC_ST/Processed_Samples`; reference is sibling `2021_breast_cancer_atlas.h5ad`.

There are94patient IDs,280sample directories,93frozen selected patients with one sample each; patient34 deferred for quality. Output root `workspace/outputs/phenomap_tnbc/`. Selection manifest `selection/selected_samples.csv`, candidate scores `selection/candidate_scores.csv`, policy `selection/policy.json`. Selection is outcome-blind, based on native H&E/nuclear visibility, spot/LR quality, tissue coverage and representativeness; do not reselect or force missing classes. H&E QC supports plausibility, not validated segmentation accuracy.

Configuration config/phenomap_tnbc_batch.json and scripts/run_phenomap_tnbc.py, run_phenomap_tnbc_dual.py, select_phenomap_tnbc.py are preserved. No old PID is evidence that work is still running. Do not stop unrelated jobs or launch duplicates; check live status. User's current explicit GPU instruction takes precedence over historical migrations. Other GPU work defaults to physicalGPU2.

Inferred expression is not measured UMI, and Softplus>0 is not detection. Existing panel1079common genes covers2926complete LR pairs. Spot image coordinates require micrometer calibration. User accepted approximate150um nearest-spot-pitch calibration: median.283635um/pixel, not scanner-measured scale; does not correct local anisotropy. Keep calibration provenance and do not silently substitute Visium geometry: these are older ST spots.

User rejected the original lineage classifier as reliable annotation and accepted SingleR after pilot, then authorized93-sample annotation and CCC rebuild excluding Unknown. Keep scripts/run_tnbc_singler_ccc.py, cell_annotation.py and preparation code. Sidecars: `workspace/outputs/tnbc_communication_phenotype/singler/annotations/<sample>/annotations.csv.gz`; config/log under `singler/`. Original expressions/annotations are not overwritten. Fine cancer/normal/uncertain epithelial types must not be collapsed to a single epithelial sender/receiver. Unknown participates in physical context only, never typed CCC or composition supervision. Do not manufacture B/endothelial proportions; pilot had candidate B recovery but unresolved endothelial calls, and reference holdout accuracy does not validate inferred TNBC cells.

The old TNBC and spot neural trainers/notebooks were removed during cleanup; preparation/deconvolution, saved results and downstream association utilities remain. No GNN or association fit is currently authorized by this handoff. DISSECT/HRA009001 acquisition was cancelled and deleted: do not resume/download it.

## Additional read-only CCC diagnostic (2026-10-29)

- Saved64 null draws tested leave-one-out: every randomized draw selects features; observed-branch median13.5(range5–31), baseline-branch median14(range4–30). Approximate-t screen is invalid for discovery here. See ccc_background_signatures/null_leave_one_out_audit.json.
- Supervised detectability ONLY: fixed15 prespecified mechanism axes, five140µm paired spatial-block folds, purge any training window sharing a physical member with test, A/B counterpart templates in same fold. Fixed logisticC1, no tuning. Held-out pooled AUC .8541, balanced accuracy .7872. Four prespecified family means give AUC .8788, accuracy .7984. This deliberately uses known A/B/axes as diagnostic; NEVER call it blind discovery or use its labels to select features. Report: ccc_background_signatures/detectability_diagnostic.json.
- Suggested next data-only test is externally annotated pathway aggregation of ALL mapped candidates, preserving sender→receiver direction and checking unannotated/single-axis coverage, plus empirical-null calibration before blind clustering. This is a discussion recommendation ONLY, not an approved model/target replacement or launched experiment. V1 and all notebooks unchanged.

## Composition-first conditional CCC simulation experiment (2026-10-29)

- User requested the composition-first, within-coarse CCC subdivision on the existing simulation. This is an isolated data-only experiment; V1 and notebooks are unchanged. Entry scripts/run_interface_coarse_ccc.py, settings config/interface_coarse_ccc_experiment.json, methods docs/interface_coarse_conditional_experiment.md. Six planted states with A/B merged suggest K=5 for composition and K<=6 after one optional statistically selected split; no other planted information enters fitting.
- NAS outputs/tumor_immune_interface/coarse_conditional_ccc; launcher PID2135615, GPU2. Inspect worker.log/progress.json/complete.json before restarting. Observed and unperturbed baseline each get 64 null standardization, 256 max-stat calibration and 32 holdout randomizations. Both independent fits finish before reading truth; no feature selected means no forced split. Initial startup and tests/test_coarse_conditional.py plus tests/test_ccc_background.py passed. Estimated run about 9 minutes, but consult live state; do not claim a result before completion or claim global-null calibration proves partial-null FDR.

## Composition-first communication network pilot (2026-10-29)

- Explicit user request authorized a fast network experiment on the current simulation, isolated from V1. Completed successfully; entry scripts/run_interface_coarse_ccc_network.py, config/interface_coarse_ccc_network.json, details docs/interface_coarse_ccc_network.md, NAS outputs/tumor_immune_interface/coarse_ccc_network. No current-model or notebook change.
- Composition K5 frozen; each coarse class has a 4902→192→32 encoder and 32→192→4902 decoder trained 30 epochs on native conditional directed CCC residuals to predict disjoint nearby same-coarse CCC residual means. It is an MLP with spatial-neighbor prediction, not a GNN, not a fixed program/pathway implementation. No labels/mechanisms enter either observed or independently trained baseline. K10 forces K2 per coarse; K6 selects one split by unsupervised h variance reduction. Forced partitions are not significance tests.
- Both branches finished. K6 A/B ARI -0.00040, 100% corresponding pairs same; K10 ARI -0.00071, 99.25% same. Baseline also split similarly. K6 splits coarse4, predominantly shared tumor core, not A/B interface. In post-fit coarse3 (709 A and 709 B), h A/B squared mean distance / within-region dispersion .04125; unperturbed baseline .000654. A/B signal remains weak relative to dispersion and Kmeans did not prioritize it; not independent validation. Figure spatial_results.png; no notebook changes.

## Green C3 only clustering (2026-10-29)

- User specifically selected green composition class C3. Reclustered its 3384 existing class-specific 32D communication embeddings for K=2,3,4,6 with seed40700 and ten initializations, no retraining or truth in fitting. Entry scripts/analyze_interface_green_coarse.py, NAS outputs/tumor_immune_interface/green_c3_analysis. A and B each have 709 C3 centers. Observed matched-pair agreement K2=.995769, K3=.988717, K4=.991537, K6=.988717; corresponding AB ARIs remain approximately zero (slightly negative). Unperturbed baseline shows the same pattern. These are exploratory forced partitions; no outcome-guided K selection. The green C3 also includes the lower infiltrated tumor core and interface. Fig green_C3_clustering.png.

## Composition K3 exploratory correction (2026-10-29)

- User flagged unwanted outer boundary in the composition coarse classes. Post-fit audit outputs/tumor_immune_interface/coarse_conditional_ccc/composition_boundary_postfit_audit.json: 178 C1 centers marked context lie median43.46µm outside the contour, versus the ±40µm generating boundary, but average 45.8% of their 30 members are interface-labeled; local mean cancer fraction 23.4%. This is a transition across the center-defined truth boundary. C1 A/B centers n412 have cancer fraction33.2%, while C3 A/B centers n1418 have cancer fraction50.7%. C3 also includes n1132 infiltrated core with cancer55.2%, T25.6%, close to C3 A/B cancer50.7%, T25.0%. Truth labels cannot demand separation based solely on nearly identical composition.
- My earlier assumption that 6 generating states minus the A/B distinction implies 5 distinct composition clusters was incorrect. A post-hoc diagnostic of K3/4/5/6 found composition K3 puts 914/935 centers from each A/B boundary into the same mixed class, with only 237/16400 context centers in that class. K3 was chosen after observing simulated truth and is exploratory, not a blind benchmark or universal K rule.
- At K3, same 30-epoch per-coarse conditional CCC network and independent unperturbed control completed under NAS outputs/tumor_immune_interface/three_coarse_ccc_network, entry scripts/run_interface_three_coarse_ccc.py, config/interface_coarse_ccc_network_three.json. K4 view makes one unsupervised forced split; K6 view forces both subclasses in each coarse. A/B paired same K4=99.893%, K6=98.824%; AB ARI about zero. Controls similar. K3 improves coarse boundaries but does not solve communication splitting. Picture spatial_results.png. No simulation data altered. Existing prior saved outputs retained; network runner generalized to K3 but V1 unchanged.

## Directed CCC NMF and pathway/response proposal (2026-10-29)

- User requested real literature on effect magnitude, an NMF diagnostic on computed CCC, and a final program/response proposal referencing CellCall and CytoSig. Full report docs/ccc_functional_program_strategy.md. V1, inputs and notebooks unchanged. NMF data are observed CCC from simulation (not patient data), in a composition K3 class selected post-hoc, 4275×4902, 914 A/B each. Rank32, 500 Frobenius multiplicative updates, K2/3/4/6, native and feature-RMS input; independently fit unperturbed negative control. No truth/mechanisms in factorization or primary clustering; A/B-only K2 is explicitly labeled oracle. Scripts experiment_interface_ccc_nmf.py, experiment_interface_ccc_nmf_sensitivity.py; GPU0 used as explicitly session-authorized all-GPU availability, outputs NAS outputs/tumor_immune_interface/ccc_nmf_program_pilot.
- Rank32 RMS-scaled reconstruction reduces MSE vs per-feature constant mean by27.46%, but A/B paired K2 labels agree100% (native96.50%); A/B-only oracle also fails. Input IFNG family A/B2.263 but NMF reconstructed1.00094; TGFβ B/A≈2.02 becomes≈1.054. Post-hoc rank16/64 × two seeds still fails; rank64 MSE .756 vs rank32 .929, yet IFNG ratio≈.99. This supports that this NMF setup loses a targeted signal but does not prove all NMF or low-dim methods fail. Full numeric tables and plot under NAS.
- Official CytoSig centroid commit ce53cc7f496f996dc0f759652e21afcd0f20bb8e: simulated panel has1062/4881 signature genes; IFNG 49/100 strongest genes, 30.3% full absolute weight, TGFB1 54/100, 27.8%. Availability alone does not validate score. Current combined atlas flags182/1045 unique simulated effective LR as CellChat-v2 source (694/4902 directed typed columns); not a full mapping audit. Real panels were not assessed. Script audit_simulation_cytosig_coverage.py and coverage outputs.
- Suggested proposal (not yet implemented): composition coarse classes; per pathway directed LR exposure from validated priors plus sender/receiver and contact/opportunity, receiver signed transcriptional response via CytoSig/CellCall/TF prior with coverage masks and uncertainty; directly compare calibrated program evidence first, then network with small program/response heads if useful. Keep individual axes and evidence attribution, evaluate independent samples, decouple held-out response genes/orthogonal experiments to avoid circularity. No claim that this guarantees success or establishes causal signaling. No explicit authorization to replace mainline implementation.
