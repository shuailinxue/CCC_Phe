# Biological expression simulation on the frozen three-tumor layout

The 2026-10-29 request changes simulated RNA, preserving all22,000 cells, their
coordinates, five type labels, region membership, original RNA library sizes,
30-member physical graphs and distance weights. This is development data.
The V1 GNN, its loss and K6 clustering are unchanged. The later user request
"重训 V1 搞起来" authorizes a50-epoch GPU2 fit and automatic notebook cell2 update.

## Generation

`src/phenoniche/simulation/interface_tissue.py` freezes geometry in NAS
`data/tumor_immune_interface/layout.npz`, recording the previous source hash in
`layout_origin.json`. Two rotated boundaries A/B each contain935 paired centers;
their physical neighborhood compositions and contact opportunities match up to
floating point precision. Both tumor cores retain the same reference RNA state.
The third tumor retains its original infiltrated cell mixture.

`simulation/biological_expression.py` bootstraps1000 observed cells per type from
the existing human breast cancer reference `2021_breast_cancer_atlas.h5ad`.
Within-type RNA profiles use15% mean shrinkage to avoid making every biological
axis depend on a single observed dropout. A/B share donor profiles, multinomial
random streams, per-cell libraries and effect heterogeneity. For cell i and gene g:

    p_ig = .85 * reference_cell_profile_ig + .15 * type_mean_profile_g
    p'_ig ∝ p_ig * fold(type_i, state_i, g) ** (taper_i * lognormal_effect_i)
    X_i ~ Multinomial(original_library_i, p'_i)

Unspecified genes have fold1. Probabilities are renormalized; compositional
sampling can change other observed counts without changing per-cell total RNA.
Both baseline and perturbed counts are saved. Baseline A/B counts match exactly;
paired core/context final counts also match exactly. State effects taper to zero
over the last10µm at the original interface margins. No new expression sheath
or cell-type ring is added.

All genes and folds were fixed in `config/interface_biological_expression.json`
before any new fit or cluster evaluation:

| Region | RNA state assumptions | Mechanistic motivation |
| --- | --- | --- |
| A interface | T-cell IFNG/cytotoxic expression and epithelial IFN response higher | IFNγ-associated T-cell inflammation |
| B interface | CAF TGFB1/3, CXCL12 and stromal-response genes higher | TGFβ/CXCL12-associated stromal immune exclusion |
| C core/interface | Intermediate IFN-associated state in the original infiltrated mixture | Infiltrated counterpoint with different composition |

These states can coexist. Neither expression-derived LR potential nor these
synthetic labels establish actual signaling, clinical response or patient biology.
The reference provides realistic RNA background; it is not claimed to be a
measured matched pair of such tumor boundaries. Numerical folds are simulation
assumptions, not literature effect estimates.

## Literature basis

- [Ayers et al., JCI2017](https://www.jci.org/articles/view/91190): IFNγ-associated
  inflammatory RNA profiles, cytotoxicity, antigen presentation and adaptive
  resistance motivate A. CD274 is allowed to increase with inflammation.
- [Tauriello et al., Nature2018](https://www.nature.com/articles/nature25492):
  stromal TGFβ and T-cell exclusion/immune evasion motivate B.
- [Mariathasan et al., Nature2018](https://pmc.ncbi.nlm.nih.gov/articles/PMC6028240/):
  fibroblast TGFβ-response and excluded T cells motivate stromal expression.
- [Feig et al., PNAS2013](https://pmc.ncbi.nlm.nih.gov/articles/PMC3864274/):
  CAF-derived CXCL12 and immune exclusion motivate the additional B axis.

## CCC construction and independent data checks

`ccc/observed_targets.py` uses the existing physical pair kernels and the same
real-cohort rules: full8408-LR atlas, every complex subunit measured, per-type
10% detection and physical opportunity in at least max(20,1% of centers).
Native counts are normalized to panel-total CP10k. Each directed feature is
the weighted mean sqrt(ligand_sender * receptor_receiver) over distinct physical
cell pairs, with the established opportunity denominator. Composition remains
the original physical-pair degree fractions. Effective features require0.5%
nonzero coverage and nonconstant variance. Each feature uses the existing RMS
conditioning with a0.1*median-RMS floor.

The panel retains the original3000 genes and adds four prespecified missing
genes: CD274,TAP2,TGFB2,TGFBR1. There is no LR quota, disjoint-gene requirement,
truth-specific whitelist or program/loading factorization. Actual filtering:
1410 complete measured LR,1045 retained LR,4902 directed CCC features.
Targets occupy approximately0.43GB. Required retained LR subunits fit within
the unchanged V1 encoder gene budget; cell types only define targets.

`scripts/validate_interface_biological_simulation.py` checks physical invariance
and all four prespecified mechanistic families after unrestricted filtering.
On935 paired edge centers, composition maximum absolute difference is below3e-7.
RMS-conditioned family means give IFNG A/B2.26, TGFB B/A2.02, CXCL12 B/A1.64.
CXCL9/10/11–CXCR3 changes only1.04-fold: retained signal mainly comes from CAFs,
while the epithelial perturbations do not pass the same detection filter. This
weak family is displayed alongside the three stronger ones. No hidden feature
selection, model scores or K-based simulation tuning is used.

Generation labels describe cells' prespecified locations/states. A physical
window straddling a border can legitimately be intermediate; preserving the
original geometry does not guarantee that every inferred cluster follows these
center-based state labels. This QC verifies input contrast, not V1 recovery.

## Paths and execution

All generated files are under
`/home/nas3/biod/xueshuailin/CCC_Phe/workspace/`:

- Data/reference/layout: `data/tumor_immune_interface/`.
- Source-keyed targets: directory recorded by `target_cache.json`.
- Composition and communication QC: `outputs/tumor_immune_interface/biological_validation/`.
- Notebook: `/home/xueshuailin/CCC_Phe/Multifocal_Communication_Simulation.ipynb`.

Generation and QC commands (no neural training):

```bash
PYTHONPATH=src python scripts/rebuild_interface_biological_simulation.py
PYTHONPATH=src python scripts/validate_interface_biological_simulation.py
```

QC uses physicalGPU2. The first notebook cell shows design and compact QC. Its
two result-reader cells refuse old-source predictions. Original V1 weights and
external baselines remain on NAS but describe different RNA and are incompatible.
The explicitly requested V1 run uses `scripts/run_interface_dual_mlp.py --train --notebook`,
unchanged K6,seed40700,50epochs, current full targets, and a source-checked
`outputs/tumor_immune_interface/v1_gnn` output. Losses persist in cell2 while
running; completion exports compact h, performs KMeans and embeds the result.
