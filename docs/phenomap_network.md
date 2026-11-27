# PhenoMap V1 direct-network mixture

## Full-slice default

`scripts/phenomap_network.py` uses `configs/network_v1_full.json` by default.
The vocabulary and communication features are ordered Sender × Receiver × LR.
Composition and communication use the same closed 50 µm window around each
cell. Both endpoints belong to the window and specimen. Only distinct cells
within the LR-specific cutoff can communicate. The cutoffs are 20 µm for
Contact and ECM, 100 µm for Secreted, and 50 µm for Unknown and Ambiguous.
Distances are used for eligibility. Eligible receivers receive equal weight.

An eligible sender–receiver-type–LR combination contributes one Hurdle trial
per sender cell. Its activity is the sender ligand arm score times the mean
receiver arm score across eligible receiver cells of that type, including
zero receptor values. The dictionary keeps the closed-window weighting by
local sender-type composition, support and LR reliability. The existing
occurrence and positive log-intensity likelihoods are used to learn an
independent dictionary and fit the joint composition–communication mixture.

The default has R=12, K=8, seed 73, batch size 32, diversity weight 4 and
communication temperature 1. Each center appears once per shuffled epoch,
for 30 dictionary and 24 prototype epochs. Both stages finish at the last
epoch. The historical sampling settings in the configuration preserve the
recorded fit signature. Census fitting does not consult them.

The command below verifies and reuses the completed ovarian fit, including
its network and model source hashes, aligned cell IDs and both coverage
records. With `--reuse-only`, an incompatible or missing fit raises an error.
For a new full-slice input, provide a closed cache from
`phenoniche.v1.storage.prepare` with `window=50` and `fraction=0` and omit
`--reuse-only`. Checkpoints, local activity, results and epoch coverage
are stored under a content-addressed directory inside the output root.

```bash
root=/home/nas3/biod/xueshuailin/CCC_Phe/results/spatialfusion_ovca/phenomap_polish
python scripts/phenomap_network.py \
  --network "$root/full_window_cache/network-03aef6c677034d3cc5b12452" \
  --root "$root" --reuse-only
```

## Historical sampled benchmarks

The historical entry point is `scripts/phenomap_network.py --sampled`. It uses the existing V1
network schema and sufficient statistics. The old implementation and
`scripts/phenomap_v1.py` remain available for reproducible comparisons.

## Dictionary and scores

For a nonnegative activity vector a, the occurrence logit and positive log
intensity mean are b + (a − 1)W_occ and m + (a − 1)W_int. Activity below one
suppresses the program relative to its reference. Both loading channels are
nonnegative with a joint per-program maximum of one. Their magnitudes are
penalized and the fitted intercept is anchored to a smoothed training-only
pooled edge intercept. Positive log intensity variance is fixed at one.

Use S observed sender opportunities, C detections, L the sum of positive log
intensities, and Q their sum of squares. The raw detection NLL at logit h is
S softplus(h) − C h. The positive log-intensity NLL at mean m, up to constants,
is (Q − 2mL + C m²)/2. Structural missingness has S=0.

The base weight is sender node fraction times S^(-1/2), times the training
edge reliability S_train/(S_train+32), times the bounded LR frequency weight
clip(p_LR/0.01, 0.25, 4)^(-1/4). All supported trials on the same edge,
including zeros, receive the same occurrence weight. The intensity weight
adds C_train/(C_train+12). Each head has a square-root effective evidence
denominator based on the sum of weighted trials (or detections), with
multipliers 32 and 8 respectively. These are fixed composite evidence
budgets. Unweighted prediction is evaluated on actual masked holdout trials.

Stage 1 alternates projected local updates with Adam on the global dictionary.
The local objective includes ridge λ||a−1||²/2. No niche variable enters local
inference. Dynamic sampling prioritizes least-visited spatial centers. Checkpoints
are selected by relative raw detection NLL plus relative positive log-MSE on
a disjoint internal slab with hidden edges.

Stage 2 freezes the dictionary and intercept. A candidate's score is log ω
plus capped multinomial node score plus temperature times the detection and
intensity log gains of its prototype against the pooled network. Sender node
mass organizes the conditional edge term. No extra composition loss is used.
The objective marginalizes candidates with logsumexp. Initial prototypes use
composition and independent program activity. Two joint temperature/start
configurations (0.5 and 1.0) are compared by internal edge prediction. A
three-epoch uniform warm-up and weak log-weight regularizer limit early
concentration. Candidate scores give conditional soft assignments.

Stage 3 jointly updates dictionary and prototypes at low learning rates using
independent local reconstruction plus 0.15 times the niche objective.
Dictionary and intercept are anchored to stage 1. Local activities still
optimize their communication-only objective. This is a multitask objective.
The eligibility gate requires both internal dictionary and mixture prediction
utilities to improve or tie and matched program cosine to stay at least 0.90.
The final test region is used only for reporting.

## Run and inspect

```bash
python scripts/phenomap_network.py --sampled --network /path/to/cached/network \
  --root /path/to/results/dataset/phenomap_network \
  --config configs/network_v1_sampled.json --device cuda:2
```

Use `--joint-only` for seed repeats and `--real-target REAL_NETWORK` for a
paired shuffled fit with evaluation against real observations. Use a new
output root for independent studies. The run stores the buffered split, visit
counts, dictionary and mixture checkpoints, local activities, typed loadings
and per-niche network predictions. `cells.npz` contains aligned coordinates,
node counts, activities, assignment probabilities and the four candidate-wise
score components. `network_workflow.explain` reconstructs one observed and
denoised window and provides exact per-edge log-score margins for a candidate
comparison. Positive and negative program contributions use (a−1)W.

`experiments/legacy_matched.py` reruns the legacy comparator on the same fit
split. `experiments/network_simulations.py` generates and evaluates five
controlled scenarios. `experiments/report_network_v1.py` takes frozen run
directories and performs final annotation evaluation and figure export.

For a cached fit, use `python scripts/explain_network_niche.py --network
NETWORK --fit FIT --cell CELL_ID --output NEW_DIRECTORY`. Select
`--stage refined` to inspect the separately saved joint refinement.
The CLI defaults to CPU for this single-window interpretation.

## Sampled benchmark scope

The cached Pt73 and ovarian networks contain 1,609 and 2,560 sampled centers.
The outer test regions come from the cached split. Within the original
training region, take the rightmost 17% as internal validation and exclude a
buffer of twice the neighborhood radius. Each stage uses the resulting
split. The previously cached expression scales were estimated over the
original training region, which includes the new internal validation slab.
The outer test remains excluded from the expression scale estimates.
For Pt73, the old full-section shuffle has a different outer split, so
`experiments/pt73_sampled_null.py` builds the missing matched null using the
same sampled center logic. Both real networks and the ovarian null are reused.

The reported fits use R=12 and K=4. The upper dictionary budget is 30 epochs.
Pt73 seed 73 selected epoch 27 and ovarian seed 73 selected epoch 30.
The ongoing ovarian validation improvement at the last epoch leaves
dictionary convergence open. The sampled runs do not establish performance
for full-slice inference. The final decision about refinement uses only
internal validation. Both datasets retained the frozen-dictionary mixture.
