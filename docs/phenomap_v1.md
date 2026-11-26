# PhenoMap V1

PhenoMap estimates global functional niches from cell-type counts and locally observed,
directed ligand–receptor (LR) expression. Each center has a specimen-local radius
window including itself. Both endpoints must lie inside that window. Distinct cells
of the same type can communicate. The model uses unique directed sender-to-receiver
pairs. Gene counts, specimen IDs, cell types and physical coordinates are the only
fitting inputs. Interpret scores as potential communication.

## Observations and likelihood

Each window has integer type counts `n`. For each sender cell and receiver type,
average the V0 outgoing score across distinct eligible receivers. Use the geometric
mean of transformed ligand/receptor expression, including geometric means within
complexes. Training cells define nonzero 95th-percentile gene scales after log1p(CP10K).
Freeze those scales for buffered validation and new data. A gene absent in the
training reference uses scale 1. Compare the frozen implementation to V0 Channels
in the network tests. Default radii in coordinate units are 50 for the window,
20 for Contact and ECM, 100 for Secreted and 50 for Ambiguous/Unknown.
A secreted interaction can span the window diameter.

For edge `e=(sender,receiver,LR)`, store the eligible sender count S by radius,
type pair and center. Store C, sum(log x), sum((log x)²), and sum(x) for positive
sender means in sparse rows. S=0 is missing. S>0 and C=0 is an observed zero.
The conditional measured activity is sum(x)/S, and the mean among detections
is sum(x)/C. Center shards bound construction memory. Fits load compressed
observations into host memory and expand only a mini-batch on the device.

Let a=softplus(u), h=b+aW_occ and m=d+aW_int. C has a binomial working model
with S trials and logit h. Positive log sender means have a Gaussian working
model with mean m and fixed standard deviation sigma_x (default 1). The two
nonnegative dictionaries share activities. Each program is normalized by its
maximum loading across both attributes and all typed LR edges. A mean loading
penalty favors sparse dictionaries. No rank-one or orthogonality restriction
applies. Eight pooled pseudo-observations shrink initial edge baselines toward
the global mean. A quadratic penalty anchors fitted baselines to these values.
For globally unsupported edges, predictions are extrapolations.

The multinomial score for niche k is n log pi_k. The Gaussian latent score is
−||u−eta_k||²/(2 sigma_a²), with fixed sigma_a=2. The optimization minimizes
hurdle losses minus logsumexp over the sum of multinomial, latent and log mixture
scores, plus regularization. Constants independent of fitted parameters may be
dropped. Both dictionary channels, edge intercepts, local u and all mixture
parameters are updated in the same training loop. Local Adam steps use the
analytic gradient, and a global Adam step updates the remaining parameters.
Initialization uses KMeans on training summaries to seed the mixture and
positive edge deviations. The full likelihood permits global reassignment.

The default calibrated objective multiplies the multinomial score by
min(1,16/sum(n)). In each window, average detection losses across supported
edges within each sender type, then across supported sender types. Cap each
edge's effective trials at 8 by multiplying by min(1,8/S). Apply the same
rule separately to the positive-intensity losses, using C and senders with
detections. This conditional intensity term gives rare positive observations
a separate contribution. Raw mode (`"scale":"raw"`) uses untempered counts and
sums all supported edge log likelihoods. All settings are fields of
`phenoniche.v1.model.Settings`. The softmax over niche scores is conditional
on a local MAP estimate. Overlap and correlated LR measurements limit calibration.

For interpretation only, normalize supported conditional activities for each
sender with a shared reference channel of weight tau. Multiplication by that
sender's observed fraction gives the joint edge mass. The reference mass plus
observed joint edges marginalizes to the original sender mass, including senders
with no supported edges. Tau is the median positive total activity per sender
type on the initialization sample (1 for all-zero data). It has no biological
state interpretation and introduces no additional training loss.

## Training, prediction and interpretation

Install the project with `python -m pip install -e '.[v1,dev]'`. Run commands
from the V1 checkout. The example uses a 2D/3D AnnData obsm in micrometres, a raw
count matrix/layer, exact gene names, a type column, and a specimen column.
Set specimen to null for an explicitly single specimen. The Pt73 file retains
original X/Y and declares the units unverified. The pilot follows the previous
Pt73 coordinate convention, one input unit per micrometre.

```bash
CFG=configs/pt73_v1.json
NET=$(python scripts/phenomap_v1.py prepare --config "$CFG" | tail -1)
RUN=$(python scripts/phenomap_v1.py train --config "$CFG" --network "$NET" --mode joint --seed 73 --device cuda:2 | tail -1)
python scripts/phenomap_v1.py infer --config "$CFG" --network "$NET" --run "$RUN" --device cuda:2
python scripts/phenomap_v1.py explain --network "$NET" --run "$RUN" --cell 1251985 --output /tmp/pt73-example
```

Use `--mode composition` and `--mode communication` for the two ablations.
The composition ablation scores held-out edges with its pooled training-only
edge baseline. Its latent fields have no inferential interpretation. In the
communication ablation the composition score is zero. For transfer, change the
input and set the holdout fraction to zero in a new config; prepare the new
network with `--reference-run "$RUN"`, then pass that network to `infer`.
Transfer requires the same ordered panel, compatible LR vocabulary, preprocessing
and radii. Known types may be absent. Supply a new output directory to `explain`.
It defaults to the competitor with the strongest composition evidence, or use
`--competitor K`.

Each epoch prioritizes previously undervisited centers with randomized ties and
enforces specimen-local minimum distance. All centers receive fixed-global inference.
Training coverage, including never-sampled centers, is recorded per ID.
The validation slab consists of the rightmost 10% of centers by x per specimen.
Training stops more than two window radii from the slab. Centers in the intervening
buffer still receive final inference. The pilot scores up to 1,024 validation centers
chosen with a fixed seed, hides a common 20% sample of typed LR edges during local
inference, and scores those edges against sender counts and positive log intensities.
The final full-network cell output uses all available edges. Sampling limits
redundancy, while the validation estimates apply only to this slab and specimen.

For the null, run prepare with `--null-seed 31415`, then train jointly as above.
Whole expression profiles shuffle within cell type, specimen and side of the
train/test tissue boundary. Node mass, positions, and opportunity counts remain
exact. This destroys local expression structure while retaining type-level
expression and correlated genes. Read cached V0 and Hub labels only at comparison
time with `scripts/compare_v1.py --help`.

## Saved artifacts

* Each network directory contains a provenance manifest, aligned schema, frozen
  gene scales, spatial split, and compressed sufficient-statistic shards.
* Each fit contains effective settings and software/source hashes, device and host,
  a reloadable global checkpoint, training schema, per-cell sampling counts, and
  loss components including mixture prior and entropy. Repeated commands reuse
  completed stages. Separate model configurations share the same upstream network.
* `cells.npz` holds cell IDs, coordinates, specimen, type counts, activities,
  latent coordinates, per-niche composition and latent evidence, soft memberships
  and hard assignments. `niches.npz` contains the prototypes, mixture weights
  and training opportunity counts. Count the repeated windows as overlapping
  support when interpreting these values.
* `programs.npz` contains both full dictionaries, baselines and the edge vocabulary.
  Edge index = (sender_index × number_of_types + receiver_index) × number_of_LRs
  + LR_index. `top_edges.tsv` lists the largest supported edges per program.
  `niche_networks.npz` contains prediction at softplus(eta), including detection,
  positive log mean, expected activity and difference from the mean across niches.
  The expected activity uses the log-normal mean and can exceed the normalized
  observed range. It is a model-based prediction.
* `explain` reconstructs one microenvironment on demand. Its table includes
  raw evidence, S and C, masked joint masses, extrapolated estimates, and per-program
  additive contributions on the logit and log-intensity scales. The candidate
  raw NLL contrast compares the two prototype networks at fixed activities.
  It is an indirect edge diagnostic. The exact conditional log-odds decompose
  into composition, latent-program, and mixture-prior margins.
* `metrics.json` records held-out positive/zero prediction, cluster occupancy,
  prototype separation, program redundancy and dispersion. Matched dictionary
  cosines in post-fit comparisons use both attributes. Program axes and niche
  labels can permute across runs. `coarse_by_niche.tsv` maps V0 coarse regions
  to V1 assignments when aligned coarse labels are supplied.
