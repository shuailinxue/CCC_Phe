# Composition-first communication network pilot

This is the explicitly requested simulation experiment, separate from V1. It
reuses composition-only K=5 and the within-coarse statistical graph from the
prior experiment. The graph connects physically disjoint, nearby windows with
similar composition; it is used as a statistical training relation and does
not change the original 30-member microenvironment or directed CCC target.

Within each coarse class, construct directed CCC residuals against the existing
type/depth randomization null, clip at ±5, and regress out the square-root
composition proportions using fixed ridge 0.1. A separate MLP for each coarse
class has architecture `4902 → 192 → 32 → 192 → 4902`; the shared 32-vector is
the communication representation. It predicts the weighted mean of the
composition-matched, physically disjoint neighboring windows' residual CCC,
using MSE and 30 epochs. The neighbor prediction adds spatial smoothing and can
erase sharp boundaries; it is an experimental assumption, not established fact.
Targets and both input/target features are native directed CCC residuals; no
prior pathway database, fixed LR program or truth labels are used for training.

Every class is forcibly divided into two by K-means, producing an exploratory
K=10 map. The K=6 map splits only the class with the greatest K-means variance
reduction among classes with both children at least 5%; this selection uses
no truth, but it also has no significance test. It is not evidence that these
forced groups are true biological subclasses. The entire fit is independently
repeated on the unperturbed reference-baseline RNA. Both fits are complete
before any planted truth is read for evaluation. A/B labels and prespecified
mechanisms never select features, weights or classes.

Results under NAS `outputs/tumor_immune_interface/coarse_ccc_network`: K=6
A/B adjusted Rand index -0.0004, 100% paired A/B labels equal. Forced K=10
A/B ARI -0.0007, 99.25% of paired labels equal. The unperturbed negative
control likewise yields similar splits. The selection of coarse class 4 in
the K=6 view corresponds mainly to the shared tumor core, rather than the A/B
interface. Loss decreases are neighbor-CCC training MSE and do not validate
generalization or functional niche recovery.

Post-fit only, the coarse-class-3 A/B centroid squared distance is 4.12% of
the mean within-A/B squared distance (0.065% for the baseline control). This
indicates some A/B association remains in the learned embedding but it is much
smaller than within-region variation; it is not an independent validation or a
claim of a successfully recovered niche.
