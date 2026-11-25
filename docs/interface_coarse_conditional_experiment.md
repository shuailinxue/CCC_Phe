# Composition-first conditional CCC simulation experiment

Run `python scripts/run_interface_coarse_ccc.py` with the project Python environment.
The only inputs are the existing RNA, observed directed CCC, 30-member physical
neighborhoods, composition, coordinates and cell types. The V1 model and notebooks
are not changed. Outputs and progress live under NAS
`outputs/tumor_immune_interface/coarse_conditional_ccc`.

The simulated generating design has six states. Merging A and B gives five
coarse categories, so this development experiment uses composition-only
K-means K=5 and at most one further split, K<=6. These numbers use simulation
design knowledge; generated state labels and injected mechanism names are
never supplied to the fitting or used to choose which class to split.

For each composition class, calculate within-class CCC deviations from the
previously computed type/depth-stratified whole-RNA-profile null. The statistical
graph is restricted to the same coarse class and connects physically disjoint
windows. A directed-CCC feature is eligible only if its local contact support
is sufficient. The spatial coherence statistic is centered within coarse class.
Sixty-four null draws estimate its mean and scale and 256 separate null draws
calibrate the maximum test statistic across all eligible class/CCC combinations.
The 32 further held-out randomizations check global-null behavior. These are
conditional randomization checks, not independent biological replicates; valid
interpretation depends on within-type/depth exchangeability. The max statistic
addresses the global null and is not asserted to control partial-null FDR.

If no feature passes, report all five coarse classes with no unsupported split.
Otherwise split only the coarse class with the strongest calibrated feature,
using all passing directed-CCC features for that class and K-means K=2. This
allows the known maximum six classes without using planted labels to choose the
class or its features. The complete procedure is independently repeated on the
unperturbed reference-baseline RNA. Truth is read only after both fits complete.
The baseline and randomized controls, A/B pairing, spatial plots and complete
screening tables are saved. This is a development test, not a validated real-data
analysis or a replacement of the current V1 framework.
