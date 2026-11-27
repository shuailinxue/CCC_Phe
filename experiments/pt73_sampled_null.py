"""Build the missing Pt73 null with the exact sampled network's split and scales."""
from ovca_network import build, PT

# The historical full-slice Pt73 null used a different holdout cut.
# This call shares the same sampling and preprocessing as the existing sampled cache.
build(PT,'type_lvl1',True)
