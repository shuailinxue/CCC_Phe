# CCC_Phe

The only current model is `phenoniche.spatial`. Its command-line entry point is
`scripts/run.py`. Keep research logic in the package and tests in `tests/`.
Describe any change to input, output or settings in `docs/spatial.md`.

Edit the current checkout directly. The user manages Git. Preserve existing
inputs, notebooks, checkpoints, and saved results. Do not rerun analyses or
generate large artifacts without a request. New large outputs go under
`/data1/xueshuailin/CCC_Phe/workspace/`; existing NAS inputs stay at their paths.
Use physical GPU 2 when GPU work is requested and no device is specified.

The spatial model builds physical graphs independently per specimen. Known
niches, simulation truth and phenotype labels are downstream evaluation only.
The included LR reference is human. Expression-derived LR scores are proxies
for potential communication.

Historical code and project instructions are archived under
`/home/nas3/biod/xueshuailin/CCC_Phe/Backup/retired_version_interfaces/`.
