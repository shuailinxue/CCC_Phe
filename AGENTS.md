# CCC_Phe

The only current model is `phenoniche.spatial`. Its command-line entry point is
`scripts/run.py`. Keep research logic in the package and tests in `tests/`.
Describe any change to input, output or settings in `docs/spatial.md`.

- Edit code directly in the current checkout. The user manages Git and uses
  it to restore earlier versions. Do not create code backups or copies on NAS.
- Store all new large project files under
  `/home/nas3/biod/xueshuailin/CCC_Phe/`. Put datasets under `data/` and
  results and checkpoints under `results/<dataset>/<method>/`.
- Save requested notebooks directly to
  `/home/nas3/biod/xueshuailin/CCC_Phe/notebooks/`.
- Use Python by default for scientific figures and notebooks. Do not ask the
  user to choose Python or R for future figure tasks.

Preserve existing inputs, notebooks, checkpoints, and saved results. Do not
rerun analyses or generate large artifacts without a request. Use physical
GPU 2 when GPU work is requested and no device is specified.

The spatial model builds physical graphs independently per specimen. Known
niches, simulation truth and phenotype labels are downstream evaluation only.
The included LR reference is human. Expression-derived LR scores are proxies
for potential communication.

## Experiment persistence and scope

- Persist time-consuming intermediate results, including cell composition,
  UMAP coordinates, cluster labels, communication tensors, and decompositions.
- Before each run, check existing caches first. Recompute only when a cache is
  absent or its input data, parameters, or algorithm have changed.
- Record aligned cell IDs, parameters, and the version information needed to
  verify each cache; never reuse a cache whose provenance does not match.
- Notebooks and experiment scripts must not depend on transient state from a
  previous run: after a kernel restart, load valid saved results to continue.
- Follow the user's requested experimental steps strictly. Do not expand the
  analysis or generate additional figures, statistics, or reports unrequested.

Historical code and project instructions are archived under
`/home/nas3/biod/xueshuailin/CCC_Phe/Backup/retired_version_interfaces/`.
