# CCC_Phe

CCC_Phe identifies spatial niches from local cell composition and directed
ligand–receptor expression. It accepts cell-level counts, registered 2D or 3D
coordinates in micrometres, cell types, specimen IDs, and an LR table.

## Install and test

From the checkout, with Python 3.11 or newer:

```bash
python -m pip install -e '.[dev]'
python -m pytest
```

## Run PhenoMap V1

Use `python scripts/phenomap_network.py --network CACHE --root OUTPUT` for
full-slice niche fitting on a closed 50 µm observation cache. The defaults
are R=12 and K=8. Compatible completed results are reused automatically.
See the [V1 network and cache instructions](docs/phenomap_network.md).

## Spatial CP workflow

For a single specimen with raw counts in the `counts` layer, cell types in
`obs["celltypes"]`, and micrometre coordinates in `obsm["spatial_um"]`:

```bash
python scripts/run.py \
  --input tissue.h5ad \
  --output results/tissue \
  --lr data/commuspace_human_lr_atlas.tsv \
  --single-specimen --obsm-um spatial_um
```

Use `--layer X` when `.X` contains raw counts. For multiple specimens, replace
`--single-specimen` with `--specimen-key KEY`. The output directory must be new.
Use `--cp-device cpu` or `--cp-device cuda:2` to select the CP device.
See `python scripts/run.py --help` for coordinate and settings options. Use an
LR table appropriate to the species and gene symbols in the input.

For an interface analysis, preview the enrichment without fitting CP, then
inspect `enrichment.png` and choose two coarse-class IDs. Use the same input
and coarse settings for both commands:

```bash
python scripts/run.py --input tissue.h5ad --output preview \
  --single-specimen --obsm-um spatial_um --enrichment-only

python scripts/run.py --input tissue.h5ad --output results/interface_0_1 \
  --lr data/commuspace_human_lr_atlas.tsv --single-specimen --obsm-um spatial_um \
  --interface 0 1 --interface-um 25 --aggregation mean-mean
```

Python callers use the same two stages:

```python
from phenoniche.spatial import Settings, SpatialNiches

cfg = Settings(interface_um=25, aggregation="mean-mean")
preview = SpatialNiches.coarse(cell_ids, xyz_um, cell_types, specimens, cfg)
preview.save("preview")
# Choose a and b from the enrichment heatmap.
result = SpatialNiches.fit(
    counts, genes, cell_ids, xyz_um, cell_types, specimens, lr, cfg,
    interface=(a, b),
)
```

## Source and reproducibility

- `src/phenoniche/spatial/` contains the model. Read `model.py` for the workflow
  and `settings.py` for the defaults. Python callers use
  `phenoniche.spatial.SpatialNiches.fit`.
- `scripts/run.py` handles input and output. `tests/test_spatial.py` checks the
  active model on synthetic inputs.
- `data/` contains the human LR table and its provenance record.
- [docs/spatial.md](docs/spatial.md) describes the input schema, method and outputs.

The 3D lung notebook and saved results reside on the project NAS. Retired
implementations and experiments are under `Backup/retired_version_interfaces/`
on that NAS.
