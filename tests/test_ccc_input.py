from dataclasses import replace

import anndata as ad
import numpy as np
import pandas as pd

from phenoniche.ccc.config import CCCInputConfig
from phenoniche.ccc.data import prepare_prime5k_ccc


def test_ccc_input_build_and_cache_reuse_without_external_code(tmp_path):
    n = 12
    cells = pd.DataFrame({
        "cell_id": [f"cell-{i}" for i in range(n)],
        "cell_type_coarse": ["A", "B"] * (n // 2),
    }, index=[f"cell-{i}" for i in range(n)])
    adata = ad.AnnData(np.arange(1, n * 2 + 1, dtype=np.float32).reshape(n, 2),
                      obs=cells, var=pd.DataFrame(index=["L", "R"]))
    adata.obsm["spatial"] = np.column_stack((np.arange(n), np.zeros(n))).astype(np.float32)
    adata.uns["dataset"] = "Xenium_Prime_Breast_Cancer_FFPE"
    input_path = tmp_path / "input.h5ad"
    adata.write_h5ad(input_path)
    atlas_path = tmp_path / "atlas.tsv"
    atlas_path.write_text("ligand\treceptor\nL\tR\n")
    config = replace(CCCInputConfig(), processed_h5ad=str(input_path), lr_atlas=str(atlas_path),
                     output_dir=str(tmp_path / "ccc"), device="cpu", n_neighbors=4,
                     minimum_features=1, opportunity_support_minimum=1,
                     ccc_batch_size=4, lr_batch_size=1)
    first = prepare_prime5k_ccc(config)
    assert first["matrix"].shape == (n, 4)
    assert np.isfinite(first["matrix"]).all() and (first["matrix"] >= 0).all()
    assert set(zip(first["features"].sender, first["features"].receiver)) == {
        ("A", "A"), ("A", "B"), ("B", "A"), ("B", "B")}
    assert first["cells"].cell_id.tolist() == cells.cell_id.tolist()
    matrix_path = first["paths"]["ccc"]
    original_mtime = matrix_path.stat().st_mtime_ns
    second = prepare_prime5k_ccc(config)
    assert matrix_path.stat().st_mtime_ns == original_mtime
    assert np.array_equal(first["matrix"], second["matrix"])
