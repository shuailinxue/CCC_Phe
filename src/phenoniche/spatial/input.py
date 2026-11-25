"""Only fitting inputs enter the model. Author niche and phenotype labels are unused."""

import numpy as np
import pandas as pd


def lung_coordinates(adata, aligned_directory):
    sections = (4, 10, 16, 22, 28, 34)
    labels = adata.obs["section"].astype(str).to_numpy()
    if set(labels) != {f"section_{n}" for n in sections}:
        raise ValueError("Expected six registered Pentimalli sections")
    xyz = np.empty((len(labels), 3))
    display = np.asarray(adata.obsm["spatial"])
    for layer, n in enumerate(sections):
        idx = np.flatnonzero(labels == f"section_{n}")
        xy = np.loadtxt(
            aligned_directory / f"section_{n}_aligned.csv", delimiter=",", ndmin=2
        )
        if xy.shape != (len(idx), 2):
            raise ValueError(f"Section {n} row count disagrees with AnnData")
        # STIM CSVs omit barcodes. Confirm every row against the per-cell display coordinates.
        for axis in range(2):
            slope, offset = np.polyfit(xy[:, axis], display[idx, axis], 1)
            if abs(slope) < 1e-8 or not np.allclose(
                slope * xy[:, axis] + offset, display[idx, axis], atol=1e-4, rtol=0
            ):
                raise ValueError(f"Section {n} STIM row alignment check failed")
        xyz[idx, :2] = xy / 5.6
        xyz[idx, 2] = 30 * layer
    return xyz


def table_coordinates(adata, path):
    table = pd.read_csv(path, sep="\t", dtype={"cell_id": str}).set_index(
        "cell_id", verify_integrity=True
    )
    axes = ["x_um", "y_um"]
    if "z_um" in table:
        axes.append("z_um")
    return table.loc[adata.obs_names, axes].to_numpy()
