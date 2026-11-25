"""Audit original Crowell CosMx WTx files without reading published niche labels.

Run with the cccphe Python environment. The default audit reads only slide 22's
raw metadata, FOV positions, expression header and a small polygon excerpt. It
does not download the entire expression matrix, select ROIs or fit a model.
"""

import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path
import sys
import zipfile

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from phenoniche.ccc.lr_atlas import canonicalize_gene, load_lr_atlas
from phenoniche.cohorts.remote_hdf5 import CachedRangeReader

DATA = Path("/home/nas3/biod/xueshuailin/CCC_Phe/workspace/data/crc_wtx_crowell")
SOURCE = "https://zenodo.org/api/records/15550908/files/22.zip/content"
SOURCE_BYTES = 698266606
# The archive includes two different donors. These are the original manifest's
# section assignments, not clusters or pathology-derived cell selections.
SECTION_FOVS = set(range(1, 127)) | set(range(199, 205)) | set(range(206, 210))
METADATA_COLUMNS = [
    "fov", "Area", "Area.um2", "CenterX_global_px", "CenterY_global_px",
    "nCount_RNA", "nFeature_RNA", "nCount_negprobes", "nCount_falsecode",
    "cell_ID", "cell",
]


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, data):
    temporary = path.with_suffix(path.suffix + ".incomplete")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def fetch_audit_files(full=False):
    DATA.mkdir(parents=True, exist_ok=True)
    with CachedRangeReader(SOURCE, SOURCE_BYTES, DATA / "http_cache",
                           prefetch_blocks=4) as reader:
        with zipfile.ZipFile(reader) as archive:
            write_json(DATA / "slide22_zip_inventory.json", [
                {"name": entry.filename, "size": entry.file_size,
                 "compressed": entry.compress_size, "offset": entry.header_offset}
                for entry in archive.infolist()
            ])
            for name in ("positions.csv.gz", "metadata.csv.gz"):
                target = DATA / ("slide22_" + name)
                if not target.exists():
                    # zipfile checks the complete member's CRC.
                    temporary = target.with_suffix(target.suffix + ".incomplete")
                    temporary.write_bytes(archive.read(name))
                    temporary.replace(target)
            if full:
                for name in ("counts.csv.gz", "polygons.csv.gz"):
                    target = DATA / ("slide22_" + name)
                    if target.exists():
                        continue
                    temporary = target.with_suffix(target.suffix + ".incomplete")
                    print(f"Fetching {name}; immutable HTTP blocks are cached on NAS", flush=True)
                    with archive.open(name) as member, temporary.open("wb") as output:
                        for block in iter(lambda: member.read(4 << 20), b""):
                            output.write(block)
                    if temporary.stat().st_size != archive.getinfo(name).file_size:
                        raise ValueError("Truncated ZIP member")
                    temporary.replace(target)
                    print(f"Saved {target.name}; complete ZIP member CRC verified", flush=True)
            header = DATA / "slide22_counts_header.csv"
            if not header.exists():
                with archive.open("counts.csv.gz") as member:
                    with gzip.GzipFile(fileobj=member) as counts:
                        header.write_bytes(counts.readline())
            polygons = DATA / "slide22_polygons_head.csv"
            if not polygons.exists():
                with archive.open("polygons.csv.gz") as member:
                    with gzip.GzipFile(fileobj=member) as stream:
                        polygons.write_bytes(b"".join(stream.readline() for _ in range(1001)))


def edge_connected_fov_sizes(positions):
    """Check measured, occupied FOVs only; adjacency does not imply tissue contact."""
    xy = positions[["x_global_px", "y_global_px"]].to_numpy()
    neighbors = cKDTree(xy).query_pairs(4258)
    adjacency = [set() for _ in xy]
    for first, second in neighbors:
        dx, dy = np.abs(xy[first] - xy[second])
        if (abs(dx - 4256) <= 2 and dy <= 2) or (abs(dy - 4256) <= 2 and dx <= 2):
            adjacency[first].add(second)
            adjacency[second].add(first)
    seen, sizes = set(), []
    for start in range(len(xy)):
        if start in seen:
            continue
        stack, count = [start], 0
        seen.add(start)
        while stack:
            node = stack.pop()
            count += 1
            for neighbor in adjacency[node] - seen:
                seen.add(neighbor)
                stack.append(neighbor)
        sizes.append(count)
    return sorted(sizes, reverse=True)


def audit():
    frame = pd.read_csv(DATA / "slide22_metadata.csv.gz", usecols=METADATA_COLUMNS)
    all_cells = len(frame)
    frame = frame.loc[frame.fov.isin(SECTION_FOVS)].copy()
    if len(frame) != 370074:
        raise ValueError("Section 221 cell count disagrees with original Table S1")
    if not frame.cell.is_unique or frame[["fov", "cell_ID"]].duplicated().any():
        raise ValueError("Duplicate physical cell identifiers")
    if not np.isfinite(frame[METADATA_COLUMNS[:-2]]).all().all():
        raise ValueError("Non-finite numeric metadata")
    if (frame.Area <= 0).any() or (frame["Area.um2"] <= 0).any():
        raise ValueError("Non-positive segmentation area")
    positions = pd.read_csv(DATA / "slide22_positions.csv.gz")
    positions = positions.loc[positions.FOV.isin(frame.fov.unique())].copy()
    if len(positions) != frame.fov.nunique() or not positions.FOV.is_unique:
        raise ValueError("Missing or duplicate FOV positions")
    # Original author conversion, code/_utils.R; independently check Area.um2.
    scale = np.sqrt(frame["Area.um2"] / frame.Area)
    if not np.allclose(scale, 0.12028, rtol=3e-5):
        raise ValueError("Original physical calibration is inconsistent with cell areas")
    xy = frame[["CenterX_global_px", "CenterY_global_px"]].to_numpy() * 0.12028
    # 30 members include center, matching the project's physical neighborhoods.
    distances = cKDTree(xy).query(xy, k=30, workers=6)[0][:, -1]
    with (DATA / "slide22_counts_header.csv").open() as handle:
        header = next(csv.reader(handle))
    genes = {canonicalize_gene(gene) for gene in header
             if gene not in ("fov", "cell_ID")
             and not gene.startswith(("Negative", "SystemControl"))}
    atlas = load_lr_atlas(ROOT / "data/commuspace_human_lr_atlas.tsv")
    complete = [interaction for interaction in atlas.interactions
                if set(interaction.ligand_components + interaction.receptor_components) <= genes]
    result = {
        "section": "221", "slide": "22", "raw_cells": len(frame),
        "other_section_cells_excluded": all_cells - len(frame),
        "fovs": int(frame.fov.nunique()), "all_cell_ids_unique": True,
        "all_coordinates_finite": True, "all_areas_positive": True,
        "biological_target_columns": len(genes),
        "median_counts_raw_metadata": float(frame.nCount_RNA.median()),
        "median_detected_genes_raw_metadata": float(frame.nFeature_RNA.median()),
        "physical_um_per_px": 0.12028,
        "independent_area_scale_median": float(scale.median()),
        "occupied_fov_edge_connected_component_sizes": edge_connected_fov_sizes(positions),
        "knn30_member_radius_um_quantiles": dict(zip(
            ("q50", "q90", "q95", "q99", "max"),
            np.quantile(distances, (.5, .9, .95, .99, 1)).tolist())),
        "fraction_knn30_member_radius_gt100um": float((distances > 100).mean()),
        "atlas_unique_pairs": len(atlas), "panel_complete_LR_pairs": len(complete),
        "LR_coverage_kind": "Panel completeness only; actual cell-pair support not yet measured",
        "published_niche_labels_read": False, "published_CCC_read": False,
        "published_trajectory_read": False, "model_training_started": False,
        "full_expression_matrix_downloaded": (DATA / "slide22_counts.csv.gz").exists(),
        "source": SOURCE, "source_archive_bytes": SOURCE_BYTES,
        "source_archive_md5": "7fe6cbc948b8523db6331306caf89e48",
        "source_archive_md5_verified": False,
        "download_scope": "ZIP member CRC checked for complete metadata/positions; headers are excerpts",
        "file_sha256": {path.name: sha256(path) for path in (
            DATA / "slide22_metadata.csv.gz", DATA / "slide22_positions.csv.gz",
            DATA / "slide22_counts_header.csv", DATA / "slide22_polygons_head.csv")},
        "limitations": [
            "Adjacent-section H&E is not pixel-aligned histology of the measured section",
            "FOV connectivity is not proof of uninterrupted tissue or admissible graph edges",
            "Source Table S3 notes section orientation limits invasion-depth scoring",
            "Similar composition and different communication between interfaces remain untested",
        ],
    }
    write_json(DATA / "section221_initial_audit.json", result)
    print(json.dumps(result, indent=2), flush=True)


def audit_expression():
    """Scan actual integer counts; no published annotations or spatial labels."""
    source = DATA / "slide22_counts.csv.gz"
    if not source.exists():
        raise FileNotFoundError("Complete original expression member is required")
    frame = pd.read_csv(DATA / "slide22_metadata.csv.gz", usecols=METADATA_COLUMNS)
    frame = frame.loc[frame.fov.isin(SECTION_FOVS)]
    expected = {(int(row.fov), int(row.cell_ID)): (int(row.nCount_RNA), int(row.nFeature_RNA))
                for row in frame.itertuples()}
    with gzip.open(source, "rt") as handle:
        header = next(csv.reader([handle.readline()]))
        if header[:2] != ["fov", "cell_ID"]:
            raise ValueError("Unexpected expression identifier columns")
        selected = np.array([i for i, gene in enumerate(header)
                             if i >= 2 and not gene.startswith(("Negative", "SystemControl"))])
        genes = [canonicalize_gene(header[i]) for i in selected]
        totals = np.zeros(len(genes), dtype=np.int64)
        detected = np.zeros(len(genes), dtype=np.int64)
        seen, mismatched_totals, mismatched_features = set(), 0, 0
        for line in handle:
            if int(line.split(",", 1)[0]) not in SECTION_FOVS:
                continue
            values = np.fromstring(line, dtype=np.int64, sep=",")
            if len(values) != len(header) or (values < 0).any():
                raise ValueError("Invalid original integer expression row")
            key = (int(values[0]), int(values[1]))
            if key not in expected or key in seen:
                raise ValueError("Expression/metadata identifiers disagree")
            seen.add(key)
            rna = values[selected]
            totals += rna
            detected += rna > 0
            mismatched_totals += int(rna.sum() != expected[key][0])
            mismatched_features += int(np.count_nonzero(rna) != expected[key][1])
            if len(seen) % 50000 == 0:
                print(f"Audited {len(seen):,} original section 221 expression rows", flush=True)
    if len(seen) != len(expected):
        raise ValueError("Incomplete section 221 expression coverage")
    table = pd.DataFrame({"gene": genes, "total_counts": totals,
                          "detected_cells": detected, "fraction_detected": detected / len(seen)})
    table.to_csv(DATA / "section221_observed_gene_detection.csv", index=False)
    atlas = load_lr_atlas(ROOT / "data/commuspace_human_lr_atlas.tsv")
    panel = set(genes)
    observed = set(table.loc[table.detected_cells >= 100, "gene"])
    complete = lambda interaction, available: set(
        interaction.ligand_components + interaction.receptor_components) <= available
    result = {
        "section": "221", "original_integer_expression_rows_audited": len(seen),
        "raw_count_sha256": sha256(source), "biological_target_columns": len(genes),
        "genes_detected_any_cell": int((detected > 0).sum()),
        "genes_detected_at_least100cells": len(observed),
        "panel_complete_LR_pairs": sum(complete(x, panel) for x in atlas.interactions),
        "LR_pairs_all_subunits_detected_at_least100cells": sum(
            complete(x, observed) for x in atlas.interactions),
        "expression_total_disagreements_with_metadata": mismatched_totals,
        "detected_feature_disagreements_with_metadata": mismatched_features,
        "interpretation": "Global gene detection audit only; not supported physical cell-pair CCC",
        "model_coverage_threshold_changed": False, "published_niche_labels_read": False,
    }
    write_json(DATA / "section221_expression_audit.json", result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fetch", action="store_true", help="Fetch missing public audit files")
    parser.add_argument("--fetch-full", action="store_true", help="Also fetch original counts and polygons")
    parser.add_argument("--audit-expression", action="store_true", help="Scan complete observed counts")
    args = parser.parse_args()
    if args.fetch or args.fetch_full:
        fetch_audit_files(full=args.fetch_full)
    audit()
    if args.audit_expression:
        audit_expression()
