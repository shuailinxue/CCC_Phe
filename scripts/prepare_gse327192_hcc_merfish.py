#!/usr/bin/env python3
"""Organize GSE327192 HCC MERFISH files and audit protective-CCC panel coverage."""

from __future__ import annotations

import argparse
import gzip
import re
from collections import Counter
from pathlib import Path

import pandas as pd


SAMPLES = {
    "GSM9651110": "1041_region_0",
    "GSM9651111": "121_region_0",
    "GSM9651112": "1004_region_0",
    "GSM9651113": "1005_region_0",
    "GSM9651114": "1021_region_0",
    "GSM9651115": "1037_region_0",
    "GSM9651116": "1023_region_0",
    "GSM9651117": "106_region_0",
}


def target_genes(entity: str) -> list[str]:
    """Split CellChat-style multimeric entities into required gene symbols."""
    return [part.upper() for part in re.split(r"[_+&:]", str(entity)) if part]


def read_panel(transcript_file: Path) -> tuple[list[str], list[str]]:
    targets: set[str] = set()
    for chunk in pd.read_csv(
        transcript_file,
        compression="gzip",
        usecols=["gene"],
        dtype={"gene": "string"},
        chunksize=2_000_000,
    ):
        targets.update(chunk["gene"].dropna().astype(str).str.strip())
    blanks = sorted(x for x in targets if x.lower().startswith("blank-"))
    genes = sorted(x for x in targets if x not in blanks)
    return genes, blanks


def write_coordinates(metadata_file: Path, output_file: Path, accession: str) -> int:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    first = True
    total = 0
    for chunk in pd.read_csv(
        metadata_file,
        compression="gzip",
        usecols=["EntityID", "fov", "center_x", "center_y"],
        dtype={"EntityID": "string", "fov": "Int64"},
        chunksize=500_000,
    ):
        out = chunk.rename(
            columns={"EntityID": "cell_id", "center_x": "x", "center_y": "y"}
        )
        out.insert(0, "sample", accession)
        out.to_csv(
            output_file,
            sep="\t",
            index=False,
            compression="gzip",
            mode="wt" if first else "at",
            header=first,
        )
        first = False
        total += len(out)
    return total


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--stable-ccc", type=Path, required=True)
    args = parser.parse_args()

    raw = args.data_root / "raw"
    processed = args.data_root / "processed"
    coordinates = processed / "cell_coordinates"
    processed.mkdir(parents=True, exist_ok=True)

    partials = sorted(raw.glob("*.aria2"))
    if partials:
        raise RuntimeError(f"Downloads are incomplete: {[p.name for p in partials]}")

    manifest_rows = []
    for accession, sample_name in SAMPLES.items():
        prefix = f"{accession}_{sample_name}"
        metadata_file = raw / f"{prefix}_cell_metadata.csv.gz"
        transcript_file = raw / f"{prefix}_detected_transcripts.csv.gz"
        for path in (metadata_file, transcript_file):
            if not path.is_file():
                raise FileNotFoundError(path)
        manifest_rows.append(
            {
                "accession": accession,
                "sample_name": sample_name,
                "expression_file": str(transcript_file),
                "coordinate_metadata_file": str(metadata_file),
                "expression_bytes": transcript_file.stat().st_size,
                "coordinate_metadata_bytes": metadata_file.stat().st_size,
                "cell_type_annotation_available": False,
                "tls_annotation_available": False,
                "tissue_region_annotation_available": False,
            }
        )

    manifest = pd.DataFrame(manifest_rows)
    manifest.to_csv(processed / "hcc_sample_manifest.tsv", sep="\t", index=False)

    # The smallest HCC file contains every one of the 550 assay targets and is
    # used as the canonical panel source to avoid repeatedly scanning ~8 GB of
    # compressed transcript records.
    panel_source = raw / "GSM9651111_121_region_0_detected_transcripts.csv.gz"
    genes, blanks = read_panel(panel_source)
    panel = pd.DataFrame(
        [(g, False) for g in genes] + [(b, True) for b in blanks],
        columns=["target", "is_blank_control"],
    ).sort_values(["is_blank_control", "target"])
    panel.to_csv(processed / "merfish_panel_all_targets.tsv", sep="\t", index=False)
    (processed / "merfish_panel_genes.txt").write_text("\n".join(genes) + "\n")

    stable = pd.read_csv(args.stable_ccc, sep="\t")
    protective = stable.loc[
        (stable["sign_selection_probability"] >= 0.70)
        & (stable["full_coefficient"] < 0)
    ].copy()
    gene_set = {g.upper() for g in genes}

    protective.insert(
        0,
        "ccc",
        protective.apply(
            lambda r: f"{r['sender']}->{r['receiver']}|{r['ligand']}->{r['receptor']}",
            axis=1,
        ),
    )
    protective["ligand_genes"] = protective["ligand"].map(
        lambda x: ";".join(target_genes(x))
    )
    protective["receptor_genes"] = protective["receptor"].map(
        lambda x: ";".join(target_genes(x))
    )
    protective["missing_ligand_genes"] = protective["ligand"].map(
        lambda x: ";".join(g for g in target_genes(x) if g not in gene_set)
    )
    protective["missing_receptor_genes"] = protective["receptor"].map(
        lambda x: ";".join(g for g in target_genes(x) if g not in gene_set)
    )
    protective["ligand_covered"] = protective["missing_ligand_genes"].eq("")
    protective["receptor_covered"] = protective["missing_receptor_genes"].eq("")
    protective["fully_computable"] = (
        protective["ligand_covered"] & protective["receptor_covered"]
    )
    coverage_columns = [
        "ccc",
        "sender",
        "receiver",
        "ligand",
        "receptor",
        "ligand_covered",
        "receptor_covered",
        "fully_computable",
        "ligand_genes",
        "receptor_genes",
        "missing_ligand_genes",
        "missing_receptor_genes",
        "full_coefficient",
        "sign_selection_probability",
        "feature_index",
        "lr_id",
    ]
    protective[coverage_columns].to_csv(
        processed / "protective_ccc_merfish_panel_coverage.tsv", sep="\t", index=False
    )

    missing_ligands = Counter(
        gene
        for entity in protective["ligand"]
        for gene in target_genes(entity)
        if gene not in gene_set
    )
    missing_receptors = Counter(
        gene
        for entity in protective["receptor"]
        for gene in target_genes(entity)
        if gene not in gene_set
    )
    pd.DataFrame(
        [("ligand", gene, count) for gene, count in missing_ligands.most_common()]
        + [("receptor", gene, count) for gene, count in missing_receptors.most_common()],
        columns=["role", "missing_gene", "affected_ccc_count"],
    ).to_csv(processed / "missing_lr_frequency.tsv", sep="\t", index=False)

    availability = pd.DataFrame(
        [
            ("cell_type", False, "No cell-type annotation file or column is supplied in GEO."),
            ("TLS", False, "No TLS annotation file or column is supplied in GEO."),
            (
                "tissue_region",
                False,
                "No tissue-region annotation file or column is supplied in GEO.",
            ),
        ],
        columns=["annotation", "available_from_geo", "evidence"],
    )
    availability.to_csv(processed / "annotation_availability.tsv", sep="\t", index=False)

    cell_counts = []
    for accession, sample_name in SAMPLES.items():
        metadata_file = raw / f"{accession}_{sample_name}_cell_metadata.csv.gz"
        output_file = coordinates / f"{accession}_{sample_name}_cell_coordinates.tsv.gz"
        cell_counts.append(
            {
                "accession": accession,
                "sample_name": sample_name,
                "n_cells": write_coordinates(metadata_file, output_file, accession),
                "coordinates_file": str(output_file),
            }
        )
    pd.DataFrame(cell_counts).to_csv(
        processed / "hcc_cell_coordinate_manifest.tsv", sep="\t", index=False
    )

    summary = pd.DataFrame(
        [
            ("hcc_samples", len(SAMPLES)),
            ("panel_genes", len(genes)),
            ("blank_controls", len(blanks)),
            ("protective_cccs", len(protective)),
            ("fully_computable_cccs", int(protective["fully_computable"].sum())),
        ],
        columns=["metric", "value"],
    )
    summary.to_csv(processed / "audit_summary.tsv", sep="\t", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
