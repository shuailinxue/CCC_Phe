from dataclasses import dataclass
import csv
import hashlib
from pathlib import Path
import re
import numpy as np


@dataclass(frozen=True)
class LRInteraction:
    lr_id: str
    ligand: str
    receptor: str
    ligand_components: tuple[str, ...]
    receptor_components: tuple[str, ...]


@dataclass(frozen=True)
class LRAtlas:
    interactions: tuple[LRInteraction, ...]
    audit: dict
    source_path: str

    def __len__(self):
        return len(self.interactions)


def canonicalize_gene(value):
    gene = re.sub(r"\s+", "", str(value).upper())
    if not gene or gene in {"NA", "NAN", "NONE"}:
        raise ValueError("Gene symbol is empty")
    return gene


def canonicalize_complex(value):
    components = [canonicalize_gene(part) for part in re.split(r"[_,+&|;]+", str(value)) if part.strip()]
    if not components:
        raise ValueError("Complex has no gene components")
    return tuple(sorted(set(components)))


def _column(fieldnames, candidates):
    lookup = {name.lower().strip(): name for name in fieldnames}
    return next((lookup[name] for name in candidates if name in lookup), None)


def _delimiter(path):
    if path.suffix.lower() in {".tsv", ".txt"}:
        return "\t"
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        sample = handle.read(4096)
    return csv.Sniffer().sniff(sample, delimiters=",\t").delimiter


def _checksum(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_lr_atlas(path):
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"LR atlas does not exist: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter=_delimiter(path))
        if reader.fieldnames is None:
            raise ValueError("LR atlas has no header")
        ligand_column = _column(reader.fieldnames, ("ligand", "from", "source_genesymbol", "ligand_gene"))
        receptor_column = _column(reader.fieldnames, ("receptor", "to", "target_genesymbol", "receptor_gene"))
        id_column = _column(reader.fieldnames, ("lr_id", "interaction_id", "pair_id"))
        if ligand_column is None or receptor_column is None:
            raise ValueError("LR atlas must contain ligand and receptor columns")
        rows = list(reader)
    unique = {}
    for row in rows:
        try:
            ligand_components = canonicalize_complex(row[ligand_column])
            receptor_components = canonicalize_complex(row[receptor_column])
        except ValueError:
            continue
        key = (ligand_components, receptor_components)
        if key not in unique:
            unique[key] = row.get(id_column, "").strip() if id_column else ""
    interactions = []
    for index, ((ligand_components, receptor_components), original_id) in enumerate(sorted(unique.items()), 1):
        interactions.append(LRInteraction(original_id or f"CommuSpace_HS_{index:05d}",
                                          "&".join(ligand_components), "&".join(receptor_components),
                                          ligand_components, receptor_components))
    complex_count = sum(len(row.ligand_components) > 1 or len(row.receptor_components) > 1 for row in interactions)
    audit = {"status": "loaded", "source": str(path), "sha256": _checksum(path), "n_raw_lr": len(rows),
             "n_unique_lr": len(interactions), "n_complex_lr": complex_count,
             "n_single_gene_lr": len(interactions) - complex_count, "exact_duplicates_removed": len(rows) - len(interactions),
             "directionality_preserved": True}
    return LRAtlas(tuple(interactions), audit, str(path))


def make_lr_subsets(atlas, small=100, medium=1000, seed=20261002):
    if not isinstance(atlas, LRAtlas) or len(atlas) < 1:
        raise ValueError("A nonempty LRAtlas is required")
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(atlas))
    small_indices = order[:min(small, len(order))]
    medium_indices = order[:min(medium, len(order))]
    select = lambda indices, name: LRAtlas(tuple(atlas.interactions[int(index)] for index in indices),
                                            {**atlas.audit, "subset": name, "subset_seed": seed,
                                             "n_unique_lr": len(indices)}, atlas.source_path)
    return {"Small": select(small_indices, "Small"), "Medium": select(medium_indices, "Medium"), "Full": atlas}


def discover_lr_atlas(project_root):
    root = Path(project_root).resolve()
    local_candidates = [root / "data/commuspace_human_lr_atlas.tsv",
                        root / "data/lr_atlas/commuspace_human_lr_atlas.tsv",
                        root.parent / "commuspace_human_lr_atlas.tsv"]
    audited_external = Path("/home/nas3/biod/xueshuailin/CCC_Phe/results/xenium_hbc1_atlas_ridge/00_inputs/commuspace_human_lr_atlas.tsv")
    existing = next((path for path in local_candidates if path.is_file()), None)
    return existing, [str(path) for path in (*local_candidates, audited_external)]


def feature_coordinates(index, cell_types, lr_count):
    features = cell_types * cell_types * lr_count
    if not isinstance(index, int) or not 0 <= index < features:
        raise ValueError("Feature index is outside the flattened axis")
    pair, lr_index = divmod(index, lr_count)
    sender, receiver = divmod(pair, cell_types)
    return sender, receiver, lr_index


def feature_index(sender, receiver, lr_index, cell_types, lr_count):
    if not (0 <= sender < cell_types and 0 <= receiver < cell_types and 0 <= lr_index < lr_count):
        raise ValueError("Feature coordinates are outside the flattened axis")
    return (sender * cell_types + receiver) * lr_count + lr_index
