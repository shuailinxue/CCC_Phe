import csv
import torch
from phenoniche.v1002.bulk_bridge import aggregate_complex_state, directional_communication_potential
from phenoniche.v1002.lr_atlas import feature_coordinates, feature_index, load_lr_atlas


def _atlas(tmp_path):
    path = tmp_path / "atlas.tsv"
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(("lr_id", "ligand", "receptor"))
        writer.writerow(("one", "a_b", "c"))
        writer.writerow(("duplicate", "B&A", "C"))
        writer.writerow(("reverse", "c", "a_b"))
        writer.writerow(("single", "d", "e"))
    return load_lr_atlas(path)


def test_lr_atlas_deduplication(tmp_path):
    atlas = _atlas(tmp_path)
    assert atlas.audit["n_raw_lr"] == 4
    assert atlas.audit["n_unique_lr"] == 3
    assert atlas.audit["exact_duplicates_removed"] == 1
    assert atlas.audit["n_complex_lr"] == 2


def test_lr_directionality(tmp_path):
    atlas = _atlas(tmp_path)
    keys = {(row.ligand_components, row.receptor_components) for row in atlas.interactions}
    assert (("A", "B"), ("C",)) in keys
    assert (("C",), ("A", "B")) in keys


def test_complex_lr_aggregation():
    expression = torch.tensor([[[4.0, 9.0, 2.0]]])
    value = aggregate_complex_state(expression, {"A": 0, "B": 1, "C": 2}, ("A", "B"))
    assert torch.allclose(value, torch.tensor([[6.0]]))


def test_large_feature_mapping():
    cell_types, lr_count = 8, 9311
    for sender, receiver, lr_index in ((0, 0, 0), (3, 7, 5000), (7, 7, 9310)):
        index = feature_index(sender, receiver, lr_index, cell_types, lr_count)
        assert feature_coordinates(index, cell_types, lr_count) == (sender, receiver, lr_index)


def test_st_bulk_feature_axis_identical(tmp_path):
    atlas = _atlas(tmp_path)
    expression = torch.ones(2, 3, 5)
    potential = directional_communication_potential(expression, ("A", "B", "C", "D", "E"), atlas)
    assert potential.shape == (2, 3 * 3, len(atlas))
    assert potential.reshape(2, -1).shape[1] == 3 * 3 * len(atlas)
