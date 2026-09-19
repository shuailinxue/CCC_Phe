import inspect
import torch
from phenoniche.v1002.bulk_bridge import directional_communication_potential
from phenoniche.v1002.lr_atlas import LRAtlas, LRInteraction
from phenoniche.v1002.selected_ccc import select_top_union, selected_directional_potential


def _atlas():
    rows = tuple(LRInteraction(f"L{i}->R{i}", f"L{i}", f"R{i}", (f"L{i}",), (f"R{i}",)) for i in range(5))
    return LRAtlas(rows, {"n_raw_lr": 5, "n_unique_lr": 5}, "toy")


def test_top100_union_uses_only_hi():
    hi = torch.zeros(3, 250)
    hi[0, :100] = 3
    hi[1, 50:150] = 2
    hi[2, 125:225] = 1
    selected = select_top_union(hi, 100)
    assert set(selected.tolist()) == set(range(225))
    assert tuple(inspect.signature(select_top_union).parameters) == ("hi", "top_m")


def test_selected_potential_matches_full_feature_axis():
    atlas = _atlas()
    genes = tuple([f"L{i}" for i in range(5)] + [f"R{i}" for i in range(5)])
    torch.manual_seed(7)
    expression = torch.rand(4, 3, len(genes))
    selected = torch.tensor([0, 7, 13, 29, 41])
    full = directional_communication_potential(expression, genes, atlas).reshape(4, -1)
    observed = selected_directional_potential(expression, genes, atlas, selected, chunk=2)
    assert torch.allclose(observed, full[:, selected])
