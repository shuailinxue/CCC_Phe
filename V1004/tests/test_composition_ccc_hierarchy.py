import numpy as np
import pytest

from phenoniche.v1004.composition_ccc_hierarchy import build_hierarchical_states


def test_only_observed_states_numbered_by_parent_then_substate():
    c = np.array([2, 0, 2, 0, 2, 0])
    i = np.array([1, 3, 1, 0, 4, 3])
    labels, mapping = build_hierarchical_states(c, i, 8)
    assert [(m["C_state"], m["I_state"], m["n_cells"]) for m in mapping] == [
        (0, 0, 1), (0, 3, 2), (2, 1, 2), (2, 4, 1)
    ]
    assert labels.tolist() == [3, 2, 3, 1, 4, 2]
    assert len(labels) == sum(m["n_cells"] for m in mapping)
    shuffled, remapping = build_hierarchical_states(c[::-1], i[::-1], 8)
    assert mapping == remapping
    np.testing.assert_array_equal(shuffled, labels[::-1])


def test_no_alignment_or_empty_combinations():
    labels, mapping = build_hierarchical_states([0, 0, 1], [7, 2, 7], 8)
    assert labels.tolist() == [2, 1, 3]
    assert [(m["C_state"], m["I_state"]) for m in mapping] == [(0, 2), (0, 7), (1, 7)]


@pytest.mark.parametrize("c,i", [([0], [8]), ([0, 1], [0]), ([], [])])
def test_invalid_labels_rejected(c, i):
    with pytest.raises(ValueError):
        build_hierarchical_states(c, i, 8)
