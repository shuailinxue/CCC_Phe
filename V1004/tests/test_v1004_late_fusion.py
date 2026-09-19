import numpy as np

from phenoniche.v1004.late_fusion import align_partitions, apply_alignment, build_joint_niches


def test_hungarian_recovers_known_permutation_without_changing_cells():
    c = np.repeat(np.arange(3), 4)
    i = np.array([2, 2, 2, 2, 0, 0, 0, 0, 1, 1, 1, 1])
    w_i = np.eye(3, dtype=np.float32)[i]
    alignment = align_partitions(c, i, 3)
    aligned, w_aligned = apply_alignment(i, w_i, alignment)
    np.testing.assert_array_equal(alignment.i_to_c, [1, 2, 0])
    np.testing.assert_array_equal(alignment.i_column_for_c, [2, 0, 1])
    np.testing.assert_array_equal(aligned, c)
    np.testing.assert_array_equal(w_aligned.argmax(1), aligned)
    np.testing.assert_array_equal(w_aligned[:, alignment.i_to_c], w_i)
    assert len(aligned) == len(i) and int(alignment.after.sum()) == len(c)
    assert alignment.overall_agreement == 1.0


def test_joint_niches_use_only_observed_states_and_reproducible_order():
    c = np.repeat(np.arange(8), 2)
    same, same_mapping = build_joint_niches(c, c.copy(), 8)
    assert len(same_mapping) == 8
    assert all(row["concordant"] for row in same_mapping)
    i = c.copy()
    i[0] = 1  # one observed discordant state, while (0, 0) remains present
    final, mapping = build_joint_niches(c, i, 8)
    assert len(mapping) == 9
    assert mapping[-1]["final_niche_id"] == "Final Niche 9"
    assert (mapping[-1]["C_state"], mapping[-1]["I_state_aligned"]) == (0, 1)
    assert not mapping[-1]["concordant"]
    assert sum(row["n_cells"] for row in mapping) == len(c)
    order = np.arange(len(c))[::-1]
    permuted, permuted_mapping = build_joint_niches(c[order], i[order], 8)
    assert mapping == permuted_mapping
    np.testing.assert_array_equal(final[order], permuted)
    assert set(np.unique(final)) == set(range(9))
    assert len(same) == len(final)
