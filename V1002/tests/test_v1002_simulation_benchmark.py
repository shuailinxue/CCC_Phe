import inspect
import numpy as np
from phenoniche.v1002.lr_atlas import load_lr_atlas
from phenoniche.v1002.simulation_cells import _coverage_mask, build_simulation_spec, bulk_potential, composition_truth, simulate_spatial
from phenoniche.v1002.simulation_benchmark import METHODS


def _spec():
    return build_simulation_spec(load_lr_atlas("data/commuspace_human_lr_atlas.tsv"))


def test_twin_niches_have_identical_composition_truth():
    for purity in (0.3, 0.5, 0.7):
        truth = composition_truth(purity)
        assert np.array_equal(truth[0], truth[1])
        characteristic = truth[0, [0, 1, 2]].sum()
        assert characteristic >= purity - 1e-6


def test_filtering_interface_is_observation_only():
    assert tuple(inspect.signature(_coverage_mask).parameters) == ("expression", "cell_types", "spec")
    assert METHODS == ("C-only", "I-only", "C+I")


def test_cell_level_simulation_builds_observed_views():
    spec = _spec()
    simulation = simulate_spatial(spec, 0.5, 50, 0.05, 991)
    assert simulation.coordinates.shape == (1800, 2)
    assert simulation.expression.shape == (1800, len(spec.genes))
    assert simulation.cs.shape == (1800, 8)
    assert simulation.communication.shape == (1800, 64 * len(spec.atlas))
    assert not hasattr(simulation, "os")
    assert np.allclose(simulation.cs.sum(1), 1)
    assert np.all(simulation.final_mask <= simulation.coverage_mask)


def test_bulk_potential_preserves_directed_feature_axis_without_abundance():
    spec = _spec()
    expression = np.ones((3, 8, len(spec.genes)), dtype=np.float32)
    gene_index = {gene: index for index, gene in enumerate(spec.genes)}
    for gene in spec.atlas.interactions[0].ligand_components:
        expression[:, 0, gene_index[gene]] = 4
    potential = bulk_potential(expression, spec).reshape(3, 64, len(spec.atlas))
    assert potential.shape == (3, 64, len(spec.atlas))
    assert not np.allclose(potential[:, 1, 0], potential[:, 8, 0])
