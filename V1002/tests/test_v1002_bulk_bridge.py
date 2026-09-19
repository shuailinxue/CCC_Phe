import csv
import inspect
import torch
from phenoniche.v1002.bulk_bridge import directional_communication_potential, project_fixed_niches, pseudobulk_expression
from phenoniche.v1002.lr_atlas import load_lr_atlas


def _simple_atlas(tmp_path):
    path = tmp_path / "simple.tsv"
    path.write_text("ligand\treceptor\nL1\tR1\nL2\tR2\n")
    return load_lr_atlas(path)


def test_bulk_no_spatial_information():
    parameters = set(inspect.signature(directional_communication_potential).parameters)
    assert not parameters & {"coordinates", "neighborhoods", "adjacency", "WS"}


def test_bulk_directional_potential(tmp_path):
    atlas = _simple_atlas(tmp_path)
    genes = ("L1", "R1", "L2", "R2")
    expression = torch.tensor([[[9.0, 1.0, 4.0, 1.0], [1.0, 4.0, 1.0, 9.0]]])
    potential = directional_communication_potential(expression, genes, atlas)
    assert not torch.allclose(potential[:, 0 * 2 + 1], potential[:, 1 * 2 + 0])


def test_no_abundance_double_counting(tmp_path):
    atlas = _simple_atlas(tmp_path)
    genes = ("L1", "R1", "L2", "R2")
    expression = torch.rand(2, 2, 4)
    first = directional_communication_potential(expression, genes, atlas, composition=torch.tensor([[0.9, 0.1], [0.1, 0.9]]))
    second = directional_communication_potential(expression, genes, atlas, composition=torch.full((2, 2), 0.5))
    ablated = directional_communication_potential(expression, genes, atlas, composition=torch.tensor([[0.9, 0.1], [0.1, 0.9]]),
                                                   multiply_abundance=True)
    assert torch.equal(first, second)
    assert not torch.equal(first, ablated)


def test_pseudobulk_bridge(tmp_path):
    atlas = _simple_atlas(tmp_path)
    genes = ("L1", "R1", "L2", "R2")
    samples = []
    for first in (True, False):
        if first:
            type_zero = torch.tensor([9.0, 1.0, 1.0, 1.0])
            type_one = torch.tensor([1.0, 9.0, 1.0, 1.0])
        else:
            type_zero = torch.tensor([1.0, 1.0, 1.0, 9.0])
            type_one = torch.tensor([1.0, 1.0, 9.0, 1.0])
        cells = torch.stack((type_zero, type_zero, type_one, type_one))
        samples.append(pseudobulk_expression(cells, torch.tensor([0, 0, 1, 1]), 2))
    expression = torch.stack(samples)
    potential = directional_communication_potential(expression, genes, atlas).reshape(2, -1)
    composition = torch.full((2, 2), 0.5)
    hc = torch.full((2, 2), 0.5)
    hi = potential.clone()
    combined = project_fixed_niches(composition, potential, hc, hi, steps=50, use_communication=True)
    composition_only = project_fixed_niches(composition, potential, hc, hi, steps=50, use_communication=False)
    truth = torch.eye(2)
    assert (combined - truth).square().mean() < (composition_only - truth).square().mean()
