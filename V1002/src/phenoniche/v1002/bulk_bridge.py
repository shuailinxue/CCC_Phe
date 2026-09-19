import math
import torch
from phenoniche.inference.bulk import infer_bulk_activities
from phenoniche.v1002.lr_atlas import LRAtlas


def aggregate_complex_state(expression, gene_index, components):
    if expression.ndim != 3:
        raise ValueError("Expression must have shape sample by cell type by gene")
    indices = [gene_index[component] for component in components]
    selected = expression[:, :, indices]
    return selected.prod(2).pow(1 / len(indices))


def directional_communication_potential(expression, gene_names, atlas, composition=None,
                                        multiply_abundance=False, lr_start=0, lr_stop=None):
    if not isinstance(atlas, LRAtlas):
        raise ValueError("atlas must be an LRAtlas")
    if expression.ndim != 3 or expression.shape[2] != len(gene_names):
        raise ValueError("Expression and gene names do not align")
    if (expression < 0).any() or not torch.isfinite(expression).all():
        raise ValueError("Expression must be finite and nonnegative")
    if multiply_abundance and composition is None:
        raise ValueError("Composition is required for the abundance ablation")
    gene_index = {str(gene).upper(): index for index, gene in enumerate(gene_names)}
    stop = len(atlas) if lr_stop is None else min(lr_stop, len(atlas))
    interactions = atlas.interactions[lr_start:stop]
    missing = sorted({gene for row in interactions for gene in row.ligand_components + row.receptor_components if gene not in gene_index})
    if missing:
        raise ValueError(f"Expression lacks {len(missing)} atlas genes")
    values = []
    for interaction in interactions:
        ligand = aggregate_complex_state(expression, gene_index, interaction.ligand_components)
        receptor = aggregate_complex_state(expression, gene_index, interaction.receptor_components)
        potential = torch.sqrt(ligand[:, :, None] * receptor[:, None, :])
        if multiply_abundance:
            potential = potential * composition[:, :, None] * composition[:, None, :]
        values.append(potential)
    if not values:
        return expression.new_empty((expression.shape[0], expression.shape[1] * expression.shape[1], 0))
    stacked = torch.stack(values, dim=3)
    return stacked.reshape(expression.shape[0], expression.shape[1] * expression.shape[1], len(interactions))


def pseudobulk_expression(cell_expression, cell_types, cell_type_count, weights=None):
    if cell_expression.ndim != 2 or cell_types.ndim != 1 or len(cell_types) != len(cell_expression):
        raise ValueError("Cell expression and cell types must align")
    if weights is None:
        weights = torch.ones(len(cell_types), dtype=cell_expression.dtype, device=cell_expression.device)
    result = cell_expression.new_zeros((cell_type_count, cell_expression.shape[1]))
    denominator = cell_expression.new_zeros(cell_type_count)
    result.index_add_(0, cell_types, cell_expression * weights[:, None])
    denominator.index_add_(0, cell_types, weights)
    return result / denominator.clamp_min(torch.finfo(result.dtype).tiny)[:, None]


def project_fixed_niches(composition, potential, hc, hi, steps=50, use_communication=True):
    if potential.ndim == 3:
        potential = potential.reshape(potential.shape[0], -1)
    communication_weight = 1 / potential.shape[1] if use_communication else 0.0
    return infer_bulk_activities(composition, potential, hc, hi, lambda_bc=1 / composition.shape[1],
                                 lambda_bi=communication_weight, steps=steps, create_graph=False)


def atlas_gene_vocabulary(atlas):
    return tuple(sorted({gene for row in atlas.interactions for gene in row.ligand_components + row.receptor_components}))


def simulate_cell_type_expression(wb, niche_ligand_state, niche_receptor_state, atlas, noise_fraction=0.005, seed=20261002):
    if wb.ndim != 2 or niche_ligand_state.ndim != 3 or niche_ligand_state.shape != niche_receptor_state.shape:
        raise ValueError("Bulk activities and niche molecular states do not align")
    if niche_ligand_state.shape[0] != wb.shape[1] or niche_ligand_state.shape[2] != len(atlas):
        raise ValueError("Niche molecular states must have shape K by C by LR")
    genes = atlas_gene_vocabulary(atlas)
    gene_index = {gene: index for index, gene in enumerate(genes)}
    expression = wb.new_zeros((wb.shape[0], niche_ligand_state.shape[1], len(genes)))
    counts = wb.new_zeros(len(genes))
    ligand_patient = torch.einsum("pk,kcl->pcl", wb, niche_ligand_state)
    receptor_patient = torch.einsum("pk,kcl->pcl", wb, niche_receptor_state)
    for lr_index, interaction in enumerate(atlas.interactions):
        for gene in interaction.ligand_components:
            expression[:, :, gene_index[gene]] += ligand_patient[:, :, lr_index]
            counts[gene_index[gene]] += 1
        for gene in interaction.receptor_components:
            expression[:, :, gene_index[gene]] += receptor_patient[:, :, lr_index]
            counts[gene_index[gene]] += 1
    expression /= counts.clamp_min(1)[None, None, :]
    generator = torch.Generator(device=expression.device).manual_seed(seed)
    noise = torch.randn(expression.shape, generator=generator, device=expression.device, dtype=expression.dtype)
    expression = torch.clamp_min(expression + noise_fraction * expression.square().mean().sqrt() * noise, 0)
    return expression, genes
