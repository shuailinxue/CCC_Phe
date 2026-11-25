"""Post-fit audit of background splitting versus the earlier tumor-edge example.

Uses generative labels only for descriptive comparisons. No fit, clustering,
simulation generation, canonical figure or Notebook output is changed.
"""
from pathlib import Path
import hashlib
import json

import anndata as ad
import numpy as np
import pandas as pd
import torch

WORK = Path('/home/nas3/biod/xueshuailin/CCC_Phe/workspace')
SOURCE = WORK / 'data/tumor_immune_interface'
MODEL = WORK / 'outputs/tumor_immune_interface/dual_mlp_gnn'
OLD = WORK / 'outputs/tumor_edge_gene_rich_core_exterior_active_simulation'
OLD_RESULT = WORK / 'outputs/tumor_edge_gene_rich_core_exterior_active_niches'
OUTPUT = MODEL / 'representation_comparison'


def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(2**20), b''):
            value.update(block)
    return value.hexdigest()


def split_gain(values, first, second):
    """Exact SSE reduction for splitting these two prescribed sets at their means.

    This is a partition diagnostic, not the gain of a globally refitted KMeans.
    """
    left, right = values[first].astype(np.float64), values[second].astype(np.float64)
    squared = float(np.square(left.mean(0) - right.mean(0)).sum())
    gain = len(left) * len(right) / (len(left) + len(right)) * squared
    return squared, gain


def run():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    protected_paths = [SOURCE / 'tissue.h5ad', MODEL / 'gnn/checkpoint.pt',
        MODEL / 'gnn/shared.npy', MODEL / 'gnn/training.csv',
        MODEL / 'clustering/labels.npz', OLD / 'injected.h5ad',
        OLD_RESULT / 'raw_clusters.csv']
    protected = {str(path): digest(path) for path in protected_paths}
    tissue = ad.read_h5ad(SOURCE / 'tissue.h5ad')
    region = tissue.obs.design_region_id.to_numpy()
    labels = np.load(MODEL / 'clustering/labels.npz')['automatic'] + 1
    context, edge_a, edge_b = region == 0, region == 2, region == 3
    background_ids = pd.Series(labels[context]).value_counts().index[:2].to_numpy()
    first, second = [context & (labels == label) for label in background_ids]
    features = pd.read_csv(MODEL / 'features.csv')
    mask = np.load(MODEL / 'feature_mask.npy')
    scale = np.load(MODEL / 'feature_scale.npy')
    observed = np.load(MODEL / 'input/ccc.npy', mmap_mode='r')
    scaled = np.asarray(observed[:, mask], np.float64) / scale[mask]
    z = np.load(MODEL / 'gnn/Z.npy')
    loading = np.load(MODEL / 'gnn/H.npy')
    shared = np.load(MODEL / 'gnn/shared.npy')
    records = []
    representations = {'shared_h': shared, 'observed_CCC_RMS_scaled': scaled,
        'reconstructed_CCC_RMS_scaled': (z @ loading[:, mask]) / scale[mask],
        'physical_cell_composition': np.asarray(tissue.obsm['local_cell_composition'])}
    for name, values in representations.items():
        background_distance, background_gain = split_gain(values, first, second)
        edge_distance, edge_gain = split_gain(values, edge_a, edge_b)
        records.append({'representation': name,
            'background_squared_centroid_distance': background_distance,
            'AB_squared_centroid_distance': edge_distance,
            'background_to_AB_squared_distance_ratio': background_distance / max(edge_distance, 1e-20),
            'background_split_SSE_gain': background_gain, 'AB_split_SSE_gain': edge_gain,
            'background_to_AB_split_gain_ratio': background_gain / max(edge_gain, 1e-20)})
    contrast = pd.DataFrame(records)
    contrast.to_csv(OUTPUT / 'representation_contrasts.csv', index=False)

    delta = scaled[first].mean(0) - scaled[second].mean(0)
    feature_table = features.loc[mask].copy().reset_index(drop=True)
    feature_table['background_first_scaled_mean'] = scaled[first].mean(0)
    feature_table['background_second_scaled_mean'] = scaled[second].mean(0)
    feature_table['squared_contrast'] = delta**2
    feature_table['fraction_of_background_squared_contrast'] = delta**2 / np.square(delta).sum()
    feature_table.sort_values('squared_contrast', ascending=False).to_csv(
        OUTPUT / 'background_feature_contrasts.csv', index=False)
    lr_contrast = feature_table.groupby(['lr_id', 'ligand', 'receptor'], as_index=False)[
        ['squared_contrast', 'fraction_of_background_squared_contrast']].sum().sort_values(
            'squared_contrast', ascending=False)
    lr_contrast.to_csv(OUTPUT / 'background_LR_contrasts.csv', index=False)
    top = lr_contrast.iloc[0]

    members = np.load(SOURCE / 'neighborhoods.npz')['members']
    kinds = tissue.obs.cell_type.astype(str).to_numpy()
    receptor = tissue.X[:, tissue.var_names.get_loc(top.receptor)].toarray().ravel()
    t_cells = kinds == 'T cell'
    positive_t = t_cells & (receptor > 0)
    local_positive_t = positive_t[members].sum(1)
    receptor_records = []
    for label, rows in zip(background_ids, [first, second]):
        receptor_records.append({'niche': int(label), 'background_centers': int(rows.sum()),
            'fraction_without_receptor_positive_T_cell': float((local_positive_t[rows] == 0).mean()),
            'mean_receptor_positive_T_cells': float(local_positive_t[rows].mean())})
    pd.DataFrame(receptor_records).to_csv(OUTPUT / 'receptor_availability.csv', index=False)
    pd.crosstab(labels[context], kinds[context], normalize='index').to_csv(
        OUTPUT / 'background_center_cell_type_fractions.csv')

    # Check the old example's actual protocol and both published cluster outputs.
    checkpoint = torch.load(OLD / 'gnn/checkpoint.pt', map_location='cpu', weights_only=False)
    old_features = pd.DataFrame(checkpoint['features'])
    old_mask = np.asarray(checkpoint['feature_mask'])
    old_labels = pd.read_csv(OLD_RESULT / 'raw_clusters.csv')
    old_tissue = ad.read_h5ad(OLD / 'injected.h5ad', backed='r')
    try:
        old_truth = old_tissue.obs.true_niche.to_numpy()
        for name in ['CCC_Phe GNN (fixed K)', 'CCC_Phe GNN (selected K)']:
            pd.crosstab(old_truth, old_labels[name]).to_csv(OUTPUT / (
                'old_fixed_K_crosstab.csv' if 'fixed' in name else 'old_selected_K_crosstab.csv'))
    finally:
        old_tissue.file.close()
    old_summary = json.loads((OLD_RESULT / 'summary.json').read_text())
    old_metrics = pd.read_csv(OLD_RESULT / 'metrics.csv')
    old_metrics.to_csv(OUTPUT / 'old_reported_metrics.csv', index=False)
    report = {'analysis': 'Read-only post-fit descriptive audit; no model or canonical prediction changes',
        'background_groups': [int(value) for value in background_ids],
        'background_group_sizes': [int(first.sum()), int(second.sum())],
        'cold_interface_sizes': [int(edge_a.sum()), int(edge_b.sum())],
        'comparison_caveat': 'Background groups were selected by current h-based clustering; this is not an independent signal benchmark or causal ablation',
        'SSE_caveat': 'Exact two-set centroid split identity; other partitions held fixed, not a globally optimized KMeans counterfactual',
        'dominant_background_LR': top.to_dict(),
        'receptor_T_cell_expression_coverage': float((receptor[t_cells] > 0).mean()),
        'receptor_background_T_cell_expression_coverage': float((receptor[t_cells & context] > 0).mean()),
        'old_retained_features_for_same_LR': old_features.loc[
            old_mask & (old_features.lr_id == top.lr_id)].to_dict(orient='records'),
        'old_effective_CCC_features': int(old_mask.sum()), 'current_effective_CCC_features': int(mask.sum()),
        'old_model': {'lambda_comp': checkpoint['config']['lambda_comp'],
            'clustering_representation': 'Z @ dictionary_program_projection(H, feature_scale), then 20% physical-neighbor mixing',
            'fixed_K': 4, 'automatic_K': old_summary['selected_K'],
            'joint_model_accepted': old_summary['joint_model_accepted']},
        'current_model': {'clustering_representation': 'raw shared h before independent CCC and composition MLPs',
            'K': int(len(np.unique(labels))), 'lambda_comp': 1.0},
        'unchanged_input_sha256': protected}
    assert protected == {str(path): digest(path) for path in protected_paths}
    (OUTPUT / 'audit.json').write_text(json.dumps(report, indent=2) + '\n')
    print(contrast.to_string(index=False))
    print(lr_contrast.head(3).to_string(index=False))
    print(pd.DataFrame(receptor_records).to_string(index=False))
    print('All source data, checkpoints, representations and canonical labels unchanged.')
    return OUTPUT


if __name__ == '__main__':
    print(run())
