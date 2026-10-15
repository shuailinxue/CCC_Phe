"""Explicit core/exterior activation scenario; retain the ramped source intact.

Only physical expression is augmented, before the unchanged CCC adapter runs.
No model, cluster labels, CCC targets or evaluation score enter preparation.
"""
from dataclasses import replace
from pathlib import Path
import json
import tempfile

import anndata as ad
import numpy as np
import pandas as pd

from ..storage import STORAGE_ROOT
from .config import SimulationConfig
from .coherent_background import prepare_coherent_background
from .programs import program_specs
from .response_simulation import checksum
from .rich_background import simulation_audit
from .workflow import protect


def inject_boundary_activity(adata, lr, config, floor, signal_rate):
    """Poisson LR increments with the original shared cell/program state.

    MI-3/4 have floor + (1-floor)*old activity in their own physical regions.
    MI-1/2, geometry, cell labels and all other gene counts stay exact.
    """
    if config.mi_count != 4 or config.mode != 'expression':
        raise ValueError('Boundary activation requires four expression programs')
    if not np.isfinite(floor) or not 0 <= floor <= 1 or not np.isfinite(signal_rate) or signal_rate <= 0:
        raise ValueError('Invalid boundary floor or expression rate')
    result = adata.copy()
    old = np.asarray(adata.obsm['designed_MI_activity'], dtype=np.float32)
    states = np.asarray(adata.obsm['synthetic_MI_cell_state'])
    if old.shape != (len(adata), 4) or states.shape != old.shape:
        raise ValueError('Missing/alignment error in source program activities')
    if any(not np.isfinite(v).all() or (v < 0).any() for v in (old, states)):
        raise ValueError('Invalid source program activities')
    activity = old.copy()
    radius = adata.obs.radius_um.to_numpy()
    core = radius < config.core_radius_um
    exterior = radius >= config.core_radius_um + config.edge_width_um
    activity[core, 2] = floor + (1 - floor) * old[core, 2]
    activity[exterior, 3] = floor + (1 - floor) * old[exterior, 3]
    original = np.asarray(adata.X)
    counts = original.copy()
    genes = {g: i for i, g in enumerate(adata.var_names)}
    library = adata.obs.synthetic_library_factor.to_numpy()
    if not np.isfinite(library).all() or (library <= 0).any():
        raise ValueError('Invalid library factors')
    types = adata.obs.cell_type.to_numpy()
    rng = np.random.default_rng(config.seed + 6831)
    affected = []
    for spec in program_specs(config)[2:]:
        j = spec['MI'] - 1
        links = lr[lr.mi == spec['MI']]
        if len(links) != config.mi_lr_count or not (links.loading > 0).all():
            raise ValueError('Missing/invalid core or exterior LR loadings')
        for side, role in [('ligand', spec['sender']), ('receptor', spec['receiver'])]:
            cells = np.flatnonzero(types == role)
            for gene, loading in zip(links[side], links.loading):
                g = genes[gene]
                delta = signal_rate * states[cells, j] * library[cells] * (activity[cells, j] - old[cells, j]) * loading / links.loading.max()
                if (delta < 0).any():
                    raise ValueError('Boundary control cannot lower source activity')
                counts[cells, g] += rng.poisson(delta)
                affected.append(g)
    other = np.setdiff1d(np.arange(adata.n_vars), affected)
    np.testing.assert_array_equal(counts[:, other], original[:, other])
    if not np.isfinite(counts).all() or (counts < original).any() or not np.equal(counts, np.floor(counts)).all():
        raise ValueError('Boundary injection must produce finite integer counts')
    result.X = counts
    result.obsm['designed_MI_activity'] = activity
    # The supported source is the LR-expression scenario, whose role activity
    # matrices equal its design; reject a spatial-pairing source silently reused.
    for key in ('injected_sender_activity', 'injected_receiver_activity'):
        np.testing.assert_array_equal(adata.obsm[key], old)
        result.obsm[key] = activity.copy()
    audit = {'core_exterior_activity_floor': float(floor),
             'changed_gene_count': len(set(affected)),
             'added_RNA_count_fraction': float((counts - original).sum() / max(original.sum(), 1)),
             'expression_increment_seed': config.seed + 6831,
             'MI_1_2_activity_unchanged': True, 'non_LR_expression_unchanged': True,
             'coordinates_types_anatomical_labels_unchanged': True,
             'CCC_recomputed_by_original_adapter': True,
             'training_or_clustering_used_for_generation': False}
    result.uns['boundary_activation'] = audit
    return result, audit


def prepare_boundary_background(simulation_config, profile_path):
    """Restartable entry for the declared boundary profile and fresh outputs."""
    config = SimulationConfig.from_json(simulation_config)
    profile = json.loads(Path(profile_path).read_text())
    if 'core_exterior_activity_floor' not in profile:
        return prepare_coherent_background(simulation_config, profile_path)
    floor = profile['core_exterior_activity_floor']
    if not isinstance(floor, (int, float)) or not np.isfinite(floor) or not 0 <= floor <= 1:
        raise ValueError('Boundary floor must lie in [0,1]')
    source = Path(profile['boundary_source_dir'])
    if source.is_symlink() or not source.resolve().is_relative_to(STORAGE_ROOT.resolve()):
        raise ValueError('Boundary source must be a direct NAS path')
    output = Path(config.output_dir)
    if output.resolve() == source.resolve() or output.resolve().is_relative_to(source.resolve()) or source.resolve().is_relative_to(output.resolve()):
        raise ValueError('Boundary data must preserve the source in a separate NAS directory')
    output = protect(config)
    source_config = replace(config, output_dir=str(source))
    if (source / 'config.json').exists():
        original = SimulationConfig.from_json(source / 'config.json')
        if replace(original, output_dir=config.output_dir).to_dict() != config.to_dict():
            raise ValueError('Boundary scenario must retain the original simulation/model settings')
        source_config = original
    source_profile = {k: v for k, v in profile.items() if k not in ('core_exterior_activity_floor', 'boundary_source_dir')}
    (STORAGE_ROOT / 'tmp').mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='boundary_source_', dir=STORAGE_ROOT / 'tmp') as name:
        tmp = Path(name)
        sp, pp = tmp / 'simulation.json', tmp / 'profile.json'
        sp.write_text(json.dumps(source_config.to_dict()))
        pp.write_text(json.dumps(source_profile))
        prepare_coherent_background(sp, pp)
    files = ('before_injection.h5ad', 'injected.h5ad', 'lr.csv', 'provenance.json', 'rich_simulation_audit.json')
    hashes = {name: checksum(source / name) for name in files}
    manifest = {'config': config.to_dict(), 'profile': profile, 'source_sha256': hashes,
                'implementation_sha256': checksum(Path(__file__))}
    saved = output / 'boundary_prepared.json'
    if saved.exists():
        previous = json.loads(saved.read_text())
        if previous['manifest'] != manifest:
            raise ValueError('Boundary inputs/settings changed; preserve the completed NAS scenario')
        for name, digest in previous['files'].items():
            if checksum(output / name) != digest:
                raise ValueError('Boundary cached data changed: ' + name)
        return output
    if (output / 'injected.h5ad').exists():
        raise ValueError('Unverified boundary simulation exists; refusing to overwrite')
    a = ad.read_h5ad(source / 'injected.h5ad')
    lr = pd.read_csv(source / 'lr.csv')
    after, change = inject_boundary_activity(a, lr, config, floor, profile['program_rates']['mi_signal'])
    before = ad.read_h5ad(source / 'before_injection.h5ad')
    audit_profile = dict(profile['source_background_profile'], lr_role_mean=profile['program_rates']['mi_baseline'],
                         mi_fold_change=1 + profile['program_rates']['mi_signal'] / profile['program_rates']['mi_baseline'])
    audit = json.loads((source / 'rich_simulation_audit.json').read_text())
    # Recompute count fractions for this actual expression scenario.
    audit.update(simulation_audit(before, after, lr, audit_profile))
    audit.update(boundary_activation=change, simulation_variant='rich_background_core_exterior_active',
                 source_coherent_background=str(source), response_exposure_normalizer=config.signal_strength)
    before.write_h5ad(output / 'before_injection.h5ad')
    after.write_h5ad(output / 'injected.incomplete.h5ad')
    (output / 'injected.incomplete.h5ad').replace(output / 'injected.h5ad')
    lr.to_csv(output / 'lr.csv', index=False)
    (output / 'rich_simulation_audit.json').write_text(json.dumps(audit, indent=2) + '\n')
    provenance = json.loads((source / 'provenance.json').read_text())
    provenance.update(boundary_activation=change, boundary_source=str(source), source_sha256=hashes)
    (output / 'provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')
    for name, digest in hashes.items():
        if checksum(source / name) != digest:
            raise RuntimeError('Original coherent source changed: ' + name)
    generated = {name: checksum(output / name) for name in files}
    saved.write_text(json.dumps({'manifest': manifest, 'files': generated}, indent=2) + '\n')
    print(f'Prepared core/exterior activity floor={floor:g}; fresh CCC and all models will fit this expression scenario', flush=True)
    return output
