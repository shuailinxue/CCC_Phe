"""Prepare the audited LR-rate simulation without replacing earlier results."""
from pathlib import Path
from dataclasses import replace
import json,tempfile
import numpy as np
from ..storage import STORAGE_ROOT
from .config import SimulationConfig
from .workflow import protect
from .rich_background import prepare_rich_background,simulation_audit
from .counterfactual_controls import noise_control
from .response_simulation import checksum


def prepare_coherent_background(simulation_config,profile_path):
    config=SimulationConfig.from_json(simulation_config)
    profile=json.loads(Path(profile_path).read_text())
    if profile.get('variant')!='rich_background_coherent_LR':raise ValueError('Explicit coherent LR profile required')
    if config.expression_coverage is None:raise ValueError('Coherent experiment requires a declared training coverage mask')
    rates=profile['program_rates']
    if not np.isclose(rates['mi_signal'],config.signal_strength):raise ValueError('Response exposure scale and MI expression rate disagree')
    source=Path(profile['background_source_dir']);output=protect(config)
    if output.resolve()==source.resolve():raise ValueError('Corrected data must preserve original simulation')
    original_path=source/'config.json'
    original=SimulationConfig.from_json(original_path) if original_path.exists() else replace(config,output_dir=str(source),expression_coverage=None)
    # Anatomical generation and panel must match the retained scCube background.
    for key in ('n_cells','seed','n_genes','n_lr','mi_lr_count','core_radius_um','edge_width_um','outer_radius_um','replicates','sender_type','receiver_type','program_design'):
        if getattr(original,key)!=getattr(config,key):raise ValueError('Retained background incompatible: '+key)
    if not all((source/name).exists() for name in ('before_injection.h5ad','injected.h5ad','seed_reference.h5ad','lr.csv')):
        with tempfile.TemporaryDirectory(prefix='coherent_source_',dir=STORAGE_ROOT/'tmp') as value:
            tmp=Path(value);sp=tmp/'source.json';pp=tmp/'profile.json'
            sp.write_text(json.dumps(original.to_dict()));pp.write_text(json.dumps(profile['source_background_profile']))
            prepare_rich_background(sp,pp)
    files=('before_injection.h5ad','injected.h5ad','seed_reference.h5ad','lr.csv')
    source_hashes={name:checksum(source/name) for name in files}
    manifest={'config':config.to_dict(),'profile':profile,'source_hashes':source_hashes,'implementation':{name:checksum(Path(__file__).with_name(name)) for name in ('coherent_background.py','counterfactual_controls.py')}}
    saved=output/'coherent_prepared.json'
    if saved.exists():
        complete=json.loads(saved.read_text())
        if complete['manifest']!=manifest:raise ValueError('Coherent inputs/settings changed; preserve completed results in another NAS output')
        for name,digest in complete['files'].items():
            if checksum(output/name)!=digest:raise ValueError('Coherent cached data changed: '+name)
        return output
    if (output/'injected.h5ad').exists():raise ValueError('Unverified simulation already exists; refusing to overwrite')
    after,lr=noise_control(source,output,'program_coherent',config=config,program_rates=rates)
    import anndata as ad
    before=ad.read_h5ad(output/'before_injection.h5ad')
    audit=simulation_audit(before,after,lr,dict(profile['source_background_profile'],lr_role_mean=rates['mi_baseline'],mi_fold_change=1+rates['mi_signal']/rates['mi_baseline']))
    audit.update(simulation_variant=profile['variant'],MI_injection='Independent additive latent expression rates, not observed-count multipliers',program_rates=rates,expression_coverage_threshold=config.expression_coverage,source_background=str(source),response_exposure_normalizer=config.signal_strength)
    (output/'rich_simulation_audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    for name,digest in source_hashes.items():
        if checksum(source/name)!=digest:raise RuntimeError('Original simulation mutated: '+name)
    generated={name:checksum(output/name) for name in ('before_injection.h5ad','injected.h5ad','lr.csv','rich_simulation_audit.json','provenance.json')}
    saved.write_text(json.dumps({'manifest':manifest,'files':generated},indent=2)+'\n')
    print(f'Prepared coherent LR simulation: {len(after)} cells, {after.n_vars} genes; all {len(lr)} LR retained',flush=True)
    return output
