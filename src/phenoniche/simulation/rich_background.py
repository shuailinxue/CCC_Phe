"""Independent gene-rich, moderate-LR simulation; old data/models stay intact.

Real LR identities, synthetic expression assumptions. No fabricated regulatory
prior and no promise that CCC_Phe must win. The exact CCC implementation and
the existing GNN/decoder/benchmark interfaces are reused after preparation.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import anndata as ad
from .generation import TYPES,make_lr,generate
from .programs import program_specs,spatial_activity
from .config import SimulationConfig
from .workflow import protect
from .response_simulation import checksum


def validate_profile(config,profile):
    positive=('type_genes_per_type','state_modules','genes_per_state','type_on_mean','type_off_mean','housekeeping_mean','state_mean','state_sd','library_sd','dispersion','lr_role_mean','lr_offtype_mean','mi_fold_change')
    if any(profile[k]<=0 for k in positive):raise ValueError('Positive background/expression parameters required')
    if any(not isinstance(profile[k],int) for k in ('type_genes_per_type','state_modules','genes_per_state')):raise ValueError('Integer background module sizes required')
    if profile['type_on_mean']<=profile['type_off_mean'] or profile['mi_fold_change']<=1:raise ValueError('Require type contrast and MI fold >1')
    needed=2*config.n_lr+25+5*profile['type_genes_per_type']+profile['state_modules']*profile['genes_per_state']
    if needed>=config.n_genes:raise ValueError('Gene panel must also contain housekeeping genes')
    if config.mode!='expression':raise ValueError('This experiment tests sparse expression changes, not pairing-only injection')


def rich_reference(config,profile,lr):
    """NB reference with broad type programs, shared states and housekeeping."""
    validate_profile(config,profile);rng=np.random.default_rng(config.seed+3721)
    genes=sorted(set(lr.ligand)|set(lr.receptor));records=[{'kind':'LR','module':-1,'type_role':''} for _ in genes]
    for j in range(25):genes.append(f'SYN_MARKER_{j}');records.append({'kind':'marker','module':j//5,'type_role':TYPES[j//5]})
    for t in range(5):
        for j in range(profile['type_genes_per_type']):genes.append(f'SYN_TYPE_{t}_{j:04d}');records.append({'kind':'type_program','module':t,'type_role':TYPES[t]})
    for k in range(profile['state_modules']):
        for j in range(profile['genes_per_state']):genes.append(f'SYN_STATE_{k}_{j:04d}');records.append({'kind':'shared_state','module':k,'type_role':''})
    while len(genes)<config.n_genes:genes.append(f'SYN_HOUSE_{len(genes):04d}');records.append({'kind':'housekeeping','module':-1,'type_role':''})
    var=pd.DataFrame(records,index=genes);means=np.full((5,len(genes)),profile['housekeeping_mean'],np.float32)
    lr_positions=np.flatnonzero(var.kind.to_numpy()=='LR');means[:,lr_positions]=.3
    lookup={g:i for i,g in enumerate(genes)}
    for t,name in enumerate(TYPES):
        for row in lr[lr.basal_type==name].itertuples():
            positions=[lookup[row.ligand],lookup[row.receptor]];means[:,positions]=config.basal_lr_offtype_mean;means[t,positions]=config.basal_lr_mean
    for spec in program_specs(config):
        for role,names in ((spec['sender'],lr[lr.mi==spec['MI']].ligand),(spec['receiver'],lr[lr.mi==spec['MI']].receptor)):
            positions=[lookup[g] for g in names];means[:,positions]=profile['lr_offtype_mean'];means[TYPES.index(role),positions]=profile['lr_role_mean']
    for t in range(5):
        marker=np.flatnonzero((var.kind.to_numpy()=='marker')&(var.module.to_numpy()==t));means[:,marker]=.3;means[t,marker]=12.
        group=np.flatnonzero((var.kind.to_numpy()=='type_program')&(var.module.to_numpy()==t))
        means[:,group]=profile['type_off_mean']*rng.lognormal(0,.3,(5,len(group)))
        means[t,group]=profile['type_on_mean']*rng.lognormal(0,.3,len(group))
    state=np.flatnonzero(var.kind.to_numpy()=='shared_state')
    # State genes also have type-dependent baselines; shared latent states
    # are independent of anatomy/MI and cannot mark A/B by construction.
    means[:,state]=profile['state_mean']*rng.lognormal(0,.45,(5,len(state)))
    n=config.reference_cells_per_type;types=np.repeat(np.arange(5),n);rates=means[types].copy()
    latent=rng.normal(0,profile['state_sd'],(len(types),profile['state_modules']))
    for k in range(profile['state_modules']):
        ids=np.flatnonzero((var.kind.to_numpy()=='shared_state')&(var.module.to_numpy()==k));rates[:,ids]*=np.exp(latent[:,k,None]-.5*profile['state_sd']**2)
    library=rng.lognormal(-.5*profile['library_sd']**2,profile['library_sd'],len(types));rates*=library[:,None]
    dispersion=profile['dispersion'];counts=rng.negative_binomial(dispersion,dispersion/(dispersion+rates)).astype(np.float32)
    obs=pd.DataFrame({'Cell_type':np.asarray(TYPES)[types],'Cell':[f'rich_seed_{i}' for i in range(len(types))]},index=[f'rich_seed_{i}' for i in range(len(types))])
    return ad.AnnData(counts,obs=obs,var=var)


def add_background_variability(a,profile,seed):
    """Shared module/library/overdispersion variation, exactly paired across A/B."""
    result=a.copy();n=len(a)//2;rng=np.random.default_rng(seed+771)
    if not np.array_equal(a.obs.mirror_pair.to_numpy()[:n],a.obs.mirror_pair.to_numpy()[n:]):raise ValueError('Mirror order mismatch')
    rates=np.asarray(a.X[:n],np.float32).copy();states=rng.normal(0,profile['state_sd'],(n,profile['state_modules']))
    for k in range(profile['state_modules']):
        ids=np.flatnonzero((a.var.kind.to_numpy()=='shared_state')&(a.var.module.to_numpy()==k));rates[:,ids]*=np.exp(states[:,k,None]-.5*profile['state_sd']**2)
    library=rng.lognormal(-.5*profile['library_sd']**2,profile['library_sd'],n);rates*=library[:,None]
    # NB resampling adds overdispersion to the calibrated scCube count mean.
    dispersion=profile['dispersion'];half=rng.negative_binomial(dispersion,dispersion/(dispersion+rates)).astype(np.float32)
    result.X=np.tile(half,(2,1));result.obsm['synthetic_background_state']=np.tile(states.astype(np.float32),(2,1))
    result.obs['synthetic_library_factor']=np.tile(library,2)
    return result


def inject_moderate_mi(a,lr,config,profile):
    """Poisson count increments with conditional expected fold ≤ declared fold.

    All non-LR genes remain exact; expression/spatial inputs are perturbed
    before CCC calculation. Anatomy is permitted for simulation design only.
    """
    result=a.copy();x=np.asarray(a.X,np.float32).copy();activity=spatial_activity(a,config);rng=np.random.default_rng(config.seed+9183)
    lookup={g:i for i,g in enumerate(a.var_names)};types=a.obs.cell_type.to_numpy()
    for spec in program_specs(config):
        j=spec['MI']-1;links=lr[lr.mi==spec['MI']]
        for role,names in ((spec['sender'],links.ligand),(spec['receiver'],links.receptor)):
            rows=np.flatnonzero(types==role)
            for gene,loading in zip(names,links.loading):
                g=lookup[gene];relative=loading/max(links.loading)
                rate=(profile['mi_fold_change']-1)*np.asarray(a.X[rows,g])*activity[rows,j]*relative
                x[rows,g]+=rng.poisson(rate).astype(np.float32)
    result.X=x;result.obsm['injected_sender_activity']=activity.copy();result.obsm['injected_receiver_activity']=activity.copy();result.obsm['designed_MI_activity']=activity
    result.uns['injection_mode']='expression';result.uns['simulation_variant']='rich_background_moderate_LR'
    result.uns['rich_background_profile']=profile
    return result


def simulation_audit(before,after,lr,profile):
    lookup={g:i for i,g in enumerate(after.var_names)};lr_genes=set(lr.ligand)|set(lr.receptor);edge_genes=set(lr.loc[lr.mi.isin([1,2]),'ligand'])|set(lr.loc[lr.mi.isin([1,2]),'receptor'])
    ids=[lookup[g] for g in lr_genes];edge_ids=[lookup[g] for g in edge_genes];other=[i for i,g in enumerate(after.var_names) if g not in lr_genes]
    n=len(after)//2;raw=np.asarray(after.X);edge=np.isin(after.obs.true_niche,[2,3])
    if not np.array_equal(before.X[:,other],raw[:,other]):raise AssertionError('MI injection altered non-LR genes')
    if not np.array_equal(raw[:n,other],raw[n:,other]):raise AssertionError('Background A/B match lost')
    if not np.array_equal(before.obs.cell_type.to_numpy()[:n],before.obs.cell_type.to_numpy()[n:]) or not np.allclose(before.obs.radius_um.to_numpy()[:n],before.obs.radius_um.to_numpy()[n:]):raise AssertionError('Paired composition/radial matching lost')
    if not np.allclose(before.obs.local_x.to_numpy()[:n],-before.obs.local_x.to_numpy()[n:]) or not np.allclose(before.obs.local_y.to_numpy()[:n],before.obs.local_y.to_numpy()[n:]):raise AssertionError('Mirror geometry lost')
    if not np.array_equal(before.obsm['spatial'],after.obsm['spatial']):raise AssertionError('MI injection changed spatial geometry')
    totals=np.maximum(raw.sum(1),1);normalized_background_difference=float(np.mean(np.abs(totals[:n]/totals[n:]-1)))
    return {'cells':len(after),'encoder_gene_panel':after.n_vars,'all_LR_genes':len(lr_genes),'edge_MI_genes':len(edge_genes),'edge_MI_gene_fraction':len(edge_genes)/after.n_vars,
        'all_LR_count_fraction':float(raw[:,ids].sum()/max(raw.sum(),1)),
        'edge_MI_count_fraction_in_edge':float(raw[edge][:,edge_ids].sum()/max(raw[edge].sum(),1)),
        'non_LR_background_exactly_mirrored':True,'paired_type_radius_coordinates_match':True,
        'mean_paired_library_normalization_relative_shift':normalized_background_difference,
        'background_gene_classes':after.var.kind.value_counts().to_dict(),'declared_peak_expected_fold':profile['mi_fold_change'],
        'response_exposure_normalizer':profile['lr_role_mean']*profile['mi_fold_change'],
        'scCube_fidelity_audit_scope':'expression_fidelity.json audits the calibrated scCube stage; rich_simulation_audit.json audits the final NB background and moderate injection',
        'interpretation':'Synthetic stress test; no method is guaranteed to outperform GraphST. Raw background matches exactly; library normalization can create small indirect changes.'}


def prepare_rich_background(simulation_config,profile_path):
    config=SimulationConfig.from_json(simulation_config);profile=json.loads(Path(profile_path).read_text());validate_profile(config,profile)
    # Legacy signal_strength is used by the existing synthetic response
    # generator as exposure denominator; the new LR injection uses fold change.
    if not np.isclose(config.signal_strength,profile['lr_role_mean']*profile['mi_fold_change']):raise ValueError('Response exposure scale does not match the moderate-LR profile')
    if profile.get('variant')!='rich_background_moderate_LR':raise ValueError('Explicit independent simulation variant required')
    source=protect(config);manifest={'simulation_config':config.to_dict(),'background_profile':profile,'implementation_sha256':checksum(Path(__file__))}
    marker=source/'rich_prepared.json';manifest_path=source/'rich_generation_manifest.json'
    if manifest_path.exists() and json.loads(manifest_path.read_text())!=manifest:raise ValueError('Rich simulation changed; preserve generated data and select an independent NAS output')
    if not manifest_path.exists() and any((source/p).exists() for p in ('before_injection.h5ad','injected.h5ad','gnn/checkpoint.pt')):raise ValueError('Refusing to replace a preexisting simulation')
    manifest_path.write_text(json.dumps(manifest,indent=2)+'\n')
    if marker.exists():
        saved=json.loads(marker.read_text())
        if saved['manifest']!=manifest:raise ValueError('Prepared data configuration changed')
        for name,digest in saved['files'].items():
            if checksum(source/name)!=digest:raise ValueError('Prepared rich input changed: '+name)
        return source
    lr=make_lr(config);reference=rich_reference(config,profile,lr);raw_path=source/'sccube_background_raw.h5ad'
    if raw_path.exists():raw=ad.read_h5ad(raw_path)
    else:
        raw,_=generate(config,source,reference=reference);raw.write_h5ad(raw_path)
    before=add_background_variability(raw,profile,config.seed);before.uns['simulation_variant']='rich_background_moderate_LR';before.uns['rich_background_profile']=profile
    before.uns['simulation_provenance']['expression_transform']+='; independent shared-state/library variation and NB resampling; duplicate paired background; Poisson moderate LR count increments'
    after=inject_moderate_mi(before,lr,config,profile)
    before.write_h5ad(source/'before_injection.h5ad');after.write_h5ad(source/'injected.h5ad');lr.to_csv(source/'lr.csv',index=False)
    audit=simulation_audit(before,after,lr,profile);(source/'rich_simulation_audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    files={name:checksum(source/name) for name in ('before_injection.h5ad','injected.h5ad','lr.csv','seed_reference.h5ad','rich_simulation_audit.json')}
    marker.write_text(json.dumps({'manifest':manifest,'files':files},indent=2)+'\n')
    print(f'Prepared rich simulation: {len(after)} cells, {after.n_vars} genes; LR count fraction {audit["all_LR_count_fraction"]:.2%}',flush=True)
    return source
