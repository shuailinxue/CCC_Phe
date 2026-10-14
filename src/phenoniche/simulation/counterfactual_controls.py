"""Independent expression-rate controls for diagnosing synthetic niches.

Preserve the source files, non-LR counts and geometry; never modify a CCC matrix.
The program_sparse control deliberately changes biological background assumptions
and is not a denoising estimate of the original observed dataset.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import anndata as ad
from .config import SimulationConfig, ROOT
from .programs import program_specs
from .response_simulation import checksum


def noise_control(source,output,mode,role_mean=None,fold_change=None,config=None):
    source,output=Path(source),Path(output)
    if source.resolve()==output.resolve():raise ValueError('Controls must not overwrite their source dataset')
    if mode not in ('single_nb','lr_expectation','structured_nb','structured_low_basal','program_latent','program_sparse','program_coherent'):
        raise ValueError('Unknown noise control')
    config=config or SimulationConfig.from_json(ROOT/"config/tumor_edge_gene_rich_simulation.json")
    if mode.startswith("program_") and (role_mean is not None or fold_change is not None):
        raise ValueError("Additive program controls have explicit rate settings, not a fold-change override")
    before=ad.read_h5ad(source/'before_injection.h5ad');reference=ad.read_h5ad(source/'seed_reference.h5ad');lr=pd.read_csv(source/'lr.csv')
    original=ad.read_h5ad(source/'injected.h5ad');n=len(before)//2;types=before.obs.cell_type.to_numpy();profile=dict(before.uns['rich_background_profile'])
    genes=sorted(set(lr.ligand)|set(lr.receptor));indices=before.var_names.get_indexer(genes);means=np.stack([np.asarray(reference.X[reference.obs.Cell_type==t])[:,reference.var_names.get_indexer(genes)].mean(0) for t in ['Tumor','Normal epithelial','Fibroblast','Macrophage','T cell']])
    codes=pd.Categorical(types[:n],categories=['Tumor','Normal epithelial','Fibroblast','Macrophage','T cell']).codes
    rates=means[codes]*before.obs.synthetic_library_factor.to_numpy()[:n,None]
    if role_mean is not None:
        for spec in program_specs(config):
            links=lr[lr.mi==spec['MI']]
            for side,role in [('ligand',spec['sender']),('receptor',spec['receiver'])]:
                cells=codes==['Tumor','Normal epithelial','Fibroblast','Macrophage','T cell'].index(role)
                for gene in links[side]:rates[cells,genes.index(gene)]*=role_mean/profile['lr_role_mean']
        profile['lr_role_mean']=role_mean
    if fold_change is not None:profile['mi_fold_change']=fold_change
    rng=np.random.default_rng(40700+5291)
    if mode.startswith('program_'):
        # Program-specific genes have a weak constitutive baseline. Injected
        # program activity is an independent latent expression-rate component,
        # not a multiplier of a previously observed, noisy expression count.
        for spec in program_specs(config):
            links=lr[lr.mi==spec['MI']]
            for side,role in [('ligand',spec['sender']),('receptor',spec['receiver'])]:
                cells=codes==['Tumor','Normal epithelial','Fibroblast','Macrophage','T cell'].index(role)
                for gene in links[side]:
                    g=genes.index(gene);rates[:,g]=.02;rates[cells,g]=.15
                    rates[:,g]*=before.obs.synthetic_library_factor.to_numpy()[:n]
        if mode in ('program_sparse','program_coherent'):
            for row in lr[lr.mi==0].itertuples():
                for gene in (row.ligand,row.receptor):
                    g=genes.index(gene)
                    if pd.isna(row.basal_type):rates[:,g]=.03
                    else:
                        role=['Tumor','Normal epithelial','Fibroblast','Macrophage','T cell'].index(row.basal_type)
                        rates[codes!=role,g]=.01
            if mode=='program_coherent':
                for k,row in enumerate(lr[(lr.mi==0)&lr.basal_type.isna()].itertuples()):
                    for gene in (row.ligand,row.receptor):
                        g=genes.index(gene);rates[:,g]=.01;rates[codes==k%5,g]=.3
    if mode.startswith('structured'):
        unused=lr[(lr.mi==0)&(lr.basal_type.fillna('')=='')]
        for k,row in enumerate(unused.itertuples()):
            role=k%5
            for gene in (row.ligand,row.receptor):
                g=genes.index(gene);rates[:,g]=.02;rates[codes==role,g]=.3
                rates[:,g]*=before.obs.synthetic_library_factor.to_numpy()[:n]
        if mode=='structured_low_basal':
            for row in lr[lr.basal_type.fillna('')!=''].itertuples():
                role=['Tumor','Normal epithelial','Fibroblast','Macrophage','T cell'].index(row.basal_type)
                for gene in (row.ligand,row.receptor):
                    g=genes.index(gene);rates[codes==role,g]=before.obs.synthetic_library_factor.to_numpy()[:n][codes==role]
        # Background LR sets share expression-state variation within a cell,
        # rather than creating 36 ubiquitous independent communication sources.
        state=rng.lognormal(-.5*.15**2,.15,(n,1));rates*=state
        latent=rng.gamma(profile['dispersion'],rates/profile['dispersion']).astype(np.float32);baseline=rng.poisson(latent).astype(np.float32)
    elif mode=='single_nb' or mode.startswith('program_'):
        latent=rng.gamma(profile['dispersion'],rates/profile['dispersion']).astype(np.float32);baseline=rng.poisson(latent).astype(np.float32)
    elif mode=='lr_expectation':latent=rates.astype(np.float32);baseline=latent.copy()
    else:raise ValueError('Unknown noise control')
    before.X[:,indices]=np.tile(baseline,(2,1));after=before.copy();rates_full=np.tile(latent,(2,1));activity=original.obsm['designed_MI_activity'];lookup={g:i for i,g in enumerate(genes)}
    program_state=np.tile(rng.gamma(12.,1/12.,(n,4)),(2,1)) if mode.startswith('program_') else None
    for spec in program_specs(config):
        links=lr[lr.mi==spec['MI']]
        for side,role in [('ligand',spec['sender']),('receptor',spec['receiver'])]:
            r=np.flatnonzero(types==role)
            for gene,loading in zip(links[side],links.loading):
                g=lookup[gene];delta=rates_full[r,g]*(profile['mi_fold_change']-1)*activity[r,spec['MI']-1]*loading/max(links.loading)
                if mode.startswith('program_'):delta=3.*program_state[r,spec['MI']-1]*before.obs.synthetic_library_factor.to_numpy()[r]*activity[r,spec['MI']-1]*loading/max(links.loading)
                after.X[r,indices[g]]+=(delta if mode=='lr_expectation' else rng.poisson(delta)).astype(np.float32)
    for key in ('designed_MI_activity','injected_sender_activity','injected_receiver_activity'):after.obsm[key]=original.obsm[key].copy()
    after.uns['injection_mode']='expression';after.uns['noise_control']=mode
    # Assert exact retention of non-LR measured background and geometry.
    non_lr=np.setdiff1d(np.arange(before.n_vars),indices)
    np.testing.assert_array_equal(after.X[:,non_lr],original.X[:,non_lr]);np.testing.assert_array_equal(after.obsm['spatial'],original.obsm['spatial'])
    output.mkdir(exist_ok=True,parents=True);before.write_h5ad(output/'before_injection.h5ad');after.write_h5ad(output/'injected.h5ad');lr.to_csv(output/'lr.csv',index=False)
    provenance={'source':str(source),'mode':mode,'source_counts_sha256':checksum(source/'injected.h5ad'),'non_LR_counts_unchanged':True,'MI_peak_fold':profile['mi_fold_change'],'MI_role_mean':profile['lr_role_mean'],'background_role_structure':mode.startswith('structured'),'LR_rate_assumption':'Synthetic seed type means × existing cell library factors; no region labels in baseline rates','sampling':'One Gamma-Poisson measurement draw; independent Poisson increments conditional on shared latent rates. Expectation control is fractional diagnostic, not observed data.'}
    if mode.startswith('program_'):
        provenance.update(MI_peak_fold=None,MI_role_mean=.15,MI_off_role_mean=.02,MI_additive_peak_rate=3.,MI_cell_program_state='Shared Gamma(shape=12, scale=1/12) across program genes; mirror-paired before region injection',background_role_structure=mode in ('program_sparse','program_coherent'),inactive_unassigned_LR_rate=.03 if mode=='program_sparse' else 'Original seed means',basal_LR_off_role_rate=.01 if mode in ('program_sparse','program_coherent') else 'Original seed means')
        if mode=='program_coherent':provenance.update(unassigned_LR_role_assignment='Round-robin basal types, independent of region/MI',unassigned_LR_own_role_rate=.3,unassigned_LR_off_role_rate=.01,inactive_unassigned_LR_rate=None)
        after.obsm['synthetic_MI_cell_state']=program_state.astype(np.float32)
        # Persist the explicit generative state; it remains evaluation-only.
        after.write_h5ad(output/'injected.h5ad')
    (output/'provenance.json').write_text(json.dumps(provenance,indent=2))
    return after,lr


def coverage_supervision_mask(adata, data, threshold=.1):
    """Training-only sender/receiver expression coverage, with no truth whitelist."""
    if not 0<threshold<=1:raise ValueError('Expression coverage must lie in (0,1]')
    training_cells=np.unique(data.members[data.rows[0]])
    profiles=[]
    for k in range(len(data.categories)):
        cells=training_cells[data.types[training_cells]==k]
        if not len(cells):profiles.append(np.zeros(adata.n_vars));continue
        values=adata.X[cells]
        if hasattr(values,'toarray'):values=values.toarray()
        profiles.append((np.asarray(values)>0).mean(0))
    coverage=np.stack(profiles)
    gene={g:i for i,g in enumerate(adata.var_names)};kind={t:i for i,t in enumerate(data.categories)}
    valid=np.asarray([coverage[kind[f.sender],gene[f.ligand]]>=threshold and coverage[kind[f.receiver],gene[f.receptor]]>=threshold for f in data.features.itertuples()])
    return data.feature_mask & valid
