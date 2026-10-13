"""Read-only study/coverage audit; no biological or clinical labels are invented."""
from pathlib import Path
import json
import numpy as np
import anndata as ad
import pandas as pd
from ..storage import STORAGE_ROOT
from ..ccc.config import default_config
from ..ccc.data import _cache_is_valid,_manifest_signature,cache_paths
from ..ccc.lr_atlas import load_lr_atlas,canonicalize_complex
from .biology import checksum,read_prior,compile_prior

REQUIRED = ('st_anchor','bayesprism','clinical_endpoint','feature_mask','sample_matching','cell_labels')


def protect_output(config):
    output=Path(config.output_dir).resolve(); root=STORAGE_ROOT.resolve()
    if not output.is_relative_to(root/'outputs'): raise ValueError('Real outputs must be direct NAS workspace/outputs paths')
    protected=[Path(config.legacy_output).resolve(),root/'data']
    for path in protected:
        if output==path or output.is_relative_to(path) or path.is_relative_to(output):
            raise ValueError('Output overlaps protected legacy results or source inputs')
    if Path(config.output_dir).is_symlink(): raise ValueError('Symbolic storage links are forbidden')


def audit_study(config,write=True):
    protect_output(config)
    inp=default_config(); paths=cache_paths(inp); blockers=[]
    source_valid=_cache_is_valid(paths,_manifest_signature(inp))
    if not source_valid: blockers.append('ST/CCC source cache signature mismatch')
    x=ad.read_h5ad(inp.processed_h5ad,backed='r')
    try:
        genes=tuple(str(g).upper() for g in x.var_names); annotation=str(x.uns.get('cell_type_annotation','unspecified'))
        n_cells=x.n_obs
    finally:
        x.file.close()
    panel=set(genes); atlas=load_lr_atlas(inp.lr_atlas)
    ligand={g for row in atlas.interactions for g in row.ligand_components}
    receptor={g for row in atlas.interactions for g in row.receptor_components}
    cols=np.load(Path(config.legacy_output)/'cache/retained_indices.npy')
    features=pd.read_csv(paths['features']).iloc[cols].reset_index(drop=True)
    manifest=json.loads(Path(config.prior_manifest).read_text()) if Path(config.prior_manifest).is_file() else {}
    coverage={}
    for branch in ('upstream','downstream'):
        try:
            table,provenance=read_prior(getattr(config,branch+'_prior'),manifest,branch)
            prior=compile_prior(table,features,genes,branch,config.min_targets,config.min_target_coverage)
            coverage[branch]={'provenance':provenance,**prior.audit}
            if getattr(config,'use_'+branch) and not prior.mapping.shape[1]: blockers.append(branch+': no verified independent panel targets')
        except (ValueError,OSError) as error:
            coverage[branch]={'status':'invalid','error':str(error)}
            # Invalid existing priors are never silently ignored, even in CCC-only.
            blockers.append(branch+': '+str(error))
    audit_file=Path(config.audit_file) if config.audit_file else None
    study=json.loads(audit_file.read_text()) if audit_file and audit_file.is_file() else {}
    study_status={}
    for role in REQUIRED:
        record=study.get(role,{})
        if record.get('status')=='not_applicable' and role in ('bayesprism','clinical_endpoint','sample_matching') and study.get('scope')=='independent_downstream_niche' and record.get('reason'):
            study_status[role]=record
            continue
        evidence=Path(record.get('evidence_file',''))
        valid=record.get('status')=='verified' and evidence.is_file() and record.get('sha256')==checksum(evidence)
        if not valid: blockers.append(role+': missing verified audit evidence')
        study_status[role]=record if valid else {'status':'missing_or_invalid'}
    report={'ready_for_training':not blockers,'blockers':blockers,'source_CCC_signature_valid':source_valid,
            'n_cells':n_cells,'panel_genes':len(genes),'fixed_CCC_features':len(features),
            'ligand_genes':len(ligand),'ligand_genes_on_panel':len(ligand&panel),'uncovered_ligands':sorted(ligand-panel),
            'receptor_genes':len(receptor),'receptor_genes_on_panel':len(receptor&panel),'uncovered_receptors':sorted(receptor-panel),
            'fully_measurable_LR_pairs':sum(set(r.ligand_components+r.receptor_components)<=panel for r in atlas.interactions),
            'cell_annotation':annotation,'study_audits':study_status,'regulation':coverage,
            'TF_target_coverage_note':'TF/target coverage is unknown when no audited regulatory universe is supplied; TF mRNA is not activity.',
            'biology_mode':config.mode,'phenotype_supervision':False,
            'primary_sources':{'SpiderNet':'https://github.com/ma-compbio-lab/SpiderNet','CollecTRI':'https://github.com/saezlab/CollecTRI','NicheNet':'https://github.com/saeyslab/nichenetr'}}
    if write:
        output=Path(config.output_dir);output.mkdir(parents=True,exist_ok=True)
        (output/'audit.json').write_text(json.dumps(report,indent=2)+'\n')
    return report
