"""Read-only study/coverage audit; no biological or clinical labels are invented."""
from pathlib import Path
import json
import numpy as np
import anndata as ad
import pandas as pd
from ..storage import STORAGE_ROOT
from ..ccc.config import default_config
from ..ccc.data import _cache_is_valid,_manifest_signature,cache_paths
from ..ccc.lr_atlas import load_lr_atlas,_checksum as checksum
from .inputs import resolve_input

REQUIRED = ('st_anchor','bayesprism','clinical_endpoint','feature_mask','sample_matching','cell_labels')


def protect_output(config):
    output=Path(config.output_dir).resolve(); root=STORAGE_ROOT.resolve()
    if not output.is_relative_to(root/'outputs'): raise ValueError('Real outputs must be direct NAS workspace/outputs paths')
    protected=[Path(config.legacy_output).resolve(),root/'data']
    for path in protected:
        if output==path or output.is_relative_to(path) or path.is_relative_to(output):
            raise ValueError('Output overlaps protected legacy results or source inputs')
    if Path(config.output_dir).is_symlink(): raise ValueError('Symbolic storage links are forbidden')
    if (output/'checkpoint.pt').is_file():
        import torch
        payload=torch.load(output/'checkpoint.pt',map_location='cpu',weights_only=False)
        if payload.get('architecture')!=config.architecture or payload.get('config')!=config.to_dict():
            raise ValueError('Existing checkpoint is protected: architecture/configuration differs')


def audit_study(config,write=True):
    protect_output(config)
    inp=resolve_input(config,default_config()); paths=cache_paths(inp); blockers=[]
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
    if not paths['features'].is_file():
        return {'ready_for_training':False,'blockers':blockers+['CCC cache absent; construct/audit it first'], 'n_cells':n_cells,'panel_genes':len(genes),'fixed_CCC_features':0,'fully_measurable_LR_pairs':0}
    cols=(np.load(Path(config.legacy_output)/'cache/retained_indices.npy') if config.use_legacy_feature_mask
          else np.arange(len(pd.read_csv(paths['features'])),dtype=np.int64))
    features=pd.read_csv(paths['features']).iloc[cols].reset_index(drop=True)
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
            'cell_annotation':annotation,'study_audits':study_status,
            'architecture':config.architecture,'phenotype_supervision':False,'regulatory_priors_required':False}
    if write:
        output=Path(config.output_dir);output.mkdir(parents=True,exist_ok=True)
        (output/'audit.json').write_text(json.dumps(report,indent=2)+'\n')
    return report
