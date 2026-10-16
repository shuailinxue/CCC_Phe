"""Published skin architecture for sample choice and post-fit interpretation only.

Author-defined immune aggregates and fine annotations never enter the encoder,
CCC construction, fitting, or unsupervised K selection. Aggregate source counts
are reported as the author's ``total_cells``, not as independent ground truth.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd

PAPER='https://www.nature.com/articles/s41467-026-74871-7'
IMMUNE_TYPES=('B_Plasma','Lymphoid','Myeloid','Mast')


def paper_section_audit(data_dir, output_dir):
    """Read the actual Fig. 3 workbook, without weighting by learned results."""
    root=Path(data_dir);out=Path(output_dir);out.mkdir(parents=True,exist_ok=True)
    source=root/'paper/source_data/Figure3.xlsx'
    if not source.exists():raise FileNotFoundError('Published Fig. 3 source workbook required: '+str(source))
    niches=pd.read_excel(source,sheet_name='F3AD')
    meta=pd.read_csv(root/'export/metadata.csv',low_memory=False)
    records=[]
    for (patient,image),section in meta.groupby(['patient_id_paper','sample_paper'],observed=True):
        if not (str(patient).startswith('patient_H') or str(patient).startswith('patient_DM')):continue
        authored=niches[(niches.patient_id_paper==patient)&(niches.sample_paper==image)]
        disease=authored[authored.subcluster_new=='Dermis_Disease']
        records.append({'patient':patient,'image':image,'cells':len(section),
            'fibroblast_fraction':float(section.sp_annotation.eq('Fibroblasts').mean()),
            'epidermis_fraction':float(section.sp_annotation.eq('Epidermis_keranocytes').mean()),
            'immune_fraction':float(section.sp_annotation.isin(IMMUNE_TYPES).mean()),
            'author_Dermis_Disease_aggregates':len(disease),
            'author_Dermis_Disease_total_cells':int(disease.total_cells.sum()),
            'author_largest_Dermis_Disease_cells':int(disease.total_cells.max()) if len(disease) else 0,
            'excluded_by_original_paper':patient=='patient_DM24'})
    table=pd.DataFrame(records)
    table.to_csv(out/'paper_guided_section_audit.csv',index=False)
    (out/'paper_analysis_context.json').write_text(json.dumps({
        'source':PAPER,'source_workbook':str(source),'sheet':'F3AD',
        'original_global_compartments':'hoodscanR: KNN100 neighborhood associations, 6 compartments',
        'original_immune_niches':'HDBSCAN on immune and endothelial cells; exclude <20 immune cells or >90% vascular; Bray-Curtis immune composition, PCoA, hierarchical niche classification',
        'original_excluded_section':'patient_DM24: wound-like histology, absent interfollicular keratinocytes',
        'original_visualization':'Fig3B/G: immune aggregates highlighted over gray unassigned tissue; fine immune states in zooms',
        'training_use_of_author_niches':False,'fine_annotations_for_display_only':True,
        'source_table_note':'Retain actual source values, including small healthy Dermis_Disease assignments; do not force exclusivity from the paper narrative.'},indent=2))
    return table


def niche_immune_summary(cells,composition,types):
    """Descriptive ranking of learned niches in immune-rich microenvironments."""
    comp=np.asarray(composition)
    if comp.shape!=(len(cells),len(types)):raise ValueError('Composition/cell/type dimensions differ')
    immune=comp[:,[i for i,t in enumerate(types) if t in IMMUNE_TYPES]].sum(1)
    rows=[]
    for niche,ids in cells.groupby('niche',sort=True).indices.items():
        ids=np.asarray(ids);local=cells.iloc[ids];hc=(local.condition=='HC').sum();dm=(local.condition=='DM').sum()
        hf=hc/max(int((cells.condition=='HC').sum()),1);df=dm/max(int((cells.condition=='DM').sum()),1)
        rows.append({'niche':int(niche),'HC_cells':int(hc),'DM_cells':int(dm),
            'HC_fraction':float(hf),'DM_fraction':float(df),'DM_minus_HC':float(df-hf),
            'pair_weighted_immune_fraction':float(immune[ids].mean()),
            'DM_immune_fraction':float(immune[ids][(local.condition=='DM').to_numpy()].mean()) if dm else 0.,
            'immune_case_priority':float(max(df-hf,0)*immune[ids].mean())})
    return pd.DataFrame(rows).sort_values('immune_case_priority',ascending=False).reset_index(drop=True)
