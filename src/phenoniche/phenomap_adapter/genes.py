"""Outcome-blind shared LR-first gene panel; marker genes survive SVG selection."""
from pathlib import Path
import csv,hashlib,json,re
import numpy as np,pandas as pd

MARKERS={
 'T_cells':['CD3D','CD3E','CD3G','TRAC','TRBC1','TRBC2','CD247','CD2','CD7','CD8A','CD8B','IL7R','CCL5','NKG7','KLRD1','GNLY'],
 'B_cells':['MS4A1','CD79A','CD79B','CD37','CD19','CD22','BANK1','CD74','HLA-DRA','MZB1','JCHAIN','SDC1'],
 'Myeloid':['LYZ','TYROBP','FCER1G','AIF1','C1QA','C1QB','C1QC','CD68','CD14','CSF1R','LST1','CTSS'],
 'Endothelial':['PECAM1','VWF','KDR','EMCN','RAMP2','PLVAP','ENG','CDH5','CLEC14A'],
 'Stromal':['COL1A1','COL1A2','COL3A1','DCN','LUM','COL6A1','PDGFRA','PDGFRB','ACTA2','TAGLN','RGS5','CXCL12'],
 'Epithelial':['EPCAM','KRT8','KRT18','KRT19','KRT7','KRT5','KRT14','KRT15','KRT17','CLDN3','CLDN4','ELF3','S100A14'],
}


def header(path):
    with Path(path).open(newline='') as h:return next(csv.reader(h))[1:]


def gene_parts(value):
    return [s.strip().upper() for s in re.split(r'[&+_| ]+',str(value)) if s.strip()]


def select_panel(headers,atlas,target=2000,maximum=2500):
    sets=[set(map(str.upper,h)) for h in headers]
    common=set.intersection(*sets)
    union=set.union(*sets)
    complete=[all(g in common for v in [r.ligand,r.receptor] for g in gene_parts(v)) for r in atlas.itertuples()]
    lr={g for r in atlas.loc[complete].itertuples() for v in [r.ligand,r.receptor] for g in gene_parts(v)}
    # A marker absent from one source column list must not disappear from every
    # other sample. Missing source columns remain explicitly masked downstream.
    marker={g for gs in MARKERS.values() for g in gs}&union
    # LR/marker coverage has priority over an arbitrary requested dimension.
    chosen=sorted(lr|marker)
    if len(chosen)>maximum:raise ValueError(f'{len(chosen)} required LR/marker genes exceed maximum={maximum}; raise maximum explicitly, never cut complexes silently')
    if not chosen:raise ValueError('No complete LR or marker gene coverage')
    audit={'genes':len(chosen),'requested_target':target,'hard_maximum':maximum,'LR_genes':len(lr),'marker_genes':len(marker),
        'complete_LR_pairs':int(sum(complete)),'database_pairs':len(atlas),
        'policy':'Common complete LR genes plus union of observed core lineage markers; per-sample absent markers explicitly masked; no phenotype/clinical data',
        'marker_source_coverage':{g:sum(g in s for s in sets) for g in sorted(marker)},
        'missing_markers':{k:sorted(set(gs)-union) for k,gs in MARKERS.items()},
        'markers_not_shared_by_all':sorted(marker-common)}
    return chosen,audit


def prepare_panel(c,out,sources):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    atlas=pd.read_csv(c['lr_atlas'],sep='\t')
    headers=[header(p/'cnts.csv') for p in sources]
    genes,audit=select_panel(headers,atlas,c['target_genes'],c['max_genes'])
    digest=hashlib.sha256('\n'.join(genes).encode()).hexdigest();audit['sha256']=digest
    old=out/'genes.txt'
    if old.exists() and old.read_text().splitlines()!=genes:raise ValueError('Existing shared panel differs; select a separate output_root to preserve existing results')
    old.write_text('\n'.join(genes)+'\n');(out/'gene_panel_audit.json').write_text(json.dumps(audit,indent=2))
    return genes,digest
