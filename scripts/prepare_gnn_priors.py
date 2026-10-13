"""Convert authenticated CollecTRI/NicheNet snapshots into traceable observed priors."""
from pathlib import Path
import sys,json,hashlib
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np,pandas as pd
from phenoniche.storage import STORAGE_ROOT,OUTPUT_DIR
from phenoniche.gnn.biology import checksum
from types import SimpleNamespace
from phenoniche.gnn.inputs import resolve_input
from phenoniche.ccc.config import default_config

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--config',default=str(Path(__file__).resolve().parents[1]/'config/gnn.json'));parser.add_argument('--upstream-only',action='store_true')
    args=parser.parse_args();values=json.loads(Path(args.config).read_text())
    if 'prior_manifest' not in values: raise ValueError('This archived prior-conversion utility is not used by the current composition GNN; supply an archived regulatory configuration explicitly')
    cfg=SimpleNamespace(**values)
    root=Path(cfg.prior_manifest).parent;root.mkdir(parents=True,exist_ok=True)
    raw=STORAGE_ROOT/'data/priors/raw';manifest={};selection={}
    if Path(cfg.upstream_prior).parent!=root or Path(cfg.downstream_prior).parent!=root: raise ValueError('Prior TSV files and manifest must share their configured directory')
    ct=pd.read_csv(raw/'collectri.tsv',sep='\t')
    usable=ct.is_directed & (ct.is_stimulation ^ ct.is_inhibition) & ct.references.notna() & ct.source_genesymbol.notna() & ct.target_genesymbol.notna()
    up=ct.loc[usable].copy()
    upstream=pd.DataFrame({'tf':up.source_genesymbol,'target':up.target_genesymbol,
                          'weight':np.where(up.is_stimulation,1.,-1.),'source':up.sources,'evidence':up.references})
    # Deduplicate only consistent signs; omit any conflicting source-level symbols.
    conflict=upstream.groupby(['tf','target']).weight.agg(['min','max']);bad=conflict.index[conflict['min']*conflict['max']<0]
    if len(bad): upstream=upstream.set_index(['tf','target']).drop(index=bad).reset_index()
    upstream=upstream.drop_duplicates(['tf','target']).sort_values(['tf','target'])
    upstream.to_csv(root/'upstream.tsv',sep='\t',index=False)
    manifest['upstream']={'database':'CollecTRI','version':'OmniPath human snapshot 2026-10-13','url':'https://omnipathdb.org/interactions?datasets=collectri&genesymbols=1&fields=sources,references&format=tsv','species':'human','sha256':checksum(root/'upstream.tsv'),'raw_sha256':checksum(raw/'collectri.tsv')}
    selection['upstream']={'raw_rows':len(ct),'kept_rows':len(upstream),'ambiguous_signed_rows_excluded':int((ct.is_stimulation & ct.is_inhibition).sum()),'missing_evidence_excluded':int(ct.references.isna().sum())}
    nichenet=raw/'nichenet_alternate.rds'
    if nichenet.is_file() and not args.upstream_only:
        md5=hashlib.md5(nichenet.read_bytes()).hexdigest()
        if md5!='b09606b04b2d4490418d9028c0e58b9f': raise ValueError('NicheNet official MD5 mismatch or incomplete download')
        import pyreadr
        matrix=pyreadr.read_r(str(nichenet))[None]
        inp=resolve_input(cfg,default_config());feature=pd.read_csv(Path(inp.output_dir)/'cache/features.csv')
        if cfg.use_legacy_feature_mask: feature=feature.iloc[np.load(Path(cfg.legacy_output)/'cache/retained_indices.npy')]
        requested=set(feature.ligand.str.upper())
        selected_ligands=[ligand for ligand in matrix.columns if str(ligand).upper() in requested]
        rows=[]
        # Fixed top-100 prior targets per ligand, selected across the full human
        # universe before Xenium panel matching (not outcome/expression-selected).
        for ligand in selected_ligands:
            values=matrix[ligand].sort_values(ascending=False,kind='stable').head(100)
            for gene,weight in values.items():
                if np.isfinite(weight) and weight>0:
                    rows.append({'ligand':str(ligand).upper(),'target':str(gene).upper(),'weight':float(weight),
                                 'source':'NicheNet-v2 human integrated ligand-target prior',
                                 'evidence':'doi:10.5281/zenodo.7074291; doi:10.1038/s41592-019-0667-5'})
        downstream=pd.DataFrame(rows).drop_duplicates(['ligand','target']).sort_values(['ligand','target'])
        downstream.to_csv(root/'downstream.tsv',sep='\t',index=False)
        manifest['downstream']={'database':'NicheNet','version':'NicheNet-v2 human Zenodo 7074291 (2022-09-13)','url':'https://zenodo.org/records/7074291','species':'human','sha256':checksum(root/'downstream.tsv'),'raw_sha256':checksum(nichenet),'official_md5':md5,'target_policy':'top 100 positive potential-score targets per ligand over complete prior; panel/CCC exclusions applied later; not causal or receptor-specific evidence'}
        selection['downstream']={'matrix_shape':list(matrix.shape),'kept_rows':len(downstream),'ligands':downstream.ligand.nunique(),'target_policy':'top 100 per frozen CCC ligand, before target panel matching','requested_CCC_ligands':sorted(requested),'unmatched_CCC_ligands':sorted(requested-set(str(g).upper() for g in selected_ligands))}
    (root/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    (root/'selection_audit.json').write_text(json.dumps(selection,indent=2)+'\n')
    print(json.dumps(selection,indent=2))
