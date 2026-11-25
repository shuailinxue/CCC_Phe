"""Outcome-blind comparison; preserve production labels/expression and notebooks.

Run with cccphe Python, PYTHONPATH=src. Writes only to NAS pilot output.
SingleR scores and marker checks are evidence, not inferred-cell ground truth.
"""
from pathlib import Path
import sys,json,hashlib,importlib.metadata
sys.dont_write_bytecode=True
import numpy as np
import pandas as pd
import anndata as ad
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import classification_report,balanced_accuracy_score,f1_score
from phenoniche.cell_annotation import REFERENCE_LABELS,CORE_MARKERS,SingleRAnnotator,marker_ranks
from phenoniche.storage import STORAGE_ROOT

ROOT=Path(__file__).resolve().parents[1]
BASE=STORAGE_ROOT/'outputs/phenomap_tnbc'
OUT=STORAGE_ROOT/'outputs/tnbc_singler_annotation_pilot'
SAMPLES=['TNBC1_CN1_C2','TNBC3_CN2_C1','TNBC8_CN4_E1','TNBC32_CN16_E1','TNBC76_CN38_E1']


def reference_data(config, genes):
    cache=OUT/'reference_subset.npz'
    if cache.exists():
        data=np.load(cache,allow_pickle=False)
        np.testing.assert_array_equal(data['genes'],genes)
        return data['x'],data['labels'],data['donors']
    a=ad.read_h5ad(config['reference_atlas'],backed='r')
    try:
        labels=a.obs.celltype_major.map(REFERENCE_LABELS)
        donors=a.obs.donor_id.astype(str).to_numpy()
        # Freeze three disjoint donor partitions before any query predictions.
        train,other=next(GroupShuffleSplit(test_size=.4,random_state=52173).split(np.zeros(len(a)),groups=donors))
        cal_rel,test_rel=next(GroupShuffleSplit(test_size=.5,random_state=52174).split(np.zeros(len(other)),groups=donors[other]))
        partitions=[train,other[cal_rel],other[test_rel]]
        rng=np.random.default_rng(52173);rows=[]
        for ids,limit in zip(partitions,[500,200,200]):
            for label in REFERENCE_LABELS.values():
                pool=ids[labels.iloc[ids].eq(label).to_numpy()]
                if len(pool)<10:raise ValueError(f'Reference split underpowered for {label}')
                rows.extend(rng.choice(pool,min(limit,len(pool)),replace=False))
        rows=np.sort(rows)
        symbols=a.var.feature_name.astype(str).str.upper().tolist()
        lookup={g:i for i,g in enumerate(symbols)}
        cols=[lookup[g] for g in genes]
        subset=a[rows].to_memory()
        x=subset.X[:,cols].toarray().astype(np.float32)
        if not config['reference_is_log1p']:x=np.log1p(x/np.maximum(x.sum(1,keepdims=True),1)*10000)
        labels=labels.iloc[rows].to_numpy(dtype=str);donors=donors[rows]
        split=np.zeros(len(rows),int)
        split[np.isin(rows,partitions[1])]=1;split[np.isin(rows,partitions[2])]=2
        np.savez_compressed(cache,x=x,labels=labels,donors=donors,split=split,genes=np.asarray(genes))
        return x,labels,donors
    finally:a.file.close()


def main():
    OUT.mkdir(exist_ok=True,parents=True)
    config=json.loads((ROOT/'config/phenomap_tnbc_batch.json').read_text())
    panel=(BASE/'shared/genes.txt').read_text().splitlines()
    a=ad.read_h5ad(config['reference_atlas'],backed='r')
    present=set(a.var.feature_name.astype(str).str.upper());a.file.close()
    genes=[g for g in panel if g in present];cols=pd.Index(panel).get_indexer(genes)
    print('REFERENCE',len(genes),'shared genes',flush=True)
    x,labels,donors=reference_data(config,genes)
    split=np.load(OUT/'reference_subset.npz',allow_pickle=False)['split']
    assert not set(donors[split==0])&set(donors[split!=0])
    assert not set(donors[split==1])&set(donors[split==2])
    print('FIT',np.bincount(split).tolist(),flush=True)
    model=SingleRAnnotator(x[split==0],labels[split==0],genes,threads=4)
    thresholds=model.calibrate(x[split==1],labels[split==1])
    test=model.predict(x[split==2]);test['true_label']=labels[split==2]
    test.to_csv(OUT/'reference_donor_test.csv.gz',index=False)
    validation={'balanced_accuracy_raw':float(balanced_accuracy_score(test.true_label,test.singleR_label)),
        'macro_F1_raw':float(f1_score(test.true_label,test.singleR_label,average='macro')),
        'accepted_fraction':float(test.cell_type.ne('Unknown').mean()),
        'report':classification_report(test.true_label,test.singleR_label,output_dict=True,zero_division=0)}
    print('REFERENCE VALIDATION',validation['balanced_accuracy_raw'],flush=True)
    counts=[];qc=[]
    rng=np.random.default_rng(40700)
    for name in SAMPLES:
        print('QUERY',name,flush=True)
        a=ad.read_h5ad(BASE/name/'cells.h5ad',backed='r')
        try:
            ids=np.sort(rng.choice(len(a),min(4096,len(a)),replace=False))
            values=np.asarray(a.X[ids])[:,cols]
            result=model.predict(values);result.index=a.obs_names[ids];result.index.name='cell_id'
            result['old_cell_type']=a.obs.cell_type.iloc[ids].astype(str).to_numpy()
            result.to_csv(OUT/(name+'_annotations.csv.gz'))
            ranks=marker_ranks(values,genes,list(CORE_MARKERS))
            for stage,column in [('raw','singleR_label'),('supported','cell_type'),('old','old_cell_type')]:
                for label,n in result[column].value_counts().items():
                    counts.append({'sample':name,'stage':stage,'cell_type':label,'cells':int(n),'sampled_cells':len(result)})
            for label in CORE_MARKERS:
                mask=result.singleR_label.eq(label).to_numpy()
                qc.append({'sample':name,'label':label,'raw_calls':int(mask.sum()),
                    'supported_calls':int(result.cell_type.eq(label).sum()),
                    'available_core_markers':';'.join(g for g in CORE_MARKERS[label] if g in genes),
                    'median_core_rank_in_calls':float(ranks.loc[mask,label].median()) if mask.any() else None,
                    'reference_core_rank_floor':thresholds[label]['marker_rank']})
            print(result.cell_type.value_counts().to_dict(),flush=True)
        finally:a.file.close()
    pd.DataFrame(counts).to_csv(OUT/'pilot_label_counts.csv',index=False)
    pd.DataFrame(qc).to_csv(OUT/'pilot_core_marker_QC.csv',index=False)
    audit={'method':'Official SingleR Python implementation; rank correlations, fine tuning; custom reference-calibrated rejection',
        'singler_version':importlib.metadata.version('singler'),
        'reference_atlas':config['reference_atlas'],
        'reference_bytes':Path(config['reference_atlas']).stat().st_size,
        'reference_mtime_ns':Path(config['reference_atlas']).stat().st_mtime_ns,
        'code_sha256':hashlib.sha256((ROOT/'src/phenoniche/cell_annotation.py').read_bytes()).hexdigest(),
        'gene_count':len(genes),'gene_sha256':hashlib.sha256('\n'.join(genes).encode()).hexdigest(),
        'thresholds':thresholds,'reference_validation':validation,'samples':SAMPLES,
        'reference_donors':{str(i):sorted(set(donors[split==i])) for i in range(3)},
        'limitations':['Reference donor accuracy is not inferred-RNA annotation accuracy',
            'Existing expression panel used to isolate annotation method; omitted B markers not imputed',
            'No outcomes or niche labels used; no target class proportions imposed',
            'Original production expression/labels/checkpoints/notebooks unchanged; not yet deployed']}
    (OUT/'audit.json').write_text(json.dumps(audit,indent=2))
    print('DONE',OUT,flush=True)


if __name__=='__main__':main()
