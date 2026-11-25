"""Read-only comparison of original spots, inferred cells, and reference scores."""
from pathlib import Path
import json,pickle
import numpy as np,pandas as pd,anndata as ad
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from phenoniche.phenomap_adapter.annotation import annotate,log_normalize
from phenoniche.phenomap_adapter.genes import MARKERS,header
BASE=Path('/home/nas3/biod/xueshuailin/CCC_Phe/workspace/outputs/phenomap_tnbc')
SOURCE=Path('/home/nas3/biod/xueshuailin/New_Projects/data/HBC/TNBC_ST/Processed_Samples')
OUT=Path('/home/nas3/biod/xueshuailin/CCC_Phe/workspace/outputs/tnbc_annotation_audit')

def main():
    config=json.load(open(ROOT/'config/phenomap_tnbc_batch.json'))
    with (BASE/'shared/annotation_reference.pkl').open('rb') as f:bundle=pickle.load(f)
    panel=(BASE/'shared/genes.txt').read_text().splitlines();common=bundle['genes'];rows=[];stats=[]
    for name in ['TNBC1_CN1_C2','TNBC3_CN2_C1','TNBC8_CN4_E1','TNBC32_CN16_E1','TNBC76_CN38_E1']:
        print('READ',name,flush=True)
        path=SOURCE/name/'cnts.csv';lookup={g.upper():g for g in header(path)}
        spot=pd.read_csv(path,index_col=0,usecols=['spots']+[lookup[g] for g in panel]);spot.columns=spot.columns.str.upper();spot=spot[panel]
        a=ad.read_h5ad(BASE/name/'cells.h5ad',backed='r')
        try:
            ids=np.sort(np.random.default_rng(40700).choice(len(a),min(4096,len(a)),replace=False));cells=np.asarray(a.X[ids])
            for domain,x in [('original_spots',spot.to_numpy(np.float32)),('inferred_cells',cells)]:
                x=x.astype(np.float32);norm=log_normalize(x[:,pd.Index(panel).get_indexer(common)])
                z=np.clip((norm-bundle['mean'])/bundle['sd'],-10,10);prob=bundle['classifier'].predict_proba(z)
                labels=bundle['classifier'].classes_[prob.argmax(1)]
                for j,typ in enumerate(bundle['classifier'].classes_):
                    markers=[common.index(g) for g in MARKERS[typ] if g in common]
                    rows.append({'sample':name,'domain':domain,'class':typ,'rows':len(x),'argmax':int((labels==typ).sum()),
                        'max_probability':float(prob[:,j].max()),'mean_probability':float(prob[:,j].mean()),
                        'mean_marker_logTP10K':float(norm[:,markers].mean()),'mean_marker_positive_fraction':float((x[:,pd.Index(panel).get_indexer([common[k] for k in markers])]>0).mean())})
                stats.append({'sample':name,'domain':domain,'positive_gene_fraction':float((x>0).mean()),
                    'median_total':float(np.median(x.sum(1))), 'median_per_cell_gene_cv':float(np.median(x.std(0)/np.maximum(x.mean(0),1e-8)))})
        finally:a.file.close()
    OUT.mkdir(exist_ok=True,parents=True);table=pd.DataFrame(rows);table.to_csv(OUT/'spot_vs_inferred_classifier.csv',index=False);pd.DataFrame(stats).to_csv(OUT/'spot_vs_inferred_expression_stats.csv',index=False)
    print(table[table['class'].isin(['B_cells','Endothelial'])].round(5).to_string(index=False));print(pd.DataFrame(stats).round(4).to_string(index=False))

if __name__=='__main__':main()
