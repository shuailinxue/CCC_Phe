"""Read-only checks of missing B/endothelial calls in inferred TNBC cells."""
from pathlib import Path
import json
import numpy as np,pandas as pd,anndata as ad
BASE=Path('/home/nas3/biod/xueshuailin/CCC_Phe/workspace/outputs/phenomap_tnbc')
OUT=Path('/home/nas3/biod/xueshuailin/CCC_Phe/workspace/outputs/tnbc_annotation_audit')

def main():
    OUT.mkdir(exist_ok=True,parents=True);m=pd.read_csv(BASE/'selection/selected_samples.csv');records=[]
    for row in m.itertuples():
        a=ad.read_h5ad(BASE/row.sample/'cells.h5ad',backed='r')
        try:
            obs=a.obs
            for typ in ['B_cells','Endothelial']:
                prob=obs['prob_'+typ].to_numpy();marker=obs.marker_supported.to_numpy()
                records.append({'sample':row.sample,'type':typ,'cells':len(obs),
                    'argmax_cells':int((obs[['prob_B_cells','prob_Endothelial','prob_Epithelial','prob_Myeloid','prob_Stromal','prob_T_cells']].idxmax(axis=1)=='prob_'+typ).sum()),
                    'probability_ge_05':int((prob>=.5).sum()),'max_probability':float(prob.max()),
                    'accepted':int(obs.cell_lineage.eq(typ).sum())})
        finally:a.file.close()
    table=pd.DataFrame(records);table.to_csv(OUT/'missing_lineage_probabilities.csv',index=False)
    print(table.groupby('type')[['argmax_cells','probability_ge_05','accepted']].sum().to_string())
    print(table.groupby('type').max_probability.agg(['min','median','max']).to_string())

if __name__=='__main__':main()
