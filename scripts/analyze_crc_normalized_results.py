"""Describe saved normalized CRC niches without fitting or notebook changes."""
from pathlib import Path
import json
import h5py
import numpy as np
import pandas as pd
from anndata._io.specs import read_elem

BASE=Path('/home/nas3/biod/xueshuailin/CCC_Phe/workspace/outputs/crc_neutrophil_xenium/joint_five')


def run():
    receipt=json.loads((BASE/'figures/umap_leiden_control.json').read_text())
    clusters=Path(receipt['KMeans']);fit=clusters.parents[1]
    complete=json.loads((fit/'complete.json').read_text());matrices=Path(complete['matrices'])
    out=fit/'result_audit';out.mkdir(exist_ok=True)
    with h5py.File(complete['protocol']['matrix_protocol']['source']['path'],'r') as f:
        obs=pd.DataFrame({k:read_elem(f['obs'][k]) for k in ['sample','cell_type','patient_id']})
    sample=obs['sample'].astype(str).to_numpy();types=obs.cell_type.astype(str).to_numpy()
    cats=json.loads((matrices/'complete.json').read_text())['categories']
    comp=np.load(matrices/'composition.npy')
    methods={'KMeans':np.load(clusters/'labels.npy')+1,'Leiden':np.load(Path(receipt['Leiden'])/'labels.npy')+1}
    for method,labels in methods.items():
        rows=[];prefix='N' if method=='KMeans' else 'L'
        for k in np.unique(labels):
            ids=np.flatnonzero(labels==k);mix=comp[ids].mean(0)
            fractions=pd.Series(sample[ids]).value_counts(normalize=True)
            row=dict(niche=f'{prefix}{k}',cells=len(ids),central_cancer_fraction=float(np.mean(types[ids]=='Cancer cell')),
                     dominant_specimen=str(fractions.index[0]),dominant_specimen_fraction=float(fractions.iloc[0]))
            row.update({'local_'+c:float(mix[j]) for j,c in enumerate(cats)})
            row.update({s:int(np.sum(sample[ids]==s)) for s in pd.unique(sample)})
            rows.append(row)
        summary=pd.DataFrame(rows);summary.to_csv(out/f'{method}_niche_overview.csv',index=False)
        pd.crosstab([sample,types],labels).to_csv(out/f'{method}_sample_center_type_counts.csv')
        print(method,summary[['niche','cells','central_cancer_fraction','local_Cancer cell','local_T cell',
                             'local_Macrophage','local_Fibroblast','local_Neutrophil',
                             'dominant_specimen','dominant_specimen_fraction']].round(4).to_string(index=False),flush=True)
    pd.crosstab(methods['KMeans'],methods['Leiden'],normalize='index').to_csv(out/'KMeans_to_Leiden_row_fractions.csv')
    print('KMeans N1 to Leiden',pd.Series(methods['Leiden'][methods['KMeans']==1]).value_counts(normalize=True).round(4).to_dict(),flush=True)
    old=BASE/'leiden_control/partitions/421a725a52277cc8/labels.npy'
    if old.exists():
        original=np.load(old)+1
        cross=pd.crosstab(original,methods['Leiden'],normalize='index');cross.to_csv(out/'old_to_normalized_Leiden_fractions.csv')
        print('OLD pure-cancer L4/L8/L10 to new Leiden',cross.loc[[4,8,10]].round(4).to_string(),flush=True)
    stats=np.load(matrices/'statistics.npz')
    native_total=stats['totals'];depth=[]
    for k in np.unique(methods['Leiden']):
        mask=(methods['Leiden']==k)&(types=='Cancer cell')
        if mask.any():depth.append(dict(niche=f'L{k}',cancer_cells=int(mask.sum()),median_native_panel_count=float(np.median(native_total[mask]))))
    pd.DataFrame(depth).to_csv(out/'Leiden_cancer_native_count_summary.csv',index=False)
    print('Native counts after normalized fitting',depth,flush=True)
    protocol=dict(fit=str(fit),matrices=str(matrices),Leiden=receipt['Leiden'],cells=len(obs),specimens=int(obs['sample'].nunique()),
                  patients=int(obs['patient_id'].nunique()),model_frozen=True,notebook_modified=False,
                  inference='Descriptive training-cohort audit; no biological replicate significance tests')
    (out/'analysis_protocol.json').write_text(json.dumps(protocol,indent=2))
    print('OUT',out,flush=True)


if __name__=='__main__':run()
