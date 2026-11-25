"""Quick descriptive RNA check of the existing 669-pair LGALS3–ENG case."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import anndata as ad
from phenoniche.cohorts.saved_inputs import input_root

ROOT=Path(__file__).resolve().parents[1]


def main():
    output=Path(json.loads((ROOT/'config/kidney_dkd_5k.json').read_text())['output_dir'])
    cache=input_root(output);analysis=output/'condition_niches'
    dest=analysis/'N6_LGALS3_ENG_response_check';dest.mkdir(exist_ok=True)
    pairs=pd.read_csv(analysis/'N6_case_EC_to_Fibroblast_LGALS3_ENG_HC_N3_pairs.csv')
    cells=pd.read_csv(analysis/'assignments.csv.gz');members=np.load(cache/'members.npy',mmap_mode='r')
    definitions={
        'TGF_response':['SERPINE1','CCN2','SMAD7','TGFBI','PMEPA1'],
        'Fibro_activation_ECM':['ACTA2','TAGLN','POSTN','FAP','COL1A1','COL1A2','COL3A1','FN1','LOX']}
    markers=list(dict.fromkeys(['ENG']+[g for genes in definitions.values() for g in genes]))
    a=ad.read_h5ad(cache/'tissue.h5ad',backed='r')
    try:
        order=pd.Index(a.obs.cell_id.astype(str)).get_indexer(cells.cell_id.astype(str))
        assert (order>=0).all() and len(set(order))==len(cells)
        assert np.allclose(a.obsm['spatial'][order],cells[['x','y']].to_numpy(),atol=1e-3)
        mapping={g:g if g in a.var_names else 'CTGF' if g=='CCN2' and 'CTGF' in a.var_names else None for g in markers}
        available=[g for g in markers if mapping[g] is not None]
        columns=[a.var_names.get_loc(mapping[g]) for g in available]
        ids={}
        for group,col in [('HC_matched','HC_row'),('DKD_matched','DKD_row')]:
            pool=np.unique(members[pairs[col].to_numpy()])
            ids[group]=pool[cells.iloc[pool].cell_type.to_numpy()=='Fibroblast']
        # A within-DKD descriptive sensitivity check, excluding all case-pool cells.
        others=np.flatnonzero(cells.condition.eq('DKD').to_numpy()&cells.cell_type.eq('Fibroblast').to_numpy())
        ids['DKD_other_fibro']=np.setdiff1d(others,ids['DKD_matched'])
        expression={};raws={};depth={};source=[]
        for group,rows in ids.items():
            assert len(rows)==len(set(rows))
            raw=a[order[rows],:].X.tocsr()
            totals=np.asarray(raw.sum(axis=1)).ravel()
            selected=raw[:,columns].toarray()
            raws[group]=selected
            expression[group]=np.log1p(selected*10000/np.maximum(totals[:,None],1))
            depth[group]={'unique_cells':len(rows),'median_total_counts':float(np.median(totals)),
                          'mean_total_counts':float(totals.mean())}
            source.extend({'group':group,'row':int(row),'cell_id':cells.cell_id.iloc[row]} for row in rows)
        pd.DataFrame(source).to_csv(dest/'unique_fibroblast_cells.csv',index=False)
        records=[]
        for j,gene in enumerate(available):
            row={'gene':gene,'panel_symbol':mapping[gene]}
            for group in ids:
                row[group+'_logmean']=float(expression[group][:,j].mean())
                row[group+'_mean_TP10K']=float(np.expm1(expression[group][:,j]).mean())
                row[group+'_detection']=float((raws[group][:,j]>0).mean())
            row['DKD_minus_HC_logmean']=row['DKD_matched_logmean']-row['HC_matched_logmean']
            row['log2FC_TP10K']=float(np.log2((row['DKD_matched_mean_TP10K']+.1)/(row['HC_matched_mean_TP10K']+.1)))
            row['DKD_matched_minus_other_logmean']=row['DKD_matched_logmean']-row['DKD_other_fibro_logmean']
            records.append(row)
        table=pd.DataFrame(records);table.to_csv(dest/'marker_expression.csv',index=False)
        scores=[]
        # Fixed requested lists, ENG excluded. Standardize using equal-weight HC
        # and DKD group moments, so unequal cell counts do not set the scale.
        mean=(expression['HC_matched'].mean(0)+expression['DKD_matched'].mean(0))/2
        second=((expression['HC_matched']**2).mean(0)+(expression['DKD_matched']**2).mean(0))/2
        scale=np.maximum(np.sqrt(np.maximum(second-mean**2,0)),.1)
        for name,genes in definitions.items():
            present=[g for g in genes if g in available]
            ix=[available.index(g) for g in present]
            row={'signature':name,'requested_genes':len(genes),'available_genes':len(present),'genes':';'.join(present)}
            for group,values in expression.items():
                score=((values[:,ix]-mean[ix])/scale[ix]).mean(axis=1)
                row[group+'_mean_score']=float(score.mean())
            row['DKD_minus_HC_score']=row['DKD_matched_mean_score']-row['HC_matched_mean_score']
            row['DKD_matched_minus_other_score']=row['DKD_matched_mean_score']-row['DKD_other_fibro_mean_score']
            scores.append(row)
        scores=pd.DataFrame(scores);scores.to_csv(dest/'signature_scores.csv',index=False)
        protocol={'matched_neighborhood_pairs':len(pairs),'unique_cells':depth,'panel_mapping':mapping,
            'gene_sets':definitions,'normalization':'Each unique cell: log1p(10000 * measured count / all panel counts)',
            'scores':'Mean gene z scores; equal HC/DKD moments, SD floor 0.1; ENG excluded',
            'scope':'Descriptive pooled matched-neighborhood fibroblasts, including halo; no cell-level p-values',
            'within_DKD_sensitivity':'All other observed DKD fibroblasts, excluding matched-neighborhood fibroblasts; no composition matching',
            'caution':'RNA association only; no evidence of direct LGALS3–ENG signaling, SMAD phosphorylation, or patient-level disease effect'}
        (dest/'protocol.json').write_text(json.dumps(protocol,indent=2))
        print(json.dumps(depth,indent=2));print('Panel',mapping)
        print(table[['gene','HC_matched_logmean','DKD_matched_logmean','HC_matched_detection','DKD_matched_detection','DKD_minus_HC_logmean','DKD_matched_minus_other_logmean']].round(4).to_string(index=False))
        print(scores.round(4).to_string(index=False));print('OUTPUT',dest)
    finally:a.file.close()


if __name__=='__main__':main()
