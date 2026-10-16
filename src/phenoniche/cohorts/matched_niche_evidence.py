"""Observed CCC evidence behind learned exceptional niches, after fitting only."""
from pathlib import Path
import json
import numpy as np,pandas as pd
from scipy.spatial import cKDTree


def export_matched_evidence(source,cells,control='NL',case='AD',seed=52173):
    from .paired_skin import input_root
    source=Path(source);out=source/'condition_niches';audit=json.loads((out/'audit.json').read_text())
    cache=input_root(source);comp=np.load(cache/'composition.npy');target=np.load(cache/'ccc.npy',mmap_mode='r');features=pd.read_csv(cache/'features.csv')
    z=np.load(source/'gnn/Z.npy',mmap_mode='r');h=np.load(source/'gnn/H.npy');types=json.loads((source/'summary.json').read_text())['cell_types']
    fit=np.asarray(audit['control_fit_rows'],int);tree=cKDTree(np.sqrt(comp[fit]));rng=np.random.default_rng(seed)
    valid=cells.analysis_center.to_numpy(bool) if 'analysis_center' in cells else np.ones(len(cells),bool)
    labels=sorted(cells.loc[valid,'label'].unique(),key=lambda s:(s[0]!='B',int(s[1:])))
    prog=[];composition=[];top=[]
    for label in labels:
        rows=np.flatnonzero((cells.label.to_numpy()==label)&valid);mean=np.asarray(z[rows]).mean(0);prog.append(mean);composition.append(comp[rows].mean(0));signal=mean@h
        for rank,f in enumerate(np.argsort(-signal)[:10],1):
            record=features.iloc[f];top.append({'niche':label,'rank':rank,'sender':record.sender,'receiver':record.receiver,'LR':record.ligand+'–'+record.receptor,'reconstructed_CCC':float(signal[f])})
    pd.DataFrame(prog,index=labels,columns=np.arange(1,33)).to_csv(out/'niche_program_activity.csv',index_label='niche')
    pd.DataFrame(composition,index=labels,columns=types).to_csv(out/'niche_composition.csv',index_label='niche')
    pd.DataFrame(top).to_csv(out/'niche_top10_CCC.csv',index=False)
    match_rows=[];profiles=[];comp_rows=[];feature_rows=[];summary=[]
    table=pd.read_csv(out/'candidate_niches.csv')
    for label in table.niche:
        cases=np.flatnonzero((cells.label.to_numpy()==label)&(cells.condition.to_numpy()==case))
        if not len(cases):continue
        query=np.sort(rng.choice(cases,min(512,len(cases)),replace=False));_,near=tree.query(np.sqrt(comp[query]),k=min(32,len(fit)))
        near=np.asarray(near).reshape(len(query),-1);choices=fit[near]
        distance=np.linalg.norm(np.sqrt(comp[query,None])-np.sqrt(comp[choices]),axis=-1)
        objective=distance.copy()
        if 'local_log_library' in cells:
            lib=cells.local_log_library.to_numpy();scale=max(float(lib[fit].std()),.1)
            objective+=.15*np.abs(lib[query,None]-lib[choices])/scale
        match=choices[np.arange(len(query)),objective.argmin(1)]
        hd=np.linalg.norm(np.sqrt(comp[query])-np.sqrt(comp[match]),axis=1)/np.sqrt(2)
        for q,r,d in zip(query,match,hd):match_rows.append({'niche':label,'case_row':int(q),'control_row':int(r),'case_cell_id':cells.cell_id.iloc[q],'control_cell_id':cells.cell_id.iloc[r],'composition_Hellinger':float(d)})
        qraw=np.asarray(target[query]);rraw=np.asarray(target[match]);qr=qraw/np.maximum(qraw.sum(1,keepdims=True),1e-12);rr=rraw/np.maximum(rraw.sum(1,keepdims=True),1e-12)
        qmean=qr.mean(0);rmean=rr.mean(0);difference=qmean-rmean
        selected=np.argsort(-difference)[:10];profiles.extend([qmean,rmean]);comp_rows.extend([comp[query].mean(0),comp[match].mean(0)])
        for rank,f in enumerate(selected,1):
            record=features.iloc[f];feature_rows.append({'niche':label,'rank':rank,'feature':int(f),'sender':record.sender,'receiver':record.receiver,
                'LR':record.ligand+'–'+record.receptor,'case_relative_CCC':float(qmean[f]),'matched_control_relative_CCC':float(rmean[f]),
                'relative_log2_ratio':float(np.log2((qmean[f]+1e-6)/(rmean[f]+1e-6))),
                'case_detection_fraction':float((qraw[:,f]>0).mean()),'control_detection_fraction':float((rraw[:,f]>0).mean()),
                'case_native_CCC':float(qraw[:,f].mean()),'control_native_CCC':float(rraw[:,f].mean())})
        leading=selected[:5];support=int(((qmean[leading]>2*(rmean[leading]+1e-6))&((qraw[:,leading]>0).mean(0)>=.2)).sum())
        summary.append({'niche':label,'matched_case_centers':len(query),'distinct_control_centers':len(set(match)),
            'mean_matched_composition_Hellinger':float(hd.mean()),'fraction_pairs_Hellinger_le_015':float((hd<=.15).mean()),
            'observed_relative_CCC_L1_difference':float(np.abs(qmean-rmean).sum()),'supported_top5_LR':support,
            'same_composition_CCC_supported':bool((hd<=.15).mean()>=.8 and support>=3)})
    pd.DataFrame(match_rows,columns=['niche','case_row','control_row','case_cell_id','control_cell_id','composition_Hellinger']).to_csv(out/'matched_centers.csv.gz',index=False)
    pd.DataFrame(feature_rows,columns=['niche','rank','feature','sender','receiver','LR','case_relative_CCC','matched_control_relative_CCC','relative_log2_ratio','case_detection_fraction','control_detection_fraction','case_native_CCC','control_native_CCC']).to_csv(out/'matched_observed_top_CCC.csv',index=False)
    evidence=pd.DataFrame(summary,columns=['niche','matched_case_centers','distinct_control_centers','mean_matched_composition_Hellinger','fraction_pairs_Hellinger_le_015','observed_relative_CCC_L1_difference','supported_top5_LR','same_composition_CCC_supported'])
    evidence.to_csv(out/'matched_evidence_summary.csv',index=False)
    table.merge(evidence,on='niche',how='left',validate='one_to_one').to_csv(out/'candidate_niches_with_evidence.csv',index=False)
    names=[name for item in summary for name in (item['niche']+' case',item['niche']+' matched control')]
    pd.DataFrame(comp_rows,index=names,columns=types).to_csv(out/'matched_composition.csv')
    np.save(out/'matched_observed_relative_CCC.npy',np.asarray(profiles,np.float32))
    (out/'matched_evidence_audit.json').write_text(json.dumps({'uses_observed_CCC':True,'training_CCC_changed':False,
        'relative_CCC':'Original C divided by each center total C, ONLY as post-fit sensitivity to global RNA-depth scale',
        'matching':'32 nearest control composition profiles, then composition + weak local library-size distance',
        'max_query_centers_per_niche':512,'reused_control_centers_reported':True,
        'communication_support':'At least80% paired Hellinger<=0.15 and >=3 of top5 observed relative-CCC features >2-fold with >=20% case detection',
        'interpretation':'Descriptive evidence in these sections; no cell-level population p-values'},indent=2))
    return table
