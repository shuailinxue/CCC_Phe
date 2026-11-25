#!/usr/bin/env python3
"""Outcome-blind native-H&E + raw-ST patient representative selection."""
import argparse
import os
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
sys.dont_write_bytecode = True


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',default=str(ROOT/'config/phenomap_tnbc_batch.json'))
    parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--seg-gpu',choices=['0','2'])
    args=parser.parse_args()
    import json
    import hashlib
    import numpy as np
    import pandas as pd
    from phenoniche.selection.tnbc import (POLICY, atomic_json, audit_one, rank_samples,
        segmentation_fields, contact_pages, patient_id)
    config=json.loads(Path(args.config).read_text())
    output=Path(config['output_root'])/'selection'; output.mkdir(parents=True,exist_ok=True)
    cache=output/'candidates'; cache.mkdir(exist_ok=True)
    if args.seg_gpu is not None:
        os.environ['CUDA_VISIBLE_DEVICES']=args.seg_gpu
        import torch
        torch.set_num_threads(4)
        from phenoniche.phenomap_adapter.features import cellpose_model
        model=cellpose_model(config)
        folders=sorted(p for p in cache.iterdir() if (p/'fields.npz').exists())
        folders=folders[int(args.seg_gpu)//2::2]
        for n, folder in enumerate(folders):
            digest=hashlib.sha256((folder/'fields.npz').read_bytes()).hexdigest()
            path=folder/'segmentation_QC.json'
            code=hashlib.sha256((ROOT/'src/phenoniche/selection/tnbc.py').read_bytes()).hexdigest()
            if path.exists():
                old=json.loads(path.read_text())
                if old.get('fields_sha256')==digest and old.get('code_sha256')==code:
                    continue
            with np.load(folder/'fields.npz') as data:
                fields=data['rgb']
            results=[]
            for rgb in fields:
                mask,_,_=model.eval(rgb,batch_size=8,flow_threshold=.4,cellprob_threshold=0.,min_size=15,invert=False)
                results.append(segmentation_fields(mask,rgb))
            atomic_json(path, {'fields_sha256':digest,'code_sha256':code,'fields':results,
                'segmentation_supported_fraction':float(np.mean([r['supported'] for r in results])),
                'segmentation_ring_contrast':float(np.median([r['nuclear_ring_contrast'] for r in results])),
                'median_plausible_nuclei':float(np.median([r['plausible_nuclei'] for r in results])),
                'interpretation':'Native H&E segmentation plausibility pilot; not ground-truth segmentation accuracy'})
            print(f'GPU{args.seg_gpu} nuclei QC {n+1}/{len(folders)} {folder.name}',flush=True)
        return
    from concurrent.futures import ProcessPoolExecutor, as_completed
    from phenoniche.phenomap_adapter.genes import gene_parts
    import subprocess
    atomic_json(output/'policy.json', {**POLICY,'inadequate_patient_action':'Defer patient if no candidate passes absolute H&E, segmentation and spot QC; never force a poor sample into production'})
    sources=sorted(p for p in Path(config['input_root']).iterdir() if p.is_dir() and (p/'cnts.csv').exists())
    genes=(Path(config['output_root'])/'shared/genes.txt').read_text().splitlines()
    atlas=pd.read_csv(config['lr_atlas'],sep='\t')
    pairs=[gene_parts(r.ligand)+gene_parts(r.receptor) for r in atlas.itertuples()]
    pairs=[parts for parts in pairs if all(g in genes for g in parts)]
    djpeg=Path(sys.executable).parent/'djpeg'
    if not djpeg.is_file():
        raise FileNotFoundError(djpeg)
    rows=[]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        pending={pool.submit(audit_one,p,cache/p.name,genes,pairs,djpeg):p for p in sources}
        for n,future in enumerate(as_completed(pending)):
            try:
                row=future.result()
            except Exception as exc:
                source=pending[future]
                row={'sample':source.name,'patient_id':patient_id(source.name),'input_ready':False,'error':f'{type(exc).__name__}: {exc}'}
                atomic_json(cache/source.name/'read_error.json',row)
            rows.append(row)
            print(f'Raw ST + native H&E {n+1}/{len(sources)} {row["sample"]} ready={row["input_ready"]}',flush=True)
    # Use exactly the production Cellpose model, only four small fields per sample.
    processes=[]; logs=[]
    for gpu in ['0','2']:
        log=(output/f'nuclei_QC_gpu{gpu}.log').open('a',buffering=1); logs.append(log)
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpu,PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4')
        processes.append(subprocess.Popen([sys.executable,'-B',str(Path(__file__).resolve()),'--config',args.config,'--seg-gpu',gpu],env=env,stdout=log,stderr=subprocess.STDOUT))
    codes=[p.wait() for p in processes]
    for log in logs:log.close()
    if any(codes):raise RuntimeError(f'Native H&E segmentation pilots failed: {codes}; inspect nuclei_QC logs')
    for row in rows:
        if row['input_ready']:
            qc=json.loads((cache/row['sample']/'segmentation_QC.json').read_text())
            row.update({k:qc[k] for k in ['segmentation_supported_fraction','segmentation_ring_contrast','median_plausible_nuclei']})
    results=rank_samples(pd.DataFrame(rows),cache)
    # Retain review recommendations in the audit, but do not launch an inadequate sample.
    deferred=results.selected.astype(bool) & ~results.quality_pass.fillna(False).astype(bool)
    results.loc[deferred,'selected']=False
    results.loc[deferred,'selection_note']='Deferred: no patient candidate jointly passes absolute spot and H&E/segmentation QC'
    # Read patient identifiers ONLY, not outcomes, subtype, grade, or survival fields.
    clinical=Path(config['input_root']).parent/'Processed_Clinical/clinical.tsv'
    patient_ids=set(pd.read_csv(clinical,sep='\t',usecols=['Sample_ID'])['Sample_ID'].astype(int))
    selected=results[results.selected.astype(bool)].copy()
    if set(results.patient_id)!=patient_ids or not set(selected.patient_id).issubset(patient_ids) or selected.patient_id.duplicated().any():
        raise ValueError('Candidate IDs must match the clinical IDs; selected patients must be a unique subset')
    flat=results.drop(columns=['field_stats'],errors='ignore')
    flat.to_csv(output/'candidate_scores.csv',index=False)
    manifest=selected.drop(columns=['field_stats'],errors='ignore').sort_values('patient_id')
    temp=output/'selected_samples.tmp';manifest.to_csv(temp,index=False);temp.replace(output/'selected_samples.csv')
    contact_pages(results,cache,output/'QC_contact_sheets')
    review=selected[~selected.quality_pass.astype(bool)]
    atomic_json(output/'selection_summary.json',{'patient_count':len(patient_ids),'candidates':len(rows),
        'available_candidates':int(results.input_ready.sum()),'selected_count':len(selected),
        'selected_one_per_patient':True,'clinical_outcomes_used':False,'weights':POLICY['weights'],
        'deferred_patients':sorted(patient_ids-set(selected.patient_id.astype(int))),
        'review_required_patients':review.patient_id.astype(int).tolist(),
        'all_selected_pass_absolute_QC':bool(review.empty),
        'manifest_sha256':hashlib.sha256((output/'selected_samples.csv').read_bytes()).hexdigest()})
    print('SELECTED',len(selected),'patients; QC-review-needed',len(review),'; manifest',output/'selected_samples.csv',flush=True)


if __name__=='__main__':main()
