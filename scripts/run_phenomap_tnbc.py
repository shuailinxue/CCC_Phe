#!/usr/bin/env python3
"""Single-GPU worker with optional disjoint multi-GPU slice shards. --smoke never starts full-section training."""
import os,sys
# Set visibility before importing torch; each worker owns one physical GPU.
_gpu_args = sys.argv[1:]
_gpu = _gpu_args[_gpu_args.index('--gpu') + 1] if '--gpu' in _gpu_args else os.environ.get('PHENOMAP_GPU', '2')
if _gpu not in {'0', '1', '2'}: raise ValueError('Only physical GPU 0, 1 or 2 is authorized')
os.environ['CUDA_VISIBLE_DEVICES'] = _gpu
sys.dont_write_bytecode=True
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import argparse,contextlib,fcntl,json,hashlib,subprocess,traceback,time
import numpy as np,pandas as pd,torch
from phenoniche.phenomap_adapter.genes import prepare_panel
from phenoniche.phenomap_adapter.annotation import prepare_reference
from phenoniche.phenomap_adapter.pipeline import run_slice,atomic_json


class Tee:
    def __init__(self,*streams):self.streams=streams
    def write(self,text):
        for stream in self.streams:stream.write(text);stream.flush()
        return len(text)
    def flush(self):
        for stream in self.streams:stream.flush()


def repo_snapshot(path):
    return {'commit':subprocess.check_output(['git','-C',path,'rev-parse','HEAD'],text=True).strip(),
        'status':subprocess.check_output(['git','-C',path,'status','--porcelain'],text=True)}


def fingerprint(c,genes,source,before):
    code=hashlib.sha256()
    for f in sorted((ROOT/'src/phenoniche/phenomap_adapter').glob('*.py')):code.update(f.read_bytes())
    return {'config':c,'panel_sha256':genes,'upstream':before,'adapter_sha256':code.hexdigest(),
        'input_files':{n:{'bytes':(source/n).stat().st_size,'mtime':(source/n).stat().st_mtime_ns} for n in ['cnts.csv','locs.csv','tissue_image.jpg','histology_cell_multifeatures.h5','spot_diameter_pixel.txt'] if (source/n).exists()}}


def save_status(root, statuses, gpu):
    """Merge progress atomically so neither GPU overwrites the other's rows."""
    with (root/'status.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        path=root/'batch_status.csv'
        new=pd.DataFrame(statuses)
        if path.exists():
            old=pd.read_csv(path).fillna('')
            new=pd.concat([old,new],ignore_index=True).drop_duplicates('sample',keep='last')
        temp=root/f'.batch_status_gpu{gpu}.tmp'
        new.to_csv(temp,index=False)
        os.replace(temp,path)


def selected_targets(ready, manifest):
    """Selection changes scheduling only; per-sample model signatures stay valid."""
    if manifest is None:
        return ready
    table = pd.read_csv(manifest)
    if not {'sample','patient_id'}.issubset(table.columns):
        raise ValueError('Selection manifest needs sample and patient_id')
    if table.empty or table['sample'].duplicated().any() or table['patient_id'].duplicated().any():
        raise ValueError('Selection manifest must contain exactly one unique sample per patient')
    available = {p.name:p for p in ready}
    missing = set(table['sample'])-set(available)
    if missing:
        raise ValueError(f'Selected samples not runnable: {sorted(missing)}')
    return [available[name] for name in sorted(table['sample'])]


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',default=str(ROOT/'config/phenomap_tnbc_batch.json'))
    parser.add_argument('--smoke',action='store_true');parser.add_argument('--preflight',action='store_true');parser.add_argument('--sample',default=None)
    parser.add_argument('--selection-manifest', default=None)
    parser.add_argument('--gpu', choices=['0','1','2'], default=_gpu)
    parser.add_argument('--worker-index', type=int, default=0)
    parser.add_argument('--worker-count', type=int, default=1)
    args=parser.parse_args();
    if not 0 <= args.worker_index < args.worker_count: parser.error('Invalid worker shard')
    c=json.loads(Path(args.config).read_text());c['smoke']=args.smoke
    torch.set_num_threads(8);torch.manual_seed(c['seed']);np.random.seed(c['seed'])
    if not torch.cuda.is_available():raise RuntimeError(f'Physical GPU {args.gpu} is required; no silent CPU fallback')
    root=Path(c['output_root']);root=root/'smoke' if args.smoke else root;root.mkdir(parents=True,exist_ok=True)
    with (root/('batch.lock' if args.worker_count == 1 else f'worker_gpu{args.gpu}.lock')).open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        before=repo_snapshot(c['phenomap_repo'])
        with (root/'shared_setup.lock').open('a') as shared_lock:
            fcntl.flock(shared_lock,fcntl.LOCK_EX)
            atomic_json(root/'upstream_readonly.json',before)
        sources=sorted(p for p in Path(c['input_root']).iterdir() if p.is_dir() and (p/'cnts.csv').exists())
        inventory=[]
        for p in sources:
            missing=[n for n in ['cnts.csv','locs.csv','tissue_image.jpg'] if not (p/n).exists()]
            inventory.append({'sample':p.name,'input_status':'ready' if not missing else 'missing_input','missing':';'.join(missing),
                'precomputed_features':(p/'histology_cell_multifeatures.h5').exists()})
        with (root/'shared_setup.lock').open('a') as shared_lock:
            fcntl.flock(shared_lock,fcntl.LOCK_EX)
            pd.DataFrame(inventory).to_csv(root/'input_inventory.csv',index=False)
        ready=[p for p,r in zip(sources,inventory) if r['input_status']=='ready']
        print(f'Physical GPU{args.gpu}:',torch.cuda.get_device_name(0),'slices',len(sources),'ready',len(ready),'missing',len(sources)-len(ready),flush=True)
        for row in inventory:
            if row['input_status']!='ready':print('MISSING',row['sample'],row['missing'],flush=True)
        for p in [Path(c['reference_atlas']),Path(c['cellpose_weights'])]:
            if not p.is_file():raise FileNotFoundError(p)
        # Derive missing diameters ONLY from this cohort's existing calibrated
        # geometry; keep pixel units and expose the calibration record.
        calibration=Path(c['input_root'])/'TNBC1_CN1_C2'
        from scipy.spatial import cKDTree
        loc=pd.read_csv(calibration/'locs.csv',index_col=0);spacing=np.median(cKDTree(loc[['x','y']]).query(loc[['x','y']],k=2)[0][:,1])
        c['cohort_spot_diameter_to_spacing']=float((calibration/'spot_diameter_pixel.txt').read_text())/float(spacing)
        default_manifest=Path(c['output_root'])/'selection/selected_samples.csv'
        manifest=Path(args.selection_manifest) if args.selection_manifest else (default_manifest if default_manifest.exists() and not args.smoke else None)
        chosen=selected_targets(ready,manifest)
        if manifest is not None:
            print(f'One sample per patient: {len(chosen)} selected, manifest {manifest}',flush=True)
            with (root/'shared_setup.lock').open('a') as shared_lock:
                fcntl.flock(shared_lock,fcntl.LOCK_EX)
                atomic_json(root/'active_selection.json',{'manifest':str(manifest),'sha256':hashlib.sha256(manifest.read_bytes()).hexdigest(),'selected_samples':[p.name for p in chosen],'patients':len(chosen),'clinical_outcomes_used':False})
        if args.preflight:
            if repo_snapshot(c['phenomap_repo'])!=before:raise RuntimeError('Upstream repository changed')
            print('Preflight complete; no segmentation/training started.');return
        # Count matrices already exist for image-missing sections too. Include
        # their headers so supplying a missing image later cannot change panel.
        with (root/'shared_setup.lock').open('a') as shared_lock:
            fcntl.flock(shared_lock,fcntl.LOCK_EX)
            genes,panel_sha=prepare_panel(c,root/'shared',sources)
            bundle=prepare_reference(c,genes,root/'shared')
        print('Shared LR+marker genes:',len(genes),flush=True)
        print('Reference donor holdout:',bundle['validation'],flush=True)
        targets=[p for p in chosen if p.name==args.sample] if args.sample else chosen
        if args.smoke:targets=[p for p in ready if p.name=='TNBC1_CN1_C2']
        if not targets:raise ValueError('No matching runnable slices')
        targets=targets[args.worker_index::args.worker_count]
        print(f'Worker GPU{args.gpu}: {len(targets)} assigned slices',flush=True)
        statuses=[{'sample':r['sample'],'status':'missing_input','detail':r['missing']} for r in inventory if r['input_status']!='ready']
        if manifest is not None:
            selected_names={p.name for p in chosen}
            statuses += [{'sample':p.name,'status':'not_selected','detail':'One representative sample per patient; existing caches retained'} for p in ready if p.name not in selected_names]
        failures=0
        for source in targets:
            out=root/source.name;out.mkdir(exist_ok=True);signature=fingerprint(c,panel_sha,source,before)
            with (out/'slice.lock').open('a') as slice_lock:
                fcntl.flock(slice_lock,fcntl.LOCK_EX)
                if (out/'complete.json').exists():
                    if json.loads((out/'complete.json').read_text())!=signature:raise ValueError(f'Completed output has another config/code: {out}; old results preserved')
                    statuses.append({'sample':source.name,'status':'reused_complete','detail':''});print('REUSE',source.name,flush=True);continue
                if (out/'run_signature.json').exists() and json.loads((out/'run_signature.json').read_text())!=signature:
                    raise ValueError(f'Incomplete run signature mismatch: {out}; choose a separate output_root')
                atomic_json(out/'run_signature.json',signature)
                print('START',source.name,flush=True)
                try:
                    with (out/'run.log').open('a',buffering=1) as log,contextlib.redirect_stdout(Tee(sys.stdout,log)),contextlib.redirect_stderr(log):
                        path=run_slice(c,source,out,genes,bundle,args.smoke)
                    if not path.exists():raise RuntimeError('Missing final cells.h5ad')
                    atomic_json(out/'complete.json',signature);statuses.append({'sample':source.name,'status':'complete','detail':str(path)})
                    print('DONE',source.name,flush=True)
                except Exception as exc:
                    failures+=1;torch.cuda.empty_cache();statuses.append({'sample':source.name,'status':'failed','detail':str(exc)})
                    (out/'error.txt').write_text(traceback.format_exc());print('FAILED',source.name,str(exc),flush=True)
                finally:
                    save_status(root,statuses,args.gpu)
                    if repo_snapshot(c['phenomap_repo'])!=before:raise RuntimeError('Read-only upstream repository integrity check failed')
        save_status(root,statuses,args.gpu)
        if args.smoke and not failures:
            from phenoniche.phenomap_adapter.smoke import image_smoke
            image_smoke(c,targets[0],root/'image_smoke')
        atomic_json(root/('batch_summary.json' if args.worker_count == 1 else f'batch_summary_gpu{args.gpu}.json'),{'input_slices':len(sources),'input_ready':len(ready),'missing_input':len(sources)-len(ready),
            'selected_samples_total':len(chosen),'selection_manifest':str(manifest) if manifest else None,
            'attempted':len(targets),'failed':failures,'GPU_physical_index':int(args.gpu),'phenomap_unchanged':repo_snapshot(c['phenomap_repo'])==before,
            'smoke_only':args.smoke,'shared_genes':len(genes),'reference_validation':bundle['validation']})
        print('Finished. Summary:',root/('batch_summary.json' if args.worker_count == 1 else f'batch_summary_gpu{args.gpu}.json'),flush=True)
        if failures:raise SystemExit(1)


if __name__=='__main__':main()
