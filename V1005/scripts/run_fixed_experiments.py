"""Run independent seed jobs with bounded local processes, then aggregate and render."""
import sys,os,json,subprocess,concurrent.futures
from pathlib import Path
sys.dont_write_bytecode=True
root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root/'src'))
from phenoniche.v1005.experiment import DEFAULT,run_dataset
from phenoniche.v1005.data import json_write

def worker(ds,seed):
    import time
    out=root/'outputs'/ds;log=out/f'run_seed_{seed}.log'
    if (out/f'baseline_metrics_seed_{seed}.csv').exists():return ds,seed,0
    with log.open('a') as stream:
        ret=subprocess.run([sys.executable,__file__,'--worker',ds,str(seed)],stdout=stream,stderr=subprocess.STDOUT)
    print(ds,seed,'exit',ret.returncode,flush=True);return ds,seed,ret.returncode

def aggregate(ds):
    import numpy as np,pandas as pd
    from phenoniche.v1005.evaluation import stability_tables
    out=root/'outputs'/ds;labels={};programs={};tables=[]
    for seed in DEFAULT['seeds']:
        t=pd.read_csv(out/f'baseline_metrics_seed_{seed}.csv');tables.append(t)
        for method in t.method:
            labels[(method,seed)]=np.load(out/f'seed_{seed}_{method}_labels.npy')
            if method in ['CCC-tensor','Proposed','CCC-autoencoder-no-prior']:
                tag='no_prior' if method=='CCC-autoencoder-no-prior' else 'tensor'
                programs[(method,seed)]=np.load(out/f'seed_{seed}_{tag}_H.npy')
    pd.concat(tables).to_csv(out/'baseline_metrics.csv',index=False)
    stability_tables(labels,programs).to_csv(out/'stability.csv',index=False)
    (out/'run.log').write_text('\n'.join((out/f'run_seed_{seed}.log').read_text() for seed in DEFAULT['seeds']))
    json_write(out/'complete.json',{'status':'complete','five_seeds':DEFAULT['seeds'],'training_revision':'simplex_staged'})

if __name__=='__main__':
    if len(sys.argv)>1 and sys.argv[1]=='--worker':
        ds,seed=sys.argv[2],int(sys.argv[3]);config=json.loads((root/'outputs'/ds/'config.json').read_text())
        run_dataset(ds,config,only_seed=seed)
    else:
        import torch
        for ds in ['hbc1','xenium5k']:
            config=dict(DEFAULT)
            if ds=='xenium5k' and torch.cuda.device_count()>2:config['device']='cuda:2'
            path=root/'outputs'/ds/'config.json'
            if path.exists() and json.loads(path.read_text())!=config:raise ValueError('Configuration differs')
            json_write(path,config)
        jobs=[(ds,seed) for seed in DEFAULT['seeds'] for ds in ['hbc1','xenium5k']]
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            results=list(pool.map(lambda job:worker(*job),jobs))
        if any(code for _,_,code in results):raise RuntimeError('One or more seed jobs failed; see individual logs')
        for ds in ['hbc1','xenium5k']:aggregate(ds)
        subprocess.run([sys.executable,str(root/'scripts/finalize.py')],check=True)
