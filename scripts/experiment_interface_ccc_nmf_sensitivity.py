"""Post-hoc rank/seed sensitivity of the C3 observed CCC NMF pilot."""
from pathlib import Path
import json
import sys

import anndata as ad
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import adjusted_rand_score

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from phenoniche.shared.clustering import gpu_kmeans
from phenoniche.shared.io import write_json
from validate_interface_biological_simulation import family
from experiment_interface_ccc_nmf import nmf, WORK, SOURCE, OUT, DEVICE


def run():
    output=OUT/'rank_seed_sensitivity';output.mkdir(exist_ok=True)
    target=Path(json.loads((SOURCE/'target_cache.json').read_text())['folder']);ids=np.load(OUT/'cells.npy')
    scale=np.load(target/'statistics.npz')['feature_scale'];array=np.asarray(np.load(target/'ccc.npy',mmap_mode='r')[ids],dtype=np.float32)/scale
    x=torch.as_tensor(array,device=DEVICE);torch.set_num_threads(4);torch.set_float32_matmul_precision('high')
    results={}
    for rank in (16,64):
        for seed in (40700,40701):
            print('Rank',rank,'seed',seed,flush=True);w,h,history=nmf(x,rank=rank,seed=seed)
            W=w.cpu().numpy();H=h.cpu().numpy();lab={}
            for k in (2,3,4,6):lab[f'K{k}']=gpu_kmeans(W,k,seed=40700,n_init=10,device=DEVICE).labels_.astype(np.int16)
            name=f'rank{rank}_seed{seed}';np.savez_compressed(output/f'{name}.npz',activity=W,loadings=H,**lab)
            results[name]=dict(rank=rank,seed=seed,last_mse=history[-1]['mse'])
    # Evaluation labels are loaded only once all candidate fits are frozen.
    a=ad.read_h5ad(SOURCE/'tissue.h5ad',backed='r');r=a.obs.design_region_id.to_numpy()[ids]
    ia=np.flatnonzero(r==2);ib=np.flatnonzero(r==3);idx=np.r_[ia,ib]
    np.testing.assert_array_equal(a.obs.template_cell_id.to_numpy()[ids][ia],a.obs.template_cell_id.to_numpy()[ids][ib])
    f=pd.read_csv(target/'features.csv');families=np.array([family(row) for row in f.itertuples()],object)
    chosen=np.flatnonzero(families=='T → tumor/stroma · IFNG');rows=[]
    for name,info in results.items():
        d=np.load(output/f'{name}.npz');pred=d['activity']@d['loadings'][:,chosen]
        ratio=float(pred[ia].mean()/max(pred[ib].mean(),1e-12))
        for k in (2,3,4,6):
            lab=d[f'K{k}'];rows.append(dict(**info,K=k,AB_ARI=float(adjusted_rand_score(r[idx],lab[idx])),
              paired_same=float(np.mean(lab[ia]==lab[ib])),predicted_IFNG_A_over_B=ratio))
    df=pd.DataFrame(rows);df.to_csv(output/'evaluation.csv',index=False)
    write_json(output/'complete.json',dict(scope='Post-hoc rank/seed sensitivity, scaled CCC; not independent validation',
      ranks=[16,64],seeds=[40700,40701],truth_in_fit=False,rows=rows));print(df.to_string(index=False));a.file.close()


if __name__=='__main__':run()
