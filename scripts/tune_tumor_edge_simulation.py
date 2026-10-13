#!/usr/bin/env python
"""CCC+regularization only; numerical settings selected without niche ground truth."""
from pathlib import Path
import sys,json
from dataclasses import replace
import numpy as np,pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from phenoniche.simulation.config import SimulationConfig
from phenoniche.simulation.comparison import reuse_inputs
from phenoniche.simulation.workflow import run_experiment
from phenoniche.storage import STORAGE_ROOT

if __name__=='__main__':
    root=STORAGE_ROOT/'outputs/tumor_edge_simulation_ccc_tuning';root.mkdir(parents=True,exist_ok=True)
    source=STORAGE_ROOT/'outputs/tumor_edge_simulation_seedcounts'
    base=SimulationConfig.from_json(Path(__file__).resolve().parents[1]/'config/tumor_edge_simulation.json')
    settings=[('rms_huber',1.,'huber',1.),('partial_huber',.5,'huber',3.),('native_huber',0.,'huber',20.),('native_mse',0.,'mse',1.)]
    results=[];configs={};folds=np.load(source/'input/folds.npy');val=np.flatnonzero(folds==1);rms=np.load(source/'input/preprocessing.npz')['feature_scale'];target=np.load(source/'input/ccc.npy',mmap_mode='r')
    for name,power,error,delta in settings:
        cfg=replace(base,lambda_comp=0.,initialize_composition=False,lambda_reg=.005,ccc_scale_power=power,ccc_error=error,ccc_huber_delta=delta,output_dir=str(root/name))
        reuse_inputs(source,cfg);print('EXPERIMENT',name,flush=True);out=run_experiment(cfg);configs[name]=cfg.to_dict()
        z=np.load(out/'gnn/Z.npy',mmap_mode='r');h=np.load(out/'gnn/H.npy');mask=np.load(out/'input/preprocessing.npz')['feature_mask'];native_sse=0.;balanced_huber=0.;n=0
        for start in range(0,len(val),256):
            rows=val[start:start+256];e=(np.asarray(z[rows])@h-np.asarray(target[rows]))[:,mask];native_sse+=float((e.astype(np.float64)**2).sum());d=np.abs(e/rms[mask]);balanced_huber+=float(np.where(d<=1,.5*d*d,d-.5).sum());n+=e.size
        scores=pd.read_csv(out/'metrics.csv');scores=scores[(scores.method=='CCC_Phe GNN (fixed K)')&(scores.evaluation=='heldout_test')].iloc[0]
        mi=pd.read_csv(out/'MI_matches.csv');held=json.loads((out/'gnn/heldout_metrics.json').read_text())
        results.append({'experiment':name,'validation_native_MSE':native_sse/n,'validation_RMS_Huber':balanced_huber/n,'best_epoch':held['best_epoch'],'test_fixedK_ARI':scores.overall_ARI,'test_edge_F1':scores.edge_F1,'test_edge_AB_accuracy':scores.edge_AB_accuracy,'MI_test_edge_Spearman_mean':mi[mi.scope=='test_edge'].activity_Spearman.mean(),'output_dir':str(out),'robust_error':error=='huber'})
        pd.DataFrame(results).to_csv(root/'comparison.csv',index=False)
    table=pd.DataFrame(results)
    # MSE is a diagnostic control; the deployed objective remains robust Huber.
    chosen=table[table.robust_error].sort_values('validation_native_MSE').iloc[0]
    (root/'selection.json').write_text(json.dumps({'criterion':'Lowest validation native CCC MSE among robust Huber objectives; no true niche/MI/test score used','selected_config':configs[chosen.experiment]},indent=2)+'\n')
    print(table.to_string(index=False));print('SELECTED',chosen.experiment)
