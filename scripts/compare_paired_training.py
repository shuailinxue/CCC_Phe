"""Input-only then diverse-training controls on identical original holdout.

All supervised-center/update budgets are matched. Use the old training scales
to report directly comparable CCC errors; no test labels select any setting.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from phenoniche.simulation.paired_tissues import PairedConfig
from phenoniche.simulation.paired_workflow import run
from phenoniche.simulation.response_simulation import checksum

ROOT=Path(__file__).resolve().parents[1]
OLD=Path('/home/nas3/biod/xueshuailin/CCC_Phe/workspace/outputs/healthy_cancer_communication')


def compare(outputs):
    reference=np.load(OLD/'input/ccc.npy',mmap_mode='r')
    p=np.load(OLD/'input/preprocessing.npz');mask=p['feature_mask'];scale=p['feature_scale'][mask]
    fold=np.load(OLD/'input/folds.npy');records=[]
    for name,path in outputs:
        path=Path(path);prediction=np.load(path/'reconstruction.npy',mmap_mode='r')
        current_target=np.load(path/'input/ccc.npy',mmap_mode='r')
        for start in range(0,len(reference),512):
            np.testing.assert_allclose(current_target[start:start+512],reference[start:start+512],atol=1e-6)
        metrics=pd.read_csv(path/'metrics.csv').query("scope == 'heldout_slices'").iloc[0]
        summary=json.loads((path/'summary.json').read_text());history=pd.read_csv(path/'gnn/training.csv')
        best=json.loads((path/'gnn/heldout_metrics.json').read_text())
        row={'experiment':name,'encoder_genes':summary['expression_genes'],
             'training_slices':summary.get('training_slices',4),'training_centers':summary['split_sizes'][0],
             'best_epoch':best['best_epoch'],'epochs_run':len(history),'K':int(metrics.K),
             'heldout_ARI':metrics.ARI,'heldout_disease_F1':metrics.disease_niche_F1}
        for split,label in [(1,'validation'),(2,'test')]:
            ids=np.flatnonzero(fold==split);error=[];sse=0.;total=0.
            for start in range(0,len(ids),512):
                block=ids[start:start+512];target=np.asarray(reference[block])[:,mask]
                pred=np.asarray(prediction[block])[:,mask];d=np.abs((pred-target)/scale)
                error.append(np.where(d<3,.5*d*d,3*(d-1.5)).ravel());sse+=float(((pred-target)**2).sum());total+=float((target**2).sum())
            row['common_'+label+'_Huber']=float(np.concatenate(error).mean())
            row['common_'+label+'_relative_RMSE']=float(np.sqrt(sse/total))
        records.append(row)
    return pd.DataFrame(records)


def main():
    cfg=PairedConfig.from_json(ROOT/'config/healthy_cancer_communication.json')
    control=PairedConfig.from_json(ROOT/'config/healthy_cancer_input_control.json')
    hashes={f:checksum(OLD/f) for f in ('tissues.h5ad','gnn/checkpoint.pt','gnn/Z.npy','gnn/H.npy','metrics.csv','assignments.csv.gz')}
    print('Input-only control',flush=True);b=run(control)
    print('Matched-update diverse-training experiment',flush=True);c=run(cfg)
    table=compare([('Original all genes',OLD),('LR + audited markers',b),('LR + markers; diverse slices',c)])
    table.to_csv(c/'experiment_comparison.csv',index=False);print(table.to_string(index=False),flush=True)
    assert all(checksum(OLD/f)==h for f,h in hashes.items())
    (c/'comparison_audit.json').write_text(json.dumps({'original_results_unchanged':True,'source_hashes':hashes,
        'original_validation_and_test_CCC_unchanged':True,'common_loss_scales':'Original training reference, never held-out refit',
        'training_update_budget_matched':bool(table.training_centers.nunique()==1),
        'test_used_for_model_selection':False},indent=2)+'\n')


if __name__=='__main__':main()
