"""LR measurement-noise controls, preserving counts outside LR and original files."""
from pathlib import Path
from dataclasses import replace
import argparse,sys,json,copy
import numpy as np,pandas as pd,anndata as ad,torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT/'scripts'))
from diagnose_gene_rich_niches import read_data,diagnostics,sampled_fit
from phenoniche.storage import STORAGE_ROOT
from phenoniche.simulation.config import SimulationConfig
from phenoniche.simulation.adapter import adapt
from phenoniche.simulation.programs import program_specs
from phenoniche.simulation.counterfactual_controls import noise_control,coverage_supervision_mask




def main():
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['single_nb','lr_expectation','structured_nb','structured_low_basal','program_latent','program_sparse','program_coherent'],required=True);p.add_argument('--epochs',type=int,default=25);p.add_argument('--role-mean',type=float);p.add_argument('--fold-change',type=float);p.add_argument('--experiments',default='standard,lr_pair');p.add_argument('--consistent-mask',action='store_true',help='Fit scale after training-only 10% expression mask');a=p.parse_args()
    torch.set_num_threads(6);torch.cuda.set_device(2)
    source=STORAGE_ROOT/'outputs/tumor_edge_gene_rich_simulation';key=a.mode+(f'_mean{a.role_mean:g}' if a.role_mean is not None else '')+(f'_fold{a.fold_change:g}' if a.fold_change is not None else '')+('_consistent_mask' if a.consistent_mask else '');out=STORAGE_ROOT/'outputs/tumor_edge_gene_rich_diagnostics'/key
    original,data,payload=read_data(source)
    if (out/'injected.h5ad').exists():adata=ad.read_h5ad(out/'injected.h5ad');lr=pd.read_csv(out/'lr.csv')
    else:adata,lr=noise_control(source,out,a.mode,a.role_mean,a.fold_change)
    config=replace(SimulationConfig.from_json(ROOT/'config/tumor_edge_gene_rich_simulation.json'),output_dir=str(out),device='cuda:2')
    controlled=adapt(adata,lr,config,out,expression_coverage=.1 if a.consistent_mask else None)
    if a.mode.startswith('structured') or a.mode in ('program_sparse','program_coherent'):
        # Same training-only type-specific expression coverage used by the real
        # CCC workflow. Numeric CCC values/order remain intact; mask unsupported
        # sender/receiver features, not an invented true-MI whitelist.
        controlled.feature_mask=coverage_supervision_mask(adata,controlled,.1)
        np.save(out/'coverage_feature_mask.npy',controlled.feature_mask)
        controlled.features.assign(effective_supervision=controlled.feature_mask).to_csv(out/'supervision_features.csv',index=False)
        print(a.mode,'coverage-valid CCC features',int(controlled.feature_mask.sum()),flush=True)
    # Parameter/feature masks fitted on this control's training split only.
    for name in a.experiments.split(','):
        target=out/name;target.mkdir(exist_ok=True)
        if not (target/'summary.json').exists():sampled_fit(adata,controlled,payload,target,name,1600,a.epochs)
    c=np.asarray(controlled.targets.matrix);h=np.load(out/'ground_truth/MI_dictionary.npy');scores=c@h.T/np.maximum(h.sum(1),1e-12)
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import roc_auc_score,accuracy_score
    truth=adata.obs.true_niche.to_numpy();edge=np.isin(truth,[2,3]);tr=controlled.rows[0][edge[controlled.rows[0]]];va=controlled.rows[1][edge[controlled.rows[1]]];te=controlled.rows[2][edge[controlled.rows[2]]]
    clf=make_pipeline(StandardScaler(),LogisticRegression()).fit(scores[tr,:2],truth[tr]==3);records=[]
    for name,r in [('validation',va),('test',te)]:
        prob=clf.predict_proba(scores[r,:2])[:,1];records.append({'mode':a.mode,'scope':name,'known_MI_diagnostic_AUC':roc_auc_score(truth[r]==3,prob),'known_MI_accuracy':accuracy_score(truth[r]==3,prob>.5)})
    pd.DataFrame(records).to_csv(out/'signal.csv',index=False)

if __name__=='__main__':main()
