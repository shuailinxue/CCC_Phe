"""Matched loss-weight experiments; select by validation CCC, never niche truth."""
from dataclasses import replace
from pathlib import Path
import hashlib,json,shutil
import pandas as pd
from .config import SimulationConfig
from .workflow import run_experiment,protect
from ..storage import STORAGE_ROOT


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def run_weight_comparison(config,weights=(0.,.005,.05)):
    root=STORAGE_ROOT/'outputs/tumor_edge_simulation_weight_comparison';root.mkdir(parents=True,exist_ok=True)
    anchor=root/'anchor_config.json'
    if anchor.exists():base=SimulationConfig.from_json(anchor)
    else:base=config;anchor.write_text(json.dumps(base.to_dict(),indent=2)+'\n')
    source=run_experiment(base);results=[];configs={}
    identity={name:digest(source/name) for name in ('injected.h5ad','input/ccc.npy')}
    for weight in weights:
        label='lambda_'+format(weight,'g').replace('.','p')
        cfg=base if weight==base.lambda_comp else replace(base,lambda_comp=weight,output_dir=str(root/label))
        output=protect(cfg);configs[label]=cfg.to_dict()
        if output!=source:
            for name in ('before_injection.h5ad','injected.h5ad','lr.csv','seed_reference.h5ad','expression_fidelity.json','input/ccc.npy'):
                target=output/name;target.parent.mkdir(parents=True,exist_ok=True)
                if not target.exists():shutil.copy2(source/name,target)
            # AE and expression controls are independent of lambda_comp; reuse
            # their computed outputs on this exact same data and physical split.
            for name in ('Z.npy','P.npy','H.npy','activity.npy','complete.json'):
                target=output/'baselines/ccc_ae'/name;target.parent.mkdir(parents=True,exist_ok=True)
                if not target.exists():shutil.copy2(source/'baselines/ccc_ae'/name,target)
            for name,expected in identity.items():
                if digest(output/name)!=expected:raise ValueError('Ablation data changed: '+name)
        run_experiment(cfg)
        held=json.loads((output/'gnn/heldout_metrics.json').read_text());history=pd.read_csv(output/'gnn/training.csv')
        epoch=history[history.epoch==held['best_epoch']].iloc[0]
        frame=pd.read_csv(output/'metrics.csv');test=frame[(frame.method=='CCC_Phe GNN (fixed K)')&(frame.evaluation=='heldout_test')].iloc[0]
        results.append({'experiment':label,'lambda_comp':weight,'best_epoch':held['best_epoch'],'validation_CCC':float(epoch.validation_ccc),'test_CCC':held['test']['ccc'],'test_composition_KL':held['test']['composition'],**test[['overall_ARI','edge_F1','edge_AB_ARI','edge_AB_accuracy']].to_dict(),'output_dir':str(output)})
    table=pd.DataFrame(results);table.to_csv(root/'weight_comparison.csv',index=False)
    selected=table.sort_values(['validation_CCC','lambda_comp']).iloc[0]
    (root/'validation_choice.json').write_text(json.dumps({'selection_rule':'Lowest validation CCC reconstruction; no ground-truth/test metric used','selected_config':configs[selected.experiment],'shared_data_sha256':identity},indent=2)+'\n')
    return table
