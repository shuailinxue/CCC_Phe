#!/usr/bin/env python
"""Independent downstream analysis. Audit is the default action."""
from pathlib import Path
import sys
import argparse
from dataclasses import replace
import json
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from phenoniche.gnn.config import GNNConfig
from phenoniche.gnn.audit import audit_study
from phenoniche.gnn.pipeline import run,inference,retrain_vector_baseline,compare_outputs

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['audit','train','infer','baseline','compare'],nargs='?',default='audit')
    parser.add_argument('--config',default=str(Path(__file__).resolve().parents[1]/'config/gnn.json'))
    parser.add_argument('--mode',choices=['ccc_only','upstream','downstream','both'])
    parser.add_argument('--output');parser.add_argument('--other-output')
    parser.add_argument('--prior-control',choices=['none','shuffle_links','shuffle_activity'])
    parser.add_argument('--contrastive',action='store_true')
    args=parser.parse_args();config=GNNConfig.from_json(args.config)
    overrides={k:v for k,v in {'mode':args.mode,'output_dir':args.output,'prior_control':args.prior_control}.items() if v is not None}
    if args.contrastive: overrides['contrastive']=True
    config=replace(config,**overrides)
    actions={'audit':lambda:audit_study(config),'train':lambda:run(config),'infer':lambda:{'arrays':list(inference(config))},'baseline':lambda:retrain_vector_baseline(config),'compare':lambda:compare_outputs(config,args.other_output)}
    result=actions[args.action]()
    if args.action=='audit': result={key:result[key] for key in ('ready_for_training','blockers','n_cells','panel_genes','fixed_CCC_features','fully_measurable_LR_pairs')}
    print(json.dumps(result,indent=2))
