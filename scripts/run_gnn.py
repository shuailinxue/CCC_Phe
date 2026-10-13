#!/usr/bin/env python
"""Expression-only GNN: prepare/audit/train, reuse complete results, or infer."""
from pathlib import Path
import sys,argparse,json
from dataclasses import replace
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from phenoniche.gnn.config import GNNConfig
from phenoniche.gnn.audit import audit_study
from phenoniche.gnn.pipeline import run,inference,ensure_results

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['audit','train','infer','ensure'],nargs='?',default='audit')
    parser.add_argument('--config',default=str(Path(__file__).resolve().parents[1]/'config/gnn.json'))
    parser.add_argument('--output');parser.add_argument('--lambda-comp',type=float)
    args=parser.parse_args();config=GNNConfig.from_json(args.config)
    if args.output:config=replace(config,output_dir=args.output)
    if args.lambda_comp is not None:config=replace(config,lambda_comp=args.lambda_comp)
    if args.action=='ensure' and (args.output or args.lambda_comp is not None):parser.error('ensure uses its configuration file; edit the file instead of CLI overrides')
    actions={'audit':lambda:audit_study(config),'train':lambda:run(config),'infer':lambda:{'arrays':list(inference(config))},'ensure':lambda:ensure_results(args.config)}
    print(json.dumps(actions[args.action](),indent=2))
