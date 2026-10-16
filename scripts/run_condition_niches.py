"""Post-fit control-background niche discovery; does not train the GNN."""
from pathlib import Path
import sys,argparse
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from phenoniche.cohorts.condition_niches import ensure_condition_results
import pandas as pd
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--source',type=Path,required=True);ap.add_argument('--control',default='NL');ap.add_argument('--case',default='AD');args=ap.parse_args()
    out=ensure_condition_results(args.source,args.control,args.case);print(pd.read_csv(out/'candidate_niches_with_evidence.csv').to_string(index=False),flush=True)
