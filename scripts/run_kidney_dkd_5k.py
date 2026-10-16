"""Run the independent observed kidney niche experiment on NAS / GPU 2."""
from pathlib import Path
import sys,argparse
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from phenoniche.cohorts.paired_skin import run
from phenoniche.cohorts.condition_niches import ensure_condition_results

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--config',type=Path,default=ROOT/'config/kidney_dkd_5k.json')
    output=run(parser.parse_args().config)
    ensure_condition_results(output,control='HC',case='DKD',calibration_mode='composition_local')
