from pathlib import Path
import sys
import argparse
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from phenoniche.cohorts.dermatomyositis import run
if __name__=='__main__':
    parser=argparse.ArgumentParser(description='Independent paired skin communication-niche analysis')
    parser.add_argument('--config',type=Path,default=ROOT/'config/dermatomyositis_5k.json')
    run(parser.parse_args().config)
