from pathlib import Path
import sys,argparse
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from phenoniche.cohorts.paired_skin import run
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--config',type=Path,default=ROOT/'config/atopic_dermatitis_5k.json');run(parser.parse_args().config)
