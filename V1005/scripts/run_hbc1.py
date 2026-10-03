import sys
from pathlib import Path
sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from phenoniche.v1005.experiment import run_dataset
if __name__=='__main__':run_dataset('hbc1')
