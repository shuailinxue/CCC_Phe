"""Run the restartable Prime 5K graph audit without model training."""
import sys
from pathlib import Path
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from phenoniche.v1005.graph_sensitivity import run_prime_diagnostic
if __name__=='__main__':run_prime_diagnostic()
