"""Post-training three-graph fusion; never fits or changes DeepTensorCCC."""
import sys
from pathlib import Path
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from phenoniche.v1005.niche_graph import run_graph_selection

if __name__=='__main__':
    for dataset in ['hbc1','xenium5k']:run_graph_selection(dataset)
