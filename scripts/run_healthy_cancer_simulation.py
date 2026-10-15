"""Run/reuse the paired healthy/cancer simulation using the current GNN."""
from pathlib import Path
import argparse
from phenoniche.simulation.paired_tissues import PairedConfig
from phenoniche.simulation.paired_workflow import run

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',default=str(Path(__file__).resolve().parents[1]/'config/healthy_cancer_communication.json'))
    print(run(PairedConfig.from_json(p.parse_args().config)))
