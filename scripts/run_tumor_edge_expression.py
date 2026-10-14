"""Run frozen or CCC-priority joint expression experiment; outputs live on NAS."""
from pathlib import Path
import sys,argparse
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from phenoniche.simulation.response_simulation import ExpressionExperimentConfig
from phenoniche.simulation.response_experiment import run_expression_experiment
if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default=str(Path(__file__).resolve().parents[1]/'config/tumor_edge_expression.json'))
    args=parser.parse_args();print(run_expression_experiment(ExpressionExperimentConfig.from_json(args.config)))
