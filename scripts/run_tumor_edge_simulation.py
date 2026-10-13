#!/usr/bin/env python
"""Full benchmark is opt-in; --smoke executes only a 180-cell tiny experiment."""
from pathlib import Path
import sys,argparse
from dataclasses import replace
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from phenoniche.simulation.config import SimulationConfig
from phenoniche.simulation.workflow import run_experiment
from phenoniche.simulation.plotting import SimulationReport
from phenoniche.storage import STORAGE_ROOT

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',default=str(Path(__file__).resolve().parents[1]/'config/tumor_edge_simulation.json'));parser.add_argument('--smoke',action='store_true');parser.add_argument('--run-full',action='store_true')
    args=parser.parse_args();config=SimulationConfig.from_json(args.config)
    if args.smoke and args.run_full:parser.error('Choose --smoke or --run-full')
    if args.smoke:config=replace(config,n_cells=180,n_genes=128,n_lr=50,replicates=3,core_radius_um=28.,edge_width_um=24.,outer_radius_um=80.,block_um=30.,reference_cells_per_type=6,sc_cube_epochs=1,sc_cube_hidden=8,gnn_epochs=2,ae_stage1_epochs=2,ae_stage2_epochs=2,batch_size=16,hidden=12,k_max=6,device='cpu',figure_dpi=120,output_dir=str(STORAGE_ROOT/'tmp/tumor_edge_smoke'))
    elif not args.run_full:parser.error('Full simulation requires --run-full; use --smoke for lightweight validation')
    output=run_experiment(config);report=SimulationReport(output)
    import matplotlib.pyplot as plt
    for method in ('ground_truth','main_figure','recovered_programs','dictionary','composition_audit','adaptive_niches','reconstruction'):plt.close(getattr(report,method)())
    print(output)
