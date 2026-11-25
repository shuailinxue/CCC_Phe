"""Run the same communication-network experiment with three composition classes."""
from pathlib import Path
import sys

import numpy as np
from scipy import sparse

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'scripts')]
from phenoniche.ccc.coarse_conditional import within_class_graph
import run_interface_coarse_ccc_network as experiment

WORK = Path('/home/nas3/biod/xueshuailin/CCC_Phe/workspace')
PREVIOUS = WORK/'outputs/tumor_immune_interface/coarse_conditional_ccc'
OUTPUT = WORK/'outputs/tumor_immune_interface/three_coarse_ccc_network'
INPUT = OUTPUT/'composition_input'


def run():
    INPUT.mkdir(parents=True, exist_ok=True)
    coarse = np.load(PREVIOUS/'composition_K_sensitivity.npz')['3'].astype(np.int16)
    np.save(INPUT/'coarse_labels.npy', coarse)
    graph = sparse.load_npz(WORK/'outputs/tumor_immune_interface/ccc_background_signatures/statistical_graph.npz')
    sparse.save_npz(INPUT/'within_coarse_graph.npz', within_class_graph(graph, coarse))
    experiment.PREV = INPUT
    experiment.OUT = OUTPUT
    experiment.CONFIG = ROOT/'config/interface_coarse_ccc_network_three.json'
    experiment.run()


if __name__ == '__main__':
    run()
