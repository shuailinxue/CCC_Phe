import sys
from pathlib import Path
sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from phenoniche.v1005.experiment import run_dataset,DEFAULT
if __name__=='__main__':
    import torch
    c=dict(DEFAULT)
    if torch.cuda.device_count()>2:c['device']='cuda:2'
    run_dataset('xenium5k',c)
