import sys
from pathlib import Path
sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from phenoniche.v1005.experiment import run_dataset,DEFAULT
if __name__=='__main__':
    import json
    root=Path(__file__).resolve().parents[1]
    for ds in ['hbc1','xenium5k']:
        path=root/'outputs'/ds/'config.json'
        run_dataset(ds,json.loads(path.read_text()) if path.exists() else dict(DEFAULT))
