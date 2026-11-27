"""Train the default direct-network V1 on existing PhenoMap observation shards."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from phenoniche.v1.network_workflow import fit_run
from phenoniche.v1.probabilistic import Config
from phenoniche.v1.storage import Network


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--network",required=True,type=Path)
    p.add_argument("--root",required=True,type=Path)
    p.add_argument("--real-target",type=Path)
    p.add_argument("--legacy",type=Path)
    p.add_argument("--config",type=Path)
    p.add_argument("--seed",type=int,default=73)
    p.add_argument("--device",default="cuda:2")
    p.add_argument("--joint-only",action="store_true")
    a=p.parse_args()
    settings=json.loads(a.config.read_text()) if a.config else {}
    cfg=Config(**(settings | {"seed":a.seed}))
    data=Network(a.network)
    result=fit_run(data,Network(a.real_target) if a.real_target else data,a.root,cfg,a.device,not a.joint_only,a.legacy)
    print(result,flush=True)


if __name__=="__main__":
    main()
