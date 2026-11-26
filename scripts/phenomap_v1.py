"""Independent PhenoMap V1 entry point (from a checkout or editable installation)."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from phenoniche.v1.evaluate import explain, heldout
from phenoniche.v1.model import Settings
from phenoniche.v1.storage import Network, digest, key, lock, prepare
from phenoniche.v1.workflow import checkpoint, execute, predict, runtime


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action",required=True)
    for action in ("prepare","train","infer","score","explain"):
        p = sub.add_parser(action)
        if action != "explain":
            p.add_argument("--config",type=Path,required=True)
        if action != "prepare":
            p.add_argument("--network",type=Path,required=True)
        if action in ("train","infer","score"):
            p.add_argument("--device",default="cuda:2")
        if action in ("infer","explain","score"):
            p.add_argument("--run",type=Path,required=True)
        if action == "prepare":
            p.add_argument("--null-seed",type=int)
            p.add_argument("--reference-run",type=Path)
        if action == "train":
            p.add_argument("--mode",choices=["composition","communication","joint"])
            p.add_argument("--seed",type=int)
        if action == "explain":
            p.add_argument("--cell",required=True)
            p.add_argument("--competitor",type=int)
            p.add_argument("--output",required=True,type=Path)
    args = parser.parse_args()
    if args.action == "explain":
        actual = key(json.loads((args.network/"manifest.json").read_text()))
        if actual != json.loads((args.run/"manifest.json").read_text())["network_key"]:
            parser.error("This fit uses a different observed network")
        result,table = explain(args.run,args.network,args.cell,args.competitor)
        args.output.mkdir(parents=True,exist_ok=False)
        (args.output/"assignment.json").write_text(json.dumps(result,indent=2))
        table.to_csv(args.output/"edges.tsv.gz",sep="\t",index=False)
        return
    cfg = json.loads(args.config.read_text())
    if args.action == "prepare":
        path = prepare(cfg["source"],cfg["atlas"],cfg["root"],**cfg["network"],null_seed=args.null_seed,
                       reference=args.reference_run/"training_schema.npz" if args.reference_run else None)
    elif args.action == "train":
        settings = cfg["model"].copy()
        for k in ("seed","mode"):
            if getattr(args,k) is not None:
                settings[k] = getattr(args,k)
        path = execute(args.network,cfg["root"],Settings(**settings),args.device)
    elif args.action == "score":
        import numpy as np
        meta = dict(checkpoint=digest(args.run/"model.pt"),target=digest(args.network/"manifest.json"),
                    **runtime(args.device))
        path = Path(cfg["root"])/("score-"+key(meta))
        path.mkdir(parents=True,exist_ok=True)
        with lock(path):
            if not (path/"metrics.json").is_file():
                with np.load(args.run/"training_schema.npz") as ref, np.load(args.network/"schema.npz") as z:
                    for name in ("lr","names","genes","scales","radii","group","window","train","test","xyz","ids","samples"):
                        if not np.array_equal(z[name],ref[name]):
                            raise ValueError(f"Paired held-out scoring requires the same {name}")
                metrics = heldout(checkpoint(args.run,args.device)[0],Network(args.network),args.device)
                if not metrics:
                    raise ValueError("No held-out centers")
                (path/"manifest.json").write_text(json.dumps(meta,indent=2))
                tmp = path/"metrics.partial"
                tmp.write_text(json.dumps(metrics,indent=2))
                tmp.replace(path/"metrics.json")
    else:
        path = predict(args.run,args.network,cfg["root"],args.device)
    print(path,flush=True)


if __name__ == "__main__":
    main()
