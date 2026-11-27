"""Fixed-global inference for a selected V1 fit with compatible network schema."""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
import numpy as np
from phenoniche.v1.polish_io import exported_niche
from phenoniche.v1.polish import Options
from phenoniche.v1.polish_fit import load_dictionary
from phenoniche.v1.probabilistic import Config
from phenoniche.v1.polish_stream import stream
from phenoniche.v1.storage import digest,key,lock
from phenoniche.v1.workflow import runtime


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ("selected","network","reference","output"):
        p.add_argument("--"+k,required=True,type=Path)
    p.add_argument("--device",default="cuda:2")
    args=p.parse_args()
    sel=args.selected
    metric=json.loads((sel/"final73/metrics.json").read_text())
    fit=sel/"final73/niches.npz"
    base_dir=Path(metric["dictionary_fit"])
    manifest=json.loads((base_dir/"manifest.json").read_text())
    checkpoint=base_dir/"dictionary.pt"
    if not checkpoint.is_file():
        # A is a preserved historical checkpoint referenced by its full hash.
        paths=sel.parent.parent/"phenomap_network"
        matches=[f for f in paths.glob("*/dictionary.pt") if digest(f)==manifest["reference"]]
        if len(matches)!=1:
            raise ValueError("Could not uniquely resolve the preserved dictionary")
        checkpoint=matches[0]
    with np.load(args.reference/"schema.npz") as x,np.load(args.network/"schema.npz") as y:
        for name in ("names","genes","lr","group","radii","signaling","window","scales"):
            if not np.array_equal(x[name],y[name]):
                raise ValueError(f"Incompatible predictive schema: {name}")
    selected=json.loads((sel/"selection.json").read_text())["niche"]
    niche=exported_niche(sel/"final73","joint",selected["temperature"],args.device)
    meta=dict(checkpoint=digest(checkpoint),niche=digest(fit),network=digest(args.network/"manifest.json"),
              implementation=digest(__file__),stream=digest(sys.modules[stream.__module__].__file__),
              runtime=runtime(args.device))
    dest=args.output/("inference-"+key(meta))
    dest.mkdir(parents=True,exist_ok=True)
    with lock(dest):
        if not (dest/"complete").is_file():
            (dest/"manifest.json").write_text(json.dumps(meta,indent=2))
            cfg=replace(Config(**manifest["cfg"]),k=len(niche.pi),batch=64)
            model=load_dictionary(checkpoint,cfg,Options(**metric["options"]),args.device)
            diag=stream(args.network,dest,sel/"final73",model,niche,cfg,args.device)
            (dest/"metrics.json").write_text(json.dumps(diag,indent=2))
            (dest/"complete").touch()
    print(dest,flush=True)


if __name__=="__main__":
    main()
