"""Build full-window observations with the fitted sampled panel normalization."""
import argparse
from pathlib import Path
from phenoniche.v1.storage import prepare

p=argparse.ArgumentParser()
for key in ("source","reference","root","atlas"):
    p.add_argument("--"+key,required=True,type=Path)
p.add_argument("--types",required=True)
p.add_argument("--specimen")
p.add_argument("--workers",type=int,default=24)
a=p.parse_args()
print(prepare(a.source,a.atlas,a.root,coords="spatial",types=a.types,specimen=a.specimen,
              workers=a.workers,reference=a.reference/"schema.npz"),flush=True)
