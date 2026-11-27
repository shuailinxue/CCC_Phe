#!/usr/bin/env bash
set -euo pipefail
export PYTHONPATH=src OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
p=${PYTHON:-python}
base=/home/nas3/biod/xueshuailin/CCC_Phe
jobs=()
for d in pt73 ovca; do
  if [[ "$d" == pt73 ]]; then
    net="$base/results/spatialfusion_ovca/phenomap_v1/network-0bdefdf7e93932cc45c76306"
    root="$base/results/nsclc_merfish_chen2024/phenomap_network"
    source="$base/data/nsclc_merfish_chen2024/processed/Pt73.h5ad"
  else
    net="$base/results/spatialfusion_ovca/phenomap_v1/network-46c9e809ec3811e073feadd3"
    root="$base/results/spatialfusion_ovca/phenomap_network"
    source=/data1/xueshuailin/CCC_Phe/workspace/data/spatialfusion_ovca/processed_OVCA.h5ad
  fi
  readarray -t paths < <("$p" - "$root" <<'PY'
from pathlib import Path
import json,sys
fits={}
for f in Path(sys.argv[1]).glob('*/manifest.json'):
 if not (f.parent/'complete').is_file(): continue
 if f.parent.name.startswith('legacy-matched-'):
  fits['legacy']=f.parent;continue
 m=json.loads(f.read_text())
 name='null' if m['input']['null_seed'] is not None else f"seed{m['settings']['seed']}"
 fits[name]=f.parent
for name in ('seed73','seed74','null','legacy'): print(fits[name])
PY
)
  "$p" experiments/report_network_v1.py --dataset "$d" --network "$net" --source "$source" \
    --seed73 "${paths[0]}" --seed74 "${paths[1]}" --null "${paths[2]}" --legacy "${paths[3]}" &
  jobs+=("$!")
done
for pid in "${jobs[@]}"; do wait "$pid"; done
