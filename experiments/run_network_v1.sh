#!/usr/bin/env bash
set -euo pipefail
dataset=$1
condition=$2
seed=$3
device=$4
base=/home/nas3/biod/xueshuailin/CCC_Phe/results
cache="$base/spatialfusion_ovca/phenomap_v1"
extra=()
if [[ "$dataset" == pt73 ]]; then
  root="$base/nsclc_merfish_chen2024/phenomap_network"
  real="$cache/network-0bdefdf7e93932cc45c76306"
else
  root="$base/spatialfusion_ovca/phenomap_network"
  real="$cache/network-46c9e809ec3811e073feadd3"
fi
net="$real"
if [[ "$condition" == null ]]; then
  if [[ "$dataset" == pt73 ]]; then
    net=$("$PYTHON" - "$cache" <<'PY'
import json,sys
from pathlib import Path
for f in Path(sys.argv[1]).glob('network-*/manifest.json'):
    m=json.loads(f.read_text())
    if m['null_seed'] is not None and m['type_column']=='type_lvl1':
        print(f.parent);break
else: raise SystemExit('Matched sampled Pt73 null cache is missing')
PY
)
  else
    net="$cache/network-39f40f0f2eeb390a514488b6"
  fi
  extra=(--real-target "$real" --joint-only)
elif [[ "$seed" != 73 ]]; then
  extra=(--joint-only)
fi
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
exec "$PYTHON" -u scripts/phenomap_network.py --network "$net" --root "$root" --seed "$seed" --device "$device" --config configs/network_v1_sampled.json "${extra[@]}"
