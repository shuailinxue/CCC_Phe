#!/usr/bin/env bash
set -euo pipefail
export PYTHONPATH=src OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
p=$HOME/miniconda3/envs/cccphe/bin/python
root=/home/nas3/biod/xueshuailin/CCC_Phe/results/spatialfusion_ovca/phenomap_v1
actual="$root/network-46c9e809ec3811e073feadd3"
readarray -t runs < <("$p" - "$root" <<'PY'
import json,sys
from pathlib import Path
r={}
for f in Path(sys.argv[1]).glob('fit-*/manifest.json'):
 if not (f.parent/'complete').is_file(): continue
 m=json.loads(f.read_text())
 mode='null' if m['network_key']=='39f40f0f2eeb390a514488b6' else m['settings']['mode']
 r[mode]=f.parent
for mode in ('joint','composition','communication','null'): print(r[mode])
PY
)
joint=${runs[0]}
comp=${runs[1]}
comm=${runs[2]}
null=${runs[3]}
"$p" -u scripts/diagnose_pt73_v1.py --joint "$joint" --communication "$comm" --network "$actual" --device cuda:2 &
diagnosis=$!
"$p" -u scripts/phenomap_v1.py score --config configs/ovca_v1.json --network "$actual" --run "$null" --device cuda:2 &
paired=$!
wait "$diagnosis"
wait "$paired"
"$p" experiments/ovca_report.py --joint "$joint" --composition "$comp" --communication "$comm" --null "$null" --diagnosis "$root"/diagnosis-* --paired-null "$root"/score-*
