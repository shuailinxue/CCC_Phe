#!/usr/bin/env bash
set -euo pipefail
export PYTHONPATH=src OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
p=$HOME/miniconda3/envs/cccphe/bin/python
root=/home/nas3/biod/xueshuailin/CCC_Phe/results/spatialfusion_ovca/phenomap_v1
net() {
  local name=$1
  until hit=$("$p" - "$root" "$name" <<'PY'
import json,sys
from pathlib import Path
root,name=sys.argv[1:]
for f in Path(root).glob('network-*/manifest.json'):
 m=json.loads(f.read_text())
 n='pt73' if m['type_column']=='type_lvl1' else ('null' if m['null_seed'] else 'joint')
 if n==name:
  print(f.parent); break
else: raise SystemExit(1)
PY
); do sleep 5; done
  printf '%s\n' "$hit"
}
actual=$(net joint)
null=$(net null)
pt=$(net pt73)
for n in "$actual" "$pt"; do "$p" experiments/ovca_quality.py "$n"; done
for mode in joint composition communication; do
  "$p" -u scripts/phenomap_v1.py train --config configs/ovca_v1.json --network "$actual" --mode "$mode" --device cuda:2 &
done
"$p" -u scripts/phenomap_v1.py train --config configs/ovca_v1.json --network "$null" --mode joint --device cuda:2 &
wait
