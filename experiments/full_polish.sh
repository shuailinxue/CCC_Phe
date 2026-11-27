#!/usr/bin/env bash
set -euo pipefail
export PYTHONPATH="src${PYTHONPATH:+:$PYTHONPATH}" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
p=${PYTHON:-python}
base=/home/nas3/biod/xueshuailin/CCC_Phe/results
case "$1" in
  pt73)
    root="$base/nsclc_merfish_chen2024/phenomap_polish"
    ref="$base/spatialfusion_ovca/phenomap_v1/network-0bdefdf7e93932cc45c76306"
    ;;
  ovca)
    root="$base/spatialfusion_ovca/phenomap_polish"
    ref="$base/spatialfusion_ovca/phenomap_v1/network-46c9e809ec3811e073feadd3"
    ;;
  *) exit 2 ;;
esac
shopt -s nullglob
while :; do
  s=("$root"/selected-*/complete)
  n=("$root"/full_window_cache/network-*/manifest.json)
  if (( ${#s[@]} > 1 || ${#n[@]} > 1 )); then echo 'Pass explicit paths for this selection'; exit 2; fi
  if (( ${#s[@]} == 1 && ${#n[@]} == 1 )); then break; fi
  sleep 15
done
"$p" -u scripts/predict_polished_v1.py --selected "${s[0]%/complete}" --reference "$ref" \
    --network "${n[0]%/manifest.json}" --output "$root" --device "${2:-cuda:2}"
