#!/usr/bin/env bash
set -euo pipefail
export PYTHONPATH=src OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
p=${PYTHON:-python}
base=/home/nas3/biod/xueshuailin/CCC_Phe
dataset=$1
if [[ "$dataset" == pt73 ]]; then
    result="$base/results/nsclc_merfish_chen2024/phenomap_polish"
    net="$base/results/spatialfusion_ovca/phenomap_v1/network-0bdefdf7e93932cc45c76306"
    source="$base/data/nsclc_merfish_chen2024/processed/Pt73.h5ad"
elif [[ "$dataset" == ovca ]]; then
    result="$base/results/spatialfusion_ovca/phenomap_polish"
    net="$base/results/spatialfusion_ovca/phenomap_v1/network-46c9e809ec3811e073feadd3"
    source=/data1/xueshuailin/CCC_Phe/workspace/data/spatialfusion_ovca/processed_OVCA.h5ad
else
    exit 2
fi
shopt -s nullglob
while :; do
    results=("$result"/selected-*/complete)
    if (( ${#results[@]} == 1 )); then break; fi
    if (( ${#results[@]} > 1 )); then
        echo 'Multiple completed selections. Pass an explicit selection to report_polish.py.' >&2
        exit 2
    fi
    sleep 5
done
"$p" experiments/report_polish.py --dataset "$dataset" --selected "${results[0]%/complete}" --network "$net" --source "$source"
