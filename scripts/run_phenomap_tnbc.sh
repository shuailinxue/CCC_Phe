#!/usr/bin/env bash
set -euo pipefail
export PHENOMAP_GPU="${PHENOMAP_GPU:-2}"
export CUDA_VISIBLE_DEVICES="$PHENOMAP_GPU"
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export MPLCONFIGDIR=/home/nas3/biod/xueshuailin/CCC_Phe/workspace/tmp/phenomap_matplotlib
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /home/xueshuailin/CCC_Phe
exec /home/xueshuailin/miniconda3/envs/cell2st/bin/python -u scripts/run_phenomap_tnbc.py "$@"
