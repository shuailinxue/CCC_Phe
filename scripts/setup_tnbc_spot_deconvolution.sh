#!/usr/bin/env bash
set -euo pipefail
# Keep newer scVI isolated from the existing cccphe/scgpt environment.
spot_env=/home/nas3/biod/xueshuailin/CCC_Phe/env/envs/tnbc_cell2location
spot_cache=/home/nas3/biod/xueshuailin/CCC_Phe/workspace/tmp/pip_cache
mkdir -p "$spot_cache"
/home/xueshuailin/miniconda3/envs/cccphe/bin/python -m venv --system-site-packages "$spot_env"
PIP_CACHE_DIR="$spot_cache" "$spot_env/bin/pip" install cell2location==0.1.5 scvi-tools==1.4.2
PYTHONDONTWRITEBYTECODE=1 "$spot_env/bin/python" -c 'import cell2location, scvi; print(cell2location.__version__, scvi.__version__)'
