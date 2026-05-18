#!/bin/zsh
set -euo pipefail

ROOT="/Users/kw/Documents/Playground/vhee_topas"
cd "$ROOT"

env MPLCONFIGDIR=/Users/kw/Documents/Playground/tmp_mpl \
XDG_CACHE_HOME=/Users/kw/Documents/Playground/tmp_cache \
PYTHONPYCACHEPREFIX=/Users/kw/Documents/Playground/tmp_pycache \
python3.12 scripts/run_phase25a_assay_outputs.py \
  --phase25-run-root "$ROOT/runs/linac_6mv_headneck_detailed_voxel_lattice_sfrt_phase25_bio_risk_analysis_1e5" \
  "$@"
