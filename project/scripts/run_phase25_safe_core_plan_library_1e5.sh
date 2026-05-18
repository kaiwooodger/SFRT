#!/bin/zsh
set -euo pipefail

ROOT="/Users/kw/Documents/Playground/vhee_topas"
cd "$ROOT"

env MPLCONFIGDIR=/Users/kw/Documents/Playground/tmp_mpl \
XDG_CACHE_HOME=/Users/kw/Documents/Playground/tmp_cache \
PYTHONPYCACHEPREFIX=/Users/kw/Documents/Playground/tmp_pycache \
python3.12 scripts/run_phase25_bio_risk_analysis.py \
  --run-root "$ROOT/runs/linac_6mv_headneck_detailed_voxel_lattice_sfrt_phase25_safe_core_plan_library_1e5" \
  --baseline-run-root "$ROOT/runs/linac_6mv_headneck_detailed_voxel_lattice_sfrt_directplan" \
  --plan-library-json "$ROOT/data/phase25_safe_core_plan_library.json" \
  --histories 100000 \
  --threads 8 \
  --seed 33 \
  --fractions 5 \
  --plan-count 5 \
  --num-spots 2 \
  --spot-radius-mm 6.0 \
  "$@"
