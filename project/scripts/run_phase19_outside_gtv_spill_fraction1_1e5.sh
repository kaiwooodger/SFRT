#!/bin/zsh
set -euo pipefail

ROOT="/Users/kw/Documents/Playground/vhee_topas"

cd "$ROOT"

export MPLCONFIGDIR="/Users/kw/Documents/Playground/tmp_mpl"
export XDG_CACHE_HOME="/Users/kw/Documents/Playground/tmp_cache"
export PYTHONPYCACHEPREFIX="/Users/kw/Documents/Playground/tmp_pycache"

python3.12 scripts/run_phase19_outside_gtv_spill_optimization.py \
  --run-root "$ROOT/runs/linac_6mv_headneck_detailed_voxel_lattice_sfrt_phase19_fraction1_1e5" \
  --baseline-run-root "$ROOT/runs/linac_6mv_headneck_detailed_voxel_lattice_sfrt_directplan" \
  --fractions 1 \
  --histories 100000 \
  --threads 8 \
  --candidate-plan-limit 6 \
  --pde-steps 400 \
  --allow-infeasible-fallback \
  "$@"
