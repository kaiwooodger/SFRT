#!/bin/zsh
set -euo pipefail

ROOT="/Users/kw/Documents/Playground/vhee_topas"

cd "$ROOT"

export MPLCONFIGDIR="/Users/kw/Documents/Playground/tmp_mpl"
export XDG_CACHE_HOME="/Users/kw/Documents/Playground/tmp_cache"
export PYTHONPYCACHEPREFIX="/Users/kw/Documents/Playground/tmp_pycache"

python3.12 scripts/run_phase20_randomized_protocol_mix_optimization.py \
  --run-root "$ROOT/runs/linac_6mv_headneck_detailed_voxel_lattice_sfrt_phase20_randomized_protocol_mix_1e5" \
  --baseline-run-root "$ROOT/runs/linac_6mv_headneck_detailed_voxel_lattice_sfrt_directplan" \
  --fractions 5 \
  --histories 100000 \
  --threads 8 \
  --candidate-plan-limit 8 \
  --pde-steps 400 \
  --allow-infeasible-fallback \
  "$@"
