#!/bin/zsh
set -euo pipefail

ROOT="/Users/kw/Documents/Playground/vhee_topas"
SCRIPT="$ROOT/scripts/run_phase17_fraction_aware_bio_optimization.py"
OUT_ROOT="$ROOT/runs/linac_6mv_headneck_detailed_voxel_lattice_sfrt_phase17_alternating_two_pattern_1e5"

mkdir -p /Users/kw/Documents/Playground/tmp_mpl /Users/kw/Documents/Playground/tmp_cache /Users/kw/Documents/Playground/tmp_pycache

export MPLCONFIGDIR=/Users/kw/Documents/Playground/tmp_mpl
export XDG_CACHE_HOME=/Users/kw/Documents/Playground/tmp_cache
export PYTHONPYCACHEPREFIX=/Users/kw/Documents/Playground/tmp_pycache

cd "$ROOT"

python3.12 "$SCRIPT" \
  --run-root "$OUT_ROOT" \
  --fractions 5 \
  --histories 100000 \
  --threads 8 \
  --candidate-top-k-centers 18 \
  --candidate-plan-limit 5 \
  --pde-steps 400 \
  --allow-infeasible-fallback \
  --course-strategy alternating_two_pattern \
  --reuse-baseline-dose \
  "$@"
