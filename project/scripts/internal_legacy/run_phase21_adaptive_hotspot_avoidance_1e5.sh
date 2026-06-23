#!/bin/zsh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
OUT_ROOT="$ROOT/runs/linac_6mv_headneck_detailed_voxel_lattice_sfrt_phase21_adaptive_hotspot_avoidance_1e5"

mkdir -p /Users/kw/Documents/Playground/tmp_mpl /Users/kw/Documents/Playground/tmp_cache /Users/kw/Documents/Playground/tmp_pycache

env \
  MPLCONFIGDIR=/Users/kw/Documents/Playground/tmp_mpl \
  XDG_CACHE_HOME=/Users/kw/Documents/Playground/tmp_cache \
  PYTHONPYCACHEPREFIX=/Users/kw/Documents/Playground/tmp_pycache \
  python3.12 "$ROOT/scripts/run_phase21_adaptive_hotspot_avoidance_optimization.py" \
  --run-root "$OUT_ROOT" \
  --baseline-run-root "$ROOT/runs/linac_6mv_headneck_detailed_voxel_lattice_sfrt_directplan" \
  --fractions 5 \
  --histories 100000 \
  --threads 8 \
  --candidate-top-k-centers 18 \
  --candidate-plan-limit 8 \
  --pde-steps 400 \
  --allow-infeasible-fallback \
  "$@"
