#!/bin/zsh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$ROOT"

export MPLCONFIGDIR="/Users/kw/Documents/Playground/tmp_mpl"
export XDG_CACHE_HOME="/Users/kw/Documents/Playground/tmp_cache"
export PYTHONPYCACHEPREFIX="/Users/kw/Documents/Playground/tmp_pycache"

python3.12 scripts/run_phase17_fraction_aware_bio_optimization.py \
  --run-root "$ROOT/runs/linac_6mv_headneck_detailed_voxel_lattice_sfrt_phase17_clinical_gtv_core_1e5" \
  --baseline-run-root "$ROOT/runs/linac_6mv_headneck_detailed_voxel_lattice_sfrt_directplan" \
  --course-strategy clinical_gtv_core \
  --fractions 5 \
  --histories 100000 \
  --threads 8 \
  --spot-radius-mm 7.5 \
  --candidate-plan-limit 6 \
  --pde-steps 400 \
  --allow-infeasible-fallback \
  "$@"
