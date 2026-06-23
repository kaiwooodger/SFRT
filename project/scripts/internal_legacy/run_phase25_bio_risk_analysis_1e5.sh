#!/bin/zsh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT"

env MPLCONFIGDIR=/Users/kw/Documents/Playground/tmp_mpl \
XDG_CACHE_HOME=/Users/kw/Documents/Playground/tmp_cache \
PYTHONPYCACHEPREFIX=/Users/kw/Documents/Playground/tmp_pycache \
python3.12 scripts/run_phase25_bio_risk_analysis.py \
  --run-root "$ROOT/runs/linac_6mv_headneck_detailed_voxel_lattice_sfrt_phase25_bio_risk_analysis_1e5" \
  --baseline-run-root "$ROOT/runs/linac_6mv_headneck_detailed_voxel_lattice_sfrt_directplan" \
  --histories 100000 \
  --threads 8 \
  --seed 33 \
  --fractions 5 \
  --plan-count 5 \
  --num-spots 4 \
  "$@"
