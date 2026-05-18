#!/usr/bin/env zsh
set -euo pipefail

ROOT="/Users/kw/Documents/Playground/vhee_topas"
cd "$ROOT"

export MPLCONFIGDIR="/Users/kw/Documents/Playground/tmp_mpl"
export XDG_CACHE_HOME="/Users/kw/Documents/Playground/tmp_cache"
export PYTHONPYCACHEPREFIX="/Users/kw/Documents/Playground/tmp_pycache"

python3.12 -u scripts/run_phase27_publication_package.py \
  --phase25-root "$ROOT/runs/linac_6mv_headneck_detailed_voxel_lattice_sfrt_phase25_safe_core_plan_library_1e5" \
  "$@"
