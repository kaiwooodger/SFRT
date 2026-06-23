#!/usr/bin/env zsh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT"

export MPLCONFIGDIR="/Users/kw/Documents/Playground/tmp_mpl"
export XDG_CACHE_HOME="/Users/kw/Documents/Playground/tmp_cache"
export PYTHONPYCACHEPREFIX="/Users/kw/Documents/Playground/tmp_pycache"

python3.12 -u scripts/run_phase28_yang2022_sinonasal_benchmark.py "$@"
