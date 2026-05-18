#!/bin/zsh
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT"

RUN_CMD_PATTERN="scripts/run_strict_whitmore_quick_250.py --histories 100000 --threads 8 --seed 11 --run-root runs/strict_whitmore_quick_250"

# Wait for the currently running strict script to finish.
while pgrep -f "$RUN_CMD_PATTERN" >/dev/null 2>&1; do
  sleep 30
done

# If trend summary already exists, leave it untouched.
if [[ -f runs/strict_whitmore_quick_250/trend_check_250_summary.json ]]; then
  echo "Trend summary already exists; nothing to do."
  exit 0
fi

# Re-run in skip-existing mode to force analysis/trend generation from finished TOPAS outputs.
/opt/anaconda3/bin/python scripts/run_strict_whitmore_quick_250.py \
  --histories 100000 \
  --threads 8 \
  --seed 11 \
  --run-root runs/strict_whitmore_quick_250 \
  --skip-existing
