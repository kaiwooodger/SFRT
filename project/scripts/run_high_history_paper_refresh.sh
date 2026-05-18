#!/usr/bin/env zsh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python3.12}"
OUT_ROOT="${REPO_ROOT}/runs/paper_refresh_hihist_10x"
HISTORIES_BASE=120000
HISTORIES_SPOT=240000
THREADS=4
SEED=33
REPEAT_SEEDS_CSV="11,22,33"
HISTORY_SCALES_CSV="0.5,1.0,2.0"
DOSE_SMOOTHING_MM=6.0
SKIP_EXISTING=0

function usage() {
  cat <<'EOF'
Usage:
  ./scripts/run_high_history_paper_refresh.sh [options]

Options:
  --out-root PATH             Refresh root for all new outputs.
  --histories-base INT        Base-field TOPAS histories per run. Default: 120000
  --histories-spot INT        Spot/vertex TOPAS histories per run. Default: 240000
  --threads INT               TOPAS threads. Default: 4
  --seed INT                  Baseline TOPAS seed. Default: 33
  --repeat-seeds CSV          Seeds for the repeated subset. Default: 11,22,33
  --history-scales CSV        History scale factors for the repeated subset. Default: 0.5,1.0,2.0
  --dose-smoothing-mm FLOAT   Dose smoothing kernel in mm. Default: 6.0
  --resume                    Reuse finished outputs where supported.
  --python BIN                Python executable. Default: python3.12
  --help                      Show this message.

This launcher creates a new high-history paper-refresh tree containing:
  - higher-history benchmark transport
  - higher-history 10-case TOPAS cohort
  - downstream biology reinterpretation
  - repeated uncertainty subset
  - first falsification cohort + uncertainty overlay
  - stronger falsification cohort + uncertainty overlay
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --out-root)
      OUT_ROOT="$2"
      shift 2
      ;;
    --histories-base)
      HISTORIES_BASE="$2"
      shift 2
      ;;
    --histories-spot)
      HISTORIES_SPOT="$2"
      shift 2
      ;;
    --threads)
      THREADS="$2"
      shift 2
      ;;
    --seed)
      SEED="$2"
      shift 2
      ;;
    --repeat-seeds)
      REPEAT_SEEDS_CSV="$2"
      shift 2
      ;;
    --history-scales)
      HISTORY_SCALES_CSV="$2"
      shift 2
      ;;
    --dose-smoothing-mm)
      DOSE_SMOOTHING_MM="$2"
      shift 2
      ;;
    --resume)
      SKIP_EXISTING=1
      shift
      ;;
    --python)
      PYTHON_BIN="$2"
      shift 2
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "Python executable not found: ${PYTHON_BIN}" >&2
  exit 1
fi

typeset -a REPEAT_SEEDS
typeset -a HISTORY_SCALES
REPEAT_SEEDS=("${(@s:,:)REPEAT_SEEDS_CSV}")
HISTORY_SCALES=("${(@s:,:)HISTORY_SCALES_CSV}")

OUT_ROOT="$(OUT_ROOT_INPUT="${OUT_ROOT}" "${PYTHON_BIN}" - <<PY
import os
from pathlib import Path
print(Path(os.environ["OUT_ROOT_INPUT"]).expanduser().resolve())
PY
)"

PHASE30_ROOT="${OUT_ROOT}/phase30_phase28_topas_true_lattice_delivery_hihist"
PHASE33_ROOT="${OUT_ROOT}/phase33_phase32_topas_cohort_hihist"
PHASE34_ROOT="${PHASE33_ROOT}/phase34_bio_cohort"
PHASE35_ROOT="${OUT_ROOT}/phase35_subset_repeat_uncertainty_hihist"
PHASE36A_ROOT="${OUT_ROOT}/phase36a_vessel_falsification_cohort_hihist"
PHASE36B_ROOT="${OUT_ROOT}/phase36b_vessel_falsification_uncertainty_hihist"
PHASE37A_ROOT="${OUT_ROOT}/phase37a_vessel_falsification_cohort_hihist"
PHASE37B_ROOT="${OUT_ROOT}/phase37b_vessel_falsification_uncertainty_hihist"

mkdir -p "${OUT_ROOT}"

function timestamp() {
  date '+%Y-%m-%d %H:%M:%S'
}

function run_step() {
  local label="$1"
  shift
  echo ""
  echo "[$(timestamp)] ${label}"
  echo "[$(timestamp)] Command: $*"
  "$@"
}

function write_manifest() {
  cat > "${OUT_ROOT}/high_history_refresh_manifest.md" <<EOF
# High-History Paper Refresh

- Refresh root: \`${OUT_ROOT}\`
- Generated: \`$(timestamp)\`
- Python: \`${PYTHON_BIN}\`
- Base histories: \`${HISTORIES_BASE}\`
- Spot histories: \`${HISTORIES_SPOT}\`
- Threads: \`${THREADS}\`
- Seed: \`${SEED}\`
- Repeat seeds: \`${REPEAT_SEEDS_CSV}\`
- History scales: \`${HISTORY_SCALES_CSV}\`
- Dose smoothing (mm): \`${DOSE_SMOOTHING_MM}\`
- Resume mode: \`${SKIP_EXISTING}\`

## Output roots

- Benchmark transport: \`${PHASE30_ROOT}\`
- Physical cohort: \`${PHASE33_ROOT}\`
- Biology cohort: \`${PHASE34_ROOT}\`
- Repeated uncertainty subset: \`${PHASE35_ROOT}\`
- Falsification cohort A: \`${PHASE36A_ROOT}\`
- Falsification uncertainty A: \`${PHASE36B_ROOT}\`
- Falsification cohort B: \`${PHASE37A_ROOT}\`
- Falsification uncertainty B: \`${PHASE37B_ROOT}\`
EOF
}

cd "${REPO_ROOT}"

PHASE33_SKIP=()
PHASE35_SKIP=()
if [[ "${SKIP_EXISTING}" -eq 1 ]]; then
  PHASE33_SKIP=(--skip-existing)
  PHASE35_SKIP=(--skip-existing)
fi

run_step \
  "Benchmark transport refresh" \
  "${PYTHON_BIN}" "${SCRIPT_DIR}/run_phase30_phase28_topas_true_lattice_delivery.py" \
  --out-root "${PHASE30_ROOT}" \
  --histories-base "${HISTORIES_BASE}" \
  --histories-spot "${HISTORIES_SPOT}" \
  --threads "${THREADS}" \
  --seed "${SEED}"

run_step \
  "10-case higher-history TOPAS cohort" \
  "${PYTHON_BIN}" "${SCRIPT_DIR}/run_phase33_phase32_topas_cohort.py" \
  --out-root "${PHASE33_ROOT}" \
  --histories-base "${HISTORIES_BASE}" \
  --histories-spot "${HISTORIES_SPOT}" \
  --threads "${THREADS}" \
  --seed "${SEED}" \
  --dose-smoothing-mm "${DOSE_SMOOTHING_MM}" \
  "${PHASE33_SKIP[@]}"

run_step \
  "Biology reinterpretation on higher-history cohort" \
  "${PYTHON_BIN}" "${SCRIPT_DIR}/run_phase34_phase32_bio_cohort.py" \
  --phase33-root "${PHASE33_ROOT}" \
  --out-root "${PHASE34_ROOT}"

run_step \
  "Repeated uncertainty subset refresh" \
  "${PYTHON_BIN}" "${SCRIPT_DIR}/run_phase35_subset_repeat_uncertainty.py" \
  --baseline-phase33-root "${PHASE33_ROOT}" \
  --baseline-phase34-root "${PHASE34_ROOT}" \
  --out-root "${PHASE35_ROOT}" \
  --histories-base "${HISTORIES_BASE}" \
  --histories-spot "${HISTORIES_SPOT}" \
  --threads "${THREADS}" \
  --dose-smoothing-mm "${DOSE_SMOOTHING_MM}" \
  --seeds "${REPEAT_SEEDS[@]}" \
  --history-scales "${HISTORY_SCALES[@]}" \
  "${PHASE35_SKIP[@]}"

run_step \
  "Initial falsification cohort refresh" \
  "${PYTHON_BIN}" "${SCRIPT_DIR}/run_phase36a_vessel_falsification_cohort.py" \
  --phase33-root "${PHASE33_ROOT}" \
  --phase34-root "${PHASE34_ROOT}" \
  --out-root "${PHASE36A_ROOT}"

run_step \
  "Initial falsification uncertainty refresh" \
  "${PYTHON_BIN}" "${SCRIPT_DIR}/run_phase36b_vessel_falsification_uncertainty.py" \
  --phase35-root "${PHASE35_ROOT}" \
  --phase36a-root "${PHASE36A_ROOT}" \
  --out-root "${PHASE36B_ROOT}"

run_step \
  "Stronger falsification cohort refresh" \
  "${PYTHON_BIN}" "${SCRIPT_DIR}/run_phase37a_vessel_falsification_cohort.py" \
  --phase33-root "${PHASE33_ROOT}" \
  --phase34-root "${PHASE34_ROOT}" \
  --out-root "${PHASE37A_ROOT}"

run_step \
  "Stronger falsification uncertainty refresh" \
  "${PYTHON_BIN}" "${SCRIPT_DIR}/run_phase37b_vessel_falsification_uncertainty.py" \
  --phase35-root "${PHASE35_ROOT}" \
  --phase37a-root "${PHASE37A_ROOT}" \
  --out-root "${PHASE37B_ROOT}"

write_manifest

echo ""
echo "=== HIGH-HISTORY PAPER REFRESH COMPLETE ==="
echo "Refresh root: ${OUT_ROOT}"
echo "Manifest: ${OUT_ROOT}/high_history_refresh_manifest.md"
echo "Benchmark: ${PHASE30_ROOT}"
echo "Cohort: ${PHASE33_ROOT}"
echo "Biology: ${PHASE34_ROOT}"
echo "Uncertainty: ${PHASE35_ROOT}"
echo "Falsification A: ${PHASE36A_ROOT}"
echo "Falsification A uncertainty: ${PHASE36B_ROOT}"
echo "Falsification B: ${PHASE37A_ROOT}"
echo "Falsification B uncertainty: ${PHASE37B_ROOT}"
