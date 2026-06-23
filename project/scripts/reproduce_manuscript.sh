#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

MODE="public"
SKIP_FIGURES=0
PYTHON_BIN="${PYTHON_BIN:-python3}"
TOPAS_BIN="${TOPAS_BIN:-}"
G4_DATA_DIR="${G4_DATA_DIR:-}"

usage() {
  cat <<'USAGE'
Usage:
  bash project/scripts/reproduce_manuscript.sh [--mode public|full] [--skip-figures]

Modes:
  public  Validate the bundled manuscript-facing results and report the frozen
          headline numbers. No TOPAS installation is required.

  full    Run the preserved full workflow through the clean public wrappers.
          Requires TOPAS_BIN and G4_DATA_DIR.
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode)
      MODE="${2:-}"
      shift 2
      ;;
    --skip-figures)
      SKIP_FIGURES=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

run_step() {
  echo
  echo "==> $*"
  "$@"
}

cd "${REPO_ROOT}"

case "${MODE}" in
  public)
    run_step "${PYTHON_BIN}" "${REPO_ROOT}/project/scripts/verify_public_bundle.py"
    ;;
  full)
    if [[ -z "${TOPAS_BIN}" || -z "${G4_DATA_DIR}" ]]; then
      echo "Full mode requires TOPAS_BIN and G4_DATA_DIR." >&2
      exit 1
    fi
    run_step "${PYTHON_BIN}" "${REPO_ROOT}/project/scripts/generate_synthetic_cohort.py"
    run_step "${PYTHON_BIN}" "${REPO_ROOT}/project/scripts/run_topas_cohort.py" --topas-bin "${TOPAS_BIN}" --g4-data-dir "${G4_DATA_DIR}"
    run_step "${PYTHON_BIN}" "${REPO_ROOT}/project/scripts/apply_biological_reinterpretation.py"
    run_step "${PYTHON_BIN}" "${REPO_ROOT}/project/scripts/run_repeat_uncertainty.py"
    run_step "${PYTHON_BIN}" "${REPO_ROOT}/project/scripts/run_sink_falsification_cohort.py"
    run_step "${PYTHON_BIN}" "${REPO_ROOT}/project/scripts/run_sink_falsification_uncertainty.py"
    run_step "${PYTHON_BIN}" "${REPO_ROOT}/project/scripts/run_biology_parameter_robustness.py"
    run_step "${PYTHON_BIN}" "${REPO_ROOT}/project/scripts/run_smoothing_kernel_sensitivity.py"
    if [[ "${SKIP_FIGURES}" -ne 1 ]]; then
      run_step "${PYTHON_BIN}" "${REPO_ROOT}/project/scripts/regenerate_manuscript_figures.py"
    fi
    ;;
  *)
    echo "Unsupported mode: ${MODE}" >&2
    exit 1
    ;;
esac

echo
echo "Reproducibility flow complete."
