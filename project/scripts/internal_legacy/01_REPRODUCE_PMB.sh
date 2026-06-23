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
  cat <<'EOF'
Usage:
  bash project/scripts/01_REPRODUCE_PMB.sh [--mode public|full] [--skip-figures]

Modes:
  public  Use archived public results already included in the repository.
          Runs the lightweight reviewer demos and regenerates the cleaned
          manuscript figure bundle. No TOPAS installation is required.

  full    Rebuild the main manuscript workflow from phantom generation through
          transport, biology, uncertainty, falsification, and figures.
          Requires TOPAS_BIN and G4_DATA_DIR.

Environment variables:
  PYTHON_BIN   Python interpreter to use (default: python3)
  TOPAS_BIN    Path to the TOPAS executable (required for --mode full)
  G4_DATA_DIR  Path to the GEANT4 data directory (required for --mode full)
EOF
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

run_figures() {
  if [[ "${SKIP_FIGURES}" -eq 1 ]]; then
    echo
    echo "==> Skipping figure regeneration (--skip-figures)"
    return
  fi
  run_step "${PYTHON_BIN}" "${REPO_ROOT}/project/scripts/render_pmb_source_clean_figures.py"
}

cd "${REPO_ROOT}"

case "${MODE}" in
  public)
    run_step "${PYTHON_BIN}" "${REPO_ROOT}/examples/minimal_bioaware_demo.py"
    run_step "${PYTHON_BIN}" "${REPO_ROOT}/examples/minimal_vascular_sink_demo.py"
    run_step "${PYTHON_BIN}" "${REPO_ROOT}/examples/minimal_endpoint_extraction.py"
    run_figures
    ;;
  full)
    if [[ -z "${TOPAS_BIN}" || -z "${G4_DATA_DIR}" ]]; then
      echo "Full mode requires TOPAS_BIN and G4_DATA_DIR." >&2
      echo "Example:" >&2
      echo "  TOPAS_BIN=/path/to/topas G4_DATA_DIR=/path/to/GEANT4 bash project/scripts/01_REPRODUCE_PMB.sh --mode full" >&2
      exit 1
    fi
    run_step "${PYTHON_BIN}" "${REPO_ROOT}/project/scripts/run_phase32_site_specific_template_phantoms.py"
    run_step "${PYTHON_BIN}" "${REPO_ROOT}/project/scripts/run_phase33_phase32_topas_cohort.py" --topas-bin "${TOPAS_BIN}" --g4-data-dir "${G4_DATA_DIR}"
    run_step "${PYTHON_BIN}" "${REPO_ROOT}/project/scripts/run_phase34_phase32_bio_cohort.py"
    run_step "${PYTHON_BIN}" "${REPO_ROOT}/project/scripts/run_phase35_subset_repeat_uncertainty.py"
    run_step "${PYTHON_BIN}" "${REPO_ROOT}/project/scripts/run_phase37a_vessel_falsification_cohort.py"
    run_step "${PYTHON_BIN}" "${REPO_ROOT}/project/scripts/run_phase37b_vessel_falsification_uncertainty.py"
    run_figures
    ;;
  *)
    echo "Unsupported mode: ${MODE}" >&2
    usage >&2
    exit 1
    ;;
esac

echo
echo "Reproduction flow complete."
