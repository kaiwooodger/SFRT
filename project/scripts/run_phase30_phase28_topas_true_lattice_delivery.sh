#!/bin/zsh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${REPO_ROOT}"
python3.12 "${SCRIPT_DIR}/run_phase30_phase28_topas_true_lattice_delivery.py" "$@"
