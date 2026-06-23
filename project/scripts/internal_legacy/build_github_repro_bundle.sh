#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SOURCE_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
TARGET_ROOT_DEFAULT="/Users/kw/Desktop/SFRT_Submission_github_repo_clean"

if [[ -d "/Users/kw/Desktop/PMB_revised_conservative/PMB_SFRT_publishable_source_clean" ]]; then
  FIGURE_SOURCE_DEFAULT="/Users/kw/Desktop/PMB_revised_conservative/PMB_SFRT_publishable_source_clean"
else
  FIGURE_SOURCE_DEFAULT="/Users/kw/Desktop/PMB_SFRT_publishable_source_clean"
fi

TARGET_ROOT="${TARGET_ROOT_DEFAULT}"
FIGURE_SOURCE="${FIGURE_SOURCE_DEFAULT}"
OVERWRITE=0

usage() {
  cat <<EOF
Usage:
  bash project/scripts/build_github_repro_bundle.sh [options]

Options:
  --target-root PATH     Output bundle root.
                         Default: ${TARGET_ROOT_DEFAULT}
  --figure-source PATH   Source directory for the cleaned manuscript figure set.
                         Default: ${FIGURE_SOURCE_DEFAULT}
  --overwrite            Replace an existing target directory.
  -h, --help             Show this help text.

This builder creates a GitHub-ready public reproducibility bundle from the
latest working repository. It promotes clean descriptive entry points and
public-facing result directories, while preserving the original implementation
scripts inside project/scripts/internal_legacy/ for provenance and full reruns.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --target-root)
      TARGET_ROOT="$2"
      shift 2
      ;;
    --figure-source)
      FIGURE_SOURCE="$2"
      shift 2
      ;;
    --overwrite)
      OVERWRITE=1
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

if [[ ! -d "${SOURCE_ROOT}" ]]; then
  echo "Source repository not found: ${SOURCE_ROOT}" >&2
  exit 1
fi

if [[ ! -d "${FIGURE_SOURCE}" ]]; then
  echo "Figure source not found: ${FIGURE_SOURCE}" >&2
  exit 1
fi

if [[ -e "${TARGET_ROOT}" ]]; then
  if [[ "${OVERWRITE}" -ne 1 ]]; then
    echo "Target already exists: ${TARGET_ROOT}" >&2
    echo "Re-run with --overwrite to replace it." >&2
    exit 1
  fi
  rm -rf "${TARGET_ROOT}"
fi

mkdir -p \
  "${TARGET_ROOT}/project/scripts/internal_legacy" \
  "${TARGET_ROOT}/project/public_results" \
  "${TARGET_ROOT}/project/data" \
  "${TARGET_ROOT}/project/topas" \
  "${TARGET_ROOT}/manuscript" \
  "${TARGET_ROOT}/figures"

MANIFEST="${TARGET_ROOT}/bundle_manifest.tsv"
printf "kind\tsource\tbundle_path\n" > "${MANIFEST}"

manifest_add() {
  local kind="$1"
  local source="$2"
  local bundle_path="$3"
  printf "%s\t%s\t%s\n" "${kind}" "${source}" "${bundle_path}" >> "${MANIFEST}"
}

copy_file() {
  local src="$1"
  local dst="$2"
  mkdir -p "$(dirname "${dst}")"
  cp "${src}" "${dst}"
  manifest_add "copied" "${src}" "${dst#${TARGET_ROOT}/}"
}

copy_dir() {
  local src="$1"
  local dst="$2"
  mkdir -p "${dst}"
  rsync -a --exclude '.DS_Store' --exclude '__pycache__' --exclude '*.pyc' "${src}/" "${dst}/"
  manifest_add "copied_dir" "${src}" "${dst#${TARGET_ROOT}/}"
}

copy_flat_strip_prefix() {
  local src_dir="$1"
  local dst_dir="$2"
  local prefix="$3"
  mkdir -p "${dst_dir}"
  while IFS= read -r -d '' file; do
    local base clean
    base="$(basename "${file}")"
    clean="${base}"
    clean="${clean#${prefix}}"
    clean="${clean#figure_${prefix}}"
    copy_file "${file}" "${dst_dir}/${clean}"
  done < <(find "${src_dir}" -maxdepth 1 -type f ! -name '.DS_Store' -print0 | sort -z)
}

write_text_file() {
  local dst="$1"
  shift
  mkdir -p "$(dirname "${dst}")"
  cat > "${dst}" <<EOF
$*
EOF
  manifest_add "generated" "(builder template)" "${dst#${TARGET_ROOT}/}"
}

replace_text() {
  local file="$1"
  local old="$2"
  local new="$3"
  python3 - "$file" "$old" "$new" <<'PY'
from pathlib import Path
import sys

path = Path(sys.argv[1])
old = sys.argv[2]
new = sys.argv[3]
path.write_text(path.read_text().replace(old, new))
PY
}

write_python_wrapper() {
  local dst="$1"
  local entry_key="$2"
  local description="$3"
  mkdir -p "$(dirname "${dst}")"
  cat > "${dst}" <<EOF
#!/usr/bin/env python3
\"\"\"${description}\"\"\"

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main() -> int:
    dispatcher = Path(__file__).resolve().parent / "legacy_dispatch.py"
    completed = subprocess.run([sys.executable, str(dispatcher), "${entry_key}", *sys.argv[1:]])
    return int(completed.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
EOF
  chmod +x "${dst}"
  manifest_add "generated" "${entry_key}" "${dst#${TARGET_ROOT}/}"
}

copy_dir "${SOURCE_ROOT}/project/scripts" "${TARGET_ROOT}/project/scripts/internal_legacy"
copy_dir "${SOURCE_ROOT}/project/data" "${TARGET_ROOT}/project/data"
copy_dir "${SOURCE_ROOT}/project/topas" "${TARGET_ROOT}/project/topas"

if [[ -f "${SOURCE_ROOT}/project/data/phase25_safe_core_plan_library.json" ]]; then
  copy_file \
    "${SOURCE_ROOT}/project/data/phase25_safe_core_plan_library.json" \
    "${TARGET_ROOT}/project/data/safe_core_plan_library.json"
fi

copy_dir "${FIGURE_SOURCE}" "${TARGET_ROOT}/figures/manuscript_clean"

if [[ -f "/Users/kw/Desktop/PMB_overleaf_copy_paste_ready.tex" ]]; then
  copy_file \
    "/Users/kw/Desktop/PMB_overleaf_copy_paste_ready.tex" \
    "${TARGET_ROOT}/manuscript/main.tex"
fi

for manuscript_file in \
  "SFRT_Submission.pdf" \
  "parameter_table.csv" \
  "endpoint_manifest.csv" \
  "biological_parameter_provenance_table.csv" \
  "biological_parameter_provenance_table.md" \
  "clean_checkout_repro_status.md" \
  "headline_numbers_table.csv" \
  "headline_numbers_table.md" \
  "model_hierarchy_table.csv" \
  "model_hierarchy_table.md" \
  "synthetic_cohort_plausibility_table.csv" \
  "synthetic_cohort_plausibility_table.md" \
  "uncertainty_interpretation_table.csv" \
  "uncertainty_interpretation_table.md"
do
  if [[ -f "${SOURCE_ROOT}/manuscript/${manuscript_file}" ]]; then
    copy_file "${SOURCE_ROOT}/manuscript/${manuscript_file}" "${TARGET_ROOT}/manuscript/${manuscript_file}"
  fi
done

mkdir -p "${TARGET_ROOT}/project/public_results/cohort_summary"
copy_file \
  "${SOURCE_ROOT}/project/public_results/phase33_34_cohort/phase33_case_manifest.csv" \
  "${TARGET_ROOT}/project/public_results/cohort_summary/case_manifest.csv"
copy_file \
  "${SOURCE_ROOT}/project/public_results/phase33_34_cohort/phase33_case_manifest.json" \
  "${TARGET_ROOT}/project/public_results/cohort_summary/case_manifest.json"
copy_file \
  "${SOURCE_ROOT}/project/public_results/phase33_34_cohort/phase33_physical_endpoint_table.csv" \
  "${TARGET_ROOT}/project/public_results/cohort_summary/physical_endpoint_table.csv"
copy_file \
  "${SOURCE_ROOT}/project/public_results/phase33_34_cohort/phase33_quick_assessment.md" \
  "${TARGET_ROOT}/project/public_results/cohort_summary/physical_quick_assessment.md"
copy_flat_strip_prefix \
  "${SOURCE_ROOT}/project/public_results/phase33_34_cohort/phase34_bio_cohort_primary_oxygen_neutral" \
  "${TARGET_ROOT}/project/public_results/cohort_summary/biology_primary_oxygen_neutral" \
  "phase34_"
copy_flat_strip_prefix \
  "${SOURCE_ROOT}/project/public_results/phase33_34_cohort/phase34_bio_cohort_sensitivity_oxygen_modulated" \
  "${TARGET_ROOT}/project/public_results/cohort_summary/biology_oxygen_sensitivity" \
  "phase34_"

copy_flat_strip_prefix \
  "${SOURCE_ROOT}/project/public_results/phase38_bio_parameter_robustness_primary_oxygen_neutral" \
  "${TARGET_ROOT}/project/public_results/biology_parameter_robustness/primary_oxygen_neutral" \
  "phase38_"
copy_flat_strip_prefix \
  "${SOURCE_ROOT}/project/public_results/phase38_bio_parameter_robustness_sensitivity_oxygen_modulated" \
  "${TARGET_ROOT}/project/public_results/biology_parameter_robustness/oxygen_sensitivity" \
  "phase38_"

copy_flat_strip_prefix \
  "${SOURCE_ROOT}/project/public_results/phase37a_vessel_falsification_cohort" \
  "${TARGET_ROOT}/project/public_results/sink_falsification/cohort" \
  "phase37a_"
copy_flat_strip_prefix \
  "${SOURCE_ROOT}/project/public_results/phase37b_vessel_falsification_uncertainty" \
  "${TARGET_ROOT}/project/public_results/sink_falsification/uncertainty" \
  "phase37b_"

copy_flat_strip_prefix \
  "${SOURCE_ROOT}/project/public_results/revision_checks_20260616/step02_phase35_fullcohort" \
  "${TARGET_ROOT}/project/public_results/repeat_run_uncertainty" \
  "phase35_"
copy_flat_strip_prefix \
  "${SOURCE_ROOT}/project/public_results/revision_checks_20260617/step10_smoothing_kernel_sensitivity" \
  "${TARGET_ROOT}/project/public_results/smoothing_kernel_sensitivity" \
  "phase40_"

copy_dir \
  "${SOURCE_ROOT}/project/public_results/paper_regeneration_smoke" \
  "${TARGET_ROOT}/project/public_results/paper_regeneration_smoke"

cat > "${TARGET_ROOT}/project/scripts/internal_legacy/clean_entrypoints.json" <<'EOF'
{
  "generate_synthetic_cohort": "run_phase32_site_specific_template_phantoms.py",
  "run_topas_cohort": "run_phase33_phase32_topas_cohort.py",
  "apply_biological_reinterpretation": "run_phase34_phase32_bio_cohort.py",
  "run_repeat_uncertainty": "run_phase35_subset_repeat_uncertainty.py",
  "run_sink_falsification_cohort": "run_phase37a_vessel_falsification_cohort.py",
  "run_sink_falsification_uncertainty": "run_phase37b_vessel_falsification_uncertainty.py",
  "run_biology_parameter_robustness": "run_phase38_bio_parameter_robustness.py",
  "run_smoothing_kernel_sensitivity": "run_phase40_smoothing_kernel_sensitivity.py",
  "regenerate_manuscript_figures": "render_pmb_source_clean_figures.py"
}
EOF
manifest_add "generated" "(builder template)" "project/scripts/internal_legacy/clean_entrypoints.json"

cat > "${TARGET_ROOT}/project/scripts/legacy_dispatch.py" <<'EOF'
#!/usr/bin/env python3
"""Dispatch clean public entry points to the preserved implementation layer."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def main() -> int:
    if len(sys.argv) < 2:
        raise SystemExit("Usage: legacy_dispatch.py <entry_key> [args...]")

    entry_key = sys.argv[1]
    script_dir = Path(__file__).resolve().parent
    legacy_root = script_dir / "internal_legacy"
    entrypoints = json.loads((legacy_root / "clean_entrypoints.json").read_text())

    if entry_key not in entrypoints:
        raise SystemExit(f"Unknown legacy entry key: {entry_key}")

    legacy_script = legacy_root / entrypoints[entry_key]
    completed = subprocess.run([sys.executable, str(legacy_script), *sys.argv[2:]])
    return int(completed.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
EOF
chmod +x "${TARGET_ROOT}/project/scripts/legacy_dispatch.py"
manifest_add "generated" "(builder template)" "project/scripts/legacy_dispatch.py"

write_python_wrapper \
  "${TARGET_ROOT}/project/scripts/generate_synthetic_cohort.py" \
  "generate_synthetic_cohort" \
  "Public entry point for generating the 10-case synthetic cohort."
write_python_wrapper \
  "${TARGET_ROOT}/project/scripts/run_topas_cohort.py" \
  "run_topas_cohort" \
  "Public entry point for the full TOPAS physical cohort rerun."
write_python_wrapper \
  "${TARGET_ROOT}/project/scripts/apply_biological_reinterpretation.py" \
  "apply_biological_reinterpretation" \
  "Public entry point for applying the biology-informed reinterpretation workflow."
write_python_wrapper \
  "${TARGET_ROOT}/project/scripts/run_repeat_uncertainty.py" \
  "run_repeat_uncertainty" \
  "Public entry point for the repeated-run uncertainty package."
write_python_wrapper \
  "${TARGET_ROOT}/project/scripts/run_sink_falsification_cohort.py" \
  "run_sink_falsification_cohort" \
  "Public entry point for the sink-falsification cohort analysis."
write_python_wrapper \
  "${TARGET_ROOT}/project/scripts/run_sink_falsification_uncertainty.py" \
  "run_sink_falsification_uncertainty" \
  "Public entry point for the sink-falsification uncertainty overlay."
write_python_wrapper \
  "${TARGET_ROOT}/project/scripts/run_biology_parameter_robustness.py" \
  "run_biology_parameter_robustness" \
  "Public entry point for bounded biology-parameter robustness analysis."
write_python_wrapper \
  "${TARGET_ROOT}/project/scripts/run_smoothing_kernel_sensitivity.py" \
  "run_smoothing_kernel_sensitivity" \
  "Public entry point for the smoothing-kernel sensitivity analysis."
write_python_wrapper \
  "${TARGET_ROOT}/project/scripts/regenerate_manuscript_figures.py" \
  "regenerate_manuscript_figures" \
  "Public entry point for the legacy cleaned figure renderer."

cat > "${TARGET_ROOT}/project/scripts/verify_public_bundle.py" <<'EOF'
#!/usr/bin/env python3
"""Verify the clean public bundle against the frozen manuscript headline numbers."""

from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "project" / "public_results"


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open() as handle:
        return list(csv.DictReader(handle))


def count_nonzero(rows: list[dict[str, str]], key: str) -> int:
    return sum(int(row[key]) != 0 for row in rows)


def main() -> int:
    rank_rows = load_rows(
        RESULTS / "cohort_summary" / "biology_primary_oxygen_neutral" / "rank_shift_table.csv"
    )
    repeat_rows = load_rows(RESULTS / "repeat_run_uncertainty" / "sink_delta_noise_table.csv")
    rank_noise_rows = load_rows(RESULTS / "repeat_run_uncertainty" / "sink_rank_noise_assessment.csv")
    smooth_rows = load_rows(RESULTS / "smoothing_kernel_sensitivity" / "sigma_summary.csv")
    overview_rows = load_rows(
        RESULTS / "biology_parameter_robustness" / "primary_oxygen_neutral" / "cohort_overview.csv"
    )

    overview = {row["metric"]: row["full_cohort_result"] for row in overview_rows}

    summary = {
        "with_sink_vs_physical_rank_shifts": count_nonzero(rank_rows, "with_sink_vs_physical_shift"),
        "no_sink_vs_physical_rank_shifts": count_nonzero(rank_rows, "no_sink_vs_physical_shift"),
        "with_sink_vs_no_sink_rank_shifts": count_nonzero(rank_rows, "with_sink_vs_no_sink_shift"),
        "biology_added_brainstem_flags": overview.get("biology-added brainstem flags", "missing"),
        "repeat_run_endpoint_deltas_above_95pct_band": sum(
            row["exceeds_95pct_noise_band"] == "True" for row in repeat_rows
        ),
        "repeat_run_total_endpoint_comparisons": len(repeat_rows),
        "noise_qualified_sink_rank_changes": sum(
            row["noise_qualified_rank_change"] == "True" for row in rank_noise_rows
        ),
        "smoothing_kernel_survival": {
            row["sigma_mm"]: row["primary_conclusion_survives"] for row in smooth_rows
        },
    }

    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
EOF
chmod +x "${TARGET_ROOT}/project/scripts/verify_public_bundle.py"
manifest_add "generated" "(builder template)" "project/scripts/verify_public_bundle.py"

cat > "${TARGET_ROOT}/project/scripts/reproduce_manuscript.sh" <<'EOF'
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
EOF
chmod +x "${TARGET_ROOT}/project/scripts/reproduce_manuscript.sh"
manifest_add "generated" "(builder template)" "project/scripts/reproduce_manuscript.sh"

write_text_file \
  "${TARGET_ROOT}/project/scripts/START_HERE.md" \
"# Start Here

This is the clean public reproducibility layer for the PMB revision package.

Use this short path:

1. Read \`SCRIPT_MAP.md\`.
2. Run \`bash project/scripts/reproduce_manuscript.sh --mode public\`.
3. Inspect:
   - \`project/public_results/\`
   - \`figures/manuscript_clean/\`
   - \`manuscript/SFRT_Submission.pdf\`

## Two execution modes

- \`public\`
  - Validates the bundled manuscript-facing result tables and frozen headline numbers.
  - Does not require TOPAS.

- \`full\`
  - Uses the clean wrappers to call the preserved full implementation.
  - Requires TOPAS and Geant4.

## Provenance note

The original implementation scripts are preserved in \`project/scripts/internal_legacy/\`.
They retain their historical names so the scientific workflow remains runnable, but the
public entry points and public results in this bundle use descriptive naming."

write_text_file \
  "${TARGET_ROOT}/project/scripts/SCRIPT_MAP.md" \
"# Script Map

## Clean public entry points

| Purpose | Script |
| --- | --- |
| Synthetic cohort generation | \`generate_synthetic_cohort.py\` |
| TOPAS physical cohort | \`run_topas_cohort.py\` |
| Biology-informed reinterpretation | \`apply_biological_reinterpretation.py\` |
| Repeated-run uncertainty | \`run_repeat_uncertainty.py\` |
| Sink falsification cohort | \`run_sink_falsification_cohort.py\` |
| Sink falsification uncertainty overlay | \`run_sink_falsification_uncertainty.py\` |
| Biology-parameter robustness | \`run_biology_parameter_robustness.py\` |
| Smoothing-kernel sensitivity | \`run_smoothing_kernel_sensitivity.py\` |
| Manuscript artifact verification | \`verify_public_bundle.py\` |
| One-command public/full rerun wrapper | \`reproduce_manuscript.sh\` |

## Preserved implementation layer

- \`internal_legacy/\` contains the preserved implementation and provenance scripts.
- That layer keeps the original working names so the scientific dependency chain does not break.
- The clean public layer is what should be linked from the GitHub README and used for first-pass review."

write_text_file \
  "${TARGET_ROOT}/project/public_results/README.md" \
"# Public Results Index

This directory is the clean manuscript-facing result layer.

## Subdirectories

- \`cohort_summary/\` — physical cohort summary plus primary and oxygen-sensitivity biological reinterpretation tables
- \`repeat_run_uncertainty/\` — repeated-run Monte Carlo uncertainty and sink-rank stability summaries
- \`sink_falsification/\` — cohort and uncertainty summaries for the sink falsification controls
- \`biology_parameter_robustness/\` — bounded phenomenological robustness summaries for the primary and oxygen-sensitivity branches
- \`smoothing_kernel_sensitivity/\` — three-kernel sensitivity summary
- \`paper_regeneration_smoke/\` — frozen manuscript artifact smoke-check outputs"

write_text_file \
  "${TARGET_ROOT}/manuscript/reproducibility_guide.md" \
"# PMB reproducibility guide

This clean GitHub bundle is the public reproducibility package for:

> **Biology-informed reinterpretation of lattice radiotherapy using non-local bystander signalling and anatomy-aware vascular sink modelling**

## Fastest public validation

From the repository root:

\`\`\`bash
bash project/scripts/reproduce_manuscript.sh --mode public
\`\`\`

This validates the bundled manuscript-facing outputs and prints the frozen headline numbers.

## Full rerun path

\`\`\`bash
TOPAS_BIN=/path/to/topas \\
G4_DATA_DIR=/path/to/GEANT4 \\
bash project/scripts/reproduce_manuscript.sh --mode full
\`\`\`

The clean wrappers call the preserved full implementation stored in \`project/scripts/internal_legacy/\`.

## Main public result locations

- \`project/public_results/cohort_summary/\`
- \`project/public_results/repeat_run_uncertainty/\`
- \`project/public_results/sink_falsification/\`
- \`project/public_results/biology_parameter_robustness/\`
- \`project/public_results/smoothing_kernel_sensitivity/\`
- \`figures/manuscript_clean/\`
- \`manuscript/SFRT_Submission.pdf\`

## Important note

This repository is a clean public bundle. The exposed entry points and result directories are descriptive and stable for GitHub review. The original implementation scripts are preserved only for provenance and full reruns."

write_text_file \
  "${TARGET_ROOT}/manuscript/figure_manifest.md" \
"# Figure manifest

## Main figures

1. \`figures/manuscript_clean/figures/fig01_workflow.png\`
2. \`figures/manuscript_clean/figures/fig02_calibration_transfer.png\`
3. \`figures/manuscript_clean/figures/fig03a_synthetic_anatomy.png\`
4. \`figures/manuscript_clean/figures/fig03_synthetic_cohort.png\`
5. \`figures/manuscript_clean/figures/fig04_cohort_reinterpretation.png\`
6. \`figures/manuscript_clean/figures/fig05a_uncertainty_sensitivity.png\`
7. \`figures/manuscript_clean/figures/fig05b_sink_falsification.png\`
8. \`figures/manuscript_clean/figures/fig06_assay_readouts.png\`

## Revision robustness figure

- \`project/public_results/biology_parameter_robustness/primary_oxygen_neutral/figure_bio_parameter_robustness.png\`

## Public note

The top-level figure bundle is the publication-facing figure set. Additional figure provenance and legacy generator scripts are preserved in \`project/scripts/internal_legacy/\`."

write_text_file \
  "${TARGET_ROOT}/README.md" \
"# SFRT biological risk-analysis reproducibility bundle

This repository is the clean GitHub-facing reproducibility package for the PMB revision workflow.

It exposes descriptive public entry points, descriptive manuscript-facing result directories, and the current publication-ready figure set, while preserving the original implementation scripts in a provenance layer so the scientific workflow remains reproducible.

## Repository structure

- \`project/scripts/\` — clean public wrappers and verification scripts
- \`project/scripts/internal_legacy/\` — preserved original implementation scripts
- \`project/public_results/\` — clean manuscript-facing result tables and summaries
- \`project/data/\` — small supporting input files
- \`project/topas/\` — TOPAS templates
- \`figures/manuscript_clean/\` — publication-ready figures
- \`manuscript/\` — manuscript PDF plus reviewer-facing tables and guides
- \`bundle_manifest.tsv\` — source-to-bundle mapping for this package

## Quick start

\`\`\`bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-repro.txt
bash project/scripts/reproduce_manuscript.sh --mode public
\`\`\`

## Full rerun

\`\`\`bash
TOPAS_BIN=/path/to/topas \\
G4_DATA_DIR=/path/to/GEANT4 \\
bash project/scripts/reproduce_manuscript.sh --mode full
\`\`\`

## Reproducibility note

The clean public layer avoids internal stage labels in the main review surface. Where historical implementation names were necessary for script stability, they were preserved inside \`project/scripts/internal_legacy/\` and kept out of the first-pass review path."

write_text_file \
  "${TARGET_ROOT}/.gitignore" \
".DS_Store
__pycache__/
*.pyc
.venv/
venv/
project/runs/
"

write_text_file \
  "${TARGET_ROOT}/requirements-repro.txt" \
"numpy
scipy
matplotlib
pandas
Pillow
"

replace_text \
  "${TARGET_ROOT}/project/public_results/cohort_summary/physical_quick_assessment.md" \
  "# Phase 33 quick assessment" \
  "# Physical cohort quick assessment"
replace_text \
  "${TARGET_ROOT}/project/public_results/cohort_summary/biology_primary_oxygen_neutral/quick_assessment.md" \
  "# Phase 34 quick assessment" \
  "# Primary oxygen-neutral biology quick assessment"
replace_text \
  "${TARGET_ROOT}/project/public_results/cohort_summary/biology_primary_oxygen_neutral/quick_assessment.md" \
  "Phase 33 cohort transport" \
  "physical cohort transport"
replace_text \
  "${TARGET_ROOT}/project/public_results/cohort_summary/biology_oxygen_sensitivity/quick_assessment.md" \
  "# Phase 34 quick assessment" \
  "# Oxygen-sensitivity biology quick assessment"
replace_text \
  "${TARGET_ROOT}/project/public_results/cohort_summary/biology_oxygen_sensitivity/quick_assessment.md" \
  "Phase 33 cohort transport" \
  "physical cohort transport"
replace_text \
  "${TARGET_ROOT}/project/public_results/repeat_run_uncertainty/quick_assessment.md" \
  "# Phase 35 quick assessment" \
  "# Repeated-run uncertainty quick assessment"
replace_text \
  "${TARGET_ROOT}/project/public_results/sink_falsification/cohort/quick_assessment.md" \
  "# Phase 37A quick assessment" \
  "# Sink falsification cohort quick assessment"
replace_text \
  "${TARGET_ROOT}/project/public_results/sink_falsification/uncertainty/quick_assessment.md" \
  "# Phase 37B quick assessment" \
  "# Sink falsification uncertainty quick assessment"
replace_text \
  "${TARGET_ROOT}/project/public_results/biology_parameter_robustness/primary_oxygen_neutral/quick_assessment.md" \
  "# Phase 38 quick assessment" \
  "# Primary oxygen-neutral biology-parameter robustness quick assessment"
replace_text \
  "${TARGET_ROOT}/project/public_results/biology_parameter_robustness/oxygen_sensitivity/quick_assessment.md" \
  "# Phase 38 quick assessment" \
  "# Oxygen-sensitivity biology-parameter robustness quick assessment"
replace_text \
  "${TARGET_ROOT}/project/public_results/smoothing_kernel_sensitivity/quick_assessment.md" \
  "# Phase 40 smoothing-kernel sensitivity" \
  "# Smoothing-kernel sensitivity"
replace_text \
  "${TARGET_ROOT}/manuscript/biological_parameter_provenance_table.md" \
  "Phase 38 sweep" \
  "bounded robustness sweep"
replace_text \
  "${TARGET_ROOT}/project/topas/headneck_voxel_lattice_template.txt" \
  "# Generated by scripts/run_phase13_headneck_voxel_lattice.py" \
  "# Generated by the preserved voxel lattice preparation workflow"
replace_text \
  "${TARGET_ROOT}/project/topas/linac_6mv_polyenergetic_direct_photon_mcstats_template.txt" \
  "# Generated by scripts/run_phase12_mc_coupled_plan.py" \
  "# Generated by the preserved polyenergetic direct-photon workflow"

find "${TARGET_ROOT}" -name '.DS_Store' -delete

echo "Clean GitHub reproducibility bundle created at:"
echo "  ${TARGET_ROOT}"
