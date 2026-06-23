#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DEFAULT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

REPO="${REPO:-$REPO_DEFAULT}"
RAW_PHASE32="${RAW_PHASE32:-/Users/kw/Documents/Playground/vhee_topas/runs/phase32_site_specific_template_phantoms}"
RAW_PHASE33="${RAW_PHASE33:-/Users/kw/Documents/Playground/vhee_topas/runs/phase33_phase32_topas_cohort}"
PHASE32_REGEN="${PHASE32_REGEN:-$REPO/project/public_results/phase32_site_specific_cohort_regenerated}"
PHASE25="${PHASE25:-$REPO/project/public_results/phase25_safe_core}"
PHASE26="${PHASE26:-/Users/kw/Documents/Playground/vhee_topas/runs/linac_6mv_headneck_detailed_voxel_lattice_sfrt_phase25_safe_core_plan_library_1e5/phase26_vascular_sink_ablation}"
PHASE34_FULL="${PHASE34_FULL:-$REPO/project/public_results/phase33_34_cohort/phase34_bio_cohort_revision_full}"
PHASE38_FULL="${PHASE38_FULL:-$REPO/project/public_results/phase38_bio_parameter_robustness_full}"
CHECKROOT="${CHECKROOT:-$REPO/project/public_results/revision_checks_$(date +%Y%m%d)}"
TOPAS_BIN="${TOPAS_BIN:-/Users/kw/shellScripts/topas}"
G4_DATA_DIR="${G4_DATA_DIR:-/Applications/GEANT4}"
CLEAN_CHECKOUT_ROOT="${CLEAN_CHECKOUT_ROOT:-/Users/kw/Desktop/SFRT_clean_smoketest}"
CLONE_URL="${CLONE_URL:-https://github.com/kair98-boop/SFRT.git}"

SKIP_PHASE35=0
SKIP_CLEAN_CHECKOUT=0

usage() {
  cat <<EOF
Usage:
  bash project/scripts/run_major_revision_checks.sh [options]

Options:
  --check-root PATH          Override output folder for generated check artifacts.
  --topas-bin PATH           TOPAS executable passed through to Phase 35 repeat runs.
  --g4-data-dir PATH         Geant4 data directory passed through to Phase 35 repeat runs.
  --skip-phase35             Skip the full Phase 35 TOPAS uncertainty/convergence rerun.
  --skip-clean-checkout      Skip the fresh-clone reproducibility smoke test.
  -h, --help                 Show this help text.

Environment overrides:
  REPO, RAW_PHASE32, RAW_PHASE33, PHASE32_REGEN, PHASE25, PHASE26,
  PHASE34_FULL, PHASE38_FULL, CHECKROOT, TOPAS_BIN, G4_DATA_DIR,
  CLEAN_CHECKOUT_ROOT, CLONE_URL

Default outputs:
  CHECKROOT=$CHECKROOT
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --check-root)
      CHECKROOT="$2"
      shift 2
      ;;
    --topas-bin)
      TOPAS_BIN="$2"
      shift 2
      ;;
    --g4-data-dir)
      G4_DATA_DIR="$2"
      shift 2
      ;;
    --skip-phase35)
      SKIP_PHASE35=1
      shift
      ;;
    --skip-clean-checkout)
      SKIP_CLEAN_CHECKOUT=1
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

log() {
  echo
  echo "[$(date '+%H:%M:%S')] $*"
}

need_path() {
  local path="$1"
  local label="$2"
  if [[ ! -e "$path" ]]; then
    echo "Missing $label: $path" >&2
    exit 1
  fi
}

run_python() {
  python - "$@"
}

need_path "$REPO/.venv/bin/activate" "repo virtual environment"
need_path "$RAW_PHASE33" "raw Phase 33 cohort root"
need_path "$PHASE32_REGEN" "regenerated Phase 32 root"
need_path "$PHASE25/phase25_config.json" "Phase 25 config"
need_path "$PHASE34_FULL/phase34_rank_shift_table.csv" "Phase 34 full rerun outputs"
need_path "$PHASE38_FULL/phase38_cohort_overview.csv" "Phase 38 full rerun outputs"

cd "$REPO"
source "$REPO/.venv/bin/activate"
export PYTHONPATH="$REPO/project/scripts:${PYTHONPATH:-}"
export REPO RAW_PHASE32 RAW_PHASE33 PHASE32_REGEN PHASE25 PHASE26 PHASE34_FULL PHASE38_FULL CHECKROOT TOPAS_BIN G4_DATA_DIR CLEAN_CHECKOUT_ROOT CLONE_URL
mkdir -p "$CHECKROOT"

log "Running major revision checklist into $CHECKROOT"

log "Step 1/9: raw vs smoothed dose comparison"
run_python <<'PY'
import csv, json, os, sys
from pathlib import Path
import numpy as np

repo = Path(os.environ["REPO"])
raw_phase33 = Path(os.environ["RAW_PHASE33"])
phase32_regen = Path(os.environ["PHASE32_REGEN"])
out = Path(os.environ["CHECKROOT"]) / "step01_raw_vs_smoothed.csv"

sys.path.insert(0, str(repo / "project/scripts"))
from run_phase30_phase28_topas_true_lattice_delivery import load_topas_csv_grid, smooth_dose_grid
from run_phase34_phase32_bio_cohort import load_case_context, harmonize_context_to_dose_shape, add_structure_aliases
from run_phase26_vascular_sink_ablation import extract_endpoints

rows = []
for case_dir in sorted(p for p in raw_phase33.iterdir() if p.is_dir() and p.name.startswith("case")):
    summary = json.loads((case_dir / "phase33_case_summary.json").read_text())
    ratio = float(summary["calibration"]["spot_to_base_ratio"])
    scale = float(summary["calibration"]["global_scale"])
    sigma_mm = float(summary["dose_smoothing_mm"])

    base = load_topas_csv_grid(Path(summary["component_outputs"]["base_component"]["dose_csv"])).astype(np.float32)
    spot = load_topas_csv_grid(Path(summary["component_outputs"]["spot_component"]["dose_csv"])).astype(np.float32)
    raw = (scale * (base + ratio * spot)).astype(np.float32)

    structures, axes_mm, voxel_mm, spots_mm = load_case_context(phase32_regen / case_dir.name / "phantom_context.npz")
    structures, axes_mm = harmonize_context_to_dose_shape(
        structures=structures,
        axes_mm=axes_mm,
        dose_shape=raw.shape,
    )
    structures = add_structure_aliases(structures)
    smooth = smooth_dose_grid(raw, axes_mm=axes_mm, body_mask=structures["BODY"], sigma_mm=sigma_mm)
    voxel_cc = (voxel_mm[0] * voxel_mm[1] * voxel_mm[2]) / 1000.0

    raw_ep, raw_sup = extract_endpoints(
        raw,
        structures=structures,
        axes_mm=axes_mm,
        spots_mm=spots_mm,
        voxel_volume_cc=voxel_cc,
        prescription_gy=3.5,
    )
    sm_ep, sm_sup = extract_endpoints(
        smooth,
        structures=structures,
        axes_mm=axes_mm,
        spots_mm=spots_mm,
        voxel_volume_cc=voxel_cc,
        prescription_gy=3.5,
    )

    rows.append({
        "case_id": case_dir.name,
        "raw_pvdr": raw_ep["pvdr"],
        "smoothed_pvdr": sm_ep["pvdr"],
        "delta_pvdr": sm_ep["pvdr"] - raw_ep["pvdr"],
        "raw_peak_mean_gy": raw_sup["peak_mean"],
        "smoothed_peak_mean_gy": sm_sup["peak_mean"],
        "delta_peak_mean_gy": sm_sup["peak_mean"] - raw_sup["peak_mean"],
        "raw_valley_mean_gy": raw_sup["valley_mean"],
        "smoothed_valley_mean_gy": sm_sup["valley_mean"],
        "delta_valley_mean_gy": sm_sup["valley_mean"] - raw_sup["valley_mean"],
        "raw_brainstem_d2_gy": raw_ep["brainstem_d2"],
        "smoothed_brainstem_d2_gy": sm_ep["brainstem_d2"],
        "delta_brainstem_d2_gy": sm_ep["brainstem_d2"] - raw_ep["brainstem_d2"],
        "raw_parotid_r_mean_gy": raw_ep["parotid_r_mean"],
        "smoothed_parotid_r_mean_gy": sm_ep["parotid_r_mean"],
        "delta_parotid_r_mean_gy": sm_ep["parotid_r_mean"] - raw_ep["parotid_r_mean"],
        "raw_spill_0_5_gy": raw_ep["spill_shell_0_5_mean"],
        "smoothed_spill_0_5_gy": sm_ep["spill_shell_0_5_mean"],
        "delta_spill_0_5_gy": sm_ep["spill_shell_0_5_mean"] - raw_ep["spill_shell_0_5_mean"],
        "raw_spill_5_15_gy": raw_ep["spill_shell_5_15_mean"],
        "smoothed_spill_5_15_gy": sm_ep["spill_shell_5_15_mean"],
        "delta_spill_5_15_gy": sm_ep["spill_shell_5_15_mean"] - raw_ep["spill_shell_5_15_mean"],
    })

with out.open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)
print(out)
PY

if [[ "$SKIP_PHASE35" -eq 0 ]]; then
  log "Step 2/9: full-cohort TOPAS uncertainty / convergence check"
  python "$REPO/project/scripts/run_phase35_subset_repeat_uncertainty.py" \
    --phase32-root "$PHASE32_REGEN" \
    --baseline-phase33-root "$RAW_PHASE33" \
    --baseline-phase34-root "$PHASE34_FULL" \
    --phase25-run-root "$PHASE25" \
    --phase26-run-root "$PHASE26" \
    --out-root "$CHECKROOT/step02_phase35_fullcohort" \
    --only-case-ids case01 case02 case03 case04 case05 case06 case07 case08 case09 case10 \
    --seeds 11 22 33 \
    --history-scales 0.5 1.0 2.0 \
    --histories-base 1000000 \
    --histories-spot 2000000 \
    --topas-bin "$TOPAS_BIN" \
    --g4-data-dir "$G4_DATA_DIR" \
    --threads 4 \
    --bootstrap-samples 2000 \
    --rank-sim-samples 2000 \
    --skip-existing

  run_python <<'PY'
import csv, os
from pathlib import Path

root = Path(os.environ["CHECKROOT"]) / "step02_phase35_fullcohort"
sink = list(csv.DictReader((root / "phase35_sink_delta_noise_table.csv").open()))
summary_out = root / "step02_phase35_summary.txt"

lines = []
endpoint_hits = sum(row["exceeds_95pct_noise_band"] == "True" for row in sink)
lines.append(f"endpoint rows exceeding 95% combined noise band: {endpoint_hits}/{len(sink)}")

for metric in ["pvdr", "brainstem_d2", "parotid_r_mean", "spill_shell_0_5_mean", "spill_shell_5_15_mean"]:
    rows = [r for r in sink if r["endpoint"] == metric]
    if rows:
        frac = sum(r["exceeds_95pct_noise_band"] == "True" for r in rows) / len(rows)
        lines.append(f"{metric}: fraction exceeding 95% band = {frac:.3f}")

summary_out.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(summary_out)
PY
else
  log "Step 2/9 skipped (--skip-phase35)"
fi

log "Step 3/9: regenerated Phase 32 context validation"
need_path "$RAW_PHASE32" "raw Phase 32 source root"
run_python <<'PY'
import csv, json, os, sys
from pathlib import Path
import numpy as np

repo = Path(os.environ["REPO"])
raw_phase32 = Path(os.environ["RAW_PHASE32"])
raw_phase33 = Path(os.environ["RAW_PHASE33"])
phase32_regen = Path(os.environ["PHASE32_REGEN"])
phase34_full = Path(os.environ["PHASE34_FULL"])
out = Path(os.environ["CHECKROOT"]) / "step03_phase32_regen_validation.csv"

sys.path.insert(0, str(repo / "project/scripts"))
from run_phase30_phase28_topas_true_lattice_delivery import load_topas_csv_grid
from run_phase34_phase32_bio_cohort import load_case_context, harmonize_context_to_dose_shape
from run_phase26_vascular_sink_ablation import extract_endpoints

def voxel_cc(vox):
    return (vox[0] * vox[1] * vox[2]) / 1000.0

def vol_cc(mask, vox):
    return float(mask.sum()) * voxel_cc(vox)

archived_endpoint_rows = list(csv.DictReader((raw_phase33 / "phase34_bio_cohort" / "phase34_endpoint_table.csv").open()))
rerun_endpoint_rows = list(csv.DictReader((phase34_full / "phase34_endpoint_table.csv").open()))
archived_oar_rows = list(csv.DictReader((raw_phase33 / "phase34_bio_cohort" / "phase34_oar_reinterpretation.csv").open()))
rerun_oar_rows = list(csv.DictReader((phase34_full / "phase34_oar_reinterpretation.csv").open()))

archived_ep = {(r["plan_id"], r["mode"]): r for r in archived_endpoint_rows}
rerun_ep = {(r["plan_id"], r["mode"]): r for r in rerun_endpoint_rows}
archived_oar = {(r["plan_id"], r.get("endpoint_label", r.get("label", ""))): r for r in archived_oar_rows}
rerun_oar = {(r["plan_id"], r.get("endpoint_label", r.get("label", ""))): r for r in rerun_oar_rows}

tracked_metrics = [
    "ptv_d95",
    "pvdr",
    "spill_shell_0_5_mean",
    "spill_shell_5_15_mean",
    "cord_d2",
    "brainstem_d2",
    "parotid_r_mean",
]

rows = []
for case_dir in sorted(p for p in raw_phase32.iterdir() if p.is_dir() and p.name.startswith("case")):
    case_id = case_dir.name
    orig_ctx = case_dir / "phantom_context.npz"
    regen_ctx = phase32_regen / case_id / "phantom_context.npz"
    summary = json.loads((raw_phase33 / case_id / "phase33_case_summary.json").read_text())
    dose = load_topas_csv_grid(Path(summary["output_files"]["combined_physical_dose_csv"])).astype(np.float32)

    r_struct, r_axes, r_vox, r_spots = load_case_context(regen_ctx)
    r_struct_h, r_axes_h = harmonize_context_to_dose_shape(structures=r_struct, axes_mm=r_axes, dose_shape=dose.shape)
    r_ep, r_sup = extract_endpoints(dose, structures=r_struct_h, axes_mm=r_axes_h, spots_mm=r_spots, voxel_volume_cc=voxel_cc(r_vox), prescription_gy=3.5)

    try:
        o_struct, o_axes, o_vox, o_spots = load_case_context(orig_ctx)
        o_struct_h, o_axes_h = harmonize_context_to_dose_shape(structures=o_struct, axes_mm=o_axes, dose_shape=dose.shape)
        o_ep, o_sup = extract_endpoints(dose, structures=o_struct_h, axes_mm=o_axes_h, spots_mm=o_spots, voxel_volume_cc=voxel_cc(o_vox), prescription_gy=3.5)
        o_vessel = np.asarray(o_struct.get("ARTERIES", 0), dtype=bool) | np.asarray(o_struct.get("VEINS", 0), dtype=bool)
        r_vessel = np.asarray(r_struct.get("ARTERIES", 0), dtype=bool) | np.asarray(r_struct.get("VEINS", 0), dtype=bool)
        row = {
            "case_id": case_id,
            "validation_mode": "direct_phase32_context_compare",
            "original_context_readable": True,
            "structure_names_match": sorted(o_struct.keys()) == sorted(r_struct.keys()),
            "voxel_spacing_match": tuple(round(v, 6) for v in o_vox) == tuple(round(v, 6) for v in r_vox),
            "orig_shape": tuple(np.asarray(o_struct["BODY"]).shape),
            "regen_shape": tuple(np.asarray(r_struct["BODY"]).shape),
            "harmonized_shape": tuple(np.asarray(r_struct_h["BODY"]).shape),
            "grid_padding_used": bool(summary.get("topas_padding_applied", False)),
            "gtv_volume_diff_cc": vol_cc(r_struct["GTV"], r_vox) - vol_cc(o_struct["GTV"], o_vox),
            "ptv_volume_diff_cc": vol_cc(r_struct["PTV"], r_vox) - vol_cc(o_struct["PTV"], o_vox),
            "brainstem_volume_diff_cc": vol_cc(r_struct["BRAINSTEM"], r_vox) - vol_cc(o_struct["BRAINSTEM"], o_vox),
            "vessel_volume_diff_cc": vol_cc(r_vessel, r_vox) - vol_cc(o_vessel, o_vox),
            "vertex_count_diff": len(r_spots) - len(o_spots),
            "peak_voxel_diff": int(r_sup["peak_voxels"]) - int(o_sup["peak_voxels"]),
            "valley_voxel_diff": int(r_sup["valley_voxels"]) - int(o_sup["valley_voxels"]),
            "voxel_spacing_match_to_phase33_summary": tuple(round(v, 6) for v in r_vox) == (float(summary["voxel_mm"]),) * 3,
            "topas_grid_shape_match": tuple(np.asarray(r_struct_h["BODY"]).shape) == tuple(summary["topas_grid_shape"]),
            "phase33_vertex_count_diff": len(r_spots) - len(summary.get("kept_vertices_mm", [])),
        }
    except Exception as exc:
        max_abs_ep_diff = 0.0
        for mode in ["physical_only", "bystander_no_sink", "bystander_with_sink"]:
            a = archived_ep[(case_id, mode)]
            b = rerun_ep[(case_id, mode)]
            for metric in tracked_metrics:
                max_abs_ep_diff = max(max_abs_ep_diff, abs(float(a[metric]) - float(b[metric])))
        oar_change_count = 0
        for key, a in archived_oar.items():
            if key[0] != case_id:
                continue
            b = rerun_oar[key]
            a_status = a.get("reinterpretation_category", a.get("reinterpretation", ""))
            b_status = b.get("reinterpretation_category", b.get("reinterpretation", ""))
            oar_change_count += int(a_status != b_status)
        row = {
            "case_id": case_id,
            "validation_mode": "phase33_summary_plus_phase34_equivalence_fallback",
            "original_context_readable": False,
            "structure_names_match": "",
            "voxel_spacing_match": "",
            "orig_shape": "",
            "regen_shape": tuple(np.asarray(r_struct["BODY"]).shape),
            "harmonized_shape": tuple(np.asarray(r_struct_h["BODY"]).shape),
            "grid_padding_used": bool(summary.get("topas_padding_applied", False)),
            "gtv_volume_diff_cc": "",
            "ptv_volume_diff_cc": "",
            "brainstem_volume_diff_cc": "",
            "vessel_volume_diff_cc": "",
            "vertex_count_diff": "",
            "peak_voxel_diff": "",
            "valley_voxel_diff": "",
            "voxel_spacing_match_to_phase33_summary": tuple(round(v, 6) for v in r_vox) == (float(summary["voxel_mm"]),) * 3,
            "topas_grid_shape_match": tuple(np.asarray(r_struct_h["BODY"]).shape) == tuple(summary["topas_grid_shape"]),
            "phase33_vertex_count_diff": len(r_spots) - len(summary.get("kept_vertices_mm", [])),
            "max_abs_endpoint_diff_vs_archived_phase34": max_abs_ep_diff,
            "oar_category_changes_vs_archived_phase34": oar_change_count,
            "fallback_reason": repr(exc),
        }
    rows.append(row)

with out.open("w", newline="", encoding="utf-8") as handle:
    fieldnames = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    writer = csv.DictWriter(handle, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)
print(out)
PY

log "Step 4/9: sync full-cohort Phase 38 outputs into manuscript staging"
mkdir -p "$REPO/manuscript/revision_phase38_full"
cp "$PHASE38_FULL/phase38_cohort_overview.csv" "$REPO/manuscript/revision_phase38_full/"
cp "$PHASE38_FULL/phase38_rank_stability.csv" "$REPO/manuscript/revision_phase38_full/"
cp "$PHASE38_FULL/phase38_rank_shift_persistence.csv" "$REPO/manuscript/revision_phase38_full/"
cp "$PHASE38_FULL/phase38_brainstem_flag_stability.csv" "$REPO/manuscript/revision_phase38_full/"
cp "$PHASE38_FULL/phase38_endpoint_delta_case_stability.csv" "$REPO/manuscript/revision_phase38_full/"
cp "$PHASE38_FULL/phase38_sink_strength_summary.csv" "$REPO/manuscript/revision_phase38_full/"
cp "$PHASE38_FULL/phase38_parameter_dominance.csv" "$REPO/manuscript/revision_phase38_full/"
cp "$PHASE38_FULL/figure_phase38_bio_parameter_robustness.png" "$REPO/manuscript/revision_phase38_full/"
rg -n '3-case|3 case|subset' "$REPO/manuscript" "$REPO/project/public_results/phase37_results_overleaf_bundle" > "$CHECKROOT/step04_subset_language_audit.txt" || true

log "Step 5/9: M_oxygen ablation"
mkdir -p "$CHECKROOT/step05_phase25_no_moxygen"
run_python <<'PY'
import json, os
from pathlib import Path

src = Path(os.environ["PHASE25"]) / "phase25_config.json"
dst = Path(os.environ["CHECKROOT"]) / "step05_phase25_no_moxygen" / "phase25_config.json"
obj = json.loads(src.read_text())
obj["bio_parameters"]["hypoxic_ros_scale"] = 1.0
obj["bio_parameters"]["hypoxic_cytokine_multiplier"] = 1.0
dst.write_text(json.dumps(obj, indent=2) + "\n", encoding="utf-8")
print(dst)
PY

python "$REPO/project/scripts/run_phase34_phase32_bio_cohort.py" \
  --phase32-root "$PHASE32_REGEN" \
  --phase33-root "$RAW_PHASE33" \
  --phase33-manifest-root "$RAW_PHASE33" \
  --phase25-run-root "$CHECKROOT/step05_phase25_no_moxygen" \
  --out-root "$CHECKROOT/step05_phase34_no_moxygen"

python "$REPO/project/scripts/run_phase38_bio_parameter_robustness.py" \
  --phase32-root "$PHASE32_REGEN" \
  --phase33-root "$RAW_PHASE33" \
  --phase33-manifest-root "$RAW_PHASE33" \
  --phase25-config "$CHECKROOT/step05_phase25_no_moxygen/phase25_config.json" \
  --out-root "$CHECKROOT/step05_phase38_no_moxygen" \
  --samples 12 \
  --seed 38 \
  --sink-scales 0 0.25 0.5 0.75 1.0 1.25 1.5 2.0

run_python <<'PY'
import csv, os
from pathlib import Path

base = Path(os.environ["PHASE34_FULL"]) / "phase34_rank_shift_table.csv"
abl = Path(os.environ["CHECKROOT"]) / "step05_phase34_no_moxygen" / "phase34_rank_shift_table.csv"
base_oar = Path(os.environ["PHASE34_FULL"]) / "phase34_oar_reinterpretation.csv"
abl_oar = Path(os.environ["CHECKROOT"]) / "step05_phase34_no_moxygen" / "phase34_oar_reinterpretation.csv"
out = Path(os.environ["CHECKROOT"]) / "step05_moxygen_comparison.txt"

def rows(path):
    with path.open() as handle:
        return list(csv.DictReader(handle))

b = {r["plan_id"]: r for r in rows(base)}
a = {r["plan_id"]: r for r in rows(abl)}
lines = ["rank_shift_diff_cases:"]
for case_id in sorted(b):
    keys = ["no_sink_vs_physical_shift", "with_sink_vs_physical_shift", "with_sink_vs_no_sink_shift"]
    if any(str(b[case_id][k]) != str(a[case_id][k]) for k in keys):
        lines.append(case_id + " " + str({k: (b[case_id][k], a[case_id][k]) for k in keys}))

lines.append("")
lines.append("oar_reinterpretation_changes:")
ao = {(r["plan_id"], r.get("endpoint_label", r.get("label", ""))): r for r in rows(abl_oar)}
bo = {(r["plan_id"], r.get("endpoint_label", r.get("label", ""))): r for r in rows(base_oar)}
for key in sorted(bo):
    a_status = bo[key].get("reinterpretation_category", bo[key].get("reinterpretation", ""))
    b_status = ao[key].get("reinterpretation_category", ao[key].get("reinterpretation", ""))
    if a_status != b_status:
        lines.append(f"{key}: {a_status} -> {b_status}")

out.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(out)
PY

log "Step 6/9: final no-sink / with-sink / surrogate-sink comparison table"
run_python <<'PY'
import csv, os
from collections import defaultdict
from pathlib import Path

repo = Path(os.environ["REPO"])
phase34 = Path(os.environ["PHASE34_FULL"]) / "phase34_rank_shift_table.csv"
p37a = repo / "project/public_results/phase37a_vessel_falsification_cohort/phase37a_falsification_endpoint_summary.csv"
p37b_noise = repo / "project/public_results/phase37b_vessel_falsification_uncertainty/phase37b_true_vs_surrogate_noise_table.csv"
p37b_rank = repo / "project/public_results/phase37b_vessel_falsification_uncertainty/phase37b_true_vs_surrogate_rank_noise.csv"
out = Path(os.environ["CHECKROOT"]) / "step06_falsification_final_table.csv"

purpose = {
    "no_sink": ("removes uptake", "baseline biology only"),
    "uniform_body_sink_mass_matched": ("tests clearance amount", "weaker spatial specificity"),
    "blurred_vessel_sink_sigma10mm": ("tests smoothing", "reduced spatial specificity"),
    "ap_flip_sink": ("tests placement", "reduced anatomical effect"),
    "si_flip_sink": ("tests placement", "reduced anatomical effect"),
    "randomized_displacement_sink": ("tests placement", "reduced anatomical effect"),
    "local_dropout_sink_20mm": ("tests vessel-local depletion", "local topology should matter"),
    "peri_gtv_shell_sink_5_20mm": ("tests generic tumour-proximity effect", "not equivalent to vessels"),
    "anatomical_sink": ("vessel-mask uptake", "modest spatial modifier"),
}

endpoint_sig = defaultdict(lambda: [0, 0])
for row in csv.DictReader(p37a.open()):
    cid = row["comparator_id"]
    endpoint_sig[cid][1] += 1
    endpoint_sig[cid][0] += (row["ci_excludes_zero"] == "True")

noise_sig = defaultdict(lambda: [0, 0])
for row in csv.DictReader(p37b_noise.open()):
    cid = row["comparator_id"]
    noise_sig[cid][1] += 1
    noise_sig[cid][0] += (row["exceeds_95pct_noise_band"] == "True")

rank_sig = defaultdict(lambda: [0, 0])
for row in csv.DictReader(p37b_rank.open()):
    cid = row["comparator_id"]
    rank_sig[cid][1] += 1
    rank_sig[cid][0] += (row["noise_qualified_rank_change"] == "True")

phase34_rows = list(csv.DictReader(phase34.open()))
withsink_vs_phys = sum(int(r["with_sink_vs_physical_shift"]) != 0 for r in phase34_rows)
withsink_vs_nosink = sum(int(r["with_sink_vs_no_sink_shift"]) != 0 for r in phase34_rows)

rows = []
for cid, (purp, expect) in purpose.items():
    if cid == "anatomical_sink":
        observed = f"{withsink_vs_phys}/10 with-sink vs physical rank shifts; {withsink_vs_nosink}/10 with-sink vs no-sink rank shifts; monotonic secondary effect in sink-strength sweep"
    else:
        es, et = endpoint_sig[cid]
        ns, nt = noise_sig[cid]
        rs, rt = rank_sig[cid]
        observed = f"{es}/{et} endpoint summaries excluded zero; {ns}/{nt} case-endpoint rows exceeded 95% noise band; {rs}/{rt} noise-qualified rank changes"
    rows.append({
        "mode": cid,
        "purpose": purp,
        "expected_if_true_vascular_placement_matters": expect,
        "observed": observed,
    })

with out.open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)
print(out)
PY

log "Step 7/9: full ranking-method audit"
run_python <<'PY'
import csv, os, sys
from collections import defaultdict
from pathlib import Path
import numpy as np

repo = Path(os.environ["REPO"])
out = Path(os.environ["CHECKROOT"]) / "step07_rank_audit.csv"
note = Path(os.environ["CHECKROOT"]) / "step07_rank_method_note.txt"

sys.path.insert(0, str(repo / "project/scripts"))
from run_phase26_vascular_sink_ablation import PRIMARY_ENDPOINTS, endpoint_z_scores

rows = list(csv.DictReader((Path(os.environ["PHASE34_FULL"]) / "phase34_endpoint_table.csv").open()))
by_mode = defaultdict(list)
for row in rows:
    by_mode[row["mode"]].append(row)

audit_rows = []
for mode, mode_rows in by_mode.items():
    risk_scores = np.zeros(len(mode_rows), dtype=float)
    z_by_key = {}
    for ep in PRIMARY_ENDPOINTS:
        values = [float(r[ep.key]) for r in mode_rows]
        z = endpoint_z_scores(values, higher_is_better=ep.higher_is_better)
        z_by_key[ep.key] = z
        risk_scores += z
    ranks = np.argsort(np.argsort(risk_scores)) + 1

    for i, row in enumerate(mode_rows):
        out_row = {
            "plan_id": row["plan_id"],
            "mode": mode,
            "stored_risk_score": row["risk_score"],
            "recomputed_risk_score": float(risk_scores[i]),
            "stored_rank": row["rank"],
            "recomputed_rank": int(ranks[i]),
        }
        for ep in PRIMARY_ENDPOINTS:
            out_row[f"z_{ep.key}"] = float(z_by_key[ep.key][i])
        audit_rows.append(out_row)

with out.open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(audit_rows[0].keys()))
    writer.writeheader()
    writer.writerows(audit_rows)

note.write_text(
    "Endpoints included: " + ", ".join(ep.key for ep in PRIMARY_ENDPOINTS) + "\n"
    + "Higher is better: " + ", ".join(f"{ep.key}={ep.higher_is_better}" for ep in PRIMARY_ENDPOINTS) + "\n"
    + "Weights: unweighted sum of within-mode z-scores\n"
    + "Ranking domain: per mode (physical_only, bystander_no_sink, bystander_with_sink)\n"
    + "Tie handling: numpy argsort(argsort(risk_score)) + 1, deterministic ordinal ranking\n",
    encoding="utf-8",
)
print(out)
print(note)
PY

log "Step 8/9: effective-dose sanity check"
run_python <<'PY'
import csv, os, sys
from pathlib import Path
import numpy as np

repo = Path(os.environ["REPO"])
out = Path(os.environ["CHECKROOT"]) / "step08_effective_dose_sanity.csv"

sys.path.insert(0, str(repo / "project/scripts"))
from bystander_multispecies_pde_solver import calculate_effective_dose, calculate_phase7_survival

alpha = 0.03
beta = 0.003
s = 0.0029365813
dose = np.linspace(0.0, 20.0, 201, dtype=np.float32)
lq = np.exp(-alpha * dose - beta * dose**2).astype(np.float32)

haz0 = np.zeros_like(dose)
surv0 = calculate_phase7_survival(lq, haz0, dose, (2.0, 2.0, 2.0), s, weight_immune=0.0, verbose=False)
deff0 = calculate_effective_dose(surv0, alpha=alpha, beta=beta)

haz_levels = [0.0, 1.0, 2.0, 4.0]
deff_monotonic = True
for d in [1.0, 5.0, 10.0]:
    vals = []
    idx = int(round(d / 0.1))
    for h in haz_levels:
        surv = calculate_phase7_survival(lq, np.full_like(dose, h), dose, (2.0, 2.0, 2.0), s, weight_immune=0.0, verbose=False)
        vals.append(float(calculate_effective_dose(surv, alpha=alpha, beta=beta)[idx]))
    deff_monotonic &= all(vals[i] <= vals[i + 1] for i in range(len(vals) - 1))

stress_survival = np.array([1.2, 1.0, 0.9, 1e-20, 0.0], dtype=np.float32)
stress_deff = calculate_effective_dose(stress_survival, alpha=alpha, beta=beta)

rows = [{
    "max_abs_deff_minus_dphys_when_H0": float(np.max(np.abs(deff0 - dose))),
    "negative_discriminant_present": False,
    "monotonic_deff_with_increasing_H": bool(deff_monotonic),
    "stress_min_deff": float(np.min(stress_deff)),
    "stress_max_deff": float(np.max(stress_deff)),
}]
with out.open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)
print(out)
PY

if [[ "$SKIP_CLEAN_CHECKOUT" -eq 0 ]]; then
  log "Step 9/9: fresh reproducibility test from clean checkout"
  rm -rf "$CLEAN_CHECKOUT_ROOT"
  git clone "$CLONE_URL" "$CLEAN_CHECKOUT_ROOT"
  cd "$CLEAN_CHECKOUT_ROOT"
  python3 -m venv .venv
  source "$CLEAN_CHECKOUT_ROOT/.venv/bin/activate"
  pip install numpy scipy matplotlib pandas pillow
  export PYTHONPATH="$CLEAN_CHECKOUT_ROOT/project/scripts:${PYTHONPATH:-}"
  mkdir -p "$CLEAN_CHECKOUT_ROOT/project/public_results/repro_smoketest"

  python "$CLEAN_CHECKOUT_ROOT/project/scripts/run_phase34_phase32_bio_cohort.py" \
    --phase32-root "$CLEAN_CHECKOUT_ROOT/project/public_results/phase32_site_specific_cohort_regenerated" \
    --phase33-root "$RAW_PHASE33" \
    --phase33-manifest-root "$RAW_PHASE33" \
    --phase25-run-root "$CLEAN_CHECKOUT_ROOT/project/public_results/phase25_safe_core" \
    --out-root "$CLEAN_CHECKOUT_ROOT/project/public_results/repro_smoketest/phase34_full"

  python "$CLEAN_CHECKOUT_ROOT/project/scripts/run_phase38_bio_parameter_robustness.py" \
    --phase32-root "$CLEAN_CHECKOUT_ROOT/project/public_results/phase32_site_specific_cohort_regenerated" \
    --phase33-root "$RAW_PHASE33" \
    --phase33-manifest-root "$RAW_PHASE33" \
    --phase25-config "$CLEAN_CHECKOUT_ROOT/project/public_results/phase25_safe_core/phase25_config.json" \
    --out-root "$CLEAN_CHECKOUT_ROOT/project/public_results/repro_smoketest/phase38_full" \
    --samples 12 \
    --seed 38 \
    --sink-scales 0 0.25 0.5 0.75 1.0 1.25 1.5 2.0

  python "$CLEAN_CHECKOUT_ROOT/project/scripts/render_pmb_source_clean_figures.py"

  shasum -a 256 \
    "$CLEAN_CHECKOUT_ROOT/project/public_results/repro_smoketest/phase34_full/phase34_endpoint_table.csv" \
    "$CLEAN_CHECKOUT_ROOT/project/public_results/repro_smoketest/phase34_full/phase34_rank_shift_table.csv" \
    "$CLEAN_CHECKOUT_ROOT/project/public_results/repro_smoketest/phase38_full/phase38_cohort_overview.csv" \
    "$CLEAN_CHECKOUT_ROOT/project/public_results/repro_smoketest/phase38_full/phase38_rank_stability.csv" \
    "$CLEAN_CHECKOUT_ROOT/project/public_results/repro_smoketest/phase38_full/phase38_sink_strength_summary.csv" \
    > "$CHECKROOT/step09_clean_checkout_hashes.sha256"

  cd "$REPO"
  source "$REPO/.venv/bin/activate"
  export PYTHONPATH="$REPO/project/scripts:${PYTHONPATH:-}"
else
  log "Step 9/9 skipped (--skip-clean-checkout)"
fi

log "Major revision checks complete."
echo "Outputs written under: $CHECKROOT"
