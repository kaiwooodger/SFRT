#!/usr/bin/env python3
"""Generate a clean fraction-1 comparison table for Phases 17, 18, and 19.

The older Phase 17/18 summaries do not contain peri-GTV spill metrics, so this
script recomputes the effective dose and spill-region metrics from the accepted
fraction dose files to keep the comparison apples-to-apples.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Sequence

import numpy as np

from analyze_topas_outputs import load_topas_grid
from analyze_fractionated_bio_vs_physical import compute_effective_dose, compute_metrics_for_domain
from run_phase14_detailed_headneck_voxel_lattice import build_detailed_plan_phantom
from run_phase15_detailed_headneck_bioaware import build_args_from_summary, load_phase14_summary
from run_phase17_fraction_aware_bio_optimization import (
    compute_peak_valley_metrics,
    build_peak_valley_rois,
    build_spill_region_masks,
    compute_region_metrics,
    write_markdown_table,
)


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline-run-root",
        type=Path,
        default=root / "runs" / "linac_6mv_headneck_detailed_voxel_lattice_sfrt_directplan",
    )
    parser.add_argument(
        "--phase17-run-root",
        type=Path,
        default=root / "runs" / "linac_6mv_headneck_detailed_voxel_lattice_sfrt_phase17_clinical_gtv_core_fraction1_1e5",
    )
    parser.add_argument(
        "--phase18-run-root",
        type=Path,
        default=root / "runs" / "linac_6mv_headneck_detailed_voxel_lattice_sfrt_phase18_fraction1_1e5",
    )
    parser.add_argument(
        "--phase19-run-root",
        type=Path,
        default=root / "runs" / "linac_6mv_headneck_detailed_voxel_lattice_sfrt_phase19_fraction1_1e5",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=root / "runs" / "phase19_outside_gtv_spill_assets",
    )
    parser.add_argument("--prescription-gy", type=float, default=6.0)
    parser.add_argument("--alpha", type=float, default=0.03)
    parser.add_argument("--beta", type=float, default=0.003)
    parser.add_argument("--pde-steps", type=int, default=400)
    parser.add_argument("--pde-dt", type=float, default=0.12)
    parser.add_argument("--peak-radius-mm", type=float, default=8.0)
    parser.add_argument("--valley-exclusion-radius-mm", type=float, default=14.0)
    parser.add_argument("--spill-shell-1-mm", type=float, default=5.0)
    parser.add_argument("--spill-shell-2-mm", type=float, default=15.0)
    parser.add_argument("--spill-shell-3-mm", type=float, default=30.0)
    parser.add_argument("--spill-oar-adjacent-mm", type=float, default=15.0)
    return parser.parse_args()


def write_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"No rows to write for {path}")
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def find_accepted_placement_dir(run_root: Path, placement_name: str) -> Path:
    matches = sorted(run_root.glob(f"placement_*_{placement_name}"))
    if len(matches) != 1:
        raise FileNotFoundError(f"Could not uniquely resolve placement '{placement_name}' under {run_root}")
    return matches[0]


def load_scaled_physical_dose(placement_dir: Path, prescription_gy: float) -> np.ndarray:
    dose_csv = placement_dir / "case" / "dosedata.csv"
    plan_summary = json.loads((placement_dir / "analysis" / "plan_summary.json").read_text())
    dose_raw, _ = load_topas_grid(dose_csv)
    scale = plan_summary.get("physical_scale_factor")
    if scale is None:
        reference = float(plan_summary.get("normalization_reference_d95_gy", 0.0))
        if reference <= 0.0:
            raise RuntimeError(f"Cannot derive normalization scale for {placement_dir}")
        scale = float(prescription_gy) / reference
    return np.asarray(dose_raw, dtype=np.float32) * float(scale)


def build_bio_args() -> SimpleNamespace:
    return SimpleNamespace(
        tumor_cytokine_multiplier=2.0,
        hypoxic_ros_scale=0.12,
        hypoxic_cytokine_multiplier=2.7,
        artery_ros_uptake=0.05,
        artery_cyto_uptake=0.70,
        vein_ros_uptake=0.05,
        vein_cyto_uptake=0.90,
    )


def build_row(
    *,
    label: str,
    run_root: Path,
    summary: Dict[str, object],
    structures: Dict[str, np.ndarray],
    axes_mm: Dict[str, np.ndarray],
    voxel_volume_cc: float,
    voxel_size_mm: tuple[float, float, float],
    args: argparse.Namespace,
) -> Dict[str, object]:
    accepted = summary["accepted_sequence"][0]
    placement_name = str(accepted["placement_name"])
    spots_mm = [tuple(float(v) for v in spot) for spot in accepted["spot_centers_mm"]]
    placement_dir = find_accepted_placement_dir(run_root, placement_name)
    physical_dose = load_scaled_physical_dose(placement_dir, float(args.prescription_gy))
    effective_dose = compute_effective_dose(
        physical_dose=physical_dose,
        voxel_size_mm=voxel_size_mm,
        structures=structures,
        bio_args=build_bio_args(),
        alpha=float(args.alpha),
        beta=float(args.beta),
        steps=int(args.pde_steps),
        dt=float(args.pde_dt),
    )
    physical_metrics = compute_metrics_for_domain(
        physical_dose,
        structures=structures,
        voxel_volume_cc=float(voxel_volume_cc),
        prescription_gy=float(args.prescription_gy),
    )
    effective_metrics = compute_metrics_for_domain(
        effective_dose,
        structures=structures,
        voxel_volume_cc=float(voxel_volume_cc),
        prescription_gy=float(args.prescription_gy),
    )
    peak_mask, valley_mask = build_peak_valley_rois(
        structures,
        axes_mm,
        spots_mm,
        peak_radius_mm=float(args.peak_radius_mm),
        valley_exclusion_radius_mm=float(args.valley_exclusion_radius_mm),
    )
    pvdr_effective = compute_peak_valley_metrics(effective_dose, peak_mask, valley_mask)
    spill_masks = build_spill_region_masks(
        structures=structures,
        axes_mm=axes_mm,
        peak_mask=peak_mask,
        shell_1_mm=float(args.spill_shell_1_mm),
        shell_2_mm=float(args.spill_shell_2_mm),
        shell_3_mm=float(args.spill_shell_3_mm),
        oar_adjacent_mm=float(args.spill_oar_adjacent_mm),
    )
    spill_effective = compute_region_metrics(
        effective_dose,
        spill_masks,
        voxel_volume_cc=float(voxel_volume_cc),
    )
    constraints_ok = (
        float(effective_metrics["SPINAL_CORD"]["d2_gy"]) <= 85.0
        and float(effective_metrics["BRAINSTEM"]["d2_gy"]) <= 30.0
        and float(effective_metrics["PAROTID_R"]["mean_gy"]) <= 60.0
        and float(effective_metrics["THYROID"]["mean_gy"]) <= 50.0
        and float(physical_metrics["BODY"]["dmax_gy"]) <= 400.0
    )
    return {
        "phase_label": label,
        "run_root": str(run_root),
        "objective_mode": str(summary.get("objective_mode", "course_balance")),
        "placement_name": placement_name,
        "selected_spots_mm": "; ".join(f"({x:.1f},{y:.1f},{z:.1f})" for x, y, z in spots_mm),
        "constraints_ok": "yes" if bool(constraints_ok) else "no",
        "ptv_d95_eff_gy": f"{float(effective_metrics['PTV']['d95_gy']):.4f}",
        "gtv_d95_eff_gy": f"{float(effective_metrics['GTV']['d95_gy']):.4f}",
        "pvdr_eff": f"{float(pvdr_effective['pvdr']):.4f}",
        "cord_d2_eff_gy": f"{float(effective_metrics['SPINAL_CORD']['d2_gy']):.4f}",
        "brainstem_d2_eff_gy": f"{float(effective_metrics['BRAINSTEM']['d2_gy']):.4f}",
        "parotid_r_mean_eff_gy": f"{float(effective_metrics['PAROTID_R']['mean_gy']):.4f}",
        "thyroid_mean_eff_gy": f"{float(effective_metrics['THYROID']['mean_gy']):.4f}",
        "spill_shell_0_5_mean_eff_gy": f"{float(spill_effective.get('SPILL_SHELL_0_5', {}).get('mean_gy', 0.0)):.4f}",
        "spill_shell_5_15_mean_eff_gy": f"{float(spill_effective.get('SPILL_SHELL_5_15', {}).get('mean_gy', 0.0)):.4f}",
        "spill_shell_15_30_mean_eff_gy": f"{float(spill_effective.get('SPILL_SHELL_15_30', {}).get('mean_gy', 0.0)):.4f}",
        "outside_gtv_d2_eff_gy": f"{float(spill_effective.get('OUTSIDE_GTV', {}).get('d2_gy', 0.0)):.4f}",
        "ptv_valley_outside_gtv_mean_eff_gy": f"{float(spill_effective.get('PTV_VALLEY_OUTSIDE_GTV', {}).get('mean_gy', 0.0)):.4f}",
        "oar_adjacent_outside_gtv_mean_eff_gy": f"{float(spill_effective.get('OAR_ADJACENT_OUTSIDE_GTV', {}).get('mean_gy', 0.0)):.4f}",
    }


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    baseline_summary = load_phase14_summary(args.baseline_run_root.resolve())
    phantom_args = build_args_from_summary(baseline_summary)
    phantom = build_detailed_plan_phantom(phantom_args)
    structures = phantom["structures"]
    axes_mm = phantom["axes_mm"]
    voxel_volume_cc = float(phantom["meta"]["voxel_volume_cc"])
    voxel_size_mm = tuple(float(v) for v in phantom["meta"]["voxel_size_mm"])

    run_specs = [
        ("phase17_clinical_core_fraction1", args.phase17_run_root),
        ("phase18_clinical_core_backbone_fraction1", args.phase18_run_root),
        ("phase19_outside_gtv_spill_fraction1", args.phase19_run_root),
    ]

    rows: List[Dict[str, object]] = []
    for label, run_root in run_specs:
        summary = json.loads((run_root / "phase17_fraction_aware_summary.json").read_text())
        rows.append(
            build_row(
                label=label,
                run_root=run_root,
                summary=summary,
                structures=structures,
                axes_mm=axes_mm,
                voxel_volume_cc=voxel_volume_cc,
                voxel_size_mm=voxel_size_mm,
                args=args,
            )
        )

    write_csv(args.out_dir / "phase19_fraction1_spill_comparison.csv", rows)
    write_markdown_table(args.out_dir / "phase19_fraction1_spill_comparison.md", rows)
    (args.out_dir / "phase19_fraction1_spill_comparison.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"Wrote comparison assets to {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
