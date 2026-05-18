#!/usr/bin/env python
"""Calibrate the Phase 7 temporal-hazard scaling factor against the 40 mm anchor."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

from analyze_250mev_sfrt_plan import (
    build_lattice,
    centered_axis_cm,
    choose_reference_depth,
    depth_axis_cm,
    nearest_index,
    reference_peak_value,
)
from analyze_topas_outputs import load_topas_grid
from bystander_multispecies_pde_solver import (
    calculate_phase7_survival,
    calculate_systemic_immune_penalty,
    solve_multispecies_pde_3d_with_hazard,
)


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description=(
            "Recalibrate the Phase 7 temporal-hazard scaling factor so the plain "
            "40 mm anchor lattice reproduces the historical valley-survival benchmark."
        )
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=root / "runs" / "linac_6mv_polyenergetic_clinical_sfrt" / "case" / "dosedata.csv",
        help="TOPAS dose CSV for the polyenergetic clinical-upgrade kernel.",
    )
    parser.add_argument(
        "--reference-summary",
        type=Path,
        default=root
        / "runs"
        / "linac_6mv_polyenergetic_clinical_sfrt"
        / "analysis_phase2_pde_calibrated"
        / "phase2_pde_summary.json",
        help="Reference summary JSON containing the anchor survival benchmark.",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=root / "runs" / "linac_6mv_polyenergetic_clinical_sfrt" / "analysis_phase7_recalibration",
        help="Directory for Phase 7 calibration outputs.",
    )
    parser.add_argument("--pitch-mm", type=float, default=40.0, help="Anchor pitch for the recalibration.")
    parser.add_argument("--n-beams-x", type=int, default=7, help="Beam copies along x.")
    parser.add_argument("--n-beams-y", type=int, default=7, help="Beam copies along y.")
    parser.add_argument(
        "--prescribed-peak-dose-gy",
        type=float,
        default=10.0,
        help="Dose assigned to the reference peak voxel after normalization.",
    )
    parser.add_argument(
        "--uniform-dose-floor-fraction",
        type=float,
        default=0.015,
        help="Uniform dose floor added everywhere as a fraction of the prescribed peak dose.",
    )
    parser.add_argument("--alpha", type=float, default=0.10, help="LQ alpha in Gy^-1.")
    parser.add_argument("--beta", type=float, default=0.05, help="LQ beta in Gy^-2.")
    parser.add_argument("--pde-steps", type=int, default=400, help="Number of explicit PDE time steps.")
    parser.add_argument("--pde-dt", type=float, default=0.15, help="Explicit Euler time step.")
    parser.add_argument(
        "--ros-diffusion-coeff",
        type=float,
        default=0.8,
        help="ROS diffusion coefficient.",
    )
    parser.add_argument(
        "--cytokine-diffusion-coeff",
        type=float,
        default=0.4,
        help="Cytokine diffusion coefficient.",
    )
    parser.add_argument(
        "--ros-decay-coeff",
        type=float,
        default=0.2,
        help="ROS decay coefficient.",
    )
    parser.add_argument(
        "--cytokine-decay-coeff",
        type=float,
        default=0.02,
        help="Cytokine decay coefficient.",
    )
    parser.add_argument(
        "--ros-emission-emax",
        type=float,
        default=1.5,
        help="Maximum ROS emission strength.",
    )
    parser.add_argument(
        "--cytokine-emission-emax",
        type=float,
        default=0.8,
        help="Maximum cytokine emission strength.",
    )
    parser.add_argument(
        "--emission-gamma-per-gy",
        type=float,
        default=0.35,
        help="Dose-response constant gamma for the saturated emission model.",
    )
    parser.add_argument(
        "--weight-ros",
        type=float,
        default=0.40,
        help="ROS contribution to the integrated local hazard.",
    )
    parser.add_argument(
        "--weight-cyto",
        type=float,
        default=0.40,
        help="Cytokine contribution to the integrated local hazard.",
    )
    parser.add_argument(
        "--weight-immune",
        type=float,
        default=0.20,
        help="Global immune penalty weight added after hazard integration.",
    )
    parser.add_argument(
        "--icd-threshold-gy",
        type=float,
        default=10.0,
        help="Dose threshold used for the ICD-volume immune scalar.",
    )
    parser.add_argument(
        "--immune-max-penalty",
        type=float,
        default=1.0,
        help="Maximum normalized systemic immune penalty scalar.",
    )
    parser.add_argument(
        "--immune-half-volume-cm3",
        type=float,
        default=5.0,
        help="ICD volume producing half-max systemic immune penalty.",
    )
    return parser.parse_args()


def normalize_single_beam(
    single_beam: np.ndarray,
    z_cm: np.ndarray,
    prescribed_peak_dose_gy: float,
) -> Tuple[np.ndarray, Dict[str, float]]:
    ref_idx = choose_reference_depth(
        single_beam,
        z_cm,
        argparse.Namespace(reference_mode="dmax", reference_depth_cm=0.0),
    )
    ref_peak_raw, ref_peak_idx_xy = reference_peak_value(single_beam, ref_idx, center_window_bins=5)
    scale = float(prescribed_peak_dose_gy) / float(ref_peak_raw)
    return single_beam * scale, {
        "reference_depth_cm": float(z_cm[ref_idx]),
        "reference_peak_raw_gy": float(ref_peak_raw),
        "scale_factor": float(scale),
        "reference_peak_index_x": int(ref_peak_idx_xy[0]),
        "reference_peak_index_y": int(ref_peak_idx_xy[1]),
    }


def main() -> int:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    reference = json.loads(args.reference_summary.read_text(encoding="utf-8"))
    calibration_ref = reference["calibration"]
    anchor_target_valley_survival = float(calibration_ref["anchor_target_valley_survival"])
    target_depth_cm = float(calibration_ref["sampled_depth_cm"])

    single_beam, header = load_topas_grid(args.csv, retries=5, retry_delay_sec=0.5)
    dx_cm = float(header["dx_cm"])
    dy_cm = float(header["dy_cm"])
    dz_cm = float(header["dz_cm"])
    z_single_cm = depth_axis_cm(single_beam.shape[2], dz_cm)
    normalized_single, norm_meta = normalize_single_beam(
        single_beam,
        z_single_cm,
        prescribed_peak_dose_gy=float(args.prescribed_peak_dose_gy),
    )
    normalized_single = normalized_single.astype(np.float32)

    pitch_bins_x = int(round((float(args.pitch_mm) / 10.0) / dx_cm))
    pitch_bins_y = int(round((float(args.pitch_mm) / 10.0) / dy_cm))
    lattice_dose, offsets_x, _ = build_lattice(
        normalized_single,
        pitch_bins_x=pitch_bins_x,
        pitch_bins_y=pitch_bins_y,
        n_beams_x=int(args.n_beams_x),
        n_beams_y=int(args.n_beams_y),
    )
    lattice_dose = lattice_dose.astype(np.float32)
    uniform_floor_gy = float(args.uniform_dose_floor_fraction) * float(args.prescribed_peak_dose_gy)
    if uniform_floor_gy > 0.0:
        lattice_dose = lattice_dose + np.float32(uniform_floor_gy)

    voxel_size_mm = (dx_cm * 10.0, dy_cm * 10.0, dz_cm * 10.0)
    z_cm = depth_axis_cm(lattice_dose.shape[2], dz_cm)
    x_cm = centered_axis_cm(lattice_dose.shape[0], dx_cm)
    target_depth_idx = int(nearest_index(z_cm, target_depth_cm))
    center_x_idx = lattice_dose.shape[0] // 2
    center_y_idx = lattice_dose.shape[1] // 2
    x_centers_idx = [center_x_idx + off for off in offsets_x]
    valley_x_idx = (x_centers_idx[len(x_centers_idx) // 2] + x_centers_idx[(len(x_centers_idx) // 2) + 1]) // 2

    survival_lq = np.exp(-float(args.alpha) * lattice_dose - float(args.beta) * lattice_dose**2).astype(np.float32)
    local_hazard_weights = (float(args.weight_ros), float(args.weight_cyto))
    if any(value < 0.0 for value in (*local_hazard_weights, float(args.weight_immune))):
        raise ValueError("Phase 7 weights must be non-negative.")
    if float(args.weight_ros) + float(args.weight_cyto) + float(args.weight_immune) > 1.0 + 1e-6:
        raise ValueError("Phase 7 weights should not sum above 1.0.")

    zero_uptake = np.zeros((2, *lattice_dose.shape), dtype=np.float32)
    concentration, hazard_grid = solve_multispecies_pde_3d_with_hazard(
        dose_grid=lattice_dose,
        voxel_size_mm=voxel_size_mm,
        steps=int(args.pde_steps),
        dt=float(args.pde_dt),
        diffusion_coeffs=(float(args.ros_diffusion_coeff), float(args.cytokine_diffusion_coeff)),
        decay_coeffs=(float(args.ros_decay_coeff), float(args.cytokine_decay_coeff)),
        emission_emax=(float(args.ros_emission_emax), float(args.cytokine_emission_emax)),
        emission_gamma_per_gy=float(args.emission_gamma_per_gy),
        state_dependent_emission=False,
        uptake_tensor=zero_uptake,
        hazard_weights=local_hazard_weights,
        progress_interval=50,
        verbose=True,
    )

    valley_lq = float(survival_lq[valley_x_idx, center_y_idx, target_depth_idx])
    valley_hazard = float(hazard_grid[valley_x_idx, center_y_idx, target_depth_idx])
    immune_penalty, icd_volume_cm3 = calculate_systemic_immune_penalty(
        lattice_dose,
        voxel_size_mm,
        icd_threshold_gy=float(args.icd_threshold_gy),
        immune_max_penalty=float(args.immune_max_penalty),
        immune_half_volume_cm3=float(args.immune_half_volume_cm3),
    )
    valley_total_penalty = float(valley_hazard) + (float(args.weight_immune) * float(immune_penalty))
    scaling_factor = -np.log(anchor_target_valley_survival / valley_lq) / valley_total_penalty

    survival_phase7 = calculate_phase7_survival(
        survival_lq,
        hazard_grid,
        lattice_dose,
        voxel_size_mm,
        float(scaling_factor),
        weight_immune=float(args.weight_immune),
        icd_threshold_gy=float(args.icd_threshold_gy),
        immune_max_penalty=float(args.immune_max_penalty),
        immune_half_volume_cm3=float(args.immune_half_volume_cm3),
        verbose=False,
    )
    verified_valley_survival = float(survival_phase7[valley_x_idx, center_y_idx, target_depth_idx])

    summary = {
        "input_csv": str(args.csv),
        "reference_summary": str(args.reference_summary),
        "outdir": str(args.outdir),
        "normalization": norm_meta,
        "anchor": {
            "pitch_mm": float(args.pitch_mm),
            "sampled_depth_cm": float(z_cm[target_depth_idx]),
            "target_valley_survival": float(anchor_target_valley_survival),
            "lq_valley_survival": float(valley_lq),
            "verified_phase7_valley_survival": float(verified_valley_survival),
            "valley_x_cm": float(x_cm[valley_x_idx]),
        },
        "phase7_model": {
            "steps": int(args.pde_steps),
            "dt": float(args.pde_dt),
            "diffusion_coeffs": [float(args.ros_diffusion_coeff), float(args.cytokine_diffusion_coeff)],
            "decay_coeffs": [float(args.ros_decay_coeff), float(args.cytokine_decay_coeff)],
            "emission_emax": [float(args.ros_emission_emax), float(args.cytokine_emission_emax)],
            "emission_gamma_per_gy": float(args.emission_gamma_per_gy),
            "local_hazard_weights": [float(args.weight_ros), float(args.weight_cyto)],
            "weight_immune": float(args.weight_immune),
            "icd_threshold_gy": float(args.icd_threshold_gy),
            "immune_max_penalty": float(args.immune_max_penalty),
            "immune_half_volume_cm3": float(args.immune_half_volume_cm3),
            "voxel_size_mm": [float(value) for value in voxel_size_mm],
            "uniform_dose_floor_gy": float(uniform_floor_gy),
        },
        "calibration": {
            "valley_hazard": float(valley_hazard),
            "immune_penalty_scalar": float(immune_penalty),
            "icd_volume_cm3": float(icd_volume_cm3),
            "valley_total_penalty": float(valley_total_penalty),
            "scaling_factor": float(scaling_factor),
            "max_hazard": float(np.max(hazard_grid)),
            "max_ros_concentration": float(np.max(concentration[0])),
            "max_cytokine_concentration": float(np.max(concentration[1])),
        },
    }

    summary_json = args.outdir / "phase7_recalibration_summary.json"
    summary_md = args.outdir / "phase7_recalibration_summary.md"
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    summary_md.write_text(
        "\n".join(
            [
                "# Phase 7 Recalibration Summary",
                "",
                f"- Anchor target valley survival: `{anchor_target_valley_survival:.9f}` at `{z_cm[target_depth_idx]:.6f} cm`.",
                f"- LQ valley survival: `{valley_lq:.9f}`.",
                f"- Valley hazard: `{valley_hazard:.9f}`.",
                f"- Immune scalar: `{immune_penalty:.9f}` with ICD volume `{icd_volume_cm3:.6f} cm^3`.",
                f"- Total valley penalty: `{valley_total_penalty:.9f}`.",
                f"- New Phase 7 scaling factor: `{scaling_factor:.9f}`.",
                f"- Verified calibrated valley survival: `{verified_valley_survival:.9f}`.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    print(f"Phase 7 recalibration summary: {summary_json}")
    print(f"New Phase 7 scaling factor: {scaling_factor:.9f}")
    print(
        f"Verified valley survival at {z_cm[target_depth_idx]:.6f} cm: {verified_valley_survival:.9f}",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
