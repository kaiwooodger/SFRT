#!/usr/bin/env python
"""Run geometry-agnostic calibration/validation sets for bystander-model generalization.

This script calibrates the nonlocal scaling on a simple half-field geometry
using a literature-style survival anchor just outside the field edge, then
reuses the same locked transport/emission parameters on 2D stripe holdouts.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np

from bystander_multispecies_pde_solver import (
    calculate_phase7_survival,
    solve_multispecies_pde_3d_with_hazard,
)
from geometry_generalization_sets import (
    generate_2d_stripe_geometry,
    generate_half_field_geometry,
)

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception as exc:  # pragma: no cover
    raise RuntimeError("matplotlib is required for figure generation") from exc


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description=(
            "Calibrate the bystander model on a half-field geometry, then validate "
            "the locked parameters on 2D stripe geometries and summarize the train/test results."
        )
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=root / "runs" / "linac_6mv_polyenergetic_clinical_sfrt" / "analysis_phase8_geometry_generalization",
        help="Directory for the geometry generalization outputs.",
    )
    parser.add_argument(
        "--phase7-grand-summary",
        type=Path,
        default=root / "runs" / "linac_6mv_polyenergetic_clinical_sfrt" / "analysis_phase7_grand_unified_sweep" / "phase7_grand_unified_summary.json",
        help="Optional existing 3D holdout summary to reference alongside the new calibration/validation outputs.",
    )
    parser.add_argument(
        "--grid-shape",
        nargs=3,
        type=int,
        default=[161, 41, 21],
        help="Synthetic calibration/validation grid shape in project order: nx ny nz.",
    )
    parser.add_argument(
        "--voxel-size-mm",
        type=float,
        default=1.0,
        help="Isotropic voxel size used for the synthetic geometries.",
    )
    parser.add_argument(
        "--dose-peak-gy",
        type=float,
        default=10.0,
        help="Uniform peak dose assigned to the irradiated regions.",
    )
    parser.add_argument(
        "--stripe-pitches-mm",
        nargs="+",
        type=float,
        default=[20.0, 30.0, 40.0],
        help="Stripe pitches used for the 2D validation holdout set.",
    )
    parser.add_argument(
        "--stripe-beam-width-mm",
        type=float,
        default=10.0,
        help="Width of each irradiated stripe in the 2D validation set.",
    )
    parser.add_argument("--alpha", type=float, default=0.10, help="LQ alpha in Gy^-1.")
    parser.add_argument("--beta", type=float, default=0.05, help="LQ beta in Gy^-2.")
    parser.add_argument("--pde-steps", type=int, default=300, help="Number of temporal PDE steps.")
    parser.add_argument("--pde-dt", type=float, default=0.15, help="Explicit Euler time step.")
    parser.add_argument("--ros-diffusion-coeff", type=float, default=0.8, help="ROS diffusion coefficient.")
    parser.add_argument("--cytokine-diffusion-coeff", type=float, default=0.4, help="Cytokine diffusion coefficient.")
    parser.add_argument("--ros-decay-coeff", type=float, default=0.2, help="ROS decay coefficient.")
    parser.add_argument("--cytokine-decay-coeff", type=float, default=0.02, help="Cytokine decay coefficient.")
    parser.add_argument("--ros-emission-emax", type=float, default=1.5, help="Maximum ROS emission strength.")
    parser.add_argument("--cytokine-emission-emax", type=float, default=0.8, help="Maximum cytokine emission strength.")
    parser.add_argument("--emission-gamma-per-gy", type=float, default=0.35, help="Dose-response gamma.")
    parser.add_argument("--weight-ros", type=float, default=0.40, help="ROS contribution to the cumulative hazard.")
    parser.add_argument("--weight-cyto", type=float, default=0.40, help="Cytokine contribution to the cumulative hazard.")
    parser.add_argument(
        "--anchor-inside-beam-mm",
        type=float,
        default=-2.0,
        help="Inside-beam anchor position for reporting (mm from field edge).",
    )
    parser.add_argument(
        "--anchor-near-valley-mm",
        type=float,
        default=2.0,
        help="Near-edge shielded anchor position used for scaling calibration.",
    )
    parser.add_argument(
        "--anchor-far-valley-mm",
        type=float,
        default=10.0,
        help="Deep-valley anchor position used as a holdout check.",
    )
    parser.add_argument(
        "--target-inside-survival-max",
        type=float,
        default=0.01,
        help="Expected upper bound for survival inside the irradiated half-field.",
    )
    parser.add_argument(
        "--target-near-survival",
        type=float,
        default=0.70,
        help="Target survival just outside the half-field edge.",
    )
    parser.add_argument(
        "--target-far-survival",
        type=float,
        default=0.95,
        help="Target survival deep in the shielded half-field.",
    )
    parser.add_argument("--dpi", type=int, default=250, help="Figure DPI.")
    return parser.parse_args()


def centered_axis_mm(n_voxels: int, spacing_mm: float) -> np.ndarray:
    return (np.arange(int(n_voxels), dtype=np.float32) - (int(n_voxels) - 1) / 2.0) * float(spacing_mm)


def nearest_index(values: Sequence[float], target: float) -> int:
    values_arr = np.asarray(values, dtype=np.float32)
    return int(np.argmin(np.abs(values_arr - float(target))))


def write_csv(rows: List[Dict[str, object]], out_file: Path) -> None:
    if not rows:
        raise ValueError("No rows supplied for CSV output.")
    with out_file.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def compose_emission_tensor(
    dose_grid: np.ndarray,
    *,
    emission_emax: Sequence[float],
    emission_gamma_per_gy: float,
    type_modifier: np.ndarray,
    oxygen_modifier: np.ndarray,
) -> np.ndarray:
    dose_grid = np.asarray(dose_grid, dtype=np.float32)
    base_emission = (1.0 - np.exp(-float(emission_gamma_per_gy) * dose_grid)).astype(np.float32)
    emax_vector = np.asarray(emission_emax, dtype=np.float32)
    if emax_vector.shape != (type_modifier.shape[0],):
        raise ValueError("emission_emax length must match the species dimension of the modifiers.")
    return (
        emax_vector[:, None, None, None]
        * base_emission[None, :, :, :]
        * np.asarray(type_modifier, dtype=np.float32)
        * np.asarray(oxygen_modifier, dtype=np.float32)
    ).astype(np.float32, copy=False)


def calculate_lq_survival(dose_grid: np.ndarray, alpha: float, beta: float) -> np.ndarray:
    return np.exp(-float(alpha) * dose_grid - float(beta) * dose_grid**2).astype(np.float32)


def centerline_profile(volume: np.ndarray) -> np.ndarray:
    center_y = volume.shape[1] // 2
    center_z = volume.shape[2] // 2
    return np.asarray(volume[:, center_y, center_z], dtype=np.float32)


def calibrate_scaling_factor(
    lq_survival_grid: np.ndarray,
    hazard_grid: np.ndarray,
    x_mm: np.ndarray,
    *,
    anchor_x_mm: float,
    target_survival: float,
) -> Dict[str, float]:
    profile_lq = centerline_profile(lq_survival_grid)
    profile_hazard = centerline_profile(hazard_grid)
    idx = nearest_index(x_mm, float(anchor_x_mm))
    lq_at_anchor = float(profile_lq[idx])
    hazard_at_anchor = float(profile_hazard[idx])
    if hazard_at_anchor <= 0.0:
        raise ValueError("Cannot calibrate scaling_factor because hazard at the anchor point is zero.")
    if target_survival <= 0.0 or target_survival > lq_at_anchor:
        raise ValueError("target_survival must lie in the interval (0, lq_survival_at_anchor].")
    scaling_factor = -np.log(float(target_survival) / lq_at_anchor) / hazard_at_anchor
    return {
        "anchor_index": int(idx),
        "anchor_x_mm": float(x_mm[idx]),
        "lq_survival_at_anchor": lq_at_anchor,
        "hazard_at_anchor": hazard_at_anchor,
        "target_survival": float(target_survival),
        "scaling_factor": float(scaling_factor),
    }


def evaluate_positions(
    x_mm: np.ndarray,
    dose_profile: np.ndarray,
    lq_profile: np.ndarray,
    final_profile: np.ndarray,
    hazard_profile: np.ndarray,
    positions_mm: Iterable[float],
) -> Dict[str, Dict[str, float]]:
    results: Dict[str, Dict[str, float]] = {}
    for position_mm in positions_mm:
        idx = nearest_index(x_mm, float(position_mm))
        key = f"{float(position_mm):+g}mm"
        results[key] = {
            "requested_x_mm": float(position_mm),
            "sampled_x_mm": float(x_mm[idx]),
            "dose_gy": float(dose_profile[idx]),
            "survival_lq": float(lq_profile[idx]),
            "survival_total": float(final_profile[idx]),
            "hazard": float(hazard_profile[idx]),
        }
    return results


def stripe_regions_mm(
    x_mm: np.ndarray,
    *,
    pitch_mm: float,
    beam_width_mm: float,
    central_feature: str = "peak",
) -> List[tuple[float, float]]:
    nx = int(x_mm.shape[0])
    center_x = nx // 2
    dx_mm = float(abs(x_mm[1] - x_mm[0])) if nx > 1 else 1.0
    pitch_bins = max(1, int(round(float(pitch_mm) / dx_mm)))
    beam_width_bins = max(1, int(round(float(beam_width_mm) / dx_mm)))
    phase_shift_bins = 0 if central_feature == "peak" else pitch_bins // 2
    half_width_low = beam_width_bins // 2
    half_width_high = beam_width_bins - half_width_low
    regions: List[tuple[float, float]] = []
    for stripe_center in range(center_x - 4 * pitch_bins, center_x + 5 * pitch_bins, pitch_bins):
        shifted_center = int(stripe_center + phase_shift_bins)
        start_x = max(0, shifted_center - half_width_low)
        end_x = min(nx - 1, shifted_center + half_width_high - 1)
        if start_x <= end_x:
            left_mm = float(x_mm[start_x] - (dx_mm / 2.0))
            right_mm = float(x_mm[end_x] + (dx_mm / 2.0))
            regions.append((left_mm, right_mm))
    return regions


def plot_half_field_calibration_figure(
    x_mm: np.ndarray,
    dose_profile: np.ndarray,
    lq_profile: np.ndarray,
    final_profile: np.ndarray,
    anchors: Dict[str, Dict[str, float]],
    out_file: Path,
    dpi: int,
) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(10.0, 7.2), constrained_layout=True, sharex=True)

    axes[0].plot(x_mm, dose_profile, color="#34495e", linewidth=2.3)
    axes[0].axvspan(float(x_mm[0]), 0.0, color="#d9d9d9", alpha=0.25)
    axes[0].set_ylabel("Dose (Gy)")
    axes[0].set_title("Figure 1: Half-field calibration geometry")
    axes[0].grid(alpha=0.25)

    axes[1].plot(x_mm, lq_profile, color="#1f77b4", linewidth=2.2, label="Standard LQ")
    axes[1].plot(x_mm, final_profile, color="#d62728", linewidth=2.2, label="Phase 7 (calibrated)")
    axes[1].axvline(0.0, color="#555555", linestyle="--", linewidth=1.4, label="Field edge")
    target_labels = {
        "+2mm": "Target ~0.70",
        "+10mm": "Target ~0.95",
    }
    for key, values in anchors.items():
        axes[1].scatter(values["sampled_x_mm"], values["survival_total"], color="#d62728", s=36, zorder=5)
        label = target_labels.get(key)
        if label is not None:
            axes[1].annotate(
                label,
                (values["sampled_x_mm"], values["survival_total"]),
                xytext=(5, 8),
                textcoords="offset points",
                fontsize=9,
            )
    axes[1].set_xlabel("Distance from field edge x (mm)")
    axes[1].set_ylabel("Cell survival")
    axes[1].set_ylim(0.0, 1.02)
    axes[1].grid(alpha=0.25)
    axes[1].legend(loc="lower right")

    fig.savefig(out_file, dpi=dpi)
    plt.close(fig)


def plot_stripe_validation_figure(
    x_mm: np.ndarray,
    stripe_results: List[Dict[str, object]],
    *,
    beam_width_mm: float,
    out_file: Path,
    dpi: int,
) -> None:
    fig, axes = plt.subplots(len(stripe_results), 1, figsize=(10.5, 3.2 * len(stripe_results)), constrained_layout=True, sharex=True)
    if len(stripe_results) == 1:
        axes = [axes]

    for ax, result in zip(axes, stripe_results):
        dose_profile = np.asarray(result["dose_profile"], dtype=np.float32)
        lq_profile = np.asarray(result["lq_profile"], dtype=np.float32)
        final_profile = np.asarray(result["final_profile"], dtype=np.float32)
        pitch_mm = float(result["pitch_mm"])
        for left_mm, right_mm in stripe_regions_mm(x_mm, pitch_mm=pitch_mm, beam_width_mm=beam_width_mm):
            ax.axvspan(left_mm, right_mm, color="#e5e5e5", alpha=0.28)
        ax.plot(x_mm, lq_profile, color="#1f77b4", linewidth=2.0, label="Standard LQ")
        ax.plot(x_mm, final_profile, color="#d62728", linewidth=2.0, label="Phase 7 (locked)")
        ax.plot(x_mm, dose_profile / max(float(np.max(dose_profile)), 1.0), color="#555555", linestyle=":", linewidth=1.8, label="Dose / max")
        ax.set_ylabel("Survival")
        ax.set_ylim(0.0, 1.02)
        ax.set_title(f"Pitch = {pitch_mm:.0f} mm")
        ax.grid(alpha=0.25)

    axes[0].legend(loc="lower left", ncol=3)
    axes[-1].set_xlabel("Lateral position x (mm)")
    fig.suptitle("Figure 2: 2D stripe validation with locked half-field calibration")
    fig.savefig(out_file, dpi=dpi)
    plt.close(fig)


def write_markdown_report(summary: Dict[str, object], out_file: Path) -> None:
    calibration = summary["calibration"]
    lines = [
        "# Phase 8 Geometry Generalization Summary",
        "",
        "- Calibration set: half-field geometry with a 10 Gy irradiated half-space and no vessels or state-dependent asymmetry.",
        "- Validation set: 2D stripe geometries using the same locked transport and emission parameters.",
        (
            f"- Locked scaling factor from the half-field anchor at x = "
            f"`{float(calibration['scaling_anchor']['anchor_x_mm']):.1f} mm`: "
            f"`{float(calibration['scaling_anchor']['scaling_factor']):.9f}`."
        ),
        "",
        "## Half-Field Anchors",
        "",
        "| Position | Dose (Gy) | LQ Survival | Phase 7 Survival | Hazard |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]

    for key, values in calibration["anchor_points"].items():
        lines.append(
            f"| {key} | {float(values['dose_gy']):.3f} | {float(values['survival_lq']):.3f} | "
            f"{float(values['survival_total']):.3f} | {float(values['hazard']):.3f} |"
        )

    lines.extend(
        [
            "",
            "## Stripe Holdout",
            "",
            "| Pitch | Center Peak Survival | Center Valley Survival | Valley Hazard |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for pitch_key, values in summary["stripe_validation"].items():
        lines.append(
            f"| {pitch_key} mm | {float(values['center_peak_survival_total']):.3f} | "
            f"{float(values['center_valley_survival_total']):.3f} | "
            f"{float(values['center_valley_hazard']):.3f} |"
        )

    if "holdout_3d_reference" in summary:
        lines.extend(
            [
                "",
                "## Existing 3D Holdout Reference",
                "",
                "| Pitch | 3D Valley Survival at 5 cm |",
                "| --- | ---: |",
            ]
        )
        for pitch_key, values in summary["holdout_3d_reference"].items():
            lines.append(
                f"| {pitch_key} mm | {float(values['valley_survival_total']):.3f} |"
            )

    out_file.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    grid_shape = tuple(int(value) for value in args.grid_shape)
    voxel_size_mm = float(args.voxel_size_mm)
    x_mm = centered_axis_mm(grid_shape[0], voxel_size_mm)

    diffusion_coeffs = (float(args.ros_diffusion_coeff), float(args.cytokine_diffusion_coeff))
    decay_coeffs = (float(args.ros_decay_coeff), float(args.cytokine_decay_coeff))
    emission_emax = (float(args.ros_emission_emax), float(args.cytokine_emission_emax))
    hazard_weights = (float(args.weight_ros), float(args.weight_cyto))

    summary: Dict[str, object] = {
        "outdir": str(args.outdir),
        "geometry_model": {
            "grid_shape": [int(value) for value in grid_shape],
            "voxel_size_mm": float(voxel_size_mm),
            "dose_peak_gy": float(args.dose_peak_gy),
        },
        "biological_model": {
            "alpha": float(args.alpha),
            "beta": float(args.beta),
            "steps": int(args.pde_steps),
            "dt": float(args.pde_dt),
            "diffusion_coeffs": [float(value) for value in diffusion_coeffs],
            "decay_coeffs": [float(value) for value in decay_coeffs],
            "emission_emax": [float(value) for value in emission_emax],
            "emission_gamma_per_gy": float(args.emission_gamma_per_gy),
            "hazard_weights": [float(value) for value in hazard_weights],
        },
        "calibration": {},
        "stripe_validation": {},
        "outputs": {},
    }

    half_dose, half_uptake, half_oxygen, half_type = generate_half_field_geometry(
        grid_shape,
        voxel_size_mm,
        dose_peak_gy=float(args.dose_peak_gy),
        num_species=2,
        irradiated_side="left",
    )
    half_emission = compose_emission_tensor(
        half_dose,
        emission_emax=emission_emax,
        emission_gamma_per_gy=float(args.emission_gamma_per_gy),
        type_modifier=half_type,
        oxygen_modifier=half_oxygen,
    )
    half_tensor, half_hazard = solve_multispecies_pde_3d_with_hazard(
        dose_grid=half_dose,
        voxel_size_mm=voxel_size_mm,
        steps=int(args.pde_steps),
        dt=float(args.pde_dt),
        diffusion_coeffs=diffusion_coeffs,
        decay_coeffs=decay_coeffs,
        emission_emax=emission_emax,
        emission_gamma_per_gy=float(args.emission_gamma_per_gy),
        emission_tensor=half_emission,
        uptake_tensor=half_uptake,
        hazard_weights=hazard_weights,
        progress_interval=50,
        verbose=True,
    )
    half_lq = calculate_lq_survival(half_dose, float(args.alpha), float(args.beta))
    scaling_info = calibrate_scaling_factor(
        half_lq,
        half_hazard,
        x_mm,
        anchor_x_mm=float(args.anchor_near_valley_mm),
        target_survival=float(args.target_near_survival),
    )
    scaling_factor = float(scaling_info["scaling_factor"])
    half_survival = calculate_phase7_survival(
        half_lq,
        half_hazard,
        half_dose,
        voxel_size_mm,
        scaling_factor,
        weight_immune=0.0,
        verbose=False,
    )

    half_dose_profile = centerline_profile(half_dose)
    half_lq_profile = centerline_profile(half_lq)
    half_survival_profile = centerline_profile(half_survival)
    half_hazard_profile = centerline_profile(half_hazard)
    half_anchor_points = evaluate_positions(
        x_mm,
        half_dose_profile,
        half_lq_profile,
        half_survival_profile,
        half_hazard_profile,
        [
            float(args.anchor_inside_beam_mm),
            float(args.anchor_near_valley_mm),
            float(args.anchor_far_valley_mm),
        ],
    )

    figure1 = args.outdir / "figure1_half_field_calibration_profile.png"
    plot_half_field_calibration_figure(
        x_mm=x_mm,
        dose_profile=half_dose_profile,
        lq_profile=half_lq_profile,
        final_profile=half_survival_profile,
        anchors=half_anchor_points,
        out_file=figure1,
        dpi=int(args.dpi),
    )

    half_rows = [
        {
            "geometry": "half_field",
            "x_mm": float(position),
            "dose_gy": float(dose),
            "survival_lq": float(lq),
            "survival_total": float(total),
            "hazard": float(hazard),
        }
        for position, dose, lq, total, hazard in zip(
            x_mm,
            half_dose_profile,
            half_lq_profile,
            half_survival_profile,
            half_hazard_profile,
        )
    ]
    half_csv = args.outdir / "half_field_profile.csv"
    write_csv(half_rows, half_csv)

    summary["calibration"] = {
        "scaling_anchor": scaling_info,
        "target_survival_inside_max": float(args.target_inside_survival_max),
        "target_survival_near": float(args.target_near_survival),
        "target_survival_far": float(args.target_far_survival),
        "anchor_points": half_anchor_points,
        "max_ros_concentration": float(np.max(half_tensor[0])),
        "max_cytokine_concentration": float(np.max(half_tensor[1])),
        "max_hazard": float(np.max(half_hazard)),
    }

    stripe_rows: List[Dict[str, object]] = []
    stripe_results_for_plot: List[Dict[str, object]] = []
    for pitch_mm in [float(value) for value in args.stripe_pitches_mm]:
        stripe_dose, stripe_uptake, stripe_oxygen, stripe_type = generate_2d_stripe_geometry(
            grid_shape,
            voxel_size_mm,
            pitch_mm=float(pitch_mm),
            beam_width_mm=float(args.stripe_beam_width_mm),
            dose_peak_gy=float(args.dose_peak_gy),
            num_species=2,
            central_feature="peak",
        )
        stripe_emission = compose_emission_tensor(
            stripe_dose,
            emission_emax=emission_emax,
            emission_gamma_per_gy=float(args.emission_gamma_per_gy),
            type_modifier=stripe_type,
            oxygen_modifier=stripe_oxygen,
        )
        stripe_tensor, stripe_hazard = solve_multispecies_pde_3d_with_hazard(
            dose_grid=stripe_dose,
            voxel_size_mm=voxel_size_mm,
            steps=int(args.pde_steps),
            dt=float(args.pde_dt),
            diffusion_coeffs=diffusion_coeffs,
            decay_coeffs=decay_coeffs,
            emission_emax=emission_emax,
            emission_gamma_per_gy=float(args.emission_gamma_per_gy),
            emission_tensor=stripe_emission,
            uptake_tensor=stripe_uptake,
            hazard_weights=hazard_weights,
            progress_interval=50,
            verbose=True,
        )
        stripe_lq = calculate_lq_survival(stripe_dose, float(args.alpha), float(args.beta))
        stripe_survival = calculate_phase7_survival(
            stripe_lq,
            stripe_hazard,
            stripe_dose,
            voxel_size_mm,
            scaling_factor,
            weight_immune=0.0,
            verbose=False,
        )

        dose_profile = centerline_profile(stripe_dose)
        lq_profile = centerline_profile(stripe_lq)
        final_profile = centerline_profile(stripe_survival)
        hazard_profile = centerline_profile(stripe_hazard)

        center_idx = stripe_dose.shape[0] // 2
        valley_idx = nearest_index(x_mm, float(pitch_mm) / 2.0)
        pitch_key = f"{pitch_mm:g}"
        summary["stripe_validation"][pitch_key] = {
            "pitch_mm": float(pitch_mm),
            "center_peak_x_mm": float(x_mm[center_idx]),
            "center_peak_survival_total": float(final_profile[center_idx]),
            "center_peak_survival_lq": float(lq_profile[center_idx]),
            "center_valley_x_mm": float(x_mm[valley_idx]),
            "center_valley_survival_total": float(final_profile[valley_idx]),
            "center_valley_survival_lq": float(lq_profile[valley_idx]),
            "center_valley_hazard": float(hazard_profile[valley_idx]),
            "max_hazard": float(np.max(stripe_hazard)),
            "max_ros_concentration": float(np.max(stripe_tensor[0])),
            "max_cytokine_concentration": float(np.max(stripe_tensor[1])),
        }
        stripe_results_for_plot.append(
            {
                "pitch_mm": float(pitch_mm),
                "dose_profile": dose_profile,
                "lq_profile": lq_profile,
                "final_profile": final_profile,
            }
        )
        for position, dose, lq, total, hazard in zip(x_mm, dose_profile, lq_profile, final_profile, hazard_profile):
            stripe_rows.append(
                {
                    "geometry": "stripe",
                    "pitch_mm": float(pitch_mm),
                    "x_mm": float(position),
                    "dose_gy": float(dose),
                    "survival_lq": float(lq),
                    "survival_total": float(total),
                    "hazard": float(hazard),
                }
            )

    figure2 = args.outdir / "figure2_stripe_validation_profiles.png"
    plot_stripe_validation_figure(
        x_mm=x_mm,
        stripe_results=stripe_results_for_plot,
        beam_width_mm=float(args.stripe_beam_width_mm),
        out_file=figure2,
        dpi=int(args.dpi),
    )

    stripe_csv = args.outdir / "stripe_validation_profiles.csv"
    write_csv(stripe_rows, stripe_csv)

    if args.phase7_grand_summary.exists():
        phase7_summary = json.loads(args.phase7_grand_summary.read_text(encoding="utf-8"))
        summary["holdout_3d_reference"] = {
            str(pitch_key): {
                "valley_survival_total": float(pitch_values["metrics_by_depth"]["d=5cm"]["valley_survival_total"])
            }
            for pitch_key, pitch_values in phase7_summary.get("generated_pitches", {}).items()
        }
        summary["holdout_3d_reference_source"] = str(args.phase7_grand_summary)

    summary_json = args.outdir / "phase8_geometry_generalization_summary.json"
    summary_md = args.outdir / "phase8_geometry_generalization_summary.md"
    summary["outputs"] = {
        "figure1_half_field": str(figure1),
        "figure2_stripes": str(figure2),
        "half_field_profile_csv": str(half_csv),
        "stripe_validation_csv": str(stripe_csv),
        "summary_json": str(summary_json),
        "summary_md": str(summary_md),
    }
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_markdown_report(summary, summary_md)

    print(f"Phase 8 geometry summary: {summary_json}")
    print(f"Phase 8 geometry report: {summary_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
