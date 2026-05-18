#!/usr/bin/env python
"""Generate the Phase 7 grand unified sweep figures for 20/30/40 mm pitches."""

from __future__ import annotations

import argparse
import csv
import gc
import json
import time
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

from analyze_250mev_sfrt_plan import (
    centered_axis_cm,
    choose_reference_depth,
    depth_axis_cm,
    nearest_index,
    peak_valley_metrics,
    profile_from_center_strip,
    reference_peak_value,
)
from analyze_topas_outputs import load_topas_grid
from bystander_multispecies_pde_solver import (
    build_cylindrical_uptake_tensor,
    calculate_phase7_survival,
    calculate_systemic_immune_penalty,
    centered_z_offset_from_surface_depth_mm,
    solve_multispecies_pde_3d_with_hazard,
)
from generate_phase2_multispecies_vessel_valley_figures import (
    build_lattice_with_x_shift,
    plot_multispecies_lateral_profile_figure,
    vessel_span_cm,
    write_csv,
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
            "Build the Phase 7 grand unified sweep figures with true-valley geometry, "
            "central vessel, right-peak hypoxia, Phase 5 immune weights, and temporal hazard integration."
        )
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=root / "runs" / "linac_6mv_polyenergetic_clinical_sfrt" / "case" / "dosedata.csv",
        help="TOPAS dose CSV for the polyenergetic clinical-upgrade kernel.",
    )
    parser.add_argument(
        "--phase7-calibration-summary",
        type=Path,
        default=root / "runs" / "linac_6mv_polyenergetic_clinical_sfrt" / "analysis_phase7_recalibration" / "phase7_recalibration_summary.json",
        help="Phase 7 recalibration summary JSON containing the calibrated scaling factor.",
    )
    parser.add_argument(
        "--phase7-scaling-factor",
        type=float,
        default=None,
        help="Optional direct override for the Phase 7 scaling factor. When supplied, this takes precedence over --phase7-calibration-summary.",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=root / "runs" / "linac_6mv_polyenergetic_clinical_sfrt" / "analysis_phase7_grand_unified_sweep",
        help="Directory for the grand unified Phase 7 sweep outputs.",
    )
    parser.add_argument(
        "--phase-label",
        type=str,
        default="Phase 7",
        help="Human-readable phase label used in figure titles and reports.",
    )
    parser.add_argument(
        "--summary-stem",
        type=str,
        default="phase7_grand_unified",
        help="Filename stem used for the summary/report/metrics outputs.",
    )
    parser.add_argument(
        "--pitches-mm",
        nargs="+",
        type=float,
        default=[20.0, 30.0, 40.0],
        help="Synthetic lattice pitches in mm.",
    )
    parser.add_argument("--n-beams-x", type=int, default=7, help="Beam copies along x.")
    parser.add_argument("--n-beams-y", type=int, default=7, help="Beam copies along y.")
    parser.add_argument(
        "--prescribed-peak-dose-gy",
        type=float,
        default=10.0,
        help="Dose assigned to the reference peak voxel after normalization.",
    )
    parser.add_argument(
        "--depths-cm",
        nargs=2,
        type=float,
        default=[3.0, 5.0],
        help="Fixed depths to show alongside dmax.",
    )
    parser.add_argument(
        "--profile-half-width-bins",
        type=int,
        default=1,
        help="Half-width of the y-strip averaged into the lateral profile.",
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
    parser.add_argument("--ros-diffusion-coeff", type=float, default=0.8, help="ROS diffusion coefficient.")
    parser.add_argument("--cytokine-diffusion-coeff", type=float, default=0.4, help="Cytokine diffusion coefficient.")
    parser.add_argument("--ros-decay-coeff", type=float, default=0.2, help="ROS decay coefficient.")
    parser.add_argument("--cytokine-decay-coeff", type=float, default=0.02, help="Cytokine decay coefficient.")
    parser.add_argument("--ros-emission-emax", type=float, default=1.5, help="Maximum ROS emission strength.")
    parser.add_argument("--cytokine-emission-emax", type=float, default=0.8, help="Maximum cytokine emission strength.")
    parser.add_argument("--emission-gamma-per-gy", type=float, default=0.35, help="Dose-response gamma.")
    parser.add_argument("--tumor-radius-mm", type=float, default=100.0, help="Tumor mask radius.")
    parser.add_argument("--tumor-center-offset-x-mm", type=float, default=20.0, help="Tumor/hypoxia x-offset from lattice center.")
    parser.add_argument("--tumor-center-offset-y-mm", type=float, default=0.0, help="Tumor/hypoxia y-offset from lattice center.")
    parser.add_argument("--tumor-center-offset-z-mm", type=float, default=0.0, help="Tumor z-offset from lattice center.")
    parser.add_argument("--tumor-cytokine-multiplier", type=float, default=1.0, help="Tumor cytokine multiplier.")
    parser.add_argument("--hypoxic-radius-mm", type=float, default=10.0, help="Hypoxic mask radius.")
    parser.add_argument("--hypoxic-depth-from-surface-mm", type=float, default=50.0, help="Hypoxic center depth from phantom surface.")
    parser.add_argument("--hypoxic-ros-scale", type=float, default=0.10, help="Hypoxic ROS suppression scale.")
    parser.add_argument("--hypoxic-cytokine-multiplier", type=float, default=3.0, help="Hypoxic cytokine boost.")
    parser.add_argument("--vessel-radius-mm", type=float, default=3.0, help="Radius of the cylindrical vascular sink.")
    parser.add_argument("--vessel-center-offset-x-mm", type=float, default=0.0, help="Vessel x-offset from lattice center.")
    parser.add_argument("--vessel-center-offset-y-mm", type=float, default=0.0, help="Vessel y-offset from lattice center.")
    parser.add_argument("--ros-vessel-uptake", type=float, default=0.05, help="ROS uptake inside the vessel.")
    parser.add_argument("--cytokine-vessel-uptake", type=float, default=0.60, help="Cytokine uptake inside the vessel.")
    parser.add_argument("--weight-ros", type=float, default=0.40, help="ROS contribution to the local cumulative hazard.")
    parser.add_argument("--weight-cyto", type=float, default=0.40, help="Cytokine contribution to the local cumulative hazard.")
    parser.add_argument("--weight-immune", type=float, default=0.20, help="Systemic immune penalty weight.")
    parser.add_argument("--icd-threshold-gy", type=float, default=10.0, help="ICD threshold for immune activation.")
    parser.add_argument("--immune-max-penalty", type=float, default=1.0, help="Maximum immune penalty scalar.")
    parser.add_argument("--immune-half-volume-cm3", type=float, default=5.0, help="Half-max ICD volume.")
    parser.add_argument(
        "--x-shift-fraction-of-pitch",
        type=float,
        default=0.5,
        help="Uniform x-shift applied to the entire lattice as a fraction of pitch.",
    )
    parser.add_argument("--dpi", type=int, default=250, help="Figure DPI.")
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
    scale = float(prescribed_peak_dose_gy) / ref_peak_raw
    return single_beam * scale, {
        "reference_depth_cm": float(z_cm[ref_idx]),
        "reference_peak_raw_gy": float(ref_peak_raw),
        "scale_factor": float(scale),
        "reference_peak_index_x": int(ref_peak_idx_xy[0]),
        "reference_peak_index_y": int(ref_peak_idx_xy[1]),
    }


def plot_phase7_biological_comparison_figure(
    x_cm: np.ndarray,
    z_cm: np.ndarray,
    survival_lq: np.ndarray,
    survival_total: np.ndarray,
    survival_loss: np.ndarray,
    guide_lines: List[Tuple[float, str, str, str]],
    title: str,
    out_file: Path,
    dpi: int,
    vessel_span: tuple[float, float] | None,
    vessel_note: str,
) -> None:
    center_y = survival_lq.shape[1] // 2
    lq_slice = survival_lq[:, center_y, :].T
    total_slice = survival_total[:, center_y, :].T
    loss_slice = survival_loss[:, center_y, :].T
    extent = [float(x_cm[0]), float(x_cm[-1]), float(z_cm[0]), float(z_cm[-1])]

    fig, axes = plt.subplots(1, 3, figsize=(16.0, 5.6), constrained_layout=True, sharey=True)
    panel_specs = [
        (lq_slice, "Standard LQ", "viridis", 0.0, 1.0),
        (total_slice, "Phase 7 cumulative hazard", "viridis", 0.0, 1.0),
        (loss_slice, "Survival loss", "magma", 0.0, max(float(np.max(loss_slice)), 1.0e-6)),
    ]

    for panel_index, (ax, (values, panel_title, cmap, vmin, vmax)) in enumerate(zip(axes, panel_specs)):
        image = ax.imshow(
            values,
            origin="lower",
            aspect="auto",
            extent=extent,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
        )
        if vessel_span is not None:
            ax.axvspan(vessel_span[0], vessel_span[1], color="#ffffff", alpha=0.12)
        for depth_cm, label, color, linestyle in guide_lines:
            ax.axhline(depth_cm, color=color, linestyle=linestyle, linewidth=1.7, label=label)
        ax.set_xlabel("Lateral position x (cm)")
        ax.set_title(panel_title)
        if panel_index == 0:
            ax.set_ylabel("Depth z (cm)")
            ax.legend(loc="upper right", fontsize=9)
            ax.text(
                0.02,
                0.98,
                vessel_note,
                transform=ax.transAxes,
                fontsize=9,
                va="top",
                ha="left",
                bbox={"facecolor": "white", "alpha": 0.86, "edgecolor": "#b5b5b5"},
            )
        fig.colorbar(image, ax=ax, shrink=0.88)

    fig.suptitle(title)
    fig.savefig(out_file, dpi=dpi)
    plt.close(fig)


def write_markdown_report(summary: Dict[str, object], out_file: Path) -> None:
    phase_label = str(summary.get("phase_label", "Phase 7"))
    lines = [
        f"# {phase_label} Grand Unified Sweep Summary",
        "",
        (
            f"- Temporal scaling factor inherited from "
            f"`{Path(str(summary['phase7_calibration_source'])).name}`: "
            f"`{float(summary['phase7_model']['scaling_factor']):.9f}`."
        ),
        (
            f"- Solver settings: `steps={int(summary['phase7_model']['steps'])}`, "
            f"`dt={float(summary['phase7_model']['dt']):.3f}`, "
            f"`D=[{float(summary['phase7_model']['diffusion_coeffs'][0]):.2f}, "
            f"{float(summary['phase7_model']['diffusion_coeffs'][1]):.2f}]`, "
            f"`lambda=[{float(summary['phase7_model']['decay_coeffs'][0]):.2f}, "
            f"{float(summary['phase7_model']['decay_coeffs'][1]):.2f}]`."
        ),
        (
            f"- Geometry: true-valley x-shift `{float(summary['lattice_geometry']['x_shift_fraction_of_pitch']):.2f}` pitch, "
            f"vessel radius `{float(summary['vessel_model']['radius_mm']):.1f} mm`."
        ),
        (
            f"- Pathology: state-dependent emission with hypoxic center "
            f"`({float(summary['emission_model']['hypoxic_center_offset_mm'][0]):.1f}, "
            f"{float(summary['emission_model']['hypoxic_center_offset_mm'][1]):.1f}, "
            f"{float(summary['emission_model']['hypoxic_center_offset_mm'][2]):.3f}) mm`."
        ),
        (
            f"- Phase 7 weights: `w_ROS={float(summary['phase7_model']['local_hazard_weights'][0]):.2f}`, "
            f"`w_cyto={float(summary['phase7_model']['local_hazard_weights'][1]):.2f}`, "
            f"`w_immune={float(summary['phase7_model']['weight_immune']):.2f}`."
        ),
        "",
        "| Pitch | Depth | Valley Survival (LQ) | Valley Survival (Phase 7) | Valley Hazard | Total Penalty | ICD Volume | P_immune |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for pitch_key, pitch_summary in summary["generated_pitches"].items():
        depth_metrics = pitch_summary["metrics_by_depth"]["d=5cm"]
        lines.append(
            f"| {float(pitch_summary['pitch_mm']):.0f} mm | {float(depth_metrics['sampled_depth_cm']):.3f} cm | "
            f"{float(depth_metrics['valley_survival_lq']):.3f} | "
            f"{float(depth_metrics['valley_survival_total']):.3f} | "
            f"{float(depth_metrics['valley_hazard']):.3f} | "
            f"{float(depth_metrics['valley_total_penalty']):.3f} | "
            f"{float(pitch_summary['icd_volume_cm3']):.3f} | "
            f"{float(pitch_summary['systemic_immune_penalty']):.3f} |"
        )
    out_file.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    phase_slug = "".join(ch for ch in str(args.phase_label).lower() if ch.isalnum())

    if args.phase7_scaling_factor is None:
        phase7_calibration = json.loads(args.phase7_calibration_summary.read_text(encoding="utf-8"))
        scaling_factor = float(phase7_calibration["calibration"]["scaling_factor"])
        phase7_calibration_source = str(args.phase7_calibration_summary)
    else:
        scaling_factor = float(args.phase7_scaling_factor)
        phase7_calibration_source = "direct_cli_override"

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
    voxel_size_mm = (dx_cm * 10.0, dy_cm * 10.0, dz_cm * 10.0)

    local_hazard_weights = [float(args.weight_ros), float(args.weight_cyto)]
    tumor_center_offset_mm = (
        float(args.tumor_center_offset_x_mm),
        float(args.tumor_center_offset_y_mm),
        float(args.tumor_center_offset_z_mm),
    )
    hypoxic_center_offset_mm = (
        float(tumor_center_offset_mm[0]),
        float(tumor_center_offset_mm[1]),
        float(
            centered_z_offset_from_surface_depth_mm(
                normalized_single.shape[2],
                float(voxel_size_mm[2]),
                float(args.hypoxic_depth_from_surface_mm),
            )
        ),
    )

    colors = {"dmax": "#0b3c5d", "d=3cm": "#c0392b", "d=5cm": "#27ae60"}
    linestyles = {"dmax": "-", "d=3cm": "--", "d=5cm": ":"}
    legend_labels = {"dmax": "d_max", "d=3cm": "d = 3 cm", "d=5cm": "d = 5 cm"}
    requested_depths_cm = [float(value) for value in args.depths_cm]
    metrics_rows: List[Dict[str, object]] = []

    summary: Dict[str, object] = {
        "input_csv": str(args.csv),
        "outdir": str(args.outdir),
        "normalization": norm_meta,
        "phase7_calibration_source": phase7_calibration_source,
        "phase_label": str(args.phase_label),
        "phase7_model": {
            "steps": int(args.pde_steps),
            "dt": float(args.pde_dt),
            "diffusion_coeffs": [float(args.ros_diffusion_coeff), float(args.cytokine_diffusion_coeff)],
            "decay_coeffs": [float(args.ros_decay_coeff), float(args.cytokine_decay_coeff)],
            "emission_emax": [float(args.ros_emission_emax), float(args.cytokine_emission_emax)],
            "emission_gamma_per_gy": float(args.emission_gamma_per_gy),
            "local_hazard_weights": [float(value) for value in local_hazard_weights],
            "weight_immune": float(args.weight_immune),
            "scaling_factor": float(scaling_factor),
            "voxel_size_mm": [float(value) for value in voxel_size_mm],
            "icd_threshold_gy": float(args.icd_threshold_gy),
            "immune_max_penalty": float(args.immune_max_penalty),
            "immune_half_volume_cm3": float(args.immune_half_volume_cm3),
        },
        "emission_model": {
            "type": "state_dependent",
            "tumor_radius_mm": float(args.tumor_radius_mm),
            "tumor_center_offset_mm": [float(value) for value in tumor_center_offset_mm],
            "tumor_cytokine_multiplier": float(args.tumor_cytokine_multiplier),
            "hypoxic_radius_mm": float(args.hypoxic_radius_mm),
            "hypoxic_depth_from_surface_mm": float(args.hypoxic_depth_from_surface_mm),
            "hypoxic_center_offset_mm": [float(value) for value in hypoxic_center_offset_mm],
            "hypoxic_ros_scale": float(args.hypoxic_ros_scale),
            "hypoxic_cytokine_multiplier": float(args.hypoxic_cytokine_multiplier),
        },
        "vessel_model": {
            "radius_mm": float(args.vessel_radius_mm),
            "center_offset_mm": [float(args.vessel_center_offset_x_mm), float(args.vessel_center_offset_y_mm)],
            "uptake_rates_in_vessel": [float(args.ros_vessel_uptake), float(args.cytokine_vessel_uptake)],
        },
        "lattice_geometry": {
            "n_beams_x": int(args.n_beams_x),
            "n_beams_y": int(args.n_beams_y),
            "x_shift_fraction_of_pitch": float(args.x_shift_fraction_of_pitch),
        },
        "physical_floor_model": {
            "uniform_dose_floor_fraction_of_peak": float(args.uniform_dose_floor_fraction),
            "uniform_dose_floor_gy": float(args.uniform_dose_floor_fraction) * float(args.prescribed_peak_dose_gy),
        },
        "generated_pitches": {},
        "outputs": {},
    }

    for pitch_index, pitch_mm in enumerate([float(value) for value in args.pitches_mm], start=1):
        pitch_label = f"{pitch_mm:g}"
        pitch_bins_x = int(round((pitch_mm / 10.0) / dx_cm))
        pitch_bins_y = int(round((pitch_mm / 10.0) / dy_cm))
        x_shift_bins = int(round(float(args.x_shift_fraction_of_pitch) * float(pitch_bins_x)))

        lattice_dose, offsets_x, _, x_centers_idx = build_lattice_with_x_shift(
            normalized_single,
            pitch_bins_x=pitch_bins_x,
            pitch_bins_y=pitch_bins_y,
            n_beams_x=int(args.n_beams_x),
            n_beams_y=int(args.n_beams_y),
            x_shift_bins=int(x_shift_bins),
        )
        lattice_dose = lattice_dose.astype(np.float32)
        uniform_floor_gy = float(args.uniform_dose_floor_fraction) * float(args.prescribed_peak_dose_gy)
        if uniform_floor_gy > 0.0:
            lattice_dose = lattice_dose + np.float32(uniform_floor_gy)

        z_cm = depth_axis_cm(lattice_dose.shape[2], dz_cm)
        x_cm = centered_axis_cm(lattice_dose.shape[0], dx_cm)
        depth_profile = np.sum(lattice_dose, axis=(0, 1))
        depth_indices = {
            "dmax": int(np.argmax(depth_profile)),
            "d=3cm": nearest_index(z_cm, requested_depths_cm[0]),
            "d=5cm": nearest_index(z_cm, requested_depths_cm[1]),
        }
        center_x_idx = lattice_dose.shape[0] // 2
        center_y_idx = lattice_dose.shape[1] // 2
        peak_x_idx = min(x_centers_idx, key=lambda idx: abs(idx - center_x_idx) if idx != center_x_idx else pitch_bins_x)
        valley_x_idx = center_x_idx

        survival_lq = np.exp(-float(args.alpha) * lattice_dose - float(args.beta) * lattice_dose**2).astype(np.float32)
        profiles = {
            label: profile_from_center_strip(lattice_dose, idx, int(args.profile_half_width_bins))
            for label, idx in depth_indices.items()
        }

        uptake_tensor, vessel_mask = build_cylindrical_uptake_tensor(
            lattice_dose.shape,
            voxel_size_mm,
            num_species=2,
            vessel_radius_mm=float(args.vessel_radius_mm),
            vessel_center_offset_mm=(float(args.vessel_center_offset_x_mm), float(args.vessel_center_offset_y_mm)),
            uptake_rates_in_vessel=(float(args.ros_vessel_uptake), float(args.cytokine_vessel_uptake)),
        )
        vessel_span = vessel_span_cm(x_cm, vessel_mask, center_y_idx, dx_cm)
        vessel_note = (
            f"True-valley vessel: x={float(args.vessel_center_offset_x_mm):.1f} mm, "
            f"r={float(args.vessel_radius_mm):.1f} mm, x-shift={float(args.x_shift_fraction_of_pitch):.2f} pitch"
        )

        print(
            f"Running {str(args.phase_label)} grand unified solve for pitch={pitch_mm:g} mm "
            f"({int(args.pde_steps)} steps, dt={float(args.pde_dt):.2f}, x-shift={float(args.x_shift_fraction_of_pitch):.2f} pitch)...",
            flush=True,
        )
        start_time = time.time()
        multispecies_tensor, hazard_grid = solve_multispecies_pde_3d_with_hazard(
            dose_grid=lattice_dose,
            voxel_size_mm=voxel_size_mm,
            steps=int(args.pde_steps),
            dt=float(args.pde_dt),
            diffusion_coeffs=(float(args.ros_diffusion_coeff), float(args.cytokine_diffusion_coeff)),
            decay_coeffs=(float(args.ros_decay_coeff), float(args.cytokine_decay_coeff)),
            emission_emax=(float(args.ros_emission_emax), float(args.cytokine_emission_emax)),
            emission_gamma_per_gy=float(args.emission_gamma_per_gy),
            state_dependent_emission=True,
            tumor_radius_mm=float(args.tumor_radius_mm),
            tumor_center_offset_mm=tumor_center_offset_mm,
            tumor_cytokine_multiplier=float(args.tumor_cytokine_multiplier),
            hypoxic_radius_mm=float(args.hypoxic_radius_mm),
            hypoxic_center_offset_mm=hypoxic_center_offset_mm,
            hypoxic_ros_scale=float(args.hypoxic_ros_scale),
            hypoxic_cytokine_multiplier=float(args.hypoxic_cytokine_multiplier),
            uptake_tensor=uptake_tensor,
            hazard_weights=tuple(local_hazard_weights),
            progress_interval=50,
            verbose=True,
        )
        runtime_sec = time.time() - start_time

        systemic_immune_penalty, icd_volume_cm3 = calculate_systemic_immune_penalty(
            lattice_dose,
            voxel_size_mm,
            icd_threshold_gy=float(args.icd_threshold_gy),
            immune_max_penalty=float(args.immune_max_penalty),
            immune_half_volume_cm3=float(args.immune_half_volume_cm3),
        )
        print(
            f"  -> ICD Volume (>{float(args.icd_threshold_gy):.2f} Gy): {icd_volume_cm3:.2f} cm^3 | "
            f"P_immune scalar: {systemic_immune_penalty:.4f}",
            flush=True,
        )

        survival_total = calculate_phase7_survival(
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
        total_penalty = (hazard_grid + (np.float32(float(args.weight_immune)) * np.float32(systemic_immune_penalty))).astype(np.float32)
        survival_loss = (survival_lq - survival_total).astype(np.float32)

        guide_lines = [
            (float(z_cm[depth_indices["dmax"]]), legend_labels["dmax"], colors["dmax"], linestyles["dmax"]),
            (float(z_cm[depth_indices["d=3cm"]]), legend_labels["d=3cm"], colors["d=3cm"], linestyles["d=3cm"]),
            (float(z_cm[depth_indices["d=5cm"]]), legend_labels["d=5cm"], colors["d=5cm"], linestyles["d=5cm"]),
        ]

        physical_figure_number = (2 * pitch_index) - 1
        bio_figure_number = 2 * pitch_index
        physical_file = args.outdir / f"figure{physical_figure_number}_physical_lateral_profile_pitch{pitch_label}.png"
        bio_file = args.outdir / f"figure{bio_figure_number}_{phase_slug}_biological_comparison_pitch{pitch_label}.png"

        plot_multispecies_lateral_profile_figure(
            x_cm=x_cm,
            profiles=[
                (profiles["dmax"], legend_labels["dmax"], colors["dmax"], linestyles["dmax"]),
                (profiles["d=3cm"], legend_labels["d=3cm"], colors["d=3cm"], linestyles["d=3cm"]),
                (profiles["d=5cm"], legend_labels["d=5cm"], colors["d=5cm"], linestyles["d=5cm"]),
            ],
            title=f"Figure {physical_figure_number}: Physical lateral dose profiles (pitch = {pitch_label} mm)",
            out_file=physical_file,
            dpi=int(args.dpi),
            vessel_span=vessel_span,
            vessel_note=vessel_note,
        )
        plot_phase7_biological_comparison_figure(
            x_cm=x_cm,
            z_cm=z_cm,
            survival_lq=survival_lq,
            survival_total=survival_total,
            survival_loss=survival_loss,
            guide_lines=guide_lines,
            title=f"Figure {bio_figure_number}: {str(args.phase_label)} grand unified biological comparison (pitch = {pitch_label} mm)",
            out_file=bio_file,
            dpi=int(args.dpi),
            vessel_span=vessel_span,
            vessel_note=vessel_note,
        )

        pitch_summary: Dict[str, object] = {
            "pitch_mm": float(pitch_mm),
            "physical_figure": str(physical_file),
            "biological_figure": str(bio_file),
            "solver_runtime_sec": float(runtime_sec),
            "icd_volume_cm3": float(icd_volume_cm3),
            "systemic_immune_penalty": float(systemic_immune_penalty),
            "x_shift_bins": int(x_shift_bins),
            "max_ros_concentration": float(np.max(multispecies_tensor[0])),
            "max_cytokine_concentration": float(np.max(multispecies_tensor[1])),
            "max_hazard": float(np.max(hazard_grid)),
            "max_total_penalty": float(np.max(total_penalty)),
            "metrics_by_depth": {},
        }

        for label, idx in depth_indices.items():
            pvdr = peak_valley_metrics(profiles[label], x_centers_idx, pitch_bins_x)
            row = {
                "pitch_mm": float(pitch_mm),
                "depth_label": label,
                "sampled_depth_cm": float(z_cm[idx]),
                "pvdr": float(pvdr["pvdr"]),
                "mean_peak_gy": float(pvdr["mean_peak_gy"]),
                "mean_valley_gy": float(pvdr["mean_valley_gy"]),
                "peak_dose_gy": float(lattice_dose[peak_x_idx, center_y_idx, idx]),
                "valley_dose_gy": float(lattice_dose[valley_x_idx, center_y_idx, idx]),
                "peak_survival_lq": float(survival_lq[peak_x_idx, center_y_idx, idx]),
                "valley_survival_lq": float(survival_lq[valley_x_idx, center_y_idx, idx]),
                "peak_ros_concentration": float(multispecies_tensor[0, peak_x_idx, center_y_idx, idx]),
                "valley_ros_concentration": float(multispecies_tensor[0, valley_x_idx, center_y_idx, idx]),
                "peak_cytokine_concentration": float(multispecies_tensor[1, peak_x_idx, center_y_idx, idx]),
                "valley_cytokine_concentration": float(multispecies_tensor[1, valley_x_idx, center_y_idx, idx]),
                "systemic_immune_penalty": float(systemic_immune_penalty),
                "icd_volume_cm3": float(icd_volume_cm3),
                "peak_hazard": float(hazard_grid[peak_x_idx, center_y_idx, idx]),
                "valley_hazard": float(hazard_grid[valley_x_idx, center_y_idx, idx]),
                "peak_total_penalty": float(total_penalty[peak_x_idx, center_y_idx, idx]),
                "valley_total_penalty": float(total_penalty[valley_x_idx, center_y_idx, idx]),
                "peak_survival_total": float(survival_total[peak_x_idx, center_y_idx, idx]),
                "valley_survival_total": float(survival_total[valley_x_idx, center_y_idx, idx]),
                "peak_survival_loss": float(survival_loss[peak_x_idx, center_y_idx, idx]),
                "valley_survival_loss": float(survival_loss[valley_x_idx, center_y_idx, idx]),
            }
            metrics_rows.append(row)
            pitch_summary["metrics_by_depth"][label] = row

        summary["generated_pitches"][pitch_label] = pitch_summary

        del lattice_dose
        del survival_lq
        del uptake_tensor
        del vessel_mask
        del multispecies_tensor
        del hazard_grid
        del total_penalty
        del survival_total
        del survival_loss
        gc.collect()

    metrics_csv = args.outdir / f"{args.summary_stem}_metrics.csv"
    summary_json = args.outdir / f"{args.summary_stem}_summary.json"
    summary_md = args.outdir / f"{args.summary_stem}_summary.md"
    write_csv(metrics_rows, metrics_csv)
    summary["outputs"] = {
        "metrics_csv": str(metrics_csv),
        "summary_json": str(summary_json),
        "summary_md": str(summary_md),
    }
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_markdown_report(summary, summary_md)

    print(f"{str(args.phase_label)} grand unified summary: {summary_json}")
    print(f"{str(args.phase_label)} grand unified report: {summary_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
