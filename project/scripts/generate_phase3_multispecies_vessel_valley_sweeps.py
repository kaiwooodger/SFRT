#!/usr/bin/env python
"""Generate multi-species vessel-in-valley Phase 3 sensitivity sweeps."""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from analyze_250mev_sfrt_plan import build_lattice, centered_offsets, choose_reference_depth, depth_axis_cm, nearest_index, reference_peak_value
from analyze_topas_outputs import load_topas_grid
from bystander_multispecies_pde_solver import (
    build_cylindrical_uptake_tensor,
    centered_z_offset_from_surface_depth_mm,
    combine_multispecies_toxicity,
    solve_multispecies_pde_3d,
)
from bystander_pde_solver import cfl_stability_limit_3d

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
            "Sweep cytokine decay rate for spatial reach and sweep the inherited "
            "multi-species toxicity scaling factor for the vessel-in-valley branch."
        )
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=root / "runs" / "linac_6mv_polyenergetic_clinical_sfrt" / "case" / "dosedata.csv",
        help="TOPAS dose CSV for the polyenergetic clinical-upgrade kernel.",
    )
    parser.add_argument(
        "--phase2-summary",
        type=Path,
        default=root / "runs" / "linac_6mv_polyenergetic_clinical_sfrt" / "analysis_phase2_multispecies_vessel_valley" / "phase2_multispecies_summary.json",
        help="Phase 2 multi-species summary JSON supplying the inherited scaling factor.",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=root / "runs" / "linac_6mv_polyenergetic_clinical_sfrt" / "analysis_phase3_multispecies_vessel_valley_sweeps",
        help="Directory for multi-species vessel-in-valley Phase 3 sweep outputs.",
    )
    parser.add_argument(
        "--pitches-mm",
        nargs="+",
        type=float,
        default=[20.0, 30.0, 40.0],
        help="Lattice pitches to compare.",
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
        "--target-depth-cm",
        type=float,
        default=5.0,
        help="Depth where valley survival is sampled for the sweep figures.",
    )
    parser.add_argument(
        "--uniform-dose-floor-fraction",
        type=float,
        default=0.015,
        help="Uniform dose floor added everywhere as a fraction of the prescribed peak dose.",
    )
    parser.add_argument("--alpha", type=float, default=0.10, help="LQ alpha in Gy^-1.")
    parser.add_argument("--beta", type=float, default=0.05, help="LQ beta in Gy^-2.")
    parser.add_argument(
        "--pde-steps",
        type=int,
        default=400,
        help="Number of explicit multi-species PDE time steps per solve.",
    )
    parser.add_argument(
        "--pde-dt",
        type=float,
        default=0.15,
        help="Explicit Euler time step. Must respect the CFL limit.",
    )
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
        help="ROS decay coefficient held fixed during the spatial-reach sweep.",
    )
    parser.add_argument(
        "--cytokine-k-values",
        nargs="+",
        type=float,
        default=[0.5, 0.125, 0.05, 0.02, 0.005],
        help="Cytokine decay-rate sweep values for Figure 7.",
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
        "--state-dependent-emission",
        action="store_true",
        help="Use tumor and hypoxia masks to modulate the multi-species emission tensor.",
    )
    parser.add_argument(
        "--tumor-radius-mm",
        type=float,
        default=15.0,
        help="Radius of the spherical tumor-core mask used by the state-dependent emission model.",
    )
    parser.add_argument(
        "--tumor-center-offset-x-mm",
        type=float,
        default=0.0,
        help="Tumor-core x-offset from the lattice center for state-dependent emission.",
    )
    parser.add_argument(
        "--tumor-center-offset-y-mm",
        type=float,
        default=0.0,
        help="Tumor-core y-offset from the lattice center for state-dependent emission.",
    )
    parser.add_argument(
        "--tumor-center-offset-z-mm",
        type=float,
        default=0.0,
        help="Tumor-core z-offset from the lattice center for state-dependent emission.",
    )
    parser.add_argument(
        "--tumor-cytokine-multiplier",
        type=float,
        default=2.0,
        help="Cytokine emission multiplier inside the tumor-core mask.",
    )
    parser.add_argument(
        "--hypoxic-radius-mm",
        type=float,
        default=5.0,
        help="Radius of the hypoxic core mask used by the state-dependent emission model.",
    )
    parser.add_argument(
        "--hypoxic-depth-from-surface-mm",
        type=float,
        default=None,
        help=(
            "Optional hypoxic-center depth measured from the phantom entrance surface. "
            "When provided, this overrides the hypoxic z location implied by "
            "--tumor-center-offset-z-mm while keeping the hypoxic x/y center aligned "
            "with the tumor offsets."
        ),
    )
    parser.add_argument(
        "--hypoxic-ros-scale",
        type=float,
        default=0.10,
        help="ROS emission scale factor inside the hypoxic mask.",
    )
    parser.add_argument(
        "--hypoxic-cytokine-multiplier",
        type=float,
        default=3.0,
        help="Cytokine emission multiplier inside the hypoxic mask.",
    )
    parser.add_argument(
        "--vessel-radius-mm",
        type=float,
        default=3.0,
        help="Radius of the cylindrical vascular sink.",
    )
    parser.add_argument(
        "--vessel-center-offset-x-mm",
        type=float,
        default=0.0,
        help="Lateral x-offset of the vascular sink center from the lattice center.",
    )
    parser.add_argument(
        "--vessel-center-offset-y-mm",
        type=float,
        default=0.0,
        help="Lateral y-offset of the vascular sink center from the lattice center.",
    )
    parser.add_argument(
        "--ros-vessel-uptake",
        type=float,
        default=0.05,
        help="In-vessel ROS uptake/scavenging coefficient.",
    )
    parser.add_argument(
        "--cytokine-vessel-uptake",
        type=float,
        default=0.60,
        help="In-vessel cytokine uptake/scavenging coefficient.",
    )
    parser.add_argument(
        "--species-weights",
        nargs=2,
        type=float,
        default=[0.5, 0.5],
        help="Linear weights used to collapse [ROS, cytokine] into total toxicity.",
    )
    parser.add_argument(
        "--scaling-multipliers",
        nargs="+",
        type=float,
        default=[0.5, 0.75, 1.0, 1.25, 1.5],
        help="Multipliers applied to the inherited scaling factor for Figure 8.",
    )
    parser.add_argument(
        "--x-shift-fraction-of-pitch",
        type=float,
        default=0.0,
        help="Uniform x-shift applied to the entire lattice as a fraction of pitch. Use 0.5 to center a valley at x=0.",
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


def write_csv(rows: List[Dict[str, object]], out_file: Path) -> None:
    if not rows:
        raise ValueError("No rows supplied for CSV output.")
    with out_file.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def effective_diffusion_length_mm(diffusion_coeff: float, k_decay: float) -> float:
    return math.sqrt(float(diffusion_coeff) / float(k_decay))


def build_lattice_with_x_shift(
    kernel: np.ndarray,
    *,
    pitch_bins_x: int,
    pitch_bins_y: int,
    n_beams_x: int,
    n_beams_y: int,
    x_shift_bins: int,
) -> Tuple[np.ndarray, List[int], List[int], List[int]]:
    offsets_x = centered_offsets(n_beams_x, pitch_bins_x)
    offsets_y = centered_offsets(n_beams_y, pitch_bins_y)
    out_shape = (
        int(kernel.shape[0] + max(0, n_beams_x - 1) * max(0, pitch_bins_x) + (2 * abs(int(x_shift_bins)))),
        int(kernel.shape[1] + max(0, n_beams_y - 1) * max(0, pitch_bins_y)),
        int(kernel.shape[2]),
    )
    lattice = np.zeros(out_shape, dtype=np.float64)

    center_x = lattice.shape[0] // 2
    center_y = lattice.shape[1] // 2
    kernel_center_x = kernel.shape[0] // 2
    kernel_center_y = kernel.shape[1] // 2
    x_centers_idx: List[int] = []

    for off_x in offsets_x:
        beam_center_x = center_x + int(off_x) + int(x_shift_bins)
        x_centers_idx.append(int(beam_center_x))
        start_x = beam_center_x - kernel_center_x
        end_x = start_x + kernel.shape[0]
        for off_y in offsets_y:
            start_y = center_y + int(off_y) - kernel_center_y
            end_y = start_y + kernel.shape[1]
            lattice[start_x:end_x, start_y:end_y, :] += kernel

    return lattice, offsets_x, offsets_y, x_centers_idx


def build_pitch_case(
    normalized_single: np.ndarray,
    dx_cm: float,
    dy_cm: float,
    dz_cm: float,
    pitch_mm: float,
    n_beams_x: int,
    n_beams_y: int,
    prescribed_peak_dose_gy: float,
    uniform_dose_floor_fraction: float,
    target_depth_cm: float,
    alpha: float,
    beta: float,
    voxel_size_mm: tuple[float, float, float],
    vessel_radius_mm: float,
    vessel_center_offset_mm: tuple[float, float],
    uptake_rates_in_vessel: List[float],
    x_shift_fraction_of_pitch: float,
) -> Dict[str, object]:
    pitch_bins_x = int(round((float(pitch_mm) / 10.0) / dx_cm))
    pitch_bins_y = int(round((float(pitch_mm) / 10.0) / dy_cm))
    x_shift_bins = int(round(float(x_shift_fraction_of_pitch) * float(pitch_bins_x)))
    if x_shift_bins == 0:
        lattice_dose, offsets_x, _ = build_lattice(
            normalized_single,
            pitch_bins_x=pitch_bins_x,
            pitch_bins_y=pitch_bins_y,
            n_beams_x=int(n_beams_x),
            n_beams_y=int(n_beams_y),
        )
        center_x_idx = lattice_dose.shape[0] // 2
        x_centers_idx = [center_x_idx + off for off in offsets_x]
    else:
        lattice_dose, offsets_x, _, x_centers_idx = build_lattice_with_x_shift(
            normalized_single,
            pitch_bins_x=pitch_bins_x,
            pitch_bins_y=pitch_bins_y,
            n_beams_x=int(n_beams_x),
            n_beams_y=int(n_beams_y),
            x_shift_bins=int(x_shift_bins),
        )
    lattice_dose = lattice_dose.astype(np.float32)
    uniform_floor_gy = float(uniform_dose_floor_fraction) * float(prescribed_peak_dose_gy)
    if uniform_floor_gy > 0.0:
        lattice_dose = lattice_dose + np.float32(uniform_floor_gy)

    survival_lq = np.exp(-float(alpha) * lattice_dose - float(beta) * lattice_dose**2).astype(np.float32)
    z_cm = depth_axis_cm(lattice_dose.shape[2], dz_cm)
    target_idx = nearest_index(z_cm, float(target_depth_cm))
    center_x_idx = lattice_dose.shape[0] // 2
    center_y_idx = lattice_dose.shape[1] // 2
    if x_shift_bins == 0:
        x_centers_idx = [center_x_idx + off for off in offsets_x]
    valley_x_idx = center_x_idx if x_shift_bins != 0 else (
        (x_centers_idx[len(x_centers_idx) // 2] + x_centers_idx[(len(x_centers_idx) // 2) + 1]) // 2
    )

    uptake_tensor, vessel_mask = build_cylindrical_uptake_tensor(
        lattice_dose.shape,
        voxel_size_mm,
        num_species=2,
        vessel_radius_mm=float(vessel_radius_mm),
        vessel_center_offset_mm=vessel_center_offset_mm,
        uptake_rates_in_vessel=uptake_rates_in_vessel,
    )

    return {
        "pitch_mm": float(pitch_mm),
        "lattice_dose": lattice_dose,
        "survival_lq": survival_lq,
        "sampled_depth_cm": float(z_cm[target_idx]),
        "target_depth_idx": int(target_idx),
        "center_y_idx": int(center_y_idx),
        "valley_x_idx": int(valley_x_idx),
        "valley_dose_gy": float(lattice_dose[valley_x_idx, center_y_idx, target_idx]),
        "valley_survival_lq": float(survival_lq[valley_x_idx, center_y_idx, target_idx]),
        "uniform_dose_floor_gy": float(uniform_floor_gy),
        "uptake_tensor": uptake_tensor,
        "vessel_mask": vessel_mask,
        "x_shift_bins": int(x_shift_bins),
    }


def plot_sweep_figure(
    pitches_mm: List[float],
    standard_values: List[float],
    swept_lines: List[Tuple[str, List[float], str, str]],
    title: str,
    settings_text: str,
    out_file: Path,
    dpi: int,
) -> None:
    fig, ax = plt.subplots(figsize=(10.4, 6.0), constrained_layout=True)
    ax.plot(
        pitches_mm,
        standard_values,
        color="#111111",
        linestyle="--",
        linewidth=2.5,
        marker="o",
        label="Standard LQ",
    )
    for label, values, color, marker in swept_lines:
        ax.plot(
            pitches_mm,
            values,
            color=color,
            linewidth=2.2,
            marker=marker,
            label=label,
        )
    ax.set_xticks(pitches_mm)
    ax.set_xlabel("Lattice pitch (mm)")
    ax.set_ylabel("Valley survival at 5 cm")
    ax.set_ylim(0.0, 1.02)
    ax.grid(alpha=0.25)
    ax.set_title(title)
    ax.legend(loc="lower right")
    ax.text(
        0.02,
        0.02,
        settings_text,
        transform=ax.transAxes,
        fontsize=9,
        va="bottom",
        ha="left",
        bbox={"facecolor": "white", "alpha": 0.88, "edgecolor": "#b5b5b5"},
    )
    fig.savefig(out_file, dpi=dpi)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    phase2_summary = json.loads(args.phase2_summary.read_text(encoding="utf-8"))
    scaling_factor = float(phase2_summary["scaling_factor"])

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

    ros_diffusion = float(args.ros_diffusion_coeff)
    cytokine_diffusion = float(args.cytokine_diffusion_coeff)
    ros_decay = float(args.ros_decay_coeff)
    emission_emax = [float(args.ros_emission_emax), float(args.cytokine_emission_emax)]
    uptake_rates = [float(args.ros_vessel_uptake), float(args.cytokine_vessel_uptake)]
    species_weights = [float(value) for value in args.species_weights]
    vessel_center_offset_mm = (
        float(args.vessel_center_offset_x_mm),
        float(args.vessel_center_offset_y_mm),
    )
    tumor_center_offset_mm = (
        float(args.tumor_center_offset_x_mm),
        float(args.tumor_center_offset_y_mm),
        float(args.tumor_center_offset_z_mm),
    )
    hypoxic_center_offset_mm = tumor_center_offset_mm
    if args.hypoxic_depth_from_surface_mm is not None:
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

    cfl_dt_limit = cfl_stability_limit_3d(voxel_size_mm, max(ros_diffusion, cytokine_diffusion))
    if float(args.pde_dt) > cfl_dt_limit:
        raise ValueError(
            f"Chosen dt={float(args.pde_dt):.6f} exceeds the CFL stability limit {cfl_dt_limit:.6f}."
        )

    pitches_mm = [float(value) for value in args.pitches_mm]
    cytokine_k_values = [float(value) for value in args.cytokine_k_values]
    scaling_multipliers = [float(value) for value in args.scaling_multipliers]

    print("Building multi-species vessel-in-valley pitch cases...", flush=True)
    pitch_cases: Dict[float, Dict[str, object]] = {}
    for pitch_mm in pitches_mm:
        pitch_cases[pitch_mm] = build_pitch_case(
            normalized_single=normalized_single,
            dx_cm=dx_cm,
            dy_cm=dy_cm,
            dz_cm=dz_cm,
            pitch_mm=pitch_mm,
            n_beams_x=int(args.n_beams_x),
            n_beams_y=int(args.n_beams_y),
            prescribed_peak_dose_gy=float(args.prescribed_peak_dose_gy),
            uniform_dose_floor_fraction=float(args.uniform_dose_floor_fraction),
            target_depth_cm=float(args.target_depth_cm),
            alpha=float(args.alpha),
            beta=float(args.beta),
            voxel_size_mm=voxel_size_mm,
            vessel_radius_mm=float(args.vessel_radius_mm),
            vessel_center_offset_mm=vessel_center_offset_mm,
            uptake_rates_in_vessel=uptake_rates,
            x_shift_fraction_of_pitch=float(args.x_shift_fraction_of_pitch),
        )

    standard_values = [float(pitch_cases[pitch]["valley_survival_lq"]) for pitch in pitches_mm]
    sampled_depth_cm = float(next(iter(pitch_cases.values()))["sampled_depth_cm"])

    baseline_rows: List[Dict[str, object]] = []
    k_rows: List[Dict[str, object]] = []
    scaling_rows: List[Dict[str, object]] = []

    for pitch_mm in pitches_mm:
        case = pitch_cases[pitch_mm]
        baseline_rows.append(
            {
                "pitch_mm": float(pitch_mm),
                "sampled_depth_cm": float(case["sampled_depth_cm"]),
                "valley_dose_gy": float(case["valley_dose_gy"]),
                "valley_survival_lq": float(case["valley_survival_lq"]),
                "uniform_dose_floor_gy": float(case["uniform_dose_floor_gy"]),
            }
        )

    k_results: Dict[float, Dict[float, float]] = {value: {} for value in cytokine_k_values}
    scaling_results: Dict[float, Dict[float, float]] = {}
    baseline_toxicity_by_pitch: Dict[float, np.ndarray] = {}
    baseline_solver_meta_by_pitch: Dict[float, Dict[str, object]] = {}

    for cytokine_k in cytokine_k_values:
        lambda_eff_mm = effective_diffusion_length_mm(cytokine_diffusion, cytokine_k)
        print(
            f"Running multi-species sweep at cytokine k={cytokine_k:g} "
            f"(lambda_eff~{lambda_eff_mm:.2f} mm)...",
            flush=True,
        )
        for pitch_mm in pitches_mm:
            case = pitch_cases[pitch_mm]
            start_time = time.time()
            multispecies_tensor = solve_multispecies_pde_3d(
                case["lattice_dose"],
                voxel_size_mm,
                steps=int(args.pde_steps),
                dt=float(args.pde_dt),
                diffusion_coeffs=[ros_diffusion, cytokine_diffusion],
                decay_coeffs=[ros_decay, float(cytokine_k)],
                emission_emax=emission_emax,
                emission_gamma_per_gy=float(args.emission_gamma_per_gy),
                state_dependent_emission=bool(args.state_dependent_emission),
                tumor_radius_mm=float(args.tumor_radius_mm),
                tumor_center_offset_mm=tumor_center_offset_mm,
                tumor_cytokine_multiplier=float(args.tumor_cytokine_multiplier),
                hypoxic_radius_mm=float(args.hypoxic_radius_mm),
                hypoxic_center_offset_mm=hypoxic_center_offset_mm,
                hypoxic_ros_scale=float(args.hypoxic_ros_scale),
                hypoxic_cytokine_multiplier=float(args.hypoxic_cytokine_multiplier),
                uptake_tensor=case["uptake_tensor"],
                progress_interval=50,
                verbose=True,
            )
            runtime_sec = time.time() - start_time
            total_toxicity = combine_multispecies_toxicity(
                multispecies_tensor,
                species_weights=species_weights,
            ).astype(np.float32)
            valley_total_toxicity = float(
                total_toxicity[
                    int(case["valley_x_idx"]),
                    int(case["center_y_idx"]),
                    int(case["target_depth_idx"]),
                ]
            )
            valley_survival_total = float(case["valley_survival_lq"]) * float(
                np.exp(-valley_total_toxicity * scaling_factor)
            )
            k_results[cytokine_k][pitch_mm] = valley_survival_total
            k_rows.append(
                {
                    "pitch_mm": float(pitch_mm),
                    "ros_decay_coeff": float(ros_decay),
                    "cytokine_decay_coeff": float(cytokine_k),
                    "cytokine_effective_diffusion_length_mm": float(lambda_eff_mm),
                    "scaling_factor": float(scaling_factor),
                    "sampled_depth_cm": float(case["sampled_depth_cm"]),
                    "valley_dose_gy": float(case["valley_dose_gy"]),
                    "uniform_dose_floor_gy": float(case["uniform_dose_floor_gy"]),
                    "valley_survival_lq": float(case["valley_survival_lq"]),
                    "valley_total_toxicity": float(valley_total_toxicity),
                    "valley_survival_total": float(valley_survival_total),
                    "valley_survival_loss": float(case["valley_survival_lq"]) - float(valley_survival_total),
                    "solver_runtime_sec": float(runtime_sec),
                    "max_ros_concentration": float(np.max(multispecies_tensor[0])),
                    "max_cytokine_concentration": float(np.max(multispecies_tensor[1])),
                    "max_total_toxicity": float(np.max(total_toxicity)),
                }
            )
            if abs(cytokine_k - 0.05) < 1.0e-12:
                baseline_toxicity_by_pitch[pitch_mm] = total_toxicity
                baseline_solver_meta_by_pitch[pitch_mm] = {
                    "runtime_sec": float(runtime_sec),
                    "max_ros_concentration": float(np.max(multispecies_tensor[0])),
                    "max_cytokine_concentration": float(np.max(multispecies_tensor[1])),
                    "max_total_toxicity": float(np.max(total_toxicity)),
                }
            else:
                del total_toxicity
            del multispecies_tensor
            gc.collect()

    baseline_k = 0.05
    baseline_lambda_eff_mm = effective_diffusion_length_mm(cytokine_diffusion, baseline_k)
    for multiplier in scaling_multipliers:
        scaling_value = float(multiplier) * float(scaling_factor)
        scaling_results[scaling_value] = {}
        for pitch_mm in pitches_mm:
            case = pitch_cases[pitch_mm]
            total_toxicity = baseline_toxicity_by_pitch[pitch_mm]
            valley_total_toxicity = float(
                total_toxicity[
                    int(case["valley_x_idx"]),
                    int(case["center_y_idx"]),
                    int(case["target_depth_idx"]),
                ]
            )
            valley_survival_total = float(case["valley_survival_lq"]) * float(
                np.exp(-valley_total_toxicity * scaling_value)
            )
            scaling_results[scaling_value][pitch_mm] = valley_survival_total
            scaling_rows.append(
                {
                    "pitch_mm": float(pitch_mm),
                    "ros_decay_coeff": float(ros_decay),
                    "cytokine_decay_coeff": float(baseline_k),
                    "cytokine_effective_diffusion_length_mm": float(baseline_lambda_eff_mm),
                    "scaling_multiplier": float(multiplier),
                    "scaling_factor": float(scaling_value),
                    "sampled_depth_cm": float(case["sampled_depth_cm"]),
                    "valley_dose_gy": float(case["valley_dose_gy"]),
                    "uniform_dose_floor_gy": float(case["uniform_dose_floor_gy"]),
                    "valley_survival_lq": float(case["valley_survival_lq"]),
                    "valley_total_toxicity": float(valley_total_toxicity),
                    "valley_survival_total": float(valley_survival_total),
                    "valley_survival_loss": float(case["valley_survival_lq"]) - float(valley_survival_total),
                }
            )

    k_colors = ["#176087", "#2a9d8f", "#8ab17d", "#e9c46a", "#d65f4a"]
    k_markers = ["o", "s", "^", "D", "P"]
    k_lines = [
        (
            f"Multi-species, k1={k_value:g} (lambda_eff~{effective_diffusion_length_mm(cytokine_diffusion, k_value):.2f} mm)",
            [k_results[k_value][pitch] for pitch in pitches_mm],
            k_colors[index % len(k_colors)],
            k_markers[index % len(k_markers)],
        )
        for index, k_value in enumerate(cytokine_k_values)
    ]
    k_settings = "\n".join(
        [
            "Sweep: cytokine decay rate k1",
            f"ROS fixed: D0={ros_diffusion:.2f}, k0={ros_decay:.2f}",
            f"Cytokine fixed D1={cytokine_diffusion:.2f}, k1 values = {', '.join(f'{value:g}' for value in cytokine_k_values)}",
            f"Emission Emax = [{float(args.ros_emission_emax):.2f}, {float(args.cytokine_emission_emax):.2f}], gamma = {float(args.emission_gamma_per_gy):.2f} Gy^-1",
            (
                f"Emission mode = {'state-dependent' if bool(args.state_dependent_emission) else 'uniform'}, "
                f"tumor r = {float(args.tumor_radius_mm):.1f} mm, hypoxic r = {float(args.hypoxic_radius_mm):.1f} mm"
            ),
            f"Weights = [{species_weights[0]:.2f}, {species_weights[1]:.2f}], scale = {float(scaling_factor):.9f}",
            f"Lattice x-shift = {float(args.x_shift_fraction_of_pitch):.2f} pitch",
            (
                f"Vessel sink: center=({float(args.vessel_center_offset_x_mm):.1f}, "
                f"{float(args.vessel_center_offset_y_mm):.1f}) mm, r={float(args.vessel_radius_mm):.1f} mm"
            ),
            f"Transmission floor = {100.0 * float(args.uniform_dose_floor_fraction):.1f}% of peak",
            f"steps = {int(args.pde_steps)}, dt = {float(args.pde_dt):.2f}, CFL = {float(cfl_dt_limit):.3f}",
            f"Sampled depth = {float(sampled_depth_cm):.3f} cm",
        ]
    )
    figure7 = args.outdir / "figure7_multispecies_k_sweep_valley_survival.png"
    plot_sweep_figure(
        pitches_mm=pitches_mm,
        standard_values=standard_values,
        swept_lines=k_lines,
        title="Figure 7: Multi-species valley survival sensitivity to cytokine decay rate",
        settings_text=k_settings,
        out_file=figure7,
        dpi=int(args.dpi),
    )

    scaling_colors = ["#4c1d95", "#7c3aed", "#c026d3", "#ef4444", "#f59e0b"]
    scaling_markers = ["o", "s", "^", "D", "P"]
    sorted_scaling_values = sorted(scaling_results.keys())
    scaling_lines = [
        (
            f"Multi-species, scale={value:.6f} ({(value / scaling_factor):.2f}x)",
            [scaling_results[value][pitch] for pitch in pitches_mm],
            scaling_colors[index % len(scaling_colors)],
            scaling_markers[index % len(scaling_markers)],
        )
        for index, value in enumerate(sorted_scaling_values)
    ]
    scaling_settings = "\n".join(
        [
            "Sweep: inherited multi-species scaling factor",
            f"ROS/Cytokine D = [{ros_diffusion:.2f}, {cytokine_diffusion:.2f}]",
            f"k fixed = [{ros_decay:.2f}, {baseline_k:.2f}], scale multipliers = {', '.join(f'{value:g}x' for value in scaling_multipliers)}",
            f"Emission Emax = [{float(args.ros_emission_emax):.2f}, {float(args.cytokine_emission_emax):.2f}], gamma = {float(args.emission_gamma_per_gy):.2f} Gy^-1",
            (
                f"Emission mode = {'state-dependent' if bool(args.state_dependent_emission) else 'uniform'}, "
                f"tumor r = {float(args.tumor_radius_mm):.1f} mm, hypoxic r = {float(args.hypoxic_radius_mm):.1f} mm"
            ),
            f"Weights = [{species_weights[0]:.2f}, {species_weights[1]:.2f}], baseline scale = {float(scaling_factor):.9f}",
            f"Lattice x-shift = {float(args.x_shift_fraction_of_pitch):.2f} pitch",
            (
                f"Vessel sink: center=({float(args.vessel_center_offset_x_mm):.1f}, "
                f"{float(args.vessel_center_offset_y_mm):.1f}) mm, r={float(args.vessel_radius_mm):.1f} mm"
            ),
            f"Transmission floor = {100.0 * float(args.uniform_dose_floor_fraction):.1f}% of peak",
            f"steps = {int(args.pde_steps)}, dt = {float(args.pde_dt):.2f}, CFL = {float(cfl_dt_limit):.3f}",
            f"Sampled depth = {float(sampled_depth_cm):.3f} cm",
        ]
    )
    figure8 = args.outdir / "figure8_multispecies_scaling_sweep_valley_survival.png"
    plot_sweep_figure(
        pitches_mm=pitches_mm,
        standard_values=standard_values,
        swept_lines=scaling_lines,
        title="Figure 8: Multi-species valley survival sensitivity to inherited scaling",
        settings_text=scaling_settings,
        out_file=figure8,
        dpi=int(args.dpi),
    )

    baseline_csv = args.outdir / "phase3_multispecies_standard_lq_baseline.csv"
    k_csv = args.outdir / "phase3_multispecies_k_sweep.csv"
    scaling_csv = args.outdir / "phase3_multispecies_scaling_sweep.csv"
    write_csv(baseline_rows, baseline_csv)
    write_csv(k_rows, k_csv)
    write_csv(scaling_rows, scaling_csv)

    summary = {
        "input_csv": str(args.csv),
        "outdir": str(args.outdir),
        "normalization": norm_meta,
        "phase2_summary": str(args.phase2_summary),
        "scaling_factor": float(scaling_factor),
        "multispecies_model": {
            "species_names": ["ROS", "Cytokine"],
            "steps": int(args.pde_steps),
            "dt": float(args.pde_dt),
            "cfl_dt_limit": float(cfl_dt_limit),
            "diffusion_coeffs": [float(ros_diffusion), float(cytokine_diffusion)],
            "baseline_decay_coeffs": [float(ros_decay), float(baseline_k)],
            "emission_emax": emission_emax,
            "emission_gamma_per_gy": float(args.emission_gamma_per_gy),
            "species_weights": species_weights,
            "voxel_size_mm": [float(value) for value in voxel_size_mm],
        },
        "emission_model": {
            "type": "state_dependent" if bool(args.state_dependent_emission) else "uniform_saturated",
            "emission_emax": emission_emax,
            "emission_gamma_per_gy": float(args.emission_gamma_per_gy),
            "tumor_radius_mm": float(args.tumor_radius_mm),
            "tumor_center_offset_mm": [float(value) for value in tumor_center_offset_mm],
            "tumor_cytokine_multiplier": float(args.tumor_cytokine_multiplier),
            "hypoxic_radius_mm": float(args.hypoxic_radius_mm),
            "hypoxic_depth_from_surface_mm": (
                None
                if args.hypoxic_depth_from_surface_mm is None
                else float(args.hypoxic_depth_from_surface_mm)
            ),
            "hypoxic_center_offset_mm": [float(value) for value in hypoxic_center_offset_mm],
            "hypoxic_ros_scale": float(args.hypoxic_ros_scale),
            "hypoxic_cytokine_multiplier": float(args.hypoxic_cytokine_multiplier),
        },
        "vessel_model": {
            "radius_mm": float(args.vessel_radius_mm),
            "center_offset_mm": [
                float(args.vessel_center_offset_x_mm),
                float(args.vessel_center_offset_y_mm),
            ],
            "uptake_rates_in_vessel": uptake_rates,
        },
        "lattice_geometry": {
            "n_beams_x": int(args.n_beams_x),
            "n_beams_y": int(args.n_beams_y),
            "x_shift_fraction_of_pitch": float(args.x_shift_fraction_of_pitch),
        },
        "baseline": {
            "target_depth_cm_requested": float(args.target_depth_cm),
            "target_depth_cm_sampled": float(sampled_depth_cm),
            "rows_csv": str(baseline_csv),
        },
        "k_sweep": {
            "cytokine_k_values": [float(value) for value in cytokine_k_values],
            "rows_csv": str(k_csv),
            "figure": str(figure7),
            "solver_runtime_sec_by_pitch_at_baseline_k1_0p05": {
                str(int(pitch)): float(baseline_solver_meta_by_pitch[pitch]["runtime_sec"]) for pitch in pitches_mm
            },
        },
        "scaling_sweep": {
            "scaling_multipliers": [float(value) for value in scaling_multipliers],
            "baseline_cytokine_decay_coeff": float(baseline_k),
            "rows_csv": str(scaling_csv),
            "figure": str(figure8),
        },
    }
    summary_file = args.outdir / "phase3_multispecies_sweep_summary.json"
    summary_file.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Multi-species Phase 3 summary: {summary_file}")
    print(f"Multi-species Figure 7: {figure7}")
    print(f"Multi-species Figure 8: {figure8}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
