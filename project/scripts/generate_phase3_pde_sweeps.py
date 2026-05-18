#!/usr/bin/env python
"""Generate PDE-based Phase 3 sensitivity sweeps from the clinical-upgrade lattice set."""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import time
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np

from analyze_250mev_sfrt_plan import build_lattice, choose_reference_depth, depth_axis_cm, nearest_index, reference_peak_value
from analyze_topas_outputs import load_topas_grid
from bystander_pde_solver import anisotropic_laplacian_3d, cfl_stability_limit_3d
from generate_phase2_bystander_figures import build_emission_map

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
            "Sweep PDE decay rate k for spatial reach and sweep the calibrated survival "
            "scaling factor for toxicity amplitude."
        )
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=root / "runs" / "linac_6mv_polyenergetic_clinical_sfrt" / "case" / "dosedata.csv",
        help="TOPAS dose CSV for the polyenergetic clinical-upgrade kernel.",
    )
    parser.add_argument(
        "--calibration-summary",
        type=Path,
        default=root / "runs" / "linac_6mv_polyenergetic_clinical_sfrt" / "analysis_phase2_pde_calibrated" / "phase2_pde_summary.json",
        help="PDE calibration summary JSON containing the calibrated scaling factor.",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=root / "runs" / "linac_6mv_polyenergetic_clinical_sfrt" / "analysis_phase3_pde_sweeps",
        help="Directory for PDE-based sweep figures and summaries.",
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
    parser.add_argument(
        "--continuous-emission",
        action="store_true",
        default=True,
        help="Use a saturated continuous emission model instead of a binary threshold.",
    )
    parser.add_argument(
        "--emission-threshold-gy",
        type=float,
        default=5.0,
        help="Compatibility threshold used only if binary emission is requested.",
    )
    parser.add_argument("--emission-emax", type=float, default=1.0, help="Maximum emission strength.")
    parser.add_argument(
        "--emission-gamma-per-gy",
        type=float,
        default=0.35,
        help="Dose-response constant gamma for E(D) = Emax * (1 - exp(-gamma * D)).",
    )
    parser.add_argument("--alpha", type=float, default=0.10, help="LQ alpha in Gy^-1.")
    parser.add_argument("--beta", type=float, default=0.05, help="LQ beta in Gy^-2.")
    parser.add_argument(
        "--pde-steps",
        type=int,
        default=300,
        help="Number of explicit PDE time steps per solve.",
    )
    parser.add_argument(
        "--pde-dt",
        type=float,
        default=0.30,
        help="Explicit Euler time step. Must respect the CFL limit.",
    )
    parser.add_argument(
        "--diffusion-coeff",
        type=float,
        default=0.5,
        help="Uniform diffusion coefficient D held fixed during the k sweep.",
    )
    parser.add_argument(
        "--k-values",
        nargs="+",
        type=float,
        default=[0.5, 0.125, 0.05, 0.02, 0.005],
        help="Decay-rate sweep values for Figure 7.",
    )
    parser.add_argument(
        "--scaling-multipliers",
        nargs="+",
        type=float,
        default=[0.5, 0.75, 1.0, 1.25, 1.5],
        help="Multipliers applied to the calibrated PDE scaling factor for Figure 8.",
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


def sci_label(value: float) -> str:
    mantissa, exponent = f"{float(value):.2e}".split("e")
    mantissa = mantissa.rstrip("0").rstrip(".")
    exponent = int(exponent)
    return f"{mantissa}e{exponent}"


def effective_diffusion_length_mm(diffusion_coeff: float, k_decay: float) -> float:
    return math.sqrt(float(diffusion_coeff) / float(k_decay))


def solve_pde_concentration(
    dose_grid: np.ndarray,
    voxel_size_mm: Tuple[float, float, float],
    *,
    steps: int,
    dt: float,
    diffusion_coeff: float,
    k_decay: float,
    emission_threshold_gy: float,
    continuous_emission: bool,
    emission_emax: float,
    emission_gamma_per_gy: float,
) -> Tuple[np.ndarray, Dict[str, object]]:
    source_term = build_emission_map(
        lattice_dose=dose_grid,
        emission_threshold_gy=float(emission_threshold_gy),
        continuous_emission=bool(continuous_emission),
        emission_emax=float(emission_emax),
        emission_gamma_per_gy=float(emission_gamma_per_gy),
    ).astype(np.float32)

    concentration = np.zeros_like(dose_grid, dtype=np.float32)
    max_history: List[float] = []
    start = time.time()
    for _ in range(int(steps)):
        laplacian = anisotropic_laplacian_3d(concentration, voxel_size_mm)
        dcdt = (float(diffusion_coeff) * laplacian) + source_term - (float(k_decay) * concentration)
        concentration += dcdt * float(dt)
        max_history.append(float(np.max(concentration)))

    runtime_sec = time.time() - start
    tail = max_history[-10:]
    tail_delta = [tail[i + 1] - tail[i] for i in range(len(tail) - 1)] if len(tail) > 1 else []
    return concentration, {
        "runtime_sec": float(runtime_sec),
        "max_concentration": float(np.max(concentration)),
        "max_history_last10": [float(value) for value in tail],
        "max_history_delta_last10": [float(value) for value in tail_delta],
    }


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
) -> Dict[str, object]:
    pitch_bins_x = int(round((float(pitch_mm) / 10.0) / dx_cm))
    pitch_bins_y = int(round((float(pitch_mm) / 10.0) / dy_cm))
    lattice_dose, offsets_x, _ = build_lattice(
        normalized_single,
        pitch_bins_x=pitch_bins_x,
        pitch_bins_y=pitch_bins_y,
        n_beams_x=int(n_beams_x),
        n_beams_y=int(n_beams_y),
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
    x_centers_idx = [center_x_idx + off for off in offsets_x]
    valley_x_idx = (x_centers_idx[len(x_centers_idx) // 2] + x_centers_idx[(len(x_centers_idx) // 2) + 1]) // 2

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


def write_csv(rows: List[Dict[str, object]], out_file: Path) -> None:
    if not rows:
        raise ValueError("No rows supplied for CSV output.")
    with out_file.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    calibration_summary = json.loads(args.calibration_summary.read_text(encoding="utf-8"))
    calibrated_scaling = float(calibration_summary["calibration"]["scaling_factor"])

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
    cfl_dt_limit = cfl_stability_limit_3d(voxel_size_mm, float(args.diffusion_coeff))
    if float(args.pde_dt) > cfl_dt_limit:
        raise ValueError(
            f"Chosen dt={float(args.pde_dt):.6f} exceeds the CFL stability limit {cfl_dt_limit:.6f}."
        )

    pitches_mm = [float(v) for v in args.pitches_mm]
    k_values = [float(v) for v in args.k_values]
    scaling_multipliers = [float(v) for v in args.scaling_multipliers]

    print("Building PDE sweep pitch cases...", flush=True)
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
        )

    standard_values = [float(pitch_cases[pitch]["valley_survival_lq"]) for pitch in pitches_mm]
    sampled_depth_cm = float(next(iter(pitch_cases.values()))["sampled_depth_cm"])

    k_rows: List[Dict[str, object]] = []
    scaling_rows: List[Dict[str, object]] = []
    baseline_rows: List[Dict[str, object]] = []

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

    k_results: Dict[float, Dict[float, float]] = {value: {} for value in k_values}
    scaling_results: Dict[float, Dict[float, float]] = {}
    baseline_concentration_by_pitch: Dict[float, np.ndarray] = {}
    baseline_solver_meta_by_pitch: Dict[float, Dict[str, object]] = {}

    for k_decay in k_values:
        lambda_eff_mm = effective_diffusion_length_mm(float(args.diffusion_coeff), k_decay)
        print(f"Running k sweep at k={k_decay:g} (lambda_eff~{lambda_eff_mm:.2f} mm)...", flush=True)
        for pitch_mm in pitches_mm:
            case = pitch_cases[pitch_mm]
            concentration, solver_meta = solve_pde_concentration(
                dose_grid=case["lattice_dose"],
                voxel_size_mm=voxel_size_mm,
                steps=int(args.pde_steps),
                dt=float(args.pde_dt),
                diffusion_coeff=float(args.diffusion_coeff),
                k_decay=float(k_decay),
                emission_threshold_gy=float(args.emission_threshold_gy),
                continuous_emission=bool(args.continuous_emission),
                emission_emax=float(args.emission_emax),
                emission_gamma_per_gy=float(args.emission_gamma_per_gy),
            )
            valley_concentration = float(
                concentration[
                    int(case["valley_x_idx"]),
                    int(case["center_y_idx"]),
                    int(case["target_depth_idx"]),
                ]
            )
            valley_survival_total = float(case["valley_survival_lq"]) * float(
                np.exp(-valley_concentration * calibrated_scaling)
            )
            k_results[k_decay][pitch_mm] = valley_survival_total
            k_rows.append(
                {
                    "pitch_mm": float(pitch_mm),
                    "k_decay": float(k_decay),
                    "effective_diffusion_length_mm": float(lambda_eff_mm),
                    "scaling_factor": float(calibrated_scaling),
                    "sampled_depth_cm": float(case["sampled_depth_cm"]),
                    "valley_dose_gy": float(case["valley_dose_gy"]),
                    "uniform_dose_floor_gy": float(case["uniform_dose_floor_gy"]),
                    "valley_survival_lq": float(case["valley_survival_lq"]),
                    "valley_concentration": float(valley_concentration),
                    "valley_survival_total": float(valley_survival_total),
                    "valley_survival_loss": float(case["valley_survival_lq"]) - float(valley_survival_total),
                    "solver_runtime_sec": float(solver_meta["runtime_sec"]),
                    "max_concentration": float(solver_meta["max_concentration"]),
                }
            )
            if abs(k_decay - 0.05) < 1.0e-12:
                baseline_concentration_by_pitch[pitch_mm] = concentration
                baseline_solver_meta_by_pitch[pitch_mm] = solver_meta
            else:
                del concentration
                gc.collect()

    for multiplier in scaling_multipliers:
        scaling_value = float(multiplier) * float(calibrated_scaling)
        scaling_results[scaling_value] = {}
        for pitch_mm in pitches_mm:
            case = pitch_cases[pitch_mm]
            concentration = baseline_concentration_by_pitch[pitch_mm]
            valley_concentration = float(
                concentration[
                    int(case["valley_x_idx"]),
                    int(case["center_y_idx"]),
                    int(case["target_depth_idx"]),
                ]
            )
            valley_survival_total = float(case["valley_survival_lq"]) * float(np.exp(-valley_concentration * scaling_value))
            scaling_results[scaling_value][pitch_mm] = valley_survival_total
            scaling_rows.append(
                {
                    "pitch_mm": float(pitch_mm),
                    "k_decay": 0.05,
                    "effective_diffusion_length_mm": float(effective_diffusion_length_mm(float(args.diffusion_coeff), 0.05)),
                    "scaling_multiplier": float(multiplier),
                    "scaling_factor": float(scaling_value),
                    "sampled_depth_cm": float(case["sampled_depth_cm"]),
                    "valley_dose_gy": float(case["valley_dose_gy"]),
                    "uniform_dose_floor_gy": float(case["uniform_dose_floor_gy"]),
                    "valley_survival_lq": float(case["valley_survival_lq"]),
                    "valley_concentration": float(valley_concentration),
                    "valley_survival_total": float(valley_survival_total),
                    "valley_survival_loss": float(case["valley_survival_lq"]) - float(valley_survival_total),
                }
            )

    k_colors = ["#176087", "#2a9d8f", "#8ab17d", "#e9c46a", "#d65f4a"]
    k_markers = ["o", "s", "^", "D", "P"]
    k_lines = [
        (
            f"PDE, k={k_value:g} (lambda_eff~{effective_diffusion_length_mm(float(args.diffusion_coeff), k_value):.2f} mm)",
            [k_results[k_value][pitch] for pitch in pitches_mm],
            k_colors[index % len(k_colors)],
            k_markers[index % len(k_markers)],
        )
        for index, k_value in enumerate(k_values)
    ]
    k_settings = "\n".join(
        [
            "Sweep: decay rate k",
            f"D fixed = {float(args.diffusion_coeff):.3f}",
            f"k values = {', '.join(f'{value:g}' for value in k_values)}",
            f"calibrated scaling = {float(calibrated_scaling):.9f}",
            f"steps = {int(args.pde_steps)}, dt = {float(args.pde_dt):.2f}, CFL = {float(cfl_dt_limit):.3f}",
            f"Emission: E(D)=Emax*(1-exp(-gamma*D)), Emax={float(args.emission_emax):.2f}, gamma={float(args.emission_gamma_per_gy):.2f} Gy^-1",
            f"Transmission floor = {100.0 * float(args.uniform_dose_floor_fraction):.1f}% of peak",
            f"Sampled depth = {float(sampled_depth_cm):.3f} cm",
            f"Pitches = {', '.join(f'{pitch:g}' for pitch in pitches_mm)} mm",
        ]
    )
    figure7 = args.outdir / "figure7_pde_k_sweep_valley_survival.png"
    plot_sweep_figure(
        pitches_mm=pitches_mm,
        standard_values=standard_values,
        swept_lines=k_lines,
        title="Figure 7: PDE valley survival sensitivity to cytokine decay rate",
        settings_text=k_settings,
        out_file=figure7,
        dpi=int(args.dpi),
    )

    scaling_colors = ["#4c1d95", "#7c3aed", "#c026d3", "#ef4444", "#f59e0b"]
    scaling_markers = ["o", "s", "^", "D", "P"]
    sorted_scaling_values = sorted(scaling_results.keys())
    scaling_lines = [
        (
            f"PDE, scale={value:.6f} ({(value / calibrated_scaling):.2f}x)",
            [scaling_results[value][pitch] for pitch in pitches_mm],
            scaling_colors[index % len(scaling_colors)],
            scaling_markers[index % len(scaling_markers)],
        )
        for index, value in enumerate(sorted_scaling_values)
    ]
    scaling_settings = "\n".join(
        [
            "Sweep: calibrated scaling factor",
            f"D fixed = {float(args.diffusion_coeff):.3f}, k fixed = 0.050",
            f"baseline scale = {float(calibrated_scaling):.9f}",
            f"multipliers = {', '.join(f'{value:g}x' for value in scaling_multipliers)}",
            f"steps = {int(args.pde_steps)}, dt = {float(args.pde_dt):.2f}, CFL = {float(cfl_dt_limit):.3f}",
            f"Emission: E(D)=Emax*(1-exp(-gamma*D)), Emax={float(args.emission_emax):.2f}, gamma={float(args.emission_gamma_per_gy):.2f} Gy^-1",
            f"Transmission floor = {100.0 * float(args.uniform_dose_floor_fraction):.1f}% of peak",
            f"Sampled depth = {float(sampled_depth_cm):.3f} cm",
            f"Pitches = {', '.join(f'{pitch:g}' for pitch in pitches_mm)} mm",
        ]
    )
    figure8 = args.outdir / "figure8_pde_scaling_sweep_valley_survival.png"
    plot_sweep_figure(
        pitches_mm=pitches_mm,
        standard_values=standard_values,
        swept_lines=scaling_lines,
        title="Figure 8: PDE valley survival sensitivity to calibrated toxicity scaling",
        settings_text=scaling_settings,
        out_file=figure8,
        dpi=int(args.dpi),
    )

    baseline_csv = args.outdir / "phase3_pde_standard_lq_baseline.csv"
    k_csv = args.outdir / "phase3_pde_k_sweep.csv"
    scaling_csv = args.outdir / "phase3_pde_scaling_sweep.csv"
    write_csv(baseline_rows, baseline_csv)
    write_csv(k_rows, k_csv)
    write_csv(scaling_rows, scaling_csv)

    summary = {
        "input_csv": str(args.csv),
        "outdir": str(args.outdir),
        "normalization": norm_meta,
        "calibration_summary": str(args.calibration_summary),
        "calibrated_scaling_factor": float(calibrated_scaling),
        "pde_model": {
            "steps": int(args.pde_steps),
            "dt": float(args.pde_dt),
            "diffusion_coeff": float(args.diffusion_coeff),
            "cfl_dt_limit": float(cfl_dt_limit),
            "voxel_size_mm": [float(v) for v in voxel_size_mm],
        },
        "emission_model": {
            "type": "continuous_saturated" if bool(args.continuous_emission) else "binary_threshold",
            "emission_threshold_gy": float(args.emission_threshold_gy),
            "emission_emax": float(args.emission_emax),
            "emission_gamma_per_gy": float(args.emission_gamma_per_gy),
        },
        "baseline": {
            "target_depth_cm_requested": float(args.target_depth_cm),
            "target_depth_cm_sampled": float(sampled_depth_cm),
            "rows_csv": str(baseline_csv),
        },
        "k_sweep": {
            "k_values": [float(v) for v in k_values],
            "rows_csv": str(k_csv),
            "figure": str(figure7),
            "solver_runtime_sec_by_pitch_at_k_0p05": {
                str(int(pitch)): float(baseline_solver_meta_by_pitch[pitch]["runtime_sec"]) for pitch in pitches_mm
            },
        },
        "scaling_sweep": {
            "scaling_multipliers": [float(v) for v in scaling_multipliers],
            "baseline_k_decay": 0.05,
            "rows_csv": str(scaling_csv),
            "figure": str(figure8),
        },
    }
    summary_file = args.outdir / "phase3_pde_sweep_summary.json"
    summary_file.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"PDE Phase 3 summary: {summary_file}")
    print(f"PDE Figure 7: {figure7}")
    print(f"PDE Figure 8: {figure8}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
