#!/usr/bin/env python
"""Generate PDE-calibrated Phase 2 bystander figures for the clinical-upgrade lattice set."""

from __future__ import annotations

import argparse
import csv
import gc
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from analyze_250mev_sfrt_plan import (
    build_lattice,
    centered_axis_cm,
    choose_reference_depth,
    depth_axis_cm,
    nearest_index,
    peak_valley_metrics,
    profile_from_center_strip,
    reference_peak_value,
)
from analyze_topas_outputs import load_topas_grid
from bystander_pde_solver import anisotropic_laplacian_3d, cfl_stability_limit_3d
from generate_phase2_bystander_figures import (
    build_emission_map,
    plot_biological_comparison_figure,
    plot_lateral_profile_figure,
)


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description=(
            "Load the clinical-upgrade TOPAS dose kernel, calibrate a PDE bystander model "
            "to the 40 mm benchmark, and write a new non-overwriting figure set."
        )
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=root / "runs" / "linac_6mv_polyenergetic_clinical_sfrt" / "case" / "dosedata.csv",
        help="TOPAS dose CSV for the polyenergetic clinical-upgrade kernel.",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=root / "runs" / "linac_6mv_polyenergetic_clinical_sfrt" / "analysis_phase2_pde_calibrated",
        help="Directory for PDE-calibrated figures and summaries.",
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
        help="Number of explicit PDE time steps.",
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
        help="Uniform diffusion coefficient in the PDE model.",
    )
    parser.add_argument(
        "--k-decay",
        type=float,
        default=0.05,
        help="Biological decay coefficient in the PDE model.",
    )
    parser.add_argument(
        "--anchor-pitch-mm",
        type=float,
        default=40.0,
        help="Pitch used to calibrate the PDE scaling factor.",
    )
    parser.add_argument(
        "--anchor-target-depth-cm",
        type=float,
        default=5.0,
        help="Depth sampled for the PDE calibration anchor.",
    )
    parser.add_argument(
        "--anchor-target-valley-survival",
        type=float,
        default=0.7697632312774658,
        help="Target multi-effect valley survival used to calibrate the PDE scaling factor.",
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


def solve_pde_with_history(
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


def write_csv(rows: List[Dict[str, object]], out_file: Path) -> None:
    if not rows:
        raise ValueError("No rows supplied for CSV output.")
    with out_file.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_markdown_report(
    args: argparse.Namespace,
    summary: Dict[str, object],
    out_file: Path,
) -> None:
    calibration = summary["calibration"]
    lines = [
        "# PDE-Calibrated Phase 2 Summary",
        "",
        (
            f"- Anchor pitch / depth: `{float(calibration['anchor_pitch_mm']):.1f} mm` at "
            f"`{float(calibration['sampled_depth_cm']):.3f} cm`."
        ),
        (
            f"- Anchor benchmark: LQ valley survival `{float(calibration['anchor_lq_valley_survival']):.6f}` "
            f"-> target multi-effect valley survival `{float(calibration['anchor_target_valley_survival']):.6f}`."
        ),
        (
            f"- PDE settings: `steps={int(summary['pde_model']['steps'])}`, "
            f"`dt={float(summary['pde_model']['dt']):.3f}`, "
            f"`D={float(summary['pde_model']['diffusion_coeff']):.3f}`, "
            f"`k={float(summary['pde_model']['k_decay']):.3f}`."
        ),
        (
            f"- CFL limit: `{float(summary['pde_model']['cfl_dt_limit']):.6f}`. "
            f"Chosen `dt` is within the stability bound."
        ),
        (
            f"- Calibrated PDE scaling factor: `{float(calibration['scaling_factor']):.9f}`."
        ),
        (
            f"- Anchor max concentration: `{float(calibration['max_concentration']):.6f}` in "
            f"`{float(calibration['runtime_sec']):.2f} s`."
        ),
        "",
        "| Pitch | Depth | Valley Survival (LQ) | Valley Survival (PDE) | Valley Concentration |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for pitch_key, pitch_summary in summary["generated_pitches"].items():
        depth_metrics = pitch_summary["metrics_by_depth"]["d=5cm"]
        lines.append(
            f"| {float(pitch_summary['pitch_mm']):.0f} mm | {float(depth_metrics['sampled_depth_cm']):.3f} cm | "
            f"{float(depth_metrics['valley_survival_lq']):.3f} | {float(depth_metrics['valley_survival_total']):.3f} | "
            f"{float(depth_metrics['valley_concentration']):.3f} |"
        )
    write_text = "\n".join(lines) + "\n"
    out_file.write_text(write_text, encoding="utf-8")


def main() -> int:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

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

    colors = {
        "dmax": "#0b3c5d",
        "d=3cm": "#c0392b",
        "d=5cm": "#27ae60",
    }
    linestyles = {
        "dmax": "-",
        "d=3cm": "--",
        "d=5cm": ":",
    }
    legend_labels = {
        "dmax": "d_max",
        "d=3cm": "d = 3 cm",
        "d=5cm": "d = 5 cm",
    }

    requested_depths_cm = [float(v) for v in args.depths_cm]
    pitch_results: Dict[float, Dict[str, object]] = {}

    for pitch_mm in [float(v) for v in args.pitches_mm]:
        pitch_bins_x = int(round((pitch_mm / 10.0) / dx_cm))
        pitch_bins_y = int(round((pitch_mm / 10.0) / dy_cm))
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

        z_cm = depth_axis_cm(lattice_dose.shape[2], dz_cm)
        x_cm = centered_axis_cm(lattice_dose.shape[0], dx_cm)
        depth_profile = np.sum(lattice_dose, axis=(0, 1))
        dmax_idx = int(np.argmax(depth_profile))
        idx_3 = nearest_index(z_cm, requested_depths_cm[0])
        idx_5 = nearest_index(z_cm, requested_depths_cm[1])
        depth_indices = {"dmax": dmax_idx, "d=3cm": idx_3, "d=5cm": idx_5}

        center_x_idx = lattice_dose.shape[0] // 2
        center_y_idx = lattice_dose.shape[1] // 2
        x_centers_idx = [center_x_idx + off for off in offsets_x]
        peak_x_idx = x_centers_idx[len(x_centers_idx) // 2]
        valley_x_idx = (x_centers_idx[len(x_centers_idx) // 2] + x_centers_idx[(len(x_centers_idx) // 2) + 1]) // 2

        survival_lq = np.exp(-float(args.alpha) * lattice_dose - float(args.beta) * lattice_dose**2).astype(np.float32)
        profiles = {
            label: profile_from_center_strip(lattice_dose, idx, int(args.profile_half_width_bins))
            for label, idx in depth_indices.items()
        }

        concentration, solver_meta = solve_pde_with_history(
            dose_grid=lattice_dose,
            voxel_size_mm=voxel_size_mm,
            steps=int(args.pde_steps),
            dt=float(args.pde_dt),
            diffusion_coeff=float(args.diffusion_coeff),
            k_decay=float(args.k_decay),
            emission_threshold_gy=float(args.emission_threshold_gy),
            continuous_emission=bool(args.continuous_emission),
            emission_emax=float(args.emission_emax),
            emission_gamma_per_gy=float(args.emission_gamma_per_gy),
        )

        pitch_results[pitch_mm] = {
            "pitch_mm": float(pitch_mm),
            "lattice_dose": lattice_dose,
            "survival_lq": survival_lq,
            "concentration": concentration,
            "solver_meta": solver_meta,
            "x_cm": x_cm,
            "z_cm": z_cm,
            "depth_indices": depth_indices,
            "profiles": profiles,
            "x_centers_idx": x_centers_idx,
            "peak_x_idx": peak_x_idx,
            "valley_x_idx": valley_x_idx,
            "center_y_idx": center_y_idx,
        }

    anchor_pitch = float(args.anchor_pitch_mm)
    anchor_result = pitch_results[anchor_pitch]
    anchor_depth_idx = anchor_result["depth_indices"]["d=5cm"]
    anchor_valley_c = float(
        anchor_result["concentration"][
            anchor_result["valley_x_idx"],
            anchor_result["center_y_idx"],
            anchor_depth_idx,
        ]
    )
    anchor_lq_valley = float(
        anchor_result["survival_lq"][
            anchor_result["valley_x_idx"],
            anchor_result["center_y_idx"],
            anchor_depth_idx,
        ]
    )
    scaling_factor = -np.log(float(args.anchor_target_valley_survival) / anchor_lq_valley) / anchor_valley_c

    metrics_rows: List[Dict[str, object]] = []
    summary: Dict[str, object] = {
        "input_csv": str(args.csv),
        "outdir": str(args.outdir),
        "normalization": norm_meta,
        "pde_model": {
            "steps": int(args.pde_steps),
            "dt": float(args.pde_dt),
            "diffusion_coeff": float(args.diffusion_coeff),
            "k_decay": float(args.k_decay),
            "cfl_dt_limit": float(cfl_dt_limit),
            "voxel_size_mm": [float(v) for v in voxel_size_mm],
        },
        "emission_model": {
            "type": "continuous_saturated" if bool(args.continuous_emission) else "binary_threshold",
            "emission_threshold_gy": float(args.emission_threshold_gy),
            "emission_emax": float(args.emission_emax),
            "emission_gamma_per_gy": float(args.emission_gamma_per_gy),
        },
        "physical_floor_model": {
            "uniform_dose_floor_fraction_of_peak": float(args.uniform_dose_floor_fraction),
            "uniform_dose_floor_gy": float(args.uniform_dose_floor_fraction) * float(args.prescribed_peak_dose_gy),
        },
        "calibration": {
            "anchor_pitch_mm": float(anchor_pitch),
            "anchor_target_valley_survival": float(args.anchor_target_valley_survival),
            "anchor_lq_valley_survival": float(anchor_lq_valley),
            "sampled_depth_cm": float(anchor_result["z_cm"][anchor_depth_idx]),
            "valley_concentration": float(anchor_valley_c),
            "max_concentration": float(anchor_result["solver_meta"]["max_concentration"]),
            "runtime_sec": float(anchor_result["solver_meta"]["runtime_sec"]),
            "max_history_last10": anchor_result["solver_meta"]["max_history_last10"],
            "max_history_delta_last10": anchor_result["solver_meta"]["max_history_delta_last10"],
            "scaling_factor": float(scaling_factor),
        },
        "generated_pitches": {},
        "outputs": {},
    }

    for pitch_index, pitch_mm in enumerate([float(v) for v in args.pitches_mm], start=1):
        pitch_result = pitch_results[pitch_mm]
        pitch_label = f"{pitch_mm:g}"
        survival_total = (
            pitch_result["survival_lq"] * np.exp(-pitch_result["concentration"] * float(scaling_factor))
        ).astype(np.float32)
        survival_loss = (pitch_result["survival_lq"] - survival_total).astype(np.float32)

        guide_lines = [
            (
                float(pitch_result["z_cm"][pitch_result["depth_indices"]["dmax"]]),
                legend_labels["dmax"],
                colors["dmax"],
                linestyles["dmax"],
            ),
            (
                float(pitch_result["z_cm"][pitch_result["depth_indices"]["d=3cm"]]),
                legend_labels["d=3cm"],
                colors["d=3cm"],
                linestyles["d=3cm"],
            ),
            (
                float(pitch_result["z_cm"][pitch_result["depth_indices"]["d=5cm"]]),
                legend_labels["d=5cm"],
                colors["d=5cm"],
                linestyles["d=5cm"],
            ),
        ]

        physical_figure_number = (2 * pitch_index) - 1
        bio_figure_number = 2 * pitch_index
        physical_file = args.outdir / f"figure{physical_figure_number}_physical_lateral_profile_pitch{pitch_label}.png"
        bio_file = args.outdir / f"figure{bio_figure_number}_pde_biological_comparison_pitch{pitch_label}.png"

        plot_lateral_profile_figure(
            x_cm=pitch_result["x_cm"],
            profiles=[
                (pitch_result["profiles"]["dmax"], legend_labels["dmax"], colors["dmax"], linestyles["dmax"]),
                (pitch_result["profiles"]["d=3cm"], legend_labels["d=3cm"], colors["d=3cm"], linestyles["d=3cm"]),
                (pitch_result["profiles"]["d=5cm"], legend_labels["d=5cm"], colors["d=5cm"], linestyles["d=5cm"]),
            ],
            title=f"Figure {physical_figure_number}: Physical lateral dose profiles (pitch = {pitch_label} mm)",
            out_file=physical_file,
            dpi=int(args.dpi),
        )
        plot_biological_comparison_figure(
            x_cm=pitch_result["x_cm"],
            z_cm=pitch_result["z_cm"],
            survival_lq=pitch_result["survival_lq"],
            survival_total=survival_total,
            survival_loss=survival_loss,
            guide_lines=guide_lines,
            title=f"Figure {bio_figure_number}: PDE-calibrated biological comparison (pitch = {pitch_label} mm)",
            out_file=bio_file,
            dpi=int(args.dpi),
        )

        pitch_summary: Dict[str, object] = {
            "pitch_mm": float(pitch_mm),
            "physical_figure": str(physical_file),
            "biological_figure": str(bio_file),
            "sampled_depths_cm": {},
            "solver_runtime_sec": float(pitch_result["solver_meta"]["runtime_sec"]),
            "max_concentration": float(pitch_result["solver_meta"]["max_concentration"]),
            "metrics_by_depth": {},
        }

        for label, idx in pitch_result["depth_indices"].items():
            pvdr = peak_valley_metrics(pitch_result["profiles"][label], pitch_result["x_centers_idx"], int(round((pitch_mm / 10.0) / dx_cm)))
            row = {
                "pitch_mm": float(pitch_mm),
                "depth_label": label,
                "sampled_depth_cm": float(pitch_result["z_cm"][idx]),
                "pvdr": float(pvdr["pvdr"]),
                "mean_peak_gy": float(pvdr["mean_peak_gy"]),
                "mean_valley_gy": float(pvdr["mean_valley_gy"]),
                "peak_dose_gy": float(pitch_result["lattice_dose"][pitch_result["peak_x_idx"], pitch_result["center_y_idx"], idx]),
                "valley_dose_gy": float(pitch_result["lattice_dose"][pitch_result["valley_x_idx"], pitch_result["center_y_idx"], idx]),
                "peak_survival_lq": float(pitch_result["survival_lq"][pitch_result["peak_x_idx"], pitch_result["center_y_idx"], idx]),
                "valley_survival_lq": float(pitch_result["survival_lq"][pitch_result["valley_x_idx"], pitch_result["center_y_idx"], idx]),
                "peak_concentration": float(pitch_result["concentration"][pitch_result["peak_x_idx"], pitch_result["center_y_idx"], idx]),
                "valley_concentration": float(pitch_result["concentration"][pitch_result["valley_x_idx"], pitch_result["center_y_idx"], idx]),
                "peak_survival_total": float(survival_total[pitch_result["peak_x_idx"], pitch_result["center_y_idx"], idx]),
                "valley_survival_total": float(survival_total[pitch_result["valley_x_idx"], pitch_result["center_y_idx"], idx]),
                "valley_survival_loss": float(survival_loss[pitch_result["valley_x_idx"], pitch_result["center_y_idx"], idx]),
            }
            metrics_rows.append(row)
            pitch_summary["sampled_depths_cm"][label] = float(pitch_result["z_cm"][idx])
            pitch_summary["metrics_by_depth"][label] = row

        summary["generated_pitches"][f"pitch_{pitch_label}mm"] = pitch_summary
        summary["outputs"][f"figure{physical_figure_number}"] = str(physical_file)
        summary["outputs"][f"figure{bio_figure_number}"] = str(bio_file)

        del survival_total
        del survival_loss
        gc.collect()

    metrics_csv = args.outdir / "phase2_pde_metrics.csv"
    write_csv(metrics_rows, metrics_csv)
    summary["metrics_csv"] = str(metrics_csv)

    summary_file = args.outdir / "phase2_pde_summary.json"
    summary_file.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    report_file = args.outdir / "phase2_pde_summary.md"
    write_markdown_report(args, summary, report_file)

    print(f"PDE Phase 2 summary: {summary_file}")
    print(f"PDE Phase 2 metrics: {metrics_csv}")
    print(f"PDE Phase 2 report: {report_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
