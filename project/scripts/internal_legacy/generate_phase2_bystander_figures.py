#!/usr/bin/env python
"""Generate Phase 2 bystander-effect figures for the direct-photon SFRT lattices."""

from __future__ import annotations

import argparse
import csv
import gc
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from scipy.signal import fftconvolve

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception as exc:  # pragma: no cover
    raise RuntimeError("matplotlib is required for figure generation") from exc

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


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description=(
            "Load the direct-photon TOPAS dose kernel, synthesize 20/30/40 mm lattices, "
            "apply a bystander FFT model, and write a new Phase 2 figure set."
        )
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=root / "runs" / "linac_6mv_direct_photon_sfrt" / "case" / "dosedata.csv",
        help="TOPAS dose CSV for the direct-photon kernel.",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=root / "runs" / "linac_6mv_direct_photon_sfrt" / "analysis_phase2_bystander",
        help="Directory for Phase 2 figures and summaries.",
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
        "--emission-threshold-gy",
        type=float,
        default=5.0,
        help="Dose threshold above which voxels emit bystander signals.",
    )
    parser.add_argument(
        "--continuous-emission",
        action="store_true",
        help="Use a saturated continuous emission model instead of a binary threshold.",
    )
    parser.add_argument(
        "--emission-emax",
        type=float,
        default=1.0,
        help="Maximum emission strength used by the continuous emission model.",
    )
    parser.add_argument(
        "--emission-gamma-per-gy",
        type=float,
        default=0.35,
        help="Dose-response constant gamma for E(D) = Emax * (1 - exp(-gamma * D)).",
    )
    parser.add_argument(
        "--diffusion-length-mm",
        type=float,
        default=10.0,
        help="Exponential bystander diffusion length in mm.",
    )
    parser.add_argument(
        "--signal-strength",
        type=float,
        default=1.0e-4,
        help="Amplitude A in the bystander kernel A * exp(-r / lambda).",
    )
    parser.add_argument(
        "--kernel-radius-lambda",
        type=float,
        default=4.0,
        help="Kernel truncation radius expressed in multiples of lambda.",
    )
    parser.add_argument(
        "--uniform-dose-floor-fraction",
        type=float,
        default=0.0,
        help="Uniform dose floor added everywhere as a fraction of the prescribed peak dose.",
    )
    parser.add_argument("--alpha", type=float, default=0.10, help="LQ alpha in Gy^-1.")
    parser.add_argument("--beta", type=float, default=0.05, help="LQ beta in Gy^-2.")
    parser.add_argument(
        "--high-dose-methodology-note",
        type=str,
        default=(
            "Numerical survival still uses the standard LQ model in this script. "
            "For peak-dose regions above about 10 Gy, an LQL or USC replacement is recommended "
            "for manuscript methodology and future peak-region refinement."
        ),
        help="Methodology note recorded in the summary for high-dose survival modeling.",
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


def build_diffusion_kernel(
    dx_cm: float,
    dy_cm: float,
    dz_cm: float,
    diffusion_length_mm: float,
    signal_strength: float,
    kernel_radius_lambda: float,
) -> np.ndarray:
    lambda_cm = float(diffusion_length_mm) / 10.0
    radius_cm = max(float(kernel_radius_lambda), 1.0) * lambda_cm
    radius_x = max(1, int(np.ceil(radius_cm / dx_cm)))
    radius_y = max(1, int(np.ceil(radius_cm / dy_cm)))
    radius_z = max(1, int(np.ceil(radius_cm / dz_cm)))

    xs = np.arange(-radius_x, radius_x + 1, dtype=np.float32) * dx_cm
    ys = np.arange(-radius_y, radius_y + 1, dtype=np.float32) * dy_cm
    zs = np.arange(-radius_z, radius_z + 1, dtype=np.float32) * dz_cm
    grid_x, grid_y, grid_z = np.meshgrid(xs, ys, zs, indexing="ij")
    radial_distance = np.sqrt(grid_x**2 + grid_y**2 + grid_z**2)
    kernel = float(signal_strength) * np.exp(-radial_distance / lambda_cm)
    return kernel.astype(np.float32)


def build_emission_map(
    lattice_dose: np.ndarray,
    emission_threshold_gy: float,
    continuous_emission: bool,
    emission_emax: float,
    emission_gamma_per_gy: float,
) -> np.ndarray:
    if continuous_emission:
        return (float(emission_emax) * (1.0 - np.exp(-float(emission_gamma_per_gy) * lattice_dose))).astype(np.float32)
    return (lattice_dose > float(emission_threshold_gy)).astype(np.float32)


def plot_lateral_profile_figure(
    x_cm: np.ndarray,
    profiles: List[Tuple[np.ndarray, str, str, str]],
    title: str,
    out_file: Path,
    dpi: int,
) -> None:
    fig, ax = plt.subplots(figsize=(9.2, 5.4), constrained_layout=True)
    for values, label, color, linestyle in profiles:
        ax.plot(x_cm, values, label=label, color=color, linestyle=linestyle, linewidth=2.2)
    ax.set_xlabel("Lateral position x (cm)")
    ax.set_ylabel("Dose (Gy)")
    ax.set_title(title)
    ax.grid(alpha=0.25)
    ax.legend()
    fig.savefig(out_file, dpi=dpi)
    plt.close(fig)


def plot_biological_comparison_figure(
    x_cm: np.ndarray,
    z_cm: np.ndarray,
    survival_lq: np.ndarray,
    survival_total: np.ndarray,
    survival_loss: np.ndarray,
    guide_lines: List[Tuple[float, str, str, str]],
    title: str,
    out_file: Path,
    dpi: int,
) -> None:
    center_y = survival_lq.shape[1] // 2
    lq_slice = survival_lq[:, center_y, :].T
    total_slice = survival_total[:, center_y, :].T
    loss_slice = survival_loss[:, center_y, :].T
    extent = [float(x_cm[0]), float(x_cm[-1]), float(z_cm[0]), float(z_cm[-1])]

    fig, axes = plt.subplots(1, 3, figsize=(16.0, 5.6), constrained_layout=True, sharey=True)
    panel_specs = [
        (lq_slice, "Standard LQ", "viridis", 0.0, 1.0),
        (total_slice, "Multi-effect", "viridis", 0.0, 1.0),
        (loss_slice, "Survival loss", "magma", 0.0, max(float(np.max(loss_slice)), 1.0e-6)),
    ]

    for ax, (values, panel_title, cmap, vmin, vmax) in zip(axes, panel_specs):
        image = ax.imshow(
            values,
            origin="lower",
            aspect="auto",
            extent=extent,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
        )
        for depth_cm, label, color, linestyle in guide_lines:
            ax.axhline(depth_cm, color=color, linestyle=linestyle, linewidth=1.7, label=label)
        ax.set_xlabel("Lateral position x (cm)")
        ax.set_title(panel_title)
        fig.colorbar(image, ax=ax, shrink=0.88)

    axes[0].set_ylabel("Depth z (cm)")
    axes[0].legend(loc="upper right")
    fig.suptitle(title)
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

    kernel = build_diffusion_kernel(
        dx_cm=dx_cm,
        dy_cm=dy_cm,
        dz_cm=dz_cm,
        diffusion_length_mm=float(args.diffusion_length_mm),
        signal_strength=float(args.signal_strength),
        kernel_radius_lambda=float(args.kernel_radius_lambda),
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

    metrics_rows: List[Dict[str, object]] = []
    summary: Dict[str, object] = {
        "input_csv": str(args.csv),
        "outdir": str(args.outdir),
        "normalization": norm_meta,
        "physical_floor_model": {
            "uniform_dose_floor_fraction_of_peak": float(args.uniform_dose_floor_fraction),
            "uniform_dose_floor_gy": float(args.uniform_dose_floor_fraction) * float(args.prescribed_peak_dose_gy),
            "rationale": "Uniform surrogate transmission floor added in Python to approximate 1%-2% block or MLC transmission.",
        },
        "bystander_model": {
            "emission_model": "continuous_saturated" if bool(args.continuous_emission) else "binary_threshold",
            "emission_threshold_gy": float(args.emission_threshold_gy),
            "emission_emax": float(args.emission_emax),
            "emission_gamma_per_gy": float(args.emission_gamma_per_gy),
            "diffusion_length_mm": float(args.diffusion_length_mm),
            "signal_strength": float(args.signal_strength),
            "kernel_radius_lambda": float(args.kernel_radius_lambda),
            "kernel_shape": [int(v) for v in kernel.shape],
            "formula": (
                "S_total = S_LQ * exp(-B), with B = E (*) [A * exp(-r / lambda)] and "
                "E(D) = Emax * (1 - exp(-gamma * D)) for continuous emission."
                if bool(args.continuous_emission)
                else "S_total = S_LQ * exp(-B), with B = E (*) [A * exp(-r / lambda)]"
            ),
        },
        "high_dose_survival_note": str(args.high_dose_methodology_note),
        "generated_pitches": {},
        "outputs": {},
    }

    requested_depths_cm = [float(v) for v in args.depths_cm]

    for pitch_index, pitch_mm in enumerate([float(v) for v in args.pitches_mm], start=1):
        pitch_label = f"{pitch_mm:g}"
        pitch_cm = pitch_mm / 10.0
        pitch_bins_x = int(round(pitch_cm / dx_cm))
        pitch_bins_y = int(round(pitch_cm / dy_cm))
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
        depth_indices = {
            "dmax": dmax_idx,
            "d=3cm": idx_3,
            "d=5cm": idx_5,
        }

        emission_map = build_emission_map(
            lattice_dose=lattice_dose,
            emission_threshold_gy=float(args.emission_threshold_gy),
            continuous_emission=bool(args.continuous_emission),
            emission_emax=float(args.emission_emax),
            emission_gamma_per_gy=float(args.emission_gamma_per_gy),
        )
        bystander_penalty = np.maximum(fftconvolve(emission_map, kernel, mode="same"), 0.0).astype(np.float32)
        survival_lq = np.exp(-float(args.alpha) * lattice_dose - float(args.beta) * lattice_dose**2).astype(np.float32)
        survival_total = (survival_lq * np.exp(-bystander_penalty)).astype(np.float32)
        survival_loss = (survival_lq - survival_total).astype(np.float32)

        center_x_idx = lattice_dose.shape[0] // 2
        center_y_idx = lattice_dose.shape[1] // 2
        x_centers_idx = [center_x_idx + off for off in offsets_x]
        peak_x_idx = x_centers_idx[len(x_centers_idx) // 2]
        valley_x_idx = (x_centers_idx[len(x_centers_idx) // 2] + x_centers_idx[(len(x_centers_idx) // 2) + 1]) // 2

        profiles = {
            label: profile_from_center_strip(lattice_dose, idx, int(args.profile_half_width_bins))
            for label, idx in depth_indices.items()
        }
        guide_lines = [
            (float(z_cm[dmax_idx]), legend_labels["dmax"], colors["dmax"], linestyles["dmax"]),
            (float(z_cm[idx_3]), legend_labels["d=3cm"], colors["d=3cm"], linestyles["d=3cm"]),
            (float(z_cm[idx_5]), legend_labels["d=5cm"], colors["d=5cm"], linestyles["d=5cm"]),
        ]

        physical_figure_number = (2 * pitch_index) - 1
        bio_figure_number = 2 * pitch_index
        physical_file = args.outdir / f"figure{physical_figure_number}_physical_lateral_profile_pitch{pitch_label}.png"
        bio_file = args.outdir / f"figure{bio_figure_number}_biological_comparison_pitch{pitch_label}.png"

        plot_lateral_profile_figure(
            x_cm=x_cm,
            profiles=[
                (profiles["dmax"], legend_labels["dmax"], colors["dmax"], linestyles["dmax"]),
                (profiles["d=3cm"], legend_labels["d=3cm"], colors["d=3cm"], linestyles["d=3cm"]),
                (profiles["d=5cm"], legend_labels["d=5cm"], colors["d=5cm"], linestyles["d=5cm"]),
            ],
            title=f"Figure {physical_figure_number}: Physical lateral dose profiles (pitch = {pitch_label} mm)",
            out_file=physical_file,
            dpi=int(args.dpi),
        )
        plot_biological_comparison_figure(
            x_cm=x_cm,
            z_cm=z_cm,
            survival_lq=survival_lq,
            survival_total=survival_total,
            survival_loss=survival_loss,
            guide_lines=guide_lines,
            title=f"Figure {bio_figure_number}: Biological comparison (pitch = {pitch_label} mm)",
            out_file=bio_file,
            dpi=int(args.dpi),
        )

        pitch_summary: Dict[str, object] = {
            "pitch_mm": float(pitch_mm),
            "physical_figure": str(physical_file),
            "biological_figure": str(bio_file),
            "sampled_depths_cm": {},
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
                "peak_emission_strength": float(emission_map[peak_x_idx, center_y_idx, idx]),
                "valley_emission_strength": float(emission_map[valley_x_idx, center_y_idx, idx]),
                "peak_bystander_penalty": float(bystander_penalty[peak_x_idx, center_y_idx, idx]),
                "valley_bystander_penalty": float(bystander_penalty[valley_x_idx, center_y_idx, idx]),
                "peak_survival_total": float(survival_total[peak_x_idx, center_y_idx, idx]),
                "valley_survival_total": float(survival_total[valley_x_idx, center_y_idx, idx]),
                "valley_survival_loss": float(survival_loss[valley_x_idx, center_y_idx, idx]),
            }
            metrics_rows.append(row)
            pitch_summary["sampled_depths_cm"][label] = float(z_cm[idx])
            pitch_summary["metrics_by_depth"][label] = row

        summary["generated_pitches"][f"pitch_{pitch_label}mm"] = pitch_summary
        summary["outputs"][f"figure{physical_figure_number}"] = str(physical_file)
        summary["outputs"][f"figure{bio_figure_number}"] = str(bio_file)

        del lattice_dose
        del emission_map
        del bystander_penalty
        del survival_lq
        del survival_total
        del survival_loss
        gc.collect()

    metrics_csv = args.outdir / "phase2_metrics.csv"
    write_csv(metrics_rows, metrics_csv)
    summary["metrics_csv"] = str(metrics_csv)

    summary_file = args.outdir / "phase2_summary.json"
    summary_file.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Phase 2 summary: {summary_file}")
    print(f"Phase 2 metrics: {metrics_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
