#!/usr/bin/env python
"""Generate multi-pitch SFRT figures from the direct-photon dose kernel."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

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
    profile_from_center_strip,
    reference_peak_value,
)
from analyze_topas_outputs import load_topas_grid


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description=(
            "Load the direct-photon TOPAS dose kernel, synthesize lattices for multiple "
            "pitches, and write Figures 1-4 with shared depth legends."
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
        default=root / "runs" / "linac_6mv_direct_photon_sfrt" / "analysis",
        help="Directory for generated figures.",
    )
    parser.add_argument(
        "--prescribed-peak-dose-gy",
        type=float,
        default=10.0,
        help="Dose assigned to the reference peak voxel after normalization.",
    )
    parser.add_argument(
        "--pitches-mm",
        nargs="+",
        type=float,
        default=[20.0, 30.0],
        help="Synthetic lattice pitches in mm, mapped to sequential figure pairs.",
    )
    parser.add_argument("--n-beams-x", type=int, default=7, help="Beam copies along x.")
    parser.add_argument("--n-beams-y", type=int, default=7, help="Beam copies along y.")
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
    parser.add_argument("--alpha", type=float, default=0.10, help="LQ alpha in Gy^-1.")
    parser.add_argument("--beta", type=float, default=0.05, help="LQ beta in Gy^-2.")
    parser.add_argument("--dpi", type=int, default=250, help="Figure DPI.")
    parser.add_argument(
        "--start-figure-number",
        type=int,
        default=1,
        help="Figure number assigned to the first generated physical profile.",
    )
    parser.add_argument(
        "--summary-file",
        type=Path,
        default=None,
        help="Optional JSON file for generation metadata. Defaults to outdir/multi_pitch_figure_summary.json.",
    )
    return parser.parse_args()


def normalize_single_beam(single_beam: np.ndarray, z_cm: np.ndarray, prescribed_peak_dose_gy: float) -> Tuple[np.ndarray, Dict[str, float]]:
    ref_idx = choose_reference_depth(single_beam, z_cm, argparse.Namespace(reference_mode="dmax", reference_depth_cm=0.0))
    ref_peak_raw, ref_peak_idx_xy = reference_peak_value(single_beam, ref_idx, center_window_bins=5)
    scale = float(prescribed_peak_dose_gy) / ref_peak_raw
    return single_beam * scale, {
        "reference_depth_cm": float(z_cm[ref_idx]),
        "reference_peak_raw_gy": float(ref_peak_raw),
        "scale_factor": float(scale),
        "reference_peak_index_x": int(ref_peak_idx_xy[0]),
        "reference_peak_index_y": int(ref_peak_idx_xy[1]),
    }


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


def plot_survival_heatmap_figure(
    x_cm: np.ndarray,
    z_cm: np.ndarray,
    survival: np.ndarray,
    guide_lines: List[Tuple[float, str, str, str]],
    title: str,
    out_file: Path,
    dpi: int,
) -> None:
    center_y = survival.shape[1] // 2
    xz_slice = survival[:, center_y, :].T
    extent = [float(x_cm[0]), float(x_cm[-1]), float(z_cm[0]), float(z_cm[-1])]
    fig, ax = plt.subplots(figsize=(9.2, 5.8), constrained_layout=True)
    image = ax.imshow(
        xz_slice,
        origin="lower",
        aspect="auto",
        extent=extent,
        cmap="viridis",
        vmin=0.0,
        vmax=1.0,
    )
    for depth_cm, label, color, linestyle in guide_lines:
        ax.axhline(depth_cm, color=color, linestyle=linestyle, linewidth=1.8, label=label)
    ax.set_xlabel("Lateral position x (cm)")
    ax.set_ylabel("Depth z (cm)")
    ax.set_title(title)
    ax.legend(loc="upper right")
    cbar = fig.colorbar(image, ax=ax)
    cbar.set_label("Survival fraction")
    fig.savefig(out_file, dpi=dpi)
    plt.close(fig)


def build_pitch_result(
    normalized_single: np.ndarray,
    dx_cm: float,
    dy_cm: float,
    dz_cm: float,
    pitch_mm: float,
    n_beams_x: int,
    n_beams_y: int,
    profile_half_width_bins: int,
    fixed_depths_cm: List[float],
    alpha: float,
    beta: float,
) -> Dict[str, object]:
    pitch_cm = float(pitch_mm) / 10.0
    pitch_bins_x = int(round(pitch_cm / dx_cm))
    pitch_bins_y = int(round(pitch_cm / dy_cm))
    lattice_dose, offsets_x, _ = build_lattice(
        normalized_single,
        pitch_bins_x=pitch_bins_x,
        pitch_bins_y=pitch_bins_y,
        n_beams_x=int(n_beams_x),
        n_beams_y=int(n_beams_y),
    )
    x_cm = centered_axis_cm(lattice_dose.shape[0], dx_cm)
    z_cm = depth_axis_cm(lattice_dose.shape[2], dz_cm)
    lattice_depth_profile = np.sum(lattice_dose, axis=(0, 1))
    dmax_idx = int(np.argmax(lattice_depth_profile))
    fixed_indices = [nearest_index(z_cm, depth) for depth in fixed_depths_cm]

    profiles = {
        "dmax": profile_from_center_strip(lattice_dose, dmax_idx, profile_half_width_bins),
        "d=3cm": profile_from_center_strip(lattice_dose, fixed_indices[0], profile_half_width_bins),
        "d=5cm": profile_from_center_strip(lattice_dose, fixed_indices[1], profile_half_width_bins),
    }
    survival = np.exp(-float(alpha) * lattice_dose - float(beta) * lattice_dose**2)
    center_x_idx = lattice_dose.shape[0] // 2
    x_centers_idx = [center_x_idx + off for off in offsets_x]

    return {
        "pitch_mm": float(pitch_mm),
        "x_cm": x_cm,
        "z_cm": z_cm,
        "lattice_dose": lattice_dose,
        "survival": survival,
        "profiles": profiles,
        "dmax_depth_cm": float(z_cm[dmax_idx]),
        "sampled_depths_cm": {
            "d=3cm": float(z_cm[fixed_indices[0]]),
            "d=5cm": float(z_cm[fixed_indices[1]]),
        },
        "x_centers_idx": x_centers_idx,
        "pitch_bins_x": int(pitch_bins_x),
    }


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

    pitches_mm = [float(v) for v in args.pitches_mm]
    fixed_depths_cm = [float(v) for v in args.depths_cm]
    pitch_results = []
    for pitch_mm in pitches_mm:
        pitch_results.append(
            build_pitch_result(
                normalized_single,
                dx_cm,
                dy_cm,
                dz_cm,
                pitch_mm=pitch_mm,
                n_beams_x=int(args.n_beams_x),
                n_beams_y=int(args.n_beams_y),
                profile_half_width_bins=int(args.profile_half_width_bins),
                fixed_depths_cm=fixed_depths_cm,
                alpha=float(args.alpha),
                beta=float(args.beta),
            )
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

    generated_outputs = {}
    generated_pitches = {}
    for idx, pitch_result in enumerate(pitch_results, start=1):
        pitch_mm = float(pitch_result["pitch_mm"])
        pitch_label = f"{pitch_mm:g}"
        physical_figure_number = int(args.start_figure_number) + (2 * (idx - 1))
        survival_figure_number = physical_figure_number + 1
        physical_file = args.outdir / f"figure{physical_figure_number}_physical_lateral_profile.png"
        survival_file = args.outdir / f"figure{survival_figure_number}_standard_lq_survival_heatmap.png"

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
        plot_survival_heatmap_figure(
            x_cm=pitch_result["x_cm"],
            z_cm=pitch_result["z_cm"],
            survival=pitch_result["survival"],
            guide_lines=[
                (pitch_result["dmax_depth_cm"], legend_labels["dmax"], colors["dmax"], linestyles["dmax"]),
                (pitch_result["sampled_depths_cm"]["d=3cm"], legend_labels["d=3cm"], colors["d=3cm"], linestyles["d=3cm"]),
                (pitch_result["sampled_depths_cm"]["d=5cm"], legend_labels["d=5cm"], colors["d=5cm"], linestyles["d=5cm"]),
            ],
            title=f"Figure {survival_figure_number}: Standard LQ survival slice (pitch = {pitch_label} mm)",
            out_file=survival_file,
            dpi=int(args.dpi),
        )

        generated_pitches[f"pitch_{pitch_label}mm"] = {
            "dmax_depth_cm": float(pitch_result["dmax_depth_cm"]),
            "sampled_depths_cm": pitch_result["sampled_depths_cm"],
            "physical_figure": str(physical_file),
            "survival_figure": str(survival_file),
        }
        generated_outputs[f"figure{physical_figure_number}"] = str(physical_file)
        generated_outputs[f"figure{survival_figure_number}"] = str(survival_file)

    summary = {
        "input_csv": str(args.csv),
        "outdir": str(args.outdir),
        "normalization": norm_meta,
        "requested_depths_cm": {
            "d=3cm": float(fixed_depths_cm[0]),
            "d=5cm": float(fixed_depths_cm[1]),
        },
        "generated_pitches": generated_pitches,
        "outputs": generated_outputs,
    }
    summary_file = args.summary_file or (args.outdir / "multi_pitch_figure_summary.json")
    summary_file.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Figure summary: {summary_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
