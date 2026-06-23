#!/usr/bin/env python3
"""Build a synthetic SFRT lattice from a TOPAS single-beam dose CSV.

Outputs:
- normalized single-beam kernel as a 3D NumPy array
- synthetic lattice dose as a 3D NumPy array
- 3D gradient magnitude map
- 3D linear-quadratic survival map
- depth-wise PVDR CSV
- two figures requested by the user
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception as exc:  # pragma: no cover
    raise RuntimeError("matplotlib is required for figure generation") from exc

from analyze_topas_outputs import load_topas_grid


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description=(
            "Load a TOPAS kernel, normalize it to a prescribed peak dose, "
            "superpose it into a synthetic lattice, compute PVDR/gradient/LQ maps, "
            "and write requested figures."
        )
    )
    parser.add_argument("--csv", type=Path, required=True, help="TOPAS dose CSV.")
    parser.add_argument(
        "--outdir",
        type=Path,
        default=root / "runs" / "sfrt_250_plan_analysis",
        help="Output directory.",
    )
    parser.add_argument(
        "--case-label",
        type=str,
        default="",
        help="Optional human-readable label for the input kernel.",
    )
    parser.add_argument(
        "--beam-label",
        type=str,
        default="250 MeV quadrupole-focused beam",
        help="Beam label used in figure titles and the summary.",
    )
    parser.add_argument(
        "--histories",
        type=int,
        default=0,
        help="Optional history count to include in the summary.",
    )
    parser.add_argument(
        "--pitch-mm",
        type=float,
        default=50.0,
        help="Center-to-center beam spacing in mm.",
    )
    parser.add_argument(
        "--n-beams-x",
        type=int,
        default=5,
        help="Number of beam copies along x.",
    )
    parser.add_argument(
        "--n-beams-y",
        type=int,
        default=5,
        help="Number of beam copies along y.",
    )
    parser.add_argument(
        "--reference-mode",
        choices=["dmax", "depth_cm"],
        default="dmax",
        help="How to choose the normalization depth.",
    )
    parser.add_argument(
        "--reference-depth-cm",
        type=float,
        default=0.0,
        help="Reference depth when --reference-mode=depth_cm.",
    )
    parser.add_argument(
        "--prescribed-peak-dose-gy",
        type=float,
        default=10.0,
        help="Dose assigned to the reference peak voxel after normalization.",
    )
    parser.add_argument(
        "--hippocampus-depth-cm",
        type=float,
        default=5.0,
        help="Virtual hippocampus depth used for PVDR reporting.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.10,
        help="LQ alpha in Gy^-1.",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=0.05,
        help="LQ beta in Gy^-2.",
    )
    parser.add_argument(
        "--center-window-bins",
        type=int,
        default=5,
        help="Half-width of the search window for the reference peak around the beam center.",
    )
    parser.add_argument(
        "--profile-half-width-bins",
        type=int,
        default=1,
        help="Half-width of the y-strip averaged into the lateral profile.",
    )
    parser.add_argument(
        "--figure1-title",
        type=str,
        default="",
        help="Optional custom title for Figure 1.",
    )
    parser.add_argument(
        "--figure2-title",
        type=str,
        default="",
        help="Optional custom title for Figure 2.",
    )
    parser.add_argument("--dpi", type=int, default=250, help="Figure DPI.")
    return parser.parse_args()


def centered_axis_cm(n: int, spacing_cm: float) -> np.ndarray:
    return (np.arange(n, dtype=float) - (n - 1) / 2.0) * spacing_cm


def depth_axis_cm(n: int, spacing_cm: float) -> np.ndarray:
    return (np.arange(n, dtype=float) + 0.5) * spacing_cm


def nearest_index(axis_cm: np.ndarray, target_cm: float) -> int:
    return int(np.argmin(np.abs(axis_cm - float(target_cm))))


def choose_reference_depth(single_beam: np.ndarray, z_cm: np.ndarray, args: argparse.Namespace) -> int:
    if args.reference_mode == "depth_cm":
        return nearest_index(z_cm, float(args.reference_depth_cm))
    integrated = np.sum(single_beam, axis=(0, 1))
    return int(np.argmax(integrated))


def reference_peak_value(
    single_beam: np.ndarray,
    ref_idx: int,
    center_window_bins: int,
) -> Tuple[float, Tuple[int, int]]:
    cx = single_beam.shape[0] // 2
    cy = single_beam.shape[1] // 2
    w = max(0, int(center_window_bins))
    plane = single_beam[:, :, ref_idx]
    xs = slice(max(0, cx - w), min(single_beam.shape[0], cx + w + 1))
    ys = slice(max(0, cy - w), min(single_beam.shape[1], cy + w + 1))
    local = plane[xs, ys]
    if local.size == 0:
        raise ValueError("Reference peak search window is empty.")
    local_flat = int(np.argmax(local))
    local_ix, local_iy = np.unravel_index(local_flat, local.shape)
    ref_val = float(local[local_ix, local_iy])
    peak_ix = max(0, cx - w) + int(local_ix)
    peak_iy = max(0, cy - w) + int(local_iy)
    if ref_val <= 0.0:
        peak_ix, peak_iy = np.unravel_index(int(np.argmax(plane)), plane.shape)
        ref_val = float(plane[peak_ix, peak_iy])
    if ref_val <= 0.0:
        raise ValueError("Reference peak value is zero; cannot normalize dose.")
    return ref_val, (int(peak_ix), int(peak_iy))


def compute_lattice_shape(
    kernel_shape: Tuple[int, int, int],
    pitch_bins_x: int,
    pitch_bins_y: int,
    n_beams_x: int,
    n_beams_y: int,
) -> Tuple[int, int, int]:
    nx, ny, nz = kernel_shape
    new_nx = nx + max(0, n_beams_x - 1) * max(0, pitch_bins_x)
    new_ny = ny + max(0, n_beams_y - 1) * max(0, pitch_bins_y)
    return int(new_nx), int(new_ny), int(nz)


def centered_offsets(n_beams: int, pitch_bins: int) -> List[int]:
    if n_beams < 1:
        raise ValueError("Number of beams must be at least 1.")
    if n_beams % 2 == 0:
        raise ValueError("An odd number of beams keeps the lattice centered.")
    half = (n_beams - 1) // 2
    return [int(i * pitch_bins) for i in range(-half, half + 1)]


def paste_kernel(target: np.ndarray, kernel: np.ndarray, start_x: int, start_y: int) -> None:
    end_x = start_x + kernel.shape[0]
    end_y = start_y + kernel.shape[1]
    target[start_x:end_x, start_y:end_y, :] += kernel


def build_lattice(
    kernel: np.ndarray,
    pitch_bins_x: int,
    pitch_bins_y: int,
    n_beams_x: int,
    n_beams_y: int,
) -> Tuple[np.ndarray, List[int], List[int]]:
    offsets_x = centered_offsets(n_beams_x, pitch_bins_x)
    offsets_y = centered_offsets(n_beams_y, pitch_bins_y)
    out_shape = compute_lattice_shape(kernel.shape, pitch_bins_x, pitch_bins_y, n_beams_x, n_beams_y)
    lattice = np.zeros(out_shape, dtype=np.float64)

    center_x = lattice.shape[0] // 2
    center_y = lattice.shape[1] // 2
    kernel_center_x = kernel.shape[0] // 2
    kernel_center_y = kernel.shape[1] // 2

    for off_x in offsets_x:
        for off_y in offsets_y:
            start_x = center_x + off_x - kernel_center_x
            start_y = center_y + off_y - kernel_center_y
            paste_kernel(lattice, kernel, start_x, start_y)
    return lattice, offsets_x, offsets_y


def profile_from_center_strip(volume: np.ndarray, z_idx: int, half_width_bins: int) -> np.ndarray:
    cy = volume.shape[1] // 2
    hw = max(0, int(half_width_bins))
    ys = slice(max(0, cy - hw), min(volume.shape[1], cy + hw + 1))
    strip = volume[:, ys, z_idx]
    return np.mean(strip, axis=1)


def single_beam_profiles_at_depth(
    single_beam: np.ndarray,
    z_idx: int,
    half_width_bins: int,
) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int]]:
    plane = single_beam[:, :, z_idx]
    peak_ix, peak_iy = np.unravel_index(int(np.argmax(plane)), plane.shape)
    hw = max(0, int(half_width_bins))

    yslice = slice(max(0, peak_iy - hw), min(single_beam.shape[1], peak_iy + hw + 1))
    xslice = slice(max(0, peak_ix - hw), min(single_beam.shape[0], peak_ix + hw + 1))

    profile_x = np.mean(plane[:, yslice], axis=1)
    profile_y = np.mean(plane[xslice, :], axis=0)
    return profile_x, profile_y, (int(peak_ix), int(peak_iy))


def interpolate_crossing(coord_a: float, value_a: float, coord_b: float, value_b: float, target: float) -> float:
    if value_a == value_b:
        return float((coord_a + coord_b) / 2.0)
    frac = (target - value_a) / (value_b - value_a)
    return float(coord_a + frac * (coord_b - coord_a))


def profile_fwhm_mm(coords_cm: np.ndarray, values: np.ndarray) -> float:
    if values.size < 3:
        return float("nan")
    peak_idx = int(np.argmax(values))
    peak_val = float(values[peak_idx])
    if peak_val <= 0.0:
        return float("nan")
    half_max = 0.5 * peak_val

    left_idx = peak_idx
    while left_idx > 0 and float(values[left_idx - 1]) >= half_max:
        left_idx -= 1
    if left_idx == 0 and float(values[left_idx]) >= half_max:
        return float("nan")
    left_cross = interpolate_crossing(
        float(coords_cm[left_idx - 1]),
        float(values[left_idx - 1]),
        float(coords_cm[left_idx]),
        float(values[left_idx]),
        half_max,
    )

    right_idx = peak_idx
    while right_idx < (values.size - 1) and float(values[right_idx + 1]) >= half_max:
        right_idx += 1
    if right_idx == (values.size - 1) and float(values[right_idx]) >= half_max:
        return float("nan")
    right_cross = interpolate_crossing(
        float(coords_cm[right_idx]),
        float(values[right_idx]),
        float(coords_cm[right_idx + 1]),
        float(values[right_idx + 1]),
        half_max,
    )
    return float((right_cross - left_cross) * 10.0)


def peak_valley_metrics(profile_x: np.ndarray, centers_idx: Iterable[int], pitch_bins: int) -> Dict[str, float]:
    centers = [int(v) for v in centers_idx]
    if len(centers) < 2:
        return {
            "mean_peak_gy": float("nan"),
            "mean_valley_gy": float("nan"),
            "pvdr": float("nan"),
        }

    win = max(1, int(round(max(1.0, pitch_bins / 4.0))))
    peaks: List[float] = []
    valleys: List[float] = []

    for center in centers:
        left = max(0, center - win)
        right = min(profile_x.size, center + win + 1)
        peaks.append(float(np.max(profile_x[left:right])))

    gap = max(1, win // 2)
    for left_center, right_center in zip(centers[:-1], centers[1:]):
        left = min(left_center, right_center) + gap
        right = max(left_center, right_center) - gap + 1
        if left < right:
            valleys.append(float(np.min(profile_x[left:right])))

    mean_peak = float(np.mean(peaks)) if peaks else float("nan")
    mean_valley = float(np.mean(valleys)) if valleys else float("nan")
    pvdr = mean_peak / mean_valley if mean_valley > 0.0 else float("inf")
    return {
        "mean_peak_gy": mean_peak,
        "mean_valley_gy": mean_valley,
        "pvdr": float(pvdr),
    }


def pvdr_depth_curve(
    lattice_dose: np.ndarray,
    z_cm: np.ndarray,
    x_centers_idx: List[int],
    pitch_bins_x: int,
    profile_half_width_bins: int,
) -> List[Dict[str, float]]:
    rows: List[Dict[str, float]] = []
    for z_idx, depth_cm in enumerate(z_cm):
        profile = profile_from_center_strip(lattice_dose, z_idx, profile_half_width_bins)
        metrics = peak_valley_metrics(profile, x_centers_idx, pitch_bins_x)
        rows.append(
            {
                "z_index": int(z_idx),
                "depth_cm": float(depth_cm),
                "mean_peak_gy": float(metrics["mean_peak_gy"]),
                "mean_valley_gy": float(metrics["mean_valley_gy"]),
                "pvdr": float(metrics["pvdr"]),
            }
        )
    return rows


def write_csv(rows: List[Dict[str, float]], out_file: Path) -> None:
    if not rows:
        raise ValueError("No rows supplied for CSV output.")
    with out_file.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_lateral_profiles(
    x_cm: np.ndarray,
    profile_dmax: np.ndarray,
    profile_hipp: np.ndarray,
    dmax_depth_cm: float,
    hippocampus_depth_cm: float,
    title: str,
    out_file: Path,
    dpi: int,
) -> None:
    fig, ax = plt.subplots(figsize=(9.0, 5.2), constrained_layout=True)
    ax.plot(x_cm, profile_dmax, linewidth=2.4, label=f"dmax slice ({dmax_depth_cm:.2f} cm)")
    ax.plot(
        x_cm,
        profile_hipp,
        linewidth=2.0,
        linestyle="--",
        label=f"virtual hippocampus ({hippocampus_depth_cm:.2f} cm)",
    )
    ax.set_xlabel("Lateral position x (cm)")
    ax.set_ylabel("Dose (Gy)")
    ax.set_title(title)
    ax.grid(alpha=0.25)
    ax.legend()
    fig.savefig(out_file, dpi=dpi)
    plt.close(fig)


def plot_survival_heatmap(
    x_cm: np.ndarray,
    z_cm: np.ndarray,
    survival: np.ndarray,
    title: str,
    out_file: Path,
    dpi: int,
) -> None:
    center_y = survival.shape[1] // 2
    xz_slice = survival[:, center_y, :].T
    extent = [float(x_cm[0]), float(x_cm[-1]), float(z_cm[0]), float(z_cm[-1])]
    fig, ax = plt.subplots(figsize=(9.0, 5.6), constrained_layout=True)
    image = ax.imshow(
        xz_slice,
        origin="lower",
        aspect="auto",
        extent=extent,
        cmap="viridis",
        vmin=0.0,
        vmax=1.0,
    )
    ax.set_xlabel("Lateral position x (cm)")
    ax.set_ylabel("Depth z (cm)")
    ax.set_title(title)
    cbar = fig.colorbar(image, ax=ax)
    cbar.set_label("Survival fraction")
    fig.savefig(out_file, dpi=dpi)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    outdir = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    single_beam, header = load_topas_grid(args.csv, retries=5, retry_delay_sec=0.5)
    dx_cm = float(header["dx_cm"])
    dy_cm = float(header["dy_cm"])
    dz_cm = float(header["dz_cm"])

    x_single_cm = centered_axis_cm(single_beam.shape[0], dx_cm)
    y_single_cm = centered_axis_cm(single_beam.shape[1], dy_cm)
    z_cm = depth_axis_cm(single_beam.shape[2], dz_cm)

    ref_idx = choose_reference_depth(single_beam, z_cm, args)
    ref_peak_raw, ref_peak_idx_xy = reference_peak_value(
        single_beam,
        ref_idx,
        center_window_bins=int(args.center_window_bins),
    )
    scale = float(args.prescribed_peak_dose_gy) / ref_peak_raw
    normalized_single = single_beam * scale

    pitch_cm = float(args.pitch_mm) / 10.0
    pitch_bins_x = int(round(pitch_cm / dx_cm))
    pitch_bins_y = int(round(pitch_cm / dy_cm))
    if pitch_bins_x < 1 or pitch_bins_y < 1:
        raise ValueError("Chosen pitch is smaller than one voxel.")

    lattice_dose, offsets_x, offsets_y = build_lattice(
        normalized_single,
        pitch_bins_x=pitch_bins_x,
        pitch_bins_y=pitch_bins_y,
        n_beams_x=int(args.n_beams_x),
        n_beams_y=int(args.n_beams_y),
    )

    x_cm = centered_axis_cm(lattice_dose.shape[0], dx_cm)
    y_cm = centered_axis_cm(lattice_dose.shape[1], dy_cm)
    z_lattice_cm = depth_axis_cm(lattice_dose.shape[2], dz_cm)

    lattice_depth_profile = np.sum(lattice_dose, axis=(0, 1))
    dmax_idx = int(np.argmax(lattice_depth_profile))
    hippocampus_idx = nearest_index(z_lattice_cm, float(args.hippocampus_depth_cm))

    center_x_idx = lattice_dose.shape[0] // 2
    x_centers_idx = [center_x_idx + off for off in offsets_x]
    profile_dmax = profile_from_center_strip(lattice_dose, dmax_idx, int(args.profile_half_width_bins))
    profile_hipp = profile_from_center_strip(lattice_dose, hippocampus_idx, int(args.profile_half_width_bins))
    single_profile_x_hipp, single_profile_y_hipp, single_peak_idx_xy_hipp = single_beam_profiles_at_depth(
        normalized_single,
        hippocampus_idx,
        int(args.profile_half_width_bins),
    )
    single_fwhm_x_mm = profile_fwhm_mm(x_single_cm, single_profile_x_hipp)
    single_fwhm_y_mm = profile_fwhm_mm(y_single_cm, single_profile_y_hipp)
    conservative_fwhm_mm = float(np.nanmax([single_fwhm_x_mm, single_fwhm_y_mm]))

    pvdr_rows = pvdr_depth_curve(
        lattice_dose=lattice_dose,
        z_cm=z_lattice_cm,
        x_centers_idx=x_centers_idx,
        pitch_bins_x=pitch_bins_x,
        profile_half_width_bins=int(args.profile_half_width_bins),
    )
    pvdr_csv = outdir / "pvdr_depth_curve.csv"
    write_csv(pvdr_rows, pvdr_csv)

    pvdr_dmax = peak_valley_metrics(profile_dmax, x_centers_idx, pitch_bins_x)
    pvdr_hipp = peak_valley_metrics(profile_hipp, x_centers_idx, pitch_bins_x)

    grad_x, grad_y, grad_z = np.gradient(lattice_dose, dx_cm, dy_cm, dz_cm)
    gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2 + grad_z**2)
    survival = np.exp(-float(args.alpha) * lattice_dose - float(args.beta) * lattice_dose**2)

    np.save(outdir / "single_beam_normalized_gy.npy", normalized_single)
    np.save(outdir / "synthetic_lattice_dose_gy.npy", lattice_dose)
    np.save(outdir / "synthetic_lattice_gradient_gy_per_cm.npy", gradient_magnitude)
    np.save(outdir / "synthetic_lq_survival.npy", survival)

    fig1 = outdir / "figure1_physical_lateral_profile.png"
    fig2 = outdir / "figure2_standard_lq_survival_heatmap.png"
    figure1_title = args.figure1_title.strip() or f"Figure 1: {args.beam_label} Lateral Dose Profile"
    figure2_title = args.figure2_title.strip() or "Figure 2: Standard LQ Survival Slice (center y)"
    plot_lateral_profiles(
        x_cm=x_cm,
        profile_dmax=profile_dmax,
        profile_hipp=profile_hipp,
        dmax_depth_cm=float(z_lattice_cm[dmax_idx]),
        hippocampus_depth_cm=float(z_lattice_cm[hippocampus_idx]),
        title=figure1_title,
        out_file=fig1,
        dpi=int(args.dpi),
    )
    plot_survival_heatmap(
        x_cm=x_cm,
        z_cm=z_lattice_cm,
        survival=survival,
        title=figure2_title,
        out_file=fig2,
        dpi=int(args.dpi),
    )

    gradient_dmax_slice = gradient_magnitude[:, :, dmax_idx]
    gradient_hipp_slice = gradient_magnitude[:, :, hippocampus_idx]

    summary = {
        "input_csv": str(args.csv),
        "case_label": args.case_label if args.case_label else args.csv.stem,
        "beam_label": str(args.beam_label),
        "histories": int(args.histories) if int(args.histories) > 0 else None,
        "voxel_grid_single_beam": {
            "shape": [int(v) for v in single_beam.shape],
            "spacing_cm": {
                "dx": float(dx_cm),
                "dy": float(dy_cm),
                "dz": float(dz_cm),
            },
            "spacing_mm": {
                "dx": float(dx_cm * 10.0),
                "dy": float(dy_cm * 10.0),
                "dz": float(dz_cm * 10.0),
            },
            "x_extent_cm": [float(x_single_cm[0]), float(x_single_cm[-1])],
            "y_extent_cm": [float(y_single_cm[0]), float(y_single_cm[-1])],
            "z_extent_cm": [float(z_cm[0]), float(z_cm[-1])],
        },
        "voxel_grid_lattice": {
            "shape": [int(v) for v in lattice_dose.shape],
            "spacing_cm": {
                "dx": float(dx_cm),
                "dy": float(dy_cm),
                "dz": float(dz_cm),
            },
            "x_extent_cm": [float(x_cm[0]), float(x_cm[-1])],
            "y_extent_cm": [float(y_cm[0]), float(y_cm[-1])],
            "z_extent_cm": [float(z_lattice_cm[0]), float(z_lattice_cm[-1])],
        },
        "normalization": {
            "reference_mode": str(args.reference_mode),
            "reference_depth_cm": float(z_cm[ref_idx]),
            "reference_peak_index_xyz_single_beam": [
                int(ref_peak_idx_xy[0]),
                int(ref_peak_idx_xy[1]),
                int(ref_idx),
            ],
            "reference_peak_raw_gy": float(ref_peak_raw),
            "prescribed_peak_dose_gy": float(args.prescribed_peak_dose_gy),
            "scale_factor": float(scale),
        },
        "single_beam_at_virtual_hippocampus": {
            "depth_cm": float(z_cm[hippocampus_idx]),
            "peak_index_xy": [int(single_peak_idx_xy_hipp[0]), int(single_peak_idx_xy_hipp[1])],
            "peak_position_cm": [
                float(x_single_cm[single_peak_idx_xy_hipp[0]]),
                float(y_single_cm[single_peak_idx_xy_hipp[1]]),
            ],
            "fwhm_x_mm": float(single_fwhm_x_mm),
            "fwhm_y_mm": float(single_fwhm_y_mm),
            "fwhm_conservative_mm": float(conservative_fwhm_mm),
        },
        "lattice": {
            "pitch_mm": float(args.pitch_mm),
            "pitch_cm": float(pitch_cm),
            "pitch_bins_x": int(pitch_bins_x),
            "pitch_bins_y": int(pitch_bins_y),
            "n_beams_x": int(args.n_beams_x),
            "n_beams_y": int(args.n_beams_y),
            "beam_center_offsets_x_bins": [int(v) for v in offsets_x],
            "beam_center_offsets_y_bins": [int(v) for v in offsets_y],
            "beam_center_x_cm": [float(x_cm[center_x_idx + off]) for off in offsets_x],
            "beam_center_y_cm": [float(y_cm[(lattice_dose.shape[1] // 2) + off]) for off in offsets_y],
        },
        "pvdr": {
            "depths_reported_cm": {
                "dmax": float(z_lattice_cm[dmax_idx]),
                "virtual_hippocampus": float(z_lattice_cm[hippocampus_idx]),
            },
            "dmax": {
                "mean_peak_gy": float(pvdr_dmax["mean_peak_gy"]),
                "mean_valley_gy": float(pvdr_dmax["mean_valley_gy"]),
                "pvdr": float(pvdr_dmax["pvdr"]),
            },
            "virtual_hippocampus": {
                "mean_peak_gy": float(pvdr_hipp["mean_peak_gy"]),
                "mean_valley_gy": float(pvdr_hipp["mean_valley_gy"]),
                "pvdr": float(pvdr_hipp["pvdr"]),
            },
            "curve_csv": str(pvdr_csv),
        },
        "gradient": {
            "units": "Gy/cm",
            "max_gradient_gy_per_cm_dmax": float(np.max(gradient_dmax_slice)),
            "mean_gradient_gy_per_cm_dmax": float(np.mean(gradient_dmax_slice)),
            "max_gradient_gy_per_cm_virtual_hippocampus": float(np.max(gradient_hipp_slice)),
            "mean_gradient_gy_per_cm_virtual_hippocampus": float(np.mean(gradient_hipp_slice)),
        },
        "lq_model": {
            "alpha_gy_inverse": float(args.alpha),
            "beta_gy_inverse_squared": float(args.beta),
            "survival_range": [float(np.min(survival)), float(np.max(survival))],
        },
        "outputs": {
            "single_beam_normalized_npy": str(outdir / "single_beam_normalized_gy.npy"),
            "synthetic_lattice_dose_npy": str(outdir / "synthetic_lattice_dose_gy.npy"),
            "gradient_magnitude_npy": str(outdir / "synthetic_lattice_gradient_gy_per_cm.npy"),
            "survival_npy": str(outdir / "synthetic_lq_survival.npy"),
            "figure1_physical": str(fig1),
            "figure2_standard_bio": str(fig2),
            "figure1_title": figure1_title,
            "figure2_title": figure2_title,
        },
        "assumptions": [
            "Depth is measured from the entrance face using TOPAS voxel centers.",
            "The reference peak is the strongest voxel within a small window around the central beam at the chosen reference depth.",
            "The LQ parameters are illustrative CNS-like defaults and are easy to override on the command line.",
        ],
    }

    summary_file = outdir / "summary.json"
    summary_file.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Summary: {summary_file}")
    print(f"Figure 1: {fig1}")
    print(f"Figure 2: {fig2}")
    print(f"PVDR curve: {pvdr_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
