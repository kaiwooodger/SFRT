#!/usr/bin/env python
"""Generate improved Phase 10 geometry visualizations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from scipy import ndimage

from analyze_250mev_sfrt_plan import centered_axis_cm, nearest_index
from geometry_generators import generate_complex_lattice_plan_geometry

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
            "Generate improved Phase 10 lattice geometry visualizations: x-z MIP, "
            "three y-band x-z slices, and a 3D isodose rendering."
        )
    )
    parser.add_argument(
        "--phase10-summary",
        type=Path,
        default=root
        / "runs"
        / "linac_6mv_polyenergetic_clinical_sfrt"
        / "analysis_phase10_complex_lattice_plan"
        / "phase10_complex_lattice_summary.json",
        help="Phase 10 summary JSON.",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=None,
        help="Optional output directory. Defaults to the directory containing --phase10-summary.",
    )
    parser.add_argument(
        "--band-half-width-mm",
        type=float,
        default=7.5,
        help="Half-width of the representative y-band slabs used in the multi-slice x-z figure.",
    )
    parser.add_argument(
        "--downsample",
        type=int,
        default=5,
        help="Downsampling factor for the 3D rendering.",
    )
    parser.add_argument("--dpi", type=int, default=250, help="Figure DPI.")
    return parser.parse_args()


def load_phase10_geometry(summary: Dict[str, object]) -> Dict[str, object]:
    inputs = summary["inputs"]
    dose_grid, uptake_tensor, _, _, meta = generate_complex_lattice_plan_geometry(
        csv=Path(str(inputs["csv"])),
        prescribed_peak_dose_gy=float(inputs["prescribed_peak_dose_gy"]),
        uniform_dose_floor_fraction=float(inputs["uniform_dose_floor_fraction"]),
    )
    voxel_size_mm = tuple(float(value) for value in meta["voxel_size_mm"])
    x_cm = np.asarray(meta["x_cm"], dtype=np.float32)
    y_cm = centered_axis_cm(dose_grid.shape[1], float(meta["voxel_size_mm"][1]) / 10.0)
    z_cm = np.asarray(meta["z_cm"], dtype=np.float32)
    vessel_mask = uptake_tensor[1] > 0.0
    return {
        "dose_grid": dose_grid.astype(np.float32, copy=False),
        "vessel_mask": vessel_mask.astype(bool, copy=False),
        "meta": meta,
        "x_cm": x_cm,
        "y_cm": y_cm,
        "z_cm": z_cm,
        "voxel_size_mm": voxel_size_mm,
    }


def load_phase10_fields(summary: Dict[str, object]) -> Dict[str, object]:
    geometry = load_phase10_geometry(summary)
    volumes_path = summary.get("outputs", {}).get("volumes_npz")
    if not volumes_path or not Path(str(volumes_path)).exists():
        raise FileNotFoundError("Phase 10 volume cache was not found.")
    volumes = np.load(Path(str(volumes_path)))
    geometry.update(
        {
            "lq_survival": np.asarray(volumes["lq_survival"], dtype=np.float32),
            "final_survival": np.asarray(volumes["final_survival"], dtype=np.float32),
            "deff_grid": np.asarray(volumes["deff_grid"], dtype=np.float32),
        }
    )
    return geometry


def choose_y_band_centers_mm(placed_spots: List[Dict[str, object]]) -> List[Tuple[str, float]]:
    y_values = np.asarray([float(spec["y_mm"]) for spec in placed_spots], dtype=np.float32)
    inferior = y_values[y_values < -10.0]
    central = y_values[(y_values >= -10.0) & (y_values <= 10.0)]
    superior = y_values[y_values > 10.0]

    bands: List[Tuple[str, float]] = []
    if inferior.size:
        bands.append(("Inferior", float(np.median(inferior))))
    if central.size:
        bands.append(("Central", float(np.median(central))))
    if superior.size:
        bands.append(("Superior", float(np.median(superior))))

    if len(bands) == 3:
        return bands

    quantiles = np.quantile(y_values, [0.2, 0.5, 0.8])
    labels = ["Inferior", "Central", "Superior"]
    return [(label, float(value)) for label, value in zip(labels, quantiles)]


def plot_xz_mip(
    *,
    dose_grid: np.ndarray,
    vessel_mask: np.ndarray,
    x_cm: np.ndarray,
    z_cm: np.ndarray,
    target_depth_cm: float,
    out_file: Path,
    dpi: int,
) -> None:
    mip = np.max(dose_grid, axis=1).T
    vessel_mip = np.any(vessel_mask, axis=1).T.astype(np.float32)

    fig, ax = plt.subplots(1, 1, figsize=(8.2, 5.2), constrained_layout=True)
    img = ax.imshow(
        mip,
        origin="lower",
        extent=[float(x_cm[0]), float(x_cm[-1]), float(z_cm[0]), float(z_cm[-1])],
        aspect="auto",
        cmap="inferno",
        vmin=0.0,
        vmax=float(np.max(dose_grid)),
    )
    ax.contour(
        x_cm,
        z_cm,
        vessel_mip,
        levels=[0.5],
        colors=["#6fd3ff"],
        linewidths=1.0,
    )
    ax.axhline(float(target_depth_cm), color="#ffffff", linestyle="--", linewidth=1.2)
    ax.set_title("Figure 9: Phase 10 x-z maximum-intensity projection over all y")
    ax.set_xlabel("x (cm)")
    ax.set_ylabel("z (cm)")
    cbar = fig.colorbar(img, ax=ax, shrink=0.92)
    cbar.set_label("Dose (Gy)")
    fig.savefig(out_file, dpi=dpi)
    plt.close(fig)


def plot_three_y_band_slices(
    *,
    dose_grid: np.ndarray,
    vessel_mask: np.ndarray,
    x_cm: np.ndarray,
    y_cm: np.ndarray,
    z_cm: np.ndarray,
    band_centers_mm: List[Tuple[str, float]],
    band_half_width_mm: float,
    out_file: Path,
    dpi: int,
) -> List[Dict[str, float]]:
    dy_mm = float(abs(y_cm[1] - y_cm[0]) * 10.0) if len(y_cm) > 1 else 1.0
    half_bins = max(1, int(round(float(band_half_width_mm) / dy_mm)))
    band_meta: List[Dict[str, float]] = []

    fig, axes = plt.subplots(1, len(band_centers_mm), figsize=(5.2 * len(band_centers_mm), 4.9), constrained_layout=True)
    if len(band_centers_mm) == 1:
        axes = [axes]

    vmax = float(np.max(dose_grid))
    for ax, (label, center_mm) in zip(axes, band_centers_mm):
        idx = nearest_index(y_cm, float(center_mm) / 10.0)
        y_start = max(0, idx - half_bins)
        y_stop = min(dose_grid.shape[1], idx + half_bins + 1)
        slab = np.mean(dose_grid[:, y_start:y_stop, :], axis=1).T
        vessel_slab = np.any(vessel_mask[:, y_start:y_stop, :], axis=1).T.astype(np.float32)
        y_center_cm = float(np.mean(y_cm[y_start:y_stop]))

        img = ax.imshow(
            slab,
            origin="lower",
            extent=[float(x_cm[0]), float(x_cm[-1]), float(z_cm[0]), float(z_cm[-1])],
            aspect="auto",
            cmap="inferno",
            vmin=0.0,
            vmax=vmax,
        )
        ax.contour(
            x_cm,
            z_cm,
            vessel_slab,
            levels=[0.5],
            colors=["#6fd3ff"],
            linewidths=0.9,
        )
        ax.set_title(f"{label} band\nmean over y = {y_center_cm:.2f} cm")
        ax.set_xlabel("x (cm)")
        ax.set_ylabel("z (cm)")
        band_meta.append(
            {
                "label": label,
                "center_y_mm_requested": float(center_mm),
                "center_y_cm_sampled": float(y_center_cm),
                "y_start_index": int(y_start),
                "y_stop_index": int(y_stop),
            }
        )

    cbar = fig.colorbar(img, ax=axes, shrink=0.92)
    cbar.set_label("Dose (Gy)")
    fig.suptitle("Figure 10: Phase 10 x-z slices through three y bands", fontsize=14)
    fig.savefig(out_file, dpi=dpi)
    plt.close(fig)
    return band_meta


def shell_points(mask: np.ndarray) -> np.ndarray:
    if not np.any(mask):
        return np.empty((0, 3), dtype=np.int32)
    eroded = ndimage.binary_erosion(mask, structure=np.ones((3, 3, 3), dtype=bool))
    shell = mask & ~eroded
    return np.argwhere(shell)


def subsample_points(points: np.ndarray, max_points: int) -> np.ndarray:
    if len(points) <= max_points:
        return points
    step = int(np.ceil(len(points) / float(max_points)))
    return points[::step]


def plot_3d_isodose_render(
    *,
    dose_grid: np.ndarray,
    vessel_mask: np.ndarray,
    x_cm: np.ndarray,
    y_cm: np.ndarray,
    z_cm: np.ndarray,
    downsample: int,
    out_file: Path,
    dpi: int,
) -> None:
    stride = max(1, int(downsample))
    dose_ds = dose_grid[::stride, ::stride, ::stride]
    vessel_ds = vessel_mask[::stride, ::stride, ::stride]
    x_ds = x_cm[::stride]
    y_ds = y_cm[::stride]
    z_ds = z_cm[::stride]

    thresholds = [
        (8.0, "#ffd166", 3000, "High isodose"),
        (4.0, "#ef476f", 4000, "Mid isodose"),
        (2.0, "#7a5cff", 5000, "Low isodose"),
    ]

    fig = plt.figure(figsize=(9.0, 7.0), constrained_layout=True)
    ax = fig.add_subplot(111, projection="3d")

    for threshold, color, max_points, label in thresholds:
        pts = shell_points(dose_ds >= float(threshold))
        pts = subsample_points(pts, max_points=max_points)
        if len(pts) == 0:
            continue
        ax.scatter(
            x_ds[pts[:, 0]],
            y_ds[pts[:, 1]],
            z_ds[pts[:, 2]],
            s=3,
            alpha=0.18,
            c=color,
            label=label if threshold == 8.0 else None,
            depthshade=False,
        )

    vessel_pts = shell_points(vessel_ds)
    vessel_pts = subsample_points(vessel_pts, max_points=3500)
    if len(vessel_pts):
        ax.scatter(
            x_ds[vessel_pts[:, 0]],
            y_ds[vessel_pts[:, 1]],
            z_ds[vessel_pts[:, 2]],
            s=4,
            alpha=0.55,
            c="#4cc9f0",
            label="Vessel sinks",
            depthshade=False,
        )

    ax.set_xlabel("x (cm)")
    ax.set_ylabel("y (cm)")
    ax.set_zlabel("z (cm)")
    ax.set_title("Figure 11: Phase 10 3D isodose rendering with vessel sinks")
    ax.view_init(elev=24, azim=-58)
    ax.legend(loc="upper left")
    ax.set_box_aspect(
        (
            float(max(x_ds) - min(x_ds)),
            float(max(y_ds) - min(y_ds)),
            float(max(z_ds) - min(z_ds)),
        )
    )
    fig.savefig(out_file, dpi=dpi)
    plt.close(fig)


def plot_projected_field(
    *,
    field: np.ndarray,
    vessel_mask: np.ndarray,
    x_cm: np.ndarray,
    z_cm: np.ndarray,
    target_depth_cm: float,
    out_file: Path,
    dpi: int,
    reduction: str,
    cmap: str,
    vmin: float,
    vmax: float,
    title: str,
    cbar_label: str,
) -> None:
    if reduction == "max":
        projected = np.max(field, axis=1).T
    elif reduction == "min":
        projected = np.min(field, axis=1).T
    else:
        raise ValueError("reduction must be 'max' or 'min'.")
    vessel_mip = np.any(vessel_mask, axis=1).T.astype(np.float32)

    fig, ax = plt.subplots(1, 1, figsize=(8.2, 5.2), constrained_layout=True)
    img = ax.imshow(
        projected,
        origin="lower",
        extent=[float(x_cm[0]), float(x_cm[-1]), float(z_cm[0]), float(z_cm[-1])],
        aspect="auto",
        cmap=cmap,
        vmin=float(vmin),
        vmax=float(vmax),
    )
    ax.contour(
        x_cm,
        z_cm,
        vessel_mip,
        levels=[0.5],
        colors=["#6fd3ff"],
        linewidths=1.0,
    )
    ax.axhline(float(target_depth_cm), color="#ffffff", linestyle="--", linewidth=1.2)
    ax.set_title(title)
    ax.set_xlabel("x (cm)")
    ax.set_ylabel("z (cm)")
    cbar = fig.colorbar(img, ax=ax, shrink=0.92)
    cbar.set_label(cbar_label)
    fig.savefig(out_file, dpi=dpi)
    plt.close(fig)


def plot_three_y_band_field_slices(
    *,
    field: np.ndarray,
    vessel_mask: np.ndarray,
    x_cm: np.ndarray,
    y_cm: np.ndarray,
    z_cm: np.ndarray,
    band_centers_mm: List[Tuple[str, float]],
    band_half_width_mm: float,
    out_file: Path,
    dpi: int,
    cmap: str,
    vmin: float,
    vmax: float,
    title: str,
    cbar_label: str,
) -> None:
    dy_mm = float(abs(y_cm[1] - y_cm[0]) * 10.0) if len(y_cm) > 1 else 1.0
    half_bins = max(1, int(round(float(band_half_width_mm) / dy_mm)))

    fig, axes = plt.subplots(1, len(band_centers_mm), figsize=(5.2 * len(band_centers_mm), 4.9), constrained_layout=True)
    if len(band_centers_mm) == 1:
        axes = [axes]

    for ax, (label, center_mm) in zip(axes, band_centers_mm):
        idx = nearest_index(y_cm, float(center_mm) / 10.0)
        y_start = max(0, idx - half_bins)
        y_stop = min(field.shape[1], idx + half_bins + 1)
        slab = np.mean(field[:, y_start:y_stop, :], axis=1).T
        vessel_slab = np.any(vessel_mask[:, y_start:y_stop, :], axis=1).T.astype(np.float32)
        y_center_cm = float(np.mean(y_cm[y_start:y_stop]))

        img = ax.imshow(
            slab,
            origin="lower",
            extent=[float(x_cm[0]), float(x_cm[-1]), float(z_cm[0]), float(z_cm[-1])],
            aspect="auto",
            cmap=cmap,
            vmin=float(vmin),
            vmax=float(vmax),
        )
        ax.contour(
            x_cm,
            z_cm,
            vessel_slab,
            levels=[0.5],
            colors=["#6fd3ff"],
            linewidths=0.9,
        )
        ax.set_title(f"{label} band\nmean over y = {y_center_cm:.2f} cm")
        ax.set_xlabel("x (cm)")
        ax.set_ylabel("z (cm)")

    cbar = fig.colorbar(img, ax=axes, shrink=0.92)
    cbar.set_label(cbar_label)
    fig.suptitle(title, fontsize=14)
    fig.savefig(out_file, dpi=dpi)
    plt.close(fig)


def plot_3d_scalar_shells(
    *,
    field: np.ndarray,
    vessel_mask: np.ndarray,
    x_cm: np.ndarray,
    y_cm: np.ndarray,
    z_cm: np.ndarray,
    downsample: int,
    out_file: Path,
    dpi: int,
    thresholds: List[Tuple[float, str, int, str]],
    comparison: str,
    title: str,
) -> None:
    stride = max(1, int(downsample))
    field_ds = field[::stride, ::stride, ::stride]
    vessel_ds = vessel_mask[::stride, ::stride, ::stride]
    x_ds = x_cm[::stride]
    y_ds = y_cm[::stride]
    z_ds = z_cm[::stride]

    fig = plt.figure(figsize=(9.0, 7.0), constrained_layout=True)
    ax = fig.add_subplot(111, projection="3d")

    for threshold, color, max_points, label in thresholds:
        if comparison == ">=":
            mask = field_ds >= float(threshold)
        elif comparison == "<=":
            mask = field_ds <= float(threshold)
        else:
            raise ValueError("comparison must be '>=' or '<='.")
        pts = shell_points(mask)
        pts = subsample_points(pts, max_points=max_points)
        if len(pts) == 0:
            continue
        ax.scatter(
            x_ds[pts[:, 0]],
            y_ds[pts[:, 1]],
            z_ds[pts[:, 2]],
            s=3,
            alpha=0.18,
            c=color,
            label=label if threshold == thresholds[0][0] else None,
            depthshade=False,
        )

    vessel_pts = shell_points(vessel_ds)
    vessel_pts = subsample_points(vessel_pts, max_points=3500)
    if len(vessel_pts):
        ax.scatter(
            x_ds[vessel_pts[:, 0]],
            y_ds[vessel_pts[:, 1]],
            z_ds[vessel_pts[:, 2]],
            s=4,
            alpha=0.55,
            c="#4cc9f0",
            label="Vessel sinks",
            depthshade=False,
        )

    ax.set_xlabel("x (cm)")
    ax.set_ylabel("y (cm)")
    ax.set_zlabel("z (cm)")
    ax.set_title(title)
    ax.view_init(elev=24, azim=-58)
    ax.legend(loc="upper left")
    ax.set_box_aspect(
        (
            float(max(x_ds) - min(x_ds)),
            float(max(y_ds) - min(y_ds)),
            float(max(z_ds) - min(z_ds)),
        )
    )
    fig.savefig(out_file, dpi=dpi)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    summary = json.loads(args.phase10_summary.read_text(encoding="utf-8"))
    outdir = args.outdir or args.phase10_summary.parent
    outdir.mkdir(parents=True, exist_ok=True)

    fields = load_phase10_fields(summary)
    geometry = load_phase10_geometry(summary)
    dose_grid = geometry["dose_grid"]
    vessel_mask = geometry["vessel_mask"]
    x_cm = geometry["x_cm"]
    y_cm = geometry["y_cm"]
    z_cm = geometry["z_cm"]
    final_survival = fields["final_survival"]
    deff_grid = fields["deff_grid"]

    target_depth_cm = float(summary["center_metrics"]["sampled_depth_cm"])
    band_centers_mm = choose_y_band_centers_mm(list(summary["geometry"]["placed_spots"]))

    fig9 = Path(outdir) / "figure9_phase10_xz_mip.png"
    fig10 = Path(outdir) / "figure10_phase10_xz_yband_slices.png"
    fig11 = Path(outdir) / "figure11_phase10_3d_isodose_vessels.png"
    fig12 = Path(outdir) / "figure12_phase10_survival_xz_projection.png"
    fig13 = Path(outdir) / "figure13_phase10_survival_yband_slices.png"
    fig14 = Path(outdir) / "figure14_phase10_3d_survival_shells.png"
    fig15 = Path(outdir) / "figure15_phase10_deff_xz_projection.png"
    fig16 = Path(outdir) / "figure16_phase10_deff_yband_slices.png"
    fig17 = Path(outdir) / "figure17_phase10_3d_deff_shells.png"

    plot_xz_mip(
        dose_grid=dose_grid,
        vessel_mask=vessel_mask,
        x_cm=x_cm,
        z_cm=z_cm,
        target_depth_cm=float(target_depth_cm),
        out_file=fig9,
        dpi=int(args.dpi),
    )
    band_meta = plot_three_y_band_slices(
        dose_grid=dose_grid,
        vessel_mask=vessel_mask,
        x_cm=x_cm,
        y_cm=y_cm,
        z_cm=z_cm,
        band_centers_mm=band_centers_mm,
        band_half_width_mm=float(args.band_half_width_mm),
        out_file=fig10,
        dpi=int(args.dpi),
    )
    plot_3d_isodose_render(
        dose_grid=dose_grid,
        vessel_mask=vessel_mask,
        x_cm=x_cm,
        y_cm=y_cm,
        z_cm=z_cm,
        downsample=int(args.downsample),
        out_file=fig11,
        dpi=int(args.dpi),
    )
    plot_projected_field(
        field=final_survival,
        vessel_mask=vessel_mask,
        x_cm=x_cm,
        z_cm=z_cm,
        target_depth_cm=float(target_depth_cm),
        out_file=fig12,
        dpi=int(args.dpi),
        reduction="min",
        cmap="viridis",
        vmin=0.0,
        vmax=1.0,
        title="Figure 12: Phase 10 x-z minimum-survival projection over all y",
        cbar_label="Final survival fraction",
    )
    plot_three_y_band_field_slices(
        field=final_survival,
        vessel_mask=vessel_mask,
        x_cm=x_cm,
        y_cm=y_cm,
        z_cm=z_cm,
        band_centers_mm=band_centers_mm,
        band_half_width_mm=float(args.band_half_width_mm),
        out_file=fig13,
        dpi=int(args.dpi),
        cmap="viridis",
        vmin=0.0,
        vmax=1.0,
        title="Figure 13: Phase 10 survival slices through three y bands",
        cbar_label="Final survival fraction",
    )
    plot_3d_scalar_shells(
        field=final_survival,
        vessel_mask=vessel_mask,
        x_cm=x_cm,
        y_cm=y_cm,
        z_cm=z_cm,
        downsample=int(args.downsample),
        out_file=fig14,
        dpi=int(args.dpi),
        thresholds=[
            (0.60, "#f94144", 4500, "Low-survival shell"),
            (0.75, "#f8961e", 5000, "Intermediate-survival shell"),
            (0.90, "#90be6d", 5500, "High-survival shell"),
        ],
        comparison="<=",
        title="Figure 14: Phase 10 3D survival shells with vessel sinks",
    )
    plot_projected_field(
        field=deff_grid,
        vessel_mask=vessel_mask,
        x_cm=x_cm,
        z_cm=z_cm,
        target_depth_cm=float(target_depth_cm),
        out_file=fig15,
        dpi=int(args.dpi),
        reduction="max",
        cmap="magma",
        vmin=0.0,
        vmax=float(np.percentile(deff_grid, 99.0)),
        title="Figure 15: Phase 10 x-z maximum D_eff projection over all y",
        cbar_label="Effective dose (Gy)",
    )
    plot_three_y_band_field_slices(
        field=deff_grid,
        vessel_mask=vessel_mask,
        x_cm=x_cm,
        y_cm=y_cm,
        z_cm=z_cm,
        band_centers_mm=band_centers_mm,
        band_half_width_mm=float(args.band_half_width_mm),
        out_file=fig16,
        dpi=int(args.dpi),
        cmap="magma",
        vmin=0.0,
        vmax=float(np.percentile(deff_grid, 99.0)),
        title="Figure 16: Phase 10 D_eff slices through three y bands",
        cbar_label="Effective dose (Gy)",
    )
    plot_3d_scalar_shells(
        field=deff_grid,
        vessel_mask=vessel_mask,
        x_cm=x_cm,
        y_cm=y_cm,
        z_cm=z_cm,
        downsample=int(args.downsample),
        out_file=fig17,
        dpi=int(args.dpi),
        thresholds=[
            (2.0, "#7a5cff", 5000, "Low D_eff shell"),
            (5.0, "#ef476f", 4500, "Mid D_eff shell"),
            (8.0, "#ffd166", 4000, "High D_eff shell"),
        ],
        comparison=">=",
        title="Figure 17: Phase 10 3D D_eff shells with vessel sinks",
    )

    summary_out = Path(outdir) / "phase10_geometry_views_summary.json"
    payload = {
        "phase": "Phase 10",
        "description": "Improved geometry visualizations for the complex lattice surrogate.",
        "target_depth_cm": float(target_depth_cm),
        "band_half_width_mm": float(args.band_half_width_mm),
        "chosen_y_bands": band_meta,
        "outputs": {
            "figure_xz_mip": str(fig9),
            "figure_xz_ybands": str(fig10),
            "figure_3d_isodose": str(fig11),
            "figure_survival_projection": str(fig12),
            "figure_survival_ybands": str(fig13),
            "figure_3d_survival": str(fig14),
            "figure_deff_projection": str(fig15),
            "figure_deff_ybands": str(fig16),
            "figure_3d_deff": str(fig17),
        },
    }
    summary_out.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"Phase 10 x-z MIP: {fig9}")
    print(f"Phase 10 y-band slices: {fig10}")
    print(f"Phase 10 3D render: {fig11}")
    print(f"Phase 10 survival projection: {fig12}")
    print(f"Phase 10 survival y-band slices: {fig13}")
    print(f"Phase 10 survival 3D render: {fig14}")
    print(f"Phase 10 Deff projection: {fig15}")
    print(f"Phase 10 Deff y-band slices: {fig16}")
    print(f"Phase 10 Deff 3D render: {fig17}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
