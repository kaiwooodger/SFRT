#!/usr/bin/env python
"""Generate Phase 9 lattice-geometry heat maps from the rebuilt 3D dose grids."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np

from analyze_250mev_sfrt_plan import centered_axis_cm, nearest_index
from geometry_generators import generate_3d_lattice_geometry

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle
except Exception as exc:  # pragma: no cover
    raise RuntimeError("matplotlib is required for figure generation") from exc


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description=(
            "Build heat-map visualizations of the Phase 9 true-valley 3D lattice geometry."
        )
    )
    parser.add_argument(
        "--phase9-summary",
        type=Path,
        default=root
        / "runs"
        / "linac_6mv_polyenergetic_clinical_sfrt"
        / "analysis_phase9_holdout_3d_lattice"
        / "phase9_holdout_3d_lattice_summary.json",
        help="Phase 9 summary JSON containing the locked geometry settings.",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=None,
        help="Optional output directory. Defaults to the folder containing --phase9-summary.",
    )
    parser.add_argument("--dpi", type=int, default=250, help="Figure DPI.")
    return parser.parse_args()


def plot_heatmaps(
    panels: List[Dict[str, object]],
    *,
    target_depth_cm: float,
    vessel_radius_cm: float,
    out_file: Path,
    dpi: int,
) -> None:
    max_dose = max(float(np.max(panel["dose_grid"])) for panel in panels)
    fig, axes = plt.subplots(2, len(panels), figsize=(5.0 * len(panels), 8.8), constrained_layout=True)

    for col, panel in enumerate(panels):
        dose_grid = np.asarray(panel["dose_grid"], dtype=np.float32)
        x_cm = np.asarray(panel["x_cm"], dtype=np.float32)
        y_cm = np.asarray(panel["y_cm"], dtype=np.float32)
        z_cm = np.asarray(panel["z_cm"], dtype=np.float32)
        pitch_mm = float(panel["pitch_mm"])
        z_idx = int(panel["target_depth_idx"])
        center_y = dose_grid.shape[1] // 2

        xy_slice = dose_grid[:, :, z_idx].T
        xz_slice = dose_grid[:, center_y, :].T

        xy_extent = [float(x_cm[0]), float(x_cm[-1]), float(y_cm[0]), float(y_cm[-1])]
        xz_extent = [float(x_cm[0]), float(x_cm[-1]), float(z_cm[0]), float(z_cm[-1])]

        ax_xy = axes[0, col]
        img_xy = ax_xy.imshow(
            xy_slice,
            origin="lower",
            extent=xy_extent,
            aspect="equal",
            cmap="inferno",
            vmin=0.0,
            vmax=max_dose,
        )
        ax_xy.add_patch(
            Circle((0.0, 0.0), radius=float(vessel_radius_cm), facecolor="none", edgecolor="#7FDBFF", linewidth=1.8, linestyle="--")
        )
        ax_xy.set_title(f"Pitch = {pitch_mm:.0f} mm\nx-y at z = {target_depth_cm:.2f} cm")
        ax_xy.set_xlabel("x (cm)")
        if col == 0:
            ax_xy.set_ylabel("y (cm)")

        ax_xz = axes[1, col]
        img_xz = ax_xz.imshow(
            xz_slice,
            origin="lower",
            extent=xz_extent,
            aspect="auto",
            cmap="inferno",
            vmin=0.0,
            vmax=max_dose,
        )
        ax_xz.axvspan(-float(vessel_radius_cm), float(vessel_radius_cm), color="#7FDBFF", alpha=0.15)
        ax_xz.axhline(float(target_depth_cm), color="#ffffff", linestyle="--", linewidth=1.5)
        ax_xz.set_title(f"Pitch = {pitch_mm:.0f} mm\nx-z at center y")
        ax_xz.set_xlabel("x (cm)")
        if col == 0:
            ax_xz.set_ylabel("z (cm)")

    cbar = fig.colorbar(img_xz, ax=axes.ravel().tolist(), shrink=0.92)
    cbar.set_label("Dose (Gy)")
    fig.suptitle("Figure 2: Phase 9 3D lattice geometry heat maps", fontsize=15)
    fig.savefig(out_file, dpi=dpi)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    summary = json.loads(args.phase9_summary.read_text(encoding="utf-8"))
    outdir = args.outdir or args.phase9_summary.parent
    outdir.mkdir(parents=True, exist_ok=True)

    inputs = summary["inputs"]
    target_depth_cm = float(summary["target_depth_cm"])
    vessel_radius_mm = float(inputs["vessel_radius_mm"])

    panels: List[Dict[str, object]] = []
    for pitch_mm in [float(value) for value in inputs["pitches_mm"]]:
        dose_grid, _, _, _, meta = generate_3d_lattice_geometry(
            pitch_mm=float(pitch_mm),
            csv=Path(str(inputs["csv"])),
            prescribed_peak_dose_gy=float(inputs["prescribed_peak_dose_gy"]),
            uniform_dose_floor_fraction=float(inputs["uniform_dose_floor_fraction"]),
            n_beams_x=int(inputs["n_beams_x"]),
            n_beams_y=int(inputs["n_beams_y"]),
            x_shift_fraction_of_pitch=float(inputs["x_shift_fraction_of_pitch"]),
            vessel_radius_mm=float(inputs["vessel_radius_mm"]),
            vessel_uptake=float(inputs["vessel_uptake"]),
        )
        voxel_size_mm = tuple(float(value) for value in meta["voxel_size_mm"])
        x_cm = np.asarray(meta["x_cm"], dtype=np.float32)
        y_cm = centered_axis_cm(dose_grid.shape[1], voxel_size_mm[1] / 10.0)
        z_cm = np.asarray(meta["z_cm"], dtype=np.float32)
        panels.append(
            {
                "pitch_mm": float(pitch_mm),
                "dose_grid": dose_grid,
                "x_cm": x_cm,
                "y_cm": y_cm,
                "z_cm": z_cm,
                "target_depth_idx": nearest_index(z_cm, float(target_depth_cm)),
            }
        )

    figure_file = Path(outdir) / "figure2_phase9_lattice_geometry_heatmaps.png"
    plot_heatmaps(
        panels,
        target_depth_cm=float(target_depth_cm),
        vessel_radius_cm=float(vessel_radius_mm) / 10.0,
        out_file=figure_file,
        dpi=int(args.dpi),
    )

    print(f"Phase 9 heat-map figure: {figure_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
