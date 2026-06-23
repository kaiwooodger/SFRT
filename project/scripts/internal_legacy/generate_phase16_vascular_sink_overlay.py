#!/usr/bin/env python3
"""Visualize the vessel-derived cytokine sink field with plan overlays."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from run_phase14_detailed_headneck_voxel_lattice import build_detailed_plan_phantom
from run_phase15_detailed_headneck_bioaware import (
    build_anatomical_biology_tensors,
    build_args_from_summary,
    load_phase14_summary,
)

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
except Exception as exc:  # pragma: no cover
    raise RuntimeError("matplotlib is required for sink-overlay visualization.") from exc


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Generate a vessel + cytokine sink + lattice-spot overlay for the best biology-aware plan."
    )
    parser.add_argument(
        "--phase14-root",
        type=Path,
        default=root / "runs" / "linac_6mv_headneck_detailed_voxel_lattice_sfrt_directplan",
        help="Run root containing the detailed phantom summary.",
    )
    parser.add_argument(
        "--bioopt-root",
        type=Path,
        default=root / "runs" / "linac_6mv_headneck_detailed_voxel_lattice_sfrt_bioopt_v2",
        help="Phase 16 optimization run root.",
    )
    parser.add_argument(
        "--placement-name",
        type=str,
        default="feedback_02",
        help="Placement name to visualize; defaults to the best current biology-aware plan.",
    )
    parser.add_argument(
        "--out-file",
        type=Path,
        default=None,
        help="Optional output image path.",
    )
    parser.add_argument("--tumor-cytokine-multiplier", type=float, default=2.0)
    parser.add_argument("--hypoxic-ros-scale", type=float, default=0.12)
    parser.add_argument("--hypoxic-cytokine-multiplier", type=float, default=2.7)
    parser.add_argument("--artery-ros-uptake", type=float, default=0.05)
    parser.add_argument("--artery-cyto-uptake", type=float, default=0.70)
    parser.add_argument("--vein-ros-uptake", type=float, default=0.05)
    parser.add_argument("--vein-cyto-uptake", type=float, default=0.90)
    parser.add_argument("--dpi", type=int, default=260)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    out_file = args.out_file
    if out_file is None:
        out_file = (
            args.bioopt_root
            / "candidate_plan_tradeoff_report"
            / f"figure6_vascular_sink_overlay_{args.placement_name}.png"
        )
    out_file.parent.mkdir(parents=True, exist_ok=True)

    phase14_summary = load_phase14_summary(args.phase14_root.resolve())
    phantom = build_detailed_plan_phantom(build_args_from_summary(phase14_summary))
    structures = phantom["structures"]
    axes_mm = phantom["axes_mm"]

    optimization = json.loads((args.bioopt_root.resolve() / "optimization_results.json").read_text(encoding="utf-8"))
    placement = None
    for row in optimization["placements"]:
        if row["placement_name"] == args.placement_name:
            placement = row
            break
    if placement is None:
        raise ValueError(f"Could not find placement '{args.placement_name}' in optimization_results.json")

    spot_centers_mm = np.asarray(placement["spot_centers_mm"], dtype=np.float32)
    uptake_tensor, _, _, _ = build_anatomical_biology_tensors(args, structures)
    cyto_uptake = uptake_tensor[1]

    y_index = int(np.argmin(np.abs(axes_mm["y"] - float(np.mean(spot_centers_mm[:, 1])))))
    z_index = int(np.argmin(np.abs(axes_mm["z"] - float(np.mean(spot_centers_mm[:, 2])))))

    x_cm = axes_mm["x"] / 10.0
    y_cm = axes_mm["y"] / 10.0
    z_cm = axes_mm["z"] / 10.0
    vmax = float(np.max(cyto_uptake)) if float(np.max(cyto_uptake)) > 0 else 1.0

    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.6), constrained_layout=True)
    panels = [
        (
            axes[0],
            cyto_uptake[:, :, z_index],
            structures["ARTERIES"][:, :, z_index],
            structures["VEINS"][:, :, z_index],
            structures["PTV"][:, :, z_index],
            structures["GTV"][:, :, z_index],
            x_cm,
            y_cm,
            "Coronal sink overlay",
            "x (cm)",
            "y (cm)",
            "x-y",
        ),
        (
            axes[1],
            cyto_uptake[:, y_index, :],
            structures["ARTERIES"][:, y_index, :],
            structures["VEINS"][:, y_index, :],
            structures["PTV"][:, y_index, :],
            structures["GTV"][:, y_index, :],
            x_cm,
            z_cm,
            "Axial-style sink overlay",
            "x (cm)",
            "z (cm)",
            "x-z",
        ),
    ]

    for ax, image, artery_slice, vein_slice, ptv_slice, gtv_slice, axis_a, axis_b, title, xlabel, ylabel, mode in panels:
        im = ax.imshow(
            image.T,
            origin="lower",
            extent=[float(axis_a[0]), float(axis_a[-1]), float(axis_b[0]), float(axis_b[-1])],
            cmap="viridis",
            vmin=0.0,
            vmax=vmax,
            alpha=0.96,
        )
        ax.contour(axis_a, axis_b, artery_slice.T.astype(float), levels=[0.5], colors=["#D62828"], linewidths=1.3)
        ax.contour(axis_a, axis_b, vein_slice.T.astype(float), levels=[0.5], colors=["#277DA1"], linewidths=1.3)
        ax.contour(axis_a, axis_b, ptv_slice.T.astype(float), levels=[0.5], colors=["cyan"], linewidths=1.2)
        ax.contour(axis_a, axis_b, gtv_slice.T.astype(float), levels=[0.5], colors=["magenta"], linewidths=1.0, linestyles="--")
        if mode == "x-y":
            ax.scatter(spot_centers_mm[:, 0] / 10.0, spot_centers_mm[:, 1] / 10.0, c="yellow", s=42, edgecolors="black", linewidths=0.6, zorder=6)
        else:
            ax.scatter(spot_centers_mm[:, 0] / 10.0, spot_centers_mm[:, 2] / 10.0, c="yellow", s=42, edgecolors="black", linewidths=0.6, zorder=6)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03, label="Cytokine uptake rate")

    fig.suptitle(
        f"Explicit vascular sink overlay for {args.placement_name}: vessel masks drive the cytokine sink field",
        fontsize=13,
    )
    legend_handles = [
        Line2D([0], [0], color="#D62828", lw=1.3, label="Arteries"),
        Line2D([0], [0], color="#277DA1", lw=1.3, label="Veins"),
        Line2D([0], [0], color="cyan", lw=1.2, label="PTV"),
        Line2D([0], [0], color="magenta", lw=1.0, linestyle="--", label="GTV"),
        Line2D([0], [0], marker="o", color="yellow", markeredgecolor="black", lw=0, markersize=7, label="Lattice spots"),
    ]
    axes[1].legend(handles=legend_handles, loc="lower right", fontsize=8, framealpha=0.92)
    fig.savefig(out_file, dpi=int(args.dpi))
    plt.close(fig)
    print(f"=== VASCULAR SINK OVERLAY WRITTEN TO {out_file} ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
