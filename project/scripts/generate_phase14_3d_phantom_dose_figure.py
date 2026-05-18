#!/usr/bin/env python3
"""Generate a 3D detailed phantom figure with a 3D physical dose rendering."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Iterable, List, Tuple

import numpy as np
from scipy import ndimage

from analyze_topas_outputs import load_topas_grid
from run_phase14_detailed_headneck_voxel_lattice import build_detailed_plan_phantom

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
            "Render the detailed head-and-neck voxel phantom in 3D and overlay "
            "the physical dose as 3D isodose shells."
        )
    )
    parser.add_argument(
        "--run-root",
        type=Path,
        default=root / "runs" / "linac_6mv_headneck_detailed_voxel_lattice_sfrt_directplan",
        help="Detailed direct-plan run root containing analysis and case outputs.",
    )
    parser.add_argument(
        "--out-file",
        type=Path,
        default=None,
        help="Optional output path. Defaults to <run-root>/analysis/figure4_3d_phantom_dose_profile.png.",
    )
    parser.add_argument(
        "--body-points",
        type=int,
        default=14000,
        help="Maximum number of rendered body-shell points per panel.",
    )
    parser.add_argument(
        "--structure-points",
        type=int,
        default=7000,
        help="Maximum number of rendered structure-shell points per structure.",
    )
    parser.add_argument(
        "--dose-points",
        type=int,
        default=9000,
        help="Maximum number of rendered points per dose shell.",
    )
    parser.add_argument("--dpi", type=int, default=280, help="Figure DPI.")
    return parser.parse_args()


def build_args_from_summary(summary: Dict[str, object]) -> SimpleNamespace:
    phantom = summary["phantom"]
    return SimpleNamespace(
        size_x_cm=float(phantom["size_cm"][0]),
        size_y_cm=float(phantom["size_cm"][1]),
        size_z_cm=float(phantom["size_cm"][2]),
        voxel_mm=float(phantom["voxel_size_mm"][0]),
    )


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


def plot_shell(
    ax,
    *,
    mask: np.ndarray,
    axes_cm: Dict[str, np.ndarray],
    color: str,
    alpha: float,
    size: float,
    max_points: int,
    label: str | None = None,
) -> None:
    pts = subsample_points(shell_points(mask), max_points=max_points)
    if len(pts) == 0:
        return
    ax.scatter(
        axes_cm["x"][pts[:, 0]],
        axes_cm["y"][pts[:, 1]],
        axes_cm["z"][pts[:, 2]],
        s=size,
        alpha=alpha,
        c=color,
        label=label,
        depthshade=False,
    )


def set_equal_box(ax, axes_cm: Dict[str, np.ndarray]) -> None:
    ax.set_box_aspect(
        (
            float(axes_cm["x"][-1] - axes_cm["x"][0]),
            float(axes_cm["y"][-1] - axes_cm["y"][0]),
            float(axes_cm["z"][-1] - axes_cm["z"][0]),
        )
    )
    ax.set_xlabel("x (cm)")
    ax.set_ylabel("y (cm)")
    ax.set_zlabel("z (cm)")


def dose_shells_from_prescription(
    prescription_gy: float,
    max_dose_gy: float,
) -> List[Tuple[float, str, str]]:
    shell_specs = [
        (8.0 * prescription_gy, "#ffd166", f"{8.0 * prescription_gy:.0f} Gy shell"),
        (4.0 * prescription_gy, "#ef476f", f"{4.0 * prescription_gy:.0f} Gy shell"),
        (2.0 * prescription_gy, "#ff7f50", f"{2.0 * prescription_gy:.0f} Gy shell"),
    ]
    valid = [(level, color, label) for level, color, label in shell_specs if level <= max_dose_gy]
    if valid:
        return valid
    return [(0.8 * max_dose_gy, "#ffd166", f"{0.8 * max_dose_gy:.1f} Gy shell")]


def lattice_spots_from_summary(summary: Dict[str, object]) -> List[Tuple[float, float, float]]:
    raw = summary.get("plan", {}).get("spot_centers_mm", [])
    if not raw:
        return []
    return [tuple(float(v) for v in center) for center in raw]


def main() -> int:
    args = parse_args()
    summary_path = args.run_root / "analysis" / "phase14_detailed_headneck_summary.json"
    dose_csv = args.run_root / "case" / "dosedata.csv"
    out_file = args.out_file or (args.run_root / "analysis" / "figure4_3d_phantom_dose_profile.png")

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    phantom = build_detailed_plan_phantom(build_args_from_summary(summary))
    structures = phantom["structures"]
    axes_mm = phantom["axes_mm"]
    axes_cm = {key: np.asarray(values, dtype=np.float32) / 10.0 for key, values in axes_mm.items()}

    dose_grid_raw, _ = load_topas_grid(dose_csv)
    physical_scale_factor = float(summary["physical_scale_factor"])
    dose_grid = np.asarray(dose_grid_raw, dtype=np.float32) * physical_scale_factor
    max_dose_gy = float(np.max(dose_grid))
    rx_gy = float(summary["prescription_gy"])

    fig = plt.figure(figsize=(14.5, 6.8), constrained_layout=True)
    ax1 = fig.add_subplot(121, projection="3d")
    ax2 = fig.add_subplot(122, projection="3d")

    body_mask = structures["BODY"] & ~structures["AIRWAY"]
    common_shells: Iterable[Tuple[str, np.ndarray, str, float, float, int]] = [
        ("Body", body_mask, "#c7ccd6", 0.035, 1.2, int(args.body_points)),
        ("Brain", structures["BRAIN"], "#8ecae6", 0.12, 1.8, int(args.structure_points)),
        ("Spinal cord", structures["SPINAL_CORD"], "#f1f5f9", 0.55, 3.2, int(args.structure_points)),
        ("PTV", structures["PTV"], "#41ead4", 0.24, 2.8, int(args.structure_points)),
        ("GTV", structures["GTV"], "#ff4fa3", 0.42, 3.4, int(args.structure_points)),
        ("Arteries", structures["ARTERIES"], "#ff6b6b", 0.38, 2.3, int(args.structure_points)),
        ("Veins", structures["VEINS"], "#4dabf7", 0.30, 2.3, int(args.structure_points)),
    ]

    for label, mask, color, alpha, size, max_points in common_shells:
        plot_shell(
            ax1,
            mask=mask,
            axes_cm=axes_cm,
            color=color,
            alpha=alpha,
            size=size,
            max_points=max_points,
            label=label,
        )

    for label, mask, color, alpha, size, max_points in common_shells:
        dose_panel_label = label if label in {"Body", "PTV", "GTV"} else None
        plot_shell(
            ax2,
            mask=mask,
            axes_cm=axes_cm,
            color=color,
            alpha=0.03 if label == "Body" else alpha * 0.55,
            size=size,
            max_points=max_points,
            label=dose_panel_label,
        )

    for level_gy, color, label in dose_shells_from_prescription(rx_gy, max_dose_gy):
        plot_shell(
            ax2,
            mask=dose_grid >= float(level_gy),
            axes_cm=axes_cm,
            color=color,
            alpha=0.18,
            size=2.2,
            max_points=int(args.dose_points),
            label=label,
        )

    lattice_spots = lattice_spots_from_summary(summary)
    if lattice_spots:
        xs = np.array([spot[0] for spot in lattice_spots], dtype=np.float32) / 10.0
        ys = np.array([spot[1] for spot in lattice_spots], dtype=np.float32) / 10.0
        zs = np.array([spot[2] for spot in lattice_spots], dtype=np.float32) / 10.0
        for ax in (ax1, ax2):
            ax.scatter(xs, ys, zs, s=54, c="#ffdd57", edgecolors="black", linewidths=0.5, depthshade=False, label="Lattice vertices" if ax is ax1 else None)

    ax1.set_title("3D detailed voxel phantom anatomy")
    ax2.set_title("3D physical dose profile on the detailed phantom")
    for ax in (ax1, ax2):
        set_equal_box(ax, axes_cm)
        ax.view_init(elev=23, azim=-58)
        ax.grid(False)

    ax1.legend(loc="upper left", fontsize=8)
    ax2.legend(loc="upper left", fontsize=8)
    fig.suptitle(
        (
            "Detailed head-and-neck phantom and 3D physical dose rendering "
            f"(Rx D95 = {rx_gy:.1f} Gy, dose max = {max_dose_gy:.1f} Gy)"
        ),
        fontsize=14,
    )
    out_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_file, dpi=int(args.dpi))
    plt.close(fig)

    summary_out = out_file.with_suffix(".json")
    summary_payload = {
        "run_root": str(args.run_root),
        "output_figure": str(out_file),
        "prescription_gy": rx_gy,
        "dose_max_gy": max_dose_gy,
        "dose_shells_gy": [float(level) for level, _, _ in dose_shells_from_prescription(rx_gy, max_dose_gy)],
        "lattice_spot_count": int(len(lattice_spots)),
        "structure_volumes_cc": phantom["meta"]["structure_volumes_cc"],
    }
    summary_out.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")
    print(out_file)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
