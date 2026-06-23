#!/usr/bin/env python3
"""Preview vessel-seeking Phase 17 lattice candidates over the vascular sink field."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from run_phase14_detailed_headneck_voxel_lattice import build_detailed_plan_phantom
from run_phase15_detailed_headneck_bioaware import (
    build_anatomical_biology_tensors,
    build_args_from_summary,
    load_phase14_summary,
)
from run_phase16_bio_guided_lattice_optimization import (
    build_candidate_centers,
    build_safe_candidate_centers,
    build_structure_points_mm,
)
from run_phase17_fraction_aware_bio_optimization import (
    build_strategy_candidate_sets,
    compute_guidance_oar_weights,
    score_candidate_centers,
)

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
except Exception as exc:  # pragma: no cover
    raise RuntimeError("matplotlib is required for vascular sink preview generation.") from exc


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Generate a visual preview of Phase 17 vascular sink-hugging lattice candidates."
    )
    parser.add_argument(
        "--phase14-root",
        type=Path,
        default=root / "runs" / "linac_6mv_headneck_detailed_voxel_lattice_sfrt_directplan",
        help="Detailed direct-plan run root used to rebuild the phantom and baseline lattice.",
    )
    parser.add_argument(
        "--out-file",
        type=Path,
        default=root / "runs" / "linac_6mv_headneck_detailed_voxel_lattice_sfrt_phase17_vascular_sink_hugging_1e5" / "figure0_vascular_sink_candidate_preview.png",
        help="Output figure path.",
    )
    parser.add_argument("--top-candidates", type=int, default=3, help="Number of top vascular candidates to preview.")
    parser.add_argument("--candidate-step-mm", type=float, default=6.0)
    parser.add_argument("--spot-radius-mm", type=float, default=8.0)
    parser.add_argument("--min-spot-spacing-mm", type=float, default=18.0)
    parser.add_argument("--num-spots", type=int, default=4)
    parser.add_argument("--candidate-top-k-centers", type=int, default=24)
    parser.add_argument("--candidate-plan-limit", type=int, default=6)
    parser.add_argument("--vascular-candidate-multiplier", type=int, default=2)
    parser.add_argument("--target-effective-gy-per-fraction", type=float, default=20.0)
    parser.add_argument("--target-need-cap-gy", type=float, default=10.0)
    parser.add_argument("--hard-min-dist-cord-mm", type=float, default=55.0)
    parser.add_argument("--hard-min-dist-brainstem-mm", type=float, default=50.0)
    parser.add_argument("--hard-cumulative-cord-d2-eff-gy", type=float, default=85.0)
    parser.add_argument("--hard-cumulative-brainstem-d2-eff-gy", type=float, default=30.0)
    parser.add_argument("--hard-cumulative-parotid-r-mean-eff-gy", type=float, default=60.0)
    parser.add_argument("--hard-cumulative-thyroid-mean-eff-gy", type=float, default=50.0)
    parser.add_argument("--tumor-cytokine-multiplier", type=float, default=2.0)
    parser.add_argument("--hypoxic-ros-scale", type=float, default=0.12)
    parser.add_argument("--hypoxic-cytokine-multiplier", type=float, default=2.7)
    parser.add_argument("--artery-ros-uptake", type=float, default=0.05)
    parser.add_argument("--artery-cyto-uptake", type=float, default=0.70)
    parser.add_argument("--vein-ros-uptake", type=float, default=0.05)
    parser.add_argument("--vein-cyto-uptake", type=float, default=0.90)
    parser.add_argument("--course-strategy", type=str, default="vascular_sink_hugging")
    parser.add_argument("--dpi", type=int, default=260)
    return parser.parse_args()


def zero_metrics() -> dict[str, dict[str, float]]:
    return {
        "SPINAL_CORD": {"d2_gy": 0.0},
        "BRAINSTEM": {"d2_gy": 0.0},
        "PAROTID_R": {"mean_gy": 0.0},
        "PAROTID_L": {"mean_gy": 0.0},
        "THYROID": {"mean_gy": 0.0},
        "PARATHYROIDS": {"mean_gy": 0.0},
        "BRAIN": {"mean_gy": 0.0},
        "BLOOD_BRAIN_BARRIER": {"mean_gy": 0.0},
        "MANDIBLE": {"mean_gy": 0.0},
    }


def render_row(
    ax_coronal,
    ax_axial,
    *,
    cyto_uptake: np.ndarray,
    structures: dict[str, np.ndarray],
    axes_mm: dict[str, np.ndarray],
    baseline_spots_mm: np.ndarray,
    candidate_spots_mm: np.ndarray,
    row_title: str,
    heuristic_score: float | None,
    rank: int | None,
    vmax: float,
) -> None:
    y_index = int(np.argmin(np.abs(axes_mm["y"] - float(np.mean(candidate_spots_mm[:, 1])))))
    z_index = int(np.argmin(np.abs(axes_mm["z"] - float(np.mean(candidate_spots_mm[:, 2])))))

    x_cm = axes_mm["x"] / 10.0
    y_cm = axes_mm["y"] / 10.0
    z_cm = axes_mm["z"] / 10.0

    panels = [
        (
            ax_coronal,
            cyto_uptake[:, :, z_index],
            structures["ARTERIES"][:, :, z_index],
            structures["VEINS"][:, :, z_index],
            structures["PTV"][:, :, z_index],
            structures["GTV"][:, :, z_index],
            x_cm,
            y_cm,
            "Coronal",
            "x (cm)",
            "y (cm)",
            baseline_spots_mm[:, [0, 1]] / 10.0,
            candidate_spots_mm[:, [0, 1]] / 10.0,
        ),
        (
            ax_axial,
            cyto_uptake[:, y_index, :],
            structures["ARTERIES"][:, y_index, :],
            structures["VEINS"][:, y_index, :],
            structures["PTV"][:, y_index, :],
            structures["GTV"][:, y_index, :],
            x_cm,
            z_cm,
            "Axial-style",
            "x (cm)",
            "z (cm)",
            baseline_spots_mm[:, [0, 2]] / 10.0,
            candidate_spots_mm[:, [0, 2]] / 10.0,
        ),
    ]

    for ax, image, artery_slice, vein_slice, ptv_slice, gtv_slice, axis_a, axis_b, panel_name, xlabel, ylabel, baseline_xy, cand_xy in panels:
        im = ax.imshow(
            image.T,
            origin="lower",
            extent=[float(axis_a[0]), float(axis_a[-1]), float(axis_b[0]), float(axis_b[-1])],
            cmap="viridis",
            vmin=0.0,
            vmax=vmax,
            alpha=0.96,
        )
        ax.contour(axis_a, axis_b, artery_slice.T.astype(float), levels=[0.5], colors=["#D62828"], linewidths=1.2)
        ax.contour(axis_a, axis_b, vein_slice.T.astype(float), levels=[0.5], colors=["#277DA1"], linewidths=1.2)
        ax.contour(axis_a, axis_b, ptv_slice.T.astype(float), levels=[0.5], colors=["cyan"], linewidths=1.1)
        ax.contour(axis_a, axis_b, gtv_slice.T.astype(float), levels=[0.5], colors=["magenta"], linewidths=0.9, linestyles="--")
        ax.scatter(
            baseline_xy[:, 0],
            baseline_xy[:, 1],
            facecolors="none",
            edgecolors="white",
            s=46,
            linewidths=1.0,
            zorder=5,
        )
        ax.scatter(
            cand_xy[:, 0],
            cand_xy[:, 1],
            c="#F4D35E",
            edgecolors="black",
            s=52,
            linewidths=0.6,
            zorder=6,
        )
        for idx, (px, py) in enumerate(cand_xy, start=1):
            ax.text(px + 0.18, py + 0.10, f"{idx}", color="white", fontsize=8, weight="bold")
        subtitle = panel_name
        if rank is not None and heuristic_score is not None:
            subtitle += f" | rank {rank} | h={heuristic_score:.1f}"
        ax.set_title(f"{row_title}: {subtitle}", fontsize=10)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(False)

    return im


def main() -> int:
    args = parse_args()
    args.out_file.parent.mkdir(parents=True, exist_ok=True)

    phase14_summary = load_phase14_summary(args.phase14_root.resolve())
    baseline_spots_mm = np.asarray(phase14_summary["plan"]["spot_centers_mm"], dtype=np.float32)
    phantom = build_detailed_plan_phantom(build_args_from_summary(phase14_summary))
    structures = phantom["structures"]
    axes_mm = phantom["axes_mm"]

    candidate_indices = build_candidate_centers(
        structures,
        axes_mm,
        spot_radius_mm=float(args.spot_radius_mm),
        candidate_step_mm=float(args.candidate_step_mm),
    )
    structure_points_mm = build_structure_points_mm(
        structures,
        axes_mm,
        [
            "SPINAL_CORD",
            "BRAINSTEM",
            "PAROTID_R",
            "PAROTID_L",
            "THYROID",
            "PARATHYROIDS",
            "BRAIN",
            "BLOOD_BRAIN_BARRIER",
            "MANDIBLE",
            "ARTERIES",
            "VEINS",
        ],
    )
    safe_candidate_indices = build_safe_candidate_centers(
        candidate_indices,
        axes_mm,
        structure_points_mm,
        hard_min_dist_cord_mm=float(args.hard_min_dist_cord_mm),
        hard_min_dist_brainstem_mm=float(args.hard_min_dist_brainstem_mm),
    )
    uptake_tensor, _, _, _ = build_anatomical_biology_tensors(args, structures)
    cyto_uptake = uptake_tensor[1]
    vessel_coords = np.argwhere(structures["ARTERIES"] | structures["VEINS"])
    vessel_coords_mm = np.column_stack(
        [
            axes_mm["x"][vessel_coords[:, 0]],
            axes_mm["y"][vessel_coords[:, 1]],
            axes_mm["z"][vessel_coords[:, 2]],
        ]
    ).astype(np.float32)

    scored_centers = score_candidate_centers(
        current_cumulative_effective_dose=np.zeros_like(phantom["tag_grid"], dtype=np.float32),
        current_fraction_idx=2,
        structures=structures,
        axes_mm=axes_mm,
        uptake_tensor=uptake_tensor,
        candidate_indices=safe_candidate_indices,
        target_effective_gy_per_fraction=float(args.target_effective_gy_per_fraction),
        target_need_cap_gy=float(args.target_need_cap_gy),
        oar_weights=compute_guidance_oar_weights(zero_metrics(), args)[0],
        structure_points_mm=structure_points_mm,
        vessel_coords_mm=vessel_coords_mm,
        history_counts={},
    )

    candidate_sets = build_strategy_candidate_sets(
        args=args,
        fraction_idx=2,
        baseline_spots=[tuple(float(v) for v in row) for row in baseline_spots_mm],
        current_repeat_spots=[tuple(float(v) for v in row) for row in baseline_spots_mm],
        strategy_state={"pattern_a": [tuple(float(v) for v in row) for row in baseline_spots_mm]},
        scored_centers=scored_centers,
        axes_mm=axes_mm,
        candidate_indices=safe_candidate_indices,
        num_spots=int(args.num_spots),
        min_spacing_mm=float(args.min_spot_spacing_mm),
        top_k_centers=int(args.candidate_top_k_centers),
        candidate_plan_limit=max(int(args.candidate_plan_limit), int(args.top_candidates)),
    )
    candidate_sets = candidate_sets[: int(args.top_candidates)]

    vmax = float(np.max(cyto_uptake)) if float(np.max(cyto_uptake)) > 0 else 1.0
    num_rows = 1 + len(candidate_sets)
    fig, axes = plt.subplots(num_rows, 2, figsize=(13.5, 4.6 * num_rows), constrained_layout=True)
    if num_rows == 1:
        axes = np.asarray([axes])

    im = render_row(
        axes[0, 0],
        axes[0, 1],
        cyto_uptake=cyto_uptake,
        structures=structures,
        axes_mm=axes_mm,
        baseline_spots_mm=baseline_spots_mm,
        candidate_spots_mm=baseline_spots_mm,
        row_title="Baseline reference",
        heuristic_score=None,
        rank=None,
        vmax=vmax,
    )
    for row_idx, candidate in enumerate(candidate_sets, start=1):
        candidate_spots = np.asarray(candidate["spots_mm"], dtype=np.float32)
        render_row(
            axes[row_idx, 0],
            axes[row_idx, 1],
            cyto_uptake=cyto_uptake,
            structures=structures,
            axes_mm=axes_mm,
            baseline_spots_mm=baseline_spots_mm,
            candidate_spots_mm=candidate_spots,
            row_title=str(candidate.get("candidate_origin", f"candidate_{row_idx}")),
            heuristic_score=float(candidate.get("heuristic_score", 0.0)),
            rank=row_idx,
            vmax=vmax,
        )

    fig.suptitle(
        "Phase 17 vascular sink-hugging preview: baseline lattice versus top vessel-seeking candidate sets",
        fontsize=14,
    )
    legend_handles = [
        Line2D([0], [0], color="#D62828", lw=1.2, label="Arteries"),
        Line2D([0], [0], color="#277DA1", lw=1.2, label="Veins"),
        Line2D([0], [0], color="cyan", lw=1.1, label="PTV"),
        Line2D([0], [0], color="magenta", lw=0.9, linestyle="--", label="GTV"),
        Line2D([0], [0], marker="o", markerfacecolor="none", markeredgecolor="white", lw=0, markersize=7, label="Baseline spots"),
        Line2D([0], [0], marker="o", color="#F4D35E", markeredgecolor="black", lw=0, markersize=7, label="Vessel-seeking spots"),
    ]
    axes[0, 1].legend(handles=legend_handles, loc="lower right", fontsize=8, framealpha=0.92)
    fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.016, pad=0.012, label="Cytokine uptake rate")
    fig.savefig(args.out_file, dpi=int(args.dpi))
    plt.close(fig)
    print(f"=== PHASE 17 VASCULAR PREVIEW WRITTEN TO {args.out_file} ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
