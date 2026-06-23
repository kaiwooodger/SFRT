#!/usr/bin/env python
"""Phase 11A Sweep 3: radii ladder with companion-vessel layouts."""

from __future__ import annotations

import argparse
import csv
import json
import multiprocessing as mp
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

from analyze_250mev_sfrt_plan import nearest_index
from bystander_multispecies_pde_solver import (
    calculate_effective_dose,
    calculate_phase7_survival,
    calculate_systemic_immune_penalty,
    run_pde_temporal_integration,
)
from geometry_generators import (
    default_complex_lattice_spot_specs_mm,
    generate_complex_lattice_plan_geometry,
)
from run_phase10_complex_lattice_plan import (
    LOCKED_D_CYTO,
    LOCKED_GAMMA,
    LOCKED_LAMBDA_CYTO,
    LOCKED_SCALING_FACTOR,
    W_CYTO,
    W_IMMUNE,
    W_ROS,
)

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
except Exception as exc:  # pragma: no cover
    raise RuntimeError("matplotlib is required for figure generation") from exc


SPOT_MODES = ["uniform", "nominal", "core_hot"]
LAYOUT_MODES = ["central_only", "parallel_companions", "right_shifted_companion"]
RADII_MM = [1.0, 2.0, 3.0, 4.0]


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description=(
            "Run Phase 11A Sweep 3 over vessel radii and companion layouts."
        )
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=root
        / "runs"
        / "linac_6mv_polyenergetic_clinical_sfrt"
        / "analysis_phase11_pathA_radius_companions",
        help="Directory for Phase 11A Sweep 3 outputs.",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=root / "runs" / "linac_6mv_polyenergetic_clinical_sfrt" / "case" / "dosedata.csv",
        help="TOPAS single-beam kernel CSV.",
    )
    parser.add_argument(
        "--prescribed-peak-dose-gy",
        type=float,
        default=10.0,
        help="Prescribed peak dose used to normalize the kernel.",
    )
    parser.add_argument(
        "--uniform-dose-floor-fraction",
        type=float,
        default=0.015,
        help="Uniform block/leakage floor applied to the surrogate plan.",
    )
    parser.add_argument("--alpha", type=float, default=0.03, help="LQ alpha in Gy^-1.")
    parser.add_argument("--beta", type=float, default=0.003, help="LQ beta in Gy^-2.")
    parser.add_argument("--pde-steps", type=int, default=400, help="Temporal PDE steps.")
    parser.add_argument("--pde-dt", type=float, default=0.12, help="Temporal PDE time step.")
    parser.add_argument(
        "--target-depth-cm",
        type=float,
        default=5.03,
        help="Reference depth for summary extraction.",
    )
    parser.add_argument(
        "--valley-threshold-fraction",
        type=float,
        default=0.40,
        help="Physical-dose fraction of prescribed peak used to define valley ROI.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of parallel workers to use for independent sweep cases.",
    )
    parser.add_argument("--dpi", type=int, default=250, help="Figure DPI.")
    return parser.parse_args()


def write_csv(rows: List[Dict[str, object]], out_file: Path) -> None:
    if not rows:
        raise ValueError("No rows supplied for CSV output.")
    with out_file.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def equivalent_uniform_dose_from_survival(
    survival_values: np.ndarray,
    *,
    alpha: float,
    beta: float,
) -> float:
    mean_survival = float(np.mean(np.asarray(survival_values, dtype=np.float32)))
    return float(
        calculate_effective_dose(
            np.array([mean_survival], dtype=np.float32),
            alpha=float(alpha),
            beta=float(beta),
        )[0]
    )


def spot_specs_for_mode(mode: str) -> list[dict]:
    base = default_complex_lattice_spot_specs_mm()
    if mode == "uniform":
        return [{**spec, "weight": 1.0} for spec in base]
    if mode == "nominal":
        return [{**spec} for spec in base]
    if mode == "core_hot":
        adjusted: list[dict] = []
        for spec in base:
            radius_mm = float(np.hypot(float(spec["x_mm"]), float(spec["y_mm"])))
            if radius_mm <= 22.0:
                factor = 1.18
            elif radius_mm <= 38.0:
                factor = 1.06
            else:
                factor = 0.82
            weight = float(np.clip(float(spec["weight"]) * factor, 0.65, 1.45))
            adjusted.append({**spec, "weight": weight})
        return adjusted
    raise ValueError(f"Unknown spot mode: {mode}")


def dominant_right_peak_boundary_mm(spot_specs: Sequence[dict], beam_half_width_mm: float = 5.0) -> float:
    positive_candidates = [spec for spec in spot_specs if float(spec["x_mm"]) > 0.0 and abs(float(spec["y_mm"])) <= 10.0]
    if not positive_candidates:
        positive_candidates = [spec for spec in spot_specs if float(spec["x_mm"]) > 0.0]
    chosen = min(positive_candidates, key=lambda spec: abs(float(spec["y_mm"])))
    return float(chosen["x_mm"]) - float(beam_half_width_mm)


def companion_offset_mm(radius_mm: float, peak_boundary_mm: float) -> float:
    return 0.5 * (float(radius_mm) + float(peak_boundary_mm))


def vessel_specs_for_layout(layout_mode: str, *, radius_mm: float, peak_boundary_mm: float) -> list[dict]:
    uptake = (0.07, 0.92)
    central_nodes = [(-1.0, -2.0, -90.0), (0.0, 0.0, -5.0), (1.0, 1.5, 75.0)]
    offset = companion_offset_mm(radius_mm, peak_boundary_mm)

    def shifted_nodes(x_shift_mm: float) -> list[list[float]]:
        return [[float(x + x_shift_mm), float(y), float(z)] for x, y, z in central_nodes]

    if layout_mode == "central_only":
        return [
            {
                "label": "central_only",
                "radius_mm": float(radius_mm),
                "uptake_rates_in_vessel": uptake,
                "nodes_mm": central_nodes,
            }
        ]
    if layout_mode == "parallel_companions":
        return [
            {
                "label": "central",
                "radius_mm": float(radius_mm),
                "uptake_rates_in_vessel": uptake,
                "nodes_mm": central_nodes,
            },
            {
                "label": "left_companion",
                "radius_mm": float(radius_mm),
                "uptake_rates_in_vessel": uptake,
                "nodes_mm": shifted_nodes(-offset),
            },
            {
                "label": "right_companion",
                "radius_mm": float(radius_mm),
                "uptake_rates_in_vessel": uptake,
                "nodes_mm": shifted_nodes(offset),
            },
        ]
    if layout_mode == "right_shifted_companion":
        return [
            {
                "label": "central",
                "radius_mm": float(radius_mm),
                "uptake_rates_in_vessel": uptake,
                "nodes_mm": central_nodes,
            },
            {
                "label": "right_companion",
                "radius_mm": float(radius_mm),
                "uptake_rates_in_vessel": uptake,
                "nodes_mm": shifted_nodes(offset),
            },
        ]
    raise ValueError(f"Unknown layout mode: {layout_mode}")


def classify_case(center_survival: float, valley_eud_shift_gy: float) -> str:
    if center_survival <= 0.60 or valley_eud_shift_gy >= 3.0:
        return "unsafe"
    if center_survival >= 0.75 and valley_eud_shift_gy <= 2.0:
        return "safe"
    return "intermediate"


def solve_case(
    *,
    spot_mode: str,
    layout_mode: str,
    radius_mm: float,
    args: argparse.Namespace,
) -> Dict[str, object]:
    spot_specs = spot_specs_for_mode(spot_mode)
    peak_boundary_mm = dominant_right_peak_boundary_mm(spot_specs)
    vessel_specs = vessel_specs_for_layout(
        layout_mode,
        radius_mm=float(radius_mm),
        peak_boundary_mm=float(peak_boundary_mm),
    )

    dose_grid, uptake_tensor, m_oxygen, m_type, meta = generate_complex_lattice_plan_geometry(
        csv=args.csv,
        prescribed_peak_dose_gy=float(args.prescribed_peak_dose_gy),
        uniform_dose_floor_fraction=float(args.uniform_dose_floor_fraction),
        spot_specs_mm=spot_specs,
        vessel_specs_mm=vessel_specs,
    )

    voxel_size_mm = tuple(float(value) for value in meta["voxel_size_mm"])
    z_cm = np.asarray(meta["z_cm"], dtype=np.float32)
    target_depth_idx = nearest_index(z_cm, float(args.target_depth_cm))
    center_x = dose_grid.shape[0] // 2
    center_y = dose_grid.shape[1] // 2

    lq_survival = np.exp(-float(args.alpha) * dose_grid - float(args.beta) * dose_grid**2).astype(np.float32)
    hazard_grid = run_pde_temporal_integration(
        dose_grid,
        voxel_size_mm,
        D_cyto=LOCKED_D_CYTO,
        lambda_cyto=LOCKED_LAMBDA_CYTO,
        gamma=LOCKED_GAMMA,
        u_k=uptake_tensor,
        M_oxygen=m_oxygen,
        M_type=m_type,
        D_ros=0.8,
        lambda_ros=0.2,
        Emax_ros=1.5,
        Emax_cyto=0.8,
        w_ros=W_ROS,
        w_cyto=W_CYTO,
        steps=int(args.pde_steps),
        dt=float(args.pde_dt),
        progress_interval=50,
        verbose=True,
    )
    final_survival = calculate_phase7_survival(
        lq_survival,
        hazard_grid,
        dose_grid,
        voxel_size_mm,
        float(LOCKED_SCALING_FACTOR),
        weight_immune=float(W_IMMUNE),
        verbose=False,
    )
    deff_grid = calculate_effective_dose(final_survival, alpha=float(args.alpha), beta=float(args.beta))
    immune_penalty, icd_volume_cm3 = calculate_systemic_immune_penalty(dose_grid, voxel_size_mm)

    valley_threshold_gy = float(args.valley_threshold_fraction) * float(args.prescribed_peak_dose_gy)
    valley_mask = dose_grid <= float(valley_threshold_gy)
    valley_eud_lq = equivalent_uniform_dose_from_survival(
        lq_survival[valley_mask],
        alpha=float(args.alpha),
        beta=float(args.beta),
    )
    valley_eud_total = equivalent_uniform_dose_from_survival(
        final_survival[valley_mask],
        alpha=float(args.alpha),
        beta=float(args.beta),
    )
    valley_eud_shift_gy = float(valley_eud_total - valley_eud_lq)
    center_survival = float(final_survival[center_x, center_y, target_depth_idx])

    return {
        "spot_mode": spot_mode,
        "layout_mode": layout_mode,
        "radius_mm": float(radius_mm),
        "companion_offset_mm": float(companion_offset_mm(radius_mm, peak_boundary_mm)),
        "peak_boundary_mm": float(peak_boundary_mm),
        "sampled_depth_cm": float(z_cm[target_depth_idx]),
        "center_dose_gy": float(dose_grid[center_x, center_y, target_depth_idx]),
        "center_survival_lq": float(lq_survival[center_x, center_y, target_depth_idx]),
        "center_survival_total": center_survival,
        "center_deff_gy": float(deff_grid[center_x, center_y, target_depth_idx]),
        "center_hazard": float(hazard_grid[center_x, center_y, target_depth_idx]),
        "valley_threshold_gy": float(valley_threshold_gy),
        "valley_mean_survival_total": float(np.mean(final_survival[valley_mask])),
        "valley_median_survival_total": float(np.median(final_survival[valley_mask])),
        "valley_eud_lq_gy": float(valley_eud_lq),
        "valley_eud_total_gy": float(valley_eud_total),
        "valley_eud_shift_gy": valley_eud_shift_gy,
        "immune_penalty": float(immune_penalty),
        "icd_volume_cm3": float(icd_volume_cm3),
        "vessel_branch_count": int(len(vessel_specs)),
        "vessel_voxel_count": int(meta.get("vessel_voxel_count", 0)),
        "risk_class": classify_case(center_survival, valley_eud_shift_gy),
    }


def ordered_rows(rows: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    return sorted(
        rows,
        key=lambda row: (
            SPOT_MODES.index(str(row["spot_mode"])),
            RADII_MM.index(float(row["radius_mm"])),
            LAYOUT_MODES.index(str(row["layout_mode"])),
        ),
    )


def case_specs(args: argparse.Namespace) -> List[Tuple[str, str, float, argparse.Namespace]]:
    return [
        (spot_mode, layout_mode, float(radius_mm), args)
        for spot_mode in SPOT_MODES
        for radius_mm in RADII_MM
        for layout_mode in LAYOUT_MODES
    ]


def solve_case_task(task: Tuple[str, str, float, argparse.Namespace]) -> Dict[str, object]:
    spot_mode, layout_mode, radius_mm, args = task
    return solve_case(
        spot_mode=spot_mode,
        layout_mode=layout_mode,
        radius_mm=float(radius_mm),
        args=args,
    )


def build_matrix(rows: Sequence[Dict[str, object]], *, spot_mode: str, value_key: str) -> np.ndarray:
    matrix = np.full((len(RADII_MM), len(LAYOUT_MODES)), np.nan, dtype=np.float32)
    relevant = [row for row in rows if str(row["spot_mode"]) == spot_mode]
    for row in relevant:
        i = RADII_MM.index(float(row["radius_mm"]))
        j = LAYOUT_MODES.index(str(row["layout_mode"]))
        matrix[i, j] = float(row[value_key])
    return matrix


def plot_metric_panels(
    rows: Sequence[Dict[str, object]],
    *,
    value_key: str,
    title: str,
    cbar_label: str,
    cmap,
    text_fmt: str,
    out_file: Path,
    dpi: int,
) -> None:
    fig, axes = plt.subplots(1, len(SPOT_MODES), figsize=(5.0 * len(SPOT_MODES), 5.0), constrained_layout=True)
    last_img = None
    for ax, spot_mode in zip(axes, SPOT_MODES):
        matrix = build_matrix(rows, spot_mode=spot_mode, value_key=value_key)
        last_img = ax.imshow(matrix, cmap=cmap, aspect="auto")
        ax.set_xticks(np.arange(len(LAYOUT_MODES)), labels=LAYOUT_MODES, rotation=20, ha="right")
        ax.set_yticks(np.arange(len(RADII_MM)), labels=[f"{value:.0f}" for value in RADII_MM])
        ax.set_xlabel("Sink layout")
        ax.set_ylabel("Radius (mm)")
        ax.set_title(spot_mode)
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                ax.text(j, i, format(float(matrix[i, j]), text_fmt), ha="center", va="center", color="#ffffff", fontsize=8)
    cbar = fig.colorbar(last_img, ax=axes, shrink=0.92)
    cbar.set_label(cbar_label)
    fig.suptitle(title, fontsize=14)
    fig.savefig(out_file, dpi=dpi)
    plt.close(fig)


def plot_risk_panels(
    rows: Sequence[Dict[str, object]],
    *,
    out_file: Path,
    dpi: int,
) -> None:
    risk_to_value = {"safe": 0, "intermediate": 1, "unsafe": 2}
    cmap = ListedColormap(["#2ca02c", "#ffbf00", "#d62728"])

    fig, axes = plt.subplots(1, len(SPOT_MODES), figsize=(5.0 * len(SPOT_MODES), 5.0), constrained_layout=True)
    last_img = None
    for ax, spot_mode in zip(axes, SPOT_MODES):
        matrix = np.full((len(RADII_MM), len(LAYOUT_MODES)), np.nan, dtype=np.float32)
        for row in rows:
            if str(row["spot_mode"]) != spot_mode:
                continue
            i = RADII_MM.index(float(row["radius_mm"]))
            j = LAYOUT_MODES.index(str(row["layout_mode"]))
            matrix[i, j] = risk_to_value[str(row["risk_class"])]
        last_img = ax.imshow(matrix, cmap=cmap, vmin=0.0, vmax=2.0, aspect="auto")
        ax.set_xticks(np.arange(len(LAYOUT_MODES)), labels=LAYOUT_MODES, rotation=20, ha="right")
        ax.set_yticks(np.arange(len(RADII_MM)), labels=[f"{value:.0f}" for value in RADII_MM])
        ax.set_xlabel("Sink layout")
        ax.set_ylabel("Radius (mm)")
        ax.set_title(spot_mode)
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                value = int(matrix[i, j])
                label = ["safe", "intermediate", "unsafe"][value]
                ax.text(j, i, label, ha="center", va="center", color="#ffffff", fontsize=8)
    cbar = fig.colorbar(last_img, ax=axes, shrink=0.92, ticks=[0, 1, 2])
    cbar.set_ticklabels(["safe", "intermediate", "unsafe"])
    fig.suptitle("Figure 3: Phase 11A Sweep 3 risk classification", fontsize=14)
    fig.savefig(out_file, dpi=dpi)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    metrics_csv = args.outdir / "phase11_pathA_radius_companion_metrics.csv"
    summary_json = args.outdir / "phase11_pathA_radius_companion_summary.json"
    fig1 = args.outdir / "figure1_phase11_radius_companion_center_survival.png"
    fig2 = args.outdir / "figure2_phase11_radius_companion_valley_shift.png"
    fig3 = args.outdir / "figure3_phase11_radius_companion_risk_map.png"

    rows: List[Dict[str, object]] = []
    tasks = case_specs(args)
    total = len(tasks)
    workers = max(1, int(args.workers))
    print(f"Running {total} Sweep 3 cases with {workers} worker(s).", flush=True)

    if workers == 1:
        iterator = (solve_case_task(task) for task in tasks)
    else:
        ctx = mp.get_context("spawn")
        pool = ctx.Pool(processes=workers)
        iterator = pool.imap_unordered(solve_case_task, tasks)

    try:
        for idx, row in enumerate(iterator, start=1):
            print(
                f"[{idx}/{total}] spot={row['spot_mode']} | radius={row['radius_mm']:.0f} mm | "
                f"layout={row['layout_mode']} | center_survival={row['center_survival_total']:.4f} | "
                f"valley_shift={row['valley_eud_shift_gy']:.3f} Gy | class={row['risk_class']}",
                flush=True,
            )
            rows.append(row)
            write_csv(ordered_rows(rows), metrics_csv)
    finally:
        if workers != 1:
            pool.close()
            pool.join()

    rows = ordered_rows(rows)

    write_csv(rows, metrics_csv)
    plot_metric_panels(
        rows,
        value_key="center_survival_total",
        title="Figure 1: Phase 11A Sweep 3 center survival",
        cbar_label="Center survival at 5 cm",
        cmap="viridis",
        text_fmt=".3f",
        out_file=fig1,
        dpi=int(args.dpi),
    )
    plot_metric_panels(
        rows,
        value_key="valley_eud_shift_gy",
        title="Figure 2: Phase 11A Sweep 3 valley EUD shift",
        cbar_label="Valley EUD shift (Gy)",
        cmap="magma",
        text_fmt=".2f",
        out_file=fig2,
        dpi=int(args.dpi),
    )
    plot_risk_panels(rows, out_file=fig3, dpi=int(args.dpi))

    class_counts: Dict[str, int] = {}
    for row in rows:
        class_counts[str(row["risk_class"])] = class_counts.get(str(row["risk_class"]), 0) + 1

    four_mm_cases = [row for row in rows if float(row["radius_mm"]) == 4.0]
    summary = {
        "phase": "Phase 11A",
        "description": "Sweep 3 over vessel radius ladder and companion layouts.",
        "sweep_axes": {
            "spot_modes": SPOT_MODES,
            "radii_mm": RADII_MM,
            "layout_modes": LAYOUT_MODES,
        },
        "locked_biology": {
            "D_cyto": LOCKED_D_CYTO,
            "lambda_cyto": LOCKED_LAMBDA_CYTO,
            "gamma": LOCKED_GAMMA,
            "scaling_factor": LOCKED_SCALING_FACTOR,
            "weight_ros": W_ROS,
            "weight_cyto": W_CYTO,
            "weight_immune": W_IMMUNE,
        },
        "companion_placement_rule": (
            "Companion center placed halfway between the central-vessel boundary and the dominant right-peak boundary "
            "derived from the nearest positive-x central-lane spot minus a 5 mm beam half-width."
        ),
        "risk_rules": {
            "safe": "center_survival_total >= 0.75 and valley_eud_shift_gy <= 2.0",
            "unsafe": "center_survival_total <= 0.60 or valley_eud_shift_gy >= 3.0",
            "intermediate": "otherwise",
        },
        "class_counts": class_counts,
        "best_case": max(rows, key=lambda row: float(row["center_survival_total"])),
        "worst_case": min(rows, key=lambda row: float(row["center_survival_total"])),
        "four_mm_cases": four_mm_cases,
        "outputs": {
            "metrics_csv": str(metrics_csv),
            "summary_json": str(summary_json),
            "figure_center_survival": str(fig1),
            "figure_valley_shift": str(fig2),
            "figure_risk_map": str(fig3),
        },
    }
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"\nPhase 11A Sweep 3 metrics: {metrics_csv}")
    print(f"Phase 11A Sweep 3 summary: {summary_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
