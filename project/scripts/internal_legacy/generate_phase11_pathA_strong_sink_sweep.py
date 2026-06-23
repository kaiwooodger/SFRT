#!/usr/bin/env python
"""Phase 11A follow-up: stronger anatomical sink sweep."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Sequence

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
except Exception as exc:  # pragma: no cover
    raise RuntimeError("matplotlib is required for figure generation") from exc


SPOT_MODES = ["uniform", "nominal", "core_hot"]
STRONG_SINK_MODES = ["baseline_single", "wide_center", "valley_centered", "corridor_aligned"]


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description=(
            "Run a stronger-sink Phase 11A follow-up sweep over spot weights and "
            "anatomically more aggressive vessel families."
        )
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=root
        / "runs"
        / "linac_6mv_polyenergetic_clinical_sfrt"
        / "analysis_phase11_pathA_strong_sinks",
        help="Directory for Phase 11A strong-sink outputs.",
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
        help="Uniform block/leakage floor applied to the entire surrogate plan.",
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
        help="Physical-dose fraction of the prescribed peak used to define the valley ROI.",
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


def strong_vessel_specs_for_mode(mode: str) -> list[dict]:
    if mode == "baseline_single":
        return [
            {
                "label": "baseline_center",
                "radius_mm": 1.8,
                "uptake_rates_in_vessel": (0.06, 0.72),
                "nodes_mm": [(-2.0, -4.0, -90.0), (0.0, 0.0, -10.0), (1.5, 2.0, 70.0)],
            }
        ]
    if mode == "wide_center":
        return [
            {
                "label": "wide_center",
                "radius_mm": 3.4,
                "uptake_rates_in_vessel": (0.07, 0.92),
                "nodes_mm": [(-1.5, -2.0, -90.0), (0.0, 0.0, -5.0), (1.0, 1.5, 75.0)],
            }
        ]
    if mode == "valley_centered":
        return [
            {
                "label": "central_valley",
                "radius_mm": 2.8,
                "uptake_rates_in_vessel": (0.07, 0.88),
                "nodes_mm": [(0.0, -18.0, -88.0), (0.0, -8.0, -10.0), (0.0, 4.0, 72.0)],
            },
            {
                "label": "mid_valley",
                "radius_mm": 2.6,
                "uptake_rates_in_vessel": (0.06, 0.84),
                "nodes_mm": [(0.0, 6.0, -88.0), (0.0, 16.0, -5.0), (0.0, 28.0, 72.0)],
            },
            {
                "label": "superior_valley",
                "radius_mm": 2.3,
                "uptake_rates_in_vessel": (0.06, 0.78),
                "nodes_mm": [(0.0, 30.0, -60.0), (0.0, 40.0, 5.0), (0.0, 52.0, 72.0)],
            },
        ]
    if mode == "corridor_aligned":
        return [
            {
                "label": "central_corridor",
                "radius_mm": 2.8,
                "uptake_rates_in_vessel": (0.07, 0.90),
                "nodes_mm": [(-2.0, -6.0, -90.0), (2.0, -2.0, -15.0), (8.0, 0.0, 35.0), (14.0, 4.0, 78.0)],
            },
            {
                "label": "parallel_corridor",
                "radius_mm": 2.1,
                "uptake_rates_in_vessel": (0.06, 0.82),
                "nodes_mm": [(-8.0, -10.0, -78.0), (-2.0, -4.0, -10.0), (4.0, 2.0, 42.0), (10.0, 10.0, 78.0)],
            },
            {
                "label": "return_branch",
                "radius_mm": 1.8,
                "uptake_rates_in_vessel": (0.05, 0.74),
                "nodes_mm": [(18.0, -6.0, -65.0), (14.0, 2.0, -10.0), (8.0, 12.0, 55.0)],
            },
        ]
    raise ValueError(f"Unknown strong sink mode: {mode}")


def classify_case(center_survival: float, valley_eud_shift_gy: float) -> str:
    if center_survival <= 0.60 or valley_eud_shift_gy >= 3.0:
        return "unsafe"
    if center_survival >= 0.75 and valley_eud_shift_gy <= 2.0:
        return "safe"
    return "intermediate"


def solve_case(
    *,
    spot_mode: str,
    sink_mode: str,
    args: argparse.Namespace,
) -> Dict[str, object]:
    spot_specs = spot_specs_for_mode(spot_mode)
    vessel_specs = strong_vessel_specs_for_mode(sink_mode)
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

    weights = np.asarray([float(spec["weight"]) for spec in spot_specs], dtype=np.float32)
    return {
        "spot_mode": spot_mode,
        "sink_mode": sink_mode,
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
        "weight_mean": float(np.mean(weights)),
        "weight_std": float(np.std(weights)),
        "weight_cv": float(np.std(weights) / np.mean(weights)),
        "vessel_branch_count": int(len(vessel_specs)),
        "vessel_voxel_count": int(meta.get("vessel_voxel_count", 0)),
        "risk_class": classify_case(center_survival, valley_eud_shift_gy),
    }


def matrix_from_rows(
    rows: Sequence[Dict[str, object]],
    *,
    value_key: str,
) -> np.ndarray:
    matrix = np.full((len(SPOT_MODES), len(STRONG_SINK_MODES)), np.nan, dtype=np.float32)
    for row in rows:
        i = SPOT_MODES.index(str(row["spot_mode"]))
        j = STRONG_SINK_MODES.index(str(row["sink_mode"]))
        matrix[i, j] = float(row[value_key])
    return matrix


def plot_metric_matrix(
    matrix: np.ndarray,
    *,
    title: str,
    cbar_label: str,
    cmap: str,
    out_file: Path,
    dpi: int,
    text_fmt: str,
) -> None:
    fig, ax = plt.subplots(1, 1, figsize=(8.8, 4.8), constrained_layout=True)
    img = ax.imshow(matrix, cmap=cmap, aspect="auto")
    ax.set_xticks(np.arange(len(STRONG_SINK_MODES)), labels=STRONG_SINK_MODES, rotation=20, ha="right")
    ax.set_yticks(np.arange(len(SPOT_MODES)), labels=SPOT_MODES)
    ax.set_xlabel("Strong sink mode")
    ax.set_ylabel("Spot-weight mode")
    ax.set_title(title)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix[i, j]
            ax.text(j, i, format(float(value), text_fmt), ha="center", va="center", color="#ffffff", fontsize=8)
    cbar = fig.colorbar(img, ax=ax, shrink=0.92)
    cbar.set_label(cbar_label)
    fig.savefig(out_file, dpi=dpi)
    plt.close(fig)


def plot_risk_scatter(
    rows: Sequence[Dict[str, object]],
    *,
    out_file: Path,
    dpi: int,
) -> None:
    color_map = {"safe": "#2ca02c", "intermediate": "#ffbf00", "unsafe": "#d62728"}
    marker_map = {
        "baseline_single": "o",
        "wide_center": "s",
        "valley_centered": "^",
        "corridor_aligned": "D",
    }

    fig, ax = plt.subplots(1, 1, figsize=(7.6, 5.4), constrained_layout=True)
    for row in rows:
        ax.scatter(
            float(row["center_survival_total"]),
            float(row["valley_eud_shift_gy"]),
            s=95,
            color=color_map[str(row["risk_class"])],
            marker=marker_map[str(row["sink_mode"])],
            edgecolor="#1a1a1a",
            linewidth=0.7,
        )
        ax.text(
            float(row["center_survival_total"]) + 0.004,
            float(row["valley_eud_shift_gy"]) + 0.012,
            f"{row['spot_mode']}/{row['sink_mode']}",
            fontsize=7,
        )
    ax.axvline(0.75, color="#2ca02c", linestyle="--", linewidth=1.0, alpha=0.6)
    ax.axvline(0.60, color="#d62728", linestyle="--", linewidth=1.0, alpha=0.6)
    ax.axhline(2.0, color="#2ca02c", linestyle="--", linewidth=1.0, alpha=0.6)
    ax.axhline(3.0, color="#d62728", linestyle="--", linewidth=1.0, alpha=0.6)
    ax.set_xlabel("Center survival at 5 cm")
    ax.set_ylabel("Valley EUD shift (Gy)")
    ax.set_title("Figure 3: Phase 11A stronger-sink safe vs unsafe map")
    ax.grid(alpha=0.25)
    fig.savefig(out_file, dpi=dpi)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, object]] = []
    for spot_mode in SPOT_MODES:
        for sink_mode in STRONG_SINK_MODES:
            print(f"\n=== Phase 11A strong-sink case: spot={spot_mode}, sink={sink_mode} ===", flush=True)
            row = solve_case(spot_mode=spot_mode, sink_mode=sink_mode, args=args)
            print(
                f"center_survival={row['center_survival_total']:.4f} | "
                f"valley_shift={row['valley_eud_shift_gy']:.3f} Gy | "
                f"class={row['risk_class']}",
                flush=True,
            )
            rows.append(row)

    metrics_csv = args.outdir / "phase11_pathA_strong_sink_metrics.csv"
    summary_json = args.outdir / "phase11_pathA_strong_sink_summary.json"
    fig1 = args.outdir / "figure1_phase11_strong_sink_center_survival_matrix.png"
    fig2 = args.outdir / "figure2_phase11_strong_sink_valley_eud_shift_matrix.png"
    fig3 = args.outdir / "figure3_phase11_strong_sink_safe_unsafe_map.png"

    write_csv(rows, metrics_csv)
    plot_metric_matrix(
        matrix_from_rows(rows, value_key="center_survival_total"),
        title="Figure 1: Phase 11A stronger-sink center survival matrix",
        cbar_label="Center survival at 5 cm",
        cmap="viridis",
        out_file=fig1,
        dpi=int(args.dpi),
        text_fmt=".3f",
    )
    plot_metric_matrix(
        matrix_from_rows(rows, value_key="valley_eud_shift_gy"),
        title="Figure 2: Phase 11A stronger-sink valley EUD shift matrix",
        cbar_label="Valley EUD shift (Gy)",
        cmap="magma",
        out_file=fig2,
        dpi=int(args.dpi),
        text_fmt=".2f",
    )
    plot_risk_scatter(rows, out_file=fig3, dpi=int(args.dpi))

    class_counts: Dict[str, int] = {}
    for row in rows:
        class_counts[str(row["risk_class"])] = class_counts.get(str(row["risk_class"]), 0) + 1

    summary = {
        "phase": "Phase 11A",
        "description": "Focused stronger-sink sweep over wider, valley-centered, and corridor-aligned vessel families.",
        "sweep_axes": {
            "spot_modes": SPOT_MODES,
            "strong_sink_modes": STRONG_SINK_MODES,
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
        "risk_rules": {
            "safe": "center_survival_total >= 0.75 and valley_eud_shift_gy <= 2.0",
            "unsafe": "center_survival_total <= 0.60 or valley_eud_shift_gy >= 3.0",
            "intermediate": "otherwise",
        },
        "class_counts": class_counts,
        "best_case": max(rows, key=lambda row: float(row["center_survival_total"])),
        "worst_case": min(rows, key=lambda row: float(row["center_survival_total"])),
        "max_valley_shift_case": max(rows, key=lambda row: float(row["valley_eud_shift_gy"])),
        "outputs": {
            "metrics_csv": str(metrics_csv),
            "summary_json": str(summary_json),
            "figure_center_survival": str(fig1),
            "figure_valley_shift": str(fig2),
            "figure_safe_unsafe": str(fig3),
        },
    }
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"\nPhase 11A stronger-sink metrics: {metrics_csv}")
    print(f"Phase 11A stronger-sink summary: {summary_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
