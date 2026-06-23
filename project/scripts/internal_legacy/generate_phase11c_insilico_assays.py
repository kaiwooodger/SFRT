#!/usr/bin/env python
"""Phase 11C: in silico orthogonal assay readouts for the best 4 mm Sweep 3 case."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List

import numpy as np

from analyze_250mev_sfrt_plan import centered_axis_cm, nearest_index
from bystander_multispecies_pde_solver import (
    calculate_effective_dose,
    calculate_phase7_survival,
    calculate_systemic_immune_penalty,
    solve_multispecies_pde_3d_with_hazard_observables,
)
from generate_phase11_pathA_radius_companion_sweep import (
    dominant_right_peak_boundary_mm,
    spot_specs_for_mode,
    vessel_specs_for_layout,
)
from geometry_generators import generate_complex_lattice_plan_geometry
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


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description=(
            "Generate in silico gammaH2AX, ELISA-like cytokine, and TUNEL-like "
            "assay outputs for the best 4 mm Path A geometry."
        )
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=root
        / "runs"
        / "linac_6mv_polyenergetic_clinical_sfrt"
        / "analysis_phase11c_insilico_assays",
        help="Directory for Phase 11C outputs.",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=root / "runs" / "linac_6mv_polyenergetic_clinical_sfrt" / "case" / "dosedata.csv",
        help="TOPAS single-beam kernel CSV.",
    )
    parser.add_argument(
        "--spot-mode",
        choices=["uniform", "nominal", "core_hot"],
        default="uniform",
        help="Spot weighting mode. Defaults to the best 4 mm Sweep 3 case.",
    )
    parser.add_argument(
        "--layout-mode",
        choices=["central_only", "parallel_companions", "right_shifted_companion"],
        default="parallel_companions",
        help="Vessel layout mode. Defaults to the best 4 mm Sweep 3 case.",
    )
    parser.add_argument(
        "--radius-mm",
        type=float,
        default=4.0,
        help="Central/companion vessel radius in mm. Defaults to 4.0 mm.",
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
        help="Uniform block/leakage floor applied to the synthetic geometry.",
    )
    parser.add_argument("--alpha", type=float, default=0.03, help="LQ alpha in Gy^-1.")
    parser.add_argument("--beta", type=float, default=0.003, help="LQ beta in Gy^-2.")
    parser.add_argument("--pde-steps", type=int, default=400, help="Temporal PDE steps.")
    parser.add_argument("--pde-dt", type=float, default=0.12, help="Temporal PDE time step.")
    parser.add_argument(
        "--history-interval",
        type=int,
        default=10,
        help="Step interval used to record ELISA-style cytokine histories.",
    )
    parser.add_argument(
        "--target-depth-cm",
        type=float,
        default=5.03,
        help="Reference depth for assay map slices.",
    )
    parser.add_argument(
        "--valley-threshold-fraction",
        type=float,
        default=0.40,
        help="Physical-dose fraction of prescribed peak used to define the valley ROI.",
    )
    parser.add_argument(
        "--peak-threshold-fraction",
        type=float,
        default=0.80,
        help="Physical-dose fraction of prescribed peak used to define the peak ROI.",
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


def _slice_extent_xy(x_cm: np.ndarray, y_cm: np.ndarray) -> list[float]:
    return [float(x_cm[0]), float(x_cm[-1]), float(y_cm[0]), float(y_cm[-1])]


def _slice_extent_xz(x_cm: np.ndarray, z_cm: np.ndarray) -> list[float]:
    return [float(x_cm[0]), float(x_cm[-1]), float(z_cm[0]), float(z_cm[-1])]


def build_best_case_geometry(args: argparse.Namespace) -> Dict[str, object]:
    spot_specs = spot_specs_for_mode(str(args.spot_mode))
    peak_boundary_mm = dominant_right_peak_boundary_mm(spot_specs)
    vessel_specs = vessel_specs_for_layout(
        str(args.layout_mode),
        radius_mm=float(args.radius_mm),
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
    x_cm = np.asarray(meta["x_cm"], dtype=np.float32)
    y_cm = centered_axis_cm(dose_grid.shape[1], voxel_size_mm[1] / 10.0)
    z_cm = np.asarray(meta["z_cm"], dtype=np.float32)
    return {
        "dose_grid": dose_grid.astype(np.float32, copy=False),
        "uptake_tensor": uptake_tensor.astype(np.float32, copy=False),
        "m_oxygen": m_oxygen.astype(np.float32, copy=False),
        "m_type": m_type.astype(np.float32, copy=False),
        "meta": meta,
        "x_cm": x_cm,
        "y_cm": y_cm,
        "z_cm": z_cm,
        "voxel_size_mm": voxel_size_mm,
        "spot_specs": spot_specs,
        "vessel_specs": vessel_specs,
        "peak_boundary_mm": float(peak_boundary_mm),
    }


def calculate_gamma_h2ax_proxy(
    dose_grid: np.ndarray,
    ros_peak_grid: np.ndarray,
    *,
    alpha: float,
    beta: float,
) -> np.ndarray:
    """Map physical dose and peak ROS into a gammaH2AX-like relative signal."""

    physical_drive = float(alpha) * dose_grid + float(beta) * dose_grid**2
    ros_scale = max(float(np.percentile(ros_peak_grid, 99.0)), 1.0e-6)
    ros_drive = np.clip(ros_peak_grid / ros_scale, 0.0, 3.0)
    return (1.0 - np.exp(-(physical_drive + 0.45 * ros_drive))).astype(np.float32, copy=False)


def plot_assay_map(
    *,
    volume: np.ndarray,
    vessel_mask: np.ndarray,
    title: str,
    cbar_label: str,
    cmap: str,
    vmin: float,
    vmax: float,
    x_cm: np.ndarray,
    y_cm: np.ndarray,
    z_cm: np.ndarray,
    target_depth_idx: int,
    out_file: Path,
    dpi: int,
) -> None:
    center_y = volume.shape[1] // 2
    xy_slice = volume[:, :, target_depth_idx].T
    xz_slice = volume[:, center_y, :].T
    vessel_xy = vessel_mask[:, :, target_depth_idx].T
    vessel_xz = vessel_mask[:, center_y, :].T

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.4), constrained_layout=True)
    img0 = axes[0].imshow(
        xy_slice,
        origin="lower",
        extent=_slice_extent_xy(x_cm, y_cm),
        aspect="equal",
        cmap=cmap,
        vmin=float(vmin),
        vmax=float(vmax),
    )
    axes[0].contour(
        vessel_xy.astype(np.float32),
        levels=[0.5],
        colors="#4db6ff",
        linewidths=1.0,
        origin="lower",
        extent=_slice_extent_xy(x_cm, y_cm),
    )
    axes[0].set_title(f"x-y at z = {float(z_cm[target_depth_idx]):.2f} cm")
    axes[0].set_xlabel("x (cm)")
    axes[0].set_ylabel("y (cm)")

    img1 = axes[1].imshow(
        xz_slice,
        origin="lower",
        extent=_slice_extent_xz(x_cm, z_cm),
        aspect="auto",
        cmap=cmap,
        vmin=float(vmin),
        vmax=float(vmax),
    )
    axes[1].contour(
        vessel_xz.astype(np.float32),
        levels=[0.5],
        colors="#4db6ff",
        linewidths=1.0,
        origin="lower",
        extent=_slice_extent_xz(x_cm, z_cm),
    )
    axes[1].axhline(float(z_cm[target_depth_idx]), color="#ffffff", linestyle="--", linewidth=1.2)
    axes[1].set_title("x-z at center y")
    axes[1].set_xlabel("x (cm)")
    axes[1].set_ylabel("z (cm)")

    cbar = fig.colorbar(img1, ax=axes.ravel().tolist(), shrink=0.9)
    cbar.set_label(cbar_label)
    fig.suptitle(title, fontsize=15)
    fig.savefig(out_file, dpi=dpi)
    plt.close(fig)


def plot_elisa_curve(
    *,
    time_axis: np.ndarray,
    global_history: np.ndarray,
    mask_history: dict[str, np.ndarray],
    out_file: Path,
    dpi: int,
) -> None:
    fig, ax = plt.subplots(figsize=(8.8, 5.4), constrained_layout=True)
    ax.plot(time_axis, global_history[1], color="#d62728", linewidth=2.2, label="Whole volume cytokine")
    if "peak_roi" in mask_history:
        ax.plot(time_axis, mask_history["peak_roi"][1], color="#ff7f0e", linewidth=2.0, label="Peak ROI cytokine")
    if "valley_roi" in mask_history:
        ax.plot(time_axis, mask_history["valley_roi"][1], color="#1f77b4", linewidth=2.0, label="Valley ROI cytokine")
    ax.set_xlabel("Simulation time (a.u.)")
    ax.set_ylabel("Mean cytokine concentration (a.u.)")
    ax.set_title("Figure 2: Phase 11C ELISA-like cytokine curve")
    ax.grid(True, linestyle="--", alpha=0.45)
    ax.legend(loc="best")
    fig.savefig(out_file, dpi=dpi)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    geometry = build_best_case_geometry(args)
    dose_grid = geometry["dose_grid"]
    uptake_tensor = geometry["uptake_tensor"]
    m_oxygen = geometry["m_oxygen"]
    m_type = geometry["m_type"]
    x_cm = geometry["x_cm"]
    y_cm = geometry["y_cm"]
    z_cm = geometry["z_cm"]
    voxel_size_mm = geometry["voxel_size_mm"]

    target_depth_idx = nearest_index(z_cm, float(args.target_depth_cm))
    center_x = dose_grid.shape[0] // 2
    center_y = dose_grid.shape[1] // 2
    valley_threshold_gy = float(args.valley_threshold_fraction) * float(args.prescribed_peak_dose_gy)
    peak_threshold_gy = float(args.peak_threshold_fraction) * float(args.prescribed_peak_dose_gy)
    valley_mask = dose_grid <= valley_threshold_gy
    peak_mask = dose_grid >= peak_threshold_gy
    vessel_mask = uptake_tensor[1] > 0.0

    lq_survival = np.exp(-float(args.alpha) * dose_grid - float(args.beta) * dose_grid**2).astype(np.float32)

    observables = solve_multispecies_pde_3d_with_hazard_observables(
        dose_grid,
        voxel_size_mm,
        diffusion_coeffs=(0.8, LOCKED_D_CYTO),
        decay_coeffs=(0.2, LOCKED_LAMBDA_CYTO),
        emission_emax=(1.5, 0.8),
        emission_gamma_per_gy=LOCKED_GAMMA,
        uptake_tensor=uptake_tensor,
        hazard_weights=(W_ROS, W_CYTO),
        history_masks={"peak_roi": peak_mask, "valley_roi": valley_mask},
        history_interval=int(args.history_interval),
        steps=int(args.pde_steps),
        dt=float(args.pde_dt),
        progress_interval=50,
        verbose=True,
    )

    concentration = np.asarray(observables["concentration"], dtype=np.float32)
    peak_concentration = np.asarray(observables["peak_concentration"], dtype=np.float32)
    hazard_grid = np.asarray(observables["hazard_grid"], dtype=np.float32)
    ros_hazard_grid = np.asarray(observables["ros_hazard_grid"], dtype=np.float32)
    cytokine_hazard_grid = np.asarray(observables["cytokine_hazard_grid"], dtype=np.float32)
    time_axis = np.asarray(observables["time_axis"], dtype=np.float32)
    global_history = np.asarray(observables["global_mean_history"], dtype=np.float32)
    mask_history = {
        str(name): np.asarray(values, dtype=np.float32)
        for name, values in dict(observables["mask_mean_history"]).items()
    }

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

    gamma_h2ax_map = calculate_gamma_h2ax_proxy(
        dose_grid,
        peak_concentration[0],
        alpha=float(args.alpha),
        beta=float(args.beta),
    )
    tunel_map = (1.0 - final_survival).astype(np.float32, copy=False)

    fig1 = args.outdir / "figure1_phase11c_gammah2ax_proxy.png"
    fig2 = args.outdir / "figure2_phase11c_elisa_cytokine_curve.png"
    fig3 = args.outdir / "figure3_phase11c_tunel_apoptosis_proxy.png"
    curve_csv = args.outdir / "phase11c_elisa_cytokine_curve.csv"
    centerline_csv = args.outdir / "phase11c_centerline_assays.csv"
    summary_json = args.outdir / "phase11c_assay_summary.json"
    volumes_npz = args.outdir / "phase11c_assay_volumes.npz"

    plot_assay_map(
        volume=gamma_h2ax_map,
        vessel_mask=vessel_mask,
        title="Figure 1: Phase 11C gammaH2AX proxy map",
        cbar_label="Relative gammaH2AX signal (a.u.)",
        cmap="magma",
        vmin=0.0,
        vmax=1.0,
        x_cm=x_cm,
        y_cm=y_cm,
        z_cm=z_cm,
        target_depth_idx=int(target_depth_idx),
        out_file=fig1,
        dpi=int(args.dpi),
    )
    plot_elisa_curve(
        time_axis=time_axis,
        global_history=global_history,
        mask_history=mask_history,
        out_file=fig2,
        dpi=int(args.dpi),
    )
    plot_assay_map(
        volume=tunel_map,
        vessel_mask=vessel_mask,
        title="Figure 3: Phase 11C TUNEL-like apoptosis proxy",
        cbar_label="TUNEL-positive fraction proxy",
        cmap="viridis",
        vmin=0.0,
        vmax=1.0,
        x_cm=x_cm,
        y_cm=y_cm,
        z_cm=z_cm,
        target_depth_idx=int(target_depth_idx),
        out_file=fig3,
        dpi=int(args.dpi),
    )

    cytokine_peak_roi = mask_history.get("peak_roi", np.zeros((2, 0), dtype=np.float32))
    cytokine_valley_roi = mask_history.get("valley_roi", np.zeros((2, 0), dtype=np.float32))
    curve_rows: List[Dict[str, object]] = []
    for idx, time_value in enumerate(time_axis):
        row: Dict[str, object] = {
            "time_au": float(time_value),
            "ros_mean_global": float(global_history[0, idx]),
            "cytokine_mean_global": float(global_history[1, idx]),
        }
        if cytokine_peak_roi.size:
            row["ros_mean_peak_roi"] = float(cytokine_peak_roi[0, idx])
            row["cytokine_mean_peak_roi"] = float(cytokine_peak_roi[1, idx])
        if cytokine_valley_roi.size:
            row["ros_mean_valley_roi"] = float(cytokine_valley_roi[0, idx])
            row["cytokine_mean_valley_roi"] = float(cytokine_valley_roi[1, idx])
        curve_rows.append(row)
    write_csv(curve_rows, curve_csv)

    centerline_rows: List[Dict[str, object]] = []
    for ix in range(dose_grid.shape[0]):
        centerline_rows.append(
            {
                "x_cm": float(x_cm[ix]),
                "dose_gy": float(dose_grid[ix, center_y, target_depth_idx]),
                "gammah2ax_proxy_au": float(gamma_h2ax_map[ix, center_y, target_depth_idx]),
                "ros_peak_au": float(peak_concentration[0, ix, center_y, target_depth_idx]),
                "cytokine_final_au": float(concentration[1, ix, center_y, target_depth_idx]),
                "hazard_total": float(hazard_grid[ix, center_y, target_depth_idx]),
                "tunel_proxy": float(tunel_map[ix, center_y, target_depth_idx]),
                "survival_total": float(final_survival[ix, center_y, target_depth_idx]),
                "deff_gy": float(deff_grid[ix, center_y, target_depth_idx]),
            }
        )
    write_csv(centerline_rows, centerline_csv)

    np.savez_compressed(
        volumes_npz,
        dose_grid=dose_grid.astype(np.float32, copy=False),
        lq_survival=lq_survival.astype(np.float32, copy=False),
        final_survival=final_survival.astype(np.float32, copy=False),
        deff_grid=deff_grid.astype(np.float32, copy=False),
        hazard_grid=hazard_grid.astype(np.float32, copy=False),
        ros_hazard_grid=ros_hazard_grid.astype(np.float32, copy=False),
        cytokine_hazard_grid=cytokine_hazard_grid.astype(np.float32, copy=False),
        concentration=concentration.astype(np.float32, copy=False),
        peak_concentration=peak_concentration.astype(np.float32, copy=False),
        gamma_h2ax_map=gamma_h2ax_map.astype(np.float32, copy=False),
        tunel_map=tunel_map.astype(np.float32, copy=False),
        x_cm=x_cm.astype(np.float32, copy=False),
        y_cm=y_cm.astype(np.float32, copy=False),
        z_cm=z_cm.astype(np.float32, copy=False),
        time_axis=time_axis.astype(np.float32, copy=False),
        global_history=global_history.astype(np.float32, copy=False),
    )

    summary = {
        "phase": "Phase 11C",
        "description": "In silico assay extraction for the best 4 mm Path A case.",
        "geometry_case": {
            "spot_mode": str(args.spot_mode),
            "layout_mode": str(args.layout_mode),
            "radius_mm": float(args.radius_mm),
            "peak_boundary_mm": float(geometry["peak_boundary_mm"]),
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
        "assay_proxies": {
            "gammah2ax_definition": "1 - exp(-(alpha*dose + beta*dose^2 + 0.45*normalized_peak_ROS))",
            "elisa_definition": "Mean cytokine concentration over time in whole-volume, peak ROI, and valley ROI.",
            "tunel_definition": "1 - final_survival (proxy apoptotic fraction).",
        },
        "sampled_depth_cm": float(z_cm[target_depth_idx]),
        "center_metrics": {
            "dose_gy": float(dose_grid[center_x, center_y, target_depth_idx]),
            "gammah2ax_proxy_au": float(gamma_h2ax_map[center_x, center_y, target_depth_idx]),
            "tunel_proxy": float(tunel_map[center_x, center_y, target_depth_idx]),
            "survival_total": float(final_survival[center_x, center_y, target_depth_idx]),
            "deff_gy": float(deff_grid[center_x, center_y, target_depth_idx]),
        },
        "roi_metrics": {
            "valley_threshold_gy": float(valley_threshold_gy),
            "peak_threshold_gy": float(peak_threshold_gy),
            "mean_gammah2ax_valley": float(np.mean(gamma_h2ax_map[valley_mask])),
            "mean_gammah2ax_peak": float(np.mean(gamma_h2ax_map[peak_mask])),
            "mean_tunel_valley": float(np.mean(tunel_map[valley_mask])),
            "mean_tunel_peak": float(np.mean(tunel_map[peak_mask])),
            "mean_cytokine_final_valley": float(np.mean(concentration[1][valley_mask])),
            "mean_cytokine_final_peak": float(np.mean(concentration[1][peak_mask])),
        },
        "elisa_curve_metrics": {
            "time_final_au": float(time_axis[-1]) if time_axis.size else 0.0,
            "cytokine_global_peak": float(np.max(global_history[1])) if global_history.size else 0.0,
            "cytokine_global_auc": float(np.trapz(global_history[1], time_axis)) if global_history.size else 0.0,
            "cytokine_peak_roi_auc": (
                float(np.trapz(mask_history["peak_roi"][1], time_axis))
                if "peak_roi" in mask_history and mask_history["peak_roi"].size
                else 0.0
            ),
            "cytokine_valley_roi_auc": (
                float(np.trapz(mask_history["valley_roi"][1], time_axis))
                if "valley_roi" in mask_history and mask_history["valley_roi"].size
                else 0.0
            ),
        },
        "systemic_immune_penalty": float(immune_penalty),
        "icd_volume_cm3": float(icd_volume_cm3),
        "outputs": {
            "figure_gammah2ax": str(fig1),
            "figure_elisa_curve": str(fig2),
            "figure_tunel": str(fig3),
            "elisa_curve_csv": str(curve_csv),
            "centerline_csv": str(centerline_csv),
            "volumes_npz": str(volumes_npz),
            "summary_json": str(summary_json),
        },
    }
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Phase 11C gammaH2AX map: {fig1}")
    print(f"Phase 11C ELISA curve: {fig2}")
    print(f"Phase 11C TUNEL map: {fig3}")
    print(f"Phase 11C summary: {summary_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
