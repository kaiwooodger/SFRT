#!/usr/bin/env python
"""Phase 10: synthetic complex lattice plan with a multi-vessel network."""

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
    run_pde_temporal_integration,
)
from geometry_generators import generate_complex_lattice_plan_geometry

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception as exc:  # pragma: no cover
    raise RuntimeError("matplotlib is required for figure generation") from exc


LOCKED_D_CYTO = 1.2
LOCKED_LAMBDA_CYTO = 0.001
LOCKED_GAMMA = 0.35
LOCKED_SCALING_FACTOR = 0.0029365813
W_ROS = 0.4
W_CYTO = 0.4
W_IMMUNE = 0.2


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description=(
            "Run a synthetic complex lattice surrogate plan with an irregular "
            "spot arrangement and multi-vessel sink network."
        )
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=root / "runs" / "linac_6mv_polyenergetic_clinical_sfrt" / "analysis_phase10_complex_lattice_plan",
        help="Directory for Phase 10 outputs.",
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
        help="Reference depth for summary extraction and transverse slices.",
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


def plot_complex_geometry(
    *,
    dose_grid: np.ndarray,
    x_cm: np.ndarray,
    y_cm: np.ndarray,
    z_cm: np.ndarray,
    target_depth_idx: int,
    out_file: Path,
    dpi: int,
) -> None:
    center_y = dose_grid.shape[1] // 2
    xy_slice = dose_grid[:, :, target_depth_idx].T
    xz_slice = dose_grid[:, center_y, :].T

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.4), constrained_layout=True)
    img0 = axes[0].imshow(
        xy_slice,
        origin="lower",
        extent=_slice_extent_xy(x_cm, y_cm),
        aspect="equal",
        cmap="inferno",
        vmin=0.0,
        vmax=float(np.max(dose_grid)),
    )
    axes[0].set_title(f"x-y dose at z = {float(z_cm[target_depth_idx]):.2f} cm")
    axes[0].set_xlabel("x (cm)")
    axes[0].set_ylabel("y (cm)")

    img1 = axes[1].imshow(
        xz_slice,
        origin="lower",
        extent=_slice_extent_xz(x_cm, z_cm),
        aspect="auto",
        cmap="inferno",
        vmin=0.0,
        vmax=float(np.max(dose_grid)),
    )
    axes[1].axhline(float(z_cm[target_depth_idx]), color="#ffffff", linestyle="--", linewidth=1.4)
    axes[1].set_title("x-z dose at center y")
    axes[1].set_xlabel("x (cm)")
    axes[1].set_ylabel("z (cm)")

    cbar = fig.colorbar(img1, ax=axes.ravel().tolist(), shrink=0.9)
    cbar.set_label("Dose (Gy)")
    fig.suptitle("Figure 1: Phase 10 complex lattice geometry", fontsize=15)
    fig.savefig(out_file, dpi=dpi)
    plt.close(fig)


def plot_survival_pair(
    *,
    volume: np.ndarray,
    title: str,
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

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.4), constrained_layout=True)
    img0 = axes[0].imshow(
        xy_slice,
        origin="lower",
        extent=_slice_extent_xy(x_cm, y_cm),
        aspect="equal",
        cmap="viridis",
        vmin=0.0,
        vmax=1.0,
    )
    axes[0].set_title(f"x-y at z = {float(z_cm[target_depth_idx]):.2f} cm")
    axes[0].set_xlabel("x (cm)")
    axes[0].set_ylabel("y (cm)")

    img1 = axes[1].imshow(
        xz_slice,
        origin="lower",
        extent=_slice_extent_xz(x_cm, z_cm),
        aspect="auto",
        cmap="viridis",
        vmin=0.0,
        vmax=1.0,
    )
    axes[1].axhline(float(z_cm[target_depth_idx]), color="#ffffff", linestyle="--", linewidth=1.4)
    axes[1].set_title("x-z at center y")
    axes[1].set_xlabel("x (cm)")
    axes[1].set_ylabel("z (cm)")

    cbar = fig.colorbar(img1, ax=axes.ravel().tolist(), shrink=0.9)
    cbar.set_label("Survival fraction")
    fig.suptitle(title, fontsize=15)
    fig.savefig(out_file, dpi=dpi)
    plt.close(fig)


def plot_deff(
    *,
    deff_grid: np.ndarray,
    x_cm: np.ndarray,
    y_cm: np.ndarray,
    z_cm: np.ndarray,
    target_depth_idx: int,
    out_file: Path,
    dpi: int,
) -> None:
    center_y = deff_grid.shape[1] // 2
    xy_slice = deff_grid[:, :, target_depth_idx].T
    xz_slice = deff_grid[:, center_y, :].T

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.4), constrained_layout=True)
    img0 = axes[0].imshow(
        xy_slice,
        origin="lower",
        extent=_slice_extent_xy(x_cm, y_cm),
        aspect="equal",
        cmap="magma",
        vmin=0.0,
        vmax=float(np.percentile(deff_grid, 99.5)),
    )
    axes[0].set_title(f"x-y at z = {float(z_cm[target_depth_idx]):.2f} cm")
    axes[0].set_xlabel("x (cm)")
    axes[0].set_ylabel("y (cm)")

    img1 = axes[1].imshow(
        xz_slice,
        origin="lower",
        extent=_slice_extent_xz(x_cm, z_cm),
        aspect="auto",
        cmap="magma",
        vmin=0.0,
        vmax=float(np.percentile(deff_grid, 99.5)),
    )
    axes[1].axhline(float(z_cm[target_depth_idx]), color="#ffffff", linestyle="--", linewidth=1.4)
    axes[1].set_title("x-z at center y")
    axes[1].set_xlabel("x (cm)")
    axes[1].set_ylabel("z (cm)")

    cbar = fig.colorbar(img1, ax=axes.ravel().tolist(), shrink=0.9)
    cbar.set_label("Effective dose (Gy)")
    fig.suptitle("Figure 4: Phase 10 effective dose", fontsize=15)
    fig.savefig(out_file, dpi=dpi)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    print("=== EXECUTING PHASE 10 COMPLEX LATTICE SURROGATE ===")
    print(
        f"Locked Parameters: D={LOCKED_D_CYTO}, lambda={LOCKED_LAMBDA_CYTO}, "
        f"gamma={LOCKED_GAMMA}, scaling={LOCKED_SCALING_FACTOR}"
    )

    dose_grid, uptake_tensor, m_oxygen, m_type, meta = generate_complex_lattice_plan_geometry(
        csv=args.csv,
        prescribed_peak_dose_gy=float(args.prescribed_peak_dose_gy),
        uniform_dose_floor_fraction=float(args.uniform_dose_floor_fraction),
    )

    voxel_size_mm = tuple(float(value) for value in meta["voxel_size_mm"])
    x_cm = np.asarray(meta["x_cm"], dtype=np.float32)
    y_cm = centered_axis_cm(dose_grid.shape[1], voxel_size_mm[1] / 10.0)
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
    deff_grid = calculate_effective_dose(
        final_survival,
        alpha=float(args.alpha),
        beta=float(args.beta),
    )
    immune_penalty, icd_volume_cm3 = calculate_systemic_immune_penalty(dose_grid, voxel_size_mm)

    center_metrics = {
        "sampled_depth_cm": float(z_cm[target_depth_idx]),
        "center_dose_gy": float(dose_grid[center_x, center_y, target_depth_idx]),
        "center_survival_lq": float(lq_survival[center_x, center_y, target_depth_idx]),
        "center_survival_total": float(final_survival[center_x, center_y, target_depth_idx]),
        "center_hazard": float(hazard_grid[center_x, center_y, target_depth_idx]),
        "center_deff_gy": float(deff_grid[center_x, center_y, target_depth_idx]),
    }
    print(
        f"Center readout at {float(z_cm[target_depth_idx]) * 10.0:.2f} mm: "
        f"dose={center_metrics['center_dose_gy']:.3f} Gy, "
        f"LQ={center_metrics['center_survival_lq']:.4f}, "
        f"Phase10={center_metrics['center_survival_total']:.4f}, "
        f"Deff={center_metrics['center_deff_gy']:.2f} Gy"
    )

    rows = [
        {
            "x_cm": float(x_cm[ix]),
            "dose_gy": float(dose_grid[ix, center_y, target_depth_idx]),
            "survival_lq": float(lq_survival[ix, center_y, target_depth_idx]),
            "survival_total": float(final_survival[ix, center_y, target_depth_idx]),
            "hazard": float(hazard_grid[ix, center_y, target_depth_idx]),
            "deff_gy": float(deff_grid[ix, center_y, target_depth_idx]),
        }
        for ix in range(dose_grid.shape[0])
    ]
    csv_file = args.outdir / "phase10_complex_lattice_centerline.csv"
    write_csv(rows, csv_file)

    fig1 = args.outdir / "figure1_phase10_complex_lattice_geometry.png"
    fig2 = args.outdir / "figure2_phase10_lq_survival.png"
    fig3 = args.outdir / "figure3_phase10_model_survival.png"
    fig4 = args.outdir / "figure4_phase10_effective_dose.png"
    volumes_npz = args.outdir / "phase10_complex_lattice_volumes.npz"
    plot_complex_geometry(
        dose_grid=dose_grid,
        x_cm=x_cm,
        y_cm=y_cm,
        z_cm=z_cm,
        target_depth_idx=int(target_depth_idx),
        out_file=fig1,
        dpi=int(args.dpi),
    )
    plot_survival_pair(
        volume=lq_survival,
        title="Figure 2: Phase 10 simplified LQ survival",
        x_cm=x_cm,
        y_cm=y_cm,
        z_cm=z_cm,
        target_depth_idx=int(target_depth_idx),
        out_file=fig2,
        dpi=int(args.dpi),
    )
    plot_survival_pair(
        volume=final_survival,
        title="Figure 3: Phase 10 full-model survival",
        x_cm=x_cm,
        y_cm=y_cm,
        z_cm=z_cm,
        target_depth_idx=int(target_depth_idx),
        out_file=fig3,
        dpi=int(args.dpi),
    )
    plot_deff(
        deff_grid=deff_grid,
        x_cm=x_cm,
        y_cm=y_cm,
        z_cm=z_cm,
        target_depth_idx=int(target_depth_idx),
        out_file=fig4,
        dpi=int(args.dpi),
    )
    np.savez_compressed(
        volumes_npz,
        dose_grid=dose_grid.astype(np.float32, copy=False),
        lq_survival=lq_survival.astype(np.float32, copy=False),
        final_survival=final_survival.astype(np.float32, copy=False),
        hazard_grid=hazard_grid.astype(np.float32, copy=False),
        deff_grid=deff_grid.astype(np.float32, copy=False),
        x_cm=x_cm.astype(np.float32, copy=False),
        y_cm=y_cm.astype(np.float32, copy=False),
        z_cm=z_cm.astype(np.float32, copy=False),
    )

    summary = {
        "phase": "Phase 10",
        "description": "Synthetic complex lattice surrogate with irregular weighted spots and multi-vessel sink network.",
        "locked_parameters": {
            "D_cyto": LOCKED_D_CYTO,
            "lambda_cyto": LOCKED_LAMBDA_CYTO,
            "gamma": LOCKED_GAMMA,
            "scaling_factor": LOCKED_SCALING_FACTOR,
            "weight_ros": W_ROS,
            "weight_cyto": W_CYTO,
            "weight_immune": W_IMMUNE,
        },
        "lq_model": {
            "alpha": float(args.alpha),
            "beta": float(args.beta),
        },
        "inputs": {
            "csv": str(args.csv),
            "prescribed_peak_dose_gy": float(args.prescribed_peak_dose_gy),
            "uniform_dose_floor_fraction": float(args.uniform_dose_floor_fraction),
            "target_depth_cm": float(args.target_depth_cm),
        },
        "geometry": meta,
        "systemic_immune_penalty": float(immune_penalty),
        "icd_volume_cm3": float(icd_volume_cm3),
        "center_metrics": center_metrics,
        "outputs": {
            "centerline_csv": str(csv_file),
            "figure_geometry": str(fig1),
            "figure_lq": str(fig2),
            "figure_model": str(fig3),
            "figure_deff": str(fig4),
            "volumes_npz": str(volumes_npz),
        },
    }
    summary_file = args.outdir / "phase10_complex_lattice_summary.json"
    summary_file.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Phase 10 summary: {summary_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
