#!/usr/bin/env python
"""Generate extended Phase 10 outputs: valley map, EUD shift, and local sensitivity."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from analyze_250mev_sfrt_plan import centered_axis_cm, nearest_index
from bystander_multispecies_pde_solver import (
    calculate_effective_dose,
    calculate_phase7_survival,
    calculate_systemic_immune_penalty,
    run_pde_temporal_integration,
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
            "Generate extended Phase 10 outputs including a valley survival map, "
            "nonlocal effective-dose shift, and local uncertainty/sensitivity figures."
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
        "--valley-threshold-fraction",
        type=float,
        default=0.40,
        help="Physical-dose fraction of the prescribed peak used to define the valley ROI.",
    )
    parser.add_argument(
        "--sensitivity-fraction",
        type=float,
        default=0.20,
        help="One-at-a-time relative perturbation used for local uncertainty and ranking.",
    )
    parser.add_argument(
        "--crop-half-width-x-mm",
        type=float,
        default=60.0,
        help="Half-width of the cropped ROI along x for the local sensitivity runs.",
    )
    parser.add_argument(
        "--crop-half-width-y-mm",
        type=float,
        default=60.0,
        help="Half-width of the cropped ROI along y for the local sensitivity runs.",
    )
    parser.add_argument(
        "--profile-half-width-mm",
        type=float,
        default=40.0,
        help="Half-width of the exported 1D centerline uncertainty profile.",
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


def crop_center_xy(
    grid: np.ndarray,
    *,
    dx_mm: float,
    dy_mm: float,
    half_width_x_mm: float,
    half_width_y_mm: float,
) -> tuple[np.ndarray, slice, slice]:
    center_x = int(grid.shape[0] // 2)
    center_y = int(grid.shape[1] // 2)
    half_bins_x = int(round(float(half_width_x_mm) / float(dx_mm)))
    half_bins_y = int(round(float(half_width_y_mm) / float(dy_mm)))
    x_slice = slice(max(0, center_x - half_bins_x), min(grid.shape[0], center_x + half_bins_x + 1))
    y_slice = slice(max(0, center_y - half_bins_y), min(grid.shape[1], center_y + half_bins_y + 1))
    return grid[x_slice, y_slice, :].copy(), x_slice, y_slice


def rebuild_baseline(
    summary: Dict[str, object],
) -> Dict[str, object]:
    inputs = summary["inputs"]
    lq_model = summary["lq_model"]
    dose_grid, uptake_tensor, m_oxygen, m_type, meta = generate_complex_lattice_plan_geometry(
        csv=Path(str(inputs["csv"])),
        prescribed_peak_dose_gy=float(inputs["prescribed_peak_dose_gy"]),
        uniform_dose_floor_fraction=float(inputs["uniform_dose_floor_fraction"]),
    )
    voxel_size_mm = tuple(float(value) for value in meta["voxel_size_mm"])
    alpha = float(lq_model["alpha"])
    beta = float(lq_model["beta"])
    lq_survival = np.exp(-alpha * dose_grid - beta * dose_grid**2).astype(np.float32)
    hazard_grid = run_pde_temporal_integration(
        dose_grid,
        voxel_size_mm,
        D_cyto=float(summary["locked_parameters"]["D_cyto"]),
        lambda_cyto=float(summary["locked_parameters"]["lambda_cyto"]),
        gamma=float(summary["locked_parameters"]["gamma"]),
        u_k=uptake_tensor,
        M_oxygen=m_oxygen,
        M_type=m_type,
        D_ros=0.8,
        lambda_ros=0.2,
        Emax_ros=1.5,
        Emax_cyto=0.8,
        w_ros=float(summary["locked_parameters"]["weight_ros"]),
        w_cyto=float(summary["locked_parameters"]["weight_cyto"]),
        steps=400,
        dt=0.12,
        progress_interval=50,
        verbose=True,
    )
    final_survival = calculate_phase7_survival(
        lq_survival,
        hazard_grid,
        dose_grid,
        voxel_size_mm,
        float(summary["locked_parameters"]["scaling_factor"]),
        weight_immune=float(summary["locked_parameters"]["weight_immune"]),
        verbose=False,
    )
    deff_grid = calculate_effective_dose(final_survival, alpha=alpha, beta=beta)
    x_cm = np.asarray(meta["x_cm"], dtype=np.float32)
    y_cm = centered_axis_cm(dose_grid.shape[1], float(meta["voxel_size_mm"][1]) / 10.0)
    z_cm = np.asarray(meta["z_cm"], dtype=np.float32)
    return {
        "dose_grid": dose_grid.astype(np.float32, copy=False),
        "lq_survival": lq_survival.astype(np.float32, copy=False),
        "final_survival": final_survival.astype(np.float32, copy=False),
        "hazard_grid": hazard_grid.astype(np.float32, copy=False),
        "deff_grid": deff_grid.astype(np.float32, copy=False),
        "uptake_tensor": uptake_tensor.astype(np.float32, copy=False),
        "m_oxygen": m_oxygen.astype(np.float32, copy=False),
        "m_type": m_type.astype(np.float32, copy=False),
        "meta": meta,
        "x_cm": x_cm,
        "y_cm": y_cm,
        "z_cm": z_cm,
        "voxel_size_mm": voxel_size_mm,
    }


def load_geometry_tensors(summary: Dict[str, object]) -> Dict[str, object]:
    inputs = summary["inputs"]
    dose_grid, uptake_tensor, m_oxygen, m_type, meta = generate_complex_lattice_plan_geometry(
        csv=Path(str(inputs["csv"])),
        prescribed_peak_dose_gy=float(inputs["prescribed_peak_dose_gy"]),
        uniform_dose_floor_fraction=float(inputs["uniform_dose_floor_fraction"]),
    )
    voxel_size_mm = tuple(float(value) for value in meta["voxel_size_mm"])
    x_cm = np.asarray(meta["x_cm"], dtype=np.float32)
    y_cm = centered_axis_cm(dose_grid.shape[1], float(meta["voxel_size_mm"][1]) / 10.0)
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
    }


def load_baseline_solution(summary: Dict[str, object]) -> Dict[str, object]:
    volumes_path = summary.get("outputs", {}).get("volumes_npz")
    if volumes_path and Path(str(volumes_path)).exists():
        data = np.load(Path(str(volumes_path)))
        meta = summary["geometry"]
        return {
            "dose_grid": np.asarray(data["dose_grid"], dtype=np.float32),
            "lq_survival": np.asarray(data["lq_survival"], dtype=np.float32),
            "final_survival": np.asarray(data["final_survival"], dtype=np.float32),
            "hazard_grid": np.asarray(data["hazard_grid"], dtype=np.float32),
            "deff_grid": np.asarray(data["deff_grid"], dtype=np.float32),
            "meta": meta,
            "x_cm": np.asarray(data["x_cm"], dtype=np.float32),
            "y_cm": np.asarray(data["y_cm"], dtype=np.float32),
            "z_cm": np.asarray(data["z_cm"], dtype=np.float32),
            "voxel_size_mm": tuple(float(value) for value in meta["voxel_size_mm"]),
        }
    return rebuild_baseline(summary)


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


def plot_valley_survival_map(
    *,
    final_survival: np.ndarray,
    dose_grid: np.ndarray,
    x_cm: np.ndarray,
    y_cm: np.ndarray,
    z_cm: np.ndarray,
    target_depth_idx: int,
    valley_threshold_gy: float,
    out_file: Path,
    dpi: int,
) -> None:
    xy_survival = final_survival[:, :, target_depth_idx].T
    xy_dose = dose_grid[:, :, target_depth_idx].T
    masked = np.ma.masked_where(xy_dose > float(valley_threshold_gy), xy_survival)

    fig, ax = plt.subplots(1, 1, figsize=(6.6, 5.8), constrained_layout=True)
    cmap = plt.cm.viridis.copy()
    cmap.set_bad(color="#3b3b3b")
    img = ax.imshow(
        masked,
        origin="lower",
        extent=[float(x_cm[0]), float(x_cm[-1]), float(y_cm[0]), float(y_cm[-1])],
        aspect="equal",
        cmap=cmap,
        vmin=0.0,
        vmax=1.0,
    )
    ax.contour(
        x_cm,
        y_cm,
        xy_dose,
        levels=[float(valley_threshold_gy)],
        colors=["#ffffff"],
        linewidths=1.0,
        linestyles="--",
    )
    ax.set_title(
        f"Figure 5: Phase 10 valley survival map\n"
        f"z = {float(z_cm[target_depth_idx]):.2f} cm, valley <= {float(valley_threshold_gy):.2f} Gy"
    )
    ax.set_xlabel("x (cm)")
    ax.set_ylabel("y (cm)")
    cbar = fig.colorbar(img, ax=ax, shrink=0.92)
    cbar.set_label("Final survival fraction")
    fig.savefig(out_file, dpi=dpi)
    plt.close(fig)


def plot_nonlocal_eud_shift(
    *,
    delta_deff: np.ndarray,
    x_cm: np.ndarray,
    y_cm: np.ndarray,
    z_cm: np.ndarray,
    target_depth_idx: int,
    out_file: Path,
    dpi: int,
) -> None:
    center_y = delta_deff.shape[1] // 2
    xy_slice = delta_deff[:, :, target_depth_idx].T
    xz_slice = delta_deff[:, center_y, :].T
    vmax = float(np.percentile(delta_deff, 99.0))

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.2), constrained_layout=True)
    img0 = axes[0].imshow(
        xy_slice,
        origin="lower",
        extent=[float(x_cm[0]), float(x_cm[-1]), float(y_cm[0]), float(y_cm[-1])],
        aspect="equal",
        cmap="magma",
        vmin=0.0,
        vmax=vmax,
    )
    axes[0].set_title(f"x-y dose-lift at z = {float(z_cm[target_depth_idx]):.2f} cm")
    axes[0].set_xlabel("x (cm)")
    axes[0].set_ylabel("y (cm)")

    img1 = axes[1].imshow(
        xz_slice,
        origin="lower",
        extent=[float(x_cm[0]), float(x_cm[-1]), float(z_cm[0]), float(z_cm[-1])],
        aspect="auto",
        cmap="magma",
        vmin=0.0,
        vmax=vmax,
    )
    axes[1].axhline(float(z_cm[target_depth_idx]), color="#ffffff", linestyle="--", linewidth=1.2)
    axes[1].set_title("x-z dose-lift at center y")
    axes[1].set_xlabel("x (cm)")
    axes[1].set_ylabel("z (cm)")

    cbar = fig.colorbar(img1, ax=axes.ravel().tolist(), shrink=0.92)
    cbar.set_label("D_eff - physical dose (Gy)")
    fig.suptitle("Figure 6: Phase 10 nonlocal EUD shift", fontsize=14)
    fig.savefig(out_file, dpi=dpi)
    plt.close(fig)


def solve_local_case(
    *,
    dose_grid: np.ndarray,
    uptake_tensor: np.ndarray,
    m_oxygen: np.ndarray,
    m_type: np.ndarray,
    voxel_size_mm: tuple[float, float, float],
    alpha: float,
    beta: float,
    d_cyto: float,
    lambda_cyto: float,
    gamma: float,
    emax_cyto: float,
    scaling_factor: float,
    weight_immune: float,
    pde_steps: int = 400,
    pde_dt: float = 0.12,
) -> Dict[str, np.ndarray]:
    lq_survival = np.exp(-float(alpha) * dose_grid - float(beta) * dose_grid**2).astype(np.float32)
    hazard_grid = run_pde_temporal_integration(
        dose_grid,
        voxel_size_mm,
        D_cyto=float(d_cyto),
        lambda_cyto=float(lambda_cyto),
        gamma=float(gamma),
        u_k=uptake_tensor,
        M_oxygen=m_oxygen,
        M_type=m_type,
        D_ros=0.8,
        lambda_ros=0.2,
        Emax_ros=1.5,
        Emax_cyto=float(emax_cyto),
        w_ros=float(W_ROS),
        w_cyto=float(W_CYTO),
        steps=int(pde_steps),
        dt=float(pde_dt),
        progress_interval=50,
        verbose=True,
    )
    final_survival = calculate_phase7_survival(
        lq_survival,
        hazard_grid,
        dose_grid,
        voxel_size_mm,
        float(scaling_factor),
        weight_immune=float(weight_immune),
        verbose=False,
    )
    deff_grid = calculate_effective_dose(final_survival, alpha=float(alpha), beta=float(beta))
    return {
        "lq_survival": lq_survival,
        "hazard_grid": hazard_grid,
        "final_survival": final_survival,
        "deff_grid": deff_grid,
    }


def plot_uncertainty_band(
    *,
    x_cm: np.ndarray,
    baseline_profile: np.ndarray,
    lower_profile: np.ndarray,
    upper_profile: np.ndarray,
    out_file: Path,
    dpi: int,
) -> None:
    fig, ax = plt.subplots(1, 1, figsize=(8.4, 4.8), constrained_layout=True)
    ax.fill_between(x_cm, lower_profile, upper_profile, color="#d62728", alpha=0.18, label="Local uncertainty band")
    ax.plot(x_cm, baseline_profile, color="#1f77b4", linewidth=2.0, label="Baseline Phase 10")
    ax.set_xlabel("x (cm)")
    ax.set_ylabel("Final survival fraction")
    ax.set_ylim(0.0, 1.0)
    ax.set_title("Figure 7: Phase 10 centerline uncertainty band")
    ax.grid(alpha=0.25)
    ax.legend(loc="lower right")
    fig.savefig(out_file, dpi=dpi)
    plt.close(fig)


def plot_sensitivity_ranking(
    rows: List[Dict[str, object]],
    *,
    out_file: Path,
    dpi: int,
) -> None:
    params = [str(row["parameter"]) for row in rows]
    survival_delta = [float(row["max_abs_delta_center_survival"]) for row in rows]
    eud_delta = [float(row["max_abs_delta_valley_eud_shift_gy"]) for row in rows]
    order = np.argsort(survival_delta)
    params = [params[idx] for idx in order]
    survival_delta = [survival_delta[idx] for idx in order]
    eud_delta = [eud_delta[idx] for idx in order]

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.8), constrained_layout=True)
    axes[0].barh(params, survival_delta, color="#1f77b4")
    axes[0].set_xlabel("Max |delta center survival|")
    axes[0].set_title("Local ranking of survival sensitivity")
    axes[0].grid(axis="x", alpha=0.25)

    axes[1].barh(params, eud_delta, color="#d62728")
    axes[1].set_xlabel("Max |delta valley EUD shift| (Gy)")
    axes[1].set_title("Local ranking of valley EUD shift")
    axes[1].grid(axis="x", alpha=0.25)

    fig.suptitle("Figure 8: Phase 10 parameter sensitivity ranking", fontsize=14)
    fig.savefig(out_file, dpi=dpi)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    summary = json.loads(args.phase10_summary.read_text(encoding="utf-8"))
    outdir = args.outdir or args.phase10_summary.parent
    outdir.mkdir(parents=True, exist_ok=True)

    baseline = load_baseline_solution(summary)
    dose_grid = baseline["dose_grid"]
    lq_survival = baseline["lq_survival"]
    final_survival = baseline["final_survival"]
    deff_grid = baseline["deff_grid"]
    x_cm = baseline["x_cm"]
    y_cm = baseline["y_cm"]
    z_cm = baseline["z_cm"]
    voxel_size_mm = tuple(float(value) for value in baseline["voxel_size_mm"])

    alpha = float(summary["lq_model"]["alpha"])
    beta = float(summary["lq_model"]["beta"])
    prescribed_peak_dose_gy = float(summary["inputs"]["prescribed_peak_dose_gy"])
    target_depth_cm = float(summary["inputs"]["target_depth_cm"])
    target_depth_idx = nearest_index(z_cm, target_depth_cm)

    valley_threshold_gy = float(args.valley_threshold_fraction) * float(prescribed_peak_dose_gy)
    valley_mask_3d = dose_grid <= float(valley_threshold_gy)
    valley_mask_xy = dose_grid[:, :, target_depth_idx] <= float(valley_threshold_gy)
    delta_deff = (deff_grid - dose_grid).astype(np.float32)

    valley_eud_lq = equivalent_uniform_dose_from_survival(
        lq_survival[valley_mask_3d],
        alpha=alpha,
        beta=beta,
    )
    valley_eud_phase10 = equivalent_uniform_dose_from_survival(
        final_survival[valley_mask_3d],
        alpha=alpha,
        beta=beta,
    )
    valley_eud_shift = float(valley_eud_phase10 - valley_eud_lq)

    fig5 = Path(outdir) / "figure5_phase10_valley_survival_map.png"
    fig6 = Path(outdir) / "figure6_phase10_nonlocal_eud_shift.png"
    plot_valley_survival_map(
        final_survival=final_survival,
        dose_grid=dose_grid,
        x_cm=x_cm,
        y_cm=y_cm,
        z_cm=z_cm,
        target_depth_idx=int(target_depth_idx),
        valley_threshold_gy=float(valley_threshold_gy),
        out_file=fig5,
        dpi=int(args.dpi),
    )
    plot_nonlocal_eud_shift(
        delta_deff=delta_deff,
        x_cm=x_cm,
        y_cm=y_cm,
        z_cm=z_cm,
        target_depth_idx=int(target_depth_idx),
        out_file=fig6,
        dpi=int(args.dpi),
    )

    # Rebuild once with tensors for the local sensitivity layer.
    geometry = load_geometry_tensors(summary)
    cropped_dose, x_slice, y_slice = crop_center_xy(
        geometry["dose_grid"],
        dx_mm=float(voxel_size_mm[0]),
        dy_mm=float(voxel_size_mm[1]),
        half_width_x_mm=float(args.crop_half_width_x_mm),
        half_width_y_mm=float(args.crop_half_width_y_mm),
    )
    cropped_uptake = geometry["uptake_tensor"][:, x_slice, y_slice, :].copy()
    cropped_oxygen = geometry["m_oxygen"][:, x_slice, y_slice, :].copy()
    cropped_type = geometry["m_type"][:, x_slice, y_slice, :].copy()
    x_cm_crop = geometry["x_cm"][x_slice]
    z_cm_crop = geometry["z_cm"]
    profile_mask = np.abs(x_cm_crop) <= (float(args.profile_half_width_mm) / 10.0)
    z_idx_crop = nearest_index(z_cm_crop, float(target_depth_cm))
    center_x_crop = cropped_dose.shape[0] // 2
    center_y_crop = cropped_dose.shape[1] // 2

    baseline_local = solve_local_case(
        dose_grid=cropped_dose,
        uptake_tensor=cropped_uptake,
        m_oxygen=cropped_oxygen,
        m_type=cropped_type,
        voxel_size_mm=voxel_size_mm,
        alpha=alpha,
        beta=beta,
        d_cyto=LOCKED_D_CYTO,
        lambda_cyto=LOCKED_LAMBDA_CYTO,
        gamma=LOCKED_GAMMA,
        emax_cyto=0.8,
        scaling_factor=LOCKED_SCALING_FACTOR,
        weight_immune=W_IMMUNE,
    )
    baseline_profile = baseline_local["final_survival"][:, center_y_crop, z_idx_crop][profile_mask]
    baseline_center_survival = float(baseline_local["final_survival"][center_x_crop, center_y_crop, z_idx_crop])
    cropped_valley_mask = cropped_dose <= float(valley_threshold_gy)
    baseline_valley_eud_shift = (
        equivalent_uniform_dose_from_survival(
            baseline_local["final_survival"][cropped_valley_mask],
            alpha=alpha,
            beta=beta,
        )
        - equivalent_uniform_dose_from_survival(
            baseline_local["lq_survival"][cropped_valley_mask],
            alpha=alpha,
            beta=beta,
        )
    )

    frac = float(args.sensitivity_fraction)
    param_cases = [
        ("D_cyto", LOCKED_D_CYTO * (1.0 - frac), LOCKED_D_CYTO * (1.0 + frac)),
        ("lambda_cyto", LOCKED_LAMBDA_CYTO * (1.0 - frac), LOCKED_LAMBDA_CYTO * (1.0 + frac)),
        ("gamma", LOCKED_GAMMA * (1.0 - frac), LOCKED_GAMMA * (1.0 + frac)),
        ("Emax_cyto", 0.8 * (1.0 - frac), 0.8 * (1.0 + frac)),
        ("weight_immune", W_IMMUNE * (1.0 - frac), W_IMMUNE * (1.0 + frac)),
        ("cyto_vessel_uptake_scale", 1.0 - frac, 1.0 + frac),
    ]

    profile_stack = [baseline_profile.astype(np.float32)]
    sensitivity_rows: List[Dict[str, object]] = []

    for parameter, low_value, high_value in param_cases:
        print(f"Running Phase 10 local sensitivity: {parameter}", flush=True)
        case_profiles: List[np.ndarray] = []
        case_center_survival: List[float] = []
        case_valley_eud_shift: List[float] = []

        for value in [low_value, high_value]:
            case_uptake = cropped_uptake
            d_cyto = LOCKED_D_CYTO
            lambda_cyto = LOCKED_LAMBDA_CYTO
            gamma = LOCKED_GAMMA
            emax_cyto = 0.8
            scaling_factor = LOCKED_SCALING_FACTOR
            weight_immune = W_IMMUNE

            if parameter == "D_cyto":
                d_cyto = float(value)
            elif parameter == "lambda_cyto":
                lambda_cyto = float(value)
            elif parameter == "gamma":
                gamma = float(value)
            elif parameter == "Emax_cyto":
                emax_cyto = float(value)
            elif parameter == "weight_immune":
                weight_immune = float(value)
            elif parameter == "cyto_vessel_uptake_scale":
                case_uptake = cropped_uptake.copy()
                case_uptake[1] = case_uptake[1] * float(value)

            solved = solve_local_case(
                dose_grid=cropped_dose,
                uptake_tensor=case_uptake,
                m_oxygen=cropped_oxygen,
                m_type=cropped_type,
                voxel_size_mm=voxel_size_mm,
                alpha=alpha,
                beta=beta,
                d_cyto=d_cyto,
                lambda_cyto=lambda_cyto,
                gamma=gamma,
                emax_cyto=emax_cyto,
                scaling_factor=scaling_factor,
                weight_immune=weight_immune,
            )
            profile = solved["final_survival"][:, center_y_crop, z_idx_crop][profile_mask]
            case_profiles.append(profile.astype(np.float32))
            profile_stack.append(profile.astype(np.float32))
            case_center_survival.append(float(solved["final_survival"][center_x_crop, center_y_crop, z_idx_crop]))
            case_valley_eud_shift.append(
                equivalent_uniform_dose_from_survival(
                    solved["final_survival"][cropped_valley_mask],
                    alpha=alpha,
                    beta=beta,
                )
                - equivalent_uniform_dose_from_survival(
                    solved["lq_survival"][cropped_valley_mask],
                    alpha=alpha,
                    beta=beta,
                )
            )

        sensitivity_rows.append(
            {
                "parameter": parameter,
                "fractional_perturbation": frac,
                "baseline_center_survival": baseline_center_survival,
                "low_center_survival": case_center_survival[0],
                "high_center_survival": case_center_survival[1],
                "baseline_valley_eud_shift_gy": baseline_valley_eud_shift,
                "low_valley_eud_shift_gy": case_valley_eud_shift[0],
                "high_valley_eud_shift_gy": case_valley_eud_shift[1],
                "max_abs_delta_center_survival": max(
                    abs(case_center_survival[0] - baseline_center_survival),
                    abs(case_center_survival[1] - baseline_center_survival),
                ),
                "max_abs_delta_valley_eud_shift_gy": max(
                    abs(case_valley_eud_shift[0] - baseline_valley_eud_shift),
                    abs(case_valley_eud_shift[1] - baseline_valley_eud_shift),
                ),
            }
        )

    profile_stack_array = np.asarray(profile_stack, dtype=np.float32)
    lower_profile = np.min(profile_stack_array, axis=0)
    upper_profile = np.max(profile_stack_array, axis=0)
    x_profile_cm = x_cm_crop[profile_mask]

    figure7 = Path(outdir) / "figure7_phase10_uncertainty_band.png"
    figure8 = Path(outdir) / "figure8_phase10_parameter_sensitivity.png"
    uncertainty_csv = Path(outdir) / "phase10_uncertainty_band.csv"
    sensitivity_csv = Path(outdir) / "phase10_sensitivity_metrics.csv"
    summary_json = Path(outdir) / "phase10_extended_summary.json"

    plot_uncertainty_band(
        x_cm=x_profile_cm,
        baseline_profile=baseline_profile,
        lower_profile=lower_profile,
        upper_profile=upper_profile,
        out_file=figure7,
        dpi=int(args.dpi),
    )
    plot_sensitivity_ranking(
        sensitivity_rows,
        out_file=figure8,
        dpi=int(args.dpi),
    )

    write_csv(
        [
            {
                "x_cm": float(x_value),
                "baseline_survival": float(base),
                "lower_survival": float(low),
                "upper_survival": float(high),
            }
            for x_value, base, low, high in zip(x_profile_cm, baseline_profile, lower_profile, upper_profile)
        ],
        uncertainty_csv,
    )
    write_csv(sensitivity_rows, sensitivity_csv)

    sensitivity_ranking = sorted(
        sensitivity_rows,
        key=lambda row: float(row["max_abs_delta_center_survival"]),
        reverse=True,
    )
    summary_payload = {
        "phase": "Phase 10",
        "description": "Extended Phase 10 outputs for the complex synthetic lattice surrogate.",
        "baseline_summary": str(args.phase10_summary),
        "target_depth_cm": float(z_cm[target_depth_idx]),
        "valley_threshold_gy": float(valley_threshold_gy),
        "valley_voxel_count_3d": int(np.count_nonzero(valley_mask_3d)),
        "valley_voxel_count_xy": int(np.count_nonzero(valley_mask_xy)),
        "valley_eud_lq_gy": float(valley_eud_lq),
        "valley_eud_phase10_gy": float(valley_eud_phase10),
        "valley_eud_shift_gy": float(valley_eud_shift),
        "local_sensitivity": {
            "crop_half_width_x_mm": float(args.crop_half_width_x_mm),
            "crop_half_width_y_mm": float(args.crop_half_width_y_mm),
            "profile_half_width_mm": float(args.profile_half_width_mm),
            "fractional_perturbation": float(frac),
            "ranking": sensitivity_ranking,
        },
        "outputs": {
            "figure_valley_survival_map": str(fig5),
            "figure_nonlocal_eud_shift": str(fig6),
            "figure_uncertainty_band": str(figure7),
            "figure_sensitivity_ranking": str(figure8),
            "uncertainty_csv": str(uncertainty_csv),
            "sensitivity_csv": str(sensitivity_csv),
        },
    }
    summary_json.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")

    print(f"Phase 10 valley survival map: {fig5}")
    print(f"Phase 10 nonlocal EUD shift: {fig6}")
    print(f"Phase 10 uncertainty band: {figure7}")
    print(f"Phase 10 sensitivity ranking: {figure8}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
