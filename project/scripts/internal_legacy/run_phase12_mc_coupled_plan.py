#!/usr/bin/env python
"""Phase 12: couple a Monte Carlo stochasticity field into the biological model."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np

from analyze_250mev_sfrt_plan import centered_axis_cm, depth_axis_cm, nearest_index
from analyze_topas_outputs import load_topas_report_grids
from bystander_multispecies_pde_solver import (
    build_vessel_network_uptake_tensor,
    calculate_effective_dose,
    calculate_phase7_survival,
    calculate_systemic_immune_penalty,
    solve_multispecies_pde_3d_with_hazard,
)
from geometry_generators import (
    build_weighted_spot_lattice,
    default_complex_lattice_spot_specs_mm,
    default_complex_vessel_specs_mm,
    normalize_single_beam,
)
from run_linac_6mv_polyenergetic_clinical_sfrt import (
    load_spectrum,
    render_case_file,
    run_topas_case,
)

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
            "Run a Phase 12 complex-plan branch in which TOPAS contributes both "
            "dose and a Monte Carlo stochasticity field that modulates emission."
        )
    )
    parser.add_argument(
        "--template",
        type=Path,
        default=root / "topas" / "linac_6mv_polyenergetic_direct_photon_mcstats_template.txt",
        help="TOPAS template with multi-report scorer output.",
    )
    parser.add_argument(
        "--spectrum-csv",
        type=Path,
        default=root / "data" / "linac_6mv_representative_spectrum.csv",
        help="Representative 6 MV spectrum as energy_mev,weight CSV.",
    )
    parser.add_argument(
        "--run-root",
        type=Path,
        default=root / "runs" / "linac_6mv_polyenergetic_clinical_sfrt_phase12_mc",
        help="Output root for the Phase 12 TOPAS case and coupled-plan analysis.",
    )
    parser.add_argument(
        "--topas-bin",
        type=str,
        default="/Users/kw/shellScripts/topas",
        help="TOPAS executable.",
    )
    parser.add_argument(
        "--g4-data-dir",
        type=str,
        default="/Applications/GEANT4",
        help="Directory containing Geant4 data folders.",
    )
    parser.add_argument(
        "--physics-profile",
        choices=["em_opt4_only"],
        default="em_opt4_only",
        help="Named TOPAS modular physics profile.",
    )
    parser.add_argument("--histories", type=int, default=1_000_000, help="TOPAS histories.")
    parser.add_argument("--threads", type=int, default=8, help="TOPAS threads.")
    parser.add_argument("--seed", type=int, default=33, help="TOPAS RNG seed.")
    parser.add_argument(
        "--beam-position-cutoff-mm",
        type=float,
        default=5.0,
        help="Flat elliptical cutoff radius in mm, giving a 10 mm diameter field.",
    )
    parser.add_argument("--source-z-cm", type=float, default=-0.1, help="Source plane z position.")
    parser.add_argument("--phantom-size-x-cm", type=float, default=12.0)
    parser.add_argument("--phantom-size-y-cm", type=float, default=12.0)
    parser.add_argument("--phantom-size-z-cm", type=float, default=20.0)
    parser.add_argument("--xbins", type=int, default=121)
    parser.add_argument("--ybins", type=int, default=121)
    parser.add_argument("--zbins", type=int, default=161)
    parser.add_argument("--max-step-phantom-mm", type=float, default=0.1)
    parser.add_argument("--cut-gamma-mm", type=float, default=0.01)
    parser.add_argument("--cut-electron-mm", type=float, default=0.01)
    parser.add_argument("--cut-positron-mm", type=float, default=0.01)
    parser.add_argument("--prescribed-peak-dose-gy", type=float, default=10.0)
    parser.add_argument("--uniform-dose-floor-fraction", type=float, default=0.015)
    parser.add_argument("--alpha", type=float, default=0.03)
    parser.add_argument("--beta", type=float, default=0.003)
    parser.add_argument("--pde-steps", type=int, default=400)
    parser.add_argument("--pde-dt", type=float, default=0.12)
    parser.add_argument("--target-depth-cm", type=float, default=5.03)
    parser.add_argument(
        "--sigma-coupling-strength",
        type=float,
        default=0.35,
        help="Strength of the Monte Carlo stochasticity coupling into emission.",
    )
    parser.add_argument(
        "--sigma-coupling-mode",
        choices=["normalized_sigma", "normalized_cv"],
        default="normalized_sigma",
        help="How the extra MC field is constructed from scorer stochasticity.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip the TOPAS run when the multi-report CSV already exists.",
    )
    parser.add_argument(
        "--analyze-only",
        action="store_true",
        help="Skip TOPAS execution and analyze an existing multi-report CSV.",
    )
    parser.add_argument("--dpi", type=int, default=250, help="Figure DPI.")
    return parser.parse_args()


def _slice_extent_xy(x_cm: np.ndarray, y_cm: np.ndarray) -> list[float]:
    return [float(x_cm[0]), float(x_cm[-1]), float(y_cm[0]), float(y_cm[-1])]


def _slice_extent_xz(x_cm: np.ndarray, z_cm: np.ndarray) -> list[float]:
    return [float(x_cm[0]), float(x_cm[-1]), float(z_cm[0]), float(z_cm[-1])]


def write_csv(rows: List[Dict[str, object]], out_file: Path) -> None:
    if not rows:
        raise ValueError("No rows supplied for CSV output.")
    with out_file.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def build_sigma_kernel(
    report_grids: Dict[str, np.ndarray],
    *,
    z_cm: np.ndarray,
    prescribed_peak_dose_gy: float,
    histories: int,
    coupling_mode: str,
) -> tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    dose_sum = np.asarray(report_grids["Sum"], dtype=np.float32)
    dose_norm, normalization = normalize_single_beam(
        dose_sum,
        z_cm,
        prescribed_peak_dose_gy=float(prescribed_peak_dose_gy),
    )
    scale = float(normalization["scale_factor"])
    sigma_raw = np.asarray(report_grids["Standard_Deviation"], dtype=np.float32)
    sigma_mean = sigma_raw * np.float32(scale / math.sqrt(float(histories)))

    if coupling_mode == "normalized_sigma":
        base_field = sigma_mean
    else:
        base_field = sigma_mean / np.maximum(dose_norm, np.float32(1.0e-6))

    ref = float(np.percentile(base_field[base_field > 0.0], 99.0)) if np.any(base_field > 0.0) else 1.0
    sigma_norm = np.clip(base_field / max(ref, 1.0e-6), 0.0, 1.0).astype(np.float32, copy=False)
    return dose_norm.astype(np.float32, copy=False), sigma_norm, {
        **normalization,
        "sigma_reference_p99": float(ref),
        "sigma_mode": str(coupling_mode),
    }


def build_complex_plan_with_sigma(
    dose_kernel: np.ndarray,
    sigma_kernel: np.ndarray,
    *,
    voxel_size_mm: Sequence[float],
    uniform_dose_floor_fraction: float,
    prescribed_peak_dose_gy: float,
    spot_specs_mm: Sequence[dict] | None = None,
    vessel_specs_mm: Sequence[dict] | None = None,
) -> Dict[str, object]:
    chosen_spots = list(spot_specs_mm or default_complex_lattice_spot_specs_mm())
    chosen_vessels = list(vessel_specs_mm or default_complex_vessel_specs_mm())

    dose_lattice, placed_spots = build_weighted_spot_lattice(
        dose_kernel,
        spot_specs_mm=chosen_spots,
        voxel_size_mm=voxel_size_mm,
        margin_bins=6,
    )
    sigma_variance_kernel = sigma_kernel**2
    variance_spot_specs = [{**spec, "weight": float(spec.get("weight", 1.0)) ** 2} for spec in chosen_spots]
    sigma_variance_lattice, _ = build_weighted_spot_lattice(
        sigma_variance_kernel,
        spot_specs_mm=variance_spot_specs,
        voxel_size_mm=voxel_size_mm,
        margin_bins=6,
    )
    sigma_lattice = np.sqrt(np.maximum(sigma_variance_lattice, 0.0)).astype(np.float32, copy=False)

    dose_lattice = dose_lattice.astype(np.float32, copy=False)
    floor_gy = float(uniform_dose_floor_fraction) * float(prescribed_peak_dose_gy)
    if floor_gy > 0.0:
        dose_lattice = dose_lattice + np.float32(floor_gy)

    uptake_tensor, vessel_mask = build_vessel_network_uptake_tensor(
        dose_lattice.shape,
        voxel_size_mm,
        chosen_vessels,
        num_species=2,
        dtype=np.float32,
    )
    oxygen_modifier = np.ones((2, *dose_lattice.shape), dtype=np.float32)
    type_modifier = np.ones((2, *dose_lattice.shape), dtype=np.float32)
    dx_mm, dy_mm, dz_mm = (float(v) for v in voxel_size_mm)
    return {
        "dose_grid": dose_lattice,
        "sigma_grid": sigma_lattice.astype(np.float32, copy=False),
        "uptake_tensor": uptake_tensor.astype(np.float32, copy=False),
        "m_oxygen": oxygen_modifier,
        "m_type": type_modifier,
        "vessel_mask": vessel_mask,
        "meta": {
            "voxel_size_mm": [dx_mm, dy_mm, dz_mm],
            "dose_grid_shape": [int(v) for v in dose_lattice.shape],
            "x_cm": centered_axis_cm(dose_lattice.shape[0], dx_mm / 10.0).tolist(),
            "z_cm": depth_axis_cm(dose_lattice.shape[2], dz_mm / 10.0).tolist(),
            "uniform_dose_floor_gy": float(floor_gy),
            "spot_specs_mm": chosen_spots,
            "placed_spots": placed_spots,
            "vessel_specs_mm": chosen_vessels,
            "vessel_voxel_count": int(np.count_nonzero(vessel_mask)),
        },
    }


def build_emission_tensor(
    dose_grid: np.ndarray,
    sigma_norm_grid: np.ndarray,
    *,
    gamma: float,
    emax_ros: float,
    emax_cyto: float,
    sigma_coupling_strength: float,
    m_oxygen: np.ndarray,
    m_type: np.ndarray,
) -> np.ndarray:
    base = (1.0 - np.exp(-float(gamma) * dose_grid)).astype(np.float32)
    sigma_multiplier = (1.0 + float(sigma_coupling_strength) * sigma_norm_grid).astype(np.float32)
    return (
        np.asarray([float(emax_ros), float(emax_cyto)], dtype=np.float32)[:, None, None, None]
        * base[None, :, :, :]
        * sigma_multiplier[None, :, :, :]
        * m_type
        * m_oxygen
    ).astype(np.float32, copy=False)


def plot_field_pair(
    *,
    volume: np.ndarray,
    x_cm: np.ndarray,
    y_cm: np.ndarray,
    z_cm: np.ndarray,
    target_depth_idx: int,
    title: str,
    cmap: str,
    cbar_label: str,
    vmin: float,
    vmax: float,
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
        cmap=cmap,
        vmin=float(vmin),
        vmax=float(vmax),
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
    axes[1].axhline(float(z_cm[target_depth_idx]), color="#ffffff", linestyle="--", linewidth=1.2)
    axes[1].set_title("x-z at center y")
    axes[1].set_xlabel("x (cm)")
    axes[1].set_ylabel("z (cm)")

    cbar = fig.colorbar(img1, ax=axes.ravel().tolist(), shrink=0.9)
    cbar.set_label(cbar_label)
    fig.suptitle(title, fontsize=15)
    fig.savefig(out_file, dpi=dpi)
    plt.close(fig)


def plot_centerline_comparison(
    *,
    x_cm: np.ndarray,
    z_value_cm: float,
    baseline_survival: np.ndarray,
    sigma_survival: np.ndarray,
    baseline_deff: np.ndarray,
    sigma_deff: np.ndarray,
    out_file: Path,
    dpi: int,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2), constrained_layout=True)
    axes[0].plot(x_cm, baseline_survival, color="#1f77b4", linewidth=2.0, label="Dose-only survival")
    axes[0].plot(x_cm, sigma_survival, color="#d62728", linewidth=2.0, label="MC-coupled survival")
    axes[0].set_xlabel("x (cm)")
    axes[0].set_ylabel("Survival fraction")
    axes[0].set_title(f"Centerline survival at z = {z_value_cm:.2f} cm")
    axes[0].grid(True, linestyle="--", alpha=0.45)
    axes[0].legend(loc="best")

    axes[1].plot(x_cm, baseline_deff, color="#1f77b4", linewidth=2.0, label="Dose-only Deff")
    axes[1].plot(x_cm, sigma_deff, color="#d62728", linewidth=2.0, label="MC-coupled Deff")
    axes[1].set_xlabel("x (cm)")
    axes[1].set_ylabel("Effective dose (Gy)")
    axes[1].set_title(f"Centerline Deff at z = {z_value_cm:.2f} cm")
    axes[1].grid(True, linestyle="--", alpha=0.45)
    axes[1].legend(loc="best")

    fig.suptitle("Figure 4: Phase 12 Monte Carlo coupling impact on centerline response", fontsize=15)
    fig.savefig(out_file, dpi=dpi)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    case_dir = args.run_root / "case"
    analysis_dir = args.run_root / "analysis_phase12_mc_coupled_plan"
    case_dir.mkdir(parents=True, exist_ok=True)
    analysis_dir.mkdir(parents=True, exist_ok=True)

    spectrum_energies, spectrum_weights = load_spectrum(args.spectrum_csv)
    parameter_file = case_dir / "beamline.txt"
    reports_csv = case_dir / "mcstats.csv"
    topas_log = case_dir / "topas.log"

    rendered_case = render_case_file(args, spectrum_energies, spectrum_weights).replace(
        's:Sc/Dose3D/OutputFile                = "dosedata"',
        's:Sc/Dose3D/OutputFile                = "mcstats"',
    )
    parameter_file.write_text(rendered_case, encoding="utf-8")

    if not args.analyze_only:
        if not (args.skip_existing and reports_csv.exists() and reports_csv.stat().st_size > 0):
            run_topas_case(args, case_dir, parameter_file, reports_csv, topas_log)

    report_grids, header = load_topas_report_grids(reports_csv, retries=5, retry_delay_sec=0.5)
    if "Sum" not in report_grids or "Standard_Deviation" not in report_grids:
        raise ValueError(
            f"Expected TOPAS reports 'Sum' and 'Standard_Deviation' in {reports_csv}, "
            f"found {sorted(report_grids)}"
        )

    z_kernel_cm = depth_axis_cm(report_grids["Sum"].shape[2], float(header["dz_cm"]))
    dose_kernel, sigma_kernel_norm, kernel_meta = build_sigma_kernel(
        report_grids,
        z_cm=z_kernel_cm,
        prescribed_peak_dose_gy=float(args.prescribed_peak_dose_gy),
        histories=int(args.histories),
        coupling_mode=str(args.sigma_coupling_mode),
    )

    plan = build_complex_plan_with_sigma(
        dose_kernel,
        sigma_kernel_norm,
        voxel_size_mm=(
            float(header["dx_cm"]) * 10.0,
            float(header["dy_cm"]) * 10.0,
            float(header["dz_cm"]) * 10.0,
        ),
        uniform_dose_floor_fraction=float(args.uniform_dose_floor_fraction),
        prescribed_peak_dose_gy=float(args.prescribed_peak_dose_gy),
    )
    dose_grid = np.asarray(plan["dose_grid"], dtype=np.float32)
    sigma_norm_grid = np.asarray(plan["sigma_grid"], dtype=np.float32)
    uptake_tensor = np.asarray(plan["uptake_tensor"], dtype=np.float32)
    m_oxygen = np.asarray(plan["m_oxygen"], dtype=np.float32)
    m_type = np.asarray(plan["m_type"], dtype=np.float32)
    x_cm = np.asarray(plan["meta"]["x_cm"], dtype=np.float32)
    y_cm = centered_axis_cm(dose_grid.shape[1], float(plan["meta"]["voxel_size_mm"][1]) / 10.0)
    z_cm = np.asarray(plan["meta"]["z_cm"], dtype=np.float32)
    target_depth_idx = nearest_index(z_cm, float(args.target_depth_cm))
    center_x = dose_grid.shape[0] // 2
    center_y = dose_grid.shape[1] // 2
    voxel_size_mm = tuple(float(v) for v in plan["meta"]["voxel_size_mm"])

    lq_survival = np.exp(-float(args.alpha) * dose_grid - float(args.beta) * dose_grid**2).astype(np.float32)
    baseline_emission_tensor = build_emission_tensor(
        dose_grid,
        np.zeros_like(sigma_norm_grid, dtype=np.float32),
        gamma=float(LOCKED_GAMMA),
        emax_ros=1.5,
        emax_cyto=0.8,
        sigma_coupling_strength=0.0,
        m_oxygen=m_oxygen,
        m_type=m_type,
    )
    sigma_emission_tensor = build_emission_tensor(
        dose_grid,
        sigma_norm_grid,
        gamma=float(LOCKED_GAMMA),
        emax_ros=1.5,
        emax_cyto=0.8,
        sigma_coupling_strength=float(args.sigma_coupling_strength),
        m_oxygen=m_oxygen,
        m_type=m_type,
    )

    _, baseline_hazard = solve_multispecies_pde_3d_with_hazard(
        dose_grid,
        voxel_size_mm,
        diffusion_coeffs=(0.8, LOCKED_D_CYTO),
        decay_coeffs=(0.2, LOCKED_LAMBDA_CYTO),
        emission_tensor=baseline_emission_tensor,
        uptake_tensor=uptake_tensor,
        hazard_weights=(W_ROS, W_CYTO),
        steps=int(args.pde_steps),
        dt=float(args.pde_dt),
        progress_interval=50,
        verbose=True,
    )
    _, sigma_hazard = solve_multispecies_pde_3d_with_hazard(
        dose_grid,
        voxel_size_mm,
        diffusion_coeffs=(0.8, LOCKED_D_CYTO),
        decay_coeffs=(0.2, LOCKED_LAMBDA_CYTO),
        emission_tensor=sigma_emission_tensor,
        uptake_tensor=uptake_tensor,
        hazard_weights=(W_ROS, W_CYTO),
        steps=int(args.pde_steps),
        dt=float(args.pde_dt),
        progress_interval=50,
        verbose=True,
    )

    baseline_survival = calculate_phase7_survival(
        lq_survival,
        baseline_hazard,
        dose_grid,
        voxel_size_mm,
        float(LOCKED_SCALING_FACTOR),
        weight_immune=float(W_IMMUNE),
        verbose=False,
    )
    sigma_survival = calculate_phase7_survival(
        lq_survival,
        sigma_hazard,
        dose_grid,
        voxel_size_mm,
        float(LOCKED_SCALING_FACTOR),
        weight_immune=float(W_IMMUNE),
        verbose=False,
    )
    baseline_deff = calculate_effective_dose(baseline_survival, alpha=float(args.alpha), beta=float(args.beta))
    sigma_deff = calculate_effective_dose(sigma_survival, alpha=float(args.alpha), beta=float(args.beta))
    immune_penalty, icd_volume_cm3 = calculate_systemic_immune_penalty(dose_grid, voxel_size_mm)

    delta_survival = (sigma_survival - baseline_survival).astype(np.float32)
    delta_deff = (sigma_deff - baseline_deff).astype(np.float32)

    center_metrics = {
        "sampled_depth_cm": float(z_cm[target_depth_idx]),
        "center_dose_gy": float(dose_grid[center_x, center_y, target_depth_idx]),
        "center_sigma_norm": float(sigma_norm_grid[center_x, center_y, target_depth_idx]),
        "center_survival_baseline": float(baseline_survival[center_x, center_y, target_depth_idx]),
        "center_survival_sigma_coupled": float(sigma_survival[center_x, center_y, target_depth_idx]),
        "center_survival_delta": float(delta_survival[center_x, center_y, target_depth_idx]),
        "center_deff_baseline_gy": float(baseline_deff[center_x, center_y, target_depth_idx]),
        "center_deff_sigma_coupled_gy": float(sigma_deff[center_x, center_y, target_depth_idx]),
        "center_deff_delta_gy": float(delta_deff[center_x, center_y, target_depth_idx]),
    }

    fig1 = analysis_dir / "figure1_phase12_mc_sigma_field.png"
    fig2 = analysis_dir / "figure2_phase12_sigma_coupled_survival.png"
    fig3 = analysis_dir / "figure3_phase12_sigma_coupled_deff_delta.png"
    fig4 = analysis_dir / "figure4_phase12_centerline_comparison.png"
    summary_json = analysis_dir / "phase12_mc_coupled_summary.json"
    centerline_csv = analysis_dir / "phase12_mc_coupled_centerline.csv"
    volumes_npz = analysis_dir / "phase12_mc_coupled_volumes.npz"

    plot_field_pair(
        volume=sigma_norm_grid,
        x_cm=x_cm,
        y_cm=y_cm,
        z_cm=z_cm,
        target_depth_idx=int(target_depth_idx),
        title="Figure 1: Phase 12 Monte Carlo stochasticity field",
        cmap="plasma",
        cbar_label="Normalized sigma_D proxy",
        vmin=0.0,
        vmax=1.0,
        out_file=fig1,
        dpi=int(args.dpi),
    )
    plot_field_pair(
        volume=sigma_survival,
        x_cm=x_cm,
        y_cm=y_cm,
        z_cm=z_cm,
        target_depth_idx=int(target_depth_idx),
        title="Figure 2: Phase 12 MC-coupled survival",
        cmap="viridis",
        cbar_label="Survival fraction",
        vmin=0.0,
        vmax=1.0,
        out_file=fig2,
        dpi=int(args.dpi),
    )
    plot_field_pair(
        volume=delta_deff,
        x_cm=x_cm,
        y_cm=y_cm,
        z_cm=z_cm,
        target_depth_idx=int(target_depth_idx),
        title="Figure 3: Phase 12 Deff lift from Monte Carlo coupling",
        cmap="magma",
        cbar_label="Delta Deff (Gy)",
        vmin=0.0,
        vmax=float(np.percentile(delta_deff, 99.5)),
        out_file=fig3,
        dpi=int(args.dpi),
    )
    plot_centerline_comparison(
        x_cm=x_cm,
        z_value_cm=float(z_cm[target_depth_idx]),
        baseline_survival=baseline_survival[:, center_y, target_depth_idx],
        sigma_survival=sigma_survival[:, center_y, target_depth_idx],
        baseline_deff=baseline_deff[:, center_y, target_depth_idx],
        sigma_deff=sigma_deff[:, center_y, target_depth_idx],
        out_file=fig4,
        dpi=int(args.dpi),
    )

    rows = []
    for ix in range(dose_grid.shape[0]):
        rows.append(
            {
                "x_cm": float(x_cm[ix]),
                "dose_gy": float(dose_grid[ix, center_y, target_depth_idx]),
                "sigma_norm": float(sigma_norm_grid[ix, center_y, target_depth_idx]),
                "survival_baseline": float(baseline_survival[ix, center_y, target_depth_idx]),
                "survival_sigma_coupled": float(sigma_survival[ix, center_y, target_depth_idx]),
                "survival_delta": float(delta_survival[ix, center_y, target_depth_idx]),
                "deff_baseline_gy": float(baseline_deff[ix, center_y, target_depth_idx]),
                "deff_sigma_coupled_gy": float(sigma_deff[ix, center_y, target_depth_idx]),
                "deff_delta_gy": float(delta_deff[ix, center_y, target_depth_idx]),
            }
        )
    write_csv(rows, centerline_csv)

    np.savez_compressed(
        volumes_npz,
        dose_grid=dose_grid.astype(np.float32, copy=False),
        sigma_norm_grid=sigma_norm_grid.astype(np.float32, copy=False),
        baseline_hazard=baseline_hazard.astype(np.float32, copy=False),
        sigma_hazard=sigma_hazard.astype(np.float32, copy=False),
        baseline_survival=baseline_survival.astype(np.float32, copy=False),
        sigma_survival=sigma_survival.astype(np.float32, copy=False),
        baseline_deff=baseline_deff.astype(np.float32, copy=False),
        sigma_deff=sigma_deff.astype(np.float32, copy=False),
        delta_deff=delta_deff.astype(np.float32, copy=False),
        x_cm=x_cm.astype(np.float32, copy=False),
        y_cm=y_cm.astype(np.float32, copy=False),
        z_cm=z_cm.astype(np.float32, copy=False),
    )

    summary = {
        "phase": "Phase 12",
        "description": "TOPAS Monte Carlo stochasticity field coupled into the emission tensor of the complex-plan biology model.",
        "mc_field": {
            "report_names": list(report_grids.keys()),
            "sigma_coupling_mode": str(args.sigma_coupling_mode),
            "sigma_coupling_strength": float(args.sigma_coupling_strength),
            "kernel_normalization": kernel_meta,
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
        "topas_case": {
            "parameter_file": str(parameter_file),
            "reports_csv": str(reports_csv),
            "topas_log": str(topas_log),
            "histories": int(args.histories),
            "threads": int(args.threads),
            "seed": int(args.seed),
        },
        "center_metrics": center_metrics,
        "systemic_immune_penalty": float(immune_penalty),
        "icd_volume_cm3": float(icd_volume_cm3),
        "outputs": {
            "figure_sigma_field": str(fig1),
            "figure_survival": str(fig2),
            "figure_deff_delta": str(fig3),
            "figure_centerline": str(fig4),
            "centerline_csv": str(centerline_csv),
            "volumes_npz": str(volumes_npz),
            "summary_json": str(summary_json),
        },
    }
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Phase 12 summary: {summary_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
