#!/usr/bin/env python
"""Run a Phase 6 Morris sensitivity analysis for the 40 mm Phase 5 branch.

This script evaluates the corrected true-valley / right-peak-hypoxia geometry
under biological parameter uncertainty and produces:

- a tornado-style Morris ranking based on valley-center survival at 5 cm
- a 1D lateral credible-band profile across x in the central valley
- CSV/JSON artifacts for the sampled parameters and sensitivity outputs

It uses SALib when available and falls back to a built-in Morris-compatible
sampler/analyzer when SALib is not installed.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import multiprocessing as mp
import os
import time
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

from analyze_250mev_sfrt_plan import centered_axis_cm, choose_reference_depth, depth_axis_cm, nearest_index, reference_peak_value
from analyze_topas_outputs import load_topas_grid
from bystander_multispecies_pde_solver import (
    build_cylindrical_uptake_tensor,
    calculate_phase5_multi_effect_survival,
    solve_multispecies_pde_3d,
)
from generate_phase2_multispecies_vessel_valley_figures import build_lattice_with_x_shift

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception as exc:  # pragma: no cover
    raise RuntimeError("matplotlib is required for figure generation") from exc

try:  # pragma: no cover - optional dependency
    from SALib.analyze import morris as morris_analyzer
    from SALib.sample import morris as morris_sampler

    HAVE_SALIB = True
except Exception:  # pragma: no cover
    HAVE_SALIB = False
    morris_analyzer = None
    morris_sampler = None


PROBLEM = {
    "num_vars": 6,
    "names": ["D_cyto", "lambda_cyto", "E_max_cyto", "gamma", "u_cyto", "w_immune"],
    "bounds": [
        [0.2, 0.6],
        [0.01, 0.05],
        [0.5, 1.2],
        [0.2, 0.5],
        [0.3, 0.9],
        [0.1, 0.4],
    ],
}

_WORKER_CONTEXT: Dict[str, object] | None = None


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description=(
            "Run a Phase 6 Morris sensitivity analysis for the 40 mm Phase 5 "
            "vessel-in-valley branch and generate tornado / credible-band outputs."
        )
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=root / "runs" / "linac_6mv_polyenergetic_clinical_sfrt" / "case" / "dosedata.csv",
        help="TOPAS single-beam dose CSV used to rebuild the lattice kernel.",
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=root
        / "runs"
        / "linac_6mv_polyenergetic_clinical_sfrt"
        / "analysis_phase2_multispecies_true_valley_phase5_rightpeak_depth50"
        / "phase2_multispecies_summary.json",
        help="Phase 5 summary JSON supplying geometry and baseline biology settings.",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=root
        / "runs"
        / "linac_6mv_polyenergetic_clinical_sfrt"
        / "analysis_phase6_sensitivity",
        help="Directory for Phase 6 outputs.",
    )
    parser.add_argument("--pitch-mm", type=float, default=40.0, help="Pitch analyzed for Phase 6.")
    parser.add_argument(
        "--target-depth-cm",
        type=float,
        default=5.0,
        help="Target depth used for the 1D slice and center-survival sensitivity metric.",
    )
    parser.add_argument(
        "--profile-range-mm",
        type=float,
        default=30.0,
        help="Half-width of the exported 1D lateral slice around x = 0.",
    )
    parser.add_argument(
        "--crop-half-width-x-mm",
        type=float,
        default=60.0,
        help="Half-width of the cropped ROI along x used to accelerate the PDE solves.",
    )
    parser.add_argument(
        "--crop-half-width-y-mm",
        type=float,
        default=40.0,
        help="Half-width of the cropped ROI along y used to accelerate the PDE solves.",
    )
    parser.add_argument(
        "--prescribed-peak-dose-gy",
        type=float,
        default=10.0,
        help="Dose assigned to the reference peak voxel after normalization.",
    )
    parser.add_argument("--alpha", type=float, default=0.10, help="LQ alpha in Gy^-1.")
    parser.add_argument("--beta", type=float, default=0.05, help="LQ beta in Gy^-2.")
    parser.add_argument(
        "--trajectories",
        type=int,
        default=20,
        help="Number of Morris trajectories. Total solves = trajectories * (num_vars + 1).",
    )
    parser.add_argument("--num-levels", type=int, default=4, help="Number of Morris grid levels.")
    parser.add_argument("--seed", type=int, default=33, help="Random seed for the Morris design.")
    parser.add_argument(
        "--max-workers",
        type=int,
        default=4,
        help="Maximum parallel worker processes. Use 0 to auto-select up to 6 workers.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=10,
        help="Print progress every N completed evaluations.",
    )
    parser.add_argument("--dpi", type=int, default=300, help="Figure DPI.")
    return parser.parse_args()


def normalize_single_beam(
    single_beam: np.ndarray,
    z_cm: np.ndarray,
    prescribed_peak_dose_gy: float,
) -> tuple[np.ndarray, Dict[str, float]]:
    ref_idx = choose_reference_depth(
        single_beam,
        z_cm,
        argparse.Namespace(reference_mode="dmax", reference_depth_cm=0.0),
    )
    ref_peak_raw, ref_peak_idx_xy = reference_peak_value(single_beam, ref_idx, center_window_bins=5)
    scale = float(prescribed_peak_dose_gy) / float(ref_peak_raw)
    return single_beam * scale, {
        "reference_depth_cm": float(z_cm[ref_idx]),
        "reference_peak_raw_gy": float(ref_peak_raw),
        "scale_factor": float(scale),
        "reference_peak_index_x": int(ref_peak_idx_xy[0]),
        "reference_peak_index_y": int(ref_peak_idx_xy[1]),
    }


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


def sample_morris_fallback(
    problem: Dict[str, object],
    *,
    trajectories: int,
    num_levels: int,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(int(seed))
    num_vars = int(problem["num_vars"])
    bounds = np.asarray(problem["bounds"], dtype=float)
    delta_unit = float(num_levels) / (2.0 * float(num_levels - 1))
    levels = np.linspace(0.0, 1.0, int(num_levels), dtype=float)
    samples: List[np.ndarray] = []

    for _ in range(int(trajectories)):
        x = rng.choice(levels, size=num_vars, replace=True).astype(float)
        trajectory = [x.copy()]
        for dim_idx in rng.permutation(num_vars):
            if x[dim_idx] + delta_unit <= 1.0 + 1e-9:
                direction = 1.0
            elif x[dim_idx] - delta_unit >= -1e-9:
                direction = -1.0
            else:
                replacement_choices = levels[levels <= (1.0 - delta_unit + 1e-9)]
                x[dim_idx] = float(rng.choice(replacement_choices))
                direction = 1.0
            x = x.copy()
            x[dim_idx] = float(np.clip(x[dim_idx] + (direction * delta_unit), 0.0, 1.0))
            trajectory.append(x.copy())
        samples.extend(trajectory)

    unit_samples = np.asarray(samples, dtype=float)
    lower = bounds[:, 0]
    upper = bounds[:, 1]
    return lower[None, :] + unit_samples * (upper - lower)[None, :]


def analyze_morris_fallback(
    problem: Dict[str, object],
    sample_matrix: np.ndarray,
    outputs: np.ndarray,
) -> Dict[str, np.ndarray]:
    num_vars = int(problem["num_vars"])
    names = list(problem["names"])
    effects: Dict[str, List[float]] = {name: [] for name in names}

    for start_idx in range(0, len(sample_matrix), num_vars + 1):
        block_x = sample_matrix[start_idx : start_idx + num_vars + 1]
        block_y = outputs[start_idx : start_idx + num_vars + 1]
        if block_x.shape[0] != num_vars + 1:
            continue
        for step_idx in range(num_vars):
            delta_x = block_x[step_idx + 1] - block_x[step_idx]
            changed = np.flatnonzero(np.abs(delta_x) > 1e-12)
            if changed.size != 1:
                continue
            dim_idx = int(changed[0])
            effect = float((block_y[step_idx + 1] - block_y[step_idx]) / delta_x[dim_idx])
            effects[names[dim_idx]].append(effect)

    mu = np.array([np.mean(effects[name]) for name in names], dtype=float)
    mu_star = np.array([np.mean(np.abs(effects[name])) for name in names], dtype=float)
    sigma = np.array([np.std(effects[name], ddof=1) if len(effects[name]) > 1 else 0.0 for name in names], dtype=float)
    return {"mu": mu, "mu_star": mu_star, "sigma": sigma}


def build_sensitivity_context(args: argparse.Namespace) -> Dict[str, object]:
    summary = json.loads(args.summary_json.read_text(encoding="utf-8"))
    single_beam, header = load_topas_grid(args.csv, retries=5, retry_delay_sec=0.5)
    dx_cm = float(header["dx_cm"])
    dy_cm = float(header["dy_cm"])
    dz_cm = float(header["dz_cm"])
    z_single_cm = depth_axis_cm(single_beam.shape[2], dz_cm)
    normalized_single, norm_meta = normalize_single_beam(
        single_beam,
        z_single_cm,
        prescribed_peak_dose_gy=float(args.prescribed_peak_dose_gy),
    )
    normalized_single = normalized_single.astype(np.float32)

    pitch_mm = float(args.pitch_mm)
    pitch_bins_x = int(round((pitch_mm / 10.0) / dx_cm))
    pitch_bins_y = int(round((pitch_mm / 10.0) / dy_cm))
    x_shift_fraction = float(summary["lattice_geometry"]["x_shift_fraction_of_pitch"])
    x_shift_bins = int(round(x_shift_fraction * float(pitch_bins_x)))
    lattice_dose, _, _, _ = build_lattice_with_x_shift(
        normalized_single,
        pitch_bins_x=pitch_bins_x,
        pitch_bins_y=pitch_bins_y,
        n_beams_x=int(summary["lattice_geometry"]["n_beams_x"]),
        n_beams_y=int(summary["lattice_geometry"]["n_beams_y"]),
        x_shift_bins=int(x_shift_bins),
    )
    lattice_dose = lattice_dose.astype(np.float32)
    uniform_floor_gy = float(summary["physical_floor_model"]["uniform_dose_floor_gy"])
    if uniform_floor_gy > 0.0:
        lattice_dose = lattice_dose + np.float32(uniform_floor_gy)

    crop_dose, _, _ = crop_center_xy(
        lattice_dose,
        dx_mm=float(dx_cm * 10.0),
        dy_mm=float(dy_cm * 10.0),
        half_width_x_mm=float(args.crop_half_width_x_mm),
        half_width_y_mm=float(args.crop_half_width_y_mm),
    )
    del lattice_dose

    z_cm = depth_axis_cm(crop_dose.shape[2], dz_cm)
    x_mm = centered_axis_cm(crop_dose.shape[0], dx_cm) * 10.0
    y_mm = centered_axis_cm(crop_dose.shape[1], dy_cm) * 10.0
    target_depth_idx = int(nearest_index(z_cm, float(args.target_depth_cm)))
    profile_mask = np.abs(x_mm) <= float(args.profile_range_mm)
    if not np.any(profile_mask):
        raise ValueError("Profile range did not select any x points.")

    lq_survival_grid = np.exp(
        -float(args.alpha) * crop_dose - float(args.beta) * crop_dose**2
    ).astype(np.float32)

    emission_model = summary["emission_model"]
    vessel_model = summary["vessel_model"]
    multispecies_model = summary["multispecies_model"]
    phase5_model = summary["phase5_model"]

    return {
        "dose_grid": crop_dose,
        "lq_survival_grid": lq_survival_grid,
        "voxel_size_mm": tuple(float(value) for value in multispecies_model["voxel_size_mm"]),
        "target_depth_idx": target_depth_idx,
        "sampled_depth_cm": float(z_cm[target_depth_idx]),
        "center_x_idx": int(crop_dose.shape[0] // 2),
        "center_y_idx": int(crop_dose.shape[1] // 2),
        "profile_x_mm": x_mm[profile_mask].astype(float),
        "profile_mask": profile_mask,
        "vessel_radius_mm": float(vessel_model["radius_mm"]),
        "vessel_center_offset_mm": tuple(float(value) for value in vessel_model["center_offset_mm"]),
        "ros_uptake": float(vessel_model["uptake_rates_in_vessel"][0]),
        "tumor_radius_mm": float(emission_model["tumor_radius_mm"]),
        "tumor_center_offset_mm": tuple(float(value) for value in emission_model["tumor_center_offset_mm"]),
        "tumor_cytokine_multiplier": float(emission_model["tumor_cytokine_multiplier"]),
        "hypoxic_radius_mm": float(emission_model["hypoxic_radius_mm"]),
        "hypoxic_center_offset_mm": tuple(float(value) for value in emission_model["hypoxic_center_offset_mm"]),
        "hypoxic_ros_scale": float(emission_model["hypoxic_ros_scale"]),
        "hypoxic_cytokine_multiplier": float(emission_model["hypoxic_cytokine_multiplier"]),
        "ros_diffusion": float(multispecies_model["diffusion_coeffs"][0]),
        "ros_decay": float(multispecies_model["decay_coeffs"][0]),
        "ros_emax": float(multispecies_model["emission_emax"][0]),
        "weight_ros_fixed": float(phase5_model["channel_weights"]["ros"]),
        "scaling_factor": float(summary["scaling_factor"]),
        "icd_threshold_gy": float(phase5_model["icd_threshold_gy"]),
        "immune_max_penalty": float(phase5_model["immune_max_penalty"]),
        "immune_half_volume_cm3": float(phase5_model["immune_half_volume_cm3"]),
        "steps": int(multispecies_model["steps"]),
        "dt": float(multispecies_model["dt"]),
        "normalization": norm_meta,
        "crop_half_width_x_mm": float(args.crop_half_width_x_mm),
        "crop_half_width_y_mm": float(args.crop_half_width_y_mm),
        "profile_range_mm": float(args.profile_range_mm),
        "x_shift_fraction": float(x_shift_fraction),
        "baseline_summary": summary,
    }


def _init_worker(context: Dict[str, object]) -> None:
    global _WORKER_CONTEXT
    _WORKER_CONTEXT = context


def wrapper_pde_evaluation(params: Sequence[float]) -> tuple[np.ndarray, float]:
    if _WORKER_CONTEXT is None:
        raise RuntimeError("Worker context was not initialized.")
    ctx = _WORKER_CONTEXT

    d_cyto, lambda_cyto, emax_cyto, gamma, u_cyto, w_immune = (float(value) for value in params)
    w_ros = float(ctx["weight_ros_fixed"])
    w_cyto = 1.0 - w_ros - w_immune
    if w_cyto < 0.0:
        raise ValueError("Sampled w_immune leaves a negative cytokine weight.")

    uptake_tensor, _ = build_cylindrical_uptake_tensor(
        ctx["dose_grid"].shape,
        ctx["voxel_size_mm"],
        num_species=2,
        vessel_radius_mm=float(ctx["vessel_radius_mm"]),
        vessel_center_offset_mm=tuple(ctx["vessel_center_offset_mm"]),
        uptake_rates_in_vessel=(float(ctx["ros_uptake"]), float(u_cyto)),
    )

    multispecies_tensor = solve_multispecies_pde_3d(
        dose_grid=ctx["dose_grid"],
        voxel_size_mm=ctx["voxel_size_mm"],
        steps=int(ctx["steps"]),
        dt=float(ctx["dt"]),
        diffusion_coeffs=(float(ctx["ros_diffusion"]), float(d_cyto)),
        decay_coeffs=(float(ctx["ros_decay"]), float(lambda_cyto)),
        emission_emax=(float(ctx["ros_emax"]), float(emax_cyto)),
        emission_gamma_per_gy=float(gamma),
        state_dependent_emission=True,
        tumor_radius_mm=float(ctx["tumor_radius_mm"]),
        tumor_center_offset_mm=tuple(ctx["tumor_center_offset_mm"]),
        tumor_cytokine_multiplier=float(ctx["tumor_cytokine_multiplier"]),
        hypoxic_radius_mm=float(ctx["hypoxic_radius_mm"]),
        hypoxic_center_offset_mm=tuple(ctx["hypoxic_center_offset_mm"]),
        hypoxic_ros_scale=float(ctx["hypoxic_ros_scale"]),
        hypoxic_cytokine_multiplier=float(ctx["hypoxic_cytokine_multiplier"]),
        uptake_tensor=uptake_tensor,
        progress_interval=0,
        verbose=False,
    )

    final_survival = calculate_phase5_multi_effect_survival(
        ctx["lq_survival_grid"],
        multispecies_tensor,
        ctx["dose_grid"],
        ctx["voxel_size_mm"],
        float(ctx["scaling_factor"]),
        channel_weights=(float(w_ros), float(w_cyto), float(w_immune)),
        icd_threshold_gy=float(ctx["icd_threshold_gy"]),
        immune_max_penalty=float(ctx["immune_max_penalty"]),
        immune_half_volume_cm3=float(ctx["immune_half_volume_cm3"]),
        verbose=False,
    )

    lateral_profile = final_survival[:, int(ctx["center_y_idx"]), int(ctx["target_depth_idx"])]
    profile_slice = np.asarray(lateral_profile[ctx["profile_mask"]], dtype=np.float32)
    center_survival = float(lateral_profile[int(ctx["center_x_idx"])])
    return profile_slice, center_survival


def plot_phase6_figure(
    problem: Dict[str, object],
    sensitivity: Dict[str, np.ndarray],
    x_mm: np.ndarray,
    median_slice: np.ndarray,
    lower_slice: np.ndarray,
    upper_slice: np.ndarray,
    *,
    out_file: Path,
    vessel_radius_mm: float,
    hypoxic_center_x_mm: float,
    hypoxic_radius_mm: float,
    sampled_depth_cm: float,
    ranking_source: str,
    dpi: int,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14.0, 6.0), constrained_layout=True)

    mu_star = np.asarray(sensitivity["mu_star"], dtype=float)
    sigma = np.asarray(sensitivity["sigma"], dtype=float)
    order = np.argsort(mu_star)
    names = np.asarray(problem["names"], dtype=object)[order]

    axes[0].barh(names, mu_star[order], color="#4c78a8")
    axes[0].set_xlabel("Influence on valley-center survival ($\\mu^*$)")
    axes[0].set_title("Morris Sensitivity Ranking")
    axes[0].grid(axis="x", linestyle="--", alpha=0.35)
    for row_idx, sigma_value in enumerate(sigma[order]):
        axes[0].text(
            float(mu_star[order][row_idx]) + 0.01 * max(mu_star.max(), 1e-6),
            row_idx,
            f"$\\sigma$={sigma_value:.3f}",
            va="center",
            fontsize=9,
        )

    axes[1].plot(x_mm, median_slice, color="#d62728", linewidth=2.4, label="Median Phase 5 survival")
    axes[1].fill_between(
        x_mm,
        lower_slice,
        upper_slice,
        color="#d62728",
        alpha=0.20,
        label="95% credible band",
    )
    axes[1].axvspan(-float(vessel_radius_mm), float(vessel_radius_mm), color="#76b7b2", alpha=0.18, label="Central vessel")
    axes[1].axvspan(
        float(hypoxic_center_x_mm) - float(hypoxic_radius_mm),
        float(hypoxic_center_x_mm) + float(hypoxic_radius_mm),
        color="#f39c12",
        alpha=0.12,
        label="Right-peak hypoxia",
    )
    axes[1].set_xlim(float(x_mm[0]), float(x_mm[-1]))
    axes[1].set_ylim(0.0, 1.0)
    axes[1].set_xlabel("Distance x (mm)")
    axes[1].set_ylabel("Cell survival")
    axes[1].set_title("1D Asymmetry Profile with Biological Uncertainty")
    axes[1].grid(alpha=0.25)
    axes[1].legend(loc="lower left")
    axes[1].text(
        0.98,
        0.02,
        f"Depth = {sampled_depth_cm:.3f} cm\nRanking = {ranking_source}",
        transform=axes[1].transAxes,
        fontsize=9,
        va="bottom",
        ha="right",
        bbox={"facecolor": "white", "alpha": 0.88, "edgecolor": "#b5b5b5"},
    )

    fig.suptitle("Figure 10: Phase 6 sensitivity analysis")
    fig.savefig(out_file, dpi=dpi)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    if HAVE_SALIB:
        param_values = morris_sampler.sample(
            PROBLEM,
            N=int(args.trajectories),
            num_levels=int(args.num_levels),
            optimal_trajectories=None,
            seed=int(args.seed),
        )
        ranking_source = "SALib Morris"
    else:
        print("SALib not installed; using built-in Morris-compatible sampler/analyzer.", flush=True)
        param_values = sample_morris_fallback(
            PROBLEM,
            trajectories=int(args.trajectories),
            num_levels=int(args.num_levels),
            seed=int(args.seed),
        )
        ranking_source = "fallback Morris"

    print(f"Generated {len(param_values)} parameter sets for Phase 6 sensitivity analysis.", flush=True)

    context = build_sensitivity_context(args)
    requested_workers = int(args.max_workers)
    if requested_workers <= 0:
        max_workers = max(1, min((os.cpu_count() or 2) - 1, 6))
    else:
        max_workers = max(1, int(requested_workers))
    print(
        f"Using {max_workers} worker(s) on ROI shape {context['dose_grid'].shape} "
        f"with profile window +/-{float(args.profile_range_mm):.1f} mm.",
        flush=True,
    )

    ctx = mp.get_context("spawn")
    start_time = time.time()
    results: List[tuple[np.ndarray, float]] = []
    with ctx.Pool(processes=max_workers, initializer=_init_worker, initargs=(context,)) as pool:
        for idx, result in enumerate(pool.imap(wrapper_pde_evaluation, param_values), start=1):
            results.append(result)
            if int(args.progress_every) > 0 and idx % int(args.progress_every) == 0:
                elapsed = time.time() - start_time
                print(f"  ... completed {idx}/{len(param_values)} evaluations in {elapsed:.1f} s", flush=True)

    runtime_sec = time.time() - start_time
    print(f"Completed {len(param_values)} PDE solves in {runtime_sec:.1f} seconds.", flush=True)

    slices_1d = np.stack([item[0] for item in results], axis=0)
    center_survivals = np.asarray([item[1] for item in results], dtype=float)

    if HAVE_SALIB:
        sensitivity = morris_analyzer.analyze(
            PROBLEM,
            param_values,
            center_survivals,
            print_to_console=False,
        )
    else:
        sensitivity = analyze_morris_fallback(PROBLEM, param_values, center_survivals)

    median_slice = np.percentile(slices_1d, 50.0, axis=0)
    lower_slice = np.percentile(slices_1d, 2.5, axis=0)
    upper_slice = np.percentile(slices_1d, 97.5, axis=0)

    figure_file = args.outdir / "figure10_phase6_sensitivity.png"
    plot_phase6_figure(
        PROBLEM,
        sensitivity,
        np.asarray(context["profile_x_mm"], dtype=float),
        median_slice,
        lower_slice,
        upper_slice,
        out_file=figure_file,
        vessel_radius_mm=float(context["vessel_radius_mm"]),
        hypoxic_center_x_mm=float(context["hypoxic_center_offset_mm"][0]),
        hypoxic_radius_mm=float(context["hypoxic_radius_mm"]),
        sampled_depth_cm=float(context["sampled_depth_cm"]),
        ranking_source=ranking_source,
        dpi=int(args.dpi),
    )

    params_rows: List[Dict[str, object]] = []
    for idx, values in enumerate(param_values):
        row = {"run_index": int(idx), "center_survival": float(center_survivals[idx])}
        for name, value in zip(PROBLEM["names"], values):
            row[str(name)] = float(value)
        params_rows.append(row)

    effects_rows = [
        {
            "parameter": str(name),
            "mu": float(sensitivity["mu"][idx]),
            "mu_star": float(sensitivity["mu_star"][idx]),
            "sigma": float(sensitivity["sigma"][idx]),
        }
        for idx, name in enumerate(PROBLEM["names"])
    ]
    credible_rows = [
        {
            "x_mm": float(x_value),
            "median_survival": float(median_slice[idx]),
            "lower_2p5": float(lower_slice[idx]),
            "upper_97p5": float(upper_slice[idx]),
        }
        for idx, x_value in enumerate(np.asarray(context["profile_x_mm"], dtype=float))
    ]

    params_csv = args.outdir / "phase6_parameter_samples.csv"
    effects_csv = args.outdir / "phase6_morris_indices.csv"
    credible_csv = args.outdir / "phase6_credible_band.csv"
    write_csv(params_rows, params_csv)
    write_csv(effects_rows, effects_csv)
    write_csv(credible_rows, credible_csv)

    best_idx = int(np.argmax(np.asarray(sensitivity["mu_star"], dtype=float)))
    summary = {
        "input_csv": str(args.csv),
        "input_summary": str(args.summary_json),
        "outdir": str(args.outdir),
        "runtime_sec": float(runtime_sec),
        "ranking_source": ranking_source,
        "problem": PROBLEM,
        "num_parameter_sets": int(len(param_values)),
        "max_workers": int(max_workers),
        "roi_shape": [int(value) for value in context["dose_grid"].shape],
        "profile_x_mm": [float(value) for value in np.asarray(context["profile_x_mm"], dtype=float)],
        "sampled_depth_cm": float(context["sampled_depth_cm"]),
        "top_parameter": str(PROBLEM["names"][best_idx]),
        "top_mu_star": float(sensitivity["mu_star"][best_idx]),
        "morris_indices": effects_rows,
        "outputs": {
            "figure": str(figure_file),
            "samples_csv": str(params_csv),
            "indices_csv": str(effects_csv),
            "credible_band_csv": str(credible_csv),
        },
        "baseline_phase5_model": context["baseline_summary"]["phase5_model"],
        "baseline_emission_model": context["baseline_summary"]["emission_model"],
        "normalization": context["normalization"],
    }
    summary_json = args.outdir / "phase6_sensitivity_summary.json"
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("Morris ranking (mu*):", flush=True)
    for row in sorted(effects_rows, key=lambda item: item["mu_star"], reverse=True):
        print(
            f"  {row['parameter']}: mu*={float(row['mu_star']):.6f}, "
            f"mu={float(row['mu']):.6f}, sigma={float(row['sigma']):.6f}",
            flush=True,
        )
    print(f"Saved Phase 6 outputs to {figure_file}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
