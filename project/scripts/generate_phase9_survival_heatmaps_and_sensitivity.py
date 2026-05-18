#!/usr/bin/env python
"""Generate Phase 9 survival heat maps and a focused local sensitivity analysis."""

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
from geometry_generators import generate_3d_lattice_geometry

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle
except Exception as exc:  # pragma: no cover
    raise RuntimeError("matplotlib is required for figure generation") from exc


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description=(
            "Regenerate Phase 9 LQ and full-model survival heat maps for the 3D lattice "
            "holdout and run a focused local sensitivity analysis."
        )
    )
    parser.add_argument(
        "--phase9-summary",
        type=Path,
        default=root
        / "runs"
        / "linac_6mv_polyenergetic_clinical_sfrt"
        / "analysis_phase9_holdout_3d_lattice"
        / "phase9_holdout_3d_lattice_summary.json",
        help="Phase 9 summary JSON used to recover the locked settings.",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=None,
        help="Optional output directory. Defaults to the folder containing --phase9-summary.",
    )
    parser.add_argument(
        "--sensitivity-fraction",
        type=float,
        default=0.20,
        help="One-at-a-time relative perturbation used for the local sensitivity study.",
    )
    parser.add_argument("--dpi", type=int, default=250, help="Figure DPI.")
    return parser.parse_args()


def build_phase9_solution(
    *,
    pitch_mm: float,
    csv_path: Path,
    prescribed_peak_dose_gy: float,
    uniform_dose_floor_fraction: float,
    n_beams_x: int,
    n_beams_y: int,
    x_shift_fraction_of_pitch: float,
    vessel_radius_mm: float,
    vessel_uptake: float,
    scaling_factor: float,
    alpha: float,
    beta: float,
    d_cyto: float,
    lambda_cyto: float,
    gamma: float,
    weight_ros: float,
    weight_cyto: float,
    weight_immune: float,
    pde_steps: int,
    pde_dt: float,
) -> Dict[str, object]:
    dose_grid, u_k, m_oxygen, m_type, meta = generate_3d_lattice_geometry(
        pitch_mm=float(pitch_mm),
        csv=csv_path,
        prescribed_peak_dose_gy=float(prescribed_peak_dose_gy),
        uniform_dose_floor_fraction=float(uniform_dose_floor_fraction),
        n_beams_x=int(n_beams_x),
        n_beams_y=int(n_beams_y),
        x_shift_fraction_of_pitch=float(x_shift_fraction_of_pitch),
        vessel_radius_mm=float(vessel_radius_mm),
        vessel_uptake=float(vessel_uptake),
        ros_vessel_uptake=0.05,
    )
    voxel_size_mm = tuple(float(value) for value in meta["voxel_size_mm"])
    lq_survival = np.exp(-float(alpha) * dose_grid - float(beta) * dose_grid**2).astype(np.float32)
    hazard_grid = run_pde_temporal_integration(
        dose_grid,
        voxel_size_mm,
        D_cyto=float(d_cyto),
        lambda_cyto=float(lambda_cyto),
        gamma=float(gamma),
        u_k=u_k,
        M_oxygen=m_oxygen,
        M_type=m_type,
        D_ros=0.8,
        lambda_ros=0.2,
        Emax_ros=1.5,
        Emax_cyto=0.8,
        w_ros=float(weight_ros),
        w_cyto=float(weight_cyto),
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
    immune_penalty, _ = calculate_systemic_immune_penalty(dose_grid, voxel_size_mm)
    return {
        "dose_grid": dose_grid,
        "lq_survival": lq_survival,
        "hazard_grid": hazard_grid,
        "final_survival": final_survival,
        "immune_penalty": float(immune_penalty),
        "meta": meta,
    }


def plot_survival_heatmaps(
    panels: List[Dict[str, object]],
    *,
    phase_title: str,
    target_depth_cm: float,
    vessel_radius_cm: float,
    quantity_key: str,
    out_file: Path,
    dpi: int,
) -> None:
    vmax = 1.0
    fig, axes = plt.subplots(2, len(panels), figsize=(5.0 * len(panels), 8.8), constrained_layout=True)

    for col, panel in enumerate(panels):
        volume = np.asarray(panel[quantity_key], dtype=np.float32)
        x_cm = np.asarray(panel["x_cm"], dtype=np.float32)
        y_cm = np.asarray(panel["y_cm"], dtype=np.float32)
        z_cm = np.asarray(panel["z_cm"], dtype=np.float32)
        pitch_mm = float(panel["pitch_mm"])
        z_idx = int(panel["target_depth_idx"])
        center_y = volume.shape[1] // 2

        xy_slice = volume[:, :, z_idx].T
        xz_slice = volume[:, center_y, :].T
        xy_extent = [float(x_cm[0]), float(x_cm[-1]), float(y_cm[0]), float(y_cm[-1])]
        xz_extent = [float(x_cm[0]), float(x_cm[-1]), float(z_cm[0]), float(z_cm[-1])]

        ax_xy = axes[0, col]
        img_xy = ax_xy.imshow(
            xy_slice,
            origin="lower",
            extent=xy_extent,
            aspect="equal",
            cmap="viridis",
            vmin=0.0,
            vmax=vmax,
        )
        ax_xy.add_patch(
            Circle((0.0, 0.0), radius=float(vessel_radius_cm), facecolor="none", edgecolor="#ffffff", linewidth=1.5, linestyle="--")
        )
        ax_xy.set_title(f"Pitch = {pitch_mm:.0f} mm\nx-y at z = {target_depth_cm:.2f} cm")
        ax_xy.set_xlabel("x (cm)")
        if col == 0:
            ax_xy.set_ylabel("y (cm)")

        ax_xz = axes[1, col]
        img_xz = ax_xz.imshow(
            xz_slice,
            origin="lower",
            extent=xz_extent,
            aspect="auto",
            cmap="viridis",
            vmin=0.0,
            vmax=vmax,
        )
        ax_xz.axvspan(-float(vessel_radius_cm), float(vessel_radius_cm), color="#ffffff", alpha=0.10)
        ax_xz.axhline(float(target_depth_cm), color="#ffffff", linestyle="--", linewidth=1.3)
        ax_xz.set_title(f"Pitch = {pitch_mm:.0f} mm\nx-z at center y")
        ax_xz.set_xlabel("x (cm)")
        if col == 0:
            ax_xz.set_ylabel("z (cm)")

    cbar = fig.colorbar(img_xz, ax=axes.ravel().tolist(), shrink=0.92)
    cbar.set_label("Survival fraction")
    fig.suptitle(phase_title, fontsize=15)
    fig.savefig(out_file, dpi=dpi)
    plt.close(fig)


def write_csv(rows: List[Dict[str, object]], out_file: Path) -> None:
    if not rows:
        raise ValueError("No rows supplied for CSV output.")
    with out_file.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def compute_center_valley_metrics(
    *,
    dose_grid: np.ndarray,
    lq_survival: np.ndarray,
    hazard_grid: np.ndarray,
    final_survival: np.ndarray,
    x_cm: np.ndarray,
    z_cm: np.ndarray,
    target_depth_cm: float,
    alpha: float,
    beta: float,
) -> Dict[str, float]:
    center_x = dose_grid.shape[0] // 2
    center_y = dose_grid.shape[1] // 2
    z_idx = nearest_index(z_cm, float(target_depth_cm))
    valley_survival = float(final_survival[center_x, center_y, z_idx])
    valley_deff = float(
        calculate_effective_dose(
            np.array([valley_survival], dtype=np.float32),
            alpha=float(alpha),
            beta=float(beta),
        )[0]
    )
    return {
        "sampled_depth_cm": float(z_cm[z_idx]),
        "center_valley_x_cm": float(x_cm[center_x]),
        "valley_dose_gy": float(dose_grid[center_x, center_y, z_idx]),
        "valley_survival_lq": float(lq_survival[center_x, center_y, z_idx]),
        "valley_survival_total": valley_survival,
        "valley_hazard": float(hazard_grid[center_x, center_y, z_idx]),
        "valley_effective_dose_gy": valley_deff,
    }


def plot_sensitivity_figure(
    rows: List[Dict[str, object]],
    out_file: Path,
    dpi: int,
) -> None:
    params = [str(row["parameter"]) for row in rows]
    surv_delta = [float(row["max_abs_delta_survival"]) for row in rows]
    deff_delta = [float(row["max_abs_delta_deff_gy"]) for row in rows]

    order = np.argsort(surv_delta)
    params = [params[idx] for idx in order]
    surv_delta = [surv_delta[idx] for idx in order]
    deff_delta = [deff_delta[idx] for idx in order]

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 5.0), constrained_layout=True)
    axes[0].barh(params, surv_delta, color="#1f77b4")
    axes[0].set_xlabel("Max |delta survival| at 40 mm valley")
    axes[0].set_title("Local sensitivity of survival")
    axes[0].grid(axis="x", alpha=0.25)

    axes[1].barh(params, deff_delta, color="#d62728")
    axes[1].set_xlabel("Max |delta D_eff| (Gy) at 40 mm valley")
    axes[1].set_title("Local sensitivity of effective dose")
    axes[1].grid(axis="x", alpha=0.25)

    fig.suptitle("Figure 5: Phase 9 local sensitivity analysis (40 mm center valley)", fontsize=14)
    fig.savefig(out_file, dpi=dpi)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    summary = json.loads(args.phase9_summary.read_text(encoding="utf-8"))
    outdir = args.outdir or args.phase9_summary.parent
    outdir.mkdir(parents=True, exist_ok=True)

    locked = summary["locked_parameters"]
    inputs = summary["inputs"]
    lq_model = summary["lq_model"]
    target_depth_cm = float(summary["target_depth_cm"])

    baseline_panels: List[Dict[str, object]] = []
    baseline_40mm_scalars: Dict[str, float] | None = None

    for pitch_mm in [float(value) for value in inputs["pitches_mm"]]:
        print(f"Building Phase 9 survival heat maps for pitch {pitch_mm:g} mm...", flush=True)
        solved = build_phase9_solution(
            pitch_mm=float(pitch_mm),
            csv_path=Path(str(inputs["csv"])),
            prescribed_peak_dose_gy=float(inputs["prescribed_peak_dose_gy"]),
            uniform_dose_floor_fraction=float(inputs["uniform_dose_floor_fraction"]),
            n_beams_x=int(inputs["n_beams_x"]),
            n_beams_y=int(inputs["n_beams_y"]),
            x_shift_fraction_of_pitch=float(inputs["x_shift_fraction_of_pitch"]),
            vessel_radius_mm=float(inputs["vessel_radius_mm"]),
            vessel_uptake=float(inputs["vessel_uptake"]),
            scaling_factor=float(locked["scaling_factor"]),
            alpha=float(lq_model["alpha"]),
            beta=float(lq_model["beta"]),
            d_cyto=float(locked["D_cyto"]),
            lambda_cyto=float(locked["lambda_cyto"]),
            gamma=float(locked["gamma"]),
            weight_ros=float(locked["weight_ros"]),
            weight_cyto=float(locked["weight_cyto"]),
            weight_immune=float(locked["weight_immune"]),
            pde_steps=400,
            pde_dt=0.12,
        )

        meta = solved["meta"]
        x_cm = np.asarray(meta["x_cm"], dtype=np.float32)
        y_cm = centered_axis_cm(solved["dose_grid"].shape[1], float(meta["voxel_size_mm"][1]) / 10.0)
        z_cm = np.asarray(meta["z_cm"], dtype=np.float32)
        panel = {
            "pitch_mm": float(pitch_mm),
            "x_cm": x_cm,
            "y_cm": y_cm,
            "z_cm": z_cm,
            "target_depth_idx": nearest_index(z_cm, float(target_depth_cm)),
            "lq_survival": solved["lq_survival"],
            "final_survival": solved["final_survival"],
        }
        baseline_panels.append(panel)

        if int(round(pitch_mm)) == 40:
            baseline_40mm_scalars = {
                **compute_center_valley_metrics(
                    dose_grid=solved["dose_grid"],
                    lq_survival=solved["lq_survival"],
                    hazard_grid=solved["hazard_grid"],
                    final_survival=solved["final_survival"],
                    x_cm=x_cm,
                    z_cm=z_cm,
                    target_depth_cm=float(target_depth_cm),
                    alpha=float(lq_model["alpha"]),
                    beta=float(lq_model["beta"]),
                ),
                "immune_penalty": float(solved["immune_penalty"]),
            }

    figure_lq = Path(outdir) / "figure3_phase9_lq_survival_heatmaps.png"
    figure_phase9 = Path(outdir) / "figure4_phase9_model_survival_heatmaps.png"
    plot_survival_heatmaps(
        baseline_panels,
        phase_title="Figure 3: Simplified LQ survival heat maps",
        target_depth_cm=float(target_depth_cm),
        vessel_radius_cm=float(inputs["vessel_radius_mm"]) / 10.0,
        quantity_key="lq_survival",
        out_file=figure_lq,
        dpi=int(args.dpi),
    )
    plot_survival_heatmaps(
        baseline_panels,
        phase_title="Figure 4: Phase 9 model survival heat maps",
        target_depth_cm=float(target_depth_cm),
        vessel_radius_cm=float(inputs["vessel_radius_mm"]) / 10.0,
        quantity_key="final_survival",
        out_file=figure_phase9,
        dpi=int(args.dpi),
    )

    if baseline_40mm_scalars is None:
        raise RuntimeError("40 mm baseline panel was not generated; cannot run sensitivity analysis.")

    baseline_survival = float(baseline_40mm_scalars["valley_survival_total"])
    baseline_deff = float(baseline_40mm_scalars["valley_effective_dose_gy"])
    baseline_hazard = float(baseline_40mm_scalars["valley_hazard"])
    baseline_lq = float(baseline_40mm_scalars["valley_survival_lq"])
    baseline_immune = float(baseline_40mm_scalars["immune_penalty"])
    frac = float(args.sensitivity_fraction)

    sensitivity_rows: List[Dict[str, object]] = []

    def record_parameter(parameter: str, low_survival: float, high_survival: float, low_deff: float, high_deff: float) -> None:
        sensitivity_rows.append(
            {
                "parameter": parameter,
                "fractional_perturbation": frac,
                "baseline_survival": baseline_survival,
                "low_survival": float(low_survival),
                "high_survival": float(high_survival),
                "baseline_deff_gy": baseline_deff,
                "low_deff_gy": float(low_deff),
                "high_deff_gy": float(high_deff),
                "max_abs_delta_survival": max(abs(float(low_survival) - baseline_survival), abs(float(high_survival) - baseline_survival)),
                "max_abs_delta_deff_gy": max(abs(float(low_deff) - baseline_deff), abs(float(high_deff) - baseline_deff)),
            }
        )

    for parameter, low_value, high_value in [
        ("D_cyto", float(locked["D_cyto"]) * (1.0 - frac), float(locked["D_cyto"]) * (1.0 + frac)),
        ("lambda_cyto", float(locked["lambda_cyto"]) * (1.0 - frac), float(locked["lambda_cyto"]) * (1.0 + frac)),
        ("gamma", float(locked["gamma"]) * (1.0 - frac), float(locked["gamma"]) * (1.0 + frac)),
    ]:
        values: List[Tuple[float, float]] = []
        for value in [low_value, high_value]:
            print(f"Running Phase 9 sensitivity case: {parameter}={value:.6f} on 40 mm...", flush=True)
            kwargs = {
                "pitch_mm": 40.0,
                "csv_path": Path(str(inputs["csv"])),
                "prescribed_peak_dose_gy": float(inputs["prescribed_peak_dose_gy"]),
                "uniform_dose_floor_fraction": float(inputs["uniform_dose_floor_fraction"]),
                "n_beams_x": int(inputs["n_beams_x"]),
                "n_beams_y": int(inputs["n_beams_y"]),
                "x_shift_fraction_of_pitch": float(inputs["x_shift_fraction_of_pitch"]),
                "vessel_radius_mm": float(inputs["vessel_radius_mm"]),
                "vessel_uptake": float(inputs["vessel_uptake"]),
                "scaling_factor": float(locked["scaling_factor"]),
                "alpha": float(lq_model["alpha"]),
                "beta": float(lq_model["beta"]),
                "d_cyto": float(locked["D_cyto"]),
                "lambda_cyto": float(locked["lambda_cyto"]),
                "gamma": float(locked["gamma"]),
                "weight_ros": float(locked["weight_ros"]),
                "weight_cyto": float(locked["weight_cyto"]),
                "weight_immune": float(locked["weight_immune"]),
                "pde_steps": 400,
                "pde_dt": 0.12,
            }
            if parameter == "D_cyto":
                kwargs["d_cyto"] = float(value)
            elif parameter == "lambda_cyto":
                kwargs["lambda_cyto"] = float(value)
            elif parameter == "gamma":
                kwargs["gamma"] = float(value)

            solved = build_phase9_solution(**kwargs)
            meta = solved["meta"]
            scalars = compute_center_valley_metrics(
                dose_grid=solved["dose_grid"],
                lq_survival=solved["lq_survival"],
                hazard_grid=solved["hazard_grid"],
                final_survival=solved["final_survival"],
                x_cm=np.asarray(meta["x_cm"], dtype=np.float32),
                z_cm=np.asarray(meta["z_cm"], dtype=np.float32),
                target_depth_cm=float(target_depth_cm),
                alpha=float(lq_model["alpha"]),
                beta=float(lq_model["beta"]),
            )
            values.append((float(scalars["valley_survival_total"]), float(scalars["valley_effective_dose_gy"])))
        record_parameter(parameter, values[0][0], values[1][0], values[0][1], values[1][1])

    total_penalty = -np.log(baseline_survival / baseline_lq) / float(locked["scaling_factor"])
    low_scale = float(locked["scaling_factor"]) * (1.0 - frac)
    high_scale = float(locked["scaling_factor"]) * (1.0 + frac)
    low_survival_scale = baseline_lq * np.exp(-total_penalty * low_scale)
    high_survival_scale = baseline_lq * np.exp(-total_penalty * high_scale)
    low_deff_scale = float(calculate_effective_dose(np.array([low_survival_scale], dtype=np.float32), alpha=float(lq_model["alpha"]), beta=float(lq_model["beta"]))[0])
    high_deff_scale = float(calculate_effective_dose(np.array([high_survival_scale], dtype=np.float32), alpha=float(lq_model["alpha"]), beta=float(lq_model["beta"]))[0])
    record_parameter("scaling_factor", low_survival_scale, high_survival_scale, low_deff_scale, high_deff_scale)

    low_w = float(locked["weight_immune"]) * (1.0 - frac)
    high_w = float(locked["weight_immune"]) * (1.0 + frac)
    local_hazard_only = baseline_hazard
    low_survival_w = baseline_lq * np.exp(-(local_hazard_only + low_w * baseline_immune) * float(locked["scaling_factor"]))
    high_survival_w = baseline_lq * np.exp(-(local_hazard_only + high_w * baseline_immune) * float(locked["scaling_factor"]))
    low_deff_w = float(calculate_effective_dose(np.array([low_survival_w], dtype=np.float32), alpha=float(lq_model["alpha"]), beta=float(lq_model["beta"]))[0])
    high_deff_w = float(calculate_effective_dose(np.array([high_survival_w], dtype=np.float32), alpha=float(lq_model["alpha"]), beta=float(lq_model["beta"]))[0])
    record_parameter("weight_immune", low_survival_w, high_survival_w, low_deff_w, high_deff_w)

    sensitivity_csv = Path(outdir) / "phase9_sensitivity_metrics.csv"
    sensitivity_json = Path(outdir) / "phase9_sensitivity_summary.json"
    figure_sensitivity = Path(outdir) / "figure5_phase9_local_sensitivity.png"
    write_csv(sensitivity_rows, sensitivity_csv)
    plot_sensitivity_figure(sensitivity_rows, figure_sensitivity, int(args.dpi))

    sensitivity_summary = {
        "phase": "Phase 9",
        "description": "Local one-at-a-time sensitivity analysis at the 40 mm center valley.",
        "baseline_40mm": baseline_40mm_scalars,
        "fractional_perturbation": frac,
        "parameters_tested": sensitivity_rows,
        "outputs": {
            "metrics_csv": str(sensitivity_csv),
            "summary_json": str(sensitivity_json),
            "figure": str(figure_sensitivity),
            "figure_lq_heatmaps": str(figure_lq),
            "figure_phase9_heatmaps": str(figure_phase9),
        },
    }
    sensitivity_json.write_text(json.dumps(sensitivity_summary, indent=2), encoding="utf-8")

    print(f"Phase 9 LQ heat maps: {figure_lq}")
    print(f"Phase 9 model heat maps: {figure_phase9}")
    print(f"Phase 9 sensitivity figure: {figure_sensitivity}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
