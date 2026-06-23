#!/usr/bin/env python
"""Generate the exact two-stripe Phase 8 dimensional holdout validation set."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np

from bystander_multispecies_pde_solver import (
    calculate_phase7_survival,
    solve_multispecies_pde_3d_with_hazard,
)
from geometry_generalization_sets import generate_2d_parallel_stripe_geometry

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
            "Run the exact two-stripe true-valley Phase 8 dimensional holdout using "
            "the locked half-field calibration parameters."
        )
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=root / "runs" / "linac_6mv_polyenergetic_clinical_sfrt" / "analysis_phase8_two_stripe_holdout",
        help="Directory for the exact two-stripe holdout outputs.",
    )
    parser.add_argument(
        "--grid-shape",
        nargs=3,
        type=int,
        default=[161, 41, 21],
        help="Synthetic grid shape in project order: nx ny nz.",
    )
    parser.add_argument(
        "--voxel-size-mm",
        type=float,
        default=1.0,
        help="Isotropic voxel size used for the synthetic in vitro geometry.",
    )
    parser.add_argument(
        "--pitches-mm",
        nargs="+",
        type=float,
        default=[20.0, 30.0, 40.0],
        help="Stripe separations to validate.",
    )
    parser.add_argument(
        "--beam-width-mm",
        type=float,
        default=10.0,
        help="Physical width of each infinite stripe.",
    )
    parser.add_argument(
        "--dose-peak-gy",
        type=float,
        default=10.0,
        help="High-dose value assigned to each stripe.",
    )
    parser.add_argument(
        "--leakage",
        type=float,
        default=0.0,
        help="Uniform fractional leakage floor under the stripes.",
    )
    parser.add_argument("--alpha", type=float, default=0.10, help="Locked LQ alpha from the half-field calibration.")
    parser.add_argument("--beta", type=float, default=0.05, help="Locked LQ beta from the half-field calibration.")
    parser.add_argument("--pde-steps", type=int, default=400, help="Number of temporal PDE steps.")
    parser.add_argument("--pde-dt", type=float, default=0.12, help="Explicit Euler time step.")
    parser.add_argument("--ros-diffusion-coeff", type=float, default=0.8, help="Locked ROS diffusion coefficient.")
    parser.add_argument("--cytokine-diffusion-coeff", type=float, default=1.2, help="Locked tuned cytokine diffusion coefficient.")
    parser.add_argument("--ros-decay-coeff", type=float, default=0.2, help="Locked ROS decay coefficient.")
    parser.add_argument("--cytokine-decay-coeff", type=float, default=0.001, help="Locked tuned cytokine decay coefficient.")
    parser.add_argument("--ros-emission-emax", type=float, default=1.5, help="Locked ROS emission amplitude.")
    parser.add_argument("--cytokine-emission-emax", type=float, default=0.8, help="Locked cytokine emission amplitude.")
    parser.add_argument("--emission-gamma-per-gy", type=float, default=0.35, help="Locked gamma from calibration.")
    parser.add_argument("--weight-ros", type=float, default=0.40, help="ROS contribution to local hazard.")
    parser.add_argument("--weight-cyto", type=float, default=0.40, help="Cytokine contribution to local hazard.")
    parser.add_argument(
        "--scaling-factor",
        type=float,
        default=0.0029365812996595296,
        help="Locked Phase 8 half-field scaling factor keeping the +2 mm anchor at 0.70.",
    )
    parser.add_argument("--dpi", type=int, default=250, help="Figure DPI.")
    return parser.parse_args()


def centered_axis_mm(n_voxels: int, spacing_mm: float) -> np.ndarray:
    return (np.arange(int(n_voxels), dtype=np.float32) - (int(n_voxels) - 1) / 2.0) * float(spacing_mm)


def write_csv(rows: List[Dict[str, object]], out_file: Path) -> None:
    if not rows:
        raise ValueError("No rows supplied for CSV output.")
    with out_file.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def compose_emission_tensor(
    dose_grid: np.ndarray,
    *,
    emission_emax: Sequence[float],
    emission_gamma_per_gy: float,
    type_modifier: np.ndarray,
    oxygen_modifier: np.ndarray,
) -> np.ndarray:
    base_emission = (1.0 - np.exp(-float(emission_gamma_per_gy) * np.asarray(dose_grid, dtype=np.float32))).astype(np.float32)
    emax_vector = np.asarray(emission_emax, dtype=np.float32)
    return (
        emax_vector[:, None, None, None]
        * base_emission[None, :, :, :]
        * np.asarray(type_modifier, dtype=np.float32)
        * np.asarray(oxygen_modifier, dtype=np.float32)
    ).astype(np.float32, copy=False)


def calculate_lq_survival(dose_grid: np.ndarray, alpha: float, beta: float) -> np.ndarray:
    return np.exp(-float(alpha) * dose_grid - float(beta) * dose_grid**2).astype(np.float32)


def centerline_profile(volume: np.ndarray) -> np.ndarray:
    center_y = volume.shape[1] // 2
    center_z = volume.shape[2] // 2
    return np.asarray(volume[:, center_y, center_z], dtype=np.float32)


def plot_profiles(
    x_mm: np.ndarray,
    results: List[Dict[str, object]],
    out_file: Path,
    dpi: int,
) -> None:
    fig, axes = plt.subplots(len(results), 1, figsize=(10.5, 3.3 * len(results)), constrained_layout=True, sharex=True)
    if len(results) == 1:
        axes = [axes]

    for ax, result in zip(axes, results):
        dose_profile = np.asarray(result["dose_profile"], dtype=np.float32)
        lq_profile = np.asarray(result["lq_profile"], dtype=np.float32)
        final_profile = np.asarray(result["final_profile"], dtype=np.float32)
        pitch_mm = float(result["pitch_mm"])
        ax.plot(x_mm, lq_profile, color="#1f77b4", linewidth=2.0, label="Standard LQ")
        ax.plot(x_mm, final_profile, color="#d62728", linewidth=2.0, label="Phase 8 locked")
        ax.axvline(0.0, color="#555555", linestyle="--", linewidth=1.3, label="True valley center")
        ax.plot(x_mm, dose_profile / max(float(np.max(dose_profile)), 1.0), color="#555555", linestyle=":", linewidth=1.8, label="Dose / max")
        ax.set_ylim(0.0, 1.02)
        ax.set_ylabel("Survival")
        ax.set_title(f"Pitch = {pitch_mm:.0f} mm")
        ax.grid(alpha=0.25)

    axes[0].legend(loc="lower left", ncol=3)
    axes[-1].set_xlabel("Lateral position x (mm)")
    fig.suptitle("Figure 1: Exact two-stripe true-valley Phase 8 holdout")
    fig.savefig(out_file, dpi=dpi)
    plt.close(fig)


def write_markdown_report(summary: Dict[str, object], out_file: Path) -> None:
    lines = [
        "# Phase 8 Exact Two-Stripe Holdout",
        "",
        "- Geometry: two infinite parallel Y-Z planes with the pristine true valley aligned at `x = 0`.",
        (
            f"- Locked parameters: `D_cyto={float(summary['biological_model']['diffusion_coeffs'][1]):.3f}`, "
            f"`lambda_cyto={float(summary['biological_model']['decay_coeffs'][1]):.4f}`, "
            f"`gamma={float(summary['biological_model']['emission_gamma_per_gy']):.3f}`, "
            f"`scaling={float(summary['biological_model']['scaling_factor']):.9f}`."
        ),
        "",
        "| Pitch | Center Valley Dose (Gy) | Center Valley Survival (LQ) | Center Valley Survival (Phase 8) | Center Valley Hazard |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for pitch_key, values in summary["generated_pitches"].items():
        lines.append(
            f"| {pitch_key} mm | {float(values['center_valley_dose_gy']):.3f} | "
            f"{float(values['center_valley_survival_lq']):.3f} | "
            f"{float(values['center_valley_survival_total']):.3f} | "
            f"{float(values['center_valley_hazard']):.3f} |"
        )
    out_file.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    grid_shape = tuple(int(value) for value in args.grid_shape)
    x_mm = centered_axis_mm(int(grid_shape[0]), float(args.voxel_size_mm))

    diffusion_coeffs = (float(args.ros_diffusion_coeff), float(args.cytokine_diffusion_coeff))
    decay_coeffs = (float(args.ros_decay_coeff), float(args.cytokine_decay_coeff))
    emission_emax = (float(args.ros_emission_emax), float(args.cytokine_emission_emax))
    hazard_weights = (float(args.weight_ros), float(args.weight_cyto))

    summary: Dict[str, object] = {
        "outdir": str(args.outdir),
        "geometry_model": {
            "grid_shape": [int(value) for value in grid_shape],
            "voxel_size_mm": float(args.voxel_size_mm),
            "beam_width_mm": float(args.beam_width_mm),
            "dose_peak_gy": float(args.dose_peak_gy),
            "leakage": float(args.leakage),
        },
        "biological_model": {
            "alpha": float(args.alpha),
            "beta": float(args.beta),
            "steps": int(args.pde_steps),
            "dt": float(args.pde_dt),
            "diffusion_coeffs": [float(value) for value in diffusion_coeffs],
            "decay_coeffs": [float(value) for value in decay_coeffs],
            "emission_emax": [float(value) for value in emission_emax],
            "emission_gamma_per_gy": float(args.emission_gamma_per_gy),
            "hazard_weights": [float(value) for value in hazard_weights],
            "scaling_factor": float(args.scaling_factor),
        },
        "generated_pitches": {},
        "outputs": {},
    }

    rows: List[Dict[str, object]] = []
    plot_rows: List[Dict[str, object]] = []

    for pitch_mm in [float(value) for value in args.pitches_mm]:
        dose_grid, uptake_tensor, oxygen_modifier, type_modifier = generate_2d_parallel_stripe_geometry(
            grid_shape,
            float(args.voxel_size_mm),
            pitch_mm=float(pitch_mm),
            beam_width_mm=float(args.beam_width_mm),
            dose_peak_gy=float(args.dose_peak_gy),
            leakage=float(args.leakage),
            num_species=2,
        )
        emission_tensor = compose_emission_tensor(
            dose_grid,
            emission_emax=emission_emax,
            emission_gamma_per_gy=float(args.emission_gamma_per_gy),
            type_modifier=type_modifier,
            oxygen_modifier=oxygen_modifier,
        )
        multispecies_tensor, hazard_grid = solve_multispecies_pde_3d_with_hazard(
            dose_grid=dose_grid,
            voxel_size_mm=float(args.voxel_size_mm),
            steps=int(args.pde_steps),
            dt=float(args.pde_dt),
            diffusion_coeffs=diffusion_coeffs,
            decay_coeffs=decay_coeffs,
            emission_emax=emission_emax,
            emission_gamma_per_gy=float(args.emission_gamma_per_gy),
            emission_tensor=emission_tensor,
            uptake_tensor=uptake_tensor,
            hazard_weights=hazard_weights,
            progress_interval=50,
            verbose=True,
        )
        lq_survival_grid = calculate_lq_survival(dose_grid, float(args.alpha), float(args.beta))
        final_survival = calculate_phase7_survival(
            lq_survival_grid,
            hazard_grid,
            dose_grid,
            float(args.voxel_size_mm),
            float(args.scaling_factor),
            weight_immune=0.0,
            verbose=False,
        )

        center_x = dose_grid.shape[0] // 2
        center_y = dose_grid.shape[1] // 2
        center_z = dose_grid.shape[2] // 2
        valley_survival = float(final_survival[center_x, center_y, center_z])
        print(f"Pitch {pitch_mm:g} mm | Center Valley Survival: {valley_survival:.4f}", flush=True)

        dose_profile = centerline_profile(dose_grid)
        lq_profile = centerline_profile(lq_survival_grid)
        final_profile = centerline_profile(final_survival)
        hazard_profile = centerline_profile(hazard_grid)

        pitch_key = f"{pitch_mm:g}"
        summary["generated_pitches"][pitch_key] = {
            "pitch_mm": float(pitch_mm),
            "center_valley_x_mm": float(x_mm[center_x]),
            "center_valley_dose_gy": float(dose_grid[center_x, center_y, center_z]),
            "center_valley_survival_lq": float(lq_survival_grid[center_x, center_y, center_z]),
            "center_valley_survival_total": valley_survival,
            "center_valley_hazard": float(hazard_grid[center_x, center_y, center_z]),
            "max_hazard": float(np.max(hazard_grid)),
            "max_ros_concentration": float(np.max(multispecies_tensor[0])),
            "max_cytokine_concentration": float(np.max(multispecies_tensor[1])),
        }
        plot_rows.append(
            {
                "pitch_mm": float(pitch_mm),
                "dose_profile": dose_profile,
                "lq_profile": lq_profile,
                "final_profile": final_profile,
            }
        )

        for x_value, dose_value, lq_value, total_value, hazard_value in zip(
            x_mm,
            dose_profile,
            lq_profile,
            final_profile,
            hazard_profile,
        ):
            rows.append(
                {
                    "pitch_mm": float(pitch_mm),
                    "x_mm": float(x_value),
                    "dose_gy": float(dose_value),
                    "survival_lq": float(lq_value),
                    "survival_total": float(total_value),
                    "hazard": float(hazard_value),
                }
            )

    figure_file = args.outdir / "figure1_phase8_two_stripe_holdout_profiles.png"
    csv_file = args.outdir / "phase8_two_stripe_holdout_profiles.csv"
    summary_json = args.outdir / "phase8_two_stripe_holdout_summary.json"
    summary_md = args.outdir / "phase8_two_stripe_holdout_summary.md"

    plot_profiles(x_mm, plot_rows, figure_file, int(args.dpi))
    write_csv(rows, csv_file)
    summary["outputs"] = {
        "figure": str(figure_file),
        "profiles_csv": str(csv_file),
        "summary_json": str(summary_json),
        "summary_md": str(summary_md),
    }
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_markdown_report(summary, summary_md)

    print(f"Phase 8 two-stripe summary: {summary_json}")
    print(f"Phase 8 two-stripe report: {summary_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
