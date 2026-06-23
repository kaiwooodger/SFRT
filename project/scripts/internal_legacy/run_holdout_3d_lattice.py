#!/usr/bin/env python
"""Phase 9 Part 1: run the locked-parameter 3D lattice holdout validation."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List

import numpy as np

from analyze_250mev_sfrt_plan import nearest_index
from bystander_multispecies_pde_solver import (
    calculate_effective_dose,
    calculate_phase7_survival,
    run_pde_temporal_integration,
)
from geometry_generators import generate_3d_lattice_geometry

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
            "Run the Phase 9 3D lattice holdout with locked Phase 8 parameters "
            "and output effective dose values at the central valley."
        )
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=root / "runs" / "linac_6mv_polyenergetic_clinical_sfrt" / "analysis_phase9_holdout_3d_lattice",
        help="Directory for the Phase 9 3D holdout outputs.",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=root / "runs" / "linac_6mv_polyenergetic_clinical_sfrt" / "case" / "dosedata.csv",
        help="TOPAS single-beam kernel CSV used to rebuild the 3D lattice geometry.",
    )
    parser.add_argument(
        "--pitches-mm",
        nargs="+",
        type=float,
        default=[20.0, 30.0, 40.0],
        help="3D lattice pitches to validate.",
    )
    parser.add_argument(
        "--prescribed-peak-dose-gy",
        type=float,
        default=10.0,
        help="Prescribed reference peak dose for the rebuilt lattice.",
    )
    parser.add_argument(
        "--uniform-dose-floor-fraction",
        type=float,
        default=0.015,
        help="Uniform transmission floor added across the lattice.",
    )
    parser.add_argument("--n-beams-x", type=int, default=7, help="Beam copies along x.")
    parser.add_argument("--n-beams-y", type=int, default=7, help="Beam copies along y.")
    parser.add_argument(
        "--x-shift-fraction-of-pitch",
        type=float,
        default=0.5,
        help="Half-pitch shift that places the pristine valley at x = 0.",
    )
    parser.add_argument(
        "--vessel-radius-mm",
        type=float,
        default=1.5,
        help="Radius of the central vascular sink in the true valley.",
    )
    parser.add_argument(
        "--vessel-uptake",
        type=float,
        default=0.60,
        help="Cytokine uptake coefficient inside the vessel.",
    )
    parser.add_argument(
        "--ros-vessel-uptake",
        type=float,
        default=0.05,
        help="ROS uptake coefficient inside the vessel.",
    )
    parser.add_argument("--alpha", type=float, default=0.03, help="LQ alpha used for baseline survival and D_eff inversion.")
    parser.add_argument("--beta", type=float, default=0.003, help="LQ beta used for baseline survival and D_eff inversion.")
    parser.add_argument("--pde-steps", type=int, default=400, help="Number of temporal PDE steps.")
    parser.add_argument("--pde-dt", type=float, default=0.12, help="Explicit Euler time step.")
    parser.add_argument(
        "--target-depth-cm",
        type=float,
        default=5.03,
        help="Target depth for the reported center-valley readout.",
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


def plot_deff_vs_pitch(rows: List[Dict[str, object]], out_file: Path, dpi: int) -> None:
    pitches = [float(row["pitch_mm"]) for row in rows]
    deff = [float(row["valley_effective_dose_gy"]) for row in rows]
    survival = [float(row["valley_survival_total"]) for row in rows]

    fig, ax = plt.subplots(figsize=(8.6, 5.4), constrained_layout=True)
    ax.plot(pitches, deff, color="#d62728", marker="o", linewidth=2.6, markersize=7)
    for pitch, value, sf in zip(pitches, deff, survival):
        ax.annotate(
            f"SF={sf:.3f}",
            (pitch, value),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
            fontsize=9,
        )
    ax.set_xlabel("Lattice pitch (mm)")
    ax.set_ylabel("Effective dose (Gy)")
    ax.set_title("Figure 1: Phase 9 3D holdout center-valley effective dose")
    ax.grid(alpha=0.25)
    fig.savefig(out_file, dpi=dpi)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    print("=== EXECUTING PHASE 9 3D HOLDOUT VALIDATION ===")
    print(
        f"Locked Parameters: D={LOCKED_D_CYTO}, lambda={LOCKED_LAMBDA_CYTO}, "
        f"gamma={LOCKED_GAMMA}, scaling={LOCKED_SCALING_FACTOR}"
    )

    rows: List[Dict[str, object]] = []

    for pitch_mm in [float(value) for value in args.pitches_mm]:
        print(f"\n--- Simulating {pitch_mm:g} mm 3D LATTICE ---", flush=True)

        dose_grid, u_k, M_oxygen, M_type, geometry_meta = generate_3d_lattice_geometry(
            pitch_mm=float(pitch_mm),
            csv=args.csv,
            prescribed_peak_dose_gy=float(args.prescribed_peak_dose_gy),
            uniform_dose_floor_fraction=float(args.uniform_dose_floor_fraction),
            n_beams_x=int(args.n_beams_x),
            n_beams_y=int(args.n_beams_y),
            x_shift_fraction_of_pitch=float(args.x_shift_fraction_of_pitch),
            vessel_radius_mm=float(args.vessel_radius_mm),
            vessel_uptake=float(args.vessel_uptake),
            ros_vessel_uptake=float(args.ros_vessel_uptake),
        )

        voxel_size_mm = tuple(float(value) for value in geometry_meta["voxel_size_mm"])
        z_cm = np.asarray(geometry_meta["z_cm"], dtype=np.float32)
        x_cm = np.asarray(geometry_meta["x_cm"], dtype=np.float32)
        target_depth_idx = nearest_index(z_cm, float(args.target_depth_cm))
        center_y = dose_grid.shape[1] // 2
        center_x = dose_grid.shape[0] // 2

        lq_survival_grid = np.exp(
            -float(args.alpha) * dose_grid - float(args.beta) * dose_grid**2
        ).astype(np.float32)

        hazard_grid = run_pde_temporal_integration(
            dose_grid,
            voxel_size_mm,
            D_cyto=LOCKED_D_CYTO,
            lambda_cyto=LOCKED_LAMBDA_CYTO,
            gamma=LOCKED_GAMMA,
            u_k=u_k,
            M_oxygen=M_oxygen,
            M_type=M_type,
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
            lq_survival_grid,
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

        valley_survival = float(final_survival[center_x, center_y, target_depth_idx])
        valley_deff = float(deff_grid[center_x, center_y, target_depth_idx])
        valley_dose = float(dose_grid[center_x, center_y, target_depth_idx])
        valley_hazard = float(hazard_grid[center_x, center_y, target_depth_idx])

        print(f"  -> Target Depth: {float(z_cm[target_depth_idx]) * 10.0:.3f} mm")
        print(f"  -> Final Valley Survival: {valley_survival:.4f}")
        print(f"  -> Effective Dose (Deff): {valley_deff:.2f} Gy")

        rows.append(
            {
                "pitch_mm": float(pitch_mm),
                "sampled_depth_cm": float(z_cm[target_depth_idx]),
                "center_valley_x_cm": float(x_cm[center_x]),
                "valley_dose_gy": valley_dose,
                "valley_hazard": valley_hazard,
                "valley_survival_total": valley_survival,
                "valley_effective_dose_gy": valley_deff,
                "voxel_size_x_mm": float(voxel_size_mm[0]),
                "voxel_size_y_mm": float(voxel_size_mm[1]),
                "voxel_size_z_mm": float(voxel_size_mm[2]),
                "vessel_radius_mm": float(args.vessel_radius_mm),
                "vessel_uptake": float(args.vessel_uptake),
            }
        )

    summary_csv = args.outdir / "phase9_holdout_3d_lattice_metrics.csv"
    summary_json = args.outdir / "phase9_holdout_3d_lattice_summary.json"
    figure_file = args.outdir / "figure1_phase9_holdout_3d_lattice_deff.png"

    write_csv(rows, summary_csv)
    plot_deff_vs_pitch(rows, figure_file, int(args.dpi))

    summary = {
        "phase": "Phase 9",
        "description": "Locked-parameter 3D lattice holdout with effective dose output.",
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
        "target_depth_cm": float(args.target_depth_cm),
        "inputs": {
            "csv": str(args.csv),
            "pitches_mm": [float(value) for value in args.pitches_mm],
            "prescribed_peak_dose_gy": float(args.prescribed_peak_dose_gy),
            "uniform_dose_floor_fraction": float(args.uniform_dose_floor_fraction),
            "n_beams_x": int(args.n_beams_x),
            "n_beams_y": int(args.n_beams_y),
            "x_shift_fraction_of_pitch": float(args.x_shift_fraction_of_pitch),
            "vessel_radius_mm": float(args.vessel_radius_mm),
            "vessel_uptake": float(args.vessel_uptake),
        },
        "results": rows,
        "outputs": {
            "metrics_csv": str(summary_csv),
            "summary_json": str(summary_json),
            "figure": str(figure_file),
        },
    }
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"\nPhase 9 summary: {summary_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
