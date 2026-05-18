#!/usr/bin/env python
"""Generate the final 1D asymmetry profile comparing Phase 3 and Phase 4."""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from analyze_250mev_sfrt_plan import centered_axis_cm, choose_reference_depth, depth_axis_cm, nearest_index, reference_peak_value
from analyze_topas_outputs import load_topas_grid
from bystander_multispecies_pde_solver import build_cylindrical_uptake_tensor, solve_multispecies_pde_3d
from generate_phase2_multispecies_vessel_valley_figures import build_lattice_with_x_shift

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
            "Build the final 1D asymmetry profile comparing Standard LQ, "
            "Phase 3 symmetric emission, and Phase 4 right-peak hypoxic emission."
        )
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=root / "runs" / "linac_6mv_polyenergetic_clinical_sfrt" / "case" / "dosedata.csv",
        help="TOPAS dose CSV for the polyenergetic clinical-upgrade kernel.",
    )
    parser.add_argument(
        "--phase3-summary",
        type=Path,
        default=root / "runs" / "linac_6mv_polyenergetic_clinical_sfrt" / "analysis_phase2_multispecies_true_valley" / "phase2_multispecies_summary.json",
        help="Phase 3 true-valley summary JSON used for scaling and geometry defaults.",
    )
    parser.add_argument(
        "--phase4-summary",
        type=Path,
        default=root / "runs" / "linac_6mv_polyenergetic_clinical_sfrt" / "analysis_phase2_multispecies_true_valley_phase4_rightpeak" / "phase2_multispecies_summary.json",
        help="Phase 4 right-peak summary JSON used for the asymmetric state-dependent emission settings.",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=root / "runs" / "linac_6mv_polyenergetic_clinical_sfrt" / "analysis_phase4_asymmetry_profile_rightpeak",
        help="Directory for the final asymmetry profile figure and tables.",
    )
    parser.add_argument("--pitch-mm", type=float, default=40.0, help="Pitch used for the asymmetry profile.")
    parser.add_argument("--n-beams-x", type=int, default=7, help="Beam copies along x.")
    parser.add_argument("--n-beams-y", type=int, default=7, help="Beam copies along y.")
    parser.add_argument(
        "--prescribed-peak-dose-gy",
        type=float,
        default=10.0,
        help="Dose assigned to the reference peak voxel after normalization.",
    )
    parser.add_argument(
        "--target-depth-cm",
        type=float,
        default=5.0,
        help="Depth sampled for the 1D profile.",
    )
    parser.add_argument(
        "--profile-range-mm",
        type=float,
        default=30.0,
        help="Half-range of the lateral profile window around x=0.",
    )
    parser.add_argument(
        "--uniform-dose-floor-fraction",
        type=float,
        default=0.015,
        help="Uniform dose floor added everywhere as a fraction of the prescribed peak dose.",
    )
    parser.add_argument("--alpha", type=float, default=0.10, help="LQ alpha in Gy^-1.")
    parser.add_argument("--beta", type=float, default=0.05, help="LQ beta in Gy^-2.")
    parser.add_argument("--pde-steps", type=int, default=400, help="Number of explicit PDE time steps.")
    parser.add_argument("--pde-dt", type=float, default=0.15, help="Explicit Euler time step.")
    parser.add_argument("--dpi", type=int, default=250, help="Figure DPI.")
    return parser.parse_args()


def normalize_single_beam(
    single_beam: np.ndarray,
    z_cm: np.ndarray,
    prescribed_peak_dose_gy: float,
) -> Tuple[np.ndarray, Dict[str, float]]:
    ref_idx = choose_reference_depth(
        single_beam,
        z_cm,
        argparse.Namespace(reference_mode="dmax", reference_depth_cm=0.0),
    )
    ref_peak_raw, ref_peak_idx_xy = reference_peak_value(single_beam, ref_idx, center_window_bins=5)
    scale = float(prescribed_peak_dose_gy) / ref_peak_raw
    return single_beam * scale, {
        "reference_depth_cm": float(z_cm[ref_idx]),
        "reference_peak_raw_gy": float(ref_peak_raw),
        "scale_factor": float(scale),
        "reference_peak_index_x": int(ref_peak_idx_xy[0]),
        "reference_peak_index_y": int(ref_peak_idx_xy[1]),
    }


def build_pitch_lattice(
    normalized_single: np.ndarray,
    *,
    pitch_mm: float,
    dx_cm: float,
    dy_cm: float,
    prescribed_peak_dose_gy: float,
    uniform_dose_floor_fraction: float,
    n_beams_x: int,
    n_beams_y: int,
    x_shift_fraction_of_pitch: float,
) -> Tuple[np.ndarray, np.ndarray, int, int]:
    pitch_bins_x = int(round((float(pitch_mm) / 10.0) / dx_cm))
    pitch_bins_y = int(round((float(pitch_mm) / 10.0) / dy_cm))
    x_shift_bins = int(round(float(x_shift_fraction_of_pitch) * float(pitch_bins_x)))
    lattice_dose, _, _, _ = build_lattice_with_x_shift(
        normalized_single,
        pitch_bins_x=pitch_bins_x,
        pitch_bins_y=pitch_bins_y,
        n_beams_x=int(n_beams_x),
        n_beams_y=int(n_beams_y),
        x_shift_bins=int(x_shift_bins),
    )
    lattice_dose = lattice_dose.astype(np.float32)
    uniform_floor_gy = float(uniform_dose_floor_fraction) * float(prescribed_peak_dose_gy)
    if uniform_floor_gy > 0.0:
        lattice_dose = lattice_dose + np.float32(uniform_floor_gy)
    return lattice_dose, centered_axis_cm(lattice_dose.shape[0], dx_cm), lattice_dose.shape[0] // 2, lattice_dose.shape[1] // 2


def extract_profile_window(
    x_cm: np.ndarray,
    profile: np.ndarray,
    profile_range_mm: float,
) -> Tuple[np.ndarray, np.ndarray]:
    x_mm = x_cm * 10.0
    mask = np.abs(x_mm) <= float(profile_range_mm)
    return x_mm[mask], profile[mask]


def write_csv(rows: List[Dict[str, object]], out_file: Path) -> None:
    if not rows:
        raise ValueError("No rows supplied for CSV output.")
    with out_file.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_asymmetry_profile(
    x_mm: np.ndarray,
    lq_profile: np.ndarray,
    phase3_profile: np.ndarray,
    phase4_profile: np.ndarray,
    *,
    sampled_depth_cm: float,
    vessel_radius_mm: float,
    hypoxic_center_x_mm: float,
    hypoxic_radius_mm: float,
    out_file: Path,
    dpi: int,
) -> None:
    fig, ax = plt.subplots(figsize=(10.2, 6.2), constrained_layout=True)
    ax.plot(x_mm, lq_profile, color="#111111", linestyle="--", linewidth=2.5, label="Standard LQ")
    ax.plot(x_mm, phase3_profile, color="#1f77b4", linewidth=2.4, label="Phase 3 (symmetric + vessel)")
    ax.plot(x_mm, phase4_profile, color="#d62728", linewidth=2.4, label="Phase 4 (right-peak hypoxia + vessel)")
    ax.axvspan(-float(vessel_radius_mm), float(vessel_radius_mm), color="#76b7b2", alpha=0.18, label="Central vessel")
    ax.axvspan(
        float(hypoxic_center_x_mm) - float(hypoxic_radius_mm),
        float(hypoxic_center_x_mm) + float(hypoxic_radius_mm),
        color="#f39c12",
        alpha=0.12,
        label="Right-peak hypoxia",
    )
    ax.set_xlim(float(x_mm[0]), float(x_mm[-1]))
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("Distance x (mm)")
    ax.set_ylabel("Cell survival")
    ax.set_title("Figure 9: 1D asymmetry profile at 5 cm depth (pitch = 40 mm)")
    ax.grid(alpha=0.25)
    ax.legend(loc="lower left")
    ax.text(
        0.98,
        0.02,
        (
            f"Sampled depth = {sampled_depth_cm:.3f} cm\n"
            f"Window = [{x_mm[0]:.0f}, {x_mm[-1]:.0f}] mm\n"
            f"Vessel at x = 0 mm, r = {vessel_radius_mm:.1f} mm\n"
            f"Hypoxic center = {hypoxic_center_x_mm:.1f} mm, r = {hypoxic_radius_mm:.1f} mm"
        ),
        transform=ax.transAxes,
        fontsize=9,
        va="bottom",
        ha="right",
        bbox={"facecolor": "white", "alpha": 0.88, "edgecolor": "#b5b5b5"},
    )
    fig.savefig(out_file, dpi=dpi)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    phase3_summary = json.loads(args.phase3_summary.read_text(encoding="utf-8"))
    phase4_summary = json.loads(args.phase4_summary.read_text(encoding="utf-8"))

    scaling_factor = float(phase4_summary["scaling_factor"])
    x_shift_fraction = float(phase4_summary["lattice_geometry"]["x_shift_fraction_of_pitch"])
    vessel_radius_mm = float(phase4_summary["vessel_model"]["radius_mm"])
    vessel_center_offset_x_mm = float(phase4_summary["vessel_model"]["center_offset_mm"][0])
    phase3_weights = [float(value) for value in phase3_summary["multispecies_model"]["species_weights"]]
    phase4_weights = [float(value) for value in phase4_summary["multispecies_model"]["species_weights"]]

    phase4_emission = phase4_summary["emission_model"]
    hypoxic_center_offset_mm = tuple(
        float(value)
        for value in phase4_emission.get(
            "hypoxic_center_offset_mm",
            phase4_emission["tumor_center_offset_mm"],
        )
    )
    hypoxic_center_x_mm = float(hypoxic_center_offset_mm[0])
    hypoxic_radius_mm = float(phase4_emission["hypoxic_radius_mm"])

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
    voxel_size_mm = (dx_cm * 10.0, dy_cm * 10.0, dz_cm * 10.0)

    lattice_dose, x_cm, center_x_idx, center_y_idx = build_pitch_lattice(
        normalized_single,
        pitch_mm=float(args.pitch_mm),
        dx_cm=dx_cm,
        dy_cm=dy_cm,
        prescribed_peak_dose_gy=float(args.prescribed_peak_dose_gy),
        uniform_dose_floor_fraction=float(args.uniform_dose_floor_fraction),
        n_beams_x=int(args.n_beams_x),
        n_beams_y=int(args.n_beams_y),
        x_shift_fraction_of_pitch=float(x_shift_fraction),
    )
    z_cm = depth_axis_cm(lattice_dose.shape[2], dz_cm)
    target_idx = nearest_index(z_cm, float(args.target_depth_cm))
    sampled_depth_cm = float(z_cm[target_idx])

    survival_lq = np.exp(-float(args.alpha) * lattice_dose - float(args.beta) * lattice_dose**2).astype(np.float32)
    uptake_tensor, _ = build_cylindrical_uptake_tensor(
        lattice_dose.shape,
        voxel_size_mm,
        num_species=2,
        vessel_radius_mm=vessel_radius_mm,
        vessel_center_offset_mm=(vessel_center_offset_x_mm, float(phase4_summary["vessel_model"]["center_offset_mm"][1])),
        uptake_rates_in_vessel=phase4_summary["vessel_model"]["uptake_rates_in_vessel"],
    )

    phase3_start = time.time()
    phase3_tensor = solve_multispecies_pde_3d(
        lattice_dose,
        voxel_size_mm,
        steps=int(phase3_summary["multispecies_model"]["steps"]),
        dt=float(phase3_summary["multispecies_model"]["dt"]),
        diffusion_coeffs=phase3_summary["multispecies_model"]["diffusion_coeffs"],
        decay_coeffs=phase3_summary["multispecies_model"]["decay_coeffs"],
        emission_emax=phase3_summary["multispecies_model"]["emission_emax"],
        emission_gamma_per_gy=float(phase3_summary["multispecies_model"]["emission_gamma_per_gy"]),
        state_dependent_emission=False,
        uptake_tensor=uptake_tensor,
        progress_interval=50,
        verbose=True,
    )
    phase3_runtime_sec = time.time() - phase3_start

    phase4_start = time.time()
    phase4_tensor = solve_multispecies_pde_3d(
        lattice_dose,
        voxel_size_mm,
        steps=int(phase4_summary["multispecies_model"]["steps"]),
        dt=float(phase4_summary["multispecies_model"]["dt"]),
        diffusion_coeffs=phase4_summary["multispecies_model"]["diffusion_coeffs"],
        decay_coeffs=phase4_summary["multispecies_model"]["decay_coeffs"],
        emission_emax=phase4_summary["multispecies_model"]["emission_emax"],
        emission_gamma_per_gy=float(phase4_summary["emission_model"]["emission_gamma_per_gy"]),
        state_dependent_emission=True,
        tumor_radius_mm=float(phase4_summary["emission_model"]["tumor_radius_mm"]),
        tumor_center_offset_mm=tuple(float(v) for v in phase4_summary["emission_model"]["tumor_center_offset_mm"]),
        tumor_cytokine_multiplier=float(phase4_summary["emission_model"]["tumor_cytokine_multiplier"]),
        hypoxic_radius_mm=float(phase4_summary["emission_model"]["hypoxic_radius_mm"]),
        hypoxic_center_offset_mm=hypoxic_center_offset_mm,
        hypoxic_ros_scale=float(phase4_summary["emission_model"]["hypoxic_ros_scale"]),
        hypoxic_cytokine_multiplier=float(phase4_summary["emission_model"]["hypoxic_cytokine_multiplier"]),
        uptake_tensor=uptake_tensor,
        progress_interval=50,
        verbose=True,
    )
    phase4_runtime_sec = time.time() - phase4_start

    phase3_total_toxicity = (
        float(phase3_weights[0]) * phase3_tensor[0]
        + float(phase3_weights[1]) * phase3_tensor[1]
    )
    phase4_total_toxicity = (
        float(phase4_weights[0]) * phase4_tensor[0]
        + float(phase4_weights[1]) * phase4_tensor[1]
    )
    survival_phase3 = (survival_lq * np.exp(-phase3_total_toxicity * float(scaling_factor))).astype(np.float32)
    survival_phase4 = (survival_lq * np.exp(-phase4_total_toxicity * float(scaling_factor))).astype(np.float32)

    lq_profile_full = survival_lq[:, center_y_idx, target_idx]
    phase3_profile_full = survival_phase3[:, center_y_idx, target_idx]
    phase4_profile_full = survival_phase4[:, center_y_idx, target_idx]
    x_mm, lq_profile = extract_profile_window(x_cm, lq_profile_full, float(args.profile_range_mm))
    _, phase3_profile = extract_profile_window(x_cm, phase3_profile_full, float(args.profile_range_mm))
    _, phase4_profile = extract_profile_window(x_cm, phase4_profile_full, float(args.profile_range_mm))

    figure_file = args.outdir / "figure9_phase4_asymmetry_profile_pitch40.png"
    plot_asymmetry_profile(
        x_mm=x_mm,
        lq_profile=lq_profile,
        phase3_profile=phase3_profile,
        phase4_profile=phase4_profile,
        sampled_depth_cm=sampled_depth_cm,
        vessel_radius_mm=vessel_radius_mm,
        hypoxic_center_x_mm=hypoxic_center_x_mm,
        hypoxic_radius_mm=hypoxic_radius_mm,
        out_file=figure_file,
        dpi=int(args.dpi),
    )

    rows: List[Dict[str, object]] = []
    for idx, x_value_mm in enumerate(x_mm):
        rows.append(
            {
                "x_mm": float(x_value_mm),
                "sampled_depth_cm": float(sampled_depth_cm),
                "survival_lq": float(lq_profile[idx]),
                "survival_phase3_symmetric": float(phase3_profile[idx]),
                "survival_phase4_asymmetric": float(phase4_profile[idx]),
                "delta_phase4_minus_phase3": float(phase4_profile[idx] - phase3_profile[idx]),
            }
        )

    csv_file = args.outdir / "phase4_asymmetry_profile_pitch40.csv"
    summary_file = args.outdir / "phase4_asymmetry_profile_summary.json"
    write_csv(rows, csv_file)
    summary = {
        "input_csv": str(args.csv),
        "phase3_summary": str(args.phase3_summary),
        "phase4_summary": str(args.phase4_summary),
        "outdir": str(args.outdir),
        "figure": str(figure_file),
        "csv": str(csv_file),
        "pitch_mm": float(args.pitch_mm),
        "sampled_depth_cm": float(sampled_depth_cm),
        "profile_window_mm": [-float(args.profile_range_mm), float(args.profile_range_mm)],
        "x_shift_fraction_of_pitch": float(x_shift_fraction),
        "vessel_center_x_mm": float(vessel_center_offset_x_mm),
        "vessel_radius_mm": float(vessel_radius_mm),
        "hypoxic_center_x_mm": float(hypoxic_center_x_mm),
        "hypoxic_radius_mm": float(hypoxic_radius_mm),
        "scaling_factor": float(scaling_factor),
        "phase3_species_weights": phase3_weights,
        "phase4_species_weights": phase4_weights,
        "phase3_runtime_sec": float(phase3_runtime_sec),
        "phase4_runtime_sec": float(phase4_runtime_sec),
        "normalization": norm_meta,
    }
    summary_file.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Asymmetry profile summary: {summary_file}")
    print(f"Asymmetry profile figure: {figure_file}")
    print(f"Asymmetry profile CSV: {csv_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
