#!/usr/bin/env python
"""Generate Phase 3 bystander-model parameter sweep figures."""

from __future__ import annotations

import argparse
import csv
import gc
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from scipy.signal import fftconvolve

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception as exc:  # pragma: no cover
    raise RuntimeError("matplotlib is required for figure generation") from exc

from analyze_250mev_sfrt_plan import (
    build_lattice,
    choose_reference_depth,
    depth_axis_cm,
    nearest_index,
    reference_peak_value,
)
from analyze_topas_outputs import load_topas_grid


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description=(
            "Sweep bystander-model lambda and amplitude settings and generate "
            "pitch-vs-valley-survival sensitivity plots."
        )
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=root / "runs" / "linac_6mv_direct_photon_sfrt" / "case" / "dosedata.csv",
        help="TOPAS dose CSV for the direct-photon kernel.",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=root / "runs" / "linac_6mv_direct_photon_sfrt" / "analysis_phase3_bystander_sweeps",
        help="Directory for Phase 3 sweep figures and summaries.",
    )
    parser.add_argument(
        "--pitches-mm",
        nargs="+",
        type=float,
        default=[20.0, 30.0, 40.0],
        help="Lattice pitches to compare.",
    )
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
        help="Depth where valley survival is sampled for the sweep figures.",
    )
    parser.add_argument(
        "--emission-threshold-gy",
        type=float,
        default=5.0,
        help="Dose threshold above which voxels emit bystander signals.",
    )
    parser.add_argument(
        "--continuous-emission",
        action="store_true",
        help="Use a saturated continuous emission model instead of a binary threshold.",
    )
    parser.add_argument(
        "--emission-emax",
        type=float,
        default=1.0,
        help="Maximum emission strength used by the continuous emission model.",
    )
    parser.add_argument(
        "--emission-gamma-per-gy",
        type=float,
        default=0.35,
        help="Dose-response constant gamma for E(D) = Emax * (1 - exp(-gamma * D)).",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.10,
        help="LQ alpha in Gy^-1.",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=0.05,
        help="LQ beta in Gy^-2.",
    )
    parser.add_argument(
        "--lambda-values-mm",
        nargs="+",
        type=float,
        default=[1.0, 2.0, 3.0, 5.0, 10.0],
        help="Lambda values for the spatial-reach sweep.",
    )
    parser.add_argument(
        "--lambda-sweep-amplitude",
        type=float,
        default=1.0e-4,
        help="Fixed amplitude A used during the lambda sweep.",
    )
    parser.add_argument(
        "--amplitude-values",
        nargs="+",
        type=float,
        default=[1.0e-5, 5.0e-5, 1.0e-4, 5.0e-4],
        help="Amplitude values for the toxicity sweep.",
    )
    parser.add_argument(
        "--amplitude-sweep-lambda-mm",
        type=float,
        default=3.0,
        help="Fixed lambda used during the amplitude sweep.",
    )
    parser.add_argument(
        "--kernel-radius-lambda",
        type=float,
        default=4.0,
        help="Kernel truncation radius expressed in multiples of lambda.",
    )
    parser.add_argument(
        "--uniform-dose-floor-fraction",
        type=float,
        default=0.0,
        help="Uniform dose floor added everywhere as a fraction of the prescribed peak dose.",
    )
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


def build_diffusion_kernel(
    dx_cm: float,
    dy_cm: float,
    dz_cm: float,
    diffusion_length_mm: float,
    signal_strength: float,
    kernel_radius_lambda: float,
) -> np.ndarray:
    lambda_cm = float(diffusion_length_mm) / 10.0
    radius_cm = max(float(kernel_radius_lambda), 1.0) * lambda_cm
    radius_x = max(1, int(np.ceil(radius_cm / dx_cm)))
    radius_y = max(1, int(np.ceil(radius_cm / dy_cm)))
    radius_z = max(1, int(np.ceil(radius_cm / dz_cm)))

    xs = np.arange(-radius_x, radius_x + 1, dtype=np.float32) * dx_cm
    ys = np.arange(-radius_y, radius_y + 1, dtype=np.float32) * dy_cm
    zs = np.arange(-radius_z, radius_z + 1, dtype=np.float32) * dz_cm
    grid_x, grid_y, grid_z = np.meshgrid(xs, ys, zs, indexing="ij")
    radial_distance = np.sqrt(grid_x**2 + grid_y**2 + grid_z**2)
    kernel = float(signal_strength) * np.exp(-radial_distance / lambda_cm)
    return kernel.astype(np.float32)


def build_emission_map(
    lattice_dose: np.ndarray,
    emission_threshold_gy: float,
    continuous_emission: bool,
    emission_emax: float,
    emission_gamma_per_gy: float,
) -> np.ndarray:
    if continuous_emission:
        return (float(emission_emax) * (1.0 - np.exp(-float(emission_gamma_per_gy) * lattice_dose))).astype(np.float32)
    return (lattice_dose > float(emission_threshold_gy)).astype(np.float32)


def sci_label(value: float) -> str:
    mantissa, exponent = f"{float(value):.1e}".split("e")
    mantissa = mantissa.rstrip("0").rstrip(".")
    exponent = int(exponent)
    return f"{mantissa}e{exponent}"


def plot_sweep_figure(
    pitches_mm: List[float],
    standard_values: List[float],
    swept_lines: List[Tuple[str, List[float], str, str]],
    title: str,
    settings_text: str,
    out_file: Path,
    dpi: int,
) -> None:
    fig, ax = plt.subplots(figsize=(10.2, 6.0), constrained_layout=True)
    ax.plot(
        pitches_mm,
        standard_values,
        color="#111111",
        linestyle="--",
        linewidth=2.5,
        marker="o",
        label="Standard LQ",
    )
    for label, values, color, marker in swept_lines:
        ax.plot(
            pitches_mm,
            values,
            color=color,
            linewidth=2.2,
            marker=marker,
            label=label,
        )
    ax.set_xticks(pitches_mm)
    ax.set_xlabel("Lattice pitch (mm)")
    ax.set_ylabel("Valley survival at 5 cm")
    ax.set_ylim(0.0, 1.02)
    ax.grid(alpha=0.25)
    ax.set_title(title)
    ax.legend(loc="lower right")
    ax.text(
        0.02,
        0.02,
        settings_text,
        transform=ax.transAxes,
        fontsize=9,
        va="bottom",
        ha="left",
        bbox={"facecolor": "white", "alpha": 0.88, "edgecolor": "#b5b5b5"},
    )
    fig.savefig(out_file, dpi=dpi)
    plt.close(fig)


def build_pitch_case(
    normalized_single: np.ndarray,
    dx_cm: float,
    dy_cm: float,
    dz_cm: float,
    pitch_mm: float,
    n_beams_x: int,
    n_beams_y: int,
    target_depth_cm: float,
    emission_threshold_gy: float,
    continuous_emission: bool,
    emission_emax: float,
    emission_gamma_per_gy: float,
    uniform_dose_floor_fraction: float,
    prescribed_peak_dose_gy: float,
    alpha: float,
    beta: float,
) -> Dict[str, object]:
    pitch_bins_x = int(round((float(pitch_mm) / 10.0) / dx_cm))
    pitch_bins_y = int(round((float(pitch_mm) / 10.0) / dy_cm))
    lattice_dose, offsets_x, _ = build_lattice(
        normalized_single,
        pitch_bins_x=pitch_bins_x,
        pitch_bins_y=pitch_bins_y,
        n_beams_x=int(n_beams_x),
        n_beams_y=int(n_beams_y),
    )
    lattice_dose = lattice_dose.astype(np.float32)
    uniform_floor_gy = float(uniform_dose_floor_fraction) * float(prescribed_peak_dose_gy)
    if uniform_floor_gy > 0.0:
        lattice_dose = lattice_dose + np.float32(uniform_floor_gy)
    emission_map = build_emission_map(
        lattice_dose=lattice_dose,
        emission_threshold_gy=float(emission_threshold_gy),
        continuous_emission=bool(continuous_emission),
        emission_emax=float(emission_emax),
        emission_gamma_per_gy=float(emission_gamma_per_gy),
    )
    survival_lq = np.exp(-float(alpha) * lattice_dose - float(beta) * lattice_dose**2).astype(np.float32)

    z_cm = depth_axis_cm(lattice_dose.shape[2], dz_cm)
    target_idx = nearest_index(z_cm, float(target_depth_cm))
    center_x_idx = lattice_dose.shape[0] // 2
    center_y_idx = lattice_dose.shape[1] // 2
    x_centers_idx = [center_x_idx + off for off in offsets_x]
    peak_x_idx = x_centers_idx[len(x_centers_idx) // 2]
    valley_x_idx = (x_centers_idx[len(x_centers_idx) // 2] + x_centers_idx[(len(x_centers_idx) // 2) + 1]) // 2

    return {
        "pitch_mm": float(pitch_mm),
        "lattice_dose": lattice_dose,
        "emission_map": emission_map,
        "survival_lq": survival_lq,
        "sampled_depth_cm": float(z_cm[target_idx]),
        "target_depth_idx": int(target_idx),
        "center_y_idx": int(center_y_idx),
        "peak_x_idx": int(peak_x_idx),
        "valley_x_idx": int(valley_x_idx),
        "valley_dose_gy": float(lattice_dose[valley_x_idx, center_y_idx, target_idx]),
        "valley_survival_lq": float(survival_lq[valley_x_idx, center_y_idx, target_idx]),
        "uniform_dose_floor_gy": float(uniform_floor_gy),
    }


def main() -> int:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

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
    print("Loaded and normalized the direct-photon kernel.", flush=True)

    pitches_mm = [float(v) for v in args.pitches_mm]
    lambda_values_mm = [float(v) for v in args.lambda_values_mm]
    amplitude_values = [float(v) for v in args.amplitude_values]

    lambda_rows: List[Dict[str, object]] = []
    amplitude_rows: List[Dict[str, object]] = []
    baseline_rows: List[Dict[str, object]] = []
    standard_lq_by_pitch: Dict[float, float] = {}
    sampled_depth_cm = None

    lambda_results: Dict[float, Dict[float, float]] = {value: {} for value in lambda_values_mm}
    amplitude_results: Dict[float, Dict[float, float]] = {value: {} for value in amplitude_values}

    for pitch_mm in pitches_mm:
        print(f"Processing pitch {pitch_mm:g} mm...", flush=True)
        case = build_pitch_case(
            normalized_single=normalized_single,
            dx_cm=dx_cm,
            dy_cm=dy_cm,
            dz_cm=dz_cm,
            pitch_mm=pitch_mm,
            n_beams_x=int(args.n_beams_x),
            n_beams_y=int(args.n_beams_y),
            target_depth_cm=float(args.target_depth_cm),
            emission_threshold_gy=float(args.emission_threshold_gy),
            continuous_emission=bool(args.continuous_emission),
            emission_emax=float(args.emission_emax),
            emission_gamma_per_gy=float(args.emission_gamma_per_gy),
            uniform_dose_floor_fraction=float(args.uniform_dose_floor_fraction),
            prescribed_peak_dose_gy=float(args.prescribed_peak_dose_gy),
            alpha=float(args.alpha),
            beta=float(args.beta),
        )

        standard_lq_by_pitch[pitch_mm] = float(case["valley_survival_lq"])
        sampled_depth_cm = float(case["sampled_depth_cm"])
        baseline_rows.append(
            {
                "pitch_mm": float(pitch_mm),
                "sampled_depth_cm": float(case["sampled_depth_cm"]),
                "valley_dose_gy": float(case["valley_dose_gy"]),
                "valley_survival_lq": float(case["valley_survival_lq"]),
            }
        )

        for lambda_mm in lambda_values_mm:
            print(f"  Lambda sweep: pitch {pitch_mm:g} mm, lambda {lambda_mm:g} mm", flush=True)
            kernel = build_diffusion_kernel(
                dx_cm=dx_cm,
                dy_cm=dy_cm,
                dz_cm=dz_cm,
                diffusion_length_mm=lambda_mm,
                signal_strength=float(args.lambda_sweep_amplitude),
                kernel_radius_lambda=float(args.kernel_radius_lambda),
            )
            bystander_penalty = np.maximum(fftconvolve(case["emission_map"], kernel, mode="same"), 0.0).astype(np.float32)
            valley_penalty = float(
                bystander_penalty[
                    int(case["valley_x_idx"]),
                    int(case["center_y_idx"]),
                    int(case["target_depth_idx"]),
                ]
            )
            valley_survival_total = float(case["valley_survival_lq"]) * float(np.exp(-valley_penalty))
            lambda_results[lambda_mm][pitch_mm] = valley_survival_total
            lambda_rows.append(
                {
                    "pitch_mm": float(pitch_mm),
                    "lambda_mm": float(lambda_mm),
                    "signal_strength": float(args.lambda_sweep_amplitude),
                    "sampled_depth_cm": float(case["sampled_depth_cm"]),
                    "valley_dose_gy": float(case["valley_dose_gy"]),
                    "uniform_dose_floor_gy": float(case["uniform_dose_floor_gy"]),
                    "valley_survival_lq": float(case["valley_survival_lq"]),
                    "valley_bystander_penalty": valley_penalty,
                    "valley_survival_total": valley_survival_total,
                    "valley_survival_loss": float(case["valley_survival_lq"]) - valley_survival_total,
                }
            )
            del bystander_penalty

        for amplitude in amplitude_values:
            print(
                f"  Amplitude sweep: pitch {pitch_mm:g} mm, A {sci_label(amplitude)}",
                flush=True,
            )
            kernel = build_diffusion_kernel(
                dx_cm=dx_cm,
                dy_cm=dy_cm,
                dz_cm=dz_cm,
                diffusion_length_mm=float(args.amplitude_sweep_lambda_mm),
                signal_strength=amplitude,
                kernel_radius_lambda=float(args.kernel_radius_lambda),
            )
            bystander_penalty = np.maximum(fftconvolve(case["emission_map"], kernel, mode="same"), 0.0).astype(np.float32)
            valley_penalty = float(
                bystander_penalty[
                    int(case["valley_x_idx"]),
                    int(case["center_y_idx"]),
                    int(case["target_depth_idx"]),
                ]
            )
            valley_survival_total = float(case["valley_survival_lq"]) * float(np.exp(-valley_penalty))
            amplitude_results[amplitude][pitch_mm] = valley_survival_total
            amplitude_rows.append(
                {
                    "pitch_mm": float(pitch_mm),
                    "lambda_mm": float(args.amplitude_sweep_lambda_mm),
                    "signal_strength": float(amplitude),
                    "sampled_depth_cm": float(case["sampled_depth_cm"]),
                    "valley_dose_gy": float(case["valley_dose_gy"]),
                    "uniform_dose_floor_gy": float(case["uniform_dose_floor_gy"]),
                    "valley_survival_lq": float(case["valley_survival_lq"]),
                    "valley_bystander_penalty": valley_penalty,
                    "valley_survival_total": valley_survival_total,
                    "valley_survival_loss": float(case["valley_survival_lq"]) - valley_survival_total,
                }
            )
            del bystander_penalty

        del case["lattice_dose"]
        del case["emission_map"]
        del case["survival_lq"]
        gc.collect()

    standard_values = [standard_lq_by_pitch[pitch] for pitch in pitches_mm]

    lambda_colors = ["#176087", "#2a9d8f", "#8ab17d", "#e9c46a", "#d65f4a"]
    lambda_markers = ["o", "s", "^", "D", "P"]
    lambda_lines = [
        (
            f"Multi-effect, lambda = {lambda_mm:g} mm",
            [lambda_results[lambda_mm][pitch] for pitch in pitches_mm],
            lambda_colors[index % len(lambda_colors)],
            lambda_markers[index % len(lambda_markers)],
        )
        for index, lambda_mm in enumerate(lambda_values_mm)
    ]
    lambda_settings_lines = [
        "Sweep: lambda",
        f"A fixed = {sci_label(args.lambda_sweep_amplitude)}",
        f"lambda values = {', '.join(f'{value:g}' for value in lambda_values_mm)} mm",
    ]
    if bool(args.continuous_emission):
        lambda_settings_lines.append(
            f"Emission: E(D)=Emax*(1-exp(-gamma*D)), Emax={float(args.emission_emax):.2f}, "
            f"gamma={float(args.emission_gamma_per_gy):.2f} Gy^-1"
        )
    else:
        lambda_settings_lines.append(f"Emission threshold: D > {float(args.emission_threshold_gy):.1f} Gy")
    lambda_settings_lines.extend(
        [
            f"Transmission floor = {100.0 * float(args.uniform_dose_floor_fraction):.1f}% of peak",
            f"alpha = {float(args.alpha):.2f}, beta = {float(args.beta):.2f}",
            f"Sampled depth = {float(sampled_depth_cm):.3f} cm",
            f"Pitches = {', '.join(f'{pitch:g}' for pitch in pitches_mm)} mm",
        ]
    )
    lambda_settings = "\n".join(lambda_settings_lines)
    lambda_figure = args.outdir / "figure7_lambda_sweep_valley_survival.png"
    print(f"Writing {lambda_figure.name}...", flush=True)
    plot_sweep_figure(
        pitches_mm=pitches_mm,
        standard_values=standard_values,
        swept_lines=lambda_lines,
        title="Figure 7: Valley survival sensitivity to bystander diffusion length",
        settings_text=lambda_settings,
        out_file=lambda_figure,
        dpi=int(args.dpi),
    )

    amplitude_colors = ["#4c1d95", "#7c3aed", "#c026d3", "#ef4444"]
    amplitude_markers = ["o", "s", "^", "D"]
    amplitude_lines = [
        (
            f"Multi-effect, A = {sci_label(amplitude)}",
            [amplitude_results[amplitude][pitch] for pitch in pitches_mm],
            amplitude_colors[index % len(amplitude_colors)],
            amplitude_markers[index % len(amplitude_markers)],
        )
        for index, amplitude in enumerate(amplitude_values)
    ]
    amplitude_settings_lines = [
        "Sweep: amplitude",
        f"lambda fixed = {float(args.amplitude_sweep_lambda_mm):.1f} mm",
        f"A values = {', '.join(sci_label(value) for value in amplitude_values)}",
    ]
    if bool(args.continuous_emission):
        amplitude_settings_lines.append(
            f"Emission: E(D)=Emax*(1-exp(-gamma*D)), Emax={float(args.emission_emax):.2f}, "
            f"gamma={float(args.emission_gamma_per_gy):.2f} Gy^-1"
        )
    else:
        amplitude_settings_lines.append(f"Emission threshold: D > {float(args.emission_threshold_gy):.1f} Gy")
    amplitude_settings_lines.extend(
        [
            f"Transmission floor = {100.0 * float(args.uniform_dose_floor_fraction):.1f}% of peak",
            f"alpha = {float(args.alpha):.2f}, beta = {float(args.beta):.2f}",
            f"Sampled depth = {float(sampled_depth_cm):.3f} cm",
            f"Pitches = {', '.join(f'{pitch:g}' for pitch in pitches_mm)} mm",
        ]
    )
    amplitude_settings = "\n".join(amplitude_settings_lines)
    amplitude_figure = args.outdir / "figure8_amplitude_sweep_valley_survival.png"
    print(f"Writing {amplitude_figure.name}...", flush=True)
    plot_sweep_figure(
        pitches_mm=pitches_mm,
        standard_values=standard_values,
        swept_lines=amplitude_lines,
        title="Figure 8: Valley survival sensitivity to bystander signal strength",
        settings_text=amplitude_settings,
        out_file=amplitude_figure,
        dpi=int(args.dpi),
    )

    def write_csv(rows: List[Dict[str, object]], out_file: Path) -> None:
        if not rows:
            raise ValueError("No rows supplied for CSV output.")
        with out_file.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    baseline_csv = args.outdir / "phase3_standard_lq_baseline.csv"
    lambda_csv = args.outdir / "phase3_lambda_sweep.csv"
    amplitude_csv = args.outdir / "phase3_amplitude_sweep.csv"
    write_csv(baseline_rows, baseline_csv)
    write_csv(lambda_rows, lambda_csv)
    write_csv(amplitude_rows, amplitude_csv)

    summary = {
        "input_csv": str(args.csv),
        "outdir": str(args.outdir),
        "normalization": norm_meta,
        "baseline": {
            "target_depth_cm_requested": float(args.target_depth_cm),
            "target_depth_cm_sampled": float(sampled_depth_cm),
            "rows_csv": str(baseline_csv),
            "uniform_dose_floor_fraction_of_peak": float(args.uniform_dose_floor_fraction),
        },
        "emission_model": {
            "type": "continuous_saturated" if bool(args.continuous_emission) else "binary_threshold",
            "emission_threshold_gy": float(args.emission_threshold_gy),
            "emission_emax": float(args.emission_emax),
            "emission_gamma_per_gy": float(args.emission_gamma_per_gy),
        },
        "lambda_sweep": {
            "fixed_signal_strength": float(args.lambda_sweep_amplitude),
            "lambda_values_mm": [float(v) for v in lambda_values_mm],
            "rows_csv": str(lambda_csv),
            "figure": str(lambda_figure),
        },
        "amplitude_sweep": {
            "fixed_lambda_mm": float(args.amplitude_sweep_lambda_mm),
            "signal_strength_values": [float(v) for v in amplitude_values],
            "rows_csv": str(amplitude_csv),
            "figure": str(amplitude_figure),
        },
    }
    summary_file = args.outdir / "phase3_sweep_summary.json"
    summary_file.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Phase 3 summary: {summary_file}")
    print(f"Lambda sweep figure: {lambda_figure}")
    print(f"Amplitude sweep figure: {amplitude_figure}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
