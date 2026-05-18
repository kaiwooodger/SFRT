#!/usr/bin/env python
"""Run a representative polyenergetic 6 MV photon SFRT surrogate and upgraded biology pipeline."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple

from build_asymmetric_sweep import (
    PHYSICS_PROFILES,
    build_topas_env,
    format_physics_modules,
    has_nonempty_output,
    write_text_with_retries,
)


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description=(
            "Create a representative polyenergetic 6 MV photon baseline in water, run TOPAS, "
            "and post-process the dose cube with the upgraded clinical-surrogate biology pipeline."
        )
    )
    parser.add_argument(
        "--template",
        type=Path,
        default=root / "topas" / "linac_6mv_polyenergetic_direct_photon_template.txt",
        help="TOPAS template used to build the polyenergetic case deck.",
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
        default=root / "runs" / "linac_6mv_polyenergetic_clinical_sfrt",
        help="Output directory for the TOPAS case and upgraded analysis artifacts.",
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
        choices=sorted(PHYSICS_PROFILES),
        default="em_opt4_only",
        help="Named TOPAS modular physics profile.",
    )
    parser.add_argument("--histories", type=int, default=1_000_000, help="TOPAS histories.")
    parser.add_argument("--threads", type=int, default=8, help="TOPAS threads.")
    parser.add_argument("--seed", type=int, default=33, help="TOPAS RNG seed.")
    parser.add_argument(
        "--nominal-energy-mv",
        type=float,
        default=6.0,
        help="Nominal clinical beam quality label recorded in metadata.",
    )
    parser.add_argument(
        "--beam-position-cutoff-mm",
        type=float,
        default=5.0,
        help="Flat elliptical cutoff radius in mm, giving a 10 mm diameter field.",
    )
    parser.add_argument(
        "--source-z-cm",
        type=float,
        default=-0.1,
        help="Source plane z position relative to the world/phantom entrance geometry.",
    )
    parser.add_argument("--phantom-size-x-cm", type=float, default=12.0, help="Water phantom size in x.")
    parser.add_argument("--phantom-size-y-cm", type=float, default=12.0, help="Water phantom size in y.")
    parser.add_argument("--phantom-size-z-cm", type=float, default=20.0, help="Water phantom size in z.")
    parser.add_argument("--xbins", type=int, default=121, help="Phantom x bins.")
    parser.add_argument("--ybins", type=int, default=121, help="Phantom y bins.")
    parser.add_argument("--zbins", type=int, default=161, help="Phantom z bins.")
    parser.add_argument(
        "--max-step-phantom-mm",
        type=float,
        default=0.1,
        help="Maximum tracking step in water.",
    )
    parser.add_argument("--cut-gamma-mm", type=float, default=0.01, help="Photon production cut in mm.")
    parser.add_argument("--cut-electron-mm", type=float, default=0.01, help="Electron production cut in mm.")
    parser.add_argument("--cut-positron-mm", type=float, default=0.01, help="Positron production cut in mm.")
    parser.add_argument(
        "--prescribed-peak-dose-gy",
        type=float,
        default=10.0,
        help="Peak dose assigned after normalization.",
    )
    parser.add_argument(
        "--pitches-mm",
        nargs="+",
        type=float,
        default=[20.0, 30.0, 40.0],
        help="Synthetic lattice pitches passed to the upgraded analyses.",
    )
    parser.add_argument("--n-beams-x", type=int, default=7, help="Beam copies along x.")
    parser.add_argument("--n-beams-y", type=int, default=7, help="Beam copies along y.")
    parser.add_argument(
        "--phase2-depths-cm",
        nargs=2,
        type=float,
        default=[3.0, 5.0],
        help="Depths shown alongside dmax in the upgraded Phase 2 figures.",
    )
    parser.add_argument(
        "--target-depth-cm",
        type=float,
        default=5.0,
        help="Depth sampled for the Phase 3 sensitivity plots.",
    )
    parser.add_argument(
        "--uniform-dose-floor-fraction",
        type=float,
        default=0.015,
        help="Uniform surrogate transmission floor as a fraction of the prescribed peak dose.",
    )
    parser.add_argument(
        "--continuous-emission",
        action="store_true",
        default=True,
        help="Use the continuous saturated bystander emission model.",
    )
    parser.add_argument(
        "--emission-threshold-gy",
        type=float,
        default=5.0,
        help="Threshold retained for compatibility when binary emission is used.",
    )
    parser.add_argument("--emission-emax", type=float, default=1.0, help="Maximum emission strength.")
    parser.add_argument(
        "--emission-gamma-per-gy",
        type=float,
        default=0.35,
        help="Continuous emission dose-response constant gamma.",
    )
    parser.add_argument("--phase2-alpha", type=float, default=0.10, help="LQ alpha in Gy^-1.")
    parser.add_argument("--phase2-beta", type=float, default=0.05, help="LQ beta in Gy^-2.")
    parser.add_argument(
        "--phase2-diffusion-length-mm",
        type=float,
        default=10.0,
        help="Diffusion length used for the upgraded Phase 2 figures.",
    )
    parser.add_argument(
        "--phase2-signal-strength",
        type=float,
        default=1.0e-4,
        help="Bystander signal strength used for the upgraded Phase 2 figures.",
    )
    parser.add_argument(
        "--lambda-values-mm",
        nargs="+",
        type=float,
        default=[1.0, 2.0, 3.0, 5.0, 10.0],
        help="Lambda values for the Phase 3 spatial-reach sweep.",
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
        "--high-dose-methodology-note",
        type=str,
        default=(
            "Numerical survival still uses the standard LQ model in the current code path. "
            "For manuscript methodology and future peak-region refinement, LQL or USC should replace "
            "standard LQ in regions exceeding about 10 Gy per fraction."
        ),
        help="Methodology note recorded in the summary for high-dose survival modeling.",
    )
    parser.add_argument(
        "--analyze-only",
        action="store_true",
        help="Skip TOPAS execution and analyze an existing dosedata.csv.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip the TOPAS run when a non-empty dosedata.csv already exists.",
    )
    parser.add_argument("--dpi", type=int, default=250, help="Figure DPI.")
    return parser.parse_args()


def fill_template(template_text: str, replacements: Dict[str, str]) -> str:
    rendered = template_text
    for key, value in replacements.items():
        rendered = rendered.replace(key, value)
    return rendered


def load_spectrum(spectrum_csv: Path) -> Tuple[List[float], List[float]]:
    energies: List[float] = []
    weights: List[float] = []
    with spectrum_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            energies.append(float(row["energy_mev"]))
            weights.append(float(row["weight"]))
    if not energies:
        raise ValueError(f"No spectrum rows found in {spectrum_csv}")
    total = sum(weights)
    if total <= 0.0:
        raise ValueError(f"Spectrum weights in {spectrum_csv} sum to a non-positive value.")
    normalized_weights = [value / total for value in weights]
    return energies, normalized_weights


def render_case_file(args: argparse.Namespace, spectrum_energies: List[float], spectrum_weights: List[float]) -> str:
    template_text = args.template.read_text(encoding="utf-8")
    phantom_hlx_cm = 0.5 * float(args.phantom_size_x_cm)
    phantom_hly_cm = 0.5 * float(args.phantom_size_y_cm)
    phantom_hlz_cm = 0.5 * float(args.phantom_size_z_cm)
    phantom_center_z_cm = phantom_hlz_cm
    world_hlx_cm = max(40.0, float(args.phantom_size_x_cm) + 20.0)
    world_hly_cm = max(40.0, float(args.phantom_size_y_cm) + 20.0)
    world_hlz_cm = max(60.0, phantom_center_z_cm + phantom_hlz_cm + 20.0)
    show_interval = max(1000, int(args.histories) // 10)

    replacements = {
        "__G4_DATA_DIR__": str(Path(args.g4_data_dir).expanduser()),
        "__PHYSICS_MODULES__": format_physics_modules(str(args.physics_profile)),
        "__CUT_GAMMA_MM__": f"{float(args.cut_gamma_mm):.6f}",
        "__CUT_ELECTRON_MM__": f"{float(args.cut_electron_mm):.6f}",
        "__CUT_POSITRON_MM__": f"{float(args.cut_positron_mm):.6f}",
        "__WORLD_HLX_CM__": f"{world_hlx_cm:.6f}",
        "__WORLD_HLY_CM__": f"{world_hly_cm:.6f}",
        "__WORLD_HLZ_CM__": f"{world_hlz_cm:.6f}",
        "__SOURCE_Z_CM__": f"{float(args.source_z_cm):.6f}",
        "__PHANTOM_HLX_CM__": f"{phantom_hlx_cm:.6f}",
        "__PHANTOM_HLY_CM__": f"{phantom_hly_cm:.6f}",
        "__PHANTOM_HLZ_CM__": f"{phantom_hlz_cm:.6f}",
        "__PHANTOM_CENTER_Z_CM__": f"{phantom_center_z_cm:.6f}",
        "__MAX_STEP_PHANTOM_MM__": f"{float(args.max_step_phantom_mm):.6f}",
        "__XBINS__": str(int(args.xbins)),
        "__YBINS__": str(int(args.ybins)),
        "__ZBINS__": str(int(args.zbins)),
        "__OUTPUT_STEM__": "dosedata",
        "__SPECTRUM_COUNT__": str(len(spectrum_energies)),
        "__SPECTRUM_VALUES__": " ".join(f"{value:.6f}" for value in spectrum_energies),
        "__SPECTRUM_WEIGHTS__": " ".join(f"{value:.8f}" for value in spectrum_weights),
        "__BEAM_POSITION_CUTOFF_X_MM__": f"{float(args.beam_position_cutoff_mm):.6f}",
        "__BEAM_POSITION_CUTOFF_Y_MM__": f"{float(args.beam_position_cutoff_mm):.6f}",
        "__N_HISTORIES__": str(int(args.histories)),
        "__N_THREADS__": str(int(args.threads)),
        "__SEED__": str(int(args.seed)),
        "__SHOW_HISTORY_INTERVAL__": str(int(show_interval)),
    }
    return fill_template(template_text, replacements)


def build_case_metadata(
    args: argparse.Namespace,
    case_dir: Path,
    phase2_dir: Path,
    phase3_dir: Path,
    spectrum_energies: List[float],
    spectrum_weights: List[float],
) -> Dict[str, object]:
    phantom_center_z_cm = 0.5 * float(args.phantom_size_z_cm)
    voxel_size_x_mm = 10.0 * float(args.phantom_size_x_cm) / float(args.xbins)
    voxel_size_y_mm = 10.0 * float(args.phantom_size_y_cm) / float(args.ybins)
    voxel_size_z_mm = 10.0 * float(args.phantom_size_z_cm) / float(args.zbins)
    return {
        "nominal_beam_label": f"{float(args.nominal_energy_mv):.1f} MV clinical LINAC beam",
        "direct_photon_model": {
            "particle": "gamma",
            "spectrum_type": "representative_polyenergetic_discrete",
            "spectrum_csv": str(args.spectrum_csv),
            "spectrum_energies_mev": [float(value) for value in spectrum_energies],
            "spectrum_weights": [float(value) for value in spectrum_weights],
            "position_distribution": "Flat",
            "angular_distribution": "None",
        },
        "geometry": {
            "source_z_cm": float(args.source_z_cm),
            "phantom_size_cm": {
                "x": float(args.phantom_size_x_cm),
                "y": float(args.phantom_size_y_cm),
                "z": float(args.phantom_size_z_cm),
            },
            "phantom_center_z_cm": phantom_center_z_cm,
            "voxel_grid": {
                "xbins": int(args.xbins),
                "ybins": int(args.ybins),
                "zbins": int(args.zbins),
            },
            "voxel_size_mm": {
                "x": voxel_size_x_mm,
                "y": voxel_size_y_mm,
                "z": voxel_size_z_mm,
            },
        },
        "field": {
            "diameter_mm": 2.0 * float(args.beam_position_cutoff_mm),
            "cutoff_x_mm": float(args.beam_position_cutoff_mm),
            "cutoff_y_mm": float(args.beam_position_cutoff_mm),
        },
        "simulation": {
            "histories": int(args.histories),
            "threads": int(args.threads),
            "seed": int(args.seed),
            "physics_profile": str(args.physics_profile),
        },
        "synthetic_sfrt": {
            "pitches_mm": [float(v) for v in args.pitches_mm],
            "n_beams_x": int(args.n_beams_x),
            "n_beams_y": int(args.n_beams_y),
            "prescribed_peak_dose_gy": float(args.prescribed_peak_dose_gy),
            "uniform_dose_floor_fraction_of_peak": float(args.uniform_dose_floor_fraction),
        },
        "biology": {
            "emission_model": "continuous_saturated" if bool(args.continuous_emission) else "binary_threshold",
            "emission_threshold_gy": float(args.emission_threshold_gy),
            "emission_emax": float(args.emission_emax),
            "emission_gamma_per_gy": float(args.emission_gamma_per_gy),
            "phase2_diffusion_length_mm": float(args.phase2_diffusion_length_mm),
            "phase2_signal_strength": float(args.phase2_signal_strength),
            "lambda_values_mm": [float(v) for v in args.lambda_values_mm],
            "lambda_sweep_amplitude": float(args.lambda_sweep_amplitude),
            "amplitude_values": [float(v) for v in args.amplitude_values],
            "amplitude_sweep_lambda_mm": float(args.amplitude_sweep_lambda_mm),
            "alpha_gy_inverse": float(args.phase2_alpha),
            "beta_gy_inverse_squared": float(args.phase2_beta),
            "high_dose_methodology_note": str(args.high_dose_methodology_note),
        },
        "paths": {
            "case_dir": str(case_dir),
            "phase2_dir": str(phase2_dir),
            "phase3_dir": str(phase3_dir),
            "dose_csv": str(case_dir / "dosedata.csv"),
            "parameter_file": str(case_dir / "beamline.txt"),
        },
        "assumptions": [
            "This is still a mathematical direct-photon surrogate, but now with a representative discrete 6 MV spectrum instead of a monoenergetic beam.",
            "The 1.5% uniform dose floor is a Python-side surrogate for block or MLC transmission rather than an explicit TOPAS block geometry.",
            "The continuous bystander emission model uses E(D) = Emax * (1 - exp(-gamma * D)).",
        ],
    }


def run_topas_case(args: argparse.Namespace, case_dir: Path, parameter_file: Path, dose_csv: Path, log_file: Path) -> None:
    result = subprocess.run(
        [str(args.topas_bin), parameter_file.name],
        cwd=str(case_dir),
        capture_output=True,
        text=True,
        env=build_topas_env(str(args.g4_data_dir)),
    )
    combined_log = (result.stdout or "") + ("\n" + result.stderr if result.stderr else "")
    write_text_with_retries(log_file, combined_log)
    if result.returncode != 0 or not has_nonempty_output(dose_csv):
        tail = "\n".join(combined_log.strip().splitlines()[-30:])
        raise RuntimeError(
            "TOPAS run failed for the polyenergetic 6 MV case.\n"
            f"Return code: {result.returncode}\n"
            f"Dose CSV present: {has_nonempty_output(dose_csv)}\n"
            f"Recent log:\n{tail}"
        )


def run_phase2(args: argparse.Namespace, root: Path, run_root: Path, dose_csv: Path, outdir: Path) -> Path:
    outdir.mkdir(parents=True, exist_ok=True)
    script = root / "scripts" / "generate_phase2_bystander_figures.py"
    mplconfigdir = run_root / ".mpl-cache-phase2"
    home_dir = run_root / ".home-phase2"
    xdg_cache_dir = run_root / ".xdg-cache-phase2"
    mplconfigdir.mkdir(parents=True, exist_ok=True)
    home_dir.mkdir(parents=True, exist_ok=True)
    xdg_cache_dir.mkdir(parents=True, exist_ok=True)
    env = dict(**build_topas_env(str(args.g4_data_dir)))
    env["HOME"] = str(home_dir)
    env["XDG_CACHE_HOME"] = str(xdg_cache_dir)
    env["MPLCONFIGDIR"] = str(mplconfigdir)

    cmd = [
        sys.executable,
        str(script),
        "--csv",
        str(dose_csv),
        "--outdir",
        str(outdir),
        "--pitches-mm",
        *[f"{float(v):.6f}" for v in args.pitches_mm],
        "--n-beams-x",
        str(int(args.n_beams_x)),
        "--n-beams-y",
        str(int(args.n_beams_y)),
        "--prescribed-peak-dose-gy",
        f"{float(args.prescribed_peak_dose_gy):.6f}",
        "--depths-cm",
        f"{float(args.phase2_depths_cm[0]):.6f}",
        f"{float(args.phase2_depths_cm[1]):.6f}",
        "--uniform-dose-floor-fraction",
        f"{float(args.uniform_dose_floor_fraction):.6f}",
        "--emission-threshold-gy",
        f"{float(args.emission_threshold_gy):.6f}",
        "--emission-emax",
        f"{float(args.emission_emax):.6f}",
        "--emission-gamma-per-gy",
        f"{float(args.emission_gamma_per_gy):.6f}",
        "--diffusion-length-mm",
        f"{float(args.phase2_diffusion_length_mm):.6f}",
        "--signal-strength",
        f"{float(args.phase2_signal_strength):.8f}",
        "--kernel-radius-lambda",
        f"{float(args.kernel_radius_lambda):.6f}",
        "--alpha",
        f"{float(args.phase2_alpha):.6f}",
        "--beta",
        f"{float(args.phase2_beta):.6f}",
        "--high-dose-methodology-note",
        str(args.high_dose_methodology_note),
        "--dpi",
        str(int(args.dpi)),
    ]
    if args.continuous_emission:
        cmd.append("--continuous-emission")
    result = subprocess.run(cmd, capture_output=True, text=True, env=env)
    combined_log = (result.stdout or "") + ("\n" + result.stderr if result.stderr else "")
    write_text_with_retries(outdir / "analysis.log", combined_log)
    if result.returncode != 0:
        tail = "\n".join(combined_log.strip().splitlines()[-30:])
        raise RuntimeError(f"Phase 2 analysis failed.\n{tail}")
    return outdir / "phase2_summary.json"


def run_phase3(args: argparse.Namespace, root: Path, run_root: Path, dose_csv: Path, outdir: Path) -> Path:
    outdir.mkdir(parents=True, exist_ok=True)
    script = root / "scripts" / "generate_phase3_bystander_sweeps.py"
    mplconfigdir = run_root / ".mpl-cache-phase3"
    home_dir = run_root / ".home-phase3"
    xdg_cache_dir = run_root / ".xdg-cache-phase3"
    mplconfigdir.mkdir(parents=True, exist_ok=True)
    home_dir.mkdir(parents=True, exist_ok=True)
    xdg_cache_dir.mkdir(parents=True, exist_ok=True)
    env = dict(**build_topas_env(str(args.g4_data_dir)))
    env["HOME"] = str(home_dir)
    env["XDG_CACHE_HOME"] = str(xdg_cache_dir)
    env["MPLCONFIGDIR"] = str(mplconfigdir)

    cmd = [
        sys.executable,
        str(script),
        "--csv",
        str(dose_csv),
        "--outdir",
        str(outdir),
        "--pitches-mm",
        *[f"{float(v):.6f}" for v in args.pitches_mm],
        "--n-beams-x",
        str(int(args.n_beams_x)),
        "--n-beams-y",
        str(int(args.n_beams_y)),
        "--prescribed-peak-dose-gy",
        f"{float(args.prescribed_peak_dose_gy):.6f}",
        "--target-depth-cm",
        f"{float(args.target_depth_cm):.6f}",
        "--uniform-dose-floor-fraction",
        f"{float(args.uniform_dose_floor_fraction):.6f}",
        "--emission-threshold-gy",
        f"{float(args.emission_threshold_gy):.6f}",
        "--emission-emax",
        f"{float(args.emission_emax):.6f}",
        "--emission-gamma-per-gy",
        f"{float(args.emission_gamma_per_gy):.6f}",
        "--alpha",
        f"{float(args.phase2_alpha):.6f}",
        "--beta",
        f"{float(args.phase2_beta):.6f}",
        "--lambda-values-mm",
        *[f"{float(v):.6f}" for v in args.lambda_values_mm],
        "--lambda-sweep-amplitude",
        f"{float(args.lambda_sweep_amplitude):.8f}",
        "--amplitude-values",
        *[f"{float(v):.8f}" for v in args.amplitude_values],
        "--amplitude-sweep-lambda-mm",
        f"{float(args.amplitude_sweep_lambda_mm):.6f}",
        "--kernel-radius-lambda",
        f"{float(args.kernel_radius_lambda):.6f}",
        "--dpi",
        str(int(args.dpi)),
    ]
    if args.continuous_emission:
        cmd.append("--continuous-emission")
    result = subprocess.run(cmd, capture_output=True, text=True, env=env)
    combined_log = (result.stdout or "") + ("\n" + result.stderr if result.stderr else "")
    write_text_with_retries(outdir / "analysis.log", combined_log)
    if result.returncode != 0:
        tail = "\n".join(combined_log.strip().splitlines()[-30:])
        raise RuntimeError(f"Phase 3 sweep analysis failed.\n{tail}")
    return outdir / "phase3_sweep_summary.json"


def load_csv_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_markdown_report(
    args: argparse.Namespace,
    metadata: Dict[str, object],
    phase2_summary: Dict[str, object],
    phase3_summary: Dict[str, object],
    out_file: Path,
) -> None:
    phase2_metrics_csv = Path(str(phase2_summary["metrics_csv"]))
    phase3_lambda_csv = Path(str(phase3_summary["lambda_sweep"]["rows_csv"]))
    phase3_amplitude_csv = Path(str(phase3_summary["amplitude_sweep"]["rows_csv"]))
    phase2_rows = load_csv_rows(phase2_metrics_csv)
    lambda_rows = load_csv_rows(phase3_lambda_csv)
    amplitude_rows = load_csv_rows(phase3_amplitude_csv)

    def find_row(rows: List[Dict[str, str]], **conditions: str) -> Dict[str, str]:
        for row in rows:
            if all(row.get(key) == value for key, value in conditions.items()):
                return row
        raise KeyError(f"Could not find row matching {conditions} in {rows[:2]}...")

    lines = [
        "# Polyenergetic 6 MV Clinical-Surrogate SFRT Summary",
        "",
        f"- TOPAS beam label: `{metadata['nominal_beam_label']}`.",
        (
            f"- Source model: direct `gamma` beam with a representative discrete 6 MV spectrum from "
            f"`{metadata['direct_photon_model']['spectrum_csv']}` and a `10 mm` flat circular aperture."
        ),
        (
            f"- Histories / threads / seed: `{int(args.histories)}` / `{int(args.threads)}` / `{int(args.seed)}`."
        ),
        (
            f"- Geometry: source plane at `z={float(args.source_z_cm):.3f} cm`, water phantom "
            f"`{float(args.phantom_size_x_cm):.1f} x {float(args.phantom_size_y_cm):.1f} x {float(args.phantom_size_z_cm):.1f} cm^3`, "
            f"grid `{int(args.xbins)} x {int(args.ybins)} x {int(args.zbins)}`."
        ),
        (
            f"- Python-side transmission floor: `{100.0 * float(args.uniform_dose_floor_fraction):.1f}%` of the "
            f"`{float(args.prescribed_peak_dose_gy):.1f} Gy` prescribed peak, added uniformly to the synthetic lattice."
        ),
        (
            f"- Bystander emission model: `E(D) = Emax * (1 - exp(-gamma * D))` with "
            f"`Emax={float(args.emission_emax):.2f}` and `gamma={float(args.emission_gamma_per_gy):.2f} Gy^-1`."
        ),
        f"- High-dose methodology note: {str(args.high_dose_methodology_note)}",
        "",
        "## Phase 2",
        f"- Figures folder: `{metadata['paths']['phase2_dir']}`.",
        f"- Metrics CSV: `{phase2_summary['metrics_csv']}`.",
        "",
        "| Pitch | Depth | PVDR | Valley Survival (LQ) | Valley Survival (Multi-effect) |",
        "| --- | --- | ---: | ---: | ---: |",
    ]

    for pitch in args.pitches_mm:
        row = find_row(phase2_rows, pitch_mm=f"{float(pitch):.1f}", depth_label="d=5cm")
        lines.append(
            f"| {float(pitch):.0f} mm | {float(row['sampled_depth_cm']):.3f} cm | "
            f"{float(row['pvdr']):.3f} | {float(row['valley_survival_lq']):.3f} | "
            f"{float(row['valley_survival_total']):.3f} |"
        )

    lines.extend(
        [
            "",
            "## Phase 3",
            f"- Figures: `{phase3_summary['lambda_sweep']['figure']}` and `{phase3_summary['amplitude_sweep']['figure']}`.",
            f"- Lambda sweep CSV: `{phase3_summary['lambda_sweep']['rows_csv']}`.",
            f"- Amplitude sweep CSV: `{phase3_summary['amplitude_sweep']['rows_csv']}`.",
            "",
            "| Sweep | Setting | 20 mm | 30 mm | 40 mm |",
            "| --- | --- | ---: | ---: | ---: |",
        ]
    )

    lambda_target = f"{float(args.lambda_values_mm[-1]):.1f}"
    lambda_vals = []
    for pitch in args.pitches_mm:
        row = find_row(lambda_rows, pitch_mm=f"{float(pitch):.1f}", lambda_mm=lambda_target, signal_strength=str(float(args.lambda_sweep_amplitude)))
        lambda_vals.append(f"{float(row['valley_survival_total']):.3f}")
    lines.append(
        f"| Lambda sweep | lambda = {float(args.lambda_values_mm[-1]):.1f} mm, A = {float(args.lambda_sweep_amplitude):.1e} | "
        f"{lambda_vals[0]} | {lambda_vals[1]} | {lambda_vals[2]} |"
    )

    amplitude_target = f"{float(args.amplitude_values[-1]):g}"
    amplitude_vals = []
    for pitch in args.pitches_mm:
        row = find_row(
            amplitude_rows,
            pitch_mm=f"{float(pitch):.1f}",
            lambda_mm=f"{float(args.amplitude_sweep_lambda_mm):.1f}",
            signal_strength=amplitude_target,
        )
        amplitude_vals.append(f"{float(row['valley_survival_total']):.3f}")
    lines.append(
        f"| Amplitude sweep | lambda = {float(args.amplitude_sweep_lambda_mm):.1f} mm, A = {float(args.amplitude_values[-1]):.1e} | "
        f"{amplitude_vals[0]} | {amplitude_vals[1]} | {amplitude_vals[2]} |"
    )

    write_text_with_retries(out_file, "\n".join(lines) + "\n")


def main() -> int:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    run_root = args.run_root
    case_dir = run_root / "case"
    phase2_dir = run_root / "analysis_phase2_clinical_upgrade"
    phase3_dir = run_root / "analysis_phase3_clinical_upgrade_sweeps"
    case_dir.mkdir(parents=True, exist_ok=True)
    phase2_dir.mkdir(parents=True, exist_ok=True)
    phase3_dir.mkdir(parents=True, exist_ok=True)

    parameter_file = case_dir / "beamline.txt"
    dose_csv = case_dir / "dosedata.csv"
    topas_log = case_dir / "topas.log"

    spectrum_energies, spectrum_weights = load_spectrum(args.spectrum_csv)
    rendered_case = render_case_file(args, spectrum_energies, spectrum_weights)
    write_text_with_retries(parameter_file, rendered_case)

    metadata = build_case_metadata(args, case_dir, phase2_dir, phase3_dir, spectrum_energies, spectrum_weights)
    metadata_file = run_root / "case_metadata.json"
    write_text_with_retries(metadata_file, json.dumps(metadata, indent=2) + "\n")

    if not args.analyze_only:
        if args.skip_existing and has_nonempty_output(dose_csv):
            print(f"Skipping TOPAS run because {dose_csv} already exists.")
        else:
            print(f"Running TOPAS case: {parameter_file}")
            run_topas_case(args, case_dir, parameter_file, dose_csv, topas_log)
    elif not has_nonempty_output(dose_csv):
        raise FileNotFoundError(f"--analyze-only was requested but missing {dose_csv}")

    print(f"Running upgraded Phase 2 analysis from {dose_csv}")
    phase2_summary_file = run_phase2(args, root, run_root, dose_csv, phase2_dir)
    phase2_summary = json.loads(phase2_summary_file.read_text(encoding="utf-8"))

    print(f"Running upgraded Phase 3 sweeps from {dose_csv}")
    phase3_summary_file = run_phase3(args, root, run_root, dose_csv, phase3_dir)
    phase3_summary = json.loads(phase3_summary_file.read_text(encoding="utf-8"))

    report_file = run_root / "clinical_upgrade_summary.md"
    write_markdown_report(args, metadata, phase2_summary, phase3_summary, report_file)

    print(f"Parameter file: {parameter_file}")
    print(f"Dose CSV: {dose_csv}")
    print(f"Phase 2 summary: {phase2_summary_file}")
    print(f"Phase 3 summary: {phase3_summary_file}")
    print(f"Report: {report_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
