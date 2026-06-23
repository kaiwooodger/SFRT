#!/usr/bin/env python
"""Generate and analyze a nominal 6 MV LINAC photon SFRT case in TOPAS."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict

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
            "Create a clinically grounded nominal 6 MV LINAC photon pencil-beam "
            "TOPAS case in water, run it, and post-process it into an SFRT lattice."
        )
    )
    parser.add_argument(
        "--template",
        type=Path,
        default=root / "topas" / "linac_6mv_nominal_photon_template.txt",
        help="TOPAS template used to build the case deck.",
    )
    parser.add_argument(
        "--run-root",
        type=Path,
        default=root / "runs" / "linac_6mv_nominal_sfrt",
        help="Output directory for the TOPAS case and post-processing artifacts.",
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
        "--incident-electron-energy-mev",
        type=float,
        default=6.0,
        help="Incident electron beam energy in MeV for the tungsten target model.",
    )
    parser.add_argument(
        "--energy-spread-mev",
        type=float,
        default=0.075,
        help="Absolute incident-electron energy spread in MeV.",
    )
    parser.add_argument(
        "--beam-sigma-mm",
        type=float,
        default=1.666667,
        help="Gaussian sigma for the pencil beam in mm.",
    )
    parser.add_argument(
        "--beam-position-cutoff-mm",
        type=float,
        default=5.0,
        help="Gaussian beam position cutoff radius in mm.",
    )
    parser.add_argument(
        "--beam-angular-spread-mrad",
        type=float,
        default=3.2,
        help="Gaussian angular spread in mrad.",
    )
    parser.add_argument(
        "--beam-angular-cutoff-mrad",
        type=float,
        default=90.0,
        help="Angular cutoff in mrad.",
    )
    parser.add_argument(
        "--ssd-cm",
        type=float,
        default=100.0,
        help="Source-to-surface distance from the target plane to the phantom entrance.",
    )
    parser.add_argument(
        "--target-size-x-cm",
        type=float,
        default=2.0,
        help="Tungsten target size in x.",
    )
    parser.add_argument(
        "--target-size-y-cm",
        type=float,
        default=2.0,
        help="Tungsten target size in y.",
    )
    parser.add_argument(
        "--target-thickness-mm",
        type=float,
        default=1.0,
        help="Tungsten target thickness.",
    )
    parser.add_argument(
        "--source-to-target-gap-mm",
        type=float,
        default=1.0,
        help="Gap from the incident electron source plane to the upstream target face.",
    )
    parser.add_argument(
        "--phantom-size-x-cm",
        type=float,
        default=12.0,
        help="Water phantom size in x.",
    )
    parser.add_argument(
        "--phantom-size-y-cm",
        type=float,
        default=12.0,
        help="Water phantom size in y.",
    )
    parser.add_argument(
        "--phantom-size-z-cm",
        type=float,
        default=20.0,
        help="Water phantom size in z.",
    )
    parser.add_argument("--xbins", type=int, default=121, help="Phantom x bins.")
    parser.add_argument("--ybins", type=int, default=121, help="Phantom y bins.")
    parser.add_argument("--zbins", type=int, default=161, help="Phantom z bins.")
    parser.add_argument(
        "--max-step-phantom-mm",
        type=float,
        default=0.1,
        help="Maximum tracking step in water.",
    )
    parser.add_argument(
        "--max-step-target-mm",
        type=float,
        default=0.02,
        help="Maximum tracking step in tungsten.",
    )
    parser.add_argument(
        "--cut-gamma-mm",
        type=float,
        default=0.01,
        help="Photon production cut in mm.",
    )
    parser.add_argument(
        "--cut-electron-mm",
        type=float,
        default=0.01,
        help="Electron production cut in mm.",
    )
    parser.add_argument(
        "--cut-positron-mm",
        type=float,
        default=0.01,
        help="Positron production cut in mm.",
    )
    parser.add_argument(
        "--prescribed-peak-dose-gy",
        type=float,
        default=10.0,
        help="Peak dose assigned after normalization.",
    )
    parser.add_argument(
        "--pitch-mm",
        type=float,
        default=10.0,
        help="Synthetic SFRT center-to-center lattice spacing.",
    )
    parser.add_argument("--n-beams-x", type=int, default=7, help="Beam copies along x.")
    parser.add_argument("--n-beams-y", type=int, default=7, help="Beam copies along y.")
    parser.add_argument(
        "--hippocampus-depth-cm",
        type=float,
        default=5.0,
        help="Reference depth used for a second PVDR report.",
    )
    parser.add_argument("--alpha", type=float, default=0.10, help="LQ alpha in Gy^-1.")
    parser.add_argument("--beta", type=float, default=0.05, help="LQ beta in Gy^-2.")
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


def build_case_metadata(args: argparse.Namespace, case_dir: Path, analysis_dir: Path) -> Dict[str, object]:
    target_hlz_cm = 0.5 * float(args.target_thickness_mm) / 10.0
    phantom_center_z_cm = target_hlz_cm + float(args.ssd_cm) + 0.5 * float(args.phantom_size_z_cm)
    source_z_cm = -(target_hlz_cm + float(args.source_to_target_gap_mm) / 10.0)
    voxel_size_x_mm = 10.0 * float(args.phantom_size_x_cm) / float(args.xbins)
    voxel_size_y_mm = 10.0 * float(args.phantom_size_y_cm) / float(args.ybins)
    voxel_size_z_mm = 10.0 * float(args.phantom_size_z_cm) / float(args.zbins)
    return {
        "nominal_beam_label": f"{float(args.nominal_energy_mv):.1f} MV clinical LINAC beam",
        "linac_proxy_model": {
            "incident_particle": "e-",
            "incident_electron_energy_mev": float(args.incident_electron_energy_mev),
            "incident_energy_spread_mev": float(args.energy_spread_mev),
            "target_material": "G4_W",
        },
        "source_model": {
            "distribution": "Gaussian",
            "beam_sigma_mm": float(args.beam_sigma_mm),
            "beam_position_cutoff_x_mm": float(args.beam_position_cutoff_mm),
            "beam_position_cutoff_y_mm": float(args.beam_position_cutoff_mm),
            "beam_angular_spread_x_mrad": float(args.beam_angular_spread_mrad),
            "beam_angular_spread_y_mrad": float(args.beam_angular_spread_mrad),
            "beam_angular_cutoff_mrad": float(args.beam_angular_cutoff_mrad),
        },
        "geometry": {
            "ssd_cm": float(args.ssd_cm),
            "target_size_cm": {
                "x": float(args.target_size_x_cm),
                "y": float(args.target_size_y_cm),
            },
            "target_thickness_mm": float(args.target_thickness_mm),
            "target_center_z_cm": 0.0,
            "source_to_target_gap_mm": float(args.source_to_target_gap_mm),
            "source_z_cm": source_z_cm,
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
        "simulation": {
            "histories": int(args.histories),
            "threads": int(args.threads),
            "seed": int(args.seed),
            "physics_profile": str(args.physics_profile),
        },
        "synthetic_sfrt": {
            "pitch_mm": float(args.pitch_mm),
            "n_beams_x": int(args.n_beams_x),
            "n_beams_y": int(args.n_beams_y),
            "prescribed_peak_dose_gy": float(args.prescribed_peak_dose_gy),
            "hippocampus_depth_cm": float(args.hippocampus_depth_cm),
            "alpha_gy_inverse": float(args.alpha),
            "beta_gy_inverse_squared": float(args.beta),
        },
        "paths": {
            "case_dir": str(case_dir),
            "analysis_dir": str(analysis_dir),
            "dose_csv": str(case_dir / "dosedata.csv"),
            "parameter_file": str(case_dir / "beamline.txt"),
        },
        "assumptions": [
            "A nominal clinical 6 MV photon beam is approximated by a 6 MeV incident electron pencil beam striking a tungsten target.",
            "The user-provided BeamPositionCutoffY value ended with 'm,' and was interpreted as 5.0 mm to match the requested 1 cm pencil beam.",
            "The Gaussian source sigma is taken as cutoff/3 so the beam is mostly contained inside the requested 1 cm diameter.",
            "The water phantom entrance is placed 100 cm downstream of the target plane, which is a standard clinical reference geometry.",
        ],
    }


def render_case_file(args: argparse.Namespace, case_dir: Path) -> str:
    template_text = args.template.read_text(encoding="utf-8")
    target_hlx_cm = 0.5 * float(args.target_size_x_cm)
    target_hly_cm = 0.5 * float(args.target_size_y_cm)
    target_hlz_cm = 0.5 * float(args.target_thickness_mm) / 10.0
    target_center_z_cm = 0.0
    phantom_hlx_cm = 0.5 * float(args.phantom_size_x_cm)
    phantom_hly_cm = 0.5 * float(args.phantom_size_y_cm)
    phantom_hlz_cm = 0.5 * float(args.phantom_size_z_cm)
    phantom_center_z_cm = target_hlz_cm + float(args.ssd_cm) + phantom_hlz_cm
    world_hlx_cm = max(40.0, float(args.phantom_size_x_cm) + 20.0)
    world_hly_cm = max(40.0, float(args.phantom_size_y_cm) + 20.0)
    world_hlz_cm = max(180.0, phantom_center_z_cm + phantom_hlz_cm + 20.0)
    energy_spread_rel = float(args.energy_spread_mev) / float(args.incident_electron_energy_mev)
    source_z_cm = -(target_hlz_cm + float(args.source_to_target_gap_mm) / 10.0)
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
        "__SOURCE_Z_CM__": f"{source_z_cm:.6f}",
        "__TARGET_HLX_CM__": f"{target_hlx_cm:.6f}",
        "__TARGET_HLY_CM__": f"{target_hly_cm:.6f}",
        "__TARGET_HLZ_CM__": f"{target_hlz_cm:.6f}",
        "__TARGET_CENTER_Z_CM__": f"{target_center_z_cm:.6f}",
        "__PHANTOM_HLX_CM__": f"{phantom_hlx_cm:.6f}",
        "__PHANTOM_HLY_CM__": f"{phantom_hly_cm:.6f}",
        "__PHANTOM_HLZ_CM__": f"{phantom_hlz_cm:.6f}",
        "__PHANTOM_CENTER_Z_CM__": f"{phantom_center_z_cm:.6f}",
        "__MAX_STEP_PHANTOM_MM__": f"{float(args.max_step_phantom_mm):.6f}",
        "__MAX_STEP_TARGET_MM__": f"{float(args.max_step_target_mm):.6f}",
        "__XBINS__": str(int(args.xbins)),
        "__YBINS__": str(int(args.ybins)),
        "__ZBINS__": str(int(args.zbins)),
        "__OUTPUT_STEM__": "dosedata",
        "__INCIDENT_ELECTRON_ENERGY_MEV__": f"{float(args.incident_electron_energy_mev):.6f}",
        "__ENERGY_SPREAD_REL__": f"{energy_spread_rel:.9f}",
        "__BEAM_POSITION_CUTOFF_X_MM__": f"{float(args.beam_position_cutoff_mm):.6f}",
        "__BEAM_POSITION_CUTOFF_Y_MM__": f"{float(args.beam_position_cutoff_mm):.6f}",
        "__BEAM_SIGMA_X_MM__": f"{float(args.beam_sigma_mm):.6f}",
        "__BEAM_SIGMA_Y_MM__": f"{float(args.beam_sigma_mm):.6f}",
        "__BEAM_ANGULAR_CUTOFF_MRAD__": f"{float(args.beam_angular_cutoff_mrad):.6f}",
        "__BEAM_ANGULAR_SPREAD_X_MRAD__": f"{float(args.beam_angular_spread_mrad):.6f}",
        "__BEAM_ANGULAR_SPREAD_Y_MRAD__": f"{float(args.beam_angular_spread_mrad):.6f}",
        "__N_HISTORIES__": str(int(args.histories)),
        "__N_THREADS__": str(int(args.threads)),
        "__SEED__": str(int(args.seed)),
        "__SHOW_HISTORY_INTERVAL__": str(int(show_interval)),
    }
    return fill_template(template_text, replacements)


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
            "TOPAS run failed for the nominal 6 MV case.\n"
            f"Return code: {result.returncode}\n"
            f"Dose CSV present: {has_nonempty_output(dose_csv)}\n"
            f"Recent log:\n{tail}"
        )


def run_analysis(args: argparse.Namespace, root: Path, run_root: Path, dose_csv: Path, analysis_dir: Path) -> Path:
    analysis_dir.mkdir(parents=True, exist_ok=True)
    analysis_script = root / "scripts" / "analyze_250mev_sfrt_plan.py"
    mplconfigdir = run_root / ".mpl-cache"
    mplconfigdir.mkdir(parents=True, exist_ok=True)
    env = dict(**build_topas_env(str(args.g4_data_dir)))
    env["MPLCONFIGDIR"] = str(mplconfigdir)

    cmd = [
        sys.executable,
        str(analysis_script),
        "--csv",
        str(dose_csv),
        "--outdir",
        str(analysis_dir),
        "--case-label",
        "linac_6mv_nominal_photon",
        "--beam-label",
        "Nominal 6 MV LINAC photon beam",
        "--histories",
        str(int(args.histories)),
        "--pitch-mm",
        f"{float(args.pitch_mm):.6f}",
        "--n-beams-x",
        str(int(args.n_beams_x)),
        "--n-beams-y",
        str(int(args.n_beams_y)),
        "--prescribed-peak-dose-gy",
        f"{float(args.prescribed_peak_dose_gy):.6f}",
        "--hippocampus-depth-cm",
        f"{float(args.hippocampus_depth_cm):.6f}",
        "--alpha",
        f"{float(args.alpha):.6f}",
        "--beta",
        f"{float(args.beta):.6f}",
        "--figure1-title",
        "Figure 1: Physical lateral dose profile for the nominal 6 MV LINAC SFRT lattice",
        "--figure2-title",
        "Figure 2: Standard LQ survival slice for the nominal 6 MV LINAC SFRT lattice",
        "--dpi",
        str(int(args.dpi)),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, env=env)
    combined_log = (result.stdout or "") + ("\n" + result.stderr if result.stderr else "")
    write_text_with_retries(analysis_dir / "analysis.log", combined_log)
    if result.returncode != 0:
        tail = "\n".join(combined_log.strip().splitlines()[-30:])
        raise RuntimeError(f"Analysis failed for the nominal 6 MV case.\n{tail}")
    return analysis_dir / "summary.json"


def write_markdown_report(
    args: argparse.Namespace,
    metadata: Dict[str, object],
    summary: Dict[str, object],
    out_file: Path,
) -> None:
    pvdr_curve_path = Path(str(summary["pvdr"]["curve_csv"]))
    with pvdr_curve_path.open("r", encoding="utf-8", newline="") as handle:
        pvdr_rows = list(csv.DictReader(handle))

    def nearest_pvdr_row(target_cm: float) -> Dict[str, float]:
        parsed = [
            {
                "depth_cm": float(row["depth_cm"]),
                "pvdr": float(row["pvdr"]),
            }
            for row in pvdr_rows
        ]
        return min(parsed, key=lambda row: abs(row["depth_cm"] - float(target_cm)))

    voxel_single = summary["voxel_grid_single_beam"]
    voxel_lattice = summary["voxel_grid_lattice"]
    pvdr = summary["pvdr"]
    outputs = summary["outputs"]
    max_depth_cm = max(float(row["depth_cm"]) for row in pvdr_rows)
    dmax_depth_cm = float(pvdr["depths_reported_cm"]["dmax"])
    pvdr_5cm = nearest_pvdr_row(5.0)
    pvdr_10cm = nearest_pvdr_row(10.0)
    if max_depth_cm < 30.0:
        pvdr_30_depth = "--"
        pvdr_30_value = "--"
        pvdr_30_note = f"Unavailable; phantom ends at {max_depth_cm:.3f} cm"
    else:
        pvdr_30cm = nearest_pvdr_row(30.0)
        pvdr_30_depth = f"{pvdr_30cm['depth_cm']:.3f}"
        pvdr_30_value = f"{pvdr_30cm['pvdr']:.3f}"
        pvdr_30_note = ""
    lines = [
        "# Nominal 6 MV LINAC SFRT Summary",
        "",
        f"- Nominal beam label: `{metadata['nominal_beam_label']}`.",
        (
            f"- Simplified LINAC proxy: incident `e-` beam at "
            f"`{float(args.incident_electron_energy_mev):.3f} MeV` onto a tungsten target to represent a nominal `6 MV` beam."
        ),
        (
            f"- Histories / threads / seed: `{int(args.histories)}` / `{int(args.threads)}` / "
            f"`{int(args.seed)}`."
        ),
        (
            f"- Source spot: Gaussian with sigma `{float(args.beam_sigma_mm):.3f} mm`, "
            f"cutoff `{float(args.beam_position_cutoff_mm):.3f} mm` in both x and y."
        ),
        (
            f"- Target model: tungsten `{float(args.target_size_x_cm):.1f} x {float(args.target_size_y_cm):.1f} cm^2`, "
            f"thickness `{float(args.target_thickness_mm):.3f} mm`, source-to-target gap `{float(args.source_to_target_gap_mm):.3f} mm`."
        ),
        f"- Geometry: `SSD={float(args.ssd_cm):.1f} cm`, water phantom `{float(args.phantom_size_x_cm):.1f} x {float(args.phantom_size_y_cm):.1f} x {float(args.phantom_size_z_cm):.1f} cm^3`.",
        (
            f"- Single-beam voxel grid: `{voxel_single['shape']}` with spacing "
            f"`{voxel_single['spacing_mm']['dx']:.3f} x {voxel_single['spacing_mm']['dy']:.3f} x {voxel_single['spacing_mm']['dz']:.3f} mm^3`."
        ),
        (
            f"- Synthetic lattice voxel grid: `{voxel_lattice['shape']}` with spacing "
            f"`{voxel_lattice['spacing_cm']['dx']*10.0:.3f} x {voxel_lattice['spacing_cm']['dy']*10.0:.3f} x {voxel_lattice['spacing_cm']['dz']*10.0:.3f} mm^3`."
        ),
        (
            f"- Chosen center-to-center spacing: `{float(args.pitch_mm):.1f} mm` with "
            f"`{int(args.n_beams_x)} x {int(args.n_beams_y)}` synthetic beam copies."
        ),
        "",
        "| Depth Target | Sampled Depth (cm) | PVDR | Note |",
        "| --- | ---: | ---: | --- |",
        f"| dmax | {dmax_depth_cm:.3f} | {pvdr['dmax']['pvdr']:.3f} | Maximum lattice depth-dose slice |",
        f"| 5 cm | {pvdr_5cm['depth_cm']:.3f} | {pvdr_5cm['pvdr']:.3f} | Nearest voxel-center depth |",
        f"| 10 cm | {pvdr_10cm['depth_cm']:.3f} | {pvdr_10cm['pvdr']:.3f} | Nearest voxel-center depth |",
        f"| 30 cm | {pvdr_30_depth} | {pvdr_30_value} | {pvdr_30_note} |",
        f"- Dose CSV: `{summary['input_csv']}`.",
        f"- Figure 1: `{outputs['figure1_physical']}`.",
        f"- Figure 2: `{outputs['figure2_standard_bio']}`.",
        f"- PVDR depth curve: `{pvdr['curve_csv']}`.",
    ]
    write_text_with_retries(out_file, "\n".join(lines) + "\n")


def main() -> int:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    run_root = args.run_root
    case_dir = run_root / "case"
    analysis_dir = run_root / "analysis"
    case_dir.mkdir(parents=True, exist_ok=True)
    analysis_dir.mkdir(parents=True, exist_ok=True)

    parameter_file = case_dir / "beamline.txt"
    dose_csv = case_dir / "dosedata.csv"
    topas_log = case_dir / "topas.log"

    rendered_case = render_case_file(args, case_dir)
    write_text_with_retries(parameter_file, rendered_case)

    metadata = build_case_metadata(args, case_dir, analysis_dir)
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

    print(f"Analyzing dose cube: {dose_csv}")
    summary_file = run_analysis(args, root, run_root, dose_csv, analysis_dir)
    summary = json.loads(summary_file.read_text(encoding="utf-8"))

    report_file = run_root / "run_summary.md"
    write_markdown_report(args, metadata, summary, report_file)

    print(f"Parameter file: {parameter_file}")
    print(f"Dose CSV: {dose_csv}")
    print(f"Summary JSON: {summary_file}")
    print(f"Report: {report_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
