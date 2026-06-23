#!/usr/bin/env python
"""Run a mathematical 6 MV direct-photon SFRT baseline in TOPAS."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

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
            "Create a mathematical direct-photon 6 MV baseline in water, run TOPAS, "
            "and post-process the dose cube into an SFRT lattice."
        )
    )
    parser.add_argument(
        "--template",
        type=Path,
        default=root / "topas" / "linac_6mv_direct_photon_template.txt",
        help="TOPAS template used to build the direct-photon case deck.",
    )
    parser.add_argument(
        "--run-root",
        type=Path,
        default=root / "runs" / "linac_6mv_direct_photon_sfrt",
        help="Output directory for the direct-photon case and post-processing artifacts.",
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
        "--photon-energy-mev",
        type=float,
        default=6.0,
        help="Monoenergetic photon energy for the mathematical baseline.",
    )
    parser.add_argument(
        "--energy-spread-mev",
        type=float,
        default=0.0,
        help="Absolute photon energy spread in MeV.",
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
        default=20.0,
        help="Synthetic SFRT center-to-center lattice spacing.",
    )
    parser.add_argument("--n-beams-x", type=int, default=7, help="Beam copies along x.")
    parser.add_argument("--n-beams-y", type=int, default=7, help="Beam copies along y.")
    parser.add_argument(
        "--pvdr-depths-cm",
        nargs="+",
        type=float,
        default=[5.0, 10.0, 30.0],
        help="Requested fixed depths for PVDR reporting in the summary table.",
    )
    parser.add_argument(
        "--reference-depth-cm",
        type=float,
        default=5.0,
        help="Auxiliary depth passed to the analysis script for secondary reporting fields.",
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


def render_case_file(args: argparse.Namespace) -> str:
    template_text = args.template.read_text(encoding="utf-8")
    phantom_hlx_cm = 0.5 * float(args.phantom_size_x_cm)
    phantom_hly_cm = 0.5 * float(args.phantom_size_y_cm)
    phantom_hlz_cm = 0.5 * float(args.phantom_size_z_cm)
    phantom_center_z_cm = phantom_hlz_cm
    world_hlx_cm = max(40.0, float(args.phantom_size_x_cm) + 20.0)
    world_hly_cm = max(40.0, float(args.phantom_size_y_cm) + 20.0)
    world_hlz_cm = max(60.0, phantom_center_z_cm + phantom_hlz_cm + 20.0)
    show_interval = max(1000, int(args.histories) // 10)
    energy_spread_rel = (
        0.0 if float(args.photon_energy_mev) == 0.0 else float(args.energy_spread_mev) / float(args.photon_energy_mev)
    )

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
        "__PHOTON_ENERGY_MEV__": f"{float(args.photon_energy_mev):.6f}",
        "__ENERGY_SPREAD_REL__": f"{energy_spread_rel:.9f}",
        "__BEAM_POSITION_CUTOFF_X_MM__": f"{float(args.beam_position_cutoff_mm):.6f}",
        "__BEAM_POSITION_CUTOFF_Y_MM__": f"{float(args.beam_position_cutoff_mm):.6f}",
        "__N_HISTORIES__": str(int(args.histories)),
        "__N_THREADS__": str(int(args.threads)),
        "__SEED__": str(int(args.seed)),
        "__SHOW_HISTORY_INTERVAL__": str(int(show_interval)),
    }
    return fill_template(template_text, replacements)


def build_case_metadata(args: argparse.Namespace, case_dir: Path, analysis_dir: Path) -> Dict[str, object]:
    phantom_center_z_cm = 0.5 * float(args.phantom_size_z_cm)
    voxel_size_x_mm = 10.0 * float(args.phantom_size_x_cm) / float(args.xbins)
    voxel_size_y_mm = 10.0 * float(args.phantom_size_y_cm) / float(args.ybins)
    voxel_size_z_mm = 10.0 * float(args.phantom_size_z_cm) / float(args.zbins)
    return {
        "nominal_beam_label": f"{float(args.nominal_energy_mv):.1f} MV clinical LINAC beam",
        "direct_photon_model": {
            "particle": "gamma",
            "monoenergetic_photon_energy_mev": float(args.photon_energy_mev),
            "energy_spread_mev": float(args.energy_spread_mev),
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
            "pitch_mm": float(args.pitch_mm),
            "n_beams_x": int(args.n_beams_x),
            "n_beams_y": int(args.n_beams_y),
            "prescribed_peak_dose_gy": float(args.prescribed_peak_dose_gy),
            "pvdr_depths_cm": [float(v) for v in args.pvdr_depths_cm],
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
            "This is a mathematical direct-photon baseline, not a full clinical linac head model.",
            "The requested TOPAS-style Direction angular mode was implemented as BeamAngularDistribution=None because that is the supported Beam-source parallel-beam option in TOPAS docs.",
            "The photon source is placed 0.1 cm upstream of the phantom entrance using a dedicated geometry group.",
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
            "TOPAS run failed for the direct-photon 6 MV case.\n"
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
        "linac_6mv_direct_photon",
        "--beam-label",
        "Direct 6 MeV mathematical photon beam",
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
        f"{float(args.reference_depth_cm):.6f}",
        "--alpha",
        f"{float(args.alpha):.6f}",
        "--beta",
        f"{float(args.beta):.6f}",
        "--figure1-title",
        "Figure 1: Physical lateral dose profile for the direct 6 MeV photon SFRT lattice",
        "--figure2-title",
        "Figure 2: Standard LQ survival slice for the direct 6 MeV photon SFRT lattice",
        "--dpi",
        str(int(args.dpi)),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, env=env)
    combined_log = (result.stdout or "") + ("\n" + result.stderr if result.stderr else "")
    write_text_with_retries(analysis_dir / "analysis.log", combined_log)
    if result.returncode != 0:
        tail = "\n".join(combined_log.strip().splitlines()[-30:])
        raise RuntimeError(f"Analysis failed for the direct-photon 6 MV case.\n{tail}")
    return analysis_dir / "summary.json"


def nearest_pvdr_rows(curve_path: Path, requested_depths_cm: List[float]) -> tuple[List[Dict[str, object]], float]:
    with curve_path.open("r", encoding="utf-8", newline="") as handle:
        raw_rows = list(csv.DictReader(handle))
    parsed = [
        {
            "depth_cm": float(row["depth_cm"]),
            "pvdr": float(row["pvdr"]),
        }
        for row in raw_rows
    ]
    max_depth_cm = max(row["depth_cm"] for row in parsed)
    rows: List[Dict[str, object]] = []
    for depth in requested_depths_cm:
        if max_depth_cm < float(depth):
            rows.append(
                {
                    "target_cm": float(depth),
                    "sampled_depth_cm": None,
                    "pvdr": None,
                    "note": f"Unavailable; phantom ends at {max_depth_cm:.3f} cm",
                }
            )
            continue
        nearest = min(parsed, key=lambda row: abs(row["depth_cm"] - float(depth)))
        rows.append(
            {
                "target_cm": float(depth),
                "sampled_depth_cm": float(nearest["depth_cm"]),
                "pvdr": float(nearest["pvdr"]),
                "note": "Nearest voxel-center depth",
            }
        )
    return rows, max_depth_cm


def write_markdown_report(
    args: argparse.Namespace,
    metadata: Dict[str, object],
    summary: Dict[str, object],
    out_file: Path,
) -> None:
    voxel_single = summary["voxel_grid_single_beam"]
    voxel_lattice = summary["voxel_grid_lattice"]
    curve_path = Path(str(summary["pvdr"]["curve_csv"]))
    requested_rows, _ = nearest_pvdr_rows(curve_path, [float(v) for v in args.pvdr_depths_cm])
    dmax_depth_cm = float(summary["pvdr"]["depths_reported_cm"]["dmax"])
    dmax_pvdr = float(summary["pvdr"]["dmax"]["pvdr"])
    outputs = summary["outputs"]

    lines = [
        "# Direct 6 MV Photon SFRT Summary",
        "",
        f"- Nominal beam label: `{metadata['nominal_beam_label']}`.",
        (
            f"- Source model: direct mathematical `gamma` beam at `{float(args.photon_energy_mev):.3f} MeV`, "
            f"flat `10 mm` diameter aperture, zero angular spread."
        ),
        (
            f"- Histories / threads / seed: `{int(args.histories)}` / `{int(args.threads)}` / "
            f"`{int(args.seed)}`."
        ),
        (
            f"- Geometry: source plane at `z={float(args.source_z_cm):.3f} cm`, water phantom "
            f"`{float(args.phantom_size_x_cm):.1f} x {float(args.phantom_size_y_cm):.1f} x {float(args.phantom_size_z_cm):.1f} cm^3`."
        ),
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
        f"| dmax | {dmax_depth_cm:.3f} | {dmax_pvdr:.3f} | Maximum lattice depth-dose slice |",
    ]

    for row in requested_rows:
        target_label = f"{row['target_cm']:.0f} cm"
        if row["sampled_depth_cm"] is None or row["pvdr"] is None:
            lines.append(f"| {target_label} | -- | -- | {row['note']} |")
        else:
            lines.append(
                f"| {target_label} | {float(row['sampled_depth_cm']):.3f} | "
                f"{float(row['pvdr']):.3f} | {row['note']} |"
            )

    lines.extend(
        [
            f"- Dose CSV: `{summary['input_csv']}`.",
            f"- Figure 1: `{outputs['figure1_physical']}`.",
            f"- Figure 2: `{outputs['figure2_standard_bio']}`.",
            f"- PVDR depth curve: `{summary['pvdr']['curve_csv']}`.",
        ]
    )
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

    rendered_case = render_case_file(args)
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
