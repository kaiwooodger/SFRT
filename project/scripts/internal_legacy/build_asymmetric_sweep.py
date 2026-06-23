#!/usr/bin/env python3
"""Generate and optionally run TOPAS asymmetric 4-quadrupole beamline sweeps."""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List


PHYSICS_PROFILES = {
    "topas_default": [
        "g4em-standard_opt4",
        "g4h-phy_QGSP_BIC_HP",
        "g4decay",
        "g4ion-binarycascade",
        "g4h-elastic_HP",
        "g4stopping",
    ],
    "em_opt4_only": ["g4em-standard_opt4"],
    "em_opt0_only": ["g4em-standard_opt0"],
}


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description=(
            "Build TOPAS input decks for the asymmetric 4-quad beamline beamline "
            "and optionally execute the sweep."
        )
    )
    parser.add_argument(
        "--reference",
        type=Path,
        default=root / "config" / "benchmark_reference.json",
        help="JSON reference file containing beamline and Whitmore data.",
    )
    parser.add_argument(
        "--template",
        type=Path,
        default=root / "topas" / "asymmetric_4quad_template.txt",
        help="TOPAS template with placeholders.",
    )
    parser.add_argument(
        "--run-root",
        type=Path,
        default=root / "runs",
        help="Output directory for generated case folders and manifest.",
    )
    parser.add_argument(
        "--energies",
        nargs="+",
        type=int,
        default=[100, 200, 250],
        help="Beam energies (MeV) to generate.",
    )
    parser.add_argument(
        "--g4-range",
        type=str,
        default="",
        help=(
            "Override paper scan with min:max:step for Q4 gradient in T/m, "
            "applied to each selected energy."
        ),
    )
    parser.add_argument(
        "--nominal-only",
        action="store_true",
        help="Only generate the baseline Table 2 Q4 value for each energy.",
    )
    parser.add_argument(
        "--histories",
        type=int,
        default=100000,
        help="Number of particle histories per TOPAS run.",
    )
    parser.add_argument("--threads", type=int, default=1, help="TOPAS thread count.")
    parser.add_argument("--seed", type=int, default=1, help="TOPAS RNG seed.")
    parser.add_argument("--xbins", type=int, default=101, help="Phantom X bins.")
    parser.add_argument("--ybins", type=int, default=101, help="Phantom Y bins.")
    parser.add_argument("--zbins", type=int, default=101, help="Phantom Z bins.")
    parser.add_argument(
        "--source-sigma-x-mm",
        type=float,
        default=None,
        help="Initial beam Gaussian sigma in X (mm). Default: reference value.",
    )
    parser.add_argument(
        "--source-sigma-y-mm",
        type=float,
        default=None,
        help="Initial beam Gaussian sigma in Y (mm). Default: reference value.",
    )
    parser.add_argument(
        "--source-angular-x-mrad",
        type=float,
        default=None,
        help="Initial beam angular spread in X (mrad). Default: reference value.",
    )
    parser.add_argument(
        "--source-angular-y-mrad",
        type=float,
        default=None,
        help="Initial beam angular spread in Y (mrad). Default: reference value.",
    )
    parser.add_argument(
        "--source-energy-spread-mev",
        type=float,
        default=None,
        help="Initial beam absolute energy spread (MeV). Default: reference value.",
    )
    parser.add_argument(
        "--cut-gamma-mm",
        type=float,
        default=0.01,
        help="Geant4 production cut for photons (mm).",
    )
    parser.add_argument(
        "--cut-electron-mm",
        type=float,
        default=0.01,
        help="Geant4 production cut for electrons (mm).",
    )
    parser.add_argument(
        "--cut-positron-mm",
        type=float,
        default=0.01,
        help="Geant4 production cut for positrons (mm).",
    )
    parser.add_argument(
        "--max-step-quad-mm",
        type=float,
        default=0.5,
        help="Maximum tracking step size in quadrupole volumes (mm).",
    )
    parser.add_argument(
        "--max-step-phantom-mm",
        type=float,
        default=0.1,
        help="Maximum tracking step size in the water phantom (mm).",
    )
    parser.add_argument(
        "--g4-data-dir",
        type=str,
        default="/Applications/GEANT4",
        help="Path containing Geant4 data folders (for Ts/G4DataDirectory).",
    )
    parser.add_argument(
        "--physics-profile",
        choices=sorted(PHYSICS_PROFILES),
        default="topas_default",
        help="Named TOPAS modular physics profile to write into each deck.",
    )
    parser.add_argument(
        "--gradient-x-scale",
        type=float,
        default=1.0,
        help="Multiplier applied to all quadrupole gradients written to MagneticFieldGradientX.",
    )
    parser.add_argument(
        "--gradient-y-scale",
        type=float,
        default=1.0,
        help="Multiplier applied to all quadrupole gradients written to MagneticFieldGradientY.",
    )
    parser.add_argument(
        "--quad-gradient-convention",
        choices=["scaled", "ideal_opposite"],
        default="ideal_opposite",
        help=(
            "How MagneticFieldGradientX/Y are derived from paper gradients. "
            "scaled: Gx=g*gradient-x-scale, Gy=g*gradient-y-scale (legacy); "
            "ideal_opposite: Gy=-Gx (paper-consistent quadrupole plane convention)."
        ),
    )
    parser.add_argument("--source-x-cm", type=float, default=0.0, help="Beam source X position in cm.")
    parser.add_argument("--source-y-cm", type=float, default=0.0, help="Beam source Y position in cm.")
    parser.add_argument("--source-z-cm", type=float, default=0.0, help="Beam source Z position in cm.")
    parser.add_argument(
        "--source-rotx-deg",
        type=float,
        default=0.0,
        help="Beam source rotation about X in degrees.",
    )
    parser.add_argument(
        "--source-roty-deg",
        type=float,
        default=0.0,
        help="Beam source rotation about Y in degrees.",
    )
    parser.add_argument(
        "--source-rotz-deg",
        type=float,
        default=0.0,
        help="Beam source rotation about Z in degrees.",
    )
    parser.add_argument(
        "--run-topas",
        action="store_true",
        help="Execute TOPAS for each generated case.",
    )
    parser.add_argument(
        "--topas-bin",
        type=str,
        default="topas",
        help="TOPAS executable/wrapper command.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip execution when the expected CSV output already exists.",
    )
    parser.add_argument(
        "--case-run-retries",
        type=int,
        default=2,
        help="Retry count per TOPAS case on failure or invalid/empty CSV output.",
    )
    parser.add_argument(
        "--max-cases",
        type=int,
        default=0,
        help="If >0, only process the first N generated cases (useful for smoke tests).",
    )
    return parser.parse_args()


def load_reference(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def case_id_for(energy_mev: int, g4_tpm: float) -> str:
    g4_tag = f"{g4_tpm:+.2f}".replace("+", "p").replace("-", "m").replace(".", "p")
    return f"E{energy_mev}_{g4_tag}"


def parse_g4_range(raw: str) -> List[float]:
    pieces = raw.split(":")
    if len(pieces) != 3:
        raise ValueError("g4-range must be min:max:step")
    g_min, g_max, g_step = (float(p.strip()) for p in pieces)
    if g_step <= 0:
        raise ValueError("g4-range step must be > 0")
    if g_max < g_min:
        raise ValueError("g4-range max must be >= min")
    values: List[float] = []
    index = 0
    while True:
        value = g_min + index * g_step
        if value > g_max + 1e-9:
            break
        values.append(round(value, 6))
        index += 1
    return values


def compute_layout(drifts_cm: List[float], quad_length_cm: float, phantom_hlz_cm: float) -> Dict[str, float]:
    if len(drifts_cm) != 5:
        raise ValueError("Asymmetric 4-quad layout expects 5 drift lengths [s1..s5].")

    z = 0.0
    q_centers = []
    for i in range(4):
        z += drifts_cm[i]
        q_centers.append(z + quad_length_cm / 2.0)
        z += quad_length_cm
    z += drifts_cm[4]
    phantom_center = z + phantom_hlz_cm
    return {
        "q1_z_cm": q_centers[0],
        "q2_z_cm": q_centers[1],
        "q3_z_cm": q_centers[2],
        "q4_z_cm": q_centers[3],
        "phantom_center_z_cm": phantom_center,
    }


def build_quadrupole_gradients(gradients: List[float], args: argparse.Namespace) -> tuple[List[float], List[float]]:
    gradients_x = [g * args.gradient_x_scale for g in gradients]
    if args.quad_gradient_convention == "ideal_opposite":
        if abs(args.gradient_y_scale - 1.0) > 1e-12:
            raise ValueError(
                "--gradient-y-scale must be 1.0 when --quad-gradient-convention=ideal_opposite; "
                "Gy is constrained to -Gx."
            )
        gradients_y = [-gx for gx in gradients_x]
    else:
        gradients_y = [g * args.gradient_y_scale for g in gradients]
    return gradients_x, gradients_y


def fill_template(template: str, replacements: Dict[str, str]) -> str:
    result = template
    for key, value in replacements.items():
        result = result.replace(key, value)
    return result


def format_physics_modules(profile_name: str) -> str:
    modules = PHYSICS_PROFILES[profile_name]
    quoted = " ".join(f'"{module}"' for module in modules)
    return f"{len(modules)} {quoted}"


def discover_g4_data_env(g4_data_dir: str) -> Dict[str, str]:
    base = Path(g4_data_dir).expanduser()
    search_roots = [
        base,
        base / "G4DATA",
        base / "geant4-install" / "share" / "Geant4" / "data",
    ]
    available_roots = [root for root in search_roots if root.exists() and root.is_dir()]
    if not available_roots:
        return {}

    env_to_prefixes = {
        "G4NEUTRONHPDATA": ["G4NDL"],
        "G4PARTICLEXSDATA": ["G4PARTICLEXS"],
        "G4PIIDATA": ["G4PII"],
        "G4LEVELGAMMADATA": ["PhotonEvaporation", "G4PhotonEvaporation"],
        "G4RADIOACTIVEDATA": ["RadioactiveDecay", "G4RadioactiveDecay"],
        "G4LEDATA": ["G4EMLOW"],
        "G4SAIDXSDATA": ["G4SAIDDATA"],
        "G4REALSURFACEDATA": ["RealSurface"],
        "G4ABLADATA": ["G4ABLA"],
        "G4INCLDATA": ["G4INCL"],
        "G4ENSDFSTATEDATA": ["G4ENSDFSTATE"],
        "G4CHANNELINGDATA": ["G4CHANNELING"],
    }

    resolved: Dict[str, str] = {}
    for env_var, prefixes in env_to_prefixes.items():
        for root in available_roots:
            matches = sorted(
                [
                    child
                    for child in root.iterdir()
                    if child.is_dir() and any(child.name.startswith(prefix) for prefix in prefixes)
                ],
                key=lambda p: p.name,
                reverse=True,
            )
            if matches:
                resolved[env_var] = str(matches[0].resolve())
                break

    # Compatibility alias seen in some Geant4/TOPAS stacks.
    if "G4NEUTRONXSDATA" not in resolved:
        if "G4PARTICLEXSDATA" in resolved:
            resolved["G4NEUTRONXSDATA"] = resolved["G4PARTICLEXSDATA"]
        elif "G4NEUTRONHPDATA" in resolved:
            resolved["G4NEUTRONXSDATA"] = resolved["G4NEUTRONHPDATA"]

    return resolved


def build_topas_env(g4_data_dir: str) -> Dict[str, str]:
    env = os.environ.copy()
    env["TOPAS_G4_DATA_DIR"] = str(Path(g4_data_dir).expanduser())
    env.update(discover_g4_data_env(g4_data_dir))
    return env


def run_case(
    topas_bin: str,
    case_dir: Path,
    parameter_file: Path,
    g4_data_dir: str,
) -> subprocess.CompletedProcess:
    return subprocess.run(
        [topas_bin, parameter_file.name],
        cwd=str(case_dir),
        capture_output=True,
        text=True,
        env=build_topas_env(g4_data_dir),
    )


def write_text_with_retries(
    path: Path,
    content: str,
    retries: int = 8,
    retry_delay_sec: float = 0.75,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    attempts = max(1, int(retries))
    delay = max(0.0, float(retry_delay_sec))
    last_error: Exception | None = None

    for attempt in range(1, attempts + 1):
        try:
            tmp_path.write_text(content, encoding="utf-8")
            tmp_path.replace(path)
            return
        except (OSError, TimeoutError) as exc:
            last_error = exc
            if attempt >= attempts:
                break
            time.sleep(delay)

    if last_error is not None:
        raise last_error
    raise OSError(f"Failed to write {path}")


def has_nonempty_output(path: Path) -> bool:
    try:
        return path.exists() and path.stat().st_size > 0
    except OSError:
        return False


def remove_file_if_exists(path: Path) -> None:
    try:
        if path.exists():
            path.unlink()
    except OSError:
        pass


def write_manifest(path: Path, payload: Dict) -> None:
    write_text_with_retries(path, json.dumps(payload, indent=2))


def main() -> int:
    args = parse_args()
    if args.quad_gradient_convention == "scaled":
        print(
            "WARNING: --quad-gradient-convention=scaled sets Gx and Gy with the same sign by default. "
            "For paper-consistent asymmetric quadrupole optics, use ideal_opposite."
        )
    reference = load_reference(args.reference)
    beamline = reference["asymmetric_beamline"]
    template_text = args.template.read_text(encoding="utf-8")

    drifts_cm = beamline["drifts_cm"]
    quad_length_cm = float(beamline["quad_length_cm"])
    phantom_hlz_cm = float(beamline["phantom_size_cm"]["z"]) / 2.0
    initial_beam = beamline["initial_beam"]
    source_sigma_x_mm = (
        float(args.source_sigma_x_mm)
        if args.source_sigma_x_mm is not None
        else float(initial_beam["sigma_mm"])
    )
    source_sigma_y_mm = (
        float(args.source_sigma_y_mm)
        if args.source_sigma_y_mm is not None
        else float(initial_beam["sigma_mm"])
    )
    source_angular_x_mrad = (
        float(args.source_angular_x_mrad)
        if args.source_angular_x_mrad is not None
        else float(initial_beam["angular_divergence_mrad"])
    )
    source_angular_y_mrad = (
        float(args.source_angular_y_mrad)
        if args.source_angular_y_mrad is not None
        else float(initial_beam["angular_divergence_mrad"])
    )
    source_energy_spread_mev = (
        float(args.source_energy_spread_mev)
        if args.source_energy_spread_mev is not None
        else float(initial_beam["energy_spread_mev"])
    )
    layout = compute_layout(drifts_cm, quad_length_cm, phantom_hlz_cm)
    detected_g4_env = discover_g4_data_env(args.g4_data_dir)
    required = ("G4NEUTRONHPDATA", "G4PARTICLEXSDATA", "G4NEUTRONXSDATA")
    missing = [key for key in required if key not in detected_g4_env]
    if missing:
        print(
            "WARNING: Missing Geant4 dataset env mappings for: "
            + ", ".join(missing)
            + f" under --g4-data-dir {args.g4_data_dir}"
        )

    print(
        "Layout (from drifts s1..s5, not direct centers): "
        f"Q1={layout['q1_z_cm']:.3f} cm, "
        f"Q2={layout['q2_z_cm']:.3f} cm, "
        f"Q3={layout['q3_z_cm']:.3f} cm, "
        f"Q4={layout['q4_z_cm']:.3f} cm, "
        f"PhantomCenter={layout['phantom_center_z_cm']:.3f} cm"
    )
    if abs(args.source_z_cm) > 1e-12:
        print(
            "WARNING: --source-z-cm is non-zero. This shifts beam source relative to fixed lattice "
            "centers from s1..s5 conversion and can change focal behavior."
        )

    use_custom_g4 = bool(args.g4_range.strip())
    custom_g4_values = parse_g4_range(args.g4_range) if use_custom_g4 else []

    run_root = args.run_root
    case_root = run_root / "cases"
    case_root.mkdir(parents=True, exist_ok=True)

    manifest_cases = []
    for energy in args.energies:
        key = str(energy)
        if key not in beamline["energies"]:
            raise ValueError(f"Energy {energy} MeV is not available in reference data.")

        entry = beamline["energies"][key]
        baseline_g = [float(v) for v in entry["baseline_gradients_t_per_m"]]
        benchmark = entry["benchmark_metrics_table2"]
        table3 = entry["supplementary_table3_scan"]

        if args.nominal_only:
            g4_values = [baseline_g[3]]
        elif use_custom_g4:
            g4_values = custom_g4_values
        else:
            g4_values = [float(row["g4_t_per_m"]) for row in table3]

        for g4 in g4_values:
            gradients = [baseline_g[0], baseline_g[1], baseline_g[2], float(g4)]
            gradients_x, gradients_y = build_quadrupole_gradients(gradients, args)
            cid = case_id_for(energy, g4)
            case_dir = case_root / cid
            case_dir.mkdir(parents=True, exist_ok=True)

            parameter_file = case_dir / "beamline.txt"
            dose_stem = "dose"
            dose_csv = case_dir / f"{dose_stem}.csv"

            energy_spread_rel = source_energy_spread_mev / float(energy)
            show_interval = max(1, args.histories // 10)

            replacements = {
                "__Q1_Z_CM__": f"{layout['q1_z_cm']:.6f}",
                "__Q2_Z_CM__": f"{layout['q2_z_cm']:.6f}",
                "__Q3_Z_CM__": f"{layout['q3_z_cm']:.6f}",
                "__Q4_Z_CM__": f"{layout['q4_z_cm']:.6f}",
                "__PHANTOM_CENTER_Z_CM__": f"{layout['phantom_center_z_cm']:.6f}",
                "__PHYSICS_MODULES__": format_physics_modules(args.physics_profile),
                "__CUT_GAMMA_MM__": f"{args.cut_gamma_mm:.6f}",
                "__CUT_ELECTRON_MM__": f"{args.cut_electron_mm:.6f}",
                "__CUT_POSITRON_MM__": f"{args.cut_positron_mm:.6f}",
                "__MAX_STEP_QUAD_MM__": f"{args.max_step_quad_mm:.6f}",
                "__MAX_STEP_PHANTOM_MM__": f"{args.max_step_phantom_mm:.6f}",
                "__G1X_TPM__": f"{gradients_x[0]:.6f}",
                "__G1Y_TPM__": f"{gradients_y[0]:.6f}",
                "__G2X_TPM__": f"{gradients_x[1]:.6f}",
                "__G2Y_TPM__": f"{gradients_y[1]:.6f}",
                "__G3X_TPM__": f"{gradients_x[2]:.6f}",
                "__G3Y_TPM__": f"{gradients_y[2]:.6f}",
                "__G4X_TPM__": f"{gradients_x[3]:.6f}",
                "__G4Y_TPM__": f"{gradients_y[3]:.6f}",
                "__SOURCE_X_CM__": f"{args.source_x_cm:.6f}",
                "__SOURCE_Y_CM__": f"{args.source_y_cm:.6f}",
                "__SOURCE_Z_CM__": f"{args.source_z_cm:.6f}",
                "__SOURCE_ROTX_DEG__": f"{args.source_rotx_deg:.6f}",
                "__SOURCE_ROTY_DEG__": f"{args.source_roty_deg:.6f}",
                "__SOURCE_ROTZ_DEG__": f"{args.source_rotz_deg:.6f}",
                "__SOURCE_SIGMA_X_CM__": f"{source_sigma_x_mm / 10.0:.6f}",
                "__SOURCE_SIGMA_Y_CM__": f"{source_sigma_y_mm / 10.0:.6f}",
                "__SOURCE_ANGULAR_X_MRAD__": f"{source_angular_x_mrad:.6f}",
                "__SOURCE_ANGULAR_Y_MRAD__": f"{source_angular_y_mrad:.6f}",
                "__OUTPUT_STEM__": dose_stem,
                "__ENERGY_MEV__": f"{float(energy):.6f}",
                "__ENERGY_SPREAD_REL__": f"{energy_spread_rel:.10f}",
                "__SOURCE_ENERGY_SPREAD_MEV__": f"{source_energy_spread_mev:.6f}",
                "__N_HISTORIES__": str(args.histories),
                "__N_THREADS__": str(args.threads),
                "__SEED__": str(args.seed),
                "__SHOW_HISTORY_INTERVAL__": str(show_interval),
                "__XBINS__": str(args.xbins),
                "__YBINS__": str(args.ybins),
                "__ZBINS__": str(args.zbins),
                "__G4_DATA_DIR__": str(args.g4_data_dir),
            }
            parameter_text = fill_template(template_text, replacements)
            write_text_with_retries(parameter_file, parameter_text)

            manifest_cases.append(
                {
                    "case_id": cid,
                    "energy_mev": energy,
                    "paper_gradients_t_per_m": gradients,
                    "gradient_x_t_per_m": gradients_x,
                    "gradient_y_t_per_m": gradients_y,
                    "gradient_x_scale": args.gradient_x_scale,
                    "gradient_y_scale": args.gradient_y_scale,
                    "source_sigma_x_mm": source_sigma_x_mm,
                    "source_sigma_y_mm": source_sigma_y_mm,
                    "source_angular_x_mrad": source_angular_x_mrad,
                    "source_angular_y_mrad": source_angular_y_mrad,
                    "source_energy_spread_mev": source_energy_spread_mev,
                    "source_energy_spread_relative": energy_spread_rel,
                    "cut_gamma_mm": float(args.cut_gamma_mm),
                    "cut_electron_mm": float(args.cut_electron_mm),
                    "cut_positron_mm": float(args.cut_positron_mm),
                    "max_step_quad_mm": float(args.max_step_quad_mm),
                    "max_step_phantom_mm": float(args.max_step_phantom_mm),
                    "quad_gradient_convention": args.quad_gradient_convention,
                    "parameter_file": str(parameter_file.resolve()),
                    "dose_csv": str(dose_csv.resolve()),
                    "benchmark_metrics_table2": benchmark,
                    "run_status": "generated",
                    "return_code": None,
                    "log_file": None,
                }
            )

    if args.max_cases > 0:
        manifest_cases = manifest_cases[: args.max_cases]

    manifest = {
        "reference_file": str(args.reference.resolve()),
        "template_file": str(args.template.resolve()),
        "run_root": str(run_root.resolve()),
        "histories": args.histories,
        "threads": args.threads,
        "seed": args.seed,
        "xbins": args.xbins,
        "ybins": args.ybins,
        "zbins": args.zbins,
        "g4_data_dir": str(args.g4_data_dir),
        "cut_gamma_mm": float(args.cut_gamma_mm),
        "cut_electron_mm": float(args.cut_electron_mm),
        "cut_positron_mm": float(args.cut_positron_mm),
        "max_step_quad_mm": float(args.max_step_quad_mm),
        "max_step_phantom_mm": float(args.max_step_phantom_mm),
        "drifts_cm": drifts_cm,
        "quad_length_cm": quad_length_cm,
        "layout_cm": layout,
        "quad_gradient_convention": args.quad_gradient_convention,
        "cases": manifest_cases,
    }
    manifest_path = run_root / "manifest.json"
    write_manifest(manifest_path, manifest)
    print(f"Wrote manifest: {manifest_path}")

    if args.run_topas:
        for index, case in enumerate(manifest_cases, start=1):
            case_dir = Path(case["parameter_file"]).parent
            parameter_file = Path(case["parameter_file"])
            dose_csv = Path(case["dose_csv"])
            log_file = case_dir / "topas.log"

            if args.skip_existing and has_nonempty_output(dose_csv):
                case["run_status"] = "skipped_existing"
                case["return_code"] = 0
                case["log_file"] = str(log_file.resolve())
                write_manifest(manifest_path, manifest)
                print(f"[{index}/{len(manifest_cases)}] {case['case_id']}: skip (non-empty CSV exists)")
                continue
            if args.skip_existing and dose_csv.exists():
                print(
                    f"[{index}/{len(manifest_cases)}] {case['case_id']}: "
                    "found empty/invalid CSV, rerunning"
                )

            attempts = max(1, int(args.case_run_retries) + 1)
            result = None
            combined_log = ""
            run_ok = False
            for attempt in range(1, attempts + 1):
                if attempt == 1:
                    print(f"[{index}/{len(manifest_cases)}] {case['case_id']}: running TOPAS")
                else:
                    print(
                        f"[{index}/{len(manifest_cases)}] {case['case_id']}: "
                        f"retry {attempt - 1}/{attempts - 1}"
                    )
                remove_file_if_exists(dose_csv)
                result = run_case(args.topas_bin, case_dir, parameter_file, args.g4_data_dir)
                combined_log = (result.stdout or "") + ("\n" + result.stderr if result.stderr else "")
                write_text_with_retries(
                    log_file,
                    f"=== TOPAS attempt {attempt}/{attempts} ===\n{combined_log}",
                )

                run_ok = result.returncode == 0 and has_nonempty_output(dose_csv)
                if run_ok:
                    break

                if attempt < attempts:
                    reason = (
                        f"code={result.returncode}, "
                        f"dose_csv={'non-empty' if has_nonempty_output(dose_csv) else 'missing/empty'}"
                    )
                    print(f"  WARN: {case['case_id']} attempt {attempt} failed ({reason}); retrying...")

            case["run_status"] = "ok" if run_ok else "failed"
            case["return_code"] = int(result.returncode if result is not None else 1)
            case["log_file"] = str(log_file.resolve())
            write_manifest(manifest_path, manifest)

            if not run_ok:
                tail = "\n".join(combined_log.strip().splitlines()[-20:])
                print(
                    f"  ERROR: {case['case_id']} failed "
                    f"(code {case['return_code']}, dose_csv={'non-empty' if has_nonempty_output(dose_csv) else 'missing/empty'})"
                )
                print(tail)
    write_manifest(manifest_path, manifest)

    print(f"Updated manifest: {manifest_path}")
    print(f"Generated cases: {len(manifest_cases)}")
    if args.run_topas:
        failed = [c for c in manifest_cases if c["run_status"] == "failed"]
        print(f"Successful runs: {len(manifest_cases) - len(failed)}")
        print(f"Failed runs: {len(failed)}")
        return 1 if failed else 0
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)
