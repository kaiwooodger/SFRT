#!/usr/bin/env python3
"""Auto-tune asymmetric quadrupole-beamline Q4 gradient in TOPAS until Whitmore pass."""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from analyze_topas_outputs import (
    benchmark_for_energy,
    extract_metrics_from_grid,
    load_topas_grid,
    score_against_benchmark,
)
from build_asymmetric_sweep import compute_layout, fill_template, load_reference
from build_asymmetric_sweep import format_physics_modules


@dataclass
class CandidateResult:
    energy_mev: int
    g4_t_per_m: float
    case_id: str
    parameter_file: Path
    dose_csv: Path
    log_file: Path
    return_code: int
    run_ok: bool
    metrics: Optional[Dict[str, float]]
    score: Optional[Dict[str, float]]


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description=(
            "Iteratively tune Q4 gradient for the asymmetric 4-quad beamline "
            "to match Whitmore metrics."
        )
    )
    parser.add_argument(
        "--reference",
        type=Path,
        default=root / "config" / "benchmark_reference.json",
        help="Whitmore/reference JSON.",
    )
    parser.add_argument(
        "--template",
        type=Path,
        default=root / "topas" / "asymmetric_4quad_template.txt",
        help="TOPAS template file.",
    )
    parser.add_argument(
        "--run-root",
        type=Path,
        default=root / "runs" / "autotune",
        help="Output root for autotune sessions.",
    )
    parser.add_argument(
        "--session",
        type=str,
        default="",
        help="Optional session folder name. Default: UTC timestamp.",
    )
    parser.add_argument(
        "--energies",
        nargs="+",
        type=int,
        default=[100, 200, 250],
        help="Energies (MeV) to tune.",
    )
    parser.add_argument(
        "--initial-mode",
        choices=["paper_sweep", "baseline_only"],
        default="paper_sweep",
        help=(
            "paper_sweep: evaluate all Supplementary Table 3 points first. "
            "baseline_only: start from baseline Q4 only."
        ),
    )
    parser.add_argument(
        "--initial-step",
        type=float,
        default=float("nan"),
        help="Initial refinement step in T/m. Default: inferred from paper sweep spacing.",
    )
    parser.add_argument(
        "--min-step",
        type=float,
        default=0.05,
        help="Stop refinement when step drops below this (T/m).",
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=8,
        help="Maximum refinement iterations after initial sweep.",
    )
    parser.add_argument(
        "--shrink",
        type=float,
        default=0.5,
        help="Refinement step multiplier per iteration (0<shrink<1).",
    )
    parser.add_argument(
        "--histories",
        type=int,
        default=100000,
        help="Histories per TOPAS run.",
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
        "--g4-data-dir",
        type=str,
        default="/Applications/GEANT4",
        help="Path containing Geant4 data folders (for Ts/G4DataDirectory).",
    )
    parser.add_argument(
        "--physics-profile",
        choices=["topas_default", "em_opt4_only", "em_opt0_only"],
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
    parser.add_argument("--topas-bin", type=str, default="topas", help="TOPAS command.")
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip rerun if CSV already exists for a candidate case.",
    )
    parser.add_argument(
        "--tol-z-cm", type=float, default=0.5, help="Tolerance for z_hat (cm)."
    )
    parser.add_argument(
        "--tol-sigma-x-cm", type=float, default=0.2, help="Tolerance for sigma_x (cm)."
    )
    parser.add_argument(
        "--tol-sigma-y-cm", type=float, default=0.2, help="Tolerance for sigma_y (cm)."
    )
    parser.add_argument(
        "--tol-entrance-pct",
        type=float,
        default=5.0,
        help="Tolerance for entrance dose percentage points.",
    )
    parser.add_argument(
        "--entrance-mode",
        choices=["on_axis", "plane_max"],
        default="on_axis",
        help="Which entrance metric to use in score/pass test.",
    )
    return parser.parse_args()


def case_id(energy_mev: int, g4_t_per_m: float) -> str:
    g_tag = f"{g4_t_per_m:+.4f}".replace("+", "p").replace("-", "m").replace(".", "p")
    return f"E{energy_mev}_{g_tag}"


def infer_step_from_values(values: List[float]) -> float:
    uniq = sorted(set(values))
    if len(uniq) < 2:
        return 0.5
    diffs = [b - a for a, b in zip(uniq[:-1], uniq[1:]) if (b - a) > 1e-9]
    if not diffs:
        return 0.5
    return min(diffs)


def build_case_file(
    template_text: str,
    reference: Dict,
    energy_mev: int,
    g4_t_per_m: float,
    args: argparse.Namespace,
    case_dir: Path,
) -> Tuple[Path, Path]:
    beamline = reference["asymmetric_beamline"]
    drifts_cm = beamline["drifts_cm"]
    quad_length_cm = float(beamline["quad_length_cm"])
    phantom_hlz_cm = float(beamline["phantom_size_cm"]["z"]) / 2.0
    layout = compute_layout(drifts_cm, quad_length_cm, phantom_hlz_cm)

    entry = beamline["energies"][str(energy_mev)]
    baseline_g = [float(v) for v in entry["baseline_gradients_t_per_m"]]
    gradients = [baseline_g[0], baseline_g[1], baseline_g[2], float(g4_t_per_m)]
    gradients_x = [g * args.gradient_x_scale for g in gradients]
    gradients_y = [g * args.gradient_y_scale for g in gradients]

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
    energy_spread_rel = source_energy_spread_mev / float(energy_mev)
    show_interval = max(1, args.histories // 10)

    replacements = {
        "__Q1_Z_CM__": f"{layout['q1_z_cm']:.6f}",
        "__Q2_Z_CM__": f"{layout['q2_z_cm']:.6f}",
        "__Q3_Z_CM__": f"{layout['q3_z_cm']:.6f}",
        "__Q4_Z_CM__": f"{layout['q4_z_cm']:.6f}",
        "__PHANTOM_CENTER_Z_CM__": f"{layout['phantom_center_z_cm']:.6f}",
        "__PHYSICS_MODULES__": format_physics_modules(args.physics_profile),
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
        "__OUTPUT_STEM__": "dose",
        "__ENERGY_MEV__": f"{float(energy_mev):.6f}",
        "__ENERGY_SPREAD_REL__": f"{energy_spread_rel:.10f}",
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
    parameter_file = case_dir / "beamline.txt"
    parameter_file.write_text(parameter_text, encoding="utf-8")
    dose_csv = case_dir / "dose.csv"
    return parameter_file, dose_csv


def run_topas(topas_bin: str, case_dir: Path, parameter_file: Path, log_file: Path) -> int:
    proc = subprocess.run(
        [topas_bin, parameter_file.name],
        cwd=str(case_dir),
        capture_output=True,
        text=True,
    )
    combined = (proc.stdout or "") + ("\n" + proc.stderr if proc.stderr else "")
    log_file.write_text(combined, encoding="utf-8")
    return int(proc.returncode)


def evaluate_candidate(
    template_text: str,
    reference: Dict,
    energy_mev: int,
    g4_t_per_m: float,
    args: argparse.Namespace,
    session_case_root: Path,
    cache: Dict[Tuple[int, float], CandidateResult],
) -> CandidateResult:
    key = (energy_mev, round(g4_t_per_m, 6))
    if key in cache:
        return cache[key]

    cid = case_id(energy_mev, g4_t_per_m)
    case_dir = session_case_root / cid
    case_dir.mkdir(parents=True, exist_ok=True)
    parameter_file, dose_csv = build_case_file(
        template_text=template_text,
        reference=reference,
        energy_mev=energy_mev,
        g4_t_per_m=g4_t_per_m,
        args=args,
        case_dir=case_dir,
    )
    log_file = case_dir / "topas.log"

    return_code = 0
    if args.skip_existing and dose_csv.exists():
        run_ok = True
    else:
        return_code = run_topas(args.topas_bin, case_dir, parameter_file, log_file)
        run_ok = return_code == 0 and dose_csv.exists()

    metrics = None
    score = None
    if run_ok:
        grid, header = load_topas_grid(dose_csv)
        metrics = extract_metrics_from_grid(
            grid=grid,
            dx_cm=float(header["dx_cm"]),
            dy_cm=float(header["dy_cm"]),
            dz_cm=float(header["dz_cm"]),
        )
        benchmark = benchmark_for_energy(reference, energy_mev)
        if benchmark is None:
            raise ValueError(f"No benchmark entry for energy {energy_mev}")
        score = score_against_benchmark(
            metrics=metrics,
            benchmark=benchmark,
            entrance_mode=args.entrance_mode,
            tol_z_cm=args.tol_z_cm,
            tol_sx_cm=args.tol_sigma_x_cm,
            tol_sy_cm=args.tol_sigma_y_cm,
            tol_ent_pct=args.tol_entrance_pct,
        )

    result = CandidateResult(
        energy_mev=energy_mev,
        g4_t_per_m=float(g4_t_per_m),
        case_id=cid,
        parameter_file=parameter_file,
        dose_csv=dose_csv,
        log_file=log_file,
        return_code=return_code,
        run_ok=run_ok,
        metrics=metrics,
        score=score,
    )
    cache[key] = result
    return result


def choose_best(results: List[CandidateResult]) -> CandidateResult:
    valid = [r for r in results if r.run_ok and r.score is not None]
    if not valid:
        raise RuntimeError("No valid candidate results available.")
    return min(valid, key=lambda r: float(r.score["weighted_error"]))


def write_outputs(
    session_dir: Path,
    all_results: List[CandidateResult],
    best_by_energy: Dict[int, CandidateResult],
) -> None:
    json_path = session_dir / "autotune_summary.json"
    csv_path = session_dir / "autotune_history.csv"
    best_csv_path = session_dir / "best_by_energy.csv"

    serializable = []
    for r in all_results:
        row = {
            "energy_mev": r.energy_mev,
            "g4_t_per_m": r.g4_t_per_m,
            "case_id": r.case_id,
            "parameter_file": str(r.parameter_file.resolve()),
            "dose_csv": str(r.dose_csv.resolve()),
            "log_file": str(r.log_file.resolve()),
            "return_code": r.return_code,
            "run_ok": r.run_ok,
        }
        if r.metrics:
            row.update(r.metrics)
        if r.score:
            row.update(r.score)
        serializable.append(row)

    payload = {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "num_results": len(serializable),
        "best_by_energy": {
            str(e): {
                "case_id": b.case_id,
                "g4_t_per_m": b.g4_t_per_m,
                "within_tolerance": bool(b.score and b.score.get("within_tolerance")),
                "weighted_error": float(b.score["weighted_error"]) if b.score else float("nan"),
            }
            for e, b in sorted(best_by_energy.items())
        },
        "results": serializable,
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    if serializable:
        fields = sorted({k for row in serializable for k in row.keys()})
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(serializable)

    best_rows = []
    for energy, b in sorted(best_by_energy.items()):
        row = {
            "energy_mev": energy,
            "case_id": b.case_id,
            "g4_t_per_m": b.g4_t_per_m,
            "run_ok": b.run_ok,
        }
        if b.metrics:
            row.update(b.metrics)
        if b.score:
            row.update(b.score)
        best_rows.append(row)
    if best_rows:
        fields = sorted({k for row in best_rows for k in row.keys()})
        with best_csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(best_rows)


def tune_energy(
    energy_mev: int,
    template_text: str,
    reference: Dict,
    args: argparse.Namespace,
    session_case_root: Path,
    all_results: List[CandidateResult],
) -> CandidateResult:
    entry = reference["asymmetric_beamline"]["energies"][str(energy_mev)]
    baseline_g4 = float(entry["baseline_gradients_t_per_m"][3])
    paper_points = [float(row["g4_t_per_m"]) for row in entry["supplementary_table3_scan"]]

    cache: Dict[Tuple[int, float], CandidateResult] = {}
    evaluated: List[CandidateResult] = []

    if args.initial_mode == "paper_sweep":
        start_values = sorted(set(paper_points))
    else:
        start_values = [baseline_g4]

    print(f"[E={energy_mev}] Initial sweep points: {', '.join(f'{v:.4f}' for v in start_values)}")
    for g4 in start_values:
        result = evaluate_candidate(
            template_text=template_text,
            reference=reference,
            energy_mev=energy_mev,
            g4_t_per_m=g4,
            args=args,
            session_case_root=session_case_root,
            cache=cache,
        )
        evaluated.append(result)
        all_results.append(result)
        if result.run_ok and result.score:
            print(
                f"  g4={g4:.4f} T/m | error={result.score['weighted_error']:.3f} | "
                f"{'PASS' if result.score['within_tolerance'] else 'FAIL'}"
            )
        else:
            print(f"  g4={g4:.4f} T/m | TOPAS failed")

    best = choose_best(evaluated)
    if best.score and best.score["within_tolerance"]:
        return best

    if math.isnan(args.initial_step):
        step = infer_step_from_values(start_values)
        if step <= 0:
            step = infer_step_from_values(paper_points)
        if step <= 0:
            step = 0.5
    else:
        step = float(args.initial_step)

    for iteration in range(1, args.max_iter + 1):
        if step < args.min_step:
            break
        center = best.g4_t_per_m
        candidates = sorted(set([center - step, center, center + step]))
        print(
            f"[E={energy_mev}] Iter {iteration}: center={center:.4f}, step={step:.4f}, "
            f"candidates={', '.join(f'{c:.4f}' for c in candidates)}"
        )

        for g4 in candidates:
            result = evaluate_candidate(
                template_text=template_text,
                reference=reference,
                energy_mev=energy_mev,
                g4_t_per_m=g4,
                args=args,
                session_case_root=session_case_root,
                cache=cache,
            )
            if result not in evaluated:
                evaluated.append(result)
                all_results.append(result)
            if result.run_ok and result.score:
                print(
                    f"  g4={g4:.4f} T/m | error={result.score['weighted_error']:.3f} | "
                    f"{'PASS' if result.score['within_tolerance'] else 'FAIL'}"
                )
            else:
                print(f"  g4={g4:.4f} T/m | TOPAS failed")

        new_best = choose_best(evaluated)
        if new_best.score and new_best.score["within_tolerance"]:
            best = new_best
            break

        # If there is no improvement, shrink step to fine tune.
        if new_best.case_id == best.case_id:
            step *= args.shrink
        else:
            best = new_best
            step *= args.shrink

    return best


def main() -> int:
    args = parse_args()
    if not (0.0 < args.shrink < 1.0):
        raise ValueError("--shrink must be in (0, 1).")

    reference = load_reference(args.reference)
    template_text = args.template.read_text(encoding="utf-8")

    session_name = (
        args.session.strip()
        if args.session.strip()
        else datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    )
    session_dir = args.run_root / session_name
    session_case_root = session_dir / "cases"
    session_case_root.mkdir(parents=True, exist_ok=True)

    all_results: List[CandidateResult] = []
    best_by_energy: Dict[int, CandidateResult] = {}

    for energy in args.energies:
        if str(energy) not in reference["asymmetric_beamline"]["energies"]:
            raise ValueError(f"Energy {energy} MeV not found in reference.")
        best = tune_energy(
            energy_mev=energy,
            template_text=template_text,
            reference=reference,
            args=args,
            session_case_root=session_case_root,
            all_results=all_results,
        )
        best_by_energy[energy] = best
        status = "PASS" if best.score and best.score["within_tolerance"] else "FAIL"
        err = best.score["weighted_error"] if best.score else float("nan")
        print(
            f"[E={energy}] Best g4={best.g4_t_per_m:.4f} T/m | "
            f"error={err:.3f} | {status}"
        )

    write_outputs(session_dir=session_dir, all_results=all_results, best_by_energy=best_by_energy)
    print(f"Session output: {session_dir}")
    print(f"Summary JSON: {session_dir / 'autotune_summary.json'}")
    print(f"History CSV: {session_dir / 'autotune_history.csv'}")
    print(f"Best CSV: {session_dir / 'best_by_energy.csv'}")

    all_pass = all(
        bool(best.score and best.score.get("within_tolerance"))
        for best in best_by_energy.values()
    )
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
