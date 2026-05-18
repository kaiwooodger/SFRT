#!/usr/bin/env python3
"""Run Stage-2 SFRT workflow with frozen beamline and analysis definitions."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

import pandas as pd


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    default_python = "/opt/anaconda3/bin/python" if Path("/opt/anaconda3/bin/python").exists() else sys.executable
    parser = argparse.ArgumentParser(
        description=(
            "Enforce a frozen Stage-2 SFRT workflow (beamline + analysis definitions), "
            "and generate bounded-claim summary artifacts."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=root / "config" / "sfrt_stage2_frozen.json",
        help="Frozen Stage-2 config JSON.",
    )
    parser.add_argument(
        "--run-root",
        type=Path,
        default=root / "runs" / "sfrt_stage2_frozen",
        help="Output root for frozen Stage-2 runs and summaries.",
    )
    parser.add_argument(
        "--python-bin",
        type=str,
        default=default_python,
        help="Python interpreter used for subprocess script calls.",
    )
    parser.add_argument(
        "--topas-bin",
        type=str,
        default="/Users/kw/shellScripts/topas",
        help="TOPAS executable/wrapper command.",
    )
    parser.add_argument(
        "--g4-data-dir",
        type=str,
        default="/Applications/GEANT4",
        help="Geant4 data root path.",
    )
    parser.add_argument(
        "--run-topas",
        action="store_true",
        help="Run TOPAS for each frozen case.",
    )
    parser.add_argument(
        "--analyze-existing",
        action="store_true",
        help="Run frozen analysis on existing outputs even if --run-topas is not set.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Pass through --skip-existing to build_asymmetric_sweep.py.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Write planned commands and frozen lock files without executing scripts.",
    )
    return parser.parse_args()


def run_command(cmd: List[str], cwd: Path, dry_run: bool) -> None:
    print("[cmd]", " ".join(cmd))
    if dry_run:
        return
    proc = subprocess.run(cmd, cwd=str(cwd), text=True)
    if proc.returncode != 0:
        raise SystemExit(proc.returncode)


def load_json(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def cfg_str(value: object) -> str:
    return f"{value}"


def build_case_command(
    cfg: Dict,
    args: argparse.Namespace,
    energy_mev: int,
    g4_t_per_m: float,
    case_run_root: Path,
) -> List[str]:
    beamline = cfg["beamline_frozen"]
    source = beamline["source"]
    runtime = beamline["runtime"]
    geom = beamline["scoring_geometry"]
    g4_range = f"{float(g4_t_per_m)}:{float(g4_t_per_m)}:0.1"

    cmd = [
        args.python_bin,
        "scripts/build_asymmetric_sweep.py",
        "--run-root",
        str(case_run_root),
        "--energies",
        cfg_str(int(energy_mev)),
        "--g4-range",
        g4_range,
        "--histories",
        cfg_str(int(runtime["histories"])),
        "--threads",
        cfg_str(int(runtime["threads"])),
        "--seed",
        cfg_str(int(runtime["seed"])),
        "--source-sigma-x-mm",
        cfg_str(float(source["sigma_x_mm"])),
        "--source-sigma-y-mm",
        cfg_str(float(source["sigma_y_mm"])),
        "--source-angular-x-mrad",
        cfg_str(float(source["angular_x_mrad"])),
        "--source-angular-y-mrad",
        cfg_str(float(source["angular_y_mrad"])),
        "--source-energy-spread-mev",
        cfg_str(float(source["energy_spread_mev"])),
        "--gradient-x-scale",
        cfg_str(float(beamline["gradient_x_scale"])),
        "--gradient-y-scale",
        cfg_str(float(beamline["gradient_y_scale"])),
        "--quad-gradient-convention",
        str(beamline["quad_gradient_convention"]),
        "--xbins",
        cfg_str(int(geom["xbins"])),
        "--ybins",
        cfg_str(int(geom["ybins"])),
        "--zbins",
        cfg_str(int(geom["zbins"])),
        "--physics-profile",
        str(beamline["physics_profile"]),
        "--topas-bin",
        args.topas_bin,
        "--g4-data-dir",
        args.g4_data_dir,
    ]
    if args.run_topas:
        cmd.append("--run-topas")
    if args.skip_existing:
        cmd.append("--skip-existing")
    return cmd


def build_analysis_command(args: argparse.Namespace, cfg: Dict, case_run_root: Path) -> List[str]:
    analysis = cfg["analysis_frozen"]
    return [
        args.python_bin,
        "scripts/analyze_topas_outputs.py",
        "--manifest",
        str(case_run_root / "manifest.json"),
        "--outdir",
        str(case_run_root / "analysis_frozen"),
        "--z-mode",
        str(analysis["z_mode"]),
        "--sigma-mode",
        str(analysis["sigma_mode"]),
        "--sigma-fit-mode",
        str(analysis["sigma_fit_mode"]),
        "--entrance-mode",
        str(analysis["entrance_mode"]),
        "--io-retries",
        "10",
        "--io-retry-delay-sec",
        "1.0",
    ]


def write_summary(
    run_root: Path,
    cfg: Dict,
    commands: List[List[str]],
    analyzed_runs: List[Path],
    combined_best_file: Path,
    combined_case_metrics_file: Path,
) -> None:
    benchmark_scope = cfg["benchmark_scope"]
    claim_policy = cfg["claim_policy"]
    analysis = cfg["analysis_frozen"]
    beamline = cfg["beamline_frozen"]

    lines: List[str] = []
    lines.append("# Stage-2 Frozen SFRT Summary")
    lines.append("")
    lines.append(f"- Generated (UTC): {datetime.now(timezone.utc).isoformat(timespec='seconds')}")
    lines.append(f"- Config: `config/sfrt_stage2_frozen.json`")
    lines.append(f"- Frozen profile name: `{cfg.get('name', 'unknown')}`")
    lines.append("")
    lines.append("## Frozen Beamline Definition")
    lines.append("")
    lines.append(f"- Energies (MeV): `{beamline['energies_mev']}`")
    lines.append(f"- g4 by energy (T/m): `{beamline['g4_t_per_m_by_energy']}`")
    lines.append(f"- Quadrupole convention: `{beamline['quad_gradient_convention']}`")
    lines.append(
        f"- Gradient scales: `x={beamline['gradient_x_scale']}`, `y={beamline['gradient_y_scale']}`"
    )
    lines.append(f"- Source: `{beamline['source']}`")
    lines.append(f"- Scoring geometry: `{beamline['scoring_geometry']}`")
    lines.append(f"- Runtime defaults: `{beamline['runtime']}`")
    lines.append("")
    lines.append("## Frozen Analysis Definition")
    lines.append("")
    lines.append(f"- z definition: `{analysis['z_mode']}`")
    lines.append(f"- sigma definition: `{analysis['sigma_mode']}` with `{analysis['sigma_fit_mode']}` fit")
    lines.append(f"- peak dose definition: `{analysis['peak_dose_definition']}`")
    lines.append(f"- valley dose definition: `{analysis['valley_dose_definition']}`")
    lines.append(f"- PVDR definition: `{analysis['pvdr_definition']}`")
    lines.append(f"- normalization: `{analysis['normalization_definition']}`")
    lines.append("")
    lines.append("## Benchmark Scope (Bounded)")
    lines.append("")
    lines.append(f"- Status: `{benchmark_scope['status']}`")
    lines.append(f"- Longitudinal consistency: `{benchmark_scope['longitudinal_consistency']}`")
    lines.append(f"- Transverse/dosimetric match: `{benchmark_scope['transverse_dosimetric_match']}`")
    lines.append(f"- Statement: {benchmark_scope['statement']}")
    lines.append("")
    lines.append("## Claim Policy")
    lines.append("")
    lines.append(f"- Allowed claim type: `{claim_policy['allowed_claim_type']}`")
    lines.append(f"- Allowed examples: `{claim_policy['allowed_examples']}`")
    lines.append(f"- Disallowed claim type: `{claim_policy['disallowed_claim_type']}`")
    lines.append(f"- Required caveat: {claim_policy['required_caveat']}")
    lines.append("")
    lines.append("## Outputs")
    lines.append("")
    lines.append(f"- Combined best-per-energy table: `{combined_best_file}`")
    lines.append(f"- Combined case-metrics table: `{combined_case_metrics_file}`")
    lines.append(f"- Analyzed run roots: `{[str(p) for p in analyzed_runs]}`")
    lines.append("")
    lines.append("## Commands Executed")
    lines.append("")
    for cmd in commands:
        lines.append(f"- `{' '.join(cmd)}`")
    lines.append("")
    (run_root / "stage2_frozen_summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    cfg = load_json(args.config)

    beamline = cfg["beamline_frozen"]
    energies = [int(v) for v in beamline["energies_mev"]]
    g4_map = {int(k): float(v) for k, v in beamline["g4_t_per_m_by_energy"].items()}
    missing_energy = [e for e in energies if e not in g4_map]
    if missing_energy:
        raise ValueError(f"Missing g4 mapping for energies: {missing_energy}")

    args.run_root.mkdir(parents=True, exist_ok=True)
    lock_payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "config_path": str(args.config.resolve()),
        "config": cfg,
        "execution_flags": {
            "run_topas": bool(args.run_topas),
            "analyze_existing": bool(args.analyze_existing),
            "skip_existing": bool(args.skip_existing),
            "dry_run": bool(args.dry_run),
        },
    }
    (args.run_root / "stage2_frozen_lock.json").write_text(
        json.dumps(lock_payload, indent=2),
        encoding="utf-8",
    )

    planned_commands: List[List[str]] = []
    analyzed_runs: List[Path] = []

    for energy in energies:
        g4 = g4_map[energy]
        case_run_root = args.run_root / f"E{energy}"
        case_run_root.mkdir(parents=True, exist_ok=True)

        build_cmd = build_case_command(
            cfg=cfg,
            args=args,
            energy_mev=energy,
            g4_t_per_m=g4,
            case_run_root=case_run_root,
        )
        planned_commands.append(build_cmd)
        run_command(build_cmd, repo_root, args.dry_run)

        do_analysis = args.run_topas or args.analyze_existing
        if do_analysis:
            manifest_path = case_run_root / "manifest.json"
            if args.dry_run or manifest_path.exists():
                analyze_cmd = build_analysis_command(args=args, cfg=cfg, case_run_root=case_run_root)
                planned_commands.append(analyze_cmd)
                run_command(analyze_cmd, repo_root, args.dry_run)
                analyzed_runs.append(case_run_root)

    (args.run_root / "stage2_frozen_commands.txt").write_text(
        "\n".join(" ".join(cmd) for cmd in planned_commands) + "\n",
        encoding="utf-8",
    )

    combined_best: List[pd.DataFrame] = []
    combined_case_metrics: List[pd.DataFrame] = []
    for energy in energies:
        case_run_root = args.run_root / f"E{energy}"
        best_file = case_run_root / "analysis_frozen" / "best_per_energy.csv"
        metrics_file = case_run_root / "analysis_frozen" / "case_metrics.csv"
        if best_file.exists():
            best_df = pd.read_csv(best_file)
            if not best_df.empty:
                best_df["frozen_energy_mev"] = energy
                combined_best.append(best_df)
        if metrics_file.exists():
            metrics_df = pd.read_csv(metrics_file)
            if not metrics_df.empty:
                metrics_df["frozen_energy_mev"] = energy
                combined_case_metrics.append(metrics_df)

    combined_best_file = args.run_root / "combined_best_per_energy.csv"
    combined_case_metrics_file = args.run_root / "combined_case_metrics.csv"
    if combined_best:
        pd.concat(combined_best, ignore_index=True).to_csv(combined_best_file, index=False)
    else:
        pd.DataFrame().to_csv(combined_best_file, index=False)
    if combined_case_metrics:
        pd.concat(combined_case_metrics, ignore_index=True).to_csv(combined_case_metrics_file, index=False)
    else:
        pd.DataFrame().to_csv(combined_case_metrics_file, index=False)

    write_summary(
        run_root=args.run_root,
        cfg=cfg,
        commands=planned_commands,
        analyzed_runs=analyzed_runs,
        combined_best_file=combined_best_file,
        combined_case_metrics_file=combined_case_metrics_file,
    )

    print(f"Frozen lock: {args.run_root / 'stage2_frozen_lock.json'}")
    print(f"Planned/executed commands: {args.run_root / 'stage2_frozen_commands.txt'}")
    print(f"Combined best: {combined_best_file}")
    print(f"Combined metrics: {combined_case_metrics_file}")
    print(f"Summary report: {args.run_root / 'stage2_frozen_summary.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
