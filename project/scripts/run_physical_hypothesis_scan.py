#!/usr/bin/env python3
"""Run a small matrix of physical-setup hypotheses and rank them against the Whitmore."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List


@dataclass
class Variant:
    name: str
    physics_profile: str
    gradient_x_scale: float
    gradient_y_scale: float
    description: str


DEFAULT_VARIANTS = [
    Variant(
        name="historical_em_opt4_equal",
        physics_profile="em_opt4_only",
        gradient_x_scale=1.0,
        gradient_y_scale=1.0,
        description="Matches the historical generated decks as closely as possible.",
    ),
    Variant(
        name="topas_default_equal",
        physics_profile="topas_default",
        gradient_x_scale=1.0,
        gradient_y_scale=1.0,
        description="TOPAS default modular physics with the same equal-gradient convention.",
    ),
    Variant(
        name="topas_default_gx_eq_gy_neg",
        physics_profile="topas_default",
        gradient_x_scale=1.0,
        gradient_y_scale=-1.0,
        description="TOPAS default modular physics with Y-gradient sign flipped.",
    ),
    Variant(
        name="topas_default_gx_neg_gy_eq",
        physics_profile="topas_default",
        gradient_x_scale=-1.0,
        gradient_y_scale=1.0,
        description="TOPAS default modular physics with X-gradient sign flipped.",
    ),
]


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    default_python = "/opt/anaconda3/bin/python" if Path("/opt/anaconda3/bin/python").exists() else sys.executable
    parser = argparse.ArgumentParser(
        description="Automate a targeted physical-setup scan and summarize which variant best follows the Whitmore."
    )
    parser.add_argument(
        "--run-root",
        type=Path,
        default=root / "runs" / "hypothesis_scan",
        help="Root directory for variant runs.",
    )
    parser.add_argument(
        "--energies",
        nargs="+",
        type=int,
        default=[100, 200, 250],
        help="Beam energies (MeV) to test.",
    )
    parser.add_argument(
        "--histories",
        type=int,
        default=50000,
        help="Histories per TOPAS run for each nominal variant case.",
    )
    parser.add_argument("--threads", type=int, default=8, help="TOPAS thread count.")
    parser.add_argument("--topas-bin", type=str, default="/Users/kw/shellScripts/topas", help="TOPAS executable/wrapper.")
    parser.add_argument("--g4-data-dir", type=str, default="/Applications/GEANT4", help="Geant4 data root.")
    parser.add_argument(
        "--python-bin",
        type=str,
        default=default_python,
        help="Python interpreter to use for the build/analyze scripts.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip rerunning TOPAS for a variant if the expected CSVs are already present.",
    )
    return parser.parse_args()


def run_command(cmd: List[str], cwd: Path) -> None:
    proc = subprocess.run(cmd, cwd=str(cwd), text=True)
    if proc.returncode != 0:
        raise SystemExit(proc.returncode)


def read_best_rows(best_csv: Path) -> List[Dict[str, str]]:
    with best_csv.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def summarize_variant(rows: List[Dict[str, str]]) -> Dict[str, float]:
    weighted_errors = [float(row["weighted_error"]) for row in rows]
    z_deltas = [abs(float(row["delta_z_hat_cm"])) for row in rows]
    passes = sum(1 for row in rows if row["within_tolerance"].lower() == "true")
    return {
        "avg_weighted_error": sum(weighted_errors) / len(weighted_errors),
        "avg_abs_delta_z_cm": sum(z_deltas) / len(z_deltas),
        "pass_count": passes,
    }


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    args.run_root.mkdir(parents=True, exist_ok=True)

    summary_rows: List[Dict[str, object]] = []
    for variant in DEFAULT_VARIANTS:
        variant_root = args.run_root / variant.name
        analysis_root = variant_root / "analysis"
        build_cmd = [
            args.python_bin,
            "scripts/build_asymmetric_sweep.py",
            "--run-root",
            str(variant_root),
            "--energies",
            *[str(energy) for energy in args.energies],
            "--nominal-only",
            "--histories",
            str(args.histories),
            "--threads",
            str(args.threads),
            "--run-topas",
            "--topas-bin",
            args.topas_bin,
            "--g4-data-dir",
            args.g4_data_dir,
            "--physics-profile",
            variant.physics_profile,
            "--gradient-x-scale",
            str(variant.gradient_x_scale),
            "--gradient-y-scale",
            str(variant.gradient_y_scale),
        ]
        if args.skip_existing:
            build_cmd.append("--skip-existing")

        analyze_cmd = [
            args.python_bin,
            "scripts/analyze_topas_outputs.py",
            "--manifest",
            str(variant_root / "manifest.json"),
            "--outdir",
            str(analysis_root),
            "--io-retries",
            "10",
            "--io-retry-delay-sec",
            "1.0",
        ]

        print(f"[variant] {variant.name}")
        run_command(build_cmd, repo_root)
        run_command(analyze_cmd, repo_root)

        rows = read_best_rows(analysis_root / "best_per_energy.csv")
        stats = summarize_variant(rows)
        summary_rows.append(
            {
                "variant": variant.name,
                "description": variant.description,
                "physics_profile": variant.physics_profile,
                "gradient_x_scale": variant.gradient_x_scale,
                "gradient_y_scale": variant.gradient_y_scale,
                **stats,
            }
        )

    summary_rows.sort(key=lambda row: (row["pass_count"] * -1, row["avg_weighted_error"]))

    summary_csv = args.run_root / "hypothesis_summary.csv"
    summary_md = args.run_root / "hypothesis_summary.md"
    summary_json = args.run_root / "hypothesis_summary.json"

    with summary_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    summary_json.write_text(json.dumps(summary_rows, indent=2), encoding="utf-8")

    lines = [
        "# Physical Hypothesis Scan",
        "",
        "Variants are ranked by pass count first, then average weighted error across the tested energies.",
        "",
    ]
    for row in summary_rows:
        lines.append(
            f"- `{row['variant']}`: avg weighted error `{row['avg_weighted_error']:.3f}`, "
            f"avg |delta_z| `{row['avg_abs_delta_z_cm']:.3f} cm`, passes `{row['pass_count']}`, "
            f"profile `{row['physics_profile']}`, scales `({row['gradient_x_scale']}, {row['gradient_y_scale']})`"
        )
        lines.append(f"  {row['description']}")
    summary_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Wrote {summary_csv}")
    print(f"Wrote {summary_md}")
    print(f"Wrote {summary_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
