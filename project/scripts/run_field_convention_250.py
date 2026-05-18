#!/usr/bin/env python3
"""Run a focused 250 MeV quadrupole field-convention scan."""

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
    gradient_x_scale: float
    gradient_y_scale: float
    description: str


DEFAULT_VARIANTS: List[Variant] = [
    Variant(
        name="gx_eq_gy_eq",
        gradient_x_scale=1.0,
        gradient_y_scale=1.0,
        description="Current mapping: Gx = g_i, Gy = g_i",
    ),
    Variant(
        name="gx_eq_gy_neg",
        gradient_x_scale=1.0,
        gradient_y_scale=-1.0,
        description="Y sign flipped: Gx = g_i, Gy = -g_i",
    ),
    Variant(
        name="gx_neg_gy_eq",
        gradient_x_scale=-1.0,
        gradient_y_scale=1.0,
        description="X sign flipped: Gx = -g_i, Gy = g_i",
    ),
    Variant(
        name="gx_neg_gy_neg",
        gradient_x_scale=-1.0,
        gradient_y_scale=-1.0,
        description="Both signs flipped: Gx = -g_i, Gy = -g_i",
    ),
]


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    default_python = "/opt/anaconda3/bin/python" if Path("/opt/anaconda3/bin/python").exists() else sys.executable
    parser = argparse.ArgumentParser(
        description=(
            "Run 250 MeV field convention variants at fixed Q4 and summarize which "
            "mapping best matches Whitmore metrics."
        )
    )
    parser.add_argument(
        "--run-root",
        type=Path,
        default=root / "runs" / "field_convention_250",
        help="Root output directory for field-convention variants.",
    )
    parser.add_argument("--energy", type=int, default=250, help="Beam energy in MeV.")
    parser.add_argument(
        "--g4",
        type=float,
        default=14.9,
        help="Fixed Q4 value in T/m for all variants.",
    )
    parser.add_argument("--histories", type=int, default=50000, help="Histories per variant.")
    parser.add_argument("--threads", type=int, default=8, help="TOPAS thread count.")
    parser.add_argument(
        "--physics-profile",
        type=str,
        default="topas_default",
        choices=["topas_default", "em_opt4_only", "em_opt0_only"],
        help="Physics profile used for all variants.",
    )
    parser.add_argument(
        "--topas-bin",
        type=str,
        default="/Users/kw/shellScripts/topas",
        help="TOPAS executable/wrapper.",
    )
    parser.add_argument(
        "--g4-data-dir",
        type=str,
        default="/Applications/GEANT4",
        help="Geant4 data root path.",
    )
    parser.add_argument(
        "--python-bin",
        type=str,
        default=default_python,
        help="Python interpreter used for subprocess script calls.",
    )
    parser.add_argument(
        "--source-sigma-x-mm",
        type=float,
        default=None,
        help="Optional source sigma X override (mm).",
    )
    parser.add_argument(
        "--source-sigma-y-mm",
        type=float,
        default=None,
        help="Optional source sigma Y override (mm).",
    )
    parser.add_argument(
        "--source-angular-x-mrad",
        type=float,
        default=None,
        help="Optional source angular spread X override (mrad).",
    )
    parser.add_argument(
        "--source-angular-y-mrad",
        type=float,
        default=None,
        help="Optional source angular spread Y override (mrad).",
    )
    parser.add_argument(
        "--source-energy-spread-mev",
        type=float,
        default=None,
        help="Optional source absolute energy spread override (MeV).",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip TOPAS execution if variant dose CSV already exists.",
    )
    return parser.parse_args()


def run_command(cmd: List[str], cwd: Path) -> None:
    proc = subprocess.run(cmd, cwd=str(cwd), text=True)
    if proc.returncode != 0:
        raise SystemExit(proc.returncode)


def load_best_row(best_csv: Path) -> Dict[str, object]:
    with best_csv.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise RuntimeError(f"No rows in {best_csv}")
    return rows[0]


def maybe_add_source_overrides(cmd: List[str], args: argparse.Namespace) -> None:
    if args.source_sigma_x_mm is not None:
        cmd.extend(["--source-sigma-x-mm", str(args.source_sigma_x_mm)])
    if args.source_sigma_y_mm is not None:
        cmd.extend(["--source-sigma-y-mm", str(args.source_sigma_y_mm)])
    if args.source_angular_x_mrad is not None:
        cmd.extend(["--source-angular-x-mrad", str(args.source_angular_x_mrad)])
    if args.source_angular_y_mrad is not None:
        cmd.extend(["--source-angular-y-mrad", str(args.source_angular_y_mrad)])
    if args.source_energy_spread_mev is not None:
        cmd.extend(["--source-energy-spread-mev", str(args.source_energy_spread_mev)])


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    args.run_root.mkdir(parents=True, exist_ok=True)

    summary_rows: List[Dict[str, object]] = []
    g4_step = 0.1
    g4_range = f"{args.g4}:{args.g4}:{g4_step}"

    for variant in DEFAULT_VARIANTS:
        variant_root = args.run_root / variant.name
        analysis_root = variant_root / "analysis"
        build_cmd = [
            args.python_bin,
            "scripts/build_asymmetric_sweep.py",
            "--run-root",
            str(variant_root),
            "--energies",
            str(args.energy),
            "--g4-range",
            g4_range,
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
            args.physics_profile,
            "--gradient-x-scale",
            str(variant.gradient_x_scale),
            "--gradient-y-scale",
            str(variant.gradient_y_scale),
        ]
        maybe_add_source_overrides(build_cmd, args)
        if args.skip_existing:
            build_cmd.append("--skip-existing")

        analyze_cmd = [
            args.python_bin,
            "scripts/analyze_topas_outputs.py",
            "--manifest",
            str(variant_root / "manifest.json"),
            "--outdir",
            str(analysis_root),
            "--z-mode",
            "integrated_xy",
            "--sigma-mode",
            "integrated_xy",
            "--io-retries",
            "10",
            "--io-retry-delay-sec",
            "1.0",
        ]

        print(f"[variant] {variant.name}")
        run_command(build_cmd, repo_root)
        run_command(analyze_cmd, repo_root)

        best_row = load_best_row(analysis_root / "best_per_energy.csv")
        summary_rows.append(
            {
                "variant": variant.name,
                "description": variant.description,
                "gradient_x_scale": variant.gradient_x_scale,
                "gradient_y_scale": variant.gradient_y_scale,
                "case_id": best_row.get("case_id", ""),
                "g4_t_per_m": float(best_row.get("g4_t_per_m", "nan")),
                "z_hat_selected_cm": float(best_row.get("z_hat_selected_cm", "nan")),
                "delta_z_hat_cm": float(best_row.get("delta_z_hat_cm", "nan")),
                "sigma_x_selected_cm": float(best_row.get("sigma_x_selected_cm", "nan")),
                "delta_sigma_x_cm": float(best_row.get("delta_sigma_x_cm", "nan")),
                "sigma_y_selected_cm": float(best_row.get("sigma_y_selected_cm", "nan")),
                "delta_sigma_y_cm": float(best_row.get("delta_sigma_y_cm", "nan")),
                "entrance_on_axis_pct": float(best_row.get("entrance_on_axis_pct", "nan")),
                "delta_entrance_pct": float(best_row.get("delta_entrance_pct", "nan")),
                "weighted_error": float(best_row.get("weighted_error", "nan")),
                "within_tolerance": str(best_row.get("within_tolerance", "False")).lower() == "true",
            }
        )

    summary_rows.sort(key=lambda row: float(row["weighted_error"]))
    summary_csv = args.run_root / "field_convention_summary.csv"
    summary_json = args.run_root / "field_convention_summary.json"
    summary_md = args.run_root / "field_convention_summary.md"

    with summary_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    summary_json.write_text(json.dumps(summary_rows, indent=2), encoding="utf-8")

    lines = [
        "# 250 MeV Field Convention Summary",
        "",
        "Variants are ranked by lowest weighted Whitmore error.",
        "",
    ]
    for row in summary_rows:
        lines.append(
            f"- `{row['variant']}`: error `{row['weighted_error']:.3f}`, "
            f"dz `{row['delta_z_hat_cm']:.3f} cm`, dsx `{row['delta_sigma_x_cm']:.3f} cm`, "
            f"dsy `{row['delta_sigma_y_cm']:.3f} cm`, dent `{row['delta_entrance_pct']:.3f}%`, "
            f"{'PASS' if row['within_tolerance'] else 'FAIL'}"
        )
        lines.append(f"  {row['description']}")
    summary_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Wrote {summary_csv}")
    print(f"Wrote {summary_json}")
    print(f"Wrote {summary_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
