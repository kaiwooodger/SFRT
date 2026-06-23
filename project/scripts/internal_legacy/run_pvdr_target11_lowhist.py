#!/usr/bin/env python3
"""Low-history PVDR targeting workflow around 10 mm pitch."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List


@dataclass(frozen=True)
class Variant:
    name: str
    sigma_x_mm: float
    sigma_y_mm: float
    angular_x_mrad: float
    angular_y_mrad: float
    energy_spread_mev: float
    gradient_x_scale: float
    gradient_y_scale: float
    quad_gradient_convention: str
    note: str


DEFAULT_VARIANTS: List[Variant] = [
    Variant(
        name="mildfocus_scaled",
        sigma_x_mm=2.0,
        sigma_y_mm=2.0,
        angular_x_mrad=1.2,
        angular_y_mrad=1.2,
        energy_spread_mev=0.35,
        gradient_x_scale=1.20,
        gradient_y_scale=1.20,
        quad_gradient_convention="scaled",
        note="Moderately tightened source with stronger optics, scaled XY convention.",
    ),
    Variant(
        name="mildfocus_idealopp",
        sigma_x_mm=2.0,
        sigma_y_mm=2.0,
        angular_x_mrad=1.2,
        angular_y_mrad=1.2,
        energy_spread_mev=0.35,
        gradient_x_scale=1.20,
        gradient_y_scale=1.00,
        quad_gradient_convention="ideal_opposite",
        note="Moderately tightened source with ideal opposite-plane quadrupole convention.",
    ),
    Variant(
        name="tight_idealopp",
        sigma_x_mm=1.0,
        sigma_y_mm=1.0,
        angular_x_mrad=0.8,
        angular_y_mrad=0.8,
        energy_spread_mev=0.20,
        gradient_x_scale=1.30,
        gradient_y_scale=1.00,
        quad_gradient_convention="ideal_opposite",
        note="Tight source stress test while avoiding TOPAS angular rejection instability.",
    ),
]


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    default_python = "/opt/anaconda3/bin/python" if Path("/opt/anaconda3/bin/python").exists() else sys.executable
    parser = argparse.ArgumentParser(
        description=(
            "Run a low-history Q4/source scan and optimize synthetic SFRT PVDR "
            "around 10 mm pitch."
        )
    )
    parser.add_argument(
        "--reference",
        type=Path,
        default=root / "config" / "benchmark_reference.json",
        help="Reference JSON used to get baseline Q4 values by energy.",
    )
    parser.add_argument(
        "--run-root",
        type=Path,
        default=root / "runs" / "pvdr_target11_lowhist",
        help="Output root for variant runs and optimization artifacts.",
    )
    parser.add_argument(
        "--energies",
        nargs="+",
        type=int,
        default=[250],
        help="Energies to include in this quick scan.",
    )
    parser.add_argument(
        "--q4-offsets",
        nargs="+",
        type=float,
        default=[-2.0, -1.0, 0.0, 1.0, 2.0],
        help="Offsets (T/m) applied around each energy baseline Q4.",
    )
    parser.add_argument("--histories", type=int, default=20000, help="Histories per case.")
    parser.add_argument("--threads", type=int, default=8, help="TOPAS thread count.")
    parser.add_argument("--seed", type=int, default=11, help="TOPAS RNG seed.")
    parser.add_argument(
        "--physics-profile",
        type=str,
        default="topas_default",
        choices=["topas_default", "em_opt4_only", "em_opt0_only"],
        help="Physics profile for all runs.",
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
        "--pitches-mm",
        nargs="+",
        type=float,
        default=[8, 9, 10, 11, 12],
        help="PVDR pitch scan values in mm.",
    )
    parser.add_argument(
        "--n-beams",
        type=int,
        default=11,
        help="Odd beam-count for lattice superposition in PVDR optimizer.",
    )
    parser.add_argument(
        "--target-pvdr",
        type=float,
        default=11.0,
        help="Target PVDR to compare against.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip TOPAS cases that already have non-empty dose.csv.",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Disable optimizer heatmaps for faster runs.",
    )
    return parser.parse_args()


def run_command(cmd: List[str], cwd: Path) -> int:
    print("[cmd]", " ".join(cmd))
    proc = subprocess.run(cmd, cwd=str(cwd), text=True)
    return int(proc.returncode)


def load_baseline_q4(reference: Path, energies: List[int]) -> Dict[int, float]:
    data = json.loads(reference.read_text(encoding="utf-8"))
    out: Dict[int, float] = {}
    for energy in energies:
        key = str(int(energy))
        try:
            out[int(energy)] = float(
                data["asymmetric_beamline"]["energies"][key]["baseline_gradients_t_per_m"][3]
            )
        except Exception as exc:
            raise KeyError(f"Missing baseline Q4 for E={energy} in {reference}") from exc
    return out


def q4_range_string(center: float, offsets: List[float]) -> str:
    values = sorted({round(center + float(v), 6) for v in offsets})
    if len(values) < 2:
        v = values[0]
        return f"{v}:{v}:0.1"
    step = round(values[1] - values[0], 6)
    if step <= 0:
        raise ValueError("q4 offsets produced a non-positive step.")
    for i in range(2, len(values)):
        if abs((values[i] - values[i - 1]) - step) > 1e-8:
            raise ValueError(
                "q4 offsets must form a uniform step to map into build_asymmetric_sweep --g4-range."
            )
    return f"{values[0]}:{values[-1]}:{step}"


def load_best_pvdr(meta_json: Path) -> float:
    payload = json.loads(meta_json.read_text(encoding="utf-8"))
    return float(payload.get("best_observed_pvdr", float("nan")))


def load_best_row(best_csv: Path) -> Dict[str, str]:
    with best_csv.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise RuntimeError(f"No rows in {best_csv}")
    rows.sort(key=lambda r: float(r.get("rank_metric", "nan")), reverse=True)
    return rows[0]


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    args.run_root.mkdir(parents=True, exist_ok=True)

    baseline_q4 = load_baseline_q4(args.reference, args.energies)
    summary_rows: List[Dict[str, object]] = []

    for variant in DEFAULT_VARIANTS:
        variant_root = args.run_root / variant.name
        variant_root.mkdir(parents=True, exist_ok=True)
        manifests: List[Path] = []

        for energy in args.energies:
            energy_root = variant_root / f"E{int(energy)}"
            energy_root.mkdir(parents=True, exist_ok=True)
            g4_range = q4_range_string(baseline_q4[int(energy)], args.q4_offsets)

            build_cmd = [
                args.python_bin,
                "scripts/build_asymmetric_sweep.py",
                "--run-root",
                str(energy_root),
                "--energies",
                str(int(energy)),
                "--g4-range",
                g4_range,
                "--histories",
                str(int(args.histories)),
                "--threads",
                str(int(args.threads)),
                "--seed",
                str(int(args.seed)),
                "--source-sigma-x-mm",
                str(variant.sigma_x_mm),
                "--source-sigma-y-mm",
                str(variant.sigma_y_mm),
                "--source-angular-x-mrad",
                str(variant.angular_x_mrad),
                "--source-angular-y-mrad",
                str(variant.angular_y_mrad),
                "--source-energy-spread-mev",
                str(variant.energy_spread_mev),
                "--gradient-x-scale",
                str(variant.gradient_x_scale),
                "--gradient-y-scale",
                str(variant.gradient_y_scale),
                "--quad-gradient-convention",
                variant.quad_gradient_convention,
                "--physics-profile",
                args.physics_profile,
                "--topas-bin",
                args.topas_bin,
                "--g4-data-dir",
                args.g4_data_dir,
                "--run-topas",
            ]
            if args.skip_existing:
                build_cmd.append("--skip-existing")
            rc = run_command(build_cmd, repo_root)
            if rc != 0:
                print(
                    f"WARN: build/run failed for variant={variant.name}, E={energy} (code={rc}). "
                    "Skipping this energy block."
                )
                continue
            manifests.append(energy_root / "manifest.json")

        if not manifests:
            print(f"WARN: no usable manifests for variant {variant.name}; skipping optimization.")
            continue

        optimize_root = variant_root / "optimization"
        optimize_cmd = [
            args.python_bin,
            "scripts/optimize_pvdr_from_manifests.py",
            "--manifests",
            *[str(p) for p in manifests],
            "--pitches-mm",
            *[f"{float(v):g}" for v in args.pitches_mm],
            "--n-beams",
            str(int(args.n_beams)),
            "--z-mode",
            "zf_integrated",
            "--outdir",
            str(optimize_root),
        ]
        if args.no_plots:
            optimize_cmd.append("--no-plots")
        rc = run_command(optimize_cmd, repo_root)
        if rc != 0:
            print(f"WARN: optimizer failed for variant {variant.name} (code={rc}); skipping.")
            continue

        best_csv = optimize_root / "pvdr_pitch_q4_best_per_energy.csv"
        meta_json = optimize_root / "pvdr_pitch_q4_meta.json"
        best_pvdr = load_best_pvdr(meta_json)
        best_row = load_best_row(best_csv)

        summary_rows.append(
            {
                "variant": variant.name,
                "note": variant.note,
                "energies_mev": ",".join(str(int(v)) for v in args.energies),
                "histories": int(args.histories),
                "threads": int(args.threads),
                "seed": int(args.seed),
                "q4_offsets_tpm": ",".join(f"{float(v):g}" for v in args.q4_offsets),
                "pitches_mm": ",".join(f"{float(v):g}" for v in args.pitches_mm),
                "source_sigma_x_mm": float(variant.sigma_x_mm),
                "source_sigma_y_mm": float(variant.sigma_y_mm),
                "source_angular_x_mrad": float(variant.angular_x_mrad),
                "source_angular_y_mrad": float(variant.angular_y_mrad),
                "source_energy_spread_mev": float(variant.energy_spread_mev),
                "gradient_x_scale": float(variant.gradient_x_scale),
                "gradient_y_scale": float(variant.gradient_y_scale),
                "quad_gradient_convention": variant.quad_gradient_convention,
                "best_case": str(best_row.get("case_id", "")),
                "best_energy_mev": int(float(best_row.get("energy_mev", "nan"))),
                "best_q4_t_per_m": float(best_row.get("g4_t_per_m", "nan")),
                "best_pitch_mm": float(best_row.get("pitch_mm", "nan")),
                "best_pvdr_at_zf": float(best_row.get("pvdr_at_zf", "nan")),
                "best_pvdr_entrance": float(best_row.get("pvdr_entrance", "nan")),
                "best_pvdr_exit": float(best_row.get("pvdr_exit", "nan")),
                "best_observed_pvdr": best_pvdr,
                "target_pvdr": float(args.target_pvdr),
                "pvdr_gap_to_target": float(args.target_pvdr - best_pvdr),
                "variant_root": str(variant_root),
            }
        )

    if not summary_rows:
        raise RuntimeError("No variant completed successfully; no PVDR summary available.")

    summary_rows.sort(key=lambda row: float(row["best_observed_pvdr"]), reverse=True)
    summary_csv = args.run_root / "target11_variant_summary.csv"
    summary_json = args.run_root / "target11_variant_summary.json"
    summary_md = args.run_root / "target11_variant_summary.md"

    with summary_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)
    summary_json.write_text(json.dumps(summary_rows, indent=2), encoding="utf-8")

    lines: List[str] = []
    lines.append("# Low-History PVDR Targeting Summary")
    lines.append("")
    lines.append("- Objective: maximize PVDR near pitch 10 mm and compare against target PVDR=11.")
    lines.append("- Ranking metric: PVDR at integrated single-beam focus depth (`zf_integrated`).")
    lines.append("")
    for row in summary_rows:
        lines.append(
            f"- `{row['variant']}`: best observed PVDR `{float(row['best_observed_pvdr']):.3f}` "
            f"(gap `{float(row['pvdr_gap_to_target']):.3f}`), best case `{row['best_case']}` "
            f"at E={int(row['best_energy_mev'])} MeV, Q4={float(row['best_q4_t_per_m']):.3f} T/m, "
            f"pitch={float(row['best_pitch_mm']):.1f} mm."
        )
        lines.append(f"  {row['note']}")
    summary_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Wrote {summary_csv}")
    print(f"Wrote {summary_json}")
    print(f"Wrote {summary_md}")
    if summary_rows:
        best = summary_rows[0]
        print(
            "Best variant overall: "
            f"{best['variant']} | PVDR={float(best['best_observed_pvdr']):.3f} "
            f"| gap to 11 = {float(best['pvdr_gap_to_target']):.3f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
