#!/usr/bin/env python3
"""Run a quick strict-Whitmore 250 MeV trend check with one seed."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    default_python = (
        "/opt/anaconda3/bin/python"
        if Path("/opt/anaconda3/bin/python").exists()
        else sys.executable
    )
    parser = argparse.ArgumentParser(
        description=(
            "Quick strict 250 MeV check: run TOPAS sweep, analyze with paper-style "
            "definitions, then compare focal-depth trend against Whitmore Supplementary Table 3."
        )
    )
    parser.add_argument(
        "--run-root",
        type=Path,
        default=root / "runs" / "strict_whitmore_quick_250",
        help="Output root for this quick strict run.",
    )
    parser.add_argument("--energy", type=int, default=250, help="Beam energy in MeV (default 250).")
    parser.add_argument(
        "--g4-range",
        type=str,
        default="",
        help=(
            "Optional override Q4 scan as min:max:step. "
            "If omitted, uses Whitmore Supplementary Table 3 points."
        ),
    )
    parser.add_argument(
        "--histories",
        type=int,
        default=100000,
        help="Reduced histories per case for quick trend checks.",
    )
    parser.add_argument("--threads", type=int, default=8, help="TOPAS thread count.")
    parser.add_argument("--seed", type=int, default=11, help="Single random seed.")
    parser.add_argument(
        "--physics-profile",
        type=str,
        default="topas_default",
        choices=["topas_default", "em_opt4_only", "em_opt0_only"],
        help="Physics profile passed through to build_asymmetric_sweep.py.",
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
        help="Python interpreter used for subprocess calls.",
    )
    parser.add_argument(
        "--reference",
        type=Path,
        default=root / "config" / "benchmark_reference.json",
        help="Whitmore reference JSON.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip TOPAS execution where non-empty dose CSV already exists.",
    )
    parser.add_argument(
        "--quad-gradient-convention",
        choices=["scaled", "ideal_opposite"],
        default="ideal_opposite",
        help=(
            "How TOPAS MagneticFieldGradientX/Y are generated from paper gradients. "
            "Use 'ideal_opposite' for paper-consistent asymmetric quadrupole optics."
        ),
    )
    parser.add_argument("--xbins", type=int, default=101, help="Dose grid bins in X.")
    parser.add_argument("--ybins", type=int, default=101, help="Dose grid bins in Y.")
    parser.add_argument("--zbins", type=int, default=101, help="Dose grid bins in Z.")
    parser.add_argument(
        "--z-mode",
        choices=["on_axis", "integrated_xy", "global_max"],
        default="on_axis",
        help="Depth metric used in analysis and trend comparison.",
    )
    parser.add_argument(
        "--sigma-mode",
        choices=["on_axis", "integrated_xy", "global_max"],
        default="on_axis",
        help="Sigma metric mode passed to analysis script.",
    )
    parser.add_argument(
        "--entrance-mode",
        choices=["on_axis", "integrated_xy", "plane_max"],
        default="on_axis",
        help="Entrance-dose metric mode passed to analysis script.",
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Only generate decks/manifest; do not run TOPAS or analysis.",
    )
    parser.add_argument("--dpi", type=int, default=220, help="DPI for trend plot.")
    return parser.parse_args()


def run_command(cmd: List[str], cwd: Path) -> None:
    print("[cmd]", " ".join(cmd))
    proc = subprocess.run(cmd, cwd=str(cwd), text=True)
    if proc.returncode != 0:
        raise SystemExit(proc.returncode)


def load_reference(path: Path) -> Dict:
    return json.loads(path.read_text(encoding="utf-8"))


def table3_for_energy(reference: Dict, energy: int) -> pd.DataFrame:
    rows = reference["asymmetric_beamline"]["energies"][str(int(energy))][
        "supplementary_table3_scan"
    ]
    frame = pd.DataFrame(rows).astype(float).sort_values("g4_t_per_m")
    frame = frame.rename(columns={"z_hat_cm": "z_hat_whitmore_cm"})
    return frame[["g4_t_per_m", "z_hat_whitmore_cm"]].reset_index(drop=True)


def write_trend_report(
    out_md: Path,
    out_json: Path,
    merged: pd.DataFrame,
    model_z_col: str,
    model_z_label: str,
    energy: int,
    histories: int,
    threads: int,
    seed: int,
) -> Dict[str, object]:
    if merged.empty:
        summary = {
            "energy_mev": int(energy),
            "n_points": 0,
            "trend_ok": False,
            "reason": "No overlapping model/reference points on Q4 grid.",
        }
        out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        out_md.write_text(
            "# Quick Strict Whitmore Trend Check\n\nNo overlapping trend points were found.\n",
            encoding="utf-8",
        )
        return summary

    x = merged["g4_t_per_m"].to_numpy(dtype=float)
    y_model = merged[model_z_col].to_numpy(dtype=float)
    y_ref = merged["z_hat_whitmore_cm"].to_numpy(dtype=float)

    if len(merged) >= 2:
        slope_model = float(np.polyfit(x, y_model, 1)[0])
        slope_ref = float(np.polyfit(x, y_ref, 1)[0])
        pearson = float(np.corrcoef(y_model, y_ref)[0, 1])
        spearman = float(
            pd.Series(y_model).corr(pd.Series(y_ref), method="spearman")
        )
    else:
        slope_model = float("nan")
        slope_ref = float("nan")
        pearson = float("nan")
        spearman = float("nan")

    mae_cm = float(np.mean(np.abs(y_model - y_ref)))
    rmse_cm = float(np.sqrt(np.mean((y_model - y_ref) ** 2)))

    same_direction = (
        np.isfinite(slope_model)
        and np.isfinite(slope_ref)
        and np.sign(slope_model) == np.sign(slope_ref)
        and abs(slope_ref) > 1e-9
    )
    trend_ok = bool(
        same_direction
        and np.isfinite(spearman)
        and spearman >= 0.80
    )

    summary = {
        "energy_mev": int(energy),
        "histories": int(histories),
        "threads": int(threads),
        "seed": int(seed),
        "model_metric": model_z_col,
        "model_metric_label": model_z_label,
        "n_points": int(len(merged)),
        "slope_model_cm_per_tpm": slope_model,
        "slope_whitmore_cm_per_tpm": slope_ref,
        "pearson_r": pearson,
        "spearman_rho": spearman,
        "mae_cm": mae_cm,
        "rmse_cm": rmse_cm,
        "same_direction": bool(same_direction),
        "trend_ok": trend_ok,
        "criterion": "trend_ok = same_direction AND spearman_rho >= 0.80",
    }
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    lines = [
        "# Quick Strict Whitmore Trend Check",
        "",
        f"- Energy: `{energy} MeV`",
        f"- Histories/case: `{histories:,}`",
        f"- Threads: `{threads}`",
        f"- Seed: `{seed}`",
        f"- Model z metric: `{model_z_label}`",
        f"- Points compared: `{len(merged)}`",
        "",
        "## Trend Metrics",
        "",
        f"- Model slope dz/dg4: `{slope_model:.4f} cm/(T/m)`",
        f"- Whitmore slope dz/dg4: `{slope_ref:.4f} cm/(T/m)`",
        f"- Pearson r: `{pearson:.4f}`",
        f"- Spearman rho: `{spearman:.4f}`",
        f"- MAE: `{mae_cm:.3f} cm`",
        f"- RMSE: `{rmse_cm:.3f} cm`",
        "",
        "## Verdict",
        "",
        (
            "- `OK` trend-consistent with Whitmore."
            if trend_ok
            else "- `NOT_OK` trend is not yet consistent under this quick strict check."
        ),
        "- Criterion used: `same direction` and `spearman >= 0.80`.",
        "",
    ]
    out_md.write_text("\n".join(lines), encoding="utf-8")
    return summary


def save_plot(
    merged: pd.DataFrame,
    model_z_col: str,
    model_z_label: str,
    out_png: Path,
    summary: Dict[str, object],
    dpi: int,
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return

    fig, ax = plt.subplots(figsize=(8.8, 5.4), constrained_layout=False)
    ax.plot(
        merged["g4_t_per_m"],
        merged["z_hat_whitmore_cm"],
        "--o",
        color="#d62728",
        linewidth=1.8,
        markersize=5,
        label="Whitmore Table 3 (250 MeV)",
    )
    ax.plot(
        merged["g4_t_per_m"],
        merged[model_z_col],
        "-o",
        color="#1f77b4",
        linewidth=2.1,
        markersize=5,
        label=f"TOPAS strict quick ({model_z_label})",
    )
    ax.set_xlabel("Q4 gradient (T/m)")
    ax.set_ylabel("Depth of maximum dose z_hat (cm)")
    ax.set_title("Quick Strict 250 MeV Trend Check")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8.8)
    verdict = "OK" if bool(summary.get("trend_ok", False)) else "NOT_OK"
    txt = (
        f"Verdict: {verdict} | "
        f"Spearman={float(summary.get('spearman_rho', float('nan'))):.3f} | "
        f"MAE={float(summary.get('mae_cm', float('nan'))):.2f} cm"
    )
    fig.text(0.01, 0.01, txt, ha="left", va="bottom", fontsize=9)
    fig.subplots_adjust(bottom=0.18)
    fig.savefig(out_png, dpi=dpi)
    plt.close(fig)


def z_mode_to_column(z_mode: str) -> tuple[str, str]:
    if z_mode == "integrated_xy":
        return "z_hat_integrated_cm", "integrated z_hat"
    if z_mode == "global_max":
        return "z_hat_global_cm", "global-maximum z_hat"
    return "z_hat_on_axis_cm", "on-axis z_hat"


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    args.run_root.mkdir(parents=True, exist_ok=True)

    build_cmd: List[str] = [
        args.python_bin,
        "scripts/build_asymmetric_sweep.py",
        "--run-root",
        str(args.run_root),
        "--energies",
        str(args.energy),
        "--histories",
        str(args.histories),
        "--threads",
        str(args.threads),
        "--seed",
        str(args.seed),
        "--xbins",
        str(args.xbins),
        "--ybins",
        str(args.ybins),
        "--zbins",
        str(args.zbins),
        "--source-sigma-x-mm",
        "4.0",
        "--source-sigma-y-mm",
        "4.0",
        "--source-angular-x-mrad",
        "3.2",
        "--source-angular-y-mrad",
        "3.2",
        "--source-energy-spread-mev",
        "0.75",
        "--quad-gradient-convention",
        args.quad_gradient_convention,
        "--topas-bin",
        args.topas_bin,
        "--g4-data-dir",
        args.g4_data_dir,
        "--physics-profile",
        args.physics_profile,
    ]
    if args.g4_range:
        build_cmd.extend(["--g4-range", args.g4_range])
    if args.skip_existing:
        build_cmd.append("--skip-existing")
    if not args.prepare_only:
        build_cmd.append("--run-topas")

    run_command(build_cmd, repo_root)

    if args.prepare_only:
        print("Prepared cases only (--prepare-only).")
        print(f"Manifest: {args.run_root / 'manifest.json'}")
        return 0

    analysis_dir = args.run_root / "analysis_paper_mode"
    analyze_cmd = [
        args.python_bin,
        "scripts/analyze_topas_outputs.py",
        "--manifest",
        str(args.run_root / "manifest.json"),
        "--outdir",
        str(analysis_dir),
        "--z-mode",
        args.z_mode,
        "--sigma-mode",
        args.sigma_mode,
        "--sigma-fit-mode",
        "gaussian_2d",
        "--entrance-mode",
        args.entrance_mode,
        "--io-retries",
        "10",
        "--io-retry-delay-sec",
        "1.0",
    ]
    run_command(analyze_cmd, repo_root)

    case_metrics = pd.read_csv(analysis_dir / "case_metrics.csv")
    case_metrics = case_metrics[case_metrics["energy_mev"].astype(int) == int(args.energy)].copy()
    if case_metrics.empty:
        raise RuntimeError("No case_metrics rows were found for requested energy.")
    model_z_col, model_z_label = z_mode_to_column(args.z_mode)
    if model_z_col not in case_metrics.columns:
        raise RuntimeError(
            f"Requested z-mode '{args.z_mode}' expects column '{model_z_col}', "
            "but it is missing from case_metrics.csv."
        )
    model = case_metrics[["g4_t_per_m", model_z_col]].copy()
    model["g4_key"] = model["g4_t_per_m"].astype(float).round(4)

    reference = load_reference(args.reference)
    ref = table3_for_energy(reference, int(args.energy))
    ref["g4_key"] = ref["g4_t_per_m"].astype(float).round(4)

    merged = pd.merge(
        model,
        ref[["g4_key", "z_hat_whitmore_cm"]],
        on="g4_key",
        how="inner",
    )
    merged = merged.drop_duplicates(subset=["g4_t_per_m"]).sort_values("g4_t_per_m")
    merged = merged[["g4_t_per_m", model_z_col, "z_hat_whitmore_cm"]]

    trend_csv = args.run_root / "trend_check_250.csv"
    trend_json = args.run_root / "trend_check_250_summary.json"
    trend_md = args.run_root / "trend_check_250_summary.md"
    trend_png = args.run_root / "trend_check_250_zhat_vs_g4.png"

    merged.to_csv(trend_csv, index=False)
    summary = write_trend_report(
        out_md=trend_md,
        out_json=trend_json,
        merged=merged,
        model_z_col=model_z_col,
        model_z_label=model_z_label,
        energy=int(args.energy),
        histories=int(args.histories),
        threads=int(args.threads),
        seed=int(args.seed),
    )
    save_plot(
        merged=merged,
        model_z_col=model_z_col,
        model_z_label=model_z_label,
        out_png=trend_png,
        summary=summary,
        dpi=args.dpi,
    )

    verdict = "OK" if bool(summary.get("trend_ok", False)) else "NOT_OK"
    print(f"Trend table: {trend_csv}")
    print(f"Trend summary JSON: {trend_json}")
    print(f"Trend summary MD: {trend_md}")
    if trend_png.exists():
        print(f"Trend plot: {trend_png}")
    print(
        f"Quick strict trend verdict ({int(args.energy)} MeV): {verdict} "
        f"(spearman={float(summary.get('spearman_rho', float('nan'))):.3f}, "
        f"mae={float(summary.get('mae_cm', float('nan'))):.3f} cm)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
