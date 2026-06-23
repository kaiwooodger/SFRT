#!/usr/bin/env python3
"""Run a high-statistics, multi-seed publishability subset workflow."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    default_python = "/opt/anaconda3/bin/python" if Path("/opt/anaconda3/bin/python").exists() else sys.executable
    parser = argparse.ArgumentParser(
        description=(
            "Best practical strategy for publishability: select key points from existing sweeps, "
            "rerun at higher histories across multiple seeds, aggregate uncertainty, and plot error bars."
        )
    )
    parser.add_argument(
        "--case-metrics",
        type=Path,
        default=root / "runs" / "analysis_paper2d" / "case_metrics.csv",
        help="Input case_metrics.csv from previous sweep analysis.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=root / "runs" / "manifest.json",
        help="Manifest from the reference sweep (used to recover source/gradient settings).",
    )
    parser.add_argument(
        "--reference",
        type=Path,
        default=root / "config" / "benchmark_reference.json",
        help="Whitmore reference JSON.",
    )
    parser.add_argument(
        "--run-root",
        type=Path,
        default=root / "runs" / "publishable_subset",
        help="Output root for selected runs, tables, and plots.",
    )
    parser.add_argument(
        "--energies",
        nargs="+",
        type=int,
        default=[100, 200, 250],
        help="Energies to include.",
    )
    parser.add_argument(
        "--selection",
        choices=["best_only", "best_and_depth"],
        default="best_and_depth",
        help="Selection rule per energy from existing case_metrics.",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[11, 22, 33],
        help="Seeds used for uncertainty estimation (3-5 recommended).",
    )
    parser.add_argument(
        "--histories",
        type=int,
        default=1000000,
        help="Histories per rerun (high-stat publishability set).",
    )
    parser.add_argument("--threads", type=int, default=8, help="TOPAS thread count.")
    parser.add_argument(
        "--physics-profile",
        type=str,
        default="topas_default",
        choices=["topas_default", "em_opt4_only", "em_opt0_only"],
        help="Physics profile for all reruns.",
    )
    parser.add_argument(
        "--quad-gradient-convention",
        choices=["scaled", "ideal_opposite"],
        default="ideal_opposite",
        help="How to map paper gradients to TOPAS Gx/Gy when regenerating selected cases.",
    )
    parser.add_argument(
        "--case-run-retries",
        type=int,
        default=2,
        help="Retries per TOPAS case passed through to build_asymmetric_sweep.py.",
    )
    parser.add_argument(
        "--z-mode",
        choices=["integrated_xy", "on_axis", "global_max"],
        default="integrated_xy",
        help="z-hat mode passed to analysis.",
    )
    parser.add_argument(
        "--sigma-mode",
        choices=["integrated_xy", "on_axis", "global_max"],
        default="integrated_xy",
        help="sigma mode passed to analysis.",
    )
    parser.add_argument(
        "--sigma-fit-mode",
        choices=["profile_1d", "gaussian_2d"],
        default="gaussian_2d",
        help="Sigma fit mode passed to analysis (paper-style: gaussian_2d).",
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
        help="Python interpreter for subprocess script calls.",
    )
    parser.add_argument(
        "--run-topas",
        action="store_true",
        help="Actually run TOPAS + analysis for selected points and seeds.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Pass --skip-existing to build_asymmetric_sweep.py.",
    )
    parser.add_argument(
        "--run-highres",
        action="store_true",
        help="After seed aggregation, run an optional 501^3 depth-precision check on best-by-energy points.",
    )
    parser.add_argument(
        "--highres-bins",
        type=int,
        default=501,
        help="High-resolution bins used for optional depth-precision check.",
    )
    parser.add_argument(
        "--highres-histories",
        type=int,
        default=500000,
        help="Histories for optional high-resolution depth check.",
    )
    parser.add_argument(
        "--highres-seed",
        type=int,
        default=101,
        help="Seed for optional high-resolution depth check.",
    )
    parser.add_argument("--dpi", type=int, default=170, help="DPI for generated plots.")
    return parser.parse_args()


def run_command(cmd: List[str], cwd: Path) -> None:
    print("[cmd]", " ".join(cmd))
    proc = subprocess.run(cmd, cwd=str(cwd), text=True)
    if proc.returncode != 0:
        raise SystemExit(proc.returncode)


def load_reference(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_manifest_case_map(path: Path) -> Dict[str, Dict]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    out: Dict[str, Dict] = {}
    for case in payload.get("cases", []):
        out[str(case.get("case_id", ""))] = case
    return out


def select_cases(df: pd.DataFrame, energies: List[int], selection: str) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for energy in sorted(set(int(e) for e in energies)):
        block = df[df["energy_mev"].astype(int) == energy].copy()
        if block.empty:
            continue
        best = block.loc[block["weighted_error"].astype(float).idxmin()]
        rows.append(
            {
                "energy_mev": energy,
                "case_id": str(best["case_id"]),
                "g4_t_per_m": float(best["g4_t_per_m"]),
                "selection_reason": "best_weighted_error",
                "weighted_error": float(best["weighted_error"]),
                "delta_z_hat_cm": float(best["delta_z_hat_cm"]),
            }
        )
        if selection == "best_and_depth":
            depth = block.loc[block["delta_z_hat_cm"].astype(float).abs().idxmin()]
            rows.append(
                {
                    "energy_mev": energy,
                    "case_id": str(depth["case_id"]),
                    "g4_t_per_m": float(depth["g4_t_per_m"]),
                    "selection_reason": "best_depth_match",
                    "weighted_error": float(depth["weighted_error"]),
                    "delta_z_hat_cm": float(depth["delta_z_hat_cm"]),
                }
            )

    selected = pd.DataFrame(rows)
    if selected.empty:
        return selected
    selected = selected.sort_values(["energy_mev", "g4_t_per_m", "selection_reason"]).reset_index(drop=True)
    grouped = (
        selected.groupby(["energy_mev", "case_id", "g4_t_per_m"], as_index=False)
        .agg(
            selection_reason=("selection_reason", lambda s: ";".join(sorted(set(str(v) for v in s)))),
            weighted_error=("weighted_error", "min"),
            delta_z_hat_cm=("delta_z_hat_cm", "min"),
        )
        .sort_values(["energy_mev", "g4_t_per_m"])
        .reset_index(drop=True)
    )
    return grouped


def append_case_overrides(selected: pd.DataFrame, case_map: Dict[str, Dict]) -> pd.DataFrame:
    out = selected.copy()
    for col in [
        "gradient_x_scale",
        "gradient_y_scale",
        "source_sigma_x_mm",
        "source_sigma_y_mm",
        "source_angular_x_mrad",
        "source_angular_y_mrad",
        "source_energy_spread_mev",
    ]:
        out[col] = np.nan

    for idx, row in out.iterrows():
        case = case_map.get(str(row["case_id"]))
        if not case:
            continue
        for col in [
            "gradient_x_scale",
            "gradient_y_scale",
            "source_sigma_x_mm",
            "source_sigma_y_mm",
            "source_angular_x_mrad",
            "source_angular_y_mrad",
            "source_energy_spread_mev",
        ]:
            if case.get(col) is not None:
                out.at[idx, col] = float(case[col])
    return out


def ensure_required_case_columns(df: pd.DataFrame) -> None:
    required = [
        "case_id",
        "energy_mev",
        "g4_t_per_m",
        "weighted_error",
        "delta_z_hat_cm",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in case_metrics.csv: {missing}")


def build_rerun_command(
    args: argparse.Namespace,
    repo_root: Path,
    selected_row: pd.Series,
    seed: int,
    run_root: Path,
    histories: int,
    xbins: int | None = None,
    ybins: int | None = None,
    zbins: int | None = None,
) -> List[str]:
    energy = int(selected_row["energy_mev"])
    g4 = float(selected_row["g4_t_per_m"])
    g4_range = f"{g4}:{g4}:0.1"

    cmd = [
        args.python_bin,
        "scripts/build_asymmetric_sweep.py",
        "--run-root",
        str(run_root),
        "--energies",
        str(energy),
        "--g4-range",
        g4_range,
        "--histories",
        str(histories),
        "--threads",
        str(args.threads),
        "--seed",
        str(seed),
        "--run-topas",
        "--topas-bin",
        args.topas_bin,
        "--g4-data-dir",
        args.g4_data_dir,
        "--physics-profile",
        args.physics_profile,
        "--quad-gradient-convention",
        args.quad_gradient_convention,
        "--case-run-retries",
        str(args.case_run_retries),
    ]

    if math_is_finite(selected_row.get("gradient_x_scale")):
        cmd.extend(["--gradient-x-scale", f"{float(selected_row['gradient_x_scale']):.6f}"])
    if math_is_finite(selected_row.get("gradient_y_scale")):
        cmd.extend(["--gradient-y-scale", f"{float(selected_row['gradient_y_scale']):.6f}"])
    if math_is_finite(selected_row.get("source_sigma_x_mm")):
        cmd.extend(["--source-sigma-x-mm", f"{float(selected_row['source_sigma_x_mm']):.6f}"])
    if math_is_finite(selected_row.get("source_sigma_y_mm")):
        cmd.extend(["--source-sigma-y-mm", f"{float(selected_row['source_sigma_y_mm']):.6f}"])
    if math_is_finite(selected_row.get("source_angular_x_mrad")):
        cmd.extend(["--source-angular-x-mrad", f"{float(selected_row['source_angular_x_mrad']):.6f}"])
    if math_is_finite(selected_row.get("source_angular_y_mrad")):
        cmd.extend(["--source-angular-y-mrad", f"{float(selected_row['source_angular_y_mrad']):.6f}"])
    if math_is_finite(selected_row.get("source_energy_spread_mev")):
        cmd.extend(["--source-energy-spread-mev", f"{float(selected_row['source_energy_spread_mev']):.6f}"])

    if xbins is not None and ybins is not None and zbins is not None:
        cmd.extend(["--xbins", str(xbins), "--ybins", str(ybins), "--zbins", str(zbins)])

    if args.skip_existing:
        cmd.append("--skip-existing")
    return cmd


def build_analysis_command(args: argparse.Namespace, run_root: Path, outdir: Path) -> List[str]:
    return [
        args.python_bin,
        "scripts/analyze_topas_outputs.py",
        "--manifest",
        str(run_root / "manifest.json"),
        "--outdir",
        str(outdir),
        "--z-mode",
        args.z_mode,
        "--sigma-mode",
        args.sigma_mode,
        "--sigma-fit-mode",
        args.sigma_fit_mode,
        "--io-retries",
        "10",
        "--io-retry-delay-sec",
        "1.0",
    ]


def math_is_finite(value: object) -> bool:
    try:
        val = float(value)
    except Exception:
        return False
    return np.isfinite(val)


def load_single_metric_row(case_metrics_csv: Path) -> Dict[str, object]:
    frame = pd.read_csv(case_metrics_csv)
    if frame.empty:
        raise RuntimeError(f"No rows in {case_metrics_csv}")
    return dict(frame.iloc[0])


def aggregate_replicates(df: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "z_hat_selected_cm",
        "sigma_x_selected_cm",
        "sigma_y_selected_cm",
        "entrance_on_axis_pct",
        "delta_z_hat_cm",
        "delta_sigma_x_cm",
        "delta_sigma_y_cm",
        "delta_entrance_pct",
        "weighted_error",
    ]
    group_cols = ["energy_mev", "case_id", "g4_t_per_m", "selection_reason"]
    agg: Dict[str, Tuple[str, str]] = {
        "seed": ("seed", "count"),
    }
    for col in metrics:
        agg[f"{col}_mean"] = (col, "mean")
        agg[f"{col}_std"] = (col, "std")
    for ref_col in ["benchmark_z_hat_cm", "benchmark_sigma_x_cm", "benchmark_sigma_y_cm", "benchmark_entrance_pct"]:
        if ref_col in df.columns:
            agg[ref_col] = (ref_col, "first")

    out = df.groupby(group_cols, as_index=False).agg(**agg)
    out = out.rename(columns={"seed": "n_seeds"})
    for col in out.columns:
        if col.endswith("_std"):
            out[col] = out[col].fillna(0.0)
    return out.sort_values(["energy_mev", "g4_t_per_m", "case_id"]).reset_index(drop=True)


def best_by_energy(agg: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for energy, block in agg.groupby("energy_mev"):
        rows.append(block.loc[block["weighted_error_mean"].idxmin()])
    return pd.DataFrame(rows).sort_values("energy_mev").reset_index(drop=True)


def plot_errorbars(agg: pd.DataFrame, reference: Dict, outdir: Path, histories: int, seeds: List[int], dpi: int) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover
        raise SystemExit(
            "matplotlib is required for plotting. Install with: python3 -m pip install matplotlib"
        ) from exc

    outdir.mkdir(parents=True, exist_ok=True)
    colors = {100: "#1f77b4", 200: "#2ca02c", 250: "#d62728"}
    caption_meta = f"Histories/case={histories:,}; seeds={','.join(str(s) for s in seeds)}."

    metrics = [
        ("z_hat_selected_cm", "Depth of Maximum Dose z_hat (cm)", "errorbar_zhat_vs_g4.png"),
        ("sigma_x_selected_cm", "Transverse Sigma_x at z_hat (cm)", "errorbar_sigma_x_vs_g4.png"),
        ("sigma_y_selected_cm", "Transverse Sigma_y at z_hat (cm)", "errorbar_sigma_y_vs_g4.png"),
        ("entrance_on_axis_pct", "Entrance Dose / Maximum Dose (%)", "errorbar_entrance_vs_g4.png"),
    ]

    for metric, ylabel, fname in metrics:
        fig, ax = plt.subplots(figsize=(10.5, 6.0), constrained_layout=False)
        for energy, block in agg.groupby("energy_mev"):
            block = block.sort_values("g4_t_per_m")
            ax.errorbar(
                block["g4_t_per_m"],
                block[f"{metric}_mean"],
                yerr=block[f"{metric}_std"],
                marker="o",
                linewidth=2.0,
                capsize=4,
                color=colors.get(int(energy), None),
                label=f"TOPAS {int(energy)} MeV (mean ±1σ)",
            )
            entry = reference["asymmetric_beamline"]["energies"][str(int(energy))]
            g4_nom = float(entry["baseline_gradients_t_per_m"][3])
            bm = entry["benchmark_metrics_table2"]
            bm_value = (
                float(bm["z_hat_cm"])
                if metric == "z_hat_selected_cm"
                else (
                    float(bm["sigma_x_cm"])
                    if metric == "sigma_x_selected_cm"
                    else (
                        float(bm["sigma_y_cm"])
                        if metric == "sigma_y_selected_cm"
                        else float(bm["entrance_dose_pct"])
                    )
                )
            )
            ax.scatter(
                [g4_nom],
                [bm_value],
                marker="*",
                s=180,
                color=colors.get(int(energy), None),
                edgecolor="black",
                linewidth=0.5,
                label=f"Whitmore {int(energy)} MeV",
            )

            if metric == "z_hat_selected_cm":
                t3 = pd.DataFrame(entry["supplementary_table3_scan"]).astype(float)
                ax.plot(
                    t3["g4_t_per_m"],
                    t3["z_hat_cm"],
                    linestyle="--",
                    linewidth=1.6,
                    color=colors.get(int(energy), None),
                    alpha=0.75,
                    label=f"Table 3 trend {int(energy)} MeV",
                )

        ax.set_xlabel("Q4 Gradient (T/m)")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.25)
        title = {
            "z_hat_selected_cm": "Depth Trend with Uncertainty",
            "sigma_x_selected_cm": "Sigma_x Trend with Uncertainty",
            "sigma_y_selected_cm": "Sigma_y Trend with Uncertainty",
            "entrance_on_axis_pct": "Entrance-Dose Trend with Uncertainty",
        }[metric]
        ax.set_title(title)
        ax.legend(fontsize=8.2, ncol=2)
        fig.text(
            0.01,
            0.01,
            f"Caption: Mean±1σ over seed reruns. {caption_meta}",
            ha="left",
            va="bottom",
            fontsize=9,
        )
        fig.subplots_adjust(bottom=0.16)
        fig.savefig(outdir / fname, dpi=dpi)
        plt.close(fig)


def write_markdown_report(
    out_file: Path,
    selected: pd.DataFrame,
    agg: pd.DataFrame | None,
    best_df: pd.DataFrame | None,
    args: argparse.Namespace,
) -> None:
    lines = ["# Publishable Subset Workflow Report", ""]
    lines.append(f"- Selection rule: `{args.selection}`")
    lines.append(f"- Energies: `{', '.join(str(e) for e in sorted(args.energies))}`")
    lines.append(f"- Histories/case (reruns): `{args.histories}`")
    lines.append(f"- Seeds: `{', '.join(str(s) for s in args.seeds)}`")
    lines.append(f"- Threads: `{args.threads}`")
    lines.append(f"- Sigma fit mode: `{args.sigma_fit_mode}`")
    lines.append("")
    lines.append("## Selected Cases")
    lines.append("")
    if selected.empty:
        lines.append("No cases selected.")
    else:
        lines.append("| Energy | Case ID | Q4 (T/m) | Reason |")
        lines.append("|---:|---|---:|---|")
        for _, row in selected.iterrows():
            lines.append(
                f"| {int(row['energy_mev'])} | {row['case_id']} | {float(row['g4_t_per_m']):.4f} | {row['selection_reason']} |"
            )
    lines.append("")

    if agg is not None and not agg.empty:
        lines.append("## Seed-Aggregated Results")
        lines.append("")
        lines.append(
            "| Energy | Case ID | Q4 (T/m) | n | z_hat mean±std (cm) | sigma_x mean±std (cm) | sigma_y mean±std (cm) | entrance mean±std (%) | weighted error mean±std |"
        )
        lines.append("|---:|---|---:|---:|---:|---:|---:|---:|---:|")
        for _, row in agg.sort_values(["energy_mev", "g4_t_per_m"]).iterrows():
            lines.append(
                f"| {int(row['energy_mev'])} | {row['case_id']} | {float(row['g4_t_per_m']):.4f} | "
                f"{int(row['n_seeds'])} | "
                f"{float(row['z_hat_selected_cm_mean']):.3f}±{float(row['z_hat_selected_cm_std']):.3f} | "
                f"{float(row['sigma_x_selected_cm_mean']):.3f}±{float(row['sigma_x_selected_cm_std']):.3f} | "
                f"{float(row['sigma_y_selected_cm_mean']):.3f}±{float(row['sigma_y_selected_cm_std']):.3f} | "
                f"{float(row['entrance_on_axis_pct_mean']):.2f}±{float(row['entrance_on_axis_pct_std']):.2f} | "
                f"{float(row['weighted_error_mean']):.3f}±{float(row['weighted_error_std']):.3f} |"
            )
        lines.append("")

    if best_df is not None and not best_df.empty:
        lines.append("## Best by Energy (mean weighted error)")
        lines.append("")
        lines.append("| Energy | Case ID | Q4 (T/m) | weighted error mean±std |")
        lines.append("|---:|---|---:|---:|")
        for _, row in best_df.iterrows():
            lines.append(
                f"| {int(row['energy_mev'])} | {row['case_id']} | {float(row['g4_t_per_m']):.4f} | "
                f"{float(row['weighted_error_mean']):.3f}±{float(row['weighted_error_std']):.3f} |"
            )
        lines.append("")

    out_file.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_highres_checks(
    args: argparse.Namespace,
    repo_root: Path,
    selected_best: pd.DataFrame,
    report_root: Path,
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    highres_root = report_root / "highres_501"
    highres_root.mkdir(parents=True, exist_ok=True)

    for _, row in selected_best.iterrows():
        energy = int(row["energy_mev"])
        cid = str(row["case_id"])
        run_dir = highres_root / f"{cid}_seed{args.highres_seed}"
        analysis_dir = run_dir / "analysis"
        cmd_build = build_rerun_command(
            args=args,
            repo_root=repo_root,
            selected_row=row,
            seed=args.highres_seed,
            run_root=run_dir,
            histories=args.highres_histories,
            xbins=args.highres_bins,
            ybins=args.highres_bins,
            zbins=args.highres_bins,
        )
        cmd_an = build_analysis_command(args, run_root=run_dir, outdir=analysis_dir)
        run_command(cmd_build, repo_root)
        run_command(cmd_an, repo_root)

        metric = load_single_metric_row(analysis_dir / "case_metrics.csv")
        metric["energy_mev"] = energy
        metric["case_id_base"] = cid
        metric["seed"] = args.highres_seed
        metric["xbins"] = args.highres_bins
        metric["ybins"] = args.highres_bins
        metric["zbins"] = args.highres_bins
        metric["histories"] = args.highres_histories
        metric["run_dir"] = str(run_dir.resolve())
        rows.append(metric)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(["energy_mev", "case_id_base"]).reset_index(drop=True)


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    args.run_root.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.case_metrics)
    ensure_required_case_columns(df)
    df = df[df["energy_mev"].astype(int).isin(args.energies)].copy()
    if df.empty:
        raise RuntimeError("No rows in case_metrics for selected energies.")

    selected = select_cases(df, args.energies, args.selection)
    case_map = load_manifest_case_map(args.manifest)
    selected = append_case_overrides(selected, case_map)
    selected_file = args.run_root / "selected_cases.csv"
    selected.to_csv(selected_file, index=False)
    print(f"Selected cases: {selected_file}")

    if selected.empty:
        print("No cases selected. Exiting.")
        return 1

    # Always export a command plan for reproducibility.
    plan_file = args.run_root / "planned_commands.txt"
    with plan_file.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["energy_mev", "case_id", "g4_t_per_m", "seed", "run_root", "command"])
        for _, row in selected.iterrows():
            for seed in args.seeds:
                run_label = f"{row['case_id']}_seed{seed}"
                run_root = args.run_root / "seed_runs" / run_label
                cmd = build_rerun_command(
                    args=args,
                    repo_root=repo_root,
                    selected_row=row,
                    seed=int(seed),
                    run_root=run_root,
                    histories=args.histories,
                )
                writer.writerow(
                    [
                        int(row["energy_mev"]),
                        str(row["case_id"]),
                        float(row["g4_t_per_m"]),
                        int(seed),
                        str(run_root),
                        " ".join(cmd),
                    ]
                )
    print(f"Planned commands: {plan_file}")

    replicate_rows: List[Dict[str, object]] = []
    if args.run_topas:
        for _, row in selected.iterrows():
            for seed in args.seeds:
                run_label = f"{row['case_id']}_seed{seed}"
                run_root = args.run_root / "seed_runs" / run_label
                analysis_root = run_root / "analysis"
                cmd_build = build_rerun_command(
                    args=args,
                    repo_root=repo_root,
                    selected_row=row,
                    seed=int(seed),
                    run_root=run_root,
                    histories=args.histories,
                )
                cmd_an = build_analysis_command(args, run_root=run_root, outdir=analysis_root)
                run_command(cmd_build, repo_root)
                run_command(cmd_an, repo_root)

                metric = load_single_metric_row(analysis_root / "case_metrics.csv")
                metric["seed"] = int(seed)
                metric["selection_reason"] = str(row["selection_reason"])
                metric["case_id_base"] = str(row["case_id"])
                metric["run_dir"] = str(run_root.resolve())
                replicate_rows.append(metric)
    else:
        print("--run-topas not provided: generated selection + command plan only.")

    agg_df = pd.DataFrame()
    best_df = pd.DataFrame()

    if replicate_rows:
        rep_df = pd.DataFrame(replicate_rows)
        rep_file = args.run_root / "replicate_metrics.csv"
        rep_df.to_csv(rep_file, index=False)
        print(f"Replicate metrics: {rep_file}")

        agg_df = aggregate_replicates(rep_df)
        agg_file = args.run_root / "aggregate_seed_stats.csv"
        agg_df.to_csv(agg_file, index=False)
        print(f"Aggregate seed stats: {agg_file}")

        best_df = best_by_energy(agg_df)
        # Preserve per-case generator overrides for optional high-resolution reruns.
        override_cols = [
            "energy_mev",
            "case_id",
            "g4_t_per_m",
            "gradient_x_scale",
            "gradient_y_scale",
            "source_sigma_x_mm",
            "source_sigma_y_mm",
            "source_angular_x_mrad",
            "source_angular_y_mrad",
            "source_energy_spread_mev",
        ]
        best_df = best_df.merge(
            selected[override_cols],
            on=["energy_mev", "case_id", "g4_t_per_m"],
            how="left",
        )
        best_file = args.run_root / "best_by_energy_seedmean.csv"
        best_df.to_csv(best_file, index=False)
        print(f"Best by energy (seed mean): {best_file}")

        reference = load_reference(args.reference)
        plots_dir = args.run_root / "plots"
        plot_errorbars(
            agg=agg_df,
            reference=reference,
            outdir=plots_dir,
            histories=args.histories,
            seeds=[int(s) for s in args.seeds],
            dpi=args.dpi,
        )
        print(f"Error-bar plots: {plots_dir}")

        if args.run_highres:
            highres_df = run_highres_checks(args=args, repo_root=repo_root, selected_best=best_df, report_root=args.run_root)
            if not highres_df.empty:
                highres_file = args.run_root / "highres_501" / "highres_metrics.csv"
                highres_df.to_csv(highres_file, index=False)
                print(f"High-resolution depth checks: {highres_file}")

    report_file = args.run_root / "publishable_subset_report.md"
    write_markdown_report(report_file, selected=selected, agg=agg_df, best_df=best_df, args=args)
    print(f"Report: {report_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
