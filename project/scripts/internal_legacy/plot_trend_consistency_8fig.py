#!/usr/bin/env python3
"""Generate 8 Whitmore-comparison figures for trend-consistency assessment."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
except Exception as exc:  # pragma: no cover
    raise SystemExit(
        "matplotlib is required for plotting. Install with: python3 -m pip install matplotlib"
    ) from exc

from analyze_topas_outputs import load_topas_grid


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description=(
            "Generate 8 figures to compare TOPAS beamline trends against Whitmore."
        )
    )
    parser.add_argument(
        "--case-metrics",
        type=Path,
        default=root / "runs" / "analysis_paper2d" / "case_metrics.csv",
        help="Input metrics CSV from analyze_topas_outputs.py.",
    )
    parser.add_argument(
        "--reference",
        type=Path,
        default=root / "config" / "benchmark_reference.json",
        help="Whitmore reference JSON.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=root / "runs" / "manifest.json",
        help="Manifest used for histories/thread metadata in captions.",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=root / "runs" / "plots_trend_consistency",
        help="Output directory for the 8 figures and caption index.",
    )
    parser.add_argument(
        "--energies",
        nargs="+",
        type=int,
        default=[100, 200, 250],
        help="Energies to include.",
    )
    parser.add_argument(
        "--z-col",
        type=str,
        default="z_hat_integrated_cm",
        help="Depth metric column in case_metrics.csv.",
    )
    parser.add_argument(
        "--sigma-x-col",
        type=str,
        default="sigma_x_integrated_cm",
        help="Sigma-x metric column in case_metrics.csv.",
    )
    parser.add_argument(
        "--sigma-y-col",
        type=str,
        default="sigma_y_integrated_cm",
        help="Sigma-y metric column in case_metrics.csv.",
    )
    parser.add_argument(
        "--entrance-col",
        type=str,
        default="entrance_on_axis_pct",
        help="Entrance dose column in case_metrics.csv.",
    )
    parser.add_argument(
        "--exit-col",
        type=str,
        default="exit_on_axis_pct",
        help="Exit dose column in case_metrics.csv.",
    )
    parser.add_argument("--dpi", type=int, default=170, help="Figure DPI.")
    return parser.parse_args()


def load_reference(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def table2_metrics(reference: Dict, energy_mev: int) -> Dict[str, float]:
    entry = reference["asymmetric_beamline"]["energies"][str(int(energy_mev))]
    m = entry["benchmark_metrics_table2"]
    return {
        "z_hat_cm": float(m["z_hat_cm"]),
        "sigma_x_cm": float(m["sigma_x_cm"]),
        "sigma_y_cm": float(m["sigma_y_cm"]),
        "entrance_dose_pct": float(m["entrance_dose_pct"]),
    }


def table3_scan(reference: Dict, energy_mev: int) -> pd.DataFrame:
    entry = reference["asymmetric_beamline"]["energies"][str(int(energy_mev))]
    rows = []
    for row in entry["supplementary_table3_scan"]:
        rows.append(
            {
                "g4_t_per_m": float(row["g4_t_per_m"]),
                "z_hat_cm": float(row["z_hat_cm"]),
            }
        )
    return pd.DataFrame(rows).sort_values("g4_t_per_m")


def baseline_q4(reference: Dict, energy_mev: int) -> float:
    entry = reference["asymmetric_beamline"]["energies"][str(int(energy_mev))]
    return float(entry["baseline_gradients_t_per_m"][3])


def caption_metadata(manifest_path: Path, df: pd.DataFrame) -> str:
    histories = "unknown"
    threads = "unknown"
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest.get("histories") is not None:
                histories = f"{int(manifest['histories']):,}"
            if manifest.get("threads") is not None:
                threads = str(manifest["threads"])
        except Exception:
            pass
    if {"nx", "ny", "nz"}.issubset(df.columns):
        nx = int(round(float(pd.to_numeric(df["nx"], errors="coerce").dropna().mode().iloc[0])))
        ny = int(round(float(pd.to_numeric(df["ny"], errors="coerce").dropna().mode().iloc[0])))
        nz = int(round(float(pd.to_numeric(df["nz"], errors="coerce").dropna().mode().iloc[0])))
        grid = f"{nx}x{ny}x{nz}"
    else:
        grid = "unknown"
    return f"Histories/case={histories}, threads={threads}, phantom grid={grid}."


def energy_scan_summary(df: pd.DataFrame, energies: List[int]) -> str:
    parts: List[str] = []
    for energy in energies:
        block = df[df["energy_mev"].astype(int) == energy]
        if block.empty:
            continue
        gmin = float(block["g4_t_per_m"].min())
        gmax = float(block["g4_t_per_m"].max())
        n = int(len(block))
        parts.append(f"{energy} MeV: Q4 {gmin:.1f}-{gmax:.1f} T/m (n={n})")
    return "; ".join(parts)


def benchmark_summary(reference: Dict, energies: List[int]) -> str:
    parts: List[str] = []
    for energy in energies:
        bench = table2_metrics(reference, int(energy))
        g4_0 = baseline_q4(reference, int(energy))
        parts.append(
            (
                f"{energy} MeV (Q4={g4_0:.1f} T/m): "
                f"z_hat={bench['z_hat_cm']:.2f} cm, sigma_x={bench['sigma_x_cm']:.2f} cm, "
                f"sigma_y={bench['sigma_y_cm']:.2f} cm, entrance={bench['entrance_dose_pct']:.1f}%"
            )
        )
    return "; ".join(parts)


def best_case_summary(df: pd.DataFrame, energies: List[int], args: argparse.Namespace) -> str:
    parts: List[str] = []
    for energy in energies:
        block = df[df["energy_mev"].astype(int) == int(energy)]
        if block.empty:
            continue
        best = block.loc[block["weighted_error"].idxmin()]
        parts.append(
            (
                f"{energy} MeV best at Q4={float(best['g4_t_per_m']):.2f} T/m: "
                f"z_hat={float(best[args.z_col]):.2f} cm, "
                f"sigma_x={float(best[args.sigma_x_col]):.2f} cm, "
                f"sigma_y={float(best[args.sigma_y_col]):.2f} cm, "
                f"entrance={float(best[args.entrance_col]):.1f}%, "
                f"exit={float(best[args.exit_col]):.1f}%"
            )
        )
    return "; ".join(parts)


def ensure_columns(df: pd.DataFrame, args: argparse.Namespace, reference: Dict) -> pd.DataFrame:
    needed = [
        "case_id",
        "energy_mev",
        "g4_t_per_m",
        args.z_col,
        args.sigma_x_col,
        args.sigma_y_col,
        args.entrance_col,
        args.exit_col,
    ]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in case_metrics.csv: {missing}")

    out = df.copy()
    for energy in sorted(out["energy_mev"].astype(int).unique().tolist()):
        b = table2_metrics(reference, int(energy))
        mask = out["energy_mev"].astype(int) == int(energy)
        out.loc[mask, "benchmark_z_hat_cm"] = b["z_hat_cm"]
        out.loc[mask, "benchmark_sigma_x_cm"] = b["sigma_x_cm"]
        out.loc[mask, "benchmark_sigma_y_cm"] = b["sigma_y_cm"]
        out.loc[mask, "benchmark_entrance_pct"] = b["entrance_dose_pct"]

    out["delta_z_hat_cm"] = pd.to_numeric(out[args.z_col], errors="coerce") - out["benchmark_z_hat_cm"]
    out["delta_sigma_x_cm"] = pd.to_numeric(out[args.sigma_x_col], errors="coerce") - out["benchmark_sigma_x_cm"]
    out["delta_sigma_y_cm"] = pd.to_numeric(out[args.sigma_y_col], errors="coerce") - out["benchmark_sigma_y_cm"]
    out["delta_entrance_pct"] = pd.to_numeric(out[args.entrance_col], errors="coerce") - out["benchmark_entrance_pct"]

    out["z_over_bench"] = out[args.z_col] / out["benchmark_z_hat_cm"].replace(0.0, np.nan)
    out["sx_over_bench"] = out[args.sigma_x_col] / out["benchmark_sigma_x_cm"].replace(0.0, np.nan)
    out["sy_over_bench"] = out[args.sigma_y_col] / out["benchmark_sigma_y_cm"].replace(0.0, np.nan)
    out["entrance_over_bench"] = out[args.entrance_col] / out["benchmark_entrance_pct"].replace(0.0, np.nan)

    out["combined_sigma_abs_delta_cm"] = 0.5 * (
        out["delta_sigma_x_cm"].abs() + out["delta_sigma_y_cm"].abs()
    )
    if "weighted_error" not in out.columns:
        out["weighted_error"] = np.sqrt(
            (out["delta_z_hat_cm"] / 0.5) ** 2
            + (out["delta_sigma_x_cm"] / 0.2) ** 2
            + (out["delta_sigma_y_cm"] / 0.2) ** 2
            + (out["delta_entrance_pct"] / 5.0) ** 2
        )

    return out


def add_caption(fig: plt.Figure, caption: str) -> None:
    fig.text(0.01, 0.01, caption, ha="left", va="bottom", fontsize=9, wrap=True)
    fig.subplots_adjust(bottom=0.18)


def save_figure(fig: plt.Figure, path: Path, dpi: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def metric_std_col(df: pd.DataFrame, col: str) -> str | None:
    std_col = f"{col}_std"
    return std_col if std_col in df.columns else None


def plot_metric_series(
    ax: plt.Axes,
    x: pd.Series,
    y: pd.Series,
    yerr: pd.Series | None,
    *,
    marker: str,
    color: str | None,
    linewidth: float,
    label: str,
) -> None:
    if yerr is not None:
        ax.errorbar(
            x,
            y,
            yerr=yerr,
            marker=marker,
            color=color,
            linewidth=linewidth,
            capsize=4,
            label=label,
        )
    else:
        ax.plot(x, y, marker=marker, color=color, linewidth=linewidth, label=label)


def main() -> int:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    reference = load_reference(args.reference)
    df = pd.read_csv(args.case_metrics)
    df = df[df["energy_mev"].astype(int).isin(args.energies)].copy()
    if df.empty:
        raise RuntimeError("No rows found for requested energies.")
    df = ensure_columns(df, args, reference)

    z_std_col = metric_std_col(df, args.z_col)
    sx_std_col = metric_std_col(df, args.sigma_x_col)
    sy_std_col = metric_std_col(df, args.sigma_y_col)
    en_std_col = metric_std_col(df, args.entrance_col)
    ex_std_col = metric_std_col(df, args.exit_col)

    if z_std_col is not None:
        df["delta_z_hat_cm_std"] = pd.to_numeric(df[z_std_col], errors="coerce").fillna(0.0)
        df["z_over_bench_std"] = pd.to_numeric(df[z_std_col], errors="coerce") / df["benchmark_z_hat_cm"].replace(0.0, np.nan)
    if sx_std_col is not None:
        df["delta_sigma_x_cm_std"] = pd.to_numeric(df[sx_std_col], errors="coerce").fillna(0.0)
        df["sx_over_bench_std"] = pd.to_numeric(df[sx_std_col], errors="coerce") / df["benchmark_sigma_x_cm"].replace(0.0, np.nan)
    if sy_std_col is not None:
        df["delta_sigma_y_cm_std"] = pd.to_numeric(df[sy_std_col], errors="coerce").fillna(0.0)
        df["sy_over_bench_std"] = pd.to_numeric(df[sy_std_col], errors="coerce") / df["benchmark_sigma_y_cm"].replace(0.0, np.nan)
    if en_std_col is not None:
        df["delta_entrance_pct_std"] = pd.to_numeric(df[en_std_col], errors="coerce").fillna(0.0)
        df["entrance_over_bench_std"] = pd.to_numeric(df[en_std_col], errors="coerce") / df["benchmark_entrance_pct"].replace(0.0, np.nan)

    meta = caption_metadata(args.manifest, df)
    has_seed_uncertainty = any(col is not None for col in [z_std_col, sx_std_col, sy_std_col, en_std_col, ex_std_col])
    uncertainty_note = " Error bars show ±1σ across seeds." if has_seed_uncertainty else ""
    colors = {100: "#1f77b4", 200: "#2ca02c", 250: "#d62728"}
    markers = {100: "o", 200: "s", 250: "^"}
    energies = sorted(df["energy_mev"].astype(int).unique().tolist())
    scan_summary = energy_scan_summary(df, energies)
    bench_summary = benchmark_summary(reference, energies)
    best_summary = best_case_summary(df, energies, args)

    captions: List[Tuple[str, str]] = []

    # Figure 1: z_hat trend vs Whitmore Table 3
    fig, ax = plt.subplots(figsize=(10.8, 6.2), constrained_layout=False)
    for energy in energies:
        block = df[df["energy_mev"].astype(int) == energy].sort_values("g4_t_per_m")
        t3 = table3_scan(reference, energy)
        yerr = block[z_std_col] if z_std_col is not None else None
        plot_metric_series(
            ax,
            block["g4_t_per_m"],
            block[args.z_col],
            yerr,
            marker=markers.get(energy, "o"),
            color=colors.get(energy, None),
            linewidth=2.0,
            label=f"TOPAS {energy} MeV" + (" (mean ±1σ)" if z_std_col is not None else ""),
        )
        ax.plot(
            t3["g4_t_per_m"],
            t3["z_hat_cm"],
            linestyle="--",
            color=colors.get(energy, None),
            linewidth=1.8,
            alpha=0.9,
            label=f"Whitmore {energy} MeV (Table 3)",
        )
    ax.set_xlabel("Q4 Gradient (T/m)")
    ax.set_ylabel("Depth of Maximum Dose, z_hat (cm)")
    ax.set_title("Figure 1: Focal-Depth Trend vs Q4 Gradient")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=9, ncol=2)
    caption = (
        "Caption: Focal depth trend (z_hat) across Q4 sweep for 100/200/250 MeV. "
        "Solid curves are TOPAS; dashed curves are Whitmore Supplementary Table 3. "
        f"Sweep ranges: {scan_summary}. "
        f"Table 2 Whitmore anchors: {bench_summary}. "
        f"{meta}{uncertainty_note}"
    )
    add_caption(fig, caption)
    path = args.outdir / "fig01_zhat_vs_g4.png"
    save_figure(fig, path, args.dpi)
    captions.append((path.name, caption))

    # Figure 2: sigma_x trend
    fig, ax = plt.subplots(figsize=(10.8, 6.2), constrained_layout=False)
    for energy in energies:
        block = df[df["energy_mev"].astype(int) == energy].sort_values("g4_t_per_m")
        bench = table2_metrics(reference, energy)
        g4_0 = baseline_q4(reference, energy)
        yerr = block[sx_std_col] if sx_std_col is not None else None
        plot_metric_series(
            ax,
            block["g4_t_per_m"],
            block[args.sigma_x_col],
            yerr,
            marker=markers.get(energy, "o"),
            color=colors.get(energy, None),
            linewidth=2.0,
            label=f"TOPAS {energy} MeV" + (" (mean ±1σ)" if sx_std_col is not None else ""),
        )
        ax.scatter(
            [g4_0],
            [bench["sigma_x_cm"]],
            marker="*",
            s=180,
            color=colors.get(energy, None),
            edgecolor="black",
            linewidth=0.6,
            label=f"Whitmore {energy} MeV (Table 2 point)",
        )
    ax.set_xlabel("Q4 Gradient (T/m)")
    ax.set_ylabel("Transverse Sigma_x at z_hat (cm)")
    ax.set_title("Figure 2: Sigma_x Trend vs Q4 Gradient")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8.5, ncol=2)
    caption = (
        "Caption: Sigma_x versus Q4 gradient for all energies. "
        "Stars mark Table 2 Whitmore values at nominal Q4 for each energy. "
        f"Best observed sigma_x points: {best_summary}. "
        f"{meta}{uncertainty_note}"
    )
    add_caption(fig, caption)
    path = args.outdir / "fig02_sigma_x_vs_g4.png"
    save_figure(fig, path, args.dpi)
    captions.append((path.name, caption))

    # Figure 3: sigma_y trend
    fig, ax = plt.subplots(figsize=(10.8, 6.2), constrained_layout=False)
    for energy in energies:
        block = df[df["energy_mev"].astype(int) == energy].sort_values("g4_t_per_m")
        bench = table2_metrics(reference, energy)
        g4_0 = baseline_q4(reference, energy)
        yerr = block[sy_std_col] if sy_std_col is not None else None
        plot_metric_series(
            ax,
            block["g4_t_per_m"],
            block[args.sigma_y_col],
            yerr,
            marker=markers.get(energy, "o"),
            color=colors.get(energy, None),
            linewidth=2.0,
            label=f"TOPAS {energy} MeV" + (" (mean ±1σ)" if sy_std_col is not None else ""),
        )
        ax.scatter(
            [g4_0],
            [bench["sigma_y_cm"]],
            marker="*",
            s=180,
            color=colors.get(energy, None),
            edgecolor="black",
            linewidth=0.6,
            label=f"Whitmore {energy} MeV (Table 2 point)",
        )
    ax.set_xlabel("Q4 Gradient (T/m)")
    ax.set_ylabel("Transverse Sigma_y at z_hat (cm)")
    ax.set_title("Figure 3: Sigma_y Trend vs Q4 Gradient")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8.5, ncol=2)
    caption = (
        "Caption: Sigma_y versus Q4 gradient for all energies. "
        "Stars mark Table 2 Whitmore values at nominal Q4 for each energy. "
        f"Table 2 Whitmore anchors: {bench_summary}. "
        f"{meta}{uncertainty_note}"
    )
    add_caption(fig, caption)
    path = args.outdir / "fig03_sigma_y_vs_g4.png"
    save_figure(fig, path, args.dpi)
    captions.append((path.name, caption))

    # Figure 4: entrance + exit trends
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 5.4), constrained_layout=False)
    ax_en, ax_ex = axes
    for energy in energies:
        block = df[df["energy_mev"].astype(int) == energy].sort_values("g4_t_per_m")
        bench = table2_metrics(reference, energy)
        g4_0 = baseline_q4(reference, energy)
        yerr_en = block[en_std_col] if en_std_col is not None else None
        plot_metric_series(
            ax_en,
            block["g4_t_per_m"],
            block[args.entrance_col],
            yerr_en,
            marker=markers.get(energy, "o"),
            color=colors.get(energy, None),
            linewidth=2.0,
            label=f"TOPAS {energy} MeV" + (" (mean ±1σ)" if en_std_col is not None else ""),
        )
        ax_en.scatter(
            [g4_0],
            [bench["entrance_dose_pct"]],
            marker="*",
            s=180,
            color=colors.get(energy, None),
            edgecolor="black",
            linewidth=0.6,
            label=f"Whitmore {energy} MeV (Table 2 point)",
        )
        yerr_ex = block[ex_std_col] if ex_std_col is not None else None
        plot_metric_series(
            ax_ex,
            block["g4_t_per_m"],
            block[args.exit_col],
            yerr_ex,
            marker=markers.get(energy, "o"),
            color=colors.get(energy, None),
            linewidth=2.0,
            label=f"TOPAS {energy} MeV" + (" (mean ±1σ)" if ex_std_col is not None else ""),
        )
    ax_en.set_xlabel("Q4 Gradient (T/m)")
    ax_en.set_ylabel("Entrance Dose / Maximum Dose (%)")
    ax_en.set_title("Entrance Dose Trend")
    ax_en.grid(alpha=0.25)
    ax_en.legend(fontsize=8.2, ncol=1)

    ax_ex.set_xlabel("Q4 Gradient (T/m)")
    ax_ex.set_ylabel("Exit Dose / Maximum Dose (%)")
    ax_ex.set_title("Exit Dose Trend")
    ax_ex.grid(alpha=0.25)
    ax_ex.legend(fontsize=8.2, ncol=1)

    fig.suptitle("Figure 4: Entrance/Exit Normalized Dose Trends", y=0.98)
    caption = (
        "Caption: Left panel shows entrance dose normalized to each case maximum and compared to Table 2 Whitmore points; "
        "right panel shows exit dose trend (paper does not report a Table 2 exit Whitmore). "
        "Normalization uses 100 * dose(z)/dose(max) per case. "
        f"Best observed points: {best_summary}. "
        f"{meta}{uncertainty_note}"
    )
    add_caption(fig, caption)
    path = args.outdir / "fig04_entrance_exit_vs_g4.png"
    save_figure(fig, path, args.dpi)
    captions.append((path.name, caption))

    # Figure 5: normalized ratios
    fig, axes = plt.subplots(2, 2, figsize=(12.6, 8.6), constrained_layout=False)
    panels = [
        ("z_over_bench", "z_hat / Whitmore (-)", "z_over_bench_std"),
        ("sx_over_bench", "sigma_x / Whitmore (-)", "sx_over_bench_std"),
        ("sy_over_bench", "sigma_y / Whitmore (-)", "sy_over_bench_std"),
        ("entrance_over_bench", "Entrance / Whitmore (-)", "entrance_over_bench_std"),
    ]
    for ax, (col, ylab, std_col) in zip(axes.flatten(), panels):
        for energy in energies:
            block = df[df["energy_mev"].astype(int) == energy].sort_values("g4_t_per_m")
            yerr = block[std_col] if std_col in block.columns else None
            plot_metric_series(
                ax,
                block["g4_t_per_m"],
                block[col],
                yerr,
                marker=markers.get(energy, "o"),
                color=colors.get(energy, None),
                linewidth=2.0,
                label=f"{energy} MeV",
            )
        ax.axhline(1.0, color="black", linestyle="--", linewidth=1.0)
        ax.set_xlabel("Q4 Gradient (T/m)")
        ax.set_ylabel(ylab)
        ax.grid(alpha=0.25)
    axes[0, 0].legend(fontsize=9)
    fig.suptitle("Figure 5: Normalized Trend Ratios vs Whitmore", y=0.98)
    caption = (
        "Caption: Normalized trend ratios where 1.0 indicates exact Whitmore agreement. "
        "This view highlights trend similarity independent of absolute scale. "
        "Ratios are TOPAS metric divided by the energy-matched Table 2 Whitmore metric. "
        f"Table 2 Whitmore anchors: {bench_summary}. "
        f"{meta}{uncertainty_note}"
    )
    add_caption(fig, caption)
    path = args.outdir / "fig05_normalized_ratios.png"
    save_figure(fig, path, args.dpi)
    captions.append((path.name, caption))

    # Figure 6: deltas
    fig, axes = plt.subplots(2, 2, figsize=(12.6, 8.6), constrained_layout=False)
    panels = [
        ("delta_z_hat_cm", "Delta z_hat (cm)", "delta_z_hat_cm_std"),
        ("delta_sigma_x_cm", "Delta sigma_x (cm)", "delta_sigma_x_cm_std"),
        ("delta_sigma_y_cm", "Delta sigma_y (cm)", "delta_sigma_y_cm_std"),
        ("delta_entrance_pct", "Delta entrance dose (%)", "delta_entrance_pct_std"),
    ]
    for ax, (col, ylab, std_col) in zip(axes.flatten(), panels):
        for energy in energies:
            block = df[df["energy_mev"].astype(int) == energy].sort_values("g4_t_per_m")
            yerr = block[std_col] if std_col in block.columns else None
            plot_metric_series(
                ax,
                block["g4_t_per_m"],
                block[col],
                yerr,
                marker=markers.get(energy, "o"),
                color=colors.get(energy, None),
                linewidth=2.0,
                label=f"{energy} MeV",
            )
        ax.axhline(0.0, color="black", linestyle="--", linewidth=1.0)
        ax.set_xlabel("Q4 Gradient (T/m)")
        ax.set_ylabel(ylab)
        ax.grid(alpha=0.25)
    axes[0, 0].legend(fontsize=9)
    fig.suptitle("Figure 6: Absolute Deviations from Whitmore", y=0.98)
    caption = (
        "Caption: Signed deviations from Whitmore for depth, sigmas, and entrance dose. "
        "Crossing zero indicates exact Whitmore match for that metric. "
        f"Best observed points: {best_summary}. "
        f"{meta}{uncertainty_note}"
    )
    add_caption(fig, caption)
    path = args.outdir / "fig06_deltas_vs_g4.png"
    save_figure(fig, path, args.dpi)
    captions.append((path.name, caption))

    # Figure 7: Pareto scatter
    fig, ax = plt.subplots(figsize=(10.8, 6.4), constrained_layout=False)
    vmin = float(df["delta_entrance_pct"].min())
    vmax = float(df["delta_entrance_pct"].max())
    for energy in energies:
        block = df[df["energy_mev"].astype(int) == energy]
        sc = ax.scatter(
            block["delta_z_hat_cm"].abs(),
            block["combined_sigma_abs_delta_cm"],
            c=block["delta_entrance_pct"],
            cmap="coolwarm",
            vmin=vmin,
            vmax=vmax,
            marker=markers.get(energy, "o"),
            s=70,
            edgecolor="black",
            linewidth=0.4,
            alpha=0.9,
        )
        xerr = block["delta_z_hat_cm_std"] if "delta_z_hat_cm_std" in block.columns else None
        yerr = None
        if "delta_sigma_x_cm_std" in block.columns and "delta_sigma_y_cm_std" in block.columns:
            yerr = 0.5 * (
                pd.to_numeric(block["delta_sigma_x_cm_std"], errors="coerce").fillna(0.0)
                + pd.to_numeric(block["delta_sigma_y_cm_std"], errors="coerce").fillna(0.0)
            )
        if xerr is not None or yerr is not None:
            ax.errorbar(
                block["delta_z_hat_cm"].abs(),
                block["combined_sigma_abs_delta_cm"],
                xerr=xerr,
                yerr=yerr,
                fmt="none",
                ecolor=colors.get(energy, "#555555"),
                alpha=0.55,
                capsize=3,
                linewidth=1.1,
            )
        best = block.loc[block["weighted_error"].idxmin()]
        ax.annotate(
            f"{energy} MeV best\nQ4={float(best['g4_t_per_m']):.2f} T/m",
            (abs(float(best["delta_z_hat_cm"])), float(best["combined_sigma_abs_delta_cm"])),
            textcoords="offset points",
            xytext=(7, 6),
            fontsize=8,
        )
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("Delta entrance dose (%)")
    ax.set_xlabel("|Delta z_hat| (cm)")
    ax.set_ylabel("0.5*(|Delta sigma_x| + |Delta sigma_y|) (cm)")
    ax.set_title("Figure 7: Multi-Metric Trade-Off (Pareto-Style)")
    handles = [
        Line2D([0], [0], marker=markers.get(e, "o"), color="w", markerfacecolor="#666666", markeredgecolor="black", markersize=7, linestyle="", label=f"{e} MeV")
        for e in energies
    ]
    ax.legend(handles=handles, fontsize=9, loc="upper right")
    ax.grid(alpha=0.25)
    caption = (
        "Caption: Trade-off map across all cases. Lower-left is better for depth and spot-size agreement; "
        "color encodes entrance-dose deviation. Annotated points are the best weighted-error cases per energy. "
        "Weighted error uses normalized deltas: z/0.5 cm, sigma_x/0.2 cm, sigma_y/0.2 cm, entrance/5%. "
        f"Best points: {best_summary}. "
        f"{meta}{uncertainty_note}"
    )
    add_caption(fig, caption)
    path = args.outdir / "fig07_pareto_tradeoff.png"
    save_figure(fig, path, args.dpi)
    captions.append((path.name, caption))

    # Figure 8: best-case focal slice maps + line profiles
    best_rows = []
    for energy in energies:
        block = df[df["energy_mev"].astype(int) == energy]
        best_rows.append(block.loc[block["weighted_error"].idxmin()])
    best_df = pd.DataFrame(best_rows).sort_values("energy_mev")

    fig, axes = plt.subplots(len(best_df), 3, figsize=(15.2, 4.5 * len(best_df)), constrained_layout=False)
    if len(best_df) == 1:
        axes = np.array([axes])  # type: ignore[assignment]

    for r, (_, row) in enumerate(best_df.iterrows()):
        energy = int(row["energy_mev"])
        csv_file = Path(str(row["csv_file"]))
        grid, header = load_topas_grid(csv_file, retries=5, retry_delay_sec=0.5)
        nx, ny, _ = grid.shape
        dx = float(header["dx_cm"])
        dy = float(header["dy_cm"])
        x = (np.arange(nx, dtype=float) + 0.5 - nx / 2.0) * dx
        y = (np.arange(ny, dtype=float) + 0.5 - ny / 2.0) * dy

        z_idx = int(row["peak_index_z_integrated"]) if "peak_index_z_integrated" in row else int(np.argmax(np.sum(grid, axis=(0, 1))))
        sl = grid[:, :, z_idx].astype(float)
        if np.max(sl) > 0:
            sl_norm = 100.0 * sl / float(np.max(sl))
        else:
            sl_norm = sl
        ix_peak, iy_peak = np.unravel_index(int(np.argmax(sl)), sl.shape)
        x_peak = float(x[ix_peak])
        y_peak = float(y[iy_peak])
        x_prof = sl[:, iy_peak]
        y_prof = sl[ix_peak, :]
        if np.max(x_prof) > 0:
            x_prof = 100.0 * x_prof / float(np.max(x_prof))
        if np.max(y_prof) > 0:
            y_prof = 100.0 * y_prof / float(np.max(y_prof))

        bench = table2_metrics(reference, energy)
        sx_b = bench["sigma_x_cm"]
        sy_b = bench["sigma_y_cm"]

        ax0 = axes[r, 0]
        im = ax0.imshow(
            sl_norm.T,
            origin="lower",
            extent=[float(x.min()), float(x.max()), float(y.min()), float(y.max())],
            cmap="viridis",
            aspect="equal",
            vmin=0.0,
            vmax=100.0,
        )
        ax0.scatter([x_peak], [y_peak], s=25, c="white", edgecolors="black", linewidths=0.4)
        ax0.set_xlabel("x (cm)")
        ax0.set_ylabel("y (cm)")
        ax0.set_title(
            (
                f"E={energy} MeV map at z_hat={float(row[args.z_col]):.2f} cm\n"
                f"Best Q4={float(row['g4_t_per_m']):.2f} T/m"
            ),
            fontsize=10,
        )
        cbar = fig.colorbar(im, ax=ax0, fraction=0.046, pad=0.02)
        cbar.set_label("Slice dose / slice max (%)")

        ax1 = axes[r, 1]
        ax1.plot(x, x_prof, color="#ff7f0e", linewidth=2.0, label="x-profile (norm %)")
        ax1.axvline(x_peak - sx_b, color="black", linestyle="--", linewidth=1.0, label="Peak ± Whitmore sigma_x")
        ax1.axvline(x_peak + sx_b, color="black", linestyle="--", linewidth=1.0)
        ax1.set_xlabel("x (cm)")
        ax1.set_ylabel("Normalized dose (%)")
        ax1.set_title(f"E={energy} MeV x-profile at focal slice")
        ax1.grid(alpha=0.25)
        ax1.legend(fontsize=8)

        ax2 = axes[r, 2]
        ax2.plot(y, y_prof, color="#17becf", linewidth=2.0, label="y-profile (norm %)")
        ax2.axvline(y_peak - sy_b, color="black", linestyle="--", linewidth=1.0, label="Peak ± Whitmore sigma_y")
        ax2.axvline(y_peak + sy_b, color="black", linestyle="--", linewidth=1.0)
        ax2.set_xlabel("y (cm)")
        ax2.set_ylabel("Normalized dose (%)")
        ax2.set_title(f"E={energy} MeV y-profile at focal slice")
        ax2.grid(alpha=0.25)
        ax2.legend(fontsize=8)

    fig.suptitle("Figure 8: Best-Case Focal Slice Maps and Profiles", y=0.995)
    caption = (
        "Caption: For each energy, the best weighted-error case is shown with (left) normalized 2D focal-slice map, "
        "(middle) x-profile, and (right) y-profile. Dashed lines indicate ± Whitmore sigma around local profile peak. "
        f"Best observed points: {best_summary}. "
        f"{meta}{uncertainty_note}"
    )
    add_caption(fig, caption)
    path = args.outdir / "fig08_bestcase_focal_maps_profiles.png"
    save_figure(fig, path, args.dpi)
    captions.append((path.name, caption))

    # Caption index file
    caption_file = args.outdir / "figure_captions.md"
    lines = ["# Trend-Consistency Figure Captions", ""]
    for name, text in captions:
        lines.append(f"- `{name}`: {text}")
    caption_file.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Wrote 8 figures to: {args.outdir}")
    for name, _ in captions:
        print(f" - {name}")
    print(f"Caption index: {caption_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
