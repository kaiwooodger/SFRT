#!/usr/bin/env python3
"""Generate the requested 6-figure Whitmore set with seed uncertainty."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception as exc:  # pragma: no cover
    raise SystemExit(
        "matplotlib is required for plotting. Install with: python3 -m pip install matplotlib"
    ) from exc

from analyze_topas_outputs import load_topas_grid


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Build the requested 6 Whitmore figures with seed mean ± SD uncertainty."
    )
    parser.add_argument(
        "--seed-raw-csv",
        type=Path,
        default=root / "runs" / "publishable_subset" / "latest_topas_case_metrics_raw.csv",
        help="Raw per-seed metrics CSV.",
    )
    parser.add_argument(
        "--reference",
        type=Path,
        default=root / "config" / "benchmark_reference.json",
        help="Whitmore reference JSON.",
    )
    parser.add_argument(
        "--sweep-case-metrics",
        type=Path,
        default=root / "runs" / "analysis_paper2d" / "case_metrics.csv",
        help="Prior sweep metrics table (e.g., 300k histories) for multi-point trend plotting.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=root / "runs" / "publishable_subset" / "seed_runs" / "E100_p5p70_seed11" / "manifest.json",
        help="Manifest used for metadata in captions.",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=root / "runs" / "requested_6fig_benchmark_20260319",
        help="Output directory for figures and captions.",
    )
    parser.add_argument(
        "--energies",
        nargs="+",
        type=int,
        default=[100, 200, 250],
        help="Energies to include.",
    )
    parser.add_argument("--dpi", type=int, default=450, help="Figure DPI.")
    return parser.parse_args()


def load_reference(path: Path) -> Dict:
    return json.loads(path.read_text(encoding="utf-8"))


def table2(reference: Dict, energy: int) -> Dict[str, float]:
    m = reference["asymmetric_beamline"]["energies"][str(int(energy))]["benchmark_metrics_table2"]
    return {
        "z_hat_cm": float(m["z_hat_cm"]),
        "sigma_x_cm": float(m["sigma_x_cm"]),
        "sigma_y_cm": float(m["sigma_y_cm"]),
        "entrance_pct": float(m["entrance_dose_pct"]),
    }


def table3(reference: Dict, energy: int) -> pd.DataFrame:
    rows = reference["asymmetric_beamline"]["energies"][str(int(energy))]["supplementary_table3_scan"]
    return pd.DataFrame(rows).astype(float).sort_values("g4_t_per_m")


def manifest_meta(path: Path) -> str:
    if not path.exists():
        return "Histories/case=unknown, threads=unknown."
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        h = data.get("histories", "unknown")
        t = data.get("threads", "unknown")
        if h != "unknown":
            h = f"{int(h):,}"
        return f"Histories/case={h}, threads={t}."
    except Exception:
        return "Histories/case=unknown, threads=unknown."


def norm100(arr: np.ndarray) -> np.ndarray:
    a = np.asarray(arr, dtype=float)
    mx = float(np.max(a)) if a.size else 0.0
    return (100.0 * a / mx) if mx > 0 else np.zeros_like(a)


def weighted_sigma_xy(slice_xy: np.ndarray, x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    s = np.asarray(slice_xy, dtype=float)
    total = float(np.sum(s))
    if s.ndim != 2 or total <= 0:
        return float("nan"), float("nan")
    wx = np.sum(s, axis=1)
    wy = np.sum(s, axis=0)
    mx = float(np.sum(wx * x) / np.sum(wx))
    my = float(np.sum(wy * y) / np.sum(wy))
    sx = float(np.sqrt(np.sum(wx * (x - mx) ** 2) / np.sum(wx)))
    sy = float(np.sqrt(np.sum(wy * (y - my) ** 2) / np.sum(wy)))
    return sx, sy


def nearest_seed_to_mean(block: pd.DataFrame, cols: List[str]) -> int:
    mu = {c: float(block[c].mean()) for c in cols}
    best_seed = int(block.iloc[0]["seed"])
    best_score = float("inf")
    for _, row in block.iterrows():
        score = 0.0
        for c in cols:
            den = abs(mu[c]) if abs(mu[c]) > 1e-9 else 1.0
            score += abs(float(row[c]) - mu[c]) / den
        if score < best_score:
            best_score = score
            best_seed = int(row["seed"])
    return best_seed


def aggregate_by_energy_g4(raw: pd.DataFrame, reference: Dict) -> pd.DataFrame:
    metrics = [
        "z_hat_integrated_cm",
        "sigma_x_integrated_cm",
        "sigma_y_integrated_cm",
        "entrance_on_axis_pct",
        "exit_on_axis_pct",
        "weighted_error",
    ]
    agg = raw.groupby(["energy_mev", "case_id", "g4_t_per_m"], as_index=False).agg(
        **{f"{m}_mean": (m, "mean") for m in metrics},
        **{f"{m}_std": (m, "std") for m in metrics},
        n_seeds=("seed", "count"),
    )
    for col in agg.columns:
        if col.endswith("_std"):
            agg[col] = agg[col].fillna(0.0)

    dz = []
    dz_rel = []
    for _, r in agg.iterrows():
        bench = table2(reference, int(r["energy_mev"]))
        delta = float(r["z_hat_integrated_cm_mean"]) - bench["z_hat_cm"]
        dz.append(delta)
        dz_rel.append(100.0 * delta / bench["z_hat_cm"])
    agg["delta_z_cm_mean"] = dz
    agg["delta_z_pct_mean"] = dz_rel
    agg["delta_z_cm_std"] = agg["z_hat_integrated_cm_std"]
    agg["delta_z_pct_std"] = 100.0 * agg["delta_z_cm_std"] / agg["energy_mev"].astype(int).map(
        lambda e: table2(reference, int(e))["z_hat_cm"]
    )
    return agg.sort_values(["energy_mev", "g4_t_per_m"]).reset_index(drop=True)


def select_best_per_energy(agg: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for energy, block in agg.groupby(agg["energy_mev"].astype(int)):
        row = block.loc[block["delta_z_cm_mean"].abs().idxmin()]
        rows.append(row)
    return pd.DataFrame(rows).sort_values("energy_mev").reset_index(drop=True)


def build_profile_data(raw: pd.DataFrame, selected: pd.DataFrame) -> Dict[int, Dict[str, object]]:
    out: Dict[int, Dict[str, object]] = {}
    for _, sel in selected.iterrows():
        energy = int(sel["energy_mev"])
        g4 = float(sel["g4_t_per_m"])
        cid = str(sel["case_id"])
        block = raw[
            (raw["energy_mev"].astype(int) == energy)
            & (raw["case_id"].astype(str) == cid)
            & (np.isclose(raw["g4_t_per_m"].astype(float), g4))
        ].copy()
        if block.empty:
            continue

        seeds = []
        z_ref = None
        sx_curves = []
        sy_curves = []
        entry_by_seed: Dict[int, Dict[str, object]] = {}

        for _, row in block.sort_values("seed").iterrows():
            seed = int(row["seed"])
            seeds.append(seed)
            grid, header = load_topas_grid(Path(str(row["csv_file"])), retries=8, retry_delay_sec=0.7)
            nx, ny, nz = grid.shape
            dx = float(header["dx_cm"])
            dy = float(header["dy_cm"])
            dz = float(header["dz_cm"])
            x = (np.arange(nx, dtype=float) + 0.5 - nx / 2.0) * dx
            y = (np.arange(ny, dtype=float) + 0.5 - ny / 2.0) * dy
            z = (np.arange(nz, dtype=float) + 0.5) * dz
            if z_ref is None:
                z_ref = z
            else:
                if len(z_ref) != len(z) or not np.allclose(z_ref, z):
                    raise RuntimeError(f"Inconsistent z grid across seeds for E={energy}, g4={g4}")

            sx = []
            sy = []
            for iz in range(nz):
                sx_i, sy_i = weighted_sigma_xy(grid[:, :, iz], x, y)
                sx.append(sx_i)
                sy.append(sy_i)
            sx = np.asarray(sx, dtype=float)
            sy = np.asarray(sy, dtype=float)
            sx_curves.append(sx)
            sy_curves.append(sy)

            integrated = np.sum(grid, axis=(0, 1))
            peak_idx = int(np.argmax(integrated))
            cx, cy = nx // 2, ny // 2
            entry_by_seed[seed] = {
                "grid": grid,
                "x": x,
                "y": y,
                "z": z,
                "peak_idx": peak_idx,
                "on_axis_norm": norm100(grid[cx, cy, :]),
            }

        sx_stack = np.vstack(sx_curves)
        sy_stack = np.vstack(sy_curves)
        out[energy] = {
            "g4": g4,
            "case_id": cid,
            "seeds": seeds,
            "z": z_ref,
            "sx_mean": np.mean(sx_stack, axis=0),
            "sx_std": np.std(sx_stack, axis=0, ddof=1) if len(seeds) > 1 else np.zeros_like(z_ref),
            "sy_mean": np.mean(sy_stack, axis=0),
            "sy_std": np.std(sy_stack, axis=0, ddof=1) if len(seeds) > 1 else np.zeros_like(z_ref),
            "entry_by_seed": entry_by_seed,
            "raw_block": block,
        }
    return out


def sample_block_evenly(block: pd.DataFrame, max_points: int = 5) -> pd.DataFrame:
    """Return up to max_points spanning the full Q4 range."""
    if len(block) <= max_points:
        return block
    idx = np.linspace(0, len(block) - 1, max_points)
    idx = sorted(set(int(round(i)) for i in idx))
    return block.iloc[idx].copy()


def main() -> int:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    reference = load_reference(args.reference)
    raw = pd.read_csv(args.seed_raw_csv)
    raw = raw[raw["energy_mev"].astype(int).isin(args.energies)].copy()
    if raw.empty:
        raise RuntimeError("No rows found in raw seed CSV for requested energies.")
    if not args.sweep_case_metrics.exists():
        raise FileNotFoundError(f"Sweep case_metrics.csv not found: {args.sweep_case_metrics}")
    sweep_df = pd.read_csv(args.sweep_case_metrics)
    sweep_df = sweep_df[sweep_df["energy_mev"].astype(int).isin(args.energies)].copy()
    if sweep_df.empty:
        raise RuntimeError("No rows found in sweep case_metrics.csv for requested energies.")
    if "z_hat_integrated_cm" not in sweep_df.columns:
        raise ValueError("Sweep case_metrics.csv is missing required column: z_hat_integrated_cm")

    agg = aggregate_by_energy_g4(raw, reference)
    selected = select_best_per_energy(agg)
    profiles = build_profile_data(raw, selected)

    colors = {100: "#1f77b4", 200: "#2ca02c", 250: "#d62728"}
    energies = sorted(selected["energy_mev"].astype(int).unique().tolist())
    meta = manifest_meta(args.manifest)
    seeds = sorted(raw["seed"].dropna().astype(int).unique().tolist()) if "seed" in raw.columns else []
    seed_note = f" Seeds: {', '.join(str(s) for s in seeds)}." if seeds else ""

    captions: List[Tuple[str, str]] = []

    # Figure 1: focal depth vs control parameter.
    fig, ax = plt.subplots(figsize=(10.4, 6.2), constrained_layout=False)
    for energy in energies:
        block = agg[agg["energy_mev"].astype(int) == energy].sort_values("g4_t_per_m")
        t3 = table3(reference, energy)
        ax.errorbar(
            block["g4_t_per_m"],
            block["z_hat_integrated_cm_mean"],
            yerr=block["z_hat_integrated_cm_std"],
            marker="o",
            linewidth=2.0,
            capsize=4,
            color=colors.get(energy, None),
            label=f"Model {energy} MeV mean ± SD",
        )
        ax.plot(
            t3["g4_t_per_m"],
            t3["z_hat_cm"],
            linestyle="--",
            linewidth=1.8,
            color=colors.get(energy, None),
            alpha=0.85,
            label=f"Whitmore {energy} MeV",
        )
    ax.set_xlabel("Q4 gradient (T/m)")
    ax.set_ylabel(r"Focal Depth $z_f$ (cm)")
    ax.set_title("Figure 1: Focal Depth vs Q4 Control")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8.4, ncol=2)
    cap = (
        "Caption: Primary Whitmore figure. Focal-depth response ($z_f$) to Q4 compared against Whitmore Table 3. "
        "Error bars represent seed-to-seed variability (SD). "
        f"{meta}{seed_note}"
    )
    fig.text(0.01, 0.01, cap, ha="left", va="bottom", fontsize=9, wrap=True)
    fig.subplots_adjust(bottom=0.16)
    out = args.outdir / "fig01_focal_depth_vs_q4.png"
    fig.savefig(out, dpi=args.dpi)
    plt.close(fig)
    captions.append((out.name, cap))

    # Supplementary legacy figure (cleaned): dose-ratio mismatch with external legend and no N/A labels.
    fig, axes = plt.subplots(1, len(energies), figsize=(5.2 * len(energies), 4.8), constrained_layout=False)
    if len(energies) == 1:
        axes = np.array([axes])
    categories = ["Entrance", "Peak", "Exit"]
    xpos = np.arange(3, dtype=float)
    for ax, energy in zip(axes, energies):
        row = selected[selected["energy_mev"].astype(int) == energy].iloc[0]
        bench = table2(reference, energy)
        model_vals = [
            float(row["entrance_on_axis_pct_mean"]),
            100.0,
            float(row["exit_on_axis_pct_mean"]),
        ]
        model_err = [
            float(row["entrance_on_axis_pct_std"]),
            0.0,
            float(row["exit_on_axis_pct_std"]),
        ]
        ax.bar(
            xpos - 0.12,
            model_vals,
            yerr=model_err,
            width=0.24,
            capsize=4,
            color=colors.get(energy, "#1f77b4"),
            label="Model mean ± SD",
        )
        # Whitmore data is available for entrance and peak only.
        ax.scatter(
            [xpos[0] + 0.14, xpos[1] + 0.14],
            [bench["entrance_pct"], 100.0],
            marker="*",
            s=140,
            color="#d62728",
            edgecolor="black",
            linewidth=0.5,
            label="Whitmore (available terms)",
        )
        ax.set_xticks(xpos, categories)
        ax.set_ylim(0, 110)
        ax.set_ylabel("Dose / peak dose (%)")
        ax.set_title(f"E={energy} MeV")
        ax.grid(axis="y", alpha=0.25)
        # Remove per-axis legends; use one external figure legend.
        if ax.get_legend() is not None:
            ax.get_legend().remove()

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.01),
        ncol=2,
        fontsize=9,
        frameon=False,
    )
    fig.suptitle("Supplementary: Dose-Ratio Comparison (Entrance/Peak/Exit)", y=0.98)
    fig.subplots_adjust(bottom=0.2)
    out_clean = args.outdir / "fig06_dose_ratio_mismatch.png"
    fig.savefig(out_clean, dpi=args.dpi)
    plt.close(fig)

    # Figure 2: focal depth error summary.
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.8), constrained_layout=False)
    ax_abs, ax_rel = axes
    x = np.arange(len(energies), dtype=float)
    s = selected.sort_values("energy_mev").reset_index(drop=True)
    ax_abs.errorbar(
        x,
        s["delta_z_cm_mean"],
        yerr=s["delta_z_cm_std"],
        fmt="o",
        capsize=5,
        linewidth=1.8,
        color="black",
    )
    ax_abs.axhline(0.0, color="gray", linestyle="--", linewidth=1.0)
    ax_abs.set_xticks(x, [str(e) for e in s["energy_mev"].astype(int)])
    ax_abs.set_xlabel("Energy (MeV)")
    ax_abs.set_ylabel(r"$\Delta z_f$ (cm)")
    ax_abs.set_title("Absolute Error")
    ax_abs.grid(axis="y", alpha=0.25)

    ax_rel.errorbar(
        x,
        s["delta_z_pct_mean"],
        yerr=s["delta_z_pct_std"],
        fmt="o",
        capsize=5,
        linewidth=1.8,
        color="#1f77b4",
    )
    ax_rel.axhline(0.0, color="gray", linestyle="--", linewidth=1.0)
    ax_rel.set_xticks(x, [str(e) for e in s["energy_mev"].astype(int)])
    ax_rel.set_xlabel("Energy (MeV)")
    ax_rel.set_ylabel(r"Relative $\Delta z_f$ (%)")
    ax_rel.set_title("Relative Error")
    ax_rel.grid(axis="y", alpha=0.25)
    fig.suptitle("Figure 2: Focal-Depth Error Summary", y=0.98)
    cap = (
        "Caption: Compact focal-depth agreement summary across energies. "
        "Points are model mean errors and bars are seed SD."
    )
    fig.text(0.01, 0.01, cap, ha="left", va="bottom", fontsize=9, wrap=True)
    fig.subplots_adjust(bottom=0.2)
    out = args.outdir / "fig02_focal_depth_error_summary.png"
    fig.savefig(out, dpi=args.dpi)
    plt.close(fig)
    captions.append((out.name, cap))

    # Figure 3: seed repeatability for focal depth.
    fig, ax = plt.subplots(figsize=(8.8, 5.0), constrained_layout=False)
    x = np.arange(len(energies), dtype=float)
    for i, energy in enumerate(energies):
        b = profiles[energy]["raw_block"]
        vals = b["z_hat_integrated_cm"].astype(float).to_numpy()
        jitter = np.linspace(-0.08, 0.08, len(vals))
        ax.scatter(np.full_like(vals, x[i]) + jitter, vals, s=48, color=colors.get(energy, "#1f77b4"), edgecolor="black", linewidth=0.4, label=f"{energy} MeV seeds" if i == 0 else None)
        mu = float(np.mean(vals))
        sd = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
        cv = (100.0 * sd / mu) if abs(mu) > 1e-9 else float("nan")
        ax.errorbar([x[i]], [mu], yerr=[sd], marker="D", color="black", capsize=5, linewidth=1.3, label="Mean ± SD" if i == 0 else None)
        ax.text(x[i], mu + sd + 0.1, f"CV={cv:.2f}%", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x, [str(e) for e in energies])
    ax.set_xlabel("Energy (MeV)")
    ax.set_ylabel(r"Focal Depth $z_f$ (cm)")
    ax.set_title("Figure 3: Seed Repeatability of Focal Depth")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=8.4, loc="upper left")
    cap = (
        "Caption: Seed-level repeatability plot for focal depth. "
        "Low spread indicates mismatch is systematic rather than Monte Carlo seed noise."
    )
    fig.text(0.01, 0.01, cap, ha="left", va="bottom", fontsize=9, wrap=True)
    fig.subplots_adjust(bottom=0.2)
    out = args.outdir / "fig03_seed_repeatability_focal_depth.png"
    fig.savefig(out, dpi=args.dpi)
    plt.close(fig)
    captions.append((out.name, cap))

    # Figure 4: representative lateral profiles entrance/focus/exit.
    fig, axes = plt.subplots(len(energies), 3, figsize=(14.2, 4.3 * len(energies)), constrained_layout=False)
    if len(energies) == 1:
        axes = np.array([axes])
    depth_titles = ["Entrance", "Near Focus", "Exit"]
    for r, energy in enumerate(energies):
        block = profiles[energy]["raw_block"].copy()
        seed = nearest_seed_to_mean(
            block,
            ["z_hat_integrated_cm", "sigma_x_integrated_cm", "sigma_y_integrated_cm", "entrance_on_axis_pct"],
        )
        entry = profiles[energy]["entry_by_seed"][seed]
        grid = entry["grid"]
        xaxis = entry["x"]
        yaxis = entry["y"]
        pidx = int(entry["peak_idx"])
        idxs = [0, pidx, grid.shape[2] - 1]
        for c, iz in enumerate(idxs):
            sl = grid[:, :, iz]
            cx = len(xaxis) // 2
            cy = len(yaxis) // 2
            px = norm100(sl[:, cy])
            py = norm100(sl[cx, :])
            ax = axes[r, c]
            ax.plot(xaxis, px, color="#1f77b4", linewidth=1.8, label=r"$x$ profile")
            ax.plot(yaxis, py, color="#ff7f0e", linestyle="--", linewidth=1.8, label=r"$y$ profile")
            ax.set_title(f"E={energy} MeV | {depth_titles[c]} | z={float(entry['z'][iz]):.2f} cm | seed {seed}", fontsize=9)
            ax.set_xlabel("Position (cm)")
            ax.set_ylabel("Normalized dose (%)")
            ax.set_ylim(0, 105)
            ax.grid(alpha=0.25)
            if r == 0 and c == 0:
                ax.legend(fontsize=8, loc="upper right")
    fig.suptitle("Figure 4: Representative Lateral Profiles (Entrance, Focus, Exit)", y=0.995)
    cap = "Caption: Physical-context figure showing focused beam-shape evolution at entrance, near-focus, and exit."
    fig.text(0.01, 0.01, cap, ha="left", va="bottom", fontsize=9, wrap=True)
    fig.subplots_adjust(bottom=0.08)
    out = args.outdir / "fig04_representative_lateral_profiles.png"
    fig.savefig(out, dpi=args.dpi)
    plt.close(fig)
    captions.append((out.name, cap))

    # Figure 5: spot-size mismatch vs depth.
    fig, axes = plt.subplots(1, len(energies), figsize=(5.2 * len(energies), 4.8), constrained_layout=False)
    if len(energies) == 1:
        axes = np.array([axes])
    for ax, energy in zip(axes, energies):
        z = profiles[energy]["z"]
        sx_mu = profiles[energy]["sx_mean"]
        sx_sd = profiles[energy]["sx_std"]
        sy_mu = profiles[energy]["sy_mean"]
        sy_sd = profiles[energy]["sy_std"]
        bench = table2(reference, energy)

        ax.plot(z, sx_mu, color="#ff7f0e", linewidth=2.0, label=r"Model $\sigma_x$ mean")
        ax.fill_between(z, sx_mu - sx_sd, sx_mu + sx_sd, color="#ff7f0e", alpha=0.18)
        ax.plot(z, sy_mu, color="#17becf", linewidth=2.0, label=r"Model $\sigma_y$ mean")
        ax.fill_between(z, sy_mu - sy_sd, sy_mu + sy_sd, color="#17becf", alpha=0.18)
        ax.axhline(bench["sigma_x_cm"], color="#ff7f0e", linestyle="--", linewidth=1.1, label=r"Whitmore $\sigma_x$")
        ax.axhline(bench["sigma_y_cm"], color="#17becf", linestyle="--", linewidth=1.1, label=r"Whitmore $\sigma_y$")
        ax.axvline(bench["z_hat_cm"], color="#d62728", linestyle=":", linewidth=1.1, label=r"Whitmore $z_f$")
        ax.set_title(f"E={energy} MeV")
        ax.set_xlabel("Depth in water (cm)")
        ax.set_ylabel(r"Spot Size $\sigma$ (cm)")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=7.2, loc="upper left")
    fig.suptitle("Figure 5: Spot-Size Mismatch vs Depth", y=0.98)
    cap = (
        "Caption: Boundary-condition figure: while focal placement trends can agree, "
        "transverse spot sizes remain larger than Whitmore across depth."
    )
    fig.text(0.01, 0.01, cap, ha="left", va="bottom", fontsize=9, wrap=True)
    fig.subplots_adjust(bottom=0.2)
    out = args.outdir / "fig05_spot_size_mismatch_vs_depth.png"
    fig.savefig(out, dpi=args.dpi)
    plt.close(fig)
    captions.append((out.name, cap))

    # Figure 6: stronger trend figure using prior 300k sweep (multi-point per energy).
    fig, ax = plt.subplots(figsize=(10.8, 6.2), constrained_layout=False)
    for energy in energies:
        # 300k sweep points (multi-point trend line).
        block_all = sweep_df[sweep_df["energy_mev"].astype(int) == energy].sort_values("g4_t_per_m")
        block = sample_block_evenly(block_all, max_points=5)
        ax.plot(
            block["g4_t_per_m"],
            block["z_hat_integrated_cm"],
            marker="o",
            linewidth=2.0,
            color=colors.get(energy, None),
            label=f"Model {energy} MeV (300k sweep)",
        )

        # Whitmore Table 3 curve.
        t3 = table3(reference, energy)
        ax.plot(
            t3["g4_t_per_m"],
            t3["z_hat_cm"],
            linestyle="--",
            linewidth=1.7,
            color=colors.get(energy, None),
            alpha=0.9,
            label=f"Whitmore {energy} MeV",
        )

        # Overlay high-history replicated point with uncertainty.
        row = selected[selected["energy_mev"].astype(int) == energy].iloc[0]
        ax.errorbar(
            [float(row["g4_t_per_m"])],
            [float(row["z_hat_integrated_cm_mean"])],
            yerr=[float(row["z_hat_integrated_cm_std"])],
            fmt="D",
            color=colors.get(energy, None),
            markeredgecolor="black",
            markeredgewidth=0.5,
            capsize=4,
            linewidth=1.4,
            label=f"Model {energy} MeV (1M mean ± SD)",
        )

    ax.set_xlabel("Q4 Gradient (T/m)")
    ax.set_ylabel(r"Focal Depth $z_f$ (cm)")
    ax.set_title("Figure 6: Trend-Reproduction Focal Depth vs Q4 (300k Sweep + 1M Replicate Point)")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8.2, ncol=2)
    cap = (
        "Caption: Strong trend-reproduction panel using prior 300,000-history sweeps (3-5 points/energy shown) "
        "overlaid with Whitmore Table 3 curves. Diamond markers show the 1,000,000-history replicated points "
        "with seed SD error bars."
    )
    fig.text(0.01, 0.01, cap, ha="left", va="bottom", fontsize=9, wrap=True)
    fig.subplots_adjust(bottom=0.16)
    out = args.outdir / "fig06_focal_depth_trend_reproduction.png"
    fig.savefig(out, dpi=args.dpi)
    plt.close(fig)
    captions.append((out.name, cap))

    # Save supporting tables and captions.
    agg_file = args.outdir / "aggregated_seed_stats_by_q4.csv"
    agg.to_csv(agg_file, index=False)
    sel_file = args.outdir / "selected_best_focal_depth_cases.csv"
    selected.to_csv(sel_file, index=False)

    # Compact focal-depth error table to support Figure 6 interpretation.
    depth_rows: List[Dict[str, float | int]] = []
    for _, row in selected.sort_values("energy_mev").iterrows():
        energy = int(row["energy_mev"])
        bench_z = table2(reference, energy)["z_hat_cm"]
        model_mu = float(row["z_hat_integrated_cm_mean"])
        model_sd = float(row["z_hat_integrated_cm_std"])
        abs_err = float(model_mu - bench_z)
        rel_err = 100.0 * abs_err / bench_z
        depth_rows.append(
            {
                "Energy (MeV)": energy,
                "Whitmore focal depth (cm)": bench_z,
                "Model focal depth mean (cm)": model_mu,
                "Model focal depth SD (cm)": model_sd,
                "Absolute error (cm)": abs_err,
                "Relative error (%)": rel_err,
            }
        )
    depth_table = pd.DataFrame(depth_rows).sort_values("Energy (MeV)").reset_index(drop=True)
    depth_table_csv = args.outdir / "focal_depth_error_table.csv"
    depth_table.to_csv(depth_table_csv, index=False)

    depth_table_md = args.outdir / "focal_depth_error_table.md"
    md_lines = [
        "# Focal-Depth Error Table",
        "",
        "| Energy (MeV) | Whitmore focal depth (cm) | Model focal depth mean (cm) | SD (cm) | Absolute error (cm) | Relative error (%) |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for _, r in depth_table.iterrows():
        md_lines.append(
            "| "
            f"{int(r['Energy (MeV)'])} | "
            f"{float(r['Whitmore focal depth (cm)']):.3f} | "
            f"{float(r['Model focal depth mean (cm)']):.3f} | "
            f"{float(r['Model focal depth SD (cm)']):.3f} | "
            f"{float(r['Absolute error (cm)']):+.3f} | "
            f"{float(r['Relative error (%)']):+.2f} |"
        )
    depth_table_md.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    cap_file = args.outdir / "figure_captions.md"
    lines = ["# Requested 6-Figure Whitmore Set Captions", ""]
    for name, text in captions:
        lines.append(f"- `{name}`: {text}")
    cap_file.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Wrote requested 6-figure set to: {args.outdir}")
    print(f"Aggregated stats: {agg_file}")
    print(f"Selected cases: {sel_file}")
    print(f"Focal-depth error table (CSV): {depth_table_csv}")
    print(f"Focal-depth error table (MD): {depth_table_md}")
    print(f"Captions: {cap_file}")
    for name, _ in captions:
        print(f" - {name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
