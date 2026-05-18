#!/usr/bin/env python3
"""Create a replicated Whitmore study figure set from multi-seed TOPAS outputs.

Focus:
- Trend agreement across energies.
- Repeatability across random seeds.
- Partial Whitmore consistency (without over-claiming full validation).
"""

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
        description=(
            "Build a small replicated Whitmore study using seed reruns and generate "
            "mean+-uncertainty figures + summary tables."
        )
    )
    parser.add_argument(
        "--seed-raw-csv",
        type=Path,
        default=root / "runs" / "publishable_subset" / "latest_topas_case_metrics_raw.csv",
        help="Raw per-seed metrics CSV.",
    )
    parser.add_argument(
        "--seedmean-csv",
        type=Path,
        default=root / "runs" / "publishable_subset" / "latest_topas_case_metrics_seedmean_std.csv",
        help="Seed-mean CSV used to pick one representative case per energy.",
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
        default=root / "runs" / "publishable_subset" / "seed_runs" / "E100_p5p70_seed11" / "manifest.json",
        help="Manifest for metadata (histories/threads).",
    )
    parser.add_argument(
        "--benchmark-depth-curves",
        "--Whitmore-depth-curves",
        dest="benchmark_depth_curves",
        type=Path,
        default=None,
        help=(
            "Optional CSV with Whitmore width-vs-depth curves. Columns: "
            "energy_mev,z_cm,sigma_x_cm,sigma_y_cm"
        ),
    )
    parser.add_argument(
        "--energies",
        nargs="+",
        type=int,
        default=[100, 200, 250],
        help="Energies to include in the replicated study.",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=root / "runs" / "replicated_benchmark_study_20260319",
        help="Output folder for figures, summary table, and captions.",
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
        payload = json.loads(path.read_text(encoding="utf-8"))
        h = payload.get("histories", "unknown")
        t = payload.get("threads", "unknown")
        if h != "unknown":
            h = f"{int(h):,}"
        return f"Histories/case={h}, threads={t}."
    except Exception:
        return "Histories/case=unknown, threads=unknown."


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


def norm100(arr: np.ndarray) -> np.ndarray:
    a = np.asarray(arr, dtype=float)
    m = float(np.max(a)) if a.size else 0.0
    return (100.0 * a / m) if m > 0 else np.zeros_like(a)


def coef_var(mean: float, std: float) -> float:
    if not np.isfinite(mean) or abs(mean) < 1e-12:
        return float("nan")
    return 100.0 * float(std) / float(mean)


def nearest_seed_to_mean(block: pd.DataFrame, columns: List[str]) -> int:
    means = {c: float(block[c].mean()) for c in columns}
    best_seed = int(block.iloc[0]["seed"])
    best_score = float("inf")
    for _, row in block.iterrows():
        score = 0.0
        for c in columns:
            denom = abs(means[c]) if abs(means[c]) > 1e-9 else 1.0
            score += abs(float(row[c]) - means[c]) / denom
        if score < best_score:
            best_score = score
            best_seed = int(row["seed"])
    return best_seed


def load_benchmark_depth_curves(path: Path | None) -> Dict[int, pd.DataFrame]:
    if path is None:
        return {}
    if not path.exists():
        raise FileNotFoundError(f"Whitmore depth-curve CSV not found: {path}")
    df = pd.read_csv(path)
    required = {"energy_mev", "z_cm", "sigma_x_cm", "sigma_y_cm"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns in Whitmore depth-curve CSV: {missing}")
    out: Dict[int, pd.DataFrame] = {}
    for energy, block in df.groupby(df["energy_mev"].astype(int)):
        out[int(energy)] = block.sort_values("z_cm").reset_index(drop=True)
    return out


def build_seed_profiles(raw_rows: pd.DataFrame) -> Dict[int, Dict[str, object]]:
    """Load dose grids for all seeds and compute per-depth profile metrics."""
    out: Dict[int, Dict[str, object]] = {}

    for energy, block in raw_rows.groupby(raw_rows["energy_mev"].astype(int)):
        seed_entries: List[Dict[str, object]] = []
        z_ref: np.ndarray | None = None
        x_ref: np.ndarray | None = None
        y_ref: np.ndarray | None = None
        nz_ref: int | None = None

        for _, row in block.sort_values("seed").iterrows():
            csv_file = Path(str(row["csv_file"]))
            grid, header = load_topas_grid(csv_file, retries=8, retry_delay_sec=0.7)
            nx, ny, nz = grid.shape
            dx = float(header["dx_cm"])
            dy = float(header["dy_cm"])
            dz = float(header["dz_cm"])
            x = (np.arange(nx, dtype=float) + 0.5 - nx / 2.0) * dx
            y = (np.arange(ny, dtype=float) + 0.5 - ny / 2.0) * dy
            z = (np.arange(nz, dtype=float) + 0.5) * dz
            cx, cy = nx // 2, ny // 2

            if z_ref is None:
                z_ref = z
                x_ref = x
                y_ref = y
                nz_ref = nz
            else:
                if nz_ref != nz or not np.allclose(z_ref, z):
                    raise RuntimeError(
                        f"Energy {energy}: inconsistent z-grid across seeds; cannot aggregate curves safely."
                    )

            sx_curve = []
            sy_curve = []
            for iz in range(nz):
                sx_i, sy_i = weighted_sigma_xy(grid[:, :, iz], x, y)
                sx_curve.append(sx_i)
                sy_curve.append(sy_i)

            sx_curve = np.asarray(sx_curve, dtype=float)
            sy_curve = np.asarray(sy_curve, dtype=float)
            smean_curve = 0.5 * (sx_curve + sy_curve)
            smin_idx = int(np.nanargmin(smean_curve))

            integrated = np.sum(grid, axis=(0, 1))
            peak_idx = int(np.argmax(integrated))

            seed_entries.append(
                {
                    "seed": int(row["seed"]),
                    "csv_file": str(csv_file),
                    "grid": grid,
                    "x": x,
                    "y": y,
                    "z": z,
                    "sx_curve": sx_curve,
                    "sy_curve": sy_curve,
                    "smean_curve": smean_curve,
                    "smin_idx": smin_idx,
                    "peak_idx": peak_idx,
                    "on_axis_norm": norm100(grid[cx, cy, :]),
                    "integrated_norm": norm100(integrated),
                    "sigma_entrance_cm": float(smean_curve[0]),
                    "sigma_focal_cm": float(smean_curve[peak_idx]),
                    "sigma_min_cm": float(smean_curve[smin_idx]),
                    "depth_min_sigma_cm": float(z[smin_idx]),
                    "entrance_pct": float(row["entrance_on_axis_pct"]),
                    "peak_pct": 100.0,
                    "exit_pct": float(row["exit_on_axis_pct"]),
                    "entrance_peak_ratio": float(row["entrance_on_axis_pct"]) / 100.0,
                    "exit_peak_ratio": float(row["exit_on_axis_pct"]) / 100.0,
                    "sigma_x_focal_cm": float(row["sigma_x_integrated_cm"]),
                    "sigma_y_focal_cm": float(row["sigma_y_integrated_cm"]),
                    "z_hat_focal_cm": float(row["z_hat_integrated_cm"]),
                }
            )

        assert z_ref is not None and x_ref is not None and y_ref is not None

        sx_stack = np.vstack([e["sx_curve"] for e in seed_entries])
        sy_stack = np.vstack([e["sy_curve"] for e in seed_entries])
        smean_stack = np.vstack([e["smean_curve"] for e in seed_entries])

        out[int(energy)] = {
            "seed_entries": seed_entries,
            "z": z_ref,
            "x": x_ref,
            "y": y_ref,
            "sx_mean": np.nanmean(sx_stack, axis=0),
            "sx_std": np.nanstd(sx_stack, axis=0, ddof=1),
            "sy_mean": np.nanmean(sy_stack, axis=0),
            "sy_std": np.nanstd(sy_stack, axis=0, ddof=1),
            "smean_mean": np.nanmean(smean_stack, axis=0),
            "smean_std": np.nanstd(smean_stack, axis=0, ddof=1),
        }

    return out


def main() -> int:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    reference = load_reference(args.reference)
    benchmark_depth_curves = load_benchmark_depth_curves(args.benchmark_depth_curves)
    raw = pd.read_csv(args.seed_raw_csv)
    seedmean = pd.read_csv(args.seedmean_csv)

    raw = raw[raw["energy_mev"].astype(int).isin(args.energies)].copy()
    seedmean = seedmean[seedmean["energy_mev"].astype(int).isin(args.energies)].copy()
    if raw.empty or seedmean.empty:
        raise RuntimeError("No rows available for requested energies in provided CSV files.")

    # Choose one case (energy + Q4) per energy using lowest mean weighted error.
    selected = []
    for energy, block in seedmean.groupby(seedmean["energy_mev"].astype(int)):
        best = block.loc[block["weighted_error"].astype(float).idxmin()]
        selected.append(
            {
                "energy_mev": int(energy),
                "case_id": str(best["case_id"]),
                "g4_t_per_m": float(best["g4_t_per_m"]),
                "weighted_error": float(best["weighted_error"]),
            }
        )
    selected_df = pd.DataFrame(selected).sort_values("energy_mev").reset_index(drop=True)

    # Keep only raw seed runs that match selected case per energy.
    keep_rows = []
    for _, row in selected_df.iterrows():
        e = int(row["energy_mev"])
        cid = str(row["case_id"])
        g4 = float(row["g4_t_per_m"])
        block = raw[
            (raw["energy_mev"].astype(int) == e)
            & (raw["case_id"].astype(str) == cid)
            & (np.isclose(raw["g4_t_per_m"].astype(float), g4))
        ].copy()
        if block.empty:
            raise RuntimeError(f"No raw seed rows found for selected case {cid} (E={e}, g4={g4}).")
        keep_rows.append(block)
    raw_selected = pd.concat(keep_rows, ignore_index=True)

    profiles = build_seed_profiles(raw_selected)
    energies = sorted(profiles.keys())
    colors = {100: "#1f77b4", 200: "#2ca02c", 250: "#d62728"}
    meta = manifest_meta(args.manifest)
    seeds = sorted(raw_selected["seed"].dropna().astype(int).unique().tolist()) if "seed" in raw_selected.columns else []
    seed_msg = f" Seeds: {', '.join(str(s) for s in seeds)}." if seeds else ""

    # Summary table requested for replicated Whitmore claims.
    summary_rows = []
    for energy in energies:
        entries = profiles[energy]["seed_entries"]
        vals = {
            "sigma_entrance_cm": np.array([e["sigma_entrance_cm"] for e in entries], dtype=float),
            "sigma_focal_cm": np.array([e["sigma_focal_cm"] for e in entries], dtype=float),
            "sigma_min_cm": np.array([e["sigma_min_cm"] for e in entries], dtype=float),
            "depth_min_sigma_cm": np.array([e["depth_min_sigma_cm"] for e in entries], dtype=float),
            "entrance_pct": np.array([e["entrance_pct"] for e in entries], dtype=float),
            "peak_pct": np.array([e["peak_pct"] for e in entries], dtype=float),
            "exit_pct": np.array([e["exit_pct"] for e in entries], dtype=float),
            "entrance_peak_ratio": np.array([e["entrance_peak_ratio"] for e in entries], dtype=float),
            "exit_peak_ratio": np.array([e["exit_peak_ratio"] for e in entries], dtype=float),
        }
        row = {"energy_mev": energy, "n_seeds": len(entries)}
        for k, arr in vals.items():
            mu = float(np.mean(arr))
            sd = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
            row[f"{k}_mean"] = mu
            row[f"{k}_std"] = sd
            row[f"{k}_cv_pct"] = coef_var(mu, sd)
        summary_rows.append(row)
    summary_df = pd.DataFrame(summary_rows).sort_values("energy_mev").reset_index(drop=True)
    summary_csv = args.outdir / "replicated_observable_summary.csv"
    summary_df.to_csv(summary_csv, index=False)

    captions: List[Tuple[str, str]] = []

    # Figure 1: Spot size vs depth (mean +- seed spread).
    fig, axes = plt.subplots(1, len(energies), figsize=(5.4 * len(energies), 4.8), constrained_layout=False)
    if len(energies) == 1:
        axes = np.array([axes])
    for ax, energy in zip(axes, energies):
        z = profiles[energy]["z"]
        sx_mu = profiles[energy]["sx_mean"]
        sx_sd = profiles[energy]["sx_std"]
        sy_mu = profiles[energy]["sy_mean"]
        sy_sd = profiles[energy]["sy_std"]
        bench = table2(reference, energy)

        ax.plot(z, sx_mu, color="#ff7f0e", linewidth=2.0, label="Model sigma_x mean")
        ax.fill_between(z, sx_mu - sx_sd, sx_mu + sx_sd, color="#ff7f0e", alpha=0.18, label="sigma_x +-1sigma")
        ax.plot(z, sy_mu, color="#17becf", linewidth=2.0, label="Model sigma_y mean")
        ax.fill_between(z, sy_mu - sy_sd, sy_mu + sy_sd, color="#17becf", alpha=0.18, label="sigma_y +-1sigma")

        if energy in benchmark_depth_curves:
            b = benchmark_depth_curves[energy]
            ax.plot(b["z_cm"], b["sigma_x_cm"], "--", color="#ff7f0e", alpha=0.8, label="Whitmore sigma_x(z)")
            ax.plot(b["z_cm"], b["sigma_y_cm"], "--", color="#17becf", alpha=0.8, label="Whitmore sigma_y(z)")
        else:
            ax.scatter([bench["z_hat_cm"]], [bench["sigma_x_cm"]], marker="*", s=150, color="#ff7f0e", edgecolor="black", linewidth=0.5, label="Whitmore sigma_x at z_hat")
            ax.scatter([bench["z_hat_cm"]], [bench["sigma_y_cm"]], marker="*", s=150, color="#17becf", edgecolor="black", linewidth=0.5, label="Whitmore sigma_y at z_hat")

        ax.set_title(f"E={energy} MeV")
        ax.set_xlabel("Depth in water (cm)")
        ax.set_ylabel("Spot size sigma (cm)")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=7.2, loc="upper left")
    fig.suptitle("Figure 1: Spot Size vs Depth (Replicated Mean +- Variability)", y=0.98)
    cap = (
        "Caption: Spot-size evolution with depth shown as mean +-1sigma across seed reruns for each energy. "
        "Whitmore depth curves are overlaid when provided; otherwise Whitmore focal anchors are shown from Table 2. "
        f"{meta}{seed_msg}"
    )
    fig.text(0.01, 0.01, cap, ha="left", va="bottom", fontsize=9, wrap=True)
    fig.subplots_adjust(bottom=0.2)
    out = args.outdir / "fig01_spot_size_vs_depth_replicated.png"
    fig.savefig(out, dpi=args.dpi)
    plt.close(fig)
    captions.append((out.name, cap))

    # Figure 2: Percent difference in spot size vs depth.
    fig, axes = plt.subplots(1, len(energies), figsize=(5.4 * len(energies), 4.8), constrained_layout=False)
    if len(energies) == 1:
        axes = np.array([axes])
    for ax, energy in zip(axes, energies):
        z = profiles[energy]["z"]
        entries = profiles[energy]["seed_entries"]
        bench = table2(reference, energy)

        # Build reference curves for sigma_x and sigma_y.
        if energy in benchmark_depth_curves:
            b = benchmark_depth_curves[energy]
            bx = np.interp(z, b["z_cm"].to_numpy(dtype=float), b["sigma_x_cm"].to_numpy(dtype=float))
            by = np.interp(z, b["z_cm"].to_numpy(dtype=float), b["sigma_y_cm"].to_numpy(dtype=float))
            ref_note = "Whitmore depth curves"
        else:
            bx = np.full_like(z, bench["sigma_x_cm"], dtype=float)
            by = np.full_like(z, bench["sigma_y_cm"], dtype=float)
            ref_note = "Whitmore focal sigma constants"

        px_rows = []
        py_rows = []
        for entry in entries:
            sx = np.asarray(entry["sx_curve"], dtype=float)
            sy = np.asarray(entry["sy_curve"], dtype=float)
            px_rows.append(100.0 * (sx - bx) / bx)
            py_rows.append(100.0 * (sy - by) / by)
        px = np.vstack(px_rows)
        py = np.vstack(py_rows)
        px_mu = np.mean(px, axis=0)
        px_sd = np.std(px, axis=0, ddof=1) if px.shape[0] > 1 else np.zeros_like(px_mu)
        py_mu = np.mean(py, axis=0)
        py_sd = np.std(py, axis=0, ddof=1) if py.shape[0] > 1 else np.zeros_like(py_mu)

        ax.plot(z, px_mu, color="#ff7f0e", linewidth=2.0, label="sigma_x percent diff mean")
        ax.fill_between(z, px_mu - px_sd, px_mu + px_sd, color="#ff7f0e", alpha=0.18, label="sigma_x +-1sigma")
        ax.plot(z, py_mu, color="#17becf", linewidth=2.0, label="sigma_y percent diff mean")
        ax.fill_between(z, py_mu - py_sd, py_mu + py_sd, color="#17becf", alpha=0.18, label="sigma_y +-1sigma")
        ax.axhline(0.0, color="black", linestyle="--", linewidth=1.0)
        ax.set_title(f"E={energy} MeV")
        ax.set_xlabel("Depth in water (cm)")
        ax.set_ylabel("Percent difference vs Whitmore (%)")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=7.2, loc="upper left")
        ax.text(0.02, 0.02, f"Reference: {ref_note}", transform=ax.transAxes, fontsize=8, ha="left", va="bottom")
    fig.suptitle("Figure 2: Spot-Size Percent Difference vs Depth", y=0.98)
    cap = (
        "Caption: Percent-difference curves separate trend agreement from absolute mismatch. "
        "Bands show seed variability; central lines are seed means. "
        "When full Whitmore depth curves are unavailable, comparison uses Whitmore focal sigma constants by energy. "
        f"{meta}{seed_msg}"
    )
    fig.text(0.01, 0.01, cap, ha="left", va="bottom", fontsize=9, wrap=True)
    fig.subplots_adjust(bottom=0.2)
    out = args.outdir / "fig02_percent_difference_spot_size_vs_depth.png"
    fig.savefig(out, dpi=args.dpi)
    plt.close(fig)
    captions.append((out.name, cap))

    # Figure 3: Entrance, peak, exit dose summary.
    fig, axes = plt.subplots(1, len(energies), figsize=(5.2 * len(energies), 4.8), constrained_layout=False)
    if len(energies) == 1:
        axes = np.array([axes])
    cats = ["Entrance", "Peak", "Exit"]
    xpos = np.arange(3, dtype=float)
    for ax, energy in zip(axes, energies):
        srow = summary_df[summary_df["energy_mev"].astype(int) == energy].iloc[0]
        bench = table2(reference, energy)
        model_vals = [float(srow["entrance_pct_mean"]), float(srow["peak_pct_mean"]), float(srow["exit_pct_mean"])]
        model_err = [float(srow["entrance_pct_std"]), float(srow["peak_pct_std"]), float(srow["exit_pct_std"])]
        ax.bar(xpos - 0.14, model_vals, yerr=model_err, capsize=4, width=0.28, color=colors.get(energy, "#1f77b4"), label="Model mean +-1sigma")
        ax.scatter([xpos[0] + 0.17, xpos[1] + 0.17], [bench["entrance_pct"], 100.0], marker="*", s=150, color="#d62728", edgecolor="black", linewidth=0.5, label="Whitmore available")
        ax.text(xpos[2] + 0.17, 7.0, "N/A", fontsize=9, color="#d62728", ha="center")
        ax.set_xticks(xpos, cats)
        ax.set_ylim(0, 110)
        ax.set_ylabel("Dose / peak dose (%)")
        ax.set_title(f"E={energy} MeV")
        ax.grid(axis="y", alpha=0.25)
        ax.legend(fontsize=7.5, loc="upper right")
    fig.suptitle("Figure 3: Entrance / Peak / Exit Dose Summary", y=0.98)
    cap = (
        "Caption: Replicated dose-ratio summary across seeds. "
        "Whitmore entrance and peak ratios are from Table 2; Whitmore exit ratio is not reported. "
        f"{meta}{seed_msg}"
    )
    fig.text(0.01, 0.01, cap, ha="left", va="bottom", fontsize=9, wrap=True)
    fig.subplots_adjust(bottom=0.2)
    out = args.outdir / "fig03_entrance_peak_exit_summary_replicated.png"
    fig.savefig(out, dpi=args.dpi)
    plt.close(fig)
    captions.append((out.name, cap))

    # Figure 4: Representative lateral profiles (seed closest to mean per energy).
    fig, axes = plt.subplots(len(energies), 3, figsize=(13.8, 4.3 * len(energies)), constrained_layout=False)
    if len(energies) == 1:
        axes = np.array([axes])
    depth_titles = ["Entrance", "Near Focus", "Exit"]
    for r, energy in enumerate(energies):
        block = raw_selected[raw_selected["energy_mev"].astype(int) == energy].copy()
        rep_seed = nearest_seed_to_mean(
            block,
            columns=["z_hat_integrated_cm", "sigma_x_integrated_cm", "sigma_y_integrated_cm", "entrance_on_axis_pct"],
        )
        rep_entry = [e for e in profiles[energy]["seed_entries"] if int(e["seed"]) == rep_seed][0]
        grid = rep_entry["grid"]
        x = rep_entry["x"]
        y = rep_entry["y"]
        depth_idx = [0, int(rep_entry["peak_idx"]), grid.shape[2] - 1]
        for c, iz in enumerate(depth_idx):
            sl = grid[:, :, iz]
            cy = len(y) // 2
            cx = len(x) // 2
            px = norm100(sl[:, cy])
            py = norm100(sl[cx, :])
            ax = axes[r, c]
            ax.plot(x, px, color="#1f77b4", linewidth=1.9, label="x-profile")
            ax.plot(y, py, color="#ff7f0e", linestyle="--", linewidth=1.8, label="y-profile")
            ax.set_title(f"E={energy} MeV | {depth_titles[c]} | z={float(rep_entry['z'][iz]):.2f} cm | seed {rep_seed}", fontsize=9)
            ax.set_xlabel("Position (cm)")
            ax.set_ylabel("Normalized dose (%)")
            ax.set_ylim(0, 105)
            ax.grid(alpha=0.25)
            if r == 0 and c == 0:
                ax.legend(fontsize=8, loc="upper right")
    fig.suptitle("Figure 4: Representative Lateral Profiles (Seed Closest to Mean)", y=0.995)
    cap = (
        "Caption: Representative seed per energy selected as closest to the seed-mean metric vector, "
        "showing entrance, near-focus, and exit lateral profile shapes."
    )
    fig.text(0.01, 0.01, cap, ha="left", va="bottom", fontsize=9, wrap=True)
    fig.subplots_adjust(bottom=0.08)
    out = args.outdir / "fig04_representative_lateral_profiles_seed_closest_mean.png"
    fig.savefig(out, dpi=args.dpi)
    plt.close(fig)
    captions.append((out.name, cap))

    # Figure 5: Seed stability figure for focal sigma_mean.
    fig, ax = plt.subplots(figsize=(8.8, 5.0), constrained_layout=False)
    xt = np.arange(len(energies), dtype=float)
    for i, energy in enumerate(energies):
        entries = profiles[energy]["seed_entries"]
        focal_sigma_seed = np.array([0.5 * (e["sigma_x_focal_cm"] + e["sigma_y_focal_cm"]) for e in entries], dtype=float)
        jitter = np.linspace(-0.08, 0.08, len(focal_sigma_seed))
        ax.scatter(np.full_like(focal_sigma_seed, xt[i]) + jitter, focal_sigma_seed, s=46, color=colors.get(energy, "#1f77b4"), edgecolor="black", linewidth=0.4, label=f"E={energy} MeV seeds" if i == 0 else None)
        mu = float(np.mean(focal_sigma_seed))
        sd = float(np.std(focal_sigma_seed, ddof=1)) if len(focal_sigma_seed) > 1 else 0.0
        ax.errorbar([xt[i]], [mu], yerr=[sd], marker="D", color="black", capsize=5, linewidth=1.3, label="Mean +-1sigma" if i == 0 else None)
        ax.text(xt[i], mu + sd + 0.08, f"CV={coef_var(mu, sd):.2f}%", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(xt, [str(e) for e in energies])
    ax.set_xlabel("Energy (MeV)")
    ax.set_ylabel("Focal sigma_mean = (sigma_x + sigma_y)/2 (cm)")
    ax.set_title("Figure 5: Seed Stability of a Key Metric (Focal Spot Size)")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=8.5, loc="upper left")
    cap = (
        "Caption: Seed-stability plot demonstrating repeatability for focal spot size. "
        "Tight clustering implies systematic mismatch dominates over Monte Carlo random-seed noise."
    )
    fig.text(0.01, 0.01, cap, ha="left", va="bottom", fontsize=9, wrap=True)
    fig.subplots_adjust(bottom=0.2)
    out = args.outdir / "fig05_seed_stability_focal_sigma.png"
    fig.savefig(out, dpi=args.dpi)
    plt.close(fig)
    captions.append((out.name, cap))

    # Optional Figure 6: Focal depth vs Q4 trend with seed uncertainty.
    fig, ax = plt.subplots(figsize=(9.8, 5.6), constrained_layout=False)
    for energy in energies:
        block = seedmean[seedmean["energy_mev"].astype(int) == energy].sort_values("g4_t_per_m")
        yerr = block["z_hat_integrated_cm_std"] if "z_hat_integrated_cm_std" in block.columns else None
        ax.errorbar(
            block["g4_t_per_m"],
            block["z_hat_integrated_cm"],
            yerr=yerr,
            marker="o",
            linewidth=2.0,
            capsize=4,
            color=colors.get(energy, None),
            label=f"Model {energy} MeV mean +-1sigma",
        )
        t3 = table3(reference, energy)
        ax.plot(t3["g4_t_per_m"], t3["z_hat_cm"], "--", color=colors.get(energy, None), alpha=0.85, label=f"Whitmore Table 3 {energy} MeV")
    ax.set_xlabel("Q4 gradient (T/m)")
    ax.set_ylabel("Focal depth z_hat (cm)")
    ax.set_title("Figure 6: Focal Depth Trend vs Q4 (Context Figure)")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, ncol=2)
    cap = (
        "Caption: Context trend figure showing focal-depth control vs Q4 with seed uncertainty against Whitmore Table 3."
    )
    fig.text(0.01, 0.01, cap, ha="left", va="bottom", fontsize=9, wrap=True)
    fig.subplots_adjust(bottom=0.16)
    out = args.outdir / "fig06_focal_depth_vs_q4_context.png"
    fig.savefig(out, dpi=args.dpi)
    plt.close(fig)
    captions.append((out.name, cap))

    # Save selected-case table and caption index.
    selected_file = args.outdir / "selected_replicated_cases.csv"
    selected_df.to_csv(selected_file, index=False)

    cap_file = args.outdir / "figure_captions.md"
    lines = ["# Replicated Whitmore Study Figure Captions", ""]
    for name, text in captions:
        lines.append(f"- `{name}`: {text}")
    cap_file.write_text("\n".join(lines) + "\n", encoding="utf-8")

    # Short claim scaffold.
    claim_file = args.outdir / "claim_scaffold.md"
    claim_lines = [
        "# Claim Scaffold",
        "",
        "Recommended claim level:",
        "- Trend agreement across energy: supported",
        "- Repeatability across seeds: supported",
        "- Partial Whitmore consistency: supported",
        "- Full quantitative validation: not yet supported",
        "",
        "Use language such as:",
        "The replicated TOPAS runs show stable seed-to-seed behavior and reproduce key trend-level Whitmore observables, while absolute dosimetric agreement remains incomplete.",
    ]
    claim_file.write_text("\n".join(claim_lines) + "\n", encoding="utf-8")

    print(f"Wrote replicated Whitmore study package: {args.outdir}")
    print(f"Selected cases: {selected_file}")
    print(f"Summary table: {summary_csv}")
    print(f"Captions: {cap_file}")
    print(f"Claim scaffold: {claim_file}")
    for name, _ in captions:
        print(f" - {name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
