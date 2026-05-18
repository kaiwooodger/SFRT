#!/usr/bin/env python3
"""Generate a minimum trend-agreement figure package for focused VHEE benchmarking.

This script is designed for "trend agreement without full validation":
- It shows where trends match (e.g., focal depth control vs Q4),
- where absolute values still differ (sigma and entrance/exit ratios),
- and uses mean +-1 sigma uncertainty from seed reruns.
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
            "Create minimum trend-agreement figures that separate trend-consistency "
            "from absolute Whitmore mismatch."
        )
    )
    parser.add_argument(
        "--seedmean-std-csv",
        type=Path,
        default=root / "runs" / "publishable_subset" / "latest_topas_case_metrics_seedmean_std.csv",
        help="Seed-aggregated metrics table with *_std columns.",
    )
    parser.add_argument(
        "--seed-raw-csv",
        type=Path,
        default=root / "runs" / "publishable_subset" / "latest_topas_case_metrics_raw.csv",
        help="Raw seed-level metrics table.",
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
        help="Manifest used for metadata in captions.",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=root / "runs" / "trend_agreement_minset_20260319",
        help="Output directory for figures and caption file.",
    )
    parser.add_argument(
        "--energies",
        nargs="+",
        type=int,
        default=[100, 200, 250],
        help="Energies to include.",
    )
    parser.add_argument(
        "--unfocused-csv-map",
        type=Path,
        default=None,
        help=(
            "Optional JSON map of unfocused dose CSVs by energy. "
            "Example: {\"100\": \"/abs/path/unfocused_100.csv\", ...}"
        ),
    )
    parser.add_argument("--dpi", type=int, default=450, help="Figure DPI.")
    return parser.parse_args()


def load_reference(path: Path) -> Dict:
    return json.loads(path.read_text(encoding="utf-8"))


def table2_metrics(reference: Dict, energy: int) -> Dict[str, float]:
    m = reference["asymmetric_beamline"]["energies"][str(int(energy))]["benchmark_metrics_table2"]
    return {
        "z_hat_cm": float(m["z_hat_cm"]),
        "sigma_x_cm": float(m["sigma_x_cm"]),
        "sigma_y_cm": float(m["sigma_y_cm"]),
        "entrance_dose_pct": float(m["entrance_dose_pct"]),
    }


def table3_scan(reference: Dict, energy: int) -> pd.DataFrame:
    rows = reference["asymmetric_beamline"]["energies"][str(int(energy))]["supplementary_table3_scan"]
    return pd.DataFrame(rows).astype(float).sort_values("g4_t_per_m")


def manifest_meta(manifest_path: Path) -> str:
    if not manifest_path.exists():
        return "Histories/case=unknown, threads=unknown."
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        histories = payload.get("histories", "unknown")
        threads = payload.get("threads", "unknown")
        if histories != "unknown":
            histories = f"{int(histories):,}"
        return f"Histories/case={histories}, threads={threads}."
    except Exception:
        return "Histories/case=unknown, threads=unknown."


def weighted_sigma_from_slice(slice_xy: np.ndarray, x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    s = np.asarray(slice_xy, dtype=float)
    if s.ndim != 2:
        return float("nan"), float("nan")
    total = float(np.sum(s))
    if total <= 0:
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
    mx = float(np.max(a)) if a.size else 0.0
    return (100.0 * a / mx) if mx > 0 else np.zeros_like(a)


def choose_profile_depth_indices(nz: int, peak_idx: int) -> List[int]:
    shallow = int(round(0.25 * max(0, peak_idx)))
    idxs = [0, shallow, int(peak_idx), nz - 1]
    out: List[int] = []
    for i in idxs:
        j = max(0, min(nz - 1, int(i)))
        if j not in out:
            out.append(j)
    return out


def best_rows_by_energy(seedmean: pd.DataFrame, energies: List[int]) -> pd.DataFrame:
    rows = []
    for energy in energies:
        block = seedmean[seedmean["energy_mev"].astype(int) == int(energy)]
        if block.empty:
            continue
        rows.append(block.loc[block["weighted_error"].astype(float).idxmin()])
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("energy_mev").reset_index(drop=True)


def load_unfocused_map(path: Path | None) -> Dict[int, Path]:
    if path is None:
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    out: Dict[int, Path] = {}
    for k, v in payload.items():
        try:
            e = int(k)
        except Exception:
            continue
        out[e] = Path(str(v))
    return out


def main() -> int:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    reference = load_reference(args.reference)
    seedmean = pd.read_csv(args.seedmean_std_csv)
    raw = pd.read_csv(args.seed_raw_csv)

    seedmean = seedmean[seedmean["energy_mev"].astype(int).isin(args.energies)].copy()
    raw = raw[raw["energy_mev"].astype(int).isin(args.energies)].copy()
    if seedmean.empty:
        raise RuntimeError("No rows found in seedmean table for requested energies.")

    best_df = best_rows_by_energy(seedmean, args.energies)
    if best_df.empty:
        raise RuntimeError("No best rows could be selected for requested energies.")

    energies = sorted(best_df["energy_mev"].astype(int).unique().tolist())
    colors = {100: "#1f77b4", 200: "#2ca02c", 250: "#d62728"}

    meta = manifest_meta(args.manifest)
    n_seed_msg = ""
    if "seed" in raw.columns:
        seeds = sorted(raw["seed"].dropna().astype(int).unique().tolist())
        if seeds:
            n_seed_msg = f" Seed reruns: {', '.join(str(s) for s in seeds)}."

    # Load per-energy dose grids for best points.
    dose_data: Dict[int, Dict[str, object]] = {}
    for _, row in best_df.iterrows():
        energy = int(row["energy_mev"])
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

        integrated = np.sum(grid, axis=(0, 1))
        on_axis = grid[cx, cy, :]
        peak_idx = int(row["peak_index_z_integrated"]) if "peak_index_z_integrated" in row else int(np.argmax(integrated))
        depths = choose_profile_depth_indices(nz=nz, peak_idx=peak_idx)

        sx = []
        sy = []
        for iz in range(nz):
            sxi, syi = weighted_sigma_from_slice(grid[:, :, iz], x, y)
            sx.append(sxi)
            sy.append(syi)

        dose_data[energy] = {
            "row": row,
            "grid": grid,
            "x": x,
            "y": y,
            "z": z,
            "integrated_norm": norm100(integrated),
            "on_axis_norm": norm100(on_axis),
            "sx_vs_z": np.asarray(sx, dtype=float),
            "sy_vs_z": np.asarray(sy, dtype=float),
            "peak_idx": peak_idx,
            "profile_depth_idx": depths,
        }

    captions: List[Tuple[str, str]] = []

    # Figure 1: Beam width vs depth (main trend-agreement figure).
    fig, axes = plt.subplots(1, len(energies), figsize=(5.2 * len(energies), 4.5), constrained_layout=False)
    if len(energies) == 1:
        axes = np.array([axes])
    for ax, energy in zip(axes, energies):
        d = dose_data[energy]
        z = d["z"]
        sx = d["sx_vs_z"]
        sy = d["sy_vs_z"]
        bench = table2_metrics(reference, energy)

        ax.plot(z, sx, color="#ff7f0e", linewidth=2.0, label="Model sigma_x(z)")
        ax.plot(z, sy, color="#17becf", linewidth=2.0, label="Model sigma_y(z)")
        ax.scatter(
            [bench["z_hat_cm"]],
            [bench["sigma_x_cm"]],
            marker="*",
            s=150,
            color="#ff7f0e",
            edgecolor="black",
            linewidth=0.5,
            label="Whitmore sigma_x at z_hat",
        )
        ax.scatter(
            [bench["z_hat_cm"]],
            [bench["sigma_y_cm"]],
            marker="*",
            s=150,
            color="#17becf",
            edgecolor="black",
            linewidth=0.5,
            label="Whitmore sigma_y at z_hat",
        )
        ax.set_title(f"E={energy} MeV")
        ax.set_xlabel("Depth in water (cm)")
        ax.set_ylabel("Beam width sigma (cm)")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=7.5, loc="upper left")
    fig.suptitle("Figure 1: Beam Width Evolution with Depth (Trend Focus)", y=0.98)
    cap = (
        "Caption: Beam-width versus depth for best selected focused cases at each energy. "
        "Whitmore stars mark Table 2 focal-point widths at Whitmore z_hat only (full Whitmore width-vs-depth curves not reported). "
        "This figure supports trend-level beam-shape evolution assessment rather than full absolute validation. "
        f"{meta}{n_seed_msg}"
    )
    fig.text(0.01, 0.01, cap, ha="left", va="bottom", fontsize=9, wrap=True)
    fig.subplots_adjust(bottom=0.2)
    p = args.outdir / "fig01_sigma_vs_depth_trend.png"
    fig.savefig(p, dpi=args.dpi)
    plt.close(fig)
    captions.append((p.name, cap))

    # Figure 2: Representative lateral profiles at matched depths.
    depth_labels = ["Entrance", "Shallow", "Near Focus", "Exit"]
    fig, axes = plt.subplots(len(energies), 4, figsize=(16.5, 4.2 * len(energies)), constrained_layout=False)
    if len(energies) == 1:
        axes = np.array([axes])
    for r, energy in enumerate(energies):
        d = dose_data[energy]
        x = d["x"]
        y = d["y"]
        grid = d["grid"]
        idxs = d["profile_depth_idx"]
        for c in range(4):
            ax = axes[r, c]
            iz = idxs[min(c, len(idxs) - 1)]
            sl = grid[:, :, iz]
            iy = len(y) // 2
            prof = norm100(sl[:, iy])
            sxi, _ = weighted_sigma_from_slice(sl, x, y)
            ix_peak = int(np.argmax(prof))
            x0 = float(x[ix_peak])
            if np.isfinite(sxi) and sxi > 0:
                gfit = 100.0 * np.exp(-0.5 * ((x - x0) / sxi) ** 2)
                ax.plot(x, gfit, color="#d62728", linestyle="--", linewidth=1.5, label="Gaussian fit")
            ax.plot(x, prof, color=colors.get(energy, "#1f77b4"), linewidth=1.8, label="Model lateral profile")
            ax.set_title(f"E={energy} MeV | {depth_labels[c]} | z={float(d['z'][iz]):.2f} cm", fontsize=9)
            ax.set_xlabel("x (cm)")
            ax.set_ylabel("Normalized dose (%)")
            ax.set_ylim(0, 105)
            ax.grid(alpha=0.25)
            if r == 0 and c == 0:
                ax.legend(fontsize=8, loc="upper right")
    fig.suptitle("Figure 2: Representative Lateral Profiles (Shape Agreement Check)", y=0.995)
    cap = (
        "Caption: Representative normalized lateral profiles at entrance, shallow, near-focus, and exit depths for each energy. "
        "Dashed curves are Gaussian overlays used to support sigma-based shape interpretation. "
        f"{meta}{n_seed_msg}"
    )
    fig.text(0.01, 0.01, cap, ha="left", va="bottom", fontsize=9, wrap=True)
    fig.subplots_adjust(bottom=0.08)
    p = args.outdir / "fig02_representative_lateral_profiles.png"
    fig.savefig(p, dpi=args.dpi)
    plt.close(fig)
    captions.append((p.name, cap))

    # Figure 3: Central-axis depth-dose comparison with Whitmore anchors.
    fig, axes = plt.subplots(1, len(energies), figsize=(5.2 * len(energies), 4.5), constrained_layout=False)
    if len(energies) == 1:
        axes = np.array([axes])
    for ax, energy in zip(axes, energies):
        d = dose_data[energy]
        z = d["z"]
        on_axis = d["on_axis_norm"]
        integrated = d["integrated_norm"]
        bench = table2_metrics(reference, energy)
        ax.plot(z, on_axis, color=colors.get(energy, "#1f77b4"), linewidth=2.0, label="Model on-axis depth-dose")
        ax.plot(z, integrated, color="#7f7f7f", linestyle="--", linewidth=1.7, label="Model integrated depth-dose")
        ax.scatter(
            [0.0, bench["z_hat_cm"]],
            [bench["entrance_dose_pct"], 100.0],
            marker="*",
            s=130,
            color="#d62728",
            edgecolor="black",
            linewidth=0.5,
            label="Whitmore anchors (Table 2)",
        )
        ax.axvline(bench["z_hat_cm"], color="#d62728", linestyle=":", linewidth=1.2)
        ax.set_title(f"E={energy} MeV")
        ax.set_xlabel("Depth in water (cm)")
        ax.set_ylabel("Normalized dose (%)")
        ax.set_ylim(0, 105)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=7.5, loc="upper right")
    fig.suptitle("Figure 3: Central-Axis Depth-Dose (Agreement vs Mismatch)", y=0.98)
    cap = (
        "Caption: Normalized model depth-dose curves (on-axis and integrated) compared against Whitmore Table 2 anchors "
        "(entrance ratio and z_hat peak position). This figure explicitly highlights dosimetric mismatch despite partial trend consistency. "
        f"{meta}{n_seed_msg}"
    )
    fig.text(0.01, 0.01, cap, ha="left", va="bottom", fontsize=9, wrap=True)
    fig.subplots_adjust(bottom=0.2)
    p = args.outdir / "fig03_depth_dose_with_whitmore_anchors.png"
    fig.savefig(p, dpi=args.dpi)
    plt.close(fig)
    captions.append((p.name, cap))

    # Figure 4: Entrance/Peak/Exit ratio summary.
    fig, axes = plt.subplots(1, len(energies), figsize=(5.3 * len(energies), 4.8), constrained_layout=False)
    if len(energies) == 1:
        axes = np.array([axes])
    cats = ["Entrance", "Peak", "Exit"]
    xloc = np.arange(len(cats), dtype=float)
    for ax, energy in zip(axes, energies):
        row = best_df[best_df["energy_mev"].astype(int) == energy].iloc[0]
        bench = table2_metrics(reference, energy)
        model_vals = [
            float(row["entrance_on_axis_pct"]),
            100.0,
            float(row["exit_on_axis_pct"]),
        ]
        model_err = [
            float(row["entrance_on_axis_pct_std"]) if "entrance_on_axis_pct_std" in row else 0.0,
            0.0,
            float(row["exit_on_axis_pct_std"]) if "exit_on_axis_pct_std" in row else 0.0,
        ]
        ax.bar(xloc - 0.15, model_vals, yerr=model_err, width=0.3, capsize=4, color=colors.get(energy, "#1f77b4"), label="Model mean ±1σ")

        ax.scatter([xloc[0] + 0.18, xloc[1] + 0.18], [bench["entrance_dose_pct"], 100.0], marker="*", s=140, color="#d62728", edgecolor="black", linewidth=0.5, label="Whitmore (available)")
        ax.text(xloc[2] + 0.18, 8.0, "N/A", color="#d62728", fontsize=9, ha="center")
        ax.set_xticks(xloc, cats)
        ax.set_ylim(0, 110)
        ax.set_ylabel("Dose / peak dose (%)")
        ax.set_title(f"E={energy} MeV")
        ax.grid(axis="y", alpha=0.25)
        ax.legend(fontsize=7.5, loc="upper right")
    fig.suptitle("Figure 4: Entrance/Peak/Exit Ratio Summary", y=0.98)
    cap = (
        "Caption: Compact ratio summary isolating key dosimetric quantities. "
        "Model bars are mean +-1σ across seeds. Whitmore entrance and peak are from Table 2; Whitmore exit ratio is not reported in the paper. "
        f"{meta}{n_seed_msg}"
    )
    fig.text(0.01, 0.01, cap, ha="left", va="bottom", fontsize=9, wrap=True)
    fig.subplots_adjust(bottom=0.2)
    p = args.outdir / "fig04_entrance_peak_exit_summary.png"
    fig.savefig(p, dpi=args.dpi)
    plt.close(fig)
    captions.append((p.name, cap))

    # Figure 5: Focal depth vs quadrupole setting.
    fig, ax = plt.subplots(figsize=(10.8, 6.0), constrained_layout=False)
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
            label=f"Model {energy} MeV (mean ±1σ)",
        )
        t3 = table3_scan(reference, energy)
        ax.plot(
            t3["g4_t_per_m"],
            t3["z_hat_cm"],
            linestyle="--",
            linewidth=1.6,
            color=colors.get(energy, None),
            alpha=0.85,
            label=f"Whitmore Table 3 {energy} MeV",
        )
    ax.set_xlabel("Q4 Gradient (T/m)")
    ax.set_ylabel("Depth of Maximum Dose z_hat (cm)")
    ax.set_title("Figure 5: Focal Depth Control vs Q4 Strength")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8.3, ncol=2)
    cap = (
        "Caption: Focal-depth control trend against Q4 gradient. "
        "This is the strongest direct trend-agreement observable against Whitmore Table 3. "
        f"{meta}{n_seed_msg}"
    )
    fig.text(0.01, 0.01, cap, ha="left", va="bottom", fontsize=9, wrap=True)
    fig.subplots_adjust(bottom=0.16)
    p = args.outdir / "fig05_focal_depth_vs_q4.png"
    fig.savefig(p, dpi=args.dpi)
    plt.close(fig)
    captions.append((p.name, cap))

    # Figure 6: Focused vs unfocused 2D maps (if unfocused inputs are provided).
    unfocused_map = load_unfocused_map(args.unfocused_csv_map)
    fig, axes = plt.subplots(len(energies), 2, figsize=(10.5, 4.3 * len(energies)), constrained_layout=False)
    if len(energies) == 1:
        axes = np.array([axes])
    for r, energy in enumerate(energies):
        d = dose_data[energy]
        grid = d["grid"]
        x = d["x"]
        y = d["y"]
        peak_idx = int(d["peak_idx"])
        sl_focus = norm100(grid[:, :, peak_idx])

        ax_u = axes[r, 0]
        ax_f = axes[r, 1]

        if energy in unfocused_map and unfocused_map[energy].exists():
            u_grid, u_header = load_topas_grid(unfocused_map[energy], retries=6, retry_delay_sec=0.7)
            uz = (np.arange(u_grid.shape[2], dtype=float) + 0.5) * float(u_header["dz_cm"])
            target_z = float(d["z"][peak_idx])
            u_idx = int(np.argmin(np.abs(uz - target_z)))
            u_sl = norm100(u_grid[:, :, u_idx])
            ux = (np.arange(u_grid.shape[0], dtype=float) + 0.5 - u_grid.shape[0] / 2.0) * float(u_header["dx_cm"])
            uy = (np.arange(u_grid.shape[1], dtype=float) + 0.5 - u_grid.shape[1] / 2.0) * float(u_header["dy_cm"])
            im0 = ax_u.imshow(
                u_sl.T,
                origin="lower",
                extent=[float(ux.min()), float(ux.max()), float(uy.min()), float(uy.max())],
                cmap="viridis",
                aspect="equal",
                vmin=0.0,
                vmax=100.0,
            )
            ax_u.set_title(f"E={energy} MeV unfocused map")
            fig.colorbar(im0, ax=ax_u, fraction=0.046, pad=0.02).set_label("Dose / slice max (%)")
        else:
            ax_u.text(0.5, 0.5, "Unfocused dataset\nnot provided", ha="center", va="center", fontsize=11)
            ax_u.set_title(f"E={energy} MeV unfocused map")
            ax_u.set_xlim(0, 1)
            ax_u.set_ylim(0, 1)

        im1 = ax_f.imshow(
            sl_focus.T,
            origin="lower",
            extent=[float(x.min()), float(x.max()), float(y.min()), float(y.max())],
            cmap="viridis",
            aspect="equal",
            vmin=0.0,
            vmax=100.0,
        )
        ax_f.set_title(f"E={energy} MeV focused map (z={float(d['z'][peak_idx]):.2f} cm)")
        fig.colorbar(im1, ax=ax_f, fraction=0.046, pad=0.02).set_label("Dose / slice max (%)")

        ax_u.set_xlabel("x (cm)")
        ax_u.set_ylabel("y (cm)")
        ax_f.set_xlabel("x (cm)")
        ax_f.set_ylabel("y (cm)")

    fig.suptitle("Figure 6: Focused vs Unfocused 2D Dose Maps (Qualitative)", y=0.995)
    cap = (
        "Caption: Qualitative focused-versus-unfocused map comparison at matched energies. "
        "If unfocused maps are not supplied, focused maps are still reported to visualize focal behavior class. "
        f"{meta}{n_seed_msg}"
    )
    fig.text(0.01, 0.01, cap, ha="left", va="bottom", fontsize=9, wrap=True)
    fig.subplots_adjust(bottom=0.08)
    p = args.outdir / "fig06_focused_vs_unfocused_maps.png"
    fig.savefig(p, dpi=args.dpi)
    plt.close(fig)
    captions.append((p.name, cap))

    # Residual figure (useful extra): percent width residual at Whitmore focal point.
    fig, ax = plt.subplots(figsize=(8.6, 5.2), constrained_layout=False)
    en = []
    dsx = []
    dsy = []
    for energy in energies:
        row = best_df[best_df["energy_mev"].astype(int) == energy].iloc[0]
        bench = table2_metrics(reference, energy)
        en.append(energy)
        dsx.append(100.0 * (float(row["sigma_x_integrated_cm"]) - bench["sigma_x_cm"]) / bench["sigma_x_cm"])
        dsy.append(100.0 * (float(row["sigma_y_integrated_cm"]) - bench["sigma_y_cm"]) / bench["sigma_y_cm"])
    xloc = np.arange(len(en), dtype=float)
    ax.bar(xloc - 0.15, dsx, width=0.3, color="#ff7f0e", label="sigma_x residual (%)")
    ax.bar(xloc + 0.15, dsy, width=0.3, color="#17becf", label="sigma_y residual (%)")
    ax.axhline(0.0, color="black", linestyle="--", linewidth=1.0)
    ax.set_xticks(xloc, [str(v) for v in en])
    ax.set_xlabel("Energy (MeV)")
    ax.set_ylabel("Percent residual vs Whitmore (%)")
    ax.set_title("Figure 7: Width Residual Summary at Best Selected Cases")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=8.5)
    cap = (
        "Caption: Percent residual summary for focal spot widths relative to Whitmore values. "
        "Used to show trend agreement with explicit absolute mismatch magnitude."
    )
    fig.text(0.01, 0.01, cap, ha="left", va="bottom", fontsize=9, wrap=True)
    fig.subplots_adjust(bottom=0.2)
    p = args.outdir / "fig07_width_residual_summary.png"
    fig.savefig(p, dpi=args.dpi)
    plt.close(fig)
    captions.append((p.name, cap))

    caption_file = args.outdir / "figure_captions.md"
    lines = ["# Minimum Trend-Agreement Figure Captions", ""]
    for name, text in captions:
        lines.append(f"- `{name}`: {text}")
    caption_file.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Wrote figures to: {args.outdir}")
    for name, _ in captions:
        print(f" - {name}")
    print(f"Caption index: {caption_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
