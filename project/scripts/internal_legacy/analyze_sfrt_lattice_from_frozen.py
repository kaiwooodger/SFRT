#!/usr/bin/env python3
"""Post-process frozen Stage-2 outputs into SFRT lattice PVDR trends."""

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
except Exception:
    plt = None

from analyze_topas_outputs import load_topas_grid


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description=(
            "Build synthetic SFRT stripe lattices from frozen single-beam TOPAS kernels "
            "and compute depth-wise peak/valley/PVDR trends."
        )
    )
    parser.add_argument(
        "--run-root",
        type=Path,
        default=root / "runs" / "sfrt_stage2_frozen",
        help="Stage-2 frozen run root.",
    )
    parser.add_argument(
        "--pitches-cm",
        nargs="+",
        type=float,
        default=[1.0, 2.0, 3.0],
        help="Stripe pitch values (cm) for synthetic lattice superposition.",
    )
    parser.add_argument(
        "--n-beams",
        type=int,
        default=7,
        help="Odd number of laterally shifted beams to superpose.",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=None,
        help="Output directory (default: <run-root>/sfrt_post).",
    )
    parser.add_argument("--dpi", type=int, default=250, help="Plot DPI.")
    return parser.parse_args()


def shift_x_zero(arr: np.ndarray, shift_bins: int) -> np.ndarray:
    out = np.zeros_like(arr)
    nx = arr.shape[0]
    if shift_bins == 0:
        out[:] = arr
        return out
    if shift_bins > 0:
        if shift_bins < nx:
            out[shift_bins:, :] = arr[: nx - shift_bins, :]
        return out
    k = -shift_bins
    if k < nx:
        out[: nx - k, :] = arr[k:, :]
    return out


def build_lattice_plane(plane_xy: np.ndarray, shifts_bins: List[int]) -> np.ndarray:
    total = np.zeros_like(plane_xy, dtype=float)
    for shift in shifts_bins:
        total += shift_x_zero(plane_xy, shift)
    return total


def expected_centers(nx: int, dx_cm: float, pitch_cm: float, n_beams: int) -> Tuple[List[int], List[float]]:
    if n_beams < 1 or n_beams % 2 == 0:
        raise ValueError("--n-beams must be a positive odd integer.")
    half = (n_beams - 1) // 2
    offsets_cm = [i * pitch_cm for i in range(-half, half + 1)]
    centers: List[int] = []
    kept_offsets: List[float] = []
    center_index = nx / 2.0 - 0.5
    for off_cm in offsets_cm:
        idx = int(round(center_index + off_cm / dx_cm))
        if 0 <= idx < nx:
            centers.append(idx)
            kept_offsets.append(off_cm)
    uniq: List[int] = []
    uniq_offsets: List[float] = []
    for idx, off in sorted(zip(centers, kept_offsets), key=lambda t: t[0]):
        if not uniq or idx != uniq[-1]:
            uniq.append(idx)
            uniq_offsets.append(off)
    return uniq, uniq_offsets


def peak_valley_metrics(profile_x: np.ndarray, centers_idx: List[int], pitch_bins: int) -> Dict[str, float]:
    if len(centers_idx) < 2:
        return {
            "mean_peak": float("nan"),
            "mean_valley": float("nan"),
            "pvdr": float("nan"),
            "modulation": float("nan"),
        }

    win = max(1, int(round(max(1.0, pitch_bins / 3.0))))
    peak_vals: List[float] = []
    for c in centers_idx:
        l = max(0, c - win)
        r = min(profile_x.size, c + win + 1)
        if l < r:
            peak_vals.append(float(np.max(profile_x[l:r])))

    valley_vals: List[float] = []
    gap = max(1, win // 2)
    for c1, c2 in zip(centers_idx[:-1], centers_idx[1:]):
        l = min(c1, c2) + gap
        r = max(c1, c2) - gap + 1
        if l < r:
            valley_vals.append(float(np.min(profile_x[l:r])))

    if not peak_vals or not valley_vals:
        return {
            "mean_peak": float("nan"),
            "mean_valley": float("nan"),
            "pvdr": float("nan"),
            "modulation": float("nan"),
        }

    mean_peak = float(np.mean(peak_vals))
    mean_valley = float(np.mean(valley_vals))
    if mean_valley <= 0.0:
        pvdr = float("inf")
    else:
        pvdr = mean_peak / mean_valley
    denom = mean_peak + mean_valley
    modulation = (mean_peak - mean_valley) / denom if denom > 0 else float("nan")
    return {
        "mean_peak": mean_peak,
        "mean_valley": mean_valley,
        "pvdr": float(pvdr),
        "modulation": float(modulation),
    }


def read_energy_case_map(run_root: Path) -> Dict[int, Path]:
    combined = run_root / "combined_case_metrics.csv"
    if not combined.exists():
        raise FileNotFoundError(f"Missing combined_case_metrics.csv: {combined}")
    df = pd.read_csv(combined)
    if df.empty:
        raise ValueError(f"No rows in {combined}")
    out: Dict[int, Path] = {}
    for energy, block in df.groupby(df["energy_mev"].astype(int)):
        row = block.iloc[0]
        out[int(energy)] = Path(str(row["csv_file"]))
    return out


def analyze_case(
    csv_path: Path,
    energy_mev: int,
    pitches_cm: List[float],
    n_beams: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    grid, header = load_topas_grid(csv_path, retries=10, retry_delay_sec=1.0)
    nx, ny, nz = grid.shape
    dx_cm = float(header["dx_cm"])
    dz_cm = float(header["dz_cm"])
    z_cm = (np.arange(nz, dtype=float) + 0.5) * dz_cm
    depth_integrated = np.sum(grid, axis=(0, 1))
    zf_idx = int(np.argmax(depth_integrated))
    zf_cm = float(z_cm[zf_idx])

    rows: List[Dict[str, float | int]] = []
    summary_rows: List[Dict[str, float | int]] = []
    for pitch_cm in pitches_cm:
        centers_idx, centers_offsets_cm = expected_centers(nx, dx_cm, pitch_cm, n_beams)
        if len(centers_idx) < 2:
            continue
        shifts_bins = [int(round(off / dx_cm)) for off in centers_offsets_cm]
        pitch_bins = max(1, int(round(pitch_cm / dx_cm)))

        pvdr_vals: List[float] = []
        for iz in range(nz):
            lattice_xy = build_lattice_plane(grid[:, :, iz], shifts_bins)
            profile_x = np.sum(lattice_xy, axis=1)
            m = peak_valley_metrics(profile_x, centers_idx, pitch_bins)
            rows.append(
                {
                    "energy_mev": energy_mev,
                    "pitch_cm": pitch_cm,
                    "z_cm": float(z_cm[iz]),
                    "iz": int(iz),
                    "zf_cm_integrated_singlebeam": zf_cm,
                    "mean_peak_au": m["mean_peak"],
                    "mean_valley_au": m["mean_valley"],
                    "pvdr": m["pvdr"],
                    "modulation": m["modulation"],
                }
            )
            pvdr_vals.append(m["pvdr"])

        block = pd.DataFrame([r for r in rows if r["energy_mev"] == energy_mev and r["pitch_cm"] == pitch_cm])
        if block.empty:
            continue
        ent = block.iloc[0]
        foc = block.iloc[zf_idx] if zf_idx < len(block) else block.iloc[-1]
        exi = block.iloc[-1]
        threshold_depth = float("nan")
        below = block[np.isfinite(block["pvdr"]) & (block["pvdr"] <= 2.0)]
        if not below.empty:
            threshold_depth = float(below.iloc[0]["z_cm"])
        summary_rows.append(
            {
                "energy_mev": energy_mev,
                "pitch_cm": pitch_cm,
                "n_beams": n_beams,
                "single_beam_zf_cm": zf_cm,
                "pvdr_entrance": float(ent["pvdr"]),
                "pvdr_at_zf": float(foc["pvdr"]),
                "pvdr_exit": float(exi["pvdr"]),
                "mod_entrance": float(ent["modulation"]),
                "mod_at_zf": float(foc["modulation"]),
                "mod_exit": float(exi["modulation"]),
                "depth_pvdr_le_2_cm": threshold_depth,
            }
        )

    return pd.DataFrame(rows), pd.DataFrame(summary_rows)


def plot_pvdr(depth_df: pd.DataFrame, outdir: Path, dpi: int) -> List[Path]:
    if plt is None:
        return []
    written: List[Path] = []
    for energy, block_e in depth_df.groupby(depth_df["energy_mev"].astype(int)):
        fig, ax = plt.subplots(figsize=(9.2, 5.2), constrained_layout=True)
        for pitch, block_p in block_e.groupby(block_e["pitch_cm"].astype(float)):
            b = block_p.sort_values("z_cm")
            ax.plot(b["z_cm"], b["pvdr"], linewidth=2.0, label=f"Pitch {pitch:.2f} cm")
        zf = float(block_e["zf_cm_integrated_singlebeam"].iloc[0])
        ax.axvline(zf, color="black", linestyle="--", linewidth=1.1, label=f"Single-beam zf={zf:.2f} cm")
        ax.set_xlabel("Depth in water (cm)")
        ax.set_ylabel("PVDR (-)")
        ax.set_title(f"Synthetic SFRT PVDR vs Depth (E={energy} MeV)")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=9)
        out = outdir / f"pvdr_vs_depth_E{energy}.png"
        fig.savefig(out, dpi=dpi)
        plt.close(fig)
        written.append(out)
    return written


def write_summary_md(summary_df: pd.DataFrame, out_file: Path) -> None:
    lines: List[str] = []
    lines.append("# SFRT Lattice Post-Analysis Summary")
    lines.append("")
    lines.append(
        "Synthetic stripe lattices were generated by lateral superposition of each frozen single-beam kernel."
    )
    lines.append("PVDR was computed depth-wise as mean peak / mean valley on the x-profile of each lattice slice.")
    lines.append("")
    lines.append(
        "| Energy (MeV) | Pitch (cm) | Single-beam zf (cm) | PVDR entrance | PVDR at zf | PVDR exit | Mod entrance | Mod at zf | Mod exit | First depth with PVDR <= 2 (cm) |"
    )
    lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for _, r in summary_df.sort_values(["energy_mev", "pitch_cm"]).iterrows():
        lines.append(
            "| "
            f"{int(r['energy_mev'])} | "
            f"{float(r['pitch_cm']):.2f} | "
            f"{float(r['single_beam_zf_cm']):.3f} | "
            f"{float(r['pvdr_entrance']):.3f} | "
            f"{float(r['pvdr_at_zf']):.3f} | "
            f"{float(r['pvdr_exit']):.3f} | "
            f"{float(r['mod_entrance']):.3f} | "
            f"{float(r['mod_at_zf']):.3f} | "
            f"{float(r['mod_exit']):.3f} | "
            f"{float(r['depth_pvdr_le_2_cm']):.3f} |"
        )
    out_file.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    outdir = args.outdir if args.outdir is not None else (args.run_root / "sfrt_post")
    outdir.mkdir(parents=True, exist_ok=True)

    case_map = read_energy_case_map(args.run_root)
    depth_parts: List[pd.DataFrame] = []
    summary_parts: List[pd.DataFrame] = []
    for energy in sorted(case_map):
        depth_df, summary_df = analyze_case(
            csv_path=case_map[energy],
            energy_mev=energy,
            pitches_cm=[float(v) for v in args.pitches_cm],
            n_beams=int(args.n_beams),
        )
        if not depth_df.empty:
            depth_parts.append(depth_df)
        if not summary_df.empty:
            summary_parts.append(summary_df)

    if not depth_parts or not summary_parts:
        raise RuntimeError("No SFRT lattice results were produced. Check input CSV files.")

    depth_all = pd.concat(depth_parts, ignore_index=True)
    summary_all = pd.concat(summary_parts, ignore_index=True)

    depth_csv = outdir / "sfrt_lattice_pvdr_depth.csv"
    summary_csv = outdir / "sfrt_lattice_pvdr_summary.csv"
    summary_md = outdir / "sfrt_lattice_pvdr_summary.md"
    depth_all.to_csv(depth_csv, index=False)
    summary_all.to_csv(summary_csv, index=False)
    write_summary_md(summary_all, summary_md)

    plots = plot_pvdr(depth_all, outdir, dpi=args.dpi)

    run_meta = {
        "run_root": str(args.run_root),
        "pitches_cm": [float(v) for v in args.pitches_cm],
        "n_beams": int(args.n_beams),
        "depth_csv": str(depth_csv),
        "summary_csv": str(summary_csv),
        "summary_md": str(summary_md),
        "plots": [str(p) for p in plots],
    }
    (outdir / "sfrt_lattice_run_meta.json").write_text(
        json.dumps(run_meta, indent=2),
        encoding="utf-8",
    )

    print(f"Depth-wise PVDR table: {depth_csv}")
    print(f"Summary table: {summary_csv}")
    print(f"Summary markdown: {summary_md}")
    if plots:
        print("Plots:")
        for p in plots:
            print(f" - {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
