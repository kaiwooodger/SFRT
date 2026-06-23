#!/usr/bin/env python3
"""Scan PVDR around a target pitch using existing TOPAS manifest case outputs."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from analyze_topas_outputs import load_topas_grid


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description=(
            "Compute synthetic SFRT PVDR trends for all usable cases in one or more manifests, "
            "then rank Q4/pitch combinations."
        )
    )
    parser.add_argument(
        "--manifests",
        nargs="+",
        type=Path,
        default=[
            root / "runs" / "manifest.json",
            root / "runs" / "sfrt_smallspot_250" / "manifest.json",
            root / "runs" / "sfrt_stage2_frozen" / "E100" / "manifest.json",
            root / "runs" / "sfrt_stage2_frozen" / "E200" / "manifest.json",
            root / "runs" / "sfrt_stage2_frozen" / "E250" / "manifest.json",
        ],
        help="Manifest files to include.",
    )
    parser.add_argument(
        "--pitches-mm",
        nargs="+",
        type=float,
        default=[8, 9, 10, 11, 12, 14],
        help="Pitch sweep (mm), centered around 10 mm by default.",
    )
    parser.add_argument(
        "--n-beams",
        type=int,
        default=11,
        help="Odd number of superposed laterally shifted beams.",
    )
    parser.add_argument(
        "--z-mode",
        choices=["zf_integrated", "entrance", "best_over_depth"],
        default="zf_integrated",
        help=(
            "Where to score PVDR for ranking: at integrated single-beam focus, at entrance, "
            "or best anywhere in depth."
        ),
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=root / "runs" / "pvdr_q4_optimization",
        help="Output directory.",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip heatmap generation for faster turnaround.",
    )
    parser.add_argument("--dpi", type=int, default=230, help="Plot DPI.")
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
    pvdr = float("inf") if mean_valley <= 0.0 else float(mean_peak / mean_valley)
    denom = mean_peak + mean_valley
    modulation = (mean_peak - mean_valley) / denom if denom > 0 else float("nan")
    return {
        "mean_peak": mean_peak,
        "mean_valley": mean_valley,
        "pvdr": pvdr,
        "modulation": float(modulation),
    }


def parse_g4(case: Dict) -> float:
    gradients = (
        case.get("paper_gradients_t_per_m")
        or case.get("gradients_t_per_m")
        or case.get("gradient_x_t_per_m")
        or []
    )
    if len(gradients) >= 4:
        return float(gradients[3])
    return float("nan")


def collect_cases(manifests: List[Path]) -> List[Dict]:
    rows: List[Dict] = []
    for manifest in manifests:
        if not manifest.exists():
            continue
        try:
            payload = json.loads(manifest.read_text(encoding="utf-8"))
        except Exception:
            continue
        for case in payload.get("cases", []):
            csv_path = Path(str(case.get("dose_csv", "")))
            if not csv_path.exists():
                continue
            try:
                if csv_path.stat().st_size <= 0:
                    continue
            except OSError:
                continue
            rows.append(
                {
                    "manifest": str(manifest),
                    "case_id": str(case.get("case_id", "")),
                    "energy_mev": int(case.get("energy_mev", -1)),
                    "g4_t_per_m": parse_g4(case),
                    "csv_file": str(csv_path),
                    "source_sigma_x_mm": case.get("source_sigma_x_mm"),
                    "source_sigma_y_mm": case.get("source_sigma_y_mm"),
                    "source_angular_x_mrad": case.get("source_angular_x_mrad"),
                    "source_angular_y_mrad": case.get("source_angular_y_mrad"),
                    "source_energy_spread_mev": case.get("source_energy_spread_mev"),
                    "gradient_x_scale": case.get("gradient_x_scale"),
                    "gradient_y_scale": case.get("gradient_y_scale"),
                    "quad_gradient_convention": case.get("quad_gradient_convention"),
                }
            )
    # De-duplicate by absolute CSV path.
    dedup: Dict[str, Dict] = {}
    for row in rows:
        dedup[row["csv_file"]] = row
    out = list(dedup.values())
    out.sort(key=lambda r: (r["energy_mev"], r["g4_t_per_m"], r["case_id"]))
    return out


def score_pvdr_for_case(
    case: Dict,
    pitches_cm: List[float],
    n_beams: int,
) -> pd.DataFrame:
    csv_path = Path(case["csv_file"])
    grid, header = load_topas_grid(csv_path, retries=8, retry_delay_sec=0.7)
    nx, _, nz = grid.shape
    dx_cm = float(header["dx_cm"])
    dz_cm = float(header["dz_cm"])
    z_cm = (np.arange(nz, dtype=float) + 0.5) * dz_cm
    zf_idx = int(np.argmax(np.sum(grid, axis=(0, 1))))
    zf_cm = float(z_cm[zf_idx])

    rows: List[Dict] = []
    for pitch_cm in pitches_cm:
        centers_idx, offsets_cm = expected_centers(nx, dx_cm, pitch_cm, n_beams)
        if len(centers_idx) < 2:
            continue
        shifts_bins = [int(round(off / dx_cm)) for off in offsets_cm]
        pitch_bins = max(1, int(round(pitch_cm / dx_cm)))

        pvdr_by_z: List[float] = []
        mod_by_z: List[float] = []
        for iz in range(nz):
            lattice_xy = build_lattice_plane(grid[:, :, iz], shifts_bins)
            profile_x = np.sum(lattice_xy, axis=1)
            m = peak_valley_metrics(profile_x, centers_idx, pitch_bins)
            pvdr_by_z.append(float(m["pvdr"]))
            mod_by_z.append(float(m["modulation"]))

        pvdr_arr = np.asarray(pvdr_by_z, dtype=float)
        mod_arr = np.asarray(mod_by_z, dtype=float)
        finite = np.isfinite(pvdr_arr)
        pvdr_best = float(np.max(pvdr_arr[finite])) if np.any(finite) else float("nan")
        best_idx = int(np.nanargmax(pvdr_arr)) if np.any(finite) else -1
        z_best = float(z_cm[best_idx]) if best_idx >= 0 else float("nan")

        rows.append(
            {
                **case,
                "pitch_cm": float(pitch_cm),
                "pitch_mm": float(pitch_cm * 10.0),
                "n_beams": int(n_beams),
                "zf_single_cm": zf_cm,
                "pvdr_entrance": float(pvdr_arr[0]),
                "pvdr_at_zf": float(pvdr_arr[zf_idx]),
                "pvdr_exit": float(pvdr_arr[-1]),
                "pvdr_best_over_depth": pvdr_best,
                "z_at_pvdr_best_cm": z_best,
                "mod_entrance": float(mod_arr[0]),
                "mod_at_zf": float(mod_arr[zf_idx]),
                "mod_exit": float(mod_arr[-1]),
                "mod_best_over_depth": float(np.nanmax(mod_arr)),
            }
        )
    return pd.DataFrame(rows)


def ranking_metric(df: pd.DataFrame, z_mode: str) -> pd.Series:
    if z_mode == "entrance":
        return df["pvdr_entrance"]
    if z_mode == "best_over_depth":
        return df["pvdr_best_over_depth"]
    return df["pvdr_at_zf"]


def plot_heatmaps(df: pd.DataFrame, outdir: Path, z_mode: str, dpi: int) -> List[Path]:
    if df.empty:
        return []
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return []
    metric_col = (
        "pvdr_entrance"
        if z_mode == "entrance"
        else ("pvdr_best_over_depth" if z_mode == "best_over_depth" else "pvdr_at_zf")
    )
    written: List[Path] = []
    for energy, block in df.groupby(df["energy_mev"].astype(int)):
        piv = (
            block.pivot_table(
                index="g4_t_per_m",
                columns="pitch_mm",
                values=metric_col,
                aggfunc="max",
            )
            .sort_index(axis=0)
            .sort_index(axis=1)
        )
        if piv.empty:
            continue
        fig, ax = plt.subplots(figsize=(8.2, 4.8), constrained_layout=True)
        data = piv.to_numpy(dtype=float)
        im = ax.imshow(data, aspect="auto", origin="lower", cmap="viridis")
        ax.set_xticks(np.arange(piv.shape[1]), [f"{v:.1f}" for v in piv.columns.to_numpy(dtype=float)])
        ax.set_yticks(np.arange(piv.shape[0]), [f"{v:.2f}" for v in piv.index.to_numpy(dtype=float)])
        ax.set_xlabel("Pitch (mm)")
        ax.set_ylabel("Q4 (T/m)")
        ax.set_title(f"E={energy} MeV | {metric_col}")
        cb = fig.colorbar(im, ax=ax)
        cb.set_label("PVDR (-)")
        out = outdir / f"heatmap_E{energy}_{metric_col}.png"
        fig.savefig(out, dpi=dpi)
        plt.close(fig)
        written.append(out)
    return written


def write_summary(df: pd.DataFrame, out_md: Path, z_mode: str) -> None:
    metric_label = (
        "PVDR(entrance)"
        if z_mode == "entrance"
        else ("best PVDR(depth)" if z_mode == "best_over_depth" else "PVDR(zf)")
    )
    lines: List[str] = []
    lines.append("# PVDR Optimization Summary")
    lines.append("")
    lines.append(f"- Ranking metric: `{metric_label}`")
    lines.append("- Objective target discussed by user: `PVDR = 11`")
    lines.append("")
    if df.empty:
        lines.append("No rows were available.")
        out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return
    metric = ranking_metric(df, z_mode)
    best_idx = int(metric.idxmax())
    best = df.loc[best_idx]
    lines.append("## Best Overall")
    lines.append("")
    lines.append(
        f"- Case: `{best['case_id']}` | E={int(best['energy_mev'])} MeV | Q4={float(best['g4_t_per_m']):.3f} T/m | pitch={float(best['pitch_mm']):.1f} mm"
    )
    lines.append(
        f"- {metric_label}={float(metric.loc[best_idx]):.3f}, PVDR(entrance)={float(best['pvdr_entrance']):.3f}, PVDR(zf)={float(best['pvdr_at_zf']):.3f}, PVDR(exit)={float(best['pvdr_exit']):.3f}"
    )
    lines.append(
        f"- Modulation at zf={float(best['mod_at_zf']):.3f}, best modulation over depth={float(best['mod_best_over_depth']):.3f}"
    )
    lines.append("")
    lines.append("## Best Per Energy")
    lines.append("")
    lines.append("| Energy (MeV) | Case | Q4 (T/m) | Pitch (mm) | PVDR(entrance) | PVDR(zf) | PVDR(exit) | Best PVDR(depth) |")
    lines.append("|---:|---|---:|---:|---:|---:|---:|---:|")
    for energy, block in df.groupby(df["energy_mev"].astype(int)):
        score = ranking_metric(block, z_mode)
        idx = int(score.idxmax())
        row = block.loc[idx]
        lines.append(
            "| "
            f"{energy} | {row['case_id']} | {float(row['g4_t_per_m']):.3f} | {float(row['pitch_mm']):.1f} | "
            f"{float(row['pvdr_entrance']):.3f} | {float(row['pvdr_at_zf']):.3f} | {float(row['pvdr_exit']):.3f} | {float(row['pvdr_best_over_depth']):.3f} |"
        )
    lines.append("")
    lines.append("## Top 10 Rows by Ranking Metric")
    lines.append("")
    cols = [
        "energy_mev",
        "case_id",
        "g4_t_per_m",
        "pitch_mm",
        "pvdr_entrance",
        "pvdr_at_zf",
        "pvdr_exit",
        "pvdr_best_over_depth",
        "mod_at_zf",
    ]
    top = df.assign(rank_metric=ranking_metric(df, z_mode)).sort_values("rank_metric", ascending=False).head(10)
    lines.append("| Energy | Case | Q4 | Pitch (mm) | PVDR(entry) | PVDR(zf) | PVDR(exit) | PVDR(best) | Mod(zf) |")
    lines.append("|---:|---|---:|---:|---:|---:|---:|---:|---:|")
    for _, row in top[cols].iterrows():
        lines.append(
            "| "
            f"{int(row['energy_mev'])} | {row['case_id']} | {float(row['g4_t_per_m']):.3f} | {float(row['pitch_mm']):.1f} | "
            f"{float(row['pvdr_entrance']):.3f} | {float(row['pvdr_at_zf']):.3f} | {float(row['pvdr_exit']):.3f} | "
            f"{float(row['pvdr_best_over_depth']):.3f} | {float(row['mod_at_zf']):.3f} |"
        )

    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    pitches_cm = [float(v) / 10.0 for v in args.pitches_mm]
    cases = collect_cases(args.manifests)
    if not cases:
        raise RuntimeError("No usable cases found in the provided manifests.")

    all_rows: List[pd.DataFrame] = []
    for case in cases:
        try:
            all_rows.append(score_pvdr_for_case(case, pitches_cm=pitches_cm, n_beams=int(args.n_beams)))
        except Exception:
            continue
    if not all_rows:
        raise RuntimeError("No PVDR rows could be computed from the case set.")

    df = pd.concat(all_rows, ignore_index=True)
    metric = ranking_metric(df, args.z_mode)
    df["rank_metric"] = metric
    df = df.sort_values(["energy_mev", "g4_t_per_m", "pitch_mm"]).reset_index(drop=True)

    raw_csv = args.outdir / "pvdr_pitch_q4_rows.csv"
    top_csv = args.outdir / "pvdr_pitch_q4_top.csv"
    best_energy_csv = args.outdir / "pvdr_pitch_q4_best_per_energy.csv"
    summary_md = args.outdir / "pvdr_pitch_q4_summary.md"
    meta_json = args.outdir / "pvdr_pitch_q4_meta.json"

    df.to_csv(raw_csv, index=False)
    df.sort_values("rank_metric", ascending=False).head(50).to_csv(top_csv, index=False)

    best_rows: List[pd.Series] = []
    for energy, block in df.groupby(df["energy_mev"].astype(int)):
        idx = int(block["rank_metric"].idxmax())
        best_rows.append(df.loc[idx])
    best_df = pd.DataFrame(best_rows).sort_values("energy_mev").reset_index(drop=True)
    best_df.to_csv(best_energy_csv, index=False)

    write_summary(df, summary_md, args.z_mode)
    heatmaps: List[Path] = []
    if not args.no_plots:
        heatmaps = plot_heatmaps(df, args.outdir, args.z_mode, args.dpi)

    target = 11.0
    best_value = float(np.nanmax(df["rank_metric"].to_numpy(dtype=float)))
    gap = target - best_value if math.isfinite(best_value) else float("nan")
    meta = {
        "n_cases_used": int(df[["csv_file"]].drop_duplicates().shape[0]),
        "n_rows": int(len(df)),
        "ranking_mode": args.z_mode,
        "pitches_mm": [float(v) for v in args.pitches_mm],
        "n_beams": int(args.n_beams),
        "target_pvdr": target,
        "best_observed_pvdr": best_value,
        "pvdr_gap_to_target": gap,
        "raw_csv": str(raw_csv),
        "top_csv": str(top_csv),
        "best_per_energy_csv": str(best_energy_csv),
        "summary_md": str(summary_md),
        "heatmaps": [str(p) for p in heatmaps],
    }
    meta_json.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"Raw rows: {raw_csv}")
    print(f"Top rows: {top_csv}")
    print(f"Best per energy: {best_energy_csv}")
    print(f"Summary: {summary_md}")
    print(f"Meta: {meta_json}")
    if heatmaps:
        print("Heatmaps:")
        for p in heatmaps:
            print(f" - {p}")
    print(f"Best observed PVDR ({args.z_mode}) = {best_value:.3f} | target=11.000 | gap={gap:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
