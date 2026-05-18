#!/usr/bin/env python3
"""Test waist-based spot-size extraction on existing TOPAS outputs."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from analyze_topas_outputs import fitted_sigma_2d, load_reference, load_topas_grid, weighted_sigma_2d


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description=(
            "Compute waist spot sizes (argmin sigma(z)) from existing TOPAS CSV files "
            "and compare agreement against Whitmore benchmark sigma values."
        )
    )
    parser.add_argument(
        "--inputs-glob",
        type=str,
        default="runs/publishable_subset/seed_runs/*/analysis/case_metrics.csv",
        help="Glob pattern for existing case_metrics.csv files.",
    )
    parser.add_argument(
        "--reference",
        type=Path,
        default=root / "config" / "benchmark_reference.json",
        help="Reference JSON containing Whitmore benchmark metrics.",
    )
    parser.add_argument(
        "--sigma-source",
        choices=["integrated", "on_axis", "global"],
        default="integrated",
        help="Current-method sigma columns used for comparison.",
    )
    parser.add_argument(
        "--out-csv",
        type=Path,
        default=root / "runs" / "publishable_subset" / "waist_sigma_test.csv",
        help="Output CSV path.",
    )
    parser.add_argument(
        "--out-md",
        type=Path,
        default=root / "runs" / "publishable_subset" / "waist_sigma_test_summary.md",
        help="Output markdown summary path.",
    )
    parser.add_argument("--io-retries", type=int, default=5, help="CSV read retry count.")
    parser.add_argument(
        "--io-retry-delay-sec",
        type=float,
        default=0.5,
        help="Delay in seconds between retry attempts.",
    )
    parser.add_argument(
        "--waist-min-z-cm",
        type=float,
        default=0.0,
        help="Lower z bound (cm) for waist search to avoid entrance artifacts.",
    )
    parser.add_argument(
        "--waist-max-z-cm",
        type=float,
        default=float("nan"),
        help="Optional upper z bound (cm) for waist search.",
    )
    return parser.parse_args()


def current_sigma_cols(mode: str) -> tuple[str, str]:
    if mode == "on_axis":
        return "sigma_x_on_axis_cm", "sigma_y_on_axis_cm"
    if mode == "global":
        return "sigma_x_global_cm", "sigma_y_global_cm"
    return "sigma_x_integrated_cm", "sigma_y_integrated_cm"


def benchmark_sigma(reference: Dict, energy_mev: int) -> tuple[float, float]:
    row = reference["asymmetric_beamline"]["energies"][str(int(energy_mev))]["benchmark_metrics_table2"]
    return float(row["sigma_x_cm"]), float(row["sigma_y_cm"])


def safe_float(value) -> float:
    try:
        return float(value)
    except Exception:
        return float("nan")


def build_coords(n: int, d_cm: float) -> np.ndarray:
    return (np.arange(n, dtype=float) + 0.5 - n / 2.0) * d_cm


def waist_from_grid(
    grid: np.ndarray,
    dx_cm: float,
    dy_cm: float,
    dz_cm: float,
    min_z_cm: float = 0.0,
    max_z_cm: float = float("nan"),
) -> Dict[str, float]:
    nx, ny, nz = grid.shape
    x_coords = build_coords(nx, dx_cm)
    y_coords = build_coords(ny, dy_cm)

    sigma_eff = np.full(nz, np.nan, dtype=float)
    sigma_x_weighted = np.full(nz, np.nan, dtype=float)
    sigma_y_weighted = np.full(nz, np.nan, dtype=float)

    for iz in range(nz):
        sx, sy = weighted_sigma_2d(x_coords, y_coords, grid[:, :, iz])
        sigma_x_weighted[iz] = sx
        sigma_y_weighted[iz] = sy
        if np.isfinite(sx) and np.isfinite(sy) and sx > 0.0 and sy > 0.0:
            sigma_eff[iz] = math.sqrt(sx * sy)

    z_centers_cm = (np.arange(nz, dtype=float) + 0.5) * dz_cm
    valid_mask = np.isfinite(sigma_eff)
    valid_mask &= z_centers_cm >= float(min_z_cm)
    if np.isfinite(max_z_cm):
        valid_mask &= z_centers_cm <= float(max_z_cm)

    valid_internal = np.where(valid_mask[1:-1])[0] + 1 if nz > 2 else np.array([], dtype=int)
    valid_all = np.where(valid_mask)[0]
    if valid_internal.size > 0:
        waist_idx = int(valid_internal[np.argmin(sigma_eff[valid_internal])])
    elif valid_all.size > 0:
        waist_idx = int(valid_all[np.argmin(sigma_eff[valid_all])])
    else:
        waist_idx = 0

    slice_waist = grid[:, :, waist_idx]
    sx_fit, sy_fit = fitted_sigma_2d(x_coords, y_coords, slice_waist)

    if not np.isfinite(sx_fit):
        sx_fit = float(sigma_x_weighted[waist_idx])
    if not np.isfinite(sy_fit):
        sy_fit = float(sigma_y_weighted[waist_idx])

    return {
        "waist_index_z": int(waist_idx),
        "z_waist_cm": float((waist_idx + 0.5) * dz_cm),
        "sigma_x_waist_cm": float(sx_fit),
        "sigma_y_waist_cm": float(sy_fit),
        "sigma_eff_waist_cm": float(sigma_eff[waist_idx]) if np.isfinite(sigma_eff[waist_idx]) else float("nan"),
    }


def main() -> int:
    args = parse_args()
    reference = load_reference(args.reference)
    sx_col, sy_col = current_sigma_cols(args.sigma_source)

    metrics_files = sorted(Path().glob(args.inputs_glob))
    if not metrics_files:
        raise SystemExit(f"No case_metrics files matched: {args.inputs_glob}")

    rows: List[Dict] = []
    unreadable = 0
    missing_csv = 0

    for metrics_path in metrics_files:
        df = pd.read_csv(metrics_path)
        if df.empty:
            continue
        for _, r in df.iterrows():
            csv_file = Path(str(r.get("csv_file", "")))
            if not csv_file.exists() or csv_file.stat().st_size == 0:
                missing_csv += 1
                continue

            try:
                grid, header = load_topas_grid(
                    csv_file,
                    retries=args.io_retries,
                    retry_delay_sec=args.io_retry_delay_sec,
                )
            except Exception:
                unreadable += 1
                continue

            energy = int(r["energy_mev"])
            b_sx, b_sy = benchmark_sigma(reference, energy)
            waist = waist_from_grid(
                grid=grid,
                dx_cm=float(header["dx_cm"]),
                dy_cm=float(header["dy_cm"]),
                dz_cm=float(header["dz_cm"]),
                min_z_cm=args.waist_min_z_cm,
                max_z_cm=args.waist_max_z_cm,
            )

            sx_current = safe_float(r.get(sx_col))
            sy_current = safe_float(r.get(sy_col))

            err_current = float("nan")
            if np.isfinite(sx_current) and np.isfinite(sy_current):
                err_current = abs(sx_current - b_sx) + abs(sy_current - b_sy)

            err_waist = abs(float(waist["sigma_x_waist_cm"]) - b_sx) + abs(float(waist["sigma_y_waist_cm"]) - b_sy)
            improvement = err_current - err_waist if np.isfinite(err_current) else float("nan")

            rows.append(
                {
                    "case_id": str(r.get("case_id", "")),
                    "seed_group": metrics_path.parent.parent.name,
                    "energy_mev": energy,
                    "g4_t_per_m": safe_float(r.get("g4_t_per_m")),
                    "csv_file": str(csv_file),
                    "sigma_x_benchmark_cm": b_sx,
                    "sigma_y_benchmark_cm": b_sy,
                    "sigma_x_current_cm": sx_current,
                    "sigma_y_current_cm": sy_current,
                    "sigma_x_waist_cm": float(waist["sigma_x_waist_cm"]),
                    "sigma_y_waist_cm": float(waist["sigma_y_waist_cm"]),
                    "z_waist_cm": float(waist["z_waist_cm"]),
                    "waist_index_z": int(waist["waist_index_z"]),
                    "sigma_eff_waist_cm": float(waist["sigma_eff_waist_cm"]),
                    "sigma_error_current_l1_cm": err_current,
                    "sigma_error_waist_l1_cm": err_waist,
                    "sigma_error_improvement_l1_cm": improvement,
                    "waist_better": bool(np.isfinite(improvement) and improvement > 0.0),
                }
            )

    if not rows:
        raise SystemExit("No valid rows were processed.")

    out_df = pd.DataFrame(rows).sort_values(["energy_mev", "g4_t_per_m", "seed_group"])
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.out_csv, index=False, quoting=csv.QUOTE_MINIMAL)

    summary = (
        out_df.groupby("energy_mev", as_index=False)
        .agg(
            n_cases=("case_id", "count"),
            current_l1_mean=("sigma_error_current_l1_cm", "mean"),
            waist_l1_mean=("sigma_error_waist_l1_cm", "mean"),
            improvement_mean=("sigma_error_improvement_l1_cm", "mean"),
            waist_better_count=("waist_better", "sum"),
            z_waist_mean_cm=("z_waist_cm", "mean"),
        )
        .sort_values("energy_mev")
    )

    lines = [
        "# Waist Spot-Size Test Summary",
        "",
        f"- Input files matched: `{len(metrics_files)}`",
        f"- Processed rows: `{len(out_df)}`",
        f"- Missing/empty CSV rows skipped: `{missing_csv}`",
        f"- Unreadable CSV rows skipped: `{unreadable}`",
        f"- Current sigma mode compared: `{args.sigma_source}` (`{sx_col}`, `{sy_col}`)",
        f"- Waist search z-range: `[{args.waist_min_z_cm}, "
        + (f"{args.waist_max_z_cm}" if np.isfinite(args.waist_max_z_cm) else "inf")
        + "] cm`",
        "",
        "## By Energy",
        "",
        "| Energy (MeV) | N | Mean L1 Error (Current) cm | Mean L1 Error (Waist) cm | Mean Improvement cm | Waist Better Count | Mean z_waist cm |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]

    for _, r in summary.iterrows():
        lines.append(
            f"| {int(r['energy_mev'])} | {int(r['n_cases'])} | "
            f"{float(r['current_l1_mean']):.3f} | {float(r['waist_l1_mean']):.3f} | "
            f"{float(r['improvement_mean']):.3f} | {int(r['waist_better_count'])} | "
            f"{float(r['z_waist_mean_cm']):.3f} |"
        )

    args.out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Waist test CSV: {args.out_csv}")
    print(f"Waist test summary: {args.out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
