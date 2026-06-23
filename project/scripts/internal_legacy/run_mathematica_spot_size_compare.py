#!/usr/bin/env python3
"""Run Mathematica 2D-Gaussian fits on TOPAS focal/entrance/exit slices and compare to Python metrics."""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from analyze_topas_outputs import load_topas_grid


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--seed-runs-root",
        type=Path,
        default=root / "runs" / "publishable_subset" / "seed_runs",
        help="Root containing seed run folders with manifest.json and analysis outputs.",
    )
    parser.add_argument(
        "--out-csv",
        type=Path,
        default=root / "runs" / "publishable_subset" / "mathematica_sigma_comparison.csv",
        help="Output CSV summary path.",
    )
    parser.add_argument(
        "--wl-script",
        type=Path,
        default=root / "scripts" / "fit_slice_gaussian_mathematica.wl",
        help="Mathematica slice-fit script path.",
    )
    parser.add_argument(
        "--wolfram-kernel",
        type=str,
        default="/Applications/Wolfram Engine.app/Contents/Resources/Wolfram Player.app/Contents/MacOS/WolframKernel",
        help="Absolute path to WolframKernel binary.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="If >0, only run the first N manifests.",
    )
    parser.add_argument(
        "--slice-stride",
        type=int,
        default=2,
        help="Subsample stride for x/y slice points before Mathematica fit (1 keeps full 101x101).",
    )
    return parser.parse_args()


def write_slice_csv(
    slice_2d: np.ndarray,
    x_coords_cm: np.ndarray,
    y_coords_cm: np.ndarray,
    out_csv: Path,
    stride: int,
) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    step = max(1, int(stride))
    slice_ds = slice_2d[::step, ::step]
    x_ds = x_coords_cm[::step]
    y_ds = y_coords_cm[::step]
    xx, yy = np.meshgrid(x_ds, y_ds, indexing="ij")
    stacked = np.column_stack([xx.ravel(), yy.ravel(), slice_ds.ravel()])
    np.savetxt(out_csv, stacked, delimiter=",", fmt="%.10g")


def run_wolfram_fit(
    wl_script: Path,
    slice_csv: Path,
    out_json: Path,
    wolfram_kernel: str,
    cwd: Path,
) -> Dict[str, float]:
    env = os.environ.copy()
    env["WolframKernel"] = wolfram_kernel
    cmd = [
        "wolframscript",
        "-file",
        str(wl_script),
        "--slice",
        str(slice_csv),
        "--out",
        str(out_json),
    ]
    result = subprocess.run(cmd, cwd=str(cwd), env=env, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"wolframscript failed ({result.returncode}) for {slice_csv}:\n"
            f"{result.stdout}\n{result.stderr}"
        )
    with out_json.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_python_metrics(case_metrics_csv: Path, case_id: str) -> Dict[str, float]:
    df = pd.read_csv(case_metrics_csv)
    row = df[df["case_id"] == case_id]
    if row.empty:
        row = df.iloc[[0]]
    r = row.iloc[0]
    return {
        "sigma_x_integrated_cm": float(r["sigma_x_integrated_cm"]),
        "sigma_y_integrated_cm": float(r["sigma_y_integrated_cm"]),
        "z_hat_integrated_cm": float(r["z_hat_integrated_cm"]),
        "entrance_on_axis_pct": float(r["entrance_on_axis_pct"]),
        "entrance_integrated_pct": float(r["entrance_integrated_pct"]),
        "entrance_plane_max_pct": float(r["entrance_plane_max_pct"]),
    }


def analyze_manifest(manifest_path: Path, args: argparse.Namespace) -> Dict[str, object]:
    with manifest_path.open("r", encoding="utf-8") as f:
        manifest = json.load(f)

    case = manifest["cases"][0]
    case_id = case["case_id"]
    energy = int(case["energy_mev"])
    g4 = float(case["paper_gradients_t_per_m"][3])
    csv_path = Path(case["dose_csv"])
    bench = case["benchmark_metrics_table2"]
    benchmark_sigma_x = float(bench["sigma_x_cm"])
    benchmark_sigma_y = float(bench["sigma_y_cm"])
    benchmark_z = float(bench["z_hat_cm"])
    benchmark_entrance = float(bench["entrance_dose_pct"])

    run_dir = manifest_path.parent
    analysis_dir = run_dir / "analysis"
    case_metrics_csv = analysis_dir / "case_metrics.csv"
    py = load_python_metrics(case_metrics_csv, case_id)

    grid, header = load_topas_grid(csv_path, retries=5, retry_delay_sec=0.5)
    nx, ny, nz = grid.shape
    dx = float(header["dx_cm"])
    dy = float(header["dy_cm"])
    dz = float(header["dz_cm"])
    x_coords = (np.arange(nx) + 0.5 - nx / 2.0) * dx
    y_coords = (np.arange(ny) + 0.5 - ny / 2.0) * dy

    depth_integrated = np.sum(grid, axis=(0, 1))
    z_idx = int(np.argmax(depth_integrated))
    z_hat_integrated_cm = (z_idx + 0.5) * dz

    focal = grid[:, :, z_idx]
    entrance = grid[:, :, 0]
    exit_slice = grid[:, :, nz - 1]

    tmp_dir = analysis_dir / "mathematica_tmp"
    focal_csv = tmp_dir / f"{case_id}_focal_slice.csv"
    ent_csv = tmp_dir / f"{case_id}_entrance_slice.csv"
    exit_csv = tmp_dir / f"{case_id}_exit_slice.csv"
    focal_json = analysis_dir / "metrics_mathematica_focal.json"
    ent_json = analysis_dir / "metrics_mathematica_entrance.json"
    exit_json = analysis_dir / "metrics_mathematica_exit.json"

    write_slice_csv(focal, x_coords, y_coords, focal_csv, stride=args.slice_stride)
    write_slice_csv(entrance, x_coords, y_coords, ent_csv, stride=args.slice_stride)
    write_slice_csv(exit_slice, x_coords, y_coords, exit_csv, stride=args.slice_stride)

    project_root = Path(__file__).resolve().parents[1]
    focal_fit = run_wolfram_fit(args.wl_script, focal_csv, focal_json, args.wolfram_kernel, cwd=project_root)
    ent_fit = run_wolfram_fit(args.wl_script, ent_csv, ent_json, args.wolfram_kernel, cwd=project_root)
    exit_fit = run_wolfram_fit(args.wl_script, exit_csv, exit_json, args.wolfram_kernel, cwd=project_root)

    return {
        "seed_run": run_dir.name,
        "case_id": case_id,
        "energy_mev": energy,
        "g4_t_per_m": g4,
        "z_hat_integrated_cm": z_hat_integrated_cm,
        "benchmark_z_hat_cm": benchmark_z,
        "python_sigma_x_cm": py["sigma_x_integrated_cm"],
        "python_sigma_y_cm": py["sigma_y_integrated_cm"],
        "math_sigma_x_cm": float(focal_fit["sigma_x_cm"]),
        "math_sigma_y_cm": float(focal_fit["sigma_y_cm"]),
        "benchmark_sigma_x_cm": benchmark_sigma_x,
        "benchmark_sigma_y_cm": benchmark_sigma_y,
        "python_delta_sigma_x_cm": py["sigma_x_integrated_cm"] - benchmark_sigma_x,
        "python_delta_sigma_y_cm": py["sigma_y_integrated_cm"] - benchmark_sigma_y,
        "math_delta_sigma_x_cm": float(focal_fit["sigma_x_cm"]) - benchmark_sigma_x,
        "math_delta_sigma_y_cm": float(focal_fit["sigma_y_cm"]) - benchmark_sigma_y,
        "math_sigma_x_entrance_cm": float(ent_fit["sigma_x_cm"]),
        "math_sigma_y_entrance_cm": float(ent_fit["sigma_y_cm"]),
        "math_sigma_x_exit_cm": float(exit_fit["sigma_x_cm"]),
        "math_sigma_y_exit_cm": float(exit_fit["sigma_y_cm"]),
        "benchmark_entrance_pct": benchmark_entrance,
        "python_entrance_on_axis_pct": py["entrance_on_axis_pct"],
        "python_entrance_integrated_pct": py["entrance_integrated_pct"],
        "python_entrance_plane_max_pct": py["entrance_plane_max_pct"],
    }


def main() -> int:
    args = parse_args()
    manifests = sorted(args.seed_runs_root.glob("*/manifest.json"))
    if args.limit > 0:
        manifests = manifests[: args.limit]
    if not manifests:
        raise FileNotFoundError(f"No manifests found under {args.seed_runs_root}")

    rows: List[Dict[str, object]] = []
    for manifest in manifests:
        print(f"[mathematica-fit] {manifest.parent.name}")
        rows.append(analyze_manifest(manifest, args))

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote comparison CSV: {args.out_csv}")

    df = pd.DataFrame(rows)
    by_energy = df.groupby("energy_mev", dropna=False).agg(
        python_delta_sigma_x_mean_cm=("python_delta_sigma_x_cm", "mean"),
        python_delta_sigma_y_mean_cm=("python_delta_sigma_y_cm", "mean"),
        math_delta_sigma_x_mean_cm=("math_delta_sigma_x_cm", "mean"),
        math_delta_sigma_y_mean_cm=("math_delta_sigma_y_cm", "mean"),
    )
    print("\nMean sigma deltas (cm) by energy:")
    print(by_energy.to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
