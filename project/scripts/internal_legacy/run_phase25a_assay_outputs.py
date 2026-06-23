#!/usr/bin/env python3
"""Phase 25A: in silico TUNEL, ELISA, and gammaH2AX outputs from Phase 25 plans."""

from __future__ import annotations

import argparse
import io
import json
from pathlib import Path
import time
from types import SimpleNamespace
from typing import Dict, List, Mapping, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from bystander_multispecies_pde_solver import (
    calculate_effective_dose,
    calculate_phase7_survival,
    calculate_systemic_immune_penalty,
    solve_multispecies_pde_3d_with_hazard_observables,
)
from build_asymmetric_sweep import write_text_with_retries
from generate_phase11c_insilico_assays import (
    calculate_gamma_h2ax_proxy,
    plot_assay_map,
    plot_elisa_curve,
    write_csv as write_assay_csv,
)
from run_phase10_complex_lattice_plan import (
    LOCKED_D_CYTO,
    LOCKED_GAMMA,
    LOCKED_LAMBDA_CYTO,
    LOCKED_SCALING_FACTOR,
    W_CYTO,
    W_IMMUNE,
    W_ROS,
)
from run_phase15_detailed_headneck_bioaware import build_anatomical_biology_tensors
from run_phase17_fraction_aware_bio_optimization import build_peak_valley_rois


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description=(
            "Use saved Phase 25 cumulative course volumes to generate assay-style "
            "TUNEL, ELISA, and gammaH2AX proxy outputs."
        )
    )
    parser.add_argument(
        "--phase25-run-root",
        type=Path,
        default=root / "runs" / "linac_6mv_headneck_detailed_voxel_lattice_sfrt_phase25_bio_risk_analysis",
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        default=None,
        help="Optional output root. Defaults to <phase25-run-root>/phase25a_assays.",
    )
    parser.add_argument("--history-interval", type=int, default=20)
    parser.add_argument("--dpi", type=int, default=180)
    return parser.parse_args()


def write_json(path: Path, payload: object) -> None:
    write_text_with_retries(path, json.dumps(payload, indent=2))


def load_phase25_context(run_root: Path) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Mapping[str, object]]:
    context_npz = np.load(run_root / "phase25_phantom_context.npz")
    structures = {
        key.removeprefix("struct_"): np.asarray(context_npz[key], dtype=bool)
        for key in context_npz.files
        if key.startswith("struct_")
    }
    axes_mm = {
        "x": np.asarray(context_npz["axes_x_mm"], dtype=np.float32),
        "y": np.asarray(context_npz["axes_y_mm"], dtype=np.float32),
        "z": np.asarray(context_npz["axes_z_mm"], dtype=np.float32),
    }
    config = json.loads((run_root / "phase25_config.json").read_text(encoding="utf-8"))
    return structures, axes_mm, config


def load_npz_arrays_with_retry(path: Path, *, retries: int = 4, pause_s: float = 1.0) -> Dict[str, np.ndarray]:
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            payload = path.read_bytes()
            with np.load(io.BytesIO(payload)) as data:
                return {name: np.asarray(data[name]) for name in data.files}
        except TimeoutError as exc:
            last_error = exc
            if attempt == retries - 1:
                break
            time.sleep(pause_s * float(attempt + 1))
    if last_error is not None:
        raise last_error
    raise RuntimeError(f"Failed to load NPZ data from {path}")


def gtv_target_depth_index(structures: Mapping[str, np.ndarray]) -> int:
    gtv_coords = np.argwhere(np.asarray(structures["GTV"], dtype=bool))
    if gtv_coords.size == 0:
        raise RuntimeError("GTV mask is empty in Phase 25A context.")
    return int(np.round(gtv_coords[:, 2].mean()))


def aggregate_bar_plot(
    out_file: Path,
    rows: List[Mapping[str, object]],
    *,
    peak_key: str,
    valley_key: str,
    title: str,
    ylabel: str,
    dpi: int,
) -> None:
    labels = [str(row["plan_id"]) for row in rows]
    x = np.arange(len(rows))
    width = 0.36
    fig, ax = plt.subplots(figsize=(10.8, 4.8), constrained_layout=True)
    ax.bar(x - width / 2.0, [float(row[peak_key]) for row in rows], width=width, color="#d62728", label="Peak ROI")
    ax.bar(x + width / 2.0, [float(row[valley_key]) for row in rows], width=width, color="#1f77b4", label="Valley ROI")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.savefig(out_file, dpi=int(dpi))
    plt.close(fig)


def plot_elisa_auc_summary(out_file: Path, rows: List[Mapping[str, object]], *, dpi: int) -> None:
    labels = [str(row["plan_id"]) for row in rows]
    x = np.arange(len(rows))
    width = 0.26
    fig, ax = plt.subplots(figsize=(11.0, 4.8), constrained_layout=True)
    ax.bar(x - width, [float(row["cytokine_global_auc"]) for row in rows], width=width, color="#444444", label="Global")
    ax.bar(x, [float(row["cytokine_peak_roi_auc"]) for row in rows], width=width, color="#ff7f0e", label="Peak ROI")
    ax.bar(x + width, [float(row["cytokine_valley_roi_auc"]) for row in rows], width=width, color="#1f77b4", label="Valley ROI")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("AUC (a.u.)")
    ax.set_title("Phase 25A: ELISA-like cytokine AUC comparison")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.savefig(out_file, dpi=int(dpi))
    plt.close(fig)


def main() -> int:
    args = parse_args()
    run_root = args.phase25_run_root.resolve()
    out_root = args.out_root.resolve() if args.out_root is not None else run_root / "phase25a_assays"
    out_root.mkdir(parents=True, exist_ok=True)
    print(f"Phase 25A starting for: {run_root}", flush=True)

    structures, axes_mm, config = load_phase25_context(run_root)
    print("Loaded Phase 25 context.", flush=True)
    manifest = json.loads((run_root / "phase25_plan_manifest.json").read_text(encoding="utf-8"))
    print(f"Loaded plan manifest with {len(manifest)} plans.", flush=True)
    bio_args = SimpleNamespace(**dict(config["bio_parameters"]))
    alpha = float(config["bio_parameters"]["alpha"])
    beta = float(config["bio_parameters"]["beta"])
    pde_steps = int(config["bio_parameters"]["pde_steps"])
    pde_dt = float(config["bio_parameters"]["pde_dt"])
    voxel_size_mm = (
        float(axes_mm["x"][1] - axes_mm["x"][0]),
        float(axes_mm["y"][1] - axes_mm["y"][0]),
        float(axes_mm["z"][1] - axes_mm["z"][0]),
    )
    x_cm = axes_mm["x"] / 10.0
    y_cm = axes_mm["y"] / 10.0
    z_cm = axes_mm["z"] / 10.0
    target_depth_idx = gtv_target_depth_index(structures)

    uptake_tensor, _, _, _ = build_anatomical_biology_tensors(bio_args, structures)
    vessel_mask = np.asarray(structures["ARTERIES"] | structures["VEINS"], dtype=bool)
    print("Built anatomical biology tensors.", flush=True)

    assay_rows: List[Dict[str, object]] = []
    for plan_entry in manifest:
        plan_id = str(plan_entry["plan_id"])
        print(f"Processing {plan_id}...", flush=True)
        plan_dir = out_root / plan_id
        plan_dir.mkdir(parents=True, exist_ok=True)
        summary_json = json.loads(Path(plan_entry["summary_json"]).read_text(encoding="utf-8"))
        spots_mm = [tuple(float(v) for v in row) for row in summary_json["spots_mm"]]

        volumes = load_npz_arrays_with_retry(Path(plan_entry["course_volumes_npz"]))
        cumulative_physical_dose = np.asarray(volumes["cumulative_physical_dose"], dtype=np.float32)

        peak_mask, valley_mask = build_peak_valley_rois(
            structures,
            axes_mm,
            spots_mm,
            peak_radius_mm=8.0,
            valley_exclusion_radius_mm=14.0,
        )

        lq_survival = np.exp(-alpha * cumulative_physical_dose - beta * cumulative_physical_dose**2).astype(np.float32)
        observables = solve_multispecies_pde_3d_with_hazard_observables(
            cumulative_physical_dose,
            voxel_size_mm,
            diffusion_coeffs=(0.8, LOCKED_D_CYTO),
            decay_coeffs=(0.2, LOCKED_LAMBDA_CYTO),
            emission_emax=(1.5, 0.8),
            emission_gamma_per_gy=LOCKED_GAMMA,
            uptake_tensor=uptake_tensor,
            hazard_weights=(W_ROS, W_CYTO),
            history_masks={"peak_roi": peak_mask, "valley_roi": valley_mask},
            history_interval=int(args.history_interval),
            steps=int(pde_steps),
            dt=float(pde_dt),
            progress_interval=50,
            verbose=True,
        )
        print(f"Completed PDE observables for {plan_id}.", flush=True)

        concentration = np.asarray(observables["concentration"], dtype=np.float32)
        peak_concentration = np.asarray(observables["peak_concentration"], dtype=np.float32)
        hazard_grid = np.asarray(observables["hazard_grid"], dtype=np.float32)
        time_axis = np.asarray(observables["time_axis"], dtype=np.float32)
        global_history = np.asarray(observables["global_mean_history"], dtype=np.float32)
        mask_history = {
            str(name): np.asarray(values, dtype=np.float32)
            for name, values in dict(observables["mask_mean_history"]).items()
        }

        final_survival = calculate_phase7_survival(
            lq_survival,
            hazard_grid,
            cumulative_physical_dose,
            voxel_size_mm,
            float(LOCKED_SCALING_FACTOR),
            weight_immune=float(W_IMMUNE),
            verbose=False,
        )
        deff_grid = calculate_effective_dose(final_survival, alpha=alpha, beta=beta)
        immune_penalty, icd_volume_cm3 = calculate_systemic_immune_penalty(cumulative_physical_dose, voxel_size_mm)
        gamma_h2ax_map = calculate_gamma_h2ax_proxy(
            cumulative_physical_dose,
            peak_concentration[0],
            alpha=alpha,
            beta=beta,
        )
        tunel_map = (1.0 - final_survival).astype(np.float32, copy=False)

        plot_assay_map(
            volume=gamma_h2ax_map,
            vessel_mask=vessel_mask,
            title=f"{plan_id}: gammaH2AX proxy",
            cbar_label="Relative gammaH2AX signal (a.u.)",
            cmap="magma",
            vmin=0.0,
            vmax=1.0,
            x_cm=x_cm,
            y_cm=y_cm,
            z_cm=z_cm,
            target_depth_idx=int(target_depth_idx),
            out_file=plan_dir / "gammah2ax_map.png",
            dpi=int(args.dpi),
        )
        plot_assay_map(
            volume=tunel_map,
            vessel_mask=vessel_mask,
            title=f"{plan_id}: TUNEL-like apoptosis proxy",
            cbar_label="TUNEL-positive fraction proxy",
            cmap="viridis",
            vmin=0.0,
            vmax=1.0,
            x_cm=x_cm,
            y_cm=y_cm,
            z_cm=z_cm,
            target_depth_idx=int(target_depth_idx),
            out_file=plan_dir / "tunel_map.png",
            dpi=int(args.dpi),
        )
        plot_elisa_curve(
            time_axis=time_axis,
            global_history=global_history,
            mask_history=mask_history,
            out_file=plan_dir / "elisa_curve.png",
            dpi=int(args.dpi),
        )

        centerline_rows: List[Dict[str, object]] = []
        center_y = cumulative_physical_dose.shape[1] // 2
        for ix in range(cumulative_physical_dose.shape[0]):
            centerline_rows.append(
                {
                    "x_cm": float(x_cm[ix]),
                    "dose_gy": float(cumulative_physical_dose[ix, center_y, target_depth_idx]),
                    "gammah2ax_proxy_au": float(gamma_h2ax_map[ix, center_y, target_depth_idx]),
                    "tunel_proxy": float(tunel_map[ix, center_y, target_depth_idx]),
                    "survival_total": float(final_survival[ix, center_y, target_depth_idx]),
                    "deff_gy": float(deff_grid[ix, center_y, target_depth_idx]),
                    "hazard_total": float(hazard_grid[ix, center_y, target_depth_idx]),
                }
            )
        write_assay_csv(centerline_rows, plan_dir / "centerline_assays.csv")

        assay_summary = {
            "plan_id": plan_id,
            "mean_gammah2ax_peak": float(np.mean(gamma_h2ax_map[peak_mask])),
            "mean_gammah2ax_valley": float(np.mean(gamma_h2ax_map[valley_mask])),
            "mean_tunel_peak": float(np.mean(tunel_map[peak_mask])),
            "mean_tunel_valley": float(np.mean(tunel_map[valley_mask])),
            "mean_cytokine_final_peak": float(np.mean(concentration[1][peak_mask])),
            "mean_cytokine_final_valley": float(np.mean(concentration[1][valley_mask])),
            "cytokine_global_auc": float(np.trapz(global_history[1], time_axis)),
            "cytokine_peak_roi_auc": (
                float(np.trapz(mask_history["peak_roi"][1], time_axis))
                if "peak_roi" in mask_history and mask_history["peak_roi"].size
                else 0.0
            ),
            "cytokine_valley_roi_auc": (
                float(np.trapz(mask_history["valley_roi"][1], time_axis))
                if "valley_roi" in mask_history and mask_history["valley_roi"].size
                else 0.0
            ),
            "immune_scalar": float(immune_penalty),
            "icd_volume_cm3": float(icd_volume_cm3),
        }
        write_json(plan_dir / "phase25a_assay_summary.json", assay_summary)
        assay_rows.append(assay_summary)
        print(f"Wrote assay outputs for {plan_id}.", flush=True)

    write_assay_csv(assay_rows, out_root / "phase25a_assay_summary.csv")
    write_json(out_root / "phase25a_assay_summary.json", assay_rows)
    print("Wrote aggregate assay tables.", flush=True)

    aggregate_bar_plot(
        out_root / "figure1_phase25a_gammah2ax_peak_vs_valley.png",
        assay_rows,
        peak_key="mean_gammah2ax_peak",
        valley_key="mean_gammah2ax_valley",
        title="Phase 25A: gammaH2AX peak-versus-valley comparison",
        ylabel="Relative gammaH2AX signal (a.u.)",
        dpi=int(args.dpi),
    )
    plot_elisa_auc_summary(
        out_root / "figure2_phase25a_elisa_auc_summary.png",
        assay_rows,
        dpi=int(args.dpi),
    )
    aggregate_bar_plot(
        out_root / "figure3_phase25a_tunel_peak_vs_valley.png",
        assay_rows,
        peak_key="mean_tunel_peak",
        valley_key="mean_tunel_valley",
        title="Phase 25A: TUNEL peak-versus-valley comparison",
        ylabel="TUNEL-positive fraction proxy",
        dpi=int(args.dpi),
    )

    print("=== PHASE 25A ASSAY OUTPUTS COMPLETE ===")
    print(f"Assay output root: {out_root}")
    print(f"Summary CSV: {out_root / 'phase25a_assay_summary.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
