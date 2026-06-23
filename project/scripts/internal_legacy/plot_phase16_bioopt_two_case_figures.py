#!/usr/bin/env python3
"""Generate comparison figures for two Phase-16 bioopt placements.

Outputs (per case + combined):
- SFRT plan overlays (3 projections) with lattice vertices and vasculature
- Physical vs LQ-equivalent effective dose heatmaps (2 slices x 2 domains)
- DVH panels (physical vs effective within each case)
- Overlay DVHs (case A vs case B) for key structures (physical and effective)
- 4-panel coronal comparison (A/B x physical/effective)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Sequence

import numpy as np

from analyze_topas_outputs import load_topas_grid
from bystander_multispecies_pde_solver import (
    calculate_effective_dose,
    calculate_phase7_survival,
    run_pde_temporal_integration,
)
from run_phase13_headneck_voxel_lattice import build_plan_sources, compute_dvh, compute_structure_metrics
from run_phase14_detailed_headneck_voxel_lattice import build_detailed_plan_phantom
from run_phase15_detailed_headneck_bioaware import (
    LOCKED_D_CYTO,
    LOCKED_GAMMA,
    LOCKED_LAMBDA_CYTO,
    LOCKED_SCALING_FACTOR,
    D_ROS,
    EMAX_CYTO,
    EMAX_ROS,
    LAMBDA_ROS,
    W_CYTO,
    W_IMMUNE,
    W_ROS,
    build_anatomical_biology_tensors,
    build_args_from_summary,
    load_phase14_summary,
    plot_dose_slices,
    plot_dvh_panels,
    plot_treatment_plan,
)

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception as exc:  # pragma: no cover
    raise RuntimeError("matplotlib is required") from exc


METRIC_CONFIG = {
    "PTV": {"prescription": 6.0, "vxs": [6.0, 10.0]},
    "GTV": {"prescription": 6.0, "vxs": [6.0, 10.0]},
    "SPINAL_CORD": {"prescription": None, "vxs": [5.0, 8.0]},
    "BRAINSTEM": {"prescription": None, "vxs": [5.0, 8.0]},
    "PAROTID_L": {"prescription": None, "vxs": [5.0, 10.0]},
    "PAROTID_R": {"prescription": None, "vxs": [5.0, 10.0]},
    "MANDIBLE": {"prescription": None, "vxs": [5.0, 10.0]},
    "THYROID": {"prescription": None, "vxs": [5.0, 10.0]},
    "PARATHYROIDS": {"prescription": None, "vxs": [5.0, 10.0]},
    "BRAIN": {"prescription": None, "vxs": [5.0, 10.0]},
    "BLOOD_BRAIN_BARRIER": {"prescription": None, "vxs": [5.0, 10.0]},
}

COMPARE_STRUCTURES = ["PTV", "GTV", "SPINAL_CORD", "BRAINSTEM", "PAROTID_R", "THYROID"]
COLORS = {
    "PTV": "tab:red",
    "GTV": "tab:orange",
    "SPINAL_CORD": "tab:blue",
    "BRAINSTEM": "tab:purple",
    "PAROTID_R": "tab:olive",
    "THYROID": "tab:pink",
}


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--phase14-root",
        type=Path,
        default=root / "runs" / "linac_6mv_headneck_detailed_voxel_lattice_sfrt_directplan",
        help="Run root containing phase14_detailed_headneck_summary.json.",
    )
    p.add_argument(
        "--bioopt-root",
        type=Path,
        default=root / "runs" / "linac_6mv_headneck_detailed_voxel_lattice_sfrt_bioopt",
        help="Phase-16 bioopt run root containing placement_* folders.",
    )
    p.add_argument("--case-a", type=str, default="placement_01_baseline_direct")
    p.add_argument("--case-b", type=str, default="placement_02_feedback_02")
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--prescription-gy", type=float, default=6.0)
    p.add_argument("--alpha", type=float, default=0.03)
    p.add_argument("--beta", type=float, default=0.003)
    p.add_argument("--pde-steps", type=int, default=400)
    p.add_argument("--pde-dt", type=float, default=0.12)
    p.add_argument("--tumor-cytokine-multiplier", type=float, default=2.0)
    p.add_argument("--hypoxic-ros-scale", type=float, default=0.12)
    p.add_argument("--hypoxic-cytokine-multiplier", type=float, default=2.7)
    p.add_argument("--artery-ros-uptake", type=float, default=0.05)
    p.add_argument("--artery-cyto-uptake", type=float, default=0.70)
    p.add_argument("--vein-ros-uptake", type=float, default=0.05)
    p.add_argument("--vein-cyto-uptake", type=float, default=0.90)
    p.add_argument("--spot-radius-mm", type=float, default=8.0)
    p.add_argument("--base-margin-mm", type=float, default=6.0)
    p.add_argument("--base-history-fraction", type=float, default=0.95)
    p.add_argument("--histories", type=int, default=1_000_000)
    p.add_argument("--dpi", type=int, default=220)
    return p.parse_args()


def _plan_dict_for_figures(
    plan_args: SimpleNamespace,
    axes_mm: Dict[str, np.ndarray],
    ptv_mask: np.ndarray,
    spot_centers_mm: Sequence[Sequence[float]],
) -> Dict[str, object]:
    meta = build_plan_sources(plan_args, axes_mm, ptv_mask, [tuple(map(float, row)) for row in spot_centers_mm])
    return {
        "spot_centers_mm": [list(map(float, row)) for row in spot_centers_mm],
        "ap_radius_mm": float(meta["ap_radius_mm"]),
        "lateral_radius_mm": float(meta["lateral_radius_mm"]),
        "num_lattice_spots": int(len(spot_centers_mm)),
        "num_sources": int(len(meta["sources"])),
    }


def _process_case(
    *,
    label: str,
    placement_dir: Path,
    args: argparse.Namespace,
    phantom: Dict[str, object],
    plan_args: SimpleNamespace,
    bio_args: SimpleNamespace,
) -> Dict[str, object]:
    summary_path = placement_dir / "analysis" / "plan_summary.json"
    dose_csv = placement_dir / "case" / "dosedata.csv"
    if not summary_path.exists():
        raise FileNotFoundError(summary_path)
    if not dose_csv.exists():
        raise FileNotFoundError(dose_csv)

    plan_summary_disk = json.loads(summary_path.read_text(encoding="utf-8"))
    spots = plan_summary_disk["spot_centers_mm"]
    structures = phantom["structures"]
    axes_mm = phantom["axes_mm"]
    phantom_meta = phantom["meta"]
    voxel_volume_cc = float(phantom_meta["voxel_volume_cc"])

    plan_for_plot = _plan_dict_for_figures(plan_args, axes_mm, structures["PTV"], spots)

    dose_raw, _ = load_topas_grid(dose_csv)
    ptv_raw = compute_structure_metrics(
        dose_raw,
        structures["PTV"],
        prescription_gy=float(args.prescription_gy),
        voxel_volume_cc=voxel_volume_cc,
    )
    raw_d95 = float(ptv_raw["d95_gy"])
    if raw_d95 <= 0.0:
        raise RuntimeError(f"{label}: non-positive raw PTV D95.")
    scale = float(args.prescription_gy) / raw_d95
    physical_dose = dose_raw.astype(np.float32) * np.float32(scale)

    lq_survival = np.exp(-float(args.alpha) * physical_dose - float(args.beta) * physical_dose**2).astype(np.float32)
    uptake_tensor, m_type, m_oxygen, _ = build_anatomical_biology_tensors(bio_args, structures)
    vz = tuple(float(v) for v in phantom_meta["voxel_size_mm"])
    hazard = run_pde_temporal_integration(
        physical_dose,
        vz,
        D_cyto=LOCKED_D_CYTO,
        lambda_cyto=LOCKED_LAMBDA_CYTO,
        gamma=LOCKED_GAMMA,
        u_k=uptake_tensor,
        M_oxygen=m_oxygen,
        M_type=m_type,
        D_ros=D_ROS,
        lambda_ros=LAMBDA_ROS,
        Emax_ros=EMAX_ROS,
        Emax_cyto=EMAX_CYTO,
        w_ros=W_ROS,
        w_cyto=W_CYTO,
        steps=int(args.pde_steps),
        dt=float(args.pde_dt),
        progress_interval=9999,
        verbose=False,
    )
    final_survival = calculate_phase7_survival(
        lq_survival,
        hazard,
        physical_dose,
        vz,
        float(LOCKED_SCALING_FACTOR),
        weight_immune=float(W_IMMUNE),
        verbose=False,
    )
    effective_dose = calculate_effective_dose(final_survival, alpha=float(args.alpha), beta=float(args.beta))

    dmax = max(float(np.max(physical_dose)), float(np.max(effective_dose))) * 1.05
    dose_axis = np.linspace(0.0, dmax, 400)
    physical_dvhs = {name: compute_dvh(physical_dose[structures[name]], dose_axis) for name in METRIC_CONFIG}
    effective_dvhs = {name: compute_dvh(effective_dose[structures[name]], dose_axis) for name in METRIC_CONFIG}

    return {
        "label": label,
        "placement_dir": placement_dir,
        "plan_for_plot": plan_for_plot,
        "physical_dose": physical_dose,
        "effective_dose": effective_dose,
        "dose_axis": dose_axis,
        "physical_dvhs": physical_dvhs,
        "effective_dvhs": effective_dvhs,
    }


def _resample(dose_axis_old: np.ndarray, curve: np.ndarray, dose_axis_new: np.ndarray) -> np.ndarray:
    return np.array([float(np.interp(x, dose_axis_old, curve)) for x in dose_axis_new], dtype=np.float32)


def plot_overlay_dvhs(
    out_file: Path,
    dose_axis: np.ndarray,
    dvhs_a: Dict[str, np.ndarray],
    dvhs_b: Dict[str, np.ndarray],
    name_a: str,
    name_b: str,
    title: str,
    *,
    dpi: int,
) -> None:
    fig, ax = plt.subplots(figsize=(11.0, 6.2), constrained_layout=True)
    for s in COMPARE_STRUCTURES:
        c = COLORS[s]
        ax.plot(dose_axis, dvhs_a[s], color=c, linestyle="-", linewidth=1.6, label=f"{s} ({name_a})")
        ax.plot(dose_axis, dvhs_b[s], color=c, linestyle="--", linewidth=1.3, alpha=0.85, label=f"{s} ({name_b})")
    ax.set_title(title)
    ax.set_xlabel("Dose (Gy)")
    ax.set_ylabel("Volume (%)")
    ax.set_xlim(0.0, float(dose_axis[-1]))
    ax.set_ylim(0.0, 100.0)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7, ncol=2, loc="upper right")
    fig.savefig(out_file, dpi=dpi)
    plt.close(fig)


def plot_four_panel_coronal_compare(
    out_file: Path,
    axes_mm: Dict[str, np.ndarray],
    structures: Dict[str, np.ndarray],
    body_mask: np.ndarray,
    case_a: Dict[str, object],
    case_b: Dict[str, object],
    *,
    clip_percentile: float,
    dpi: int,
) -> None:
    spots_a = np.asarray(case_a["plan_for_plot"]["spot_centers_mm"], dtype=np.float64)
    spots_b = np.asarray(case_b["plan_for_plot"]["spot_centers_mm"], dtype=np.float64)
    z_mm = 0.5 * (float(np.mean(spots_a[:, 2])) + float(np.mean(spots_b[:, 2])))
    z_index = int(np.argmin(np.abs(axes_mm["z"] - z_mm)))

    x_cm = axes_mm["x"] / 10.0
    y_cm = axes_mm["y"] / 10.0
    ptv_s = structures["PTV"][:, :, z_index]
    gtv_s = structures["GTV"][:, :, z_index]
    cord_s = structures["SPINAL_CORD"][:, :, z_index]

    def vmax_for(dose: np.ndarray) -> float:
        return float(np.percentile(dose[body_mask], clip_percentile))

    vphys = max(vmax_for(case_a["physical_dose"]), vmax_for(case_b["physical_dose"]))
    veff = max(vmax_for(case_a["effective_dose"]), vmax_for(case_b["effective_dose"]))

    phys_a = case_a["physical_dose"][:, :, z_index]
    eff_a = case_a["effective_dose"][:, :, z_index]
    phys_b = case_b["physical_dose"][:, :, z_index]
    eff_b = case_b["effective_dose"][:, :, z_index]

    fig, axes = plt.subplots(2, 2, figsize=(13.0, 10.5), constrained_layout=True)

    def draw(ax, img, vmax, spots, title: str) -> None:
        im = ax.imshow(
            img.T,
            origin="lower",
            extent=[float(x_cm[0]), float(x_cm[-1]), float(y_cm[0]), float(y_cm[-1])],
            cmap="inferno",
            vmin=0.0,
            vmax=vmax,
        )
        ax.contour(x_cm, y_cm, ptv_s.T.astype(float), levels=[0.5], colors=["cyan"], linewidths=1.1)
        ax.contour(x_cm, y_cm, gtv_s.T.astype(float), levels=[0.5], colors=["magenta"], linewidths=0.9, linestyles="--")
        ax.contour(x_cm, y_cm, cord_s.T.astype(float), levels=[0.5], colors=["white"], linewidths=0.9)
        ax.scatter(spots[:, 0] / 10.0, spots[:, 1] / 10.0, c="white", s=40, edgecolors="black", linewidths=0.6, zorder=6)
        ax.set_title(title)
        ax.set_xlabel("x (cm)")
        ax.set_ylabel("y (cm)")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02, label="Gy")

    draw(axes[0, 0], phys_a, vphys, spots_a[:, 0:2], "Case A physical (coronal)")
    draw(axes[0, 1], eff_a, veff, spots_a[:, 0:2], "Case A effective (coronal)")
    draw(axes[1, 0], phys_b, vphys, spots_b[:, 0:2], "Case B physical (coronal)")
    draw(axes[1, 1], eff_b, veff, spots_b[:, 0:2], "Case B effective (coronal)")

    fig.suptitle(
        f"Coronal comparison at z={float(axes_mm['z'][z_index]):.1f} mm (color clipped at body {clip_percentile:.1f}th percentile)",
        fontsize=12,
    )
    fig.savefig(out_file, dpi=dpi)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    phase14 = load_phase14_summary(args.phase14_root.resolve())
    phantom_args = build_args_from_summary(phase14)
    phantom = build_detailed_plan_phantom(phantom_args)

    plan_args = SimpleNamespace(
        size_x_cm=float(phase14["phantom"]["size_cm"][0]),
        size_y_cm=float(phase14["phantom"]["size_cm"][1]),
        size_z_cm=float(phase14["phantom"]["size_cm"][2]),
        voxel_mm=float(phase14["phantom"]["voxel_size_mm"][0]),
        spot_radius_mm=float(args.spot_radius_mm),
        base_margin_mm=float(args.base_margin_mm),
        base_history_fraction=float(args.base_history_fraction),
        histories=int(args.histories),
    )
    bio_args = SimpleNamespace(
        tumor_cytokine_multiplier=float(args.tumor_cytokine_multiplier),
        hypoxic_ros_scale=float(args.hypoxic_ros_scale),
        hypoxic_cytokine_multiplier=float(args.hypoxic_cytokine_multiplier),
        artery_ros_uptake=float(args.artery_ros_uptake),
        artery_cyto_uptake=float(args.artery_cyto_uptake),
        vein_ros_uptake=float(args.vein_ros_uptake),
        vein_cyto_uptake=float(args.vein_cyto_uptake),
    )

    bioopt = args.bioopt_root.resolve()
    dir_a = bioopt / args.case_a
    dir_b = bioopt / args.case_b
    out_dir = (args.out_dir or (bioopt / "figures_two_case_compare")).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    case_a = _process_case(label=args.case_a, placement_dir=dir_a, args=args, phantom=phantom, plan_args=plan_args, bio_args=bio_args)
    case_b = _process_case(label=args.case_b, placement_dir=dir_b, args=args, phantom=phantom, plan_args=plan_args, bio_args=bio_args)

    structures = phantom["structures"]
    axes_mm = phantom["axes_mm"]
    body_mask = structures["BODY"]

    name_a = str(case_a["label"])
    name_b = str(case_b["label"])
    sub_a = out_dir / name_a
    sub_b = out_dir / name_b
    sub_a.mkdir(parents=True, exist_ok=True)
    sub_b.mkdir(parents=True, exist_ok=True)

    plot_treatment_plan(sub_a / "01_treatment_plan_sfrt.png", axes_mm, structures, case_a["plan_for_plot"], dpi=int(args.dpi))
    plot_treatment_plan(sub_b / "01_treatment_plan_sfrt.png", axes_mm, structures, case_b["plan_for_plot"], dpi=int(args.dpi))

    plot_dose_slices(
        sub_a / "02_physical_vs_effective_heatmaps.png",
        axes_mm,
        case_a["physical_dose"],
        case_a["effective_dose"],
        structures,
        case_a["plan_for_plot"],
        clip_percentile=99.5,
        dpi=int(args.dpi),
    )
    plot_dose_slices(
        sub_b / "02_physical_vs_effective_heatmaps.png",
        axes_mm,
        case_b["physical_dose"],
        case_b["effective_dose"],
        structures,
        case_b["plan_for_plot"],
        clip_percentile=99.5,
        dpi=int(args.dpi),
    )

    plot_dvh_panels(
        sub_a / "03_dvh_panels_physical_and_effective.png",
        case_a["dose_axis"],
        case_a["physical_dvhs"],
        case_a["effective_dvhs"],
        dpi=int(args.dpi),
    )
    plot_dvh_panels(
        sub_b / "03_dvh_panels_physical_and_effective.png",
        case_b["dose_axis"],
        case_b["physical_dvhs"],
        case_b["effective_dvhs"],
        dpi=int(args.dpi),
    )

    d_axis = np.linspace(0.0, max(float(case_a["dose_axis"][-1]), float(case_b["dose_axis"][-1])), 400)
    phys_a = {s: _resample(case_a["dose_axis"], case_a["physical_dvhs"][s], d_axis) for s in COMPARE_STRUCTURES}
    phys_b = {s: _resample(case_b["dose_axis"], case_b["physical_dvhs"][s], d_axis) for s in COMPARE_STRUCTURES}
    eff_a = {s: _resample(case_a["dose_axis"], case_a["effective_dvhs"][s], d_axis) for s in COMPARE_STRUCTURES}
    eff_b = {s: _resample(case_b["dose_axis"], case_b["effective_dvhs"][s], d_axis) for s in COMPARE_STRUCTURES}

    plot_overlay_dvhs(
        out_dir / "compare_04_overlay_physical_dvh.png",
        d_axis,
        phys_a,
        phys_b,
        name_a,
        name_b,
        "Physical dose DVHs (solid=case A, dashed=case B)",
        dpi=int(args.dpi),
    )
    plot_overlay_dvhs(
        out_dir / "compare_05_overlay_effective_dvh.png",
        d_axis,
        eff_a,
        eff_b,
        name_a,
        name_b,
        "LQ-equivalent effective dose DVHs (solid=case A, dashed=case B)",
        dpi=int(args.dpi),
    )

    plot_four_panel_coronal_compare(
        out_dir / "compare_06_coronal_dose_four_panel.png",
        axes_mm,
        structures,
        body_mask,
        case_a,
        case_b,
        clip_percentile=99.5,
        dpi=int(args.dpi),
    )

    manifest = {
        "case_a": str(dir_a),
        "case_b": str(dir_b),
        "out_dir": str(out_dir),
        "figures": [
            str(sub_a / "01_treatment_plan_sfrt.png"),
            str(sub_a / "02_physical_vs_effective_heatmaps.png"),
            str(sub_a / "03_dvh_panels_physical_and_effective.png"),
            str(sub_b / "01_treatment_plan_sfrt.png"),
            str(sub_b / "02_physical_vs_effective_heatmaps.png"),
            str(sub_b / "03_dvh_panels_physical_and_effective.png"),
            str(out_dir / "compare_04_overlay_physical_dvh.png"),
            str(out_dir / "compare_05_overlay_effective_dvh.png"),
            str(out_dir / "compare_06_coronal_dose_four_panel.png"),
        ],
    }
    (out_dir / "figure_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"=== FIGURES WRITTEN TO {out_dir} ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

