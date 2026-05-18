#!/usr/bin/env python3
"""Apply the fully calibrated biology model to the detailed direct-planned SFRT case.

This script reuses the physical dose from the heterogeneous phantom direct-plan
run, rebuilds the detailed anatomy masks, derives anatomical vascular sink
fields directly from the artery and vein masks, and computes bio-effective
dose / DVH metrics under the locked multiscale biology model.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Sequence, Tuple

import numpy as np

from analyze_topas_outputs import load_topas_grid
from build_asymmetric_sweep import write_text_with_retries
from bystander_multispecies_pde_solver import (
    calculate_effective_dose,
    calculate_phase7_survival,
    calculate_systemic_immune_penalty,
    run_pde_temporal_integration,
)
from run_phase13_headneck_voxel_lattice import compute_dvh, compute_structure_metrics, save_csv
from run_phase14_detailed_headneck_voxel_lattice import build_detailed_plan_phantom

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.patheffects as path_effects
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
except Exception as exc:  # pragma: no cover
    raise RuntimeError("matplotlib is required for figure generation") from exc


LOCKED_D_CYTO = 1.2
LOCKED_LAMBDA_CYTO = 0.001
LOCKED_GAMMA = 0.35
LOCKED_SCALING_FACTOR = 0.0029365813
W_ROS = 0.4
W_CYTO = 0.4
W_IMMUNE = 0.2
D_ROS = 0.8
LAMBDA_ROS = 0.2
EMAX_ROS = 1.5
EMAX_CYTO = 0.8


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description=(
            "Apply the fully calibrated biology model to the detailed heterogeneous "
            "head-and-neck direct-plan dose and generate plan / DVH / effective-dose figures."
        )
    )
    parser.add_argument(
        "--plan-run-root",
        type=Path,
        default=root / "runs" / "linac_6mv_headneck_detailed_voxel_lattice_sfrt_directplan",
        help="Detailed direct-plan run root containing the physical TOPAS dose and phase14 summary.",
    )
    parser.add_argument(
        "--output-dir-name",
        type=str,
        default="analysis_bioaware",
        help="Subdirectory created under the plan run root for bio-aware outputs.",
    )
    parser.add_argument(
        "--prescription-gy",
        type=float,
        default=6.0,
        help="Physical PTV prescription used to normalize the raw TOPAS dose to D95.",
    )
    parser.add_argument("--alpha", type=float, default=0.03, help="LQ alpha for physical and effective-dose inversion.")
    parser.add_argument("--beta", type=float, default=0.003, help="LQ beta for physical and effective-dose inversion.")
    parser.add_argument("--pde-steps", type=int, default=400, help="Temporal PDE steps.")
    parser.add_argument("--pde-dt", type=float, default=0.12, help="Temporal PDE time step.")
    parser.add_argument(
        "--tumor-cytokine-multiplier",
        type=float,
        default=2.0,
        help="Tumour-specific cytokine emission multiplier.",
    )
    parser.add_argument(
        "--hypoxic-ros-scale",
        type=float,
        default=0.12,
        help="ROS emission scale in the hypoxic core.",
    )
    parser.add_argument(
        "--hypoxic-cytokine-multiplier",
        type=float,
        default=2.7,
        help="Cytokine emission multiplier in the hypoxic core.",
    )
    parser.add_argument("--artery-ros-uptake", type=float, default=0.05)
    parser.add_argument("--artery-cyto-uptake", type=float, default=0.70)
    parser.add_argument("--vein-ros-uptake", type=float, default=0.05)
    parser.add_argument("--vein-cyto-uptake", type=float, default=0.90)
    parser.add_argument("--dpi", type=int, default=260, help="Figure DPI.")
    return parser.parse_args()


def load_phase14_summary(plan_run_root: Path) -> Dict[str, object]:
    summary_path = plan_run_root / "analysis" / "phase14_detailed_headneck_summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"Could not find phase14 summary: {summary_path}")
    return json.loads(summary_path.read_text(encoding="utf-8"))


def build_args_from_summary(summary: Dict[str, object]) -> SimpleNamespace:
    size_cm = summary["phantom"]["size_cm"]
    voxel_mm = summary["phantom"]["voxel_size_mm"][0]
    return SimpleNamespace(
        size_x_cm=float(size_cm[0]),
        size_y_cm=float(size_cm[1]),
        size_z_cm=float(size_cm[2]),
        voxel_mm=float(voxel_mm),
    )


def mask_centroid_mm(mask: np.ndarray, axes_mm: Dict[str, np.ndarray]) -> Tuple[float, float, float]:
    coords = np.argwhere(mask)
    centroid_idx = np.round(coords.mean(axis=0)).astype(int)
    return (
        float(axes_mm["x"][centroid_idx[0]]),
        float(axes_mm["y"][centroid_idx[1]]),
        float(axes_mm["z"][centroid_idx[2]]),
    )


def build_anatomical_biology_tensors(
    args: argparse.Namespace,
    structures: Dict[str, np.ndarray],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, float]]:
    shape = structures["BODY"].shape
    m_type = np.ones((2, *shape), dtype=np.float32)
    m_oxygen = np.ones((2, *shape), dtype=np.float32)

    gtv_mask = structures["GTV"]
    hypoxia_mask = structures["HYPOXIA"]
    m_type[1, gtv_mask] = float(args.tumor_cytokine_multiplier)
    m_oxygen[0, hypoxia_mask] = float(args.hypoxic_ros_scale)
    m_oxygen[1, hypoxia_mask] = float(args.hypoxic_cytokine_multiplier)

    uptake = np.zeros((2, *shape), dtype=np.float32)
    arteries = structures["ARTERIES"]
    veins = structures["VEINS"]
    uptake[0, arteries] = float(args.artery_ros_uptake)
    uptake[1, arteries] = float(args.artery_cyto_uptake)
    uptake[0, veins] = np.maximum(uptake[0, veins], float(args.vein_ros_uptake))
    uptake[1, veins] = np.maximum(uptake[1, veins], float(args.vein_cyto_uptake))

    uptake_meta = {
        "artery_ros_uptake": float(args.artery_ros_uptake),
        "artery_cyto_uptake": float(args.artery_cyto_uptake),
        "vein_ros_uptake": float(args.vein_ros_uptake),
        "vein_cyto_uptake": float(args.vein_cyto_uptake),
        "arterial_volume_cc": 0.0,  # overwritten below
        "venous_volume_cc": 0.0,
    }
    return uptake, m_type, m_oxygen, uptake_meta


def plot_treatment_plan(
    out_file: Path,
    axes_mm: Dict[str, np.ndarray],
    structures: Dict[str, np.ndarray],
    plan_summary: Dict[str, object],
    *,
    dpi: int,
) -> None:
    x_cm = axes_mm["x"] / 10.0
    y_cm = axes_mm["y"] / 10.0
    z_cm = axes_mm["z"] / 10.0

    body_xy = np.any(structures["BODY"], axis=2)
    body_xz = np.any(structures["BODY"], axis=1)
    body_yz = np.any(structures["BODY"], axis=0)
    ptv_xy = np.any(structures["PTV"], axis=2)
    ptv_xz = np.any(structures["PTV"], axis=1)
    ptv_yz = np.any(structures["PTV"], axis=0)
    gtv_xy = np.any(structures["GTV"], axis=2)
    gtv_xz = np.any(structures["GTV"], axis=1)
    gtv_yz = np.any(structures["GTV"], axis=0)
    artery_xy = np.any(structures["ARTERIES"], axis=2)
    artery_xz = np.any(structures["ARTERIES"], axis=1)
    artery_yz = np.any(structures["ARTERIES"], axis=0)
    vein_xy = np.any(structures["VEINS"], axis=2)
    vein_xz = np.any(structures["VEINS"], axis=1)
    vein_yz = np.any(structures["VEINS"], axis=0)
    cord_xy = np.any(structures["SPINAL_CORD"], axis=2)
    cord_xz = np.any(structures["SPINAL_CORD"], axis=1)
    cord_yz = np.any(structures["SPINAL_CORD"], axis=0)

    ptv_centroid_mm = mask_centroid_mm(structures["PTV"], axes_mm)
    spot_centers_mm = [tuple(map(float, row)) for row in plan_summary["spot_centers_mm"]]
    ap_radius_cm = float(plan_summary["ap_radius_mm"]) / 10.0

    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.2), constrained_layout=True)
    projections = [
        (axes[0], body_xy, ptv_xy, gtv_xy, artery_xy, vein_xy, cord_xy, x_cm, y_cm, "Coronal projection", "x (cm)", "y (cm)"),
        (axes[1], body_xz, ptv_xz, gtv_xz, artery_xz, vein_xz, cord_xz, x_cm, z_cm, "Axial-style projection", "x (cm)", "z (cm)"),
        (axes[2], body_yz, ptv_yz, gtv_yz, artery_yz, vein_yz, cord_yz, y_cm, z_cm, "Sagittal projection", "y (cm)", "z (cm)"),
    ]

    for ax, body, ptv, gtv, artery, vein, cord, axis_a, axis_b, title, xlabel, ylabel in projections:
        extent = [float(axis_a[0]), float(axis_a[-1]), float(axis_b[0]), float(axis_b[-1])]
        ax.imshow(body.T, origin="lower", cmap="Greys", extent=extent, alpha=0.92)
        ax.contour(axis_a, axis_b, artery.T.astype(float), levels=[0.5], colors=["#D62828"], linewidths=1.1)
        ax.contour(axis_a, axis_b, vein.T.astype(float), levels=[0.5], colors=["#277DA1"], linewidths=1.1)
        ax.contour(axis_a, axis_b, ptv.T.astype(float), levels=[0.5], colors=["cyan"], linewidths=1.5)
        ax.contour(axis_a, axis_b, gtv.T.astype(float), levels=[0.5], colors=["magenta"], linewidths=1.2, linestyles="--")
        ax.contour(axis_a, axis_b, cord.T.astype(float), levels=[0.5], colors=["white"], linewidths=0.8)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

    ap_circle = plt.Circle(
        (ptv_centroid_mm[0] / 10.0, ptv_centroid_mm[1] / 10.0),
        ap_radius_cm,
        fill=False,
        linestyle="--",
        linewidth=1.4,
        edgecolor="#00E5FF",
    )
    axes[0].add_patch(ap_circle)
    axes[0].annotate(
        "AP base-field footprint",
        xy=(ptv_centroid_mm[0] / 10.0 + ap_radius_cm * 0.55, ptv_centroid_mm[1] / 10.0),
        xytext=(4.4, 7.3),
        color="#00E5FF",
        fontsize=9,
        arrowprops={"arrowstyle": "->", "lw": 0.9, "color": "#00E5FF"},
    )

    for idx, (sx, sy, sz) in enumerate(spot_centers_mm, start=1):
        label = f"S{idx}"
        text_kw = dict(
            color="white",
            fontsize=8,
            ha="center",
            va="center",
            path_effects=[path_effects.Stroke(linewidth=1.4, foreground="black"), path_effects.Normal()],
        )
        axes[0].scatter([sx / 10.0], [sy / 10.0], c="yellow", s=34, edgecolors="black", linewidths=0.5, zorder=5)
        axes[0].text(sx / 10.0, sy / 10.0, label, **text_kw)
        axes[1].scatter([sx / 10.0], [sz / 10.0], c="yellow", s=34, edgecolors="black", linewidths=0.5, zorder=5)
        axes[1].text(sx / 10.0, sz / 10.0, label, **text_kw)
        axes[2].scatter([sy / 10.0], [sz / 10.0], c="yellow", s=34, edgecolors="black", linewidths=0.5, zorder=5)
        axes[2].text(sy / 10.0, sz / 10.0, label, **text_kw)

    fig.suptitle(
        "Direct SFRT treatment plan on the heterogeneous head-and-neck phantom with explicit vascular anatomy",
        fontsize=13,
    )
    legend_handles = [
        Line2D([0], [0], color="#D62828", lw=1.3, label="Arterial network"),
        Line2D([0], [0], color="#277DA1", lw=1.3, label="Venous network"),
        Line2D([0], [0], color="cyan", lw=1.5, label="PTV"),
        Line2D([0], [0], color="magenta", lw=1.2, linestyle="--", label="GTV"),
        Line2D([0], [0], color="white", lw=0.9, label="Spinal cord"),
        Line2D([0], [0], marker="o", color="yellow", markeredgecolor="black", lw=0, markersize=7, label="Lattice vertices"),
        Line2D([0], [0], color="#00E5FF", lw=1.4, linestyle="--", label="AP base-field footprint"),
    ]
    axes[2].legend(handles=legend_handles, loc="lower right", fontsize=8, framealpha=0.9)
    fig.savefig(out_file, dpi=dpi)
    plt.close(fig)


def plot_sink_field(
    out_file: Path,
    axes_mm: Dict[str, np.ndarray],
    uptake_tensor: np.ndarray,
    structures: Dict[str, np.ndarray],
    *,
    dpi: int,
) -> None:
    y_index = int(np.argmin(np.abs(axes_mm["y"] - 0.0)))
    z_index = int(np.argmin(np.abs(axes_mm["z"] - 0.0)))
    x_cm = axes_mm["x"] / 10.0
    y_cm = axes_mm["y"] / 10.0
    z_cm = axes_mm["z"] / 10.0
    cyto_uptake = uptake_tensor[1]

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), constrained_layout=True)
    panels = [
        (axes[0], cyto_uptake[:, y_index, :], structures["PTV"][:, y_index, :], x_cm, z_cm, "Cytokine sink field: axial-style slice", "x (cm)", "z (cm)"),
        (axes[1], cyto_uptake[:, :, z_index], structures["PTV"][:, :, z_index], x_cm, y_cm, "Cytokine sink field: coronal slice", "x (cm)", "y (cm)"),
    ]
    for ax, image, ptv_slice, axis_a_cm, axis_b_cm, title, xlabel, ylabel in panels:
        im = ax.imshow(
            image.T,
            origin="lower",
            extent=[float(axis_a_cm[0]), float(axis_a_cm[-1]), float(axis_b_cm[0]), float(axis_b_cm[-1])],
            cmap="viridis",
            vmin=0.0,
            vmax=float(np.max(cyto_uptake)) if float(np.max(cyto_uptake)) > 0.0 else 1.0,
        )
        ax.contour(axis_a_cm, axis_b_cm, ptv_slice.T.astype(float), levels=[0.5], colors=["white"], linewidths=1.0)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Uptake rate")

    fig.savefig(out_file, dpi=dpi)
    plt.close(fig)


def plot_dose_slices(
    out_file: Path,
    axes_mm: Dict[str, np.ndarray],
    physical_dose: np.ndarray,
    bio_dose: np.ndarray,
    structures: Dict[str, np.ndarray],
    plan_summary: Dict[str, object],
    *,
    clip_percentile: float = 99.5,
    dpi: int,
) -> None:
    spot_centers_mm = np.asarray(plan_summary["spot_centers_mm"], dtype=np.float32)
    lattice_centroid_mm = np.mean(spot_centers_mm, axis=0)
    y_index = int(np.argmin(np.abs(axes_mm["y"] - float(lattice_centroid_mm[1]))))
    z_index = int(np.argmin(np.abs(axes_mm["z"] - float(lattice_centroid_mm[2]))))
    x_cm = axes_mm["x"] / 10.0
    y_cm = axes_mm["y"] / 10.0
    z_cm = axes_mm["z"] / 10.0
    body_mask = structures["BODY"]

    physical_vmax = float(np.percentile(physical_dose[body_mask], float(clip_percentile)))
    bio_vmax = float(np.percentile(bio_dose[body_mask], float(clip_percentile)))

    fig, axes = plt.subplots(2, 2, figsize=(12.5, 9.0), constrained_layout=True)
    panels = [
        (
            axes[0, 0],
            physical_dose[:, y_index, :],
            structures["PTV"][:, y_index, :],
            structures["GTV"][:, y_index, :],
            structures["SPINAL_CORD"][:, y_index, :],
            x_cm,
            z_cm,
            f"Physical dose: axial-style slice at y = {float(axes_mm['y'][y_index]):.1f} mm",
            "x (cm)",
            "z (cm)",
            physical_vmax,
            "x-z",
        ),
        (
            axes[0, 1],
            physical_dose[:, :, z_index],
            structures["PTV"][:, :, z_index],
            structures["GTV"][:, :, z_index],
            structures["SPINAL_CORD"][:, :, z_index],
            x_cm,
            y_cm,
            f"Physical dose: coronal slice at z = {float(axes_mm['z'][z_index]):.1f} mm",
            "x (cm)",
            "y (cm)",
            physical_vmax,
            "x-y",
        ),
        (
            axes[1, 0],
            bio_dose[:, y_index, :],
            structures["PTV"][:, y_index, :],
            structures["GTV"][:, y_index, :],
            structures["SPINAL_CORD"][:, y_index, :],
            x_cm,
            z_cm,
            f"LQ-equivalent effective dose: axial-style slice at y = {float(axes_mm['y'][y_index]):.1f} mm",
            "x (cm)",
            "z (cm)",
            bio_vmax,
            "x-z",
        ),
        (
            axes[1, 1],
            bio_dose[:, :, z_index],
            structures["PTV"][:, :, z_index],
            structures["GTV"][:, :, z_index],
            structures["SPINAL_CORD"][:, :, z_index],
            x_cm,
            y_cm,
            f"LQ-equivalent effective dose: coronal slice at z = {float(axes_mm['z'][z_index]):.1f} mm",
            "x (cm)",
            "y (cm)",
            bio_vmax,
            "x-y",
        ),
    ]

    marker_style = dict(
        c="white",
        s=34,
        edgecolors="black",
        linewidths=0.6,
        zorder=6,
    )
    for ax, image, ptv_slice, gtv_slice, cord_slice, axis_a_cm, axis_b_cm, title, xlabel, ylabel, vmax, projection_mode in panels:
        im = ax.imshow(
            image.T,
            origin="lower",
            extent=[float(axis_a_cm[0]), float(axis_a_cm[-1]), float(axis_b_cm[0]), float(axis_b_cm[-1])],
            cmap="inferno",
            vmin=0.0,
            vmax=vmax,
        )
        ax.contour(axis_a_cm, axis_b_cm, ptv_slice.T.astype(float), levels=[0.5], colors=["cyan"], linewidths=1.2)
        ax.contour(axis_a_cm, axis_b_cm, gtv_slice.T.astype(float), levels=[0.5], colors=["magenta"], linewidths=1.0, linestyles="--")
        ax.contour(axis_a_cm, axis_b_cm, cord_slice.T.astype(float), levels=[0.5], colors=["white"], linewidths=1.0)
        if projection_mode == "x-z":
            ax.scatter(spot_centers_mm[:, 0] / 10.0, spot_centers_mm[:, 2] / 10.0, **marker_style)
        else:
            ax.scatter(spot_centers_mm[:, 0] / 10.0, spot_centers_mm[:, 1] / 10.0, **marker_style)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Gy")

    fig.suptitle(
        f"Physical and biology-aware dose slices through the lattice centroid plane (color clipped at body {clip_percentile:.1f}th percentile)",
        fontsize=12,
    )
    fig.savefig(out_file, dpi=dpi)
    plt.close(fig)


def plot_dvh_panels(
    out_file: Path,
    dose_axis: np.ndarray,
    physical_curves: Dict[str, np.ndarray],
    bio_curves: Dict[str, np.ndarray],
    *,
    dpi: int,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9.2), constrained_layout=True)
    panel_map = [
        ("Targets and serial OARs", ["PTV", "GTV", "SPINAL_CORD", "BRAINSTEM"]),
        ("Salivary glands and mandible", ["PAROTID_L", "PAROTID_R", "MANDIBLE"]),
        ("Endocrine structures", ["THYROID", "PARATHYROIDS"]),
        ("Intracranial structures", ["BRAIN", "BLOOD_BRAIN_BARRIER"]),
    ]
    colors = {
        "PTV": "tab:red",
        "GTV": "tab:orange",
        "SPINAL_CORD": "tab:blue",
        "BRAINSTEM": "tab:purple",
        "PAROTID_L": "tab:green",
        "PAROTID_R": "tab:olive",
        "MANDIBLE": "tab:brown",
        "THYROID": "tab:pink",
        "PARATHYROIDS": "goldenrod",
        "BRAIN": "tab:cyan",
        "BLOOD_BRAIN_BARRIER": "slateblue",
    }
    for ax, (title, names) in zip(axes.ravel(), panel_map):
        for name in names:
            ax.plot(dose_axis, physical_curves[name], color=colors[name], linestyle="-", label=f"{name} physical")
            ax.plot(dose_axis, bio_curves[name], color=colors[name], linestyle="--", label=f"{name} effective")
        ax.set_title(title)
        ax.set_xlabel("Dose / Gy")
        ax.set_ylabel("Volume receiving at least dose (%)")
        ax.set_ylim(0.0, 100.0)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=7, ncol=2)
    fig.savefig(out_file, dpi=dpi)
    plt.close(fig)


def plot_metric_bars(
    out_file: Path,
    physical_metrics: Dict[str, Dict[str, float]],
    bio_metrics: Dict[str, Dict[str, float]],
    *,
    dpi: int,
) -> None:
    categories = [
        ("PTV D95", physical_metrics["PTV"]["d95_gy"], bio_metrics["PTV"]["d95_gy"]),
        ("PTV D2", physical_metrics["PTV"]["d2_gy"], bio_metrics["PTV"]["d2_gy"]),
        ("Cord D2", physical_metrics["SPINAL_CORD"]["d2_gy"], bio_metrics["SPINAL_CORD"]["d2_gy"]),
        ("Brainstem D2", physical_metrics["BRAINSTEM"]["d2_gy"], bio_metrics["BRAINSTEM"]["d2_gy"]),
        ("Parotid R mean", physical_metrics["PAROTID_R"]["mean_gy"], bio_metrics["PAROTID_R"]["mean_gy"]),
        ("Thyroid mean", physical_metrics["THYROID"]["mean_gy"], bio_metrics["THYROID"]["mean_gy"]),
        ("Brain mean", physical_metrics["BRAIN"]["mean_gy"], bio_metrics["BRAIN"]["mean_gy"]),
    ]
    x = np.arange(len(categories))
    width = 0.38
    fig, ax = plt.subplots(figsize=(12.5, 5.0), constrained_layout=True)
    ax.bar(x - width / 2.0, [row[1] for row in categories], width=width, label="Physical", color="tab:blue")
    ax.bar(x + width / 2.0, [row[2] for row in categories], width=width, label="LQ-equivalent effective", color="tab:red")
    ax.set_xticks(x)
    ax.set_xticklabels([row[0] for row in categories], rotation=20, ha="right")
    ax.set_ylabel("Gy")
    ax.set_title("Detailed direct-plan metrics: physical versus biology-aware reinterpretation")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.savefig(out_file, dpi=dpi)
    plt.close(fig)


def write_markdown_report(
    out_file: Path,
    *,
    summary: Dict[str, object],
    physical_metrics: Dict[str, Dict[str, float]],
    bio_metrics: Dict[str, Dict[str, float]],
) -> None:
    lines = [
        "# Detailed Head-and-Neck Direct Plan With Biology-Aware Reinterpretation",
        "",
        "## Plan and Phantom",
        f"- Source count: `{summary['plan']['num_sources']}`",
        f"- Lattice vertices: `{summary['plan']['num_lattice_spots']}`",
        f"- AP base radius: `{summary['plan']['ap_radius_mm']:.2f} mm`",
        f"- Prescription normalization: PTV D95 scaled to `{summary['prescription_gy']:.2f} Gy`",
        f"- Phantom note: {summary['phantom']['anatomical_note']}",
        "",
        "## Vascular Sink Model",
        f"- Artery cytokine uptake: `{summary['biology']['sink_model']['artery_cyto_uptake']:.2f}`",
        f"- Vein cytokine uptake: `{summary['biology']['sink_model']['vein_cyto_uptake']:.2f}`",
        f"- Arterial vessel volume: `{summary['biology']['arterial_volume_cc']:.2f} cc`",
        f"- Venous vessel volume: `{summary['biology']['venous_volume_cc']:.2f} cc`",
        "",
        "## Physical Metrics",
        f"- PTV D95: `{physical_metrics['PTV']['d95_gy']:.2f} Gy` | D2: `{physical_metrics['PTV']['d2_gy']:.2f} Gy` | mean: `{physical_metrics['PTV']['mean_gy']:.2f} Gy`",
        f"- Spinal cord D2: `{physical_metrics['SPINAL_CORD']['d2_gy']:.2f} Gy`",
        f"- Brainstem D2: `{physical_metrics['BRAINSTEM']['d2_gy']:.2f} Gy`",
        f"- Right parotid mean: `{physical_metrics['PAROTID_R']['mean_gy']:.2f} Gy`",
        f"- Thyroid mean: `{physical_metrics['THYROID']['mean_gy']:.2f} Gy`",
        f"- Brain mean: `{physical_metrics['BRAIN']['mean_gy']:.2f} Gy`",
        "",
        "## Biology-Aware Effective Metrics",
        f"- PTV D95(eq): `{bio_metrics['PTV']['d95_gy']:.2f} Gy` | D2(eq): `{bio_metrics['PTV']['d2_gy']:.2f} Gy` | mean(eq): `{bio_metrics['PTV']['mean_gy']:.2f} Gy`",
        f"- Spinal cord D2(eq): `{bio_metrics['SPINAL_CORD']['d2_gy']:.2f} Gy`",
        f"- Brainstem D2(eq): `{bio_metrics['BRAINSTEM']['d2_gy']:.2f} Gy`",
        f"- Right parotid mean(eq): `{bio_metrics['PAROTID_R']['mean_gy']:.2f} Gy`",
        f"- Thyroid mean(eq): `{bio_metrics['THYROID']['mean_gy']:.2f} Gy`",
        f"- Brain mean(eq): `{bio_metrics['BRAIN']['mean_gy']:.2f} Gy`",
        "",
        "## Interpretation",
        "- The direct plan remains normalized to the same physical PTV D95 target, but the biology-aware reinterpretation broadens the apparent burden through nonlocal transport and temporal hazard accumulation.",
        "- Because sink tensors were derived directly from the explicit artery and vein masks, this branch tests anatomical washout using the detailed phantom's own vascular infrastructure rather than surrogate line vessels.",
    ]
    write_text_with_retries(out_file, "\n".join(lines) + "\n")


def main() -> int:
    args = parse_args()
    plan_run_root = args.plan_run_root.resolve()
    dose_csv = plan_run_root / "case" / "dosedata.csv"
    if not dose_csv.exists():
        raise FileNotFoundError(f"Could not find detailed direct-plan dose CSV: {dose_csv}")

    phase14_summary = load_phase14_summary(plan_run_root)
    phantom_args = build_args_from_summary(phase14_summary)
    phantom = build_detailed_plan_phantom(phantom_args)
    structures = phantom["structures"]
    axes_mm = phantom["axes_mm"]
    phantom_meta = phantom["meta"]

    out_dir = plan_run_root / args.output_dir_name
    out_dir.mkdir(parents=True, exist_ok=True)

    dose_raw, _ = load_topas_grid(dose_csv)
    voxel_volume_cc = float(phantom_meta["voxel_volume_cc"])
    ptv_raw_metrics = compute_structure_metrics(
        dose_raw,
        structures["PTV"],
        prescription_gy=float(args.prescription_gy),
        voxel_volume_cc=voxel_volume_cc,
    )
    raw_d95 = float(ptv_raw_metrics["d95_gy"])
    if raw_d95 <= 0.0:
        raise RuntimeError("Detailed direct-plan raw PTV D95 is non-positive; cannot normalize the plan.")
    physical_scale_factor = float(args.prescription_gy) / raw_d95
    physical_dose = dose_raw.astype(np.float32) * np.float32(physical_scale_factor)

    lq_survival = np.exp(-float(args.alpha) * physical_dose - float(args.beta) * physical_dose**2).astype(np.float32)
    uptake_tensor, m_type, m_oxygen, uptake_meta = build_anatomical_biology_tensors(args, structures)
    uptake_meta["arterial_volume_cc"] = float(np.count_nonzero(structures["ARTERIES"]) * voxel_volume_cc)
    uptake_meta["venous_volume_cc"] = float(np.count_nonzero(structures["VEINS"]) * voxel_volume_cc)

    hazard_grid = run_pde_temporal_integration(
        physical_dose,
        (float(phantom_meta["voxel_size_mm"][0]), float(phantom_meta["voxel_size_mm"][1]), float(phantom_meta["voxel_size_mm"][2])),
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
        progress_interval=50,
        verbose=True,
    )
    final_survival = calculate_phase7_survival(
        lq_survival,
        hazard_grid,
        physical_dose,
        (float(phantom_meta["voxel_size_mm"][0]), float(phantom_meta["voxel_size_mm"][1]), float(phantom_meta["voxel_size_mm"][2])),
        float(LOCKED_SCALING_FACTOR),
        weight_immune=float(W_IMMUNE),
        verbose=False,
    )
    effective_dose = calculate_effective_dose(final_survival, alpha=float(args.alpha), beta=float(args.beta))
    immune_penalty, icd_volume_cm3 = calculate_systemic_immune_penalty(
        physical_dose,
        (float(phantom_meta["voxel_size_mm"][0]), float(phantom_meta["voxel_size_mm"][1]), float(phantom_meta["voxel_size_mm"][2])),
    )

    metric_config = {
        "PTV": {"prescription": float(args.prescription_gy), "vxs": [6.0, 10.0]},
        "GTV": {"prescription": float(args.prescription_gy), "vxs": [6.0, 10.0]},
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
    physical_metrics: Dict[str, Dict[str, float]] = {}
    bio_metrics: Dict[str, Dict[str, float]] = {}
    for structure_name, config in metric_config.items():
        physical_metrics[structure_name] = compute_structure_metrics(
            physical_dose,
            structures[structure_name],
            prescription_gy=config["prescription"],
            voxel_volume_cc=voxel_volume_cc,
            volume_thresholds_gy=config["vxs"],
        )
        bio_metrics[structure_name] = compute_structure_metrics(
            effective_dose,
            structures[structure_name],
            prescription_gy=config["prescription"],
            voxel_volume_cc=voxel_volume_cc,
            volume_thresholds_gy=config["vxs"],
        )

    dose_axis = np.linspace(0.0, max(float(np.max(physical_dose)), float(np.max(effective_dose))) * 1.05, 400)
    physical_dvhs = {name: compute_dvh(physical_dose[structures[name]], dose_axis) for name in metric_config}
    bio_dvhs = {name: compute_dvh(effective_dose[structures[name]], dose_axis) for name in metric_config}

    dvh_rows: List[Dict[str, object]] = []
    for idx, dose_value in enumerate(dose_axis):
        row: Dict[str, object] = {"dose_gy": float(dose_value)}
        for structure_name in metric_config:
            row[f"{structure_name}_physical_pct"] = float(physical_dvhs[structure_name][idx])
            row[f"{structure_name}_effective_pct"] = float(bio_dvhs[structure_name][idx])
        dvh_rows.append(row)

    delta_rows: List[Dict[str, object]] = []
    for structure_name in metric_config:
        for metric_name, physical_value in physical_metrics[structure_name].items():
            if metric_name not in bio_metrics[structure_name]:
                continue
            bio_value = float(bio_metrics[structure_name][metric_name])
            physical_value = float(physical_value)
            delta_rows.append(
                {
                    "structure": structure_name,
                    "metric": metric_name,
                    "physical": physical_value,
                    "effective": bio_value,
                    "delta_abs": bio_value - physical_value,
                    "delta_pct_of_physical": ((bio_value - physical_value) / physical_value * 100.0) if abs(physical_value) > 1e-9 else np.nan,
                }
            )

    save_csv(
        [{"domain": "physical", "structure": name, **{key: float(value) for key, value in values.items()}} for name, values in physical_metrics.items()],
        out_dir / "physical_plan_metrics.csv",
    )
    save_csv(
        [{"domain": "effective", "structure": name, **{key: float(value) for key, value in values.items()}} for name, values in bio_metrics.items()],
        out_dir / "effective_plan_metrics.csv",
    )
    save_csv(dvh_rows, out_dir / "dvh_curves.csv")
    save_csv(delta_rows, out_dir / "physical_vs_effective_metric_deltas.csv")

    plot_treatment_plan(out_dir / "figure1_treatment_plan_with_vessels.png", axes_mm, structures, phase14_summary["plan"], dpi=int(args.dpi))
    plot_sink_field(out_dir / "figure2_anatomical_cytokine_sink_field.png", axes_mm, uptake_tensor, structures, dpi=int(args.dpi))
    plot_dose_slices(
        out_dir / "figure3_physical_vs_effective_dose.png",
        axes_mm,
        physical_dose,
        effective_dose,
        structures,
        phase14_summary["plan"],
        clip_percentile=99.5,
        dpi=int(args.dpi),
    )
    plot_dvh_panels(out_dir / "figure4_physical_vs_effective_dvhs.png", dose_axis, physical_dvhs, bio_dvhs, dpi=int(args.dpi))
    plot_metric_bars(out_dir / "figure5_key_metric_comparison.png", physical_metrics, bio_metrics, dpi=int(args.dpi))

    summary = {
        "plan_run_root": str(plan_run_root),
        "prescription_gy": float(args.prescription_gy),
        "physical_scale_factor": float(physical_scale_factor),
        "plan": phase14_summary["plan"],
        "phantom": phantom_meta,
        "biology": {
            "locked_D_cyto": LOCKED_D_CYTO,
            "locked_lambda_cyto": LOCKED_LAMBDA_CYTO,
            "locked_gamma": LOCKED_GAMMA,
            "locked_scaling_factor": LOCKED_SCALING_FACTOR,
            "w_ros": W_ROS,
            "w_cyto": W_CYTO,
            "w_immune": W_IMMUNE,
            "immune_scalar": float(immune_penalty),
            "icd_volume_cm3": float(icd_volume_cm3),
            "sink_model": uptake_meta,
            "arterial_volume_cc": float(uptake_meta["arterial_volume_cc"]),
            "venous_volume_cc": float(uptake_meta["venous_volume_cc"]),
        },
        "physical_metrics": physical_metrics,
        "effective_metrics": bio_metrics,
    }
    write_text_with_retries(out_dir / "phase15_detailed_headneck_bioaware_summary.json", json.dumps(summary, indent=2))
    write_markdown_report(
        out_dir / "phase15_detailed_headneck_bioaware_summary.md",
        summary=summary,
        physical_metrics=physical_metrics,
        bio_metrics=bio_metrics,
    )

    print("=== PHASE 15 DETAILED HEAD-AND-NECK BIOAWARE ANALYSIS COMPLETE ===")
    print(f"PTV physical D95: {physical_metrics['PTV']['d95_gy']:.2f} Gy")
    print(f"PTV effective D95: {bio_metrics['PTV']['d95_gy']:.2f} Gy")
    print(f"Spinal cord physical D2: {physical_metrics['SPINAL_CORD']['d2_gy']:.2f} Gy")
    print(f"Spinal cord effective D2: {bio_metrics['SPINAL_CORD']['d2_gy']:.2f} Gy")
    print(f"Right parotid physical mean: {physical_metrics['PAROTID_R']['mean_gy']:.2f} Gy")
    print(f"Right parotid effective mean: {bio_metrics['PAROTID_R']['mean_gy']:.2f} Gy")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
