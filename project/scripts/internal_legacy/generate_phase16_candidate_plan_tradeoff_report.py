#!/usr/bin/env python3
"""Compare all Phase 16 SFRT candidate plans using physical and biology-aware tradeoffs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Sequence

import numpy as np

from analyze_topas_outputs import load_topas_grid
from build_asymmetric_sweep import write_text_with_retries
from bystander_multispecies_pde_solver import (
    calculate_effective_dose,
    calculate_phase7_survival,
    run_pde_temporal_integration,
)
from run_phase13_headneck_voxel_lattice import compute_dvh, compute_structure_metrics
from run_phase14_detailed_headneck_voxel_lattice import build_detailed_plan_phantom
from run_phase15_detailed_headneck_bioaware import (
    D_ROS,
    EMAX_CYTO,
    EMAX_ROS,
    LAMBDA_ROS,
    LOCKED_D_CYTO,
    LOCKED_GAMMA,
    LOCKED_LAMBDA_CYTO,
    LOCKED_SCALING_FACTOR,
    W_CYTO,
    W_IMMUNE,
    W_ROS,
    build_anatomical_biology_tensors,
    build_args_from_summary,
    load_phase14_summary,
)
from run_phase16_bio_guided_lattice_optimization import compute_plan_objective

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception as exc:  # pragma: no cover
    raise RuntimeError("matplotlib is required for candidate-plan comparison figures.") from exc


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
PLAN_COLORS = {
    "placement_01_baseline_direct": "#1f77b4",
    "placement_02_feedback_02": "#d62728",
    "placement_03_feedback_03": "#2ca02c",
    "placement_04_feedback_04": "#9467bd",
    "placement_05_feedback_05": "#ff7f0e",
}


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description=(
            "Generate figures and ranking tables for all SFRT candidate plans in the "
            "Phase 16 biology-guided optimization run."
        )
    )
    parser.add_argument(
        "--phase14-root",
        type=Path,
        default=root / "runs" / "linac_6mv_headneck_detailed_voxel_lattice_sfrt_directplan",
        help="Run root containing the phase14 detailed direct-plan summary.",
    )
    parser.add_argument(
        "--bioopt-root",
        type=Path,
        default=root / "runs" / "linac_6mv_headneck_detailed_voxel_lattice_sfrt_bioopt_v2",
        help="Phase 16 run root containing placement_* candidate plans.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Optional override for the candidate-plan comparison output directory.",
    )
    parser.add_argument("--prescription-gy", type=float, default=6.0)
    parser.add_argument("--alpha", type=float, default=0.03)
    parser.add_argument("--beta", type=float, default=0.003)
    parser.add_argument("--pde-steps", type=int, default=400)
    parser.add_argument("--pde-dt", type=float, default=0.12)
    parser.add_argument("--tumor-cytokine-multiplier", type=float, default=2.0)
    parser.add_argument("--hypoxic-ros-scale", type=float, default=0.12)
    parser.add_argument("--hypoxic-cytokine-multiplier", type=float, default=2.7)
    parser.add_argument("--artery-ros-uptake", type=float, default=0.05)
    parser.add_argument("--artery-cyto-uptake", type=float, default=0.70)
    parser.add_argument("--vein-ros-uptake", type=float, default=0.05)
    parser.add_argument("--vein-cyto-uptake", type=float, default=0.90)
    parser.add_argument("--dpi", type=int, default=240)
    return parser.parse_args()


def resolve_dose_csv(placement_dir: Path) -> Path:
    case_csv = placement_dir / "case" / "dosedata.csv"
    if case_csv.exists() and case_csv.stat().st_size > 0:
        return case_csv
    reused = placement_dir / "analysis" / "reused_dose_source.txt"
    if reused.exists():
        source = Path(reused.read_text(encoding="utf-8").strip())
        if source.exists():
            return source
    raise FileNotFoundError(f"No usable dose CSV found for {placement_dir}")


def plot_plan_layouts(
    out_file: Path,
    axes_mm: Dict[str, np.ndarray],
    structures: Dict[str, np.ndarray],
    cases: Sequence[Dict[str, object]],
    *,
    dpi: int,
) -> None:
    x_cm = axes_mm["x"] / 10.0
    y_cm = axes_mm["y"] / 10.0
    body_xy = np.any(structures["BODY"], axis=2)
    ptv_xy = np.any(structures["PTV"], axis=2)
    gtv_xy = np.any(structures["GTV"], axis=2)

    fig, axes = plt.subplots(1, len(cases), figsize=(4.0 * len(cases), 4.8), constrained_layout=True)
    if len(cases) == 1:
        axes = [axes]
    for ax, case in zip(axes, cases):
        label = str(case["placement_name"])
        display = label.replace("placement_", "P").replace("_", " ")
        spots = np.asarray(case["spot_centers_mm"], dtype=np.float32)
        extent = [float(x_cm[0]), float(x_cm[-1]), float(y_cm[0]), float(y_cm[-1])]
        ax.imshow(body_xy.T, origin="lower", cmap="Greys", extent=extent, alpha=0.90)
        ax.contour(x_cm, y_cm, ptv_xy.T.astype(float), levels=[0.5], colors=["cyan"], linewidths=1.2)
        ax.contour(x_cm, y_cm, gtv_xy.T.astype(float), levels=[0.5], colors=["magenta"], linewidths=1.0, linestyles="--")
        ax.scatter(
            spots[:, 0] / 10.0,
            spots[:, 1] / 10.0,
            c=PLAN_COLORS.get(label, "yellow"),
            s=40,
            edgecolors="black",
            linewidths=0.5,
            zorder=5,
        )
        for idx, (sx, sy, _) in enumerate(spots, start=1):
            ax.text(sx / 10.0, sy / 10.0, str(idx), color="white", fontsize=8, ha="center", va="center")
        ax.set_title(display)
        ax.set_xlabel("x (cm)")
        ax.set_ylabel("y (cm)")
    fig.suptitle("Candidate SFRT lattice placements in the heterogeneous head-and-neck phantom", fontsize=13)
    fig.savefig(out_file, dpi=dpi)
    plt.close(fig)


def plot_dvh_family(
    out_file: Path,
    dose_axis: np.ndarray,
    cases: Sequence[Dict[str, object]],
    dvh_key: str,
    *,
    dpi: int,
    title: str,
) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(14.0, 8.8), constrained_layout=True)
    axes = axes.ravel()
    for ax, structure in zip(axes, COMPARE_STRUCTURES):
        for case in cases:
            label = str(case["placement_name"])
            rank_text = f"B{int(case['bio_rank'])}"
            display = label.replace("placement_", "P").replace("_", " ")
            linestyle = "-" if bool(case["accepted"]) else "--"
            linewidth = 2.2 if int(case["bio_rank"]) == 1 else 1.6
            curve = case[dvh_key][structure]
            ax.plot(
                dose_axis,
                curve,
                color=PLAN_COLORS.get(label, None),
                linestyle=linestyle,
                linewidth=linewidth,
                label=f"{display} ({rank_text})",
            )
        ax.set_title(structure.replace("_", " "))
        ax.set_xlabel("Dose (Gy)")
        ax.set_ylabel("Volume (%)")
        ax.set_xlim(0.0, float(dose_axis[-1]))
        ax.set_ylim(0.0, 100.0)
        ax.grid(alpha=0.25)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=8, framealpha=0.95)
    fig.suptitle(title, fontsize=13)
    fig.savefig(out_file, dpi=dpi)
    plt.close(fig)


def plot_survival_maps(
    out_file: Path,
    axes_mm: Dict[str, np.ndarray],
    structures: Dict[str, np.ndarray],
    cases: Sequence[Dict[str, object]],
    *,
    dpi: int,
) -> None:
    x_cm = axes_mm["x"] / 10.0
    z_cm = axes_mm["z"] / 10.0
    fig, axes = plt.subplots(1, len(cases), figsize=(4.2 * len(cases), 4.8), constrained_layout=True)
    if len(cases) == 1:
        axes = [axes]
    images = []
    for ax, case in zip(axes, cases):
        y_mm = float(np.mean(np.asarray(case["spot_centers_mm"], dtype=np.float32)[:, 1]))
        y_idx = int(np.argmin(np.abs(axes_mm["y"] - y_mm)))
        surv_slice = case["final_survival"][:, y_idx, :]
        ptv_slice = structures["PTV"][:, y_idx, :]
        gtv_slice = structures["GTV"][:, y_idx, :]
        im = ax.imshow(
            surv_slice.T,
            origin="lower",
            extent=[float(x_cm[0]), float(x_cm[-1]), float(z_cm[0]), float(z_cm[-1])],
            cmap="magma_r",
            vmin=0.0,
            vmax=1.0,
        )
        images.append(im)
        ax.contour(x_cm, z_cm, ptv_slice.T.astype(float), levels=[0.5], colors=["cyan"], linewidths=1.0)
        ax.contour(x_cm, z_cm, gtv_slice.T.astype(float), levels=[0.5], colors=["magenta"], linewidths=0.9, linestyles="--")
        spots = np.asarray(case["spot_centers_mm"], dtype=np.float32)
        ax.scatter(spots[:, 0] / 10.0, spots[:, 2] / 10.0, c="white", s=28, edgecolors="black", linewidths=0.5)
        ax.set_title(f"{case['placement_name'].replace('placement_', 'P').replace('_', ' ')}\nBio rank {case['bio_rank']}")
        ax.set_xlabel("x (cm)")
        ax.set_ylabel("z (cm)")
    cbar = fig.colorbar(images[-1], ax=axes, fraction=0.018, pad=0.02)
    cbar.set_label("Final survival")
    fig.suptitle("Biology-aware survival maps through each plan's lattice-centroid plane", fontsize=13)
    fig.savefig(out_file, dpi=dpi)
    plt.close(fig)


def plot_tradeoff_scores(
    out_file: Path,
    cases: Sequence[Dict[str, object]],
    *,
    dpi: int,
) -> None:
    labels = [case["placement_name"].replace("placement_", "P").replace("_", " ") for case in cases]
    x = np.arange(len(cases))
    physical_scores = np.array([float(case["physical_score"]) for case in cases], dtype=np.float32)
    bio_scores = np.array([float(case["bio_score"]) for case in cases], dtype=np.float32)
    fig, ax = plt.subplots(figsize=(11.5, 5.8), constrained_layout=True)
    width = 0.36
    ax.bar(x - width / 2, physical_scores, width=width, color="#A0AEC0", label="Physical tradeoff score")
    ax.bar(x + width / 2, bio_scores, width=width, color="#D53F8C", label="Biological tradeoff score")
    for idx, case in enumerate(cases):
        ax.text(x[idx] + width / 2, bio_scores[idx] + 0.12, f"B{case['bio_rank']}", ha="center", va="bottom", fontsize=9)
        if bool(case["accepted"]):
            ax.text(x[idx], max(physical_scores[idx], bio_scores[idx]) + 0.45, "accepted", ha="center", va="bottom", fontsize=8, color="#2F855A")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylabel("Tradeoff score")
    ax.set_title("Candidate-plan ranking by physical-only versus biology-aware tradeoff")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(framealpha=0.95)
    fig.savefig(out_file, dpi=dpi)
    plt.close(fig)


def build_metric_summary_row(case: Dict[str, object]) -> Dict[str, object]:
    pm = case["physical_metrics"]
    em = case["effective_metrics"]
    sm = case["survival_metrics"]
    return {
        "bio_rank": int(case["bio_rank"]),
        "physical_rank": int(case["physical_rank"]),
        "placement_name": str(case["placement_name"]),
        "accepted": "yes" if bool(case["accepted"]) else "no",
        "physical_score": float(case["physical_score"]),
        "biological_score": float(case["bio_score"]),
        "ptv_d95_physical_gy": float(pm["PTV"]["d95_gy"]),
        "ptv_d95_effective_gy": float(em["PTV"]["d95_gy"]),
        "gtv_d95_effective_gy": float(em["GTV"]["d95_gy"]),
        "spinal_cord_d2_effective_gy": float(em["SPINAL_CORD"]["d2_gy"]),
        "brainstem_d2_effective_gy": float(em["BRAINSTEM"]["d2_gy"]),
        "parotid_r_mean_effective_gy": float(em["PAROTID_R"]["mean_gy"]),
        "thyroid_mean_effective_gy": float(em["THYROID"]["mean_gy"]),
        "brain_mean_effective_gy": float(em["BRAIN"]["mean_gy"]),
        "ptv_mean_survival": float(sm["PTV_mean_survival"]),
        "gtv_mean_survival": float(sm["GTV_mean_survival"]),
        "oar_mean_survival_parotid_r": float(sm["PAROTID_R_mean_survival"]),
        "spots_mm": "; ".join(f"({x:.1f},{y:.1f},{z:.1f})" for x, y, z in case["spot_centers_mm"]),
    }


def write_markdown_table(out_file: Path, rows: Sequence[Dict[str, object]]) -> None:
    headers = [
        "Bio rank",
        "Physical rank",
        "Placement",
        "Accepted",
        "Bio score",
        "Phys score",
        "PTV D95 phys",
        "PTV D95 eff",
        "GTV D95 eff",
        "Cord D2 eff",
        "Parotid R mean eff",
        "Thyroid mean eff",
        "Brain mean eff",
        "PTV mean survival",
    ]
    lines = [
        "# Candidate-plan biological tradeoff ranking",
        "",
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["bio_rank"]),
                    str(row["physical_rank"]),
                    str(row["placement_name"]),
                    str(row["accepted"]),
                    f"{float(row['biological_score']):.2f}",
                    f"{float(row['physical_score']):.2f}",
                    f"{float(row['ptv_d95_physical_gy']):.2f}",
                    f"{float(row['ptv_d95_effective_gy']):.2f}",
                    f"{float(row['gtv_d95_effective_gy']):.2f}",
                    f"{float(row['spinal_cord_d2_effective_gy']):.2f}",
                    f"{float(row['parotid_r_mean_effective_gy']):.2f}",
                    f"{float(row['thyroid_mean_effective_gy']):.2f}",
                    f"{float(row['brain_mean_effective_gy']):.2f}",
                    f"{float(row['ptv_mean_survival']):.4f}",
                ]
            )
            + " |"
        )
    write_text_with_retries(out_file, "\n".join(lines) + "\n")


def process_case(
    *,
    placement_dir: Path,
    placement_meta: Dict[str, object],
    args: argparse.Namespace,
    phantom: Dict[str, object],
) -> Dict[str, object]:
    structures = phantom["structures"]
    axes_mm = phantom["axes_mm"]
    voxel_volume_cc = float(phantom["meta"]["voxel_volume_cc"])
    dose_csv = resolve_dose_csv(placement_dir)
    dose_raw, _ = load_topas_grid(dose_csv)
    ptv_raw = compute_structure_metrics(
        dose_raw,
        structures["PTV"],
        prescription_gy=float(args.prescription_gy),
        voxel_volume_cc=voxel_volume_cc,
    )
    raw_d95 = float(ptv_raw["d95_gy"])
    if raw_d95 <= 0.0:
        raise RuntimeError(f"{placement_dir.name}: non-positive raw PTV D95.")
    scale = float(args.prescription_gy) / raw_d95
    physical_dose = dose_raw.astype(np.float32) * np.float32(scale)

    uptake_tensor, m_type, m_oxygen, _ = build_anatomical_biology_tensors(args, structures)
    lq_survival = np.exp(-float(args.alpha) * physical_dose - float(args.beta) * physical_dose**2).astype(np.float32)
    vz = tuple(float(v) for v in phantom["meta"]["voxel_size_mm"])
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

    physical_metrics: Dict[str, Dict[str, float]] = {}
    effective_metrics: Dict[str, Dict[str, float]] = {}
    for structure_name, config in METRIC_CONFIG.items():
        physical_metrics[structure_name] = compute_structure_metrics(
            physical_dose,
            structures[structure_name],
            prescription_gy=config["prescription"],
            voxel_volume_cc=voxel_volume_cc,
            volume_thresholds_gy=config["vxs"],
        )
        effective_metrics[structure_name] = compute_structure_metrics(
            effective_dose,
            structures[structure_name],
            prescription_gy=config["prescription"],
            voxel_volume_cc=voxel_volume_cc,
            volume_thresholds_gy=config["vxs"],
        )
    physical_score, _ = compute_plan_objective(physical_metrics)
    bio_score, _ = compute_plan_objective(effective_metrics)

    survival_metrics = {
        "PTV_mean_survival": float(np.mean(final_survival[structures["PTV"]])),
        "GTV_mean_survival": float(np.mean(final_survival[structures["GTV"]])),
        "PAROTID_R_mean_survival": float(np.mean(final_survival[structures["PAROTID_R"]])),
    }
    return {
        "placement_id": int(placement_meta["placement_id"]),
        "placement_name": str(placement_dir.name),
        "accepted": bool(placement_meta.get("accepted", False)),
        "spot_centers_mm": [tuple(map(float, row)) for row in placement_meta["spot_centers_mm"]],
        "physical_dose": physical_dose,
        "effective_dose": effective_dose,
        "final_survival": final_survival,
        "physical_metrics": physical_metrics,
        "effective_metrics": effective_metrics,
        "survival_metrics": survival_metrics,
        "physical_score": float(physical_score),
        "bio_score": float(bio_score),
    }


def main() -> int:
    args = parse_args()
    bioopt_root = args.bioopt_root.resolve()
    out_dir = (args.out_dir or (bioopt_root / "candidate_plan_tradeoff_report")).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    phase14 = load_phase14_summary(args.phase14_root.resolve())
    phantom_args = build_args_from_summary(phase14)
    phantom = build_detailed_plan_phantom(phantom_args)
    axes_mm = phantom["axes_mm"]
    structures = phantom["structures"]

    optimization = json.loads((bioopt_root / "optimization_results.json").read_text(encoding="utf-8"))
    placements = sorted(optimization["placements"], key=lambda row: int(row["placement_id"]))
    cases = [
        process_case(
            placement_dir=bioopt_root / f"placement_{int(meta['placement_id']):02d}_{meta['placement_name']}",
            placement_meta=meta,
            args=args,
            phantom=phantom,
        )
        for meta in placements
    ]

    bio_order = sorted(cases, key=lambda case: float(case["bio_score"]), reverse=True)
    physical_order = sorted(cases, key=lambda case: float(case["physical_score"]), reverse=True)
    for rank, case in enumerate(bio_order, start=1):
        case["bio_rank"] = rank
    for rank, case in enumerate(physical_order, start=1):
        case["physical_rank"] = rank
    cases = sorted(cases, key=lambda case: int(case["placement_id"]))

    body_mask = structures["BODY"]
    phys_max = max(float(np.max(case["physical_dose"][body_mask])) for case in cases) * 1.02
    eff_max = max(float(np.max(case["effective_dose"][body_mask])) for case in cases) * 1.02
    dose_axis_phys = np.linspace(0.0, phys_max, 500)
    dose_axis_eff = np.linspace(0.0, eff_max, 500)
    for case in cases:
        case["physical_dvhs"] = {
            structure: compute_dvh(case["physical_dose"][structures[structure]], dose_axis_phys)
            for structure in COMPARE_STRUCTURES
        }
        case["effective_dvhs"] = {
            structure: compute_dvh(case["effective_dose"][structures[structure]], dose_axis_eff)
            for structure in COMPARE_STRUCTURES
        }

    plot_plan_layouts(out_dir / "figure1_candidate_plan_layouts.png", axes_mm, structures, cases, dpi=int(args.dpi))
    plot_dvh_family(
        out_dir / "figure2_candidate_plan_physical_dvhs.png",
        dose_axis_phys,
        cases,
        "physical_dvhs",
        dpi=int(args.dpi),
        title="Physical DVHs across SFRT candidate plans",
    )
    plot_dvh_family(
        out_dir / "figure3_candidate_plan_effective_dvhs.png",
        dose_axis_eff,
        cases,
        "effective_dvhs",
        dpi=int(args.dpi),
        title="Biology-aware LQ-equivalent effective-dose DVHs across SFRT candidate plans",
    )
    plot_survival_maps(
        out_dir / "figure4_candidate_plan_survival_maps.png",
        axes_mm,
        structures,
        cases,
        dpi=int(args.dpi),
    )
    plot_tradeoff_scores(
        out_dir / "figure5_candidate_plan_tradeoff_scores.png",
        cases,
        dpi=int(args.dpi),
    )

    rows = [build_metric_summary_row(case) for case in sorted(cases, key=lambda case: int(case["bio_rank"]))]
    with (out_dir / "candidate_plan_ranking.csv").open("w", encoding="utf-8", newline="") as handle:
        import csv

        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    write_markdown_table(out_dir / "candidate_plan_ranking.md", rows)
    write_text_with_retries(
        out_dir / "candidate_plan_ranking.json",
        json.dumps(
            {
                "best_biological_plan": str(min(cases, key=lambda case: case["bio_rank"])["placement_name"]),
                "cases": rows,
                "figures": [
                    str(out_dir / "figure1_candidate_plan_layouts.png"),
                    str(out_dir / "figure2_candidate_plan_physical_dvhs.png"),
                    str(out_dir / "figure3_candidate_plan_effective_dvhs.png"),
                    str(out_dir / "figure4_candidate_plan_survival_maps.png"),
                    str(out_dir / "figure5_candidate_plan_tradeoff_scores.png"),
                ],
            },
            indent=2,
        ),
    )

    print(f"=== CANDIDATE PLAN TRADEOFF REPORT WRITTEN TO {out_dir} ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
