#!/usr/bin/env python3
"""Evaluate whether the biology model improves fractionated plan quality.

This script compares *physical* vs *bio-effective* (LQ-equivalent effective dose) metrics
for a set of per-fraction SFRT plans (e.g. Phase-16 placements treated as fractions).

It reports:
- Per-fraction target coverage (PTV/GTV D95, V95, etc.)
- Per-fraction OAR burden (mean and D2 for key OARs)
- Peak-valley preservation proxy (PVDR ~ D2/D98 within target)
- Hotspot metrics (BODY Dmax, PTV D2, OAR D2)
- Consistency across fractions (mean, std, coefficient of variation)
- Cumulative (summed) physical dose and two cumulative effective-dose proxies:
    (A) effective(sum physical)  [biology applied to accumulated physical dose]
    (B) sum(effective per fraction) [upper bound-ish; not strictly biophysical]

Outputs are written under runs/.../fractionated_eval/.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

from analyze_topas_outputs import load_topas_grid
from bystander_multispecies_pde_solver import (
    calculate_effective_dose,
    calculate_phase7_survival,
    run_pde_temporal_integration,
)
from run_phase13_headneck_voxel_lattice import compute_dvh, compute_structure_metrics
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
)

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception as exc:  # pragma: no cover
    raise RuntimeError("matplotlib is required") from exc


STRUCTURES_TARGET = ("PTV", "GTV")
STRUCTURES_OAR = ("SPINAL_CORD", "BRAINSTEM", "PAROTID_R", "PAROTID_L", "THYROID", "BRAIN", "MANDIBLE")
STRUCTURES_ALL = STRUCTURES_TARGET + STRUCTURES_OAR + ("BODY",)


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--phase14-root",
        type=Path,
        default=root / "runs" / "linac_6mv_headneck_detailed_voxel_lattice_sfrt_directplan",
        help="Run root containing phase14_detailed_headneck_summary.json (phantom definition).",
    )
    p.add_argument(
        "--bioopt-root",
        type=Path,
        default=root / "runs" / "linac_6mv_headneck_detailed_voxel_lattice_sfrt_bioopt",
        help="Phase-16 run root containing placement_* folders.",
    )
    p.add_argument(
        "--fractions",
        nargs="+",
        default=[
            "placement_01_baseline_direct",
            "placement_02_feedback_02",
            "placement_03_feedback_03",
            "placement_04_feedback_04",
            "placement_05_feedback_05",
        ],
        help="Placement subfolders to treat as sequential fractions.",
    )
    p.add_argument("--out-dir", type=Path, default=None, help="Output directory (default: bioopt-root/fractionated_eval).")
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
    p.add_argument("--dvh-bins", type=int, default=400)
    p.add_argument("--dpi", type=int, default=220)
    return p.parse_args()


def pvdr_from_metrics(metrics: Dict[str, float]) -> float:
    d2 = float(metrics.get("d2_gy", 0.0))
    d98 = float(metrics.get("d98_gy", 0.0))
    if d98 <= 1e-6:
        return float("nan")
    return float(d2 / d98)


def compute_effective_dose(
    *,
    physical_dose: np.ndarray,
    voxel_size_mm: Tuple[float, float, float],
    structures: Dict[str, np.ndarray],
    bio_args: SimpleNamespace,
    alpha: float,
    beta: float,
    steps: int,
    dt: float,
) -> np.ndarray:
    lq_survival = np.exp(-float(alpha) * physical_dose - float(beta) * physical_dose**2).astype(np.float32)
    uptake_tensor, m_type, m_oxygen, _ = build_anatomical_biology_tensors(bio_args, structures)
    hazard = run_pde_temporal_integration(
        physical_dose,
        voxel_size_mm,
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
        steps=int(steps),
        dt=float(dt),
        progress_interval=9999,
        verbose=False,
    )
    final_survival = calculate_phase7_survival(
        lq_survival,
        hazard,
        physical_dose,
        voxel_size_mm,
        float(LOCKED_SCALING_FACTOR),
        weight_immune=float(W_IMMUNE),
        verbose=False,
    )
    return calculate_effective_dose(final_survival, alpha=float(alpha), beta=float(beta))


def compute_metrics_for_domain(
    dose: np.ndarray,
    *,
    structures: Dict[str, np.ndarray],
    voxel_volume_cc: float,
    prescription_gy: float,
) -> Dict[str, Dict[str, float]]:
    metrics: Dict[str, Dict[str, float]] = {}
    for name in STRUCTURES_ALL:
        if name not in structures:
            continue
        metrics[name] = compute_structure_metrics(
            dose,
            structures[name],
            prescription_gy=float(prescription_gy) if name in STRUCTURES_TARGET else None,
            voxel_volume_cc=float(voxel_volume_cc),
            volume_thresholds_gy=[prescription_gy, 10.0] if name in STRUCTURES_TARGET else [5.0, 10.0],
        )
    return metrics


def summarize_fraction_row(
    fraction_idx: int,
    label: str,
    physical_metrics: Dict[str, Dict[str, float]],
    effective_metrics: Dict[str, Dict[str, float]],
) -> Dict[str, object]:
    ptv_pvdr_phys = pvdr_from_metrics(physical_metrics["PTV"])
    ptv_pvdr_eff = pvdr_from_metrics(effective_metrics["PTV"])
    row: Dict[str, object] = {
        "fraction": int(fraction_idx),
        "label": label,
        "PTV_D95_phys": float(physical_metrics["PTV"]["d95_gy"]),
        "PTV_D95_eff": float(effective_metrics["PTV"]["d95_gy"]),
        "GTV_D95_eff": float(effective_metrics["GTV"]["d95_gy"]),
        "PTV_PVDR_phys(D2/D98)": float(ptv_pvdr_phys),
        "PTV_PVDR_eff(D2/D98)": float(ptv_pvdr_eff),
        "CORD_D2_phys": float(physical_metrics["SPINAL_CORD"]["d2_gy"]),
        "CORD_D2_eff": float(effective_metrics["SPINAL_CORD"]["d2_gy"]),
        "BRAINSTEM_D2_eff": float(effective_metrics["BRAINSTEM"]["d2_gy"]),
        "PAROTID_R_mean_eff": float(effective_metrics["PAROTID_R"]["mean_gy"]),
        "THYROID_mean_eff": float(effective_metrics["THYROID"]["mean_gy"]),
        "BODY_Dmax_phys": float(physical_metrics["BODY"]["dmax_gy"]),
        "BODY_Dmax_eff": float(effective_metrics["BODY"]["dmax_gy"]),
    }
    return row


def write_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    if not rows:
        raise ValueError("No rows to write.")
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def consistency_table(rows: Sequence[Dict[str, object]], keys: Sequence[str]) -> List[Dict[str, object]]:
    table: List[Dict[str, object]] = []
    for key in keys:
        values = np.array([float(r[key]) for r in rows], dtype=np.float64)
        mean = float(np.mean(values))
        std = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
        cv = float(std / mean) if abs(mean) > 1e-9 else float("nan")
        table.append({"metric": key, "mean": mean, "std": std, "cv": cv, "min": float(values.min()), "max": float(values.max())})
    return table


def plot_consistency_bars(out_file: Path, table: Sequence[Dict[str, object]], *, dpi: int) -> None:
    metrics = [row["metric"] for row in table]
    means = [float(row["mean"]) for row in table]
    stds = [float(row["std"]) for row in table]
    x = np.arange(len(metrics))
    fig, ax = plt.subplots(figsize=(12.5, 5.2), constrained_layout=True)
    ax.bar(x, means, yerr=stds, capsize=4, color="tab:blue", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, rotation=20, ha="right")
    ax.set_ylabel("Gy (or ratio)")
    ax.set_title("Across-fraction consistency (mean ± 1σ)")
    ax.grid(axis="y", alpha=0.25)
    fig.savefig(out_file, dpi=dpi)
    plt.close(fig)


def plot_overlay_target_dvhs(
    out_file: Path,
    dose_axis: np.ndarray,
    dvhs_phys: List[Dict[str, np.ndarray]],
    dvhs_eff: List[Dict[str, np.ndarray]],
    labels: Sequence[str],
    *,
    dpi: int,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.6), constrained_layout=True)
    for idx, label in enumerate(labels):
        axes[0].plot(dose_axis, dvhs_phys[idx]["PTV"], alpha=0.6, linewidth=1.3, label=label)
        axes[1].plot(dose_axis, dvhs_eff[idx]["PTV"], alpha=0.6, linewidth=1.3, label=label)
    axes[0].set_title("PTV physical DVHs across fractions")
    axes[1].set_title("PTV effective DVHs across fractions")
    for ax in axes:
        ax.set_xlabel("Dose (Gy)")
        ax.set_ylabel("Volume (%)")
        ax.set_ylim(0.0, 100.0)
        ax.grid(alpha=0.25)
    axes[1].legend(fontsize=7, ncol=2)
    fig.savefig(out_file, dpi=dpi)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    phase14 = load_phase14_summary(args.phase14_root.resolve())
    phantom_args = build_args_from_summary(phase14)
    phantom = build_detailed_plan_phantom(phantom_args)
    structures = phantom["structures"]
    phantom_meta = phantom["meta"]
    voxel_volume_cc = float(phantom_meta["voxel_volume_cc"])
    voxel_size_mm = tuple(float(v) for v in phantom_meta["voxel_size_mm"])

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
    out_dir = (args.out_dir or (bioopt / "fractionated_eval")).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    fraction_rows: List[Dict[str, object]] = []
    dvhs_phys: List[Dict[str, np.ndarray]] = []
    dvhs_eff: List[Dict[str, np.ndarray]] = []
    physical_sum: np.ndarray | None = None
    effective_sum_of_fractions: np.ndarray | None = None

    # Determine a common DVH axis from an initial pass on physical dose maxima.
    dmax_candidates: List[float] = []
    io_retries = 10
    io_retry_delay = 1.0
    for frac in args.fractions:
        dose_csv = bioopt / frac / "case" / "dosedata.csv"
        if not dose_csv.exists():
            raise FileNotFoundError(dose_csv)
        dose_raw, _ = load_topas_grid(dose_csv, retries=io_retries, retry_delay_sec=io_retry_delay)
        dmax_candidates.append(float(np.max(dose_raw)))
    # Very conservative upper bound (post-normalization can increase).
    dvh_max = float(max(dmax_candidates)) * 350.0  # scale factor can be large; keep axis wide but finite
    dose_axis = np.linspace(0.0, max(60.0, dvh_max), int(args.dvh_bins))

    for i, frac in enumerate(args.fractions, start=1):
        placement_dir = bioopt / frac
        dose_csv = placement_dir / "case" / "dosedata.csv"

        dose_raw, _ = load_topas_grid(dose_csv, retries=io_retries, retry_delay_sec=io_retry_delay)
        ptv_raw = compute_structure_metrics(
            dose_raw,
            structures["PTV"],
            prescription_gy=float(args.prescription_gy),
            voxel_volume_cc=voxel_volume_cc,
        )
        raw_d95 = float(ptv_raw["d95_gy"])
        if raw_d95 <= 0.0:
            raise RuntimeError(f"{frac}: non-positive raw PTV D95; cannot normalize.")
        scale = float(args.prescription_gy) / raw_d95
        physical = dose_raw.astype(np.float32) * np.float32(scale)

        effective = compute_effective_dose(
            physical_dose=physical,
            voxel_size_mm=voxel_size_mm,
            structures=structures,
            bio_args=bio_args,
            alpha=float(args.alpha),
            beta=float(args.beta),
            steps=int(args.pde_steps),
            dt=float(args.pde_dt),
        ).astype(np.float32)

        physical_metrics = compute_metrics_for_domain(physical, structures=structures, voxel_volume_cc=voxel_volume_cc, prescription_gy=float(args.prescription_gy))
        effective_metrics = compute_metrics_for_domain(effective, structures=structures, voxel_volume_cc=voxel_volume_cc, prescription_gy=float(args.prescription_gy))

        row = summarize_fraction_row(i, frac, physical_metrics, effective_metrics)
        fraction_rows.append(row)

        dvhs_phys.append({name: compute_dvh(physical[structures[name]], dose_axis) for name in STRUCTURES_TARGET})
        dvhs_eff.append({name: compute_dvh(effective[structures[name]], dose_axis) for name in STRUCTURES_TARGET})

        physical_sum = physical if physical_sum is None else (physical_sum + physical)
        effective_sum_of_fractions = effective if effective_sum_of_fractions is None else (effective_sum_of_fractions + effective)

    assert physical_sum is not None
    assert effective_sum_of_fractions is not None

    # Cumulative physical metrics.
    cum_phys_metrics = compute_metrics_for_domain(physical_sum, structures=structures, voxel_volume_cc=voxel_volume_cc, prescription_gy=float(args.prescription_gy))

    # Proxy cumulative effective dose: apply biology to the summed physical dose.
    effective_of_sum = compute_effective_dose(
        physical_dose=physical_sum,
        voxel_size_mm=voxel_size_mm,
        structures=structures,
        bio_args=bio_args,
        alpha=float(args.alpha),
        beta=float(args.beta),
        steps=int(args.pde_steps),
        dt=float(args.pde_dt),
    ).astype(np.float32)
    cum_eff_of_sum_metrics = compute_metrics_for_domain(effective_of_sum, structures=structures, voxel_volume_cc=voxel_volume_cc, prescription_gy=float(args.prescription_gy))

    # Alternative proxy: sum of per-fraction effective doses.
    cum_eff_sum_metrics = compute_metrics_for_domain(
        effective_sum_of_fractions, structures=structures, voxel_volume_cc=voxel_volume_cc, prescription_gy=float(args.prescription_gy)
    )

    # Consistency across fractions (keys chosen to map to the user's questions).
    consistency_keys = [
        "PTV_D95_phys",
        "PTV_D95_eff",
        "PTV_PVDR_phys(D2/D98)",
        "PTV_PVDR_eff(D2/D98)",
        "CORD_D2_eff",
        "PAROTID_R_mean_eff",
        "THYROID_mean_eff",
        "BODY_Dmax_phys",
        "BODY_Dmax_eff",
    ]
    consistency = consistency_table(fraction_rows, consistency_keys)

    write_csv(out_dir / "per_fraction_metrics.csv", fraction_rows)
    write_csv(out_dir / "consistency_summary.csv", consistency)
    (out_dir / "cumulative_metrics.json").write_text(
        json.dumps(
            {
                "note": "Cumulative metrics with two effective-dose proxies; see script docstring for interpretation.",
                "fractions": list(args.fractions),
                "cumulative_physical_metrics": cum_phys_metrics,
                "cumulative_effective_of_sum_physical_metrics": cum_eff_of_sum_metrics,
                "cumulative_sum_of_effective_metrics": cum_eff_sum_metrics,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    plot_consistency_bars(out_dir / "figure_consistency_bars.png", consistency, dpi=int(args.dpi))
    plot_overlay_target_dvhs(out_dir / "figure_ptv_dvhs_across_fractions.png", dose_axis, dvhs_phys, dvhs_eff, args.fractions, dpi=int(args.dpi))

    print(f"=== FRACTIONATED EVAL WRITTEN TO {out_dir} ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

