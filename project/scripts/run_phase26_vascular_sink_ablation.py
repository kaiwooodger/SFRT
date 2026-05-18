#!/usr/bin/env python3
"""Phase 26: vascular sink ablation and uptake sensitivity for Phase 25 plans.

This phase reuses the saved Phase 25 safe-core course dose volumes and evaluates
each plan three ways:

1. physical-only
2. bystander model without vascular sink uptake
3. bystander model with anatomical vascular sink uptake

It then reports the seven clinical endpoints, rank shifts, OAR reinterpretation,
assay-like proxy outputs, and Morris-motivated uptake sensitivity bands.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Mapping, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage

from bystander_multispecies_pde_solver import (
    calculate_effective_dose,
    calculate_phase7_survival,
    calculate_systemic_immune_penalty,
    solve_multispecies_pde_3d_with_hazard_observables,
)
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
)


@dataclass(frozen=True)
class EndpointSpec:
    key: str
    label: str
    units: str
    higher_is_better: bool


PRIMARY_ENDPOINTS: Tuple[EndpointSpec, ...] = (
    EndpointSpec("ptv_d95", "PTV D95", "Gy", True),
    EndpointSpec("pvdr", "PVDR", "ratio", True),
    EndpointSpec("spill_shell_0_5_mean", "Peri-GTV 0-5 mm mean", "Gy", False),
    EndpointSpec("spill_shell_5_15_mean", "Peri-GTV 5-15 mm mean", "Gy", False),
    EndpointSpec("cord_d2", "Cord D2", "Gy", False),
    EndpointSpec("brainstem_d2", "Brainstem D2", "Gy", False),
    EndpointSpec("parotid_r_mean", "Parotid R mean", "Gy", False),
)

ASSAY_METRICS: Tuple[str, ...] = (
    "mean_gammah2ax_peak",
    "mean_gammah2ax_valley",
    "mean_tunel_peak",
    "mean_tunel_valley",
    "cytokine_global_auc",
    "cytokine_peak_roi_auc",
    "cytokine_valley_roi_auc",
    "immune_scalar",
    "icd_volume_cm3",
)

MODE_LABELS = {
    "physical_only": "Physical-only",
    "bystander_no_sink": "Bystander, no vascular sink",
    "bystander_with_sink": "Bystander, anatomical vascular sink",
}


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase25-run-root",
        type=Path,
        default=root
        / "runs"
        / "linac_6mv_headneck_detailed_voxel_lattice_sfrt_phase25_safe_core_plan_library_1e5",
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        default=None,
        help="Defaults to <phase25-run-root>/phase26_vascular_sink_ablation.",
    )
    parser.add_argument("--history-interval", type=int, default=20)
    parser.add_argument("--sensitivity-uptake-scales", type=float, nargs="+", default=[0.75, 1.0, 1.25])
    parser.add_argument("--cord-limit-gy", type=float, default=85.0)
    parser.add_argument("--brainstem-limit-gy", type=float, default=30.0)
    parser.add_argument("--parotid-r-limit-gy", type=float, default=60.0)
    parser.add_argument("--thyroid-limit-gy", type=float, default=50.0)
    parser.add_argument("--body-dmax-limit-gy", type=float, default=400.0)
    parser.add_argument("--dpi", type=int, default=180)
    parser.add_argument("--verbose-pde", action="store_true")
    parser.add_argument("--progress-interval", type=int, default=100)
    parser.add_argument("--max-plans", type=int, default=0, help="Optional smoke-test limit. 0 means all plans.")
    return parser.parse_args()


def write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        raise ValueError(f"No rows available for CSV output: {path}")
    fieldnames: List[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(str(key))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_npz_arrays_with_retry(path: Path, *, retries: int = 4, pause_s: float = 1.0) -> Dict[str, np.ndarray]:
    last_error: Exception | None = None
    for attempt in range(int(retries)):
        try:
            payload = path.read_bytes()
            with np.load(io.BytesIO(payload)) as data:
                return {name: np.asarray(data[name]) for name in data.files}
        except TimeoutError as exc:
            last_error = exc
            if attempt == retries - 1:
                break
            time.sleep(float(pause_s) * float(attempt + 1))
    if last_error is not None:
        raise last_error
    raise RuntimeError(f"Failed to load NPZ data from {path}")


def load_phase25_context(run_root: Path) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Mapping[str, object]]:
    context_npz = load_npz_arrays_with_retry(run_root / "phase25_phantom_context.npz")
    structures = {
        key.removeprefix("struct_"): np.asarray(context_npz[key], dtype=bool)
        for key in context_npz
        if key.startswith("struct_")
    }
    axes_mm = {
        "x": np.asarray(context_npz["axes_x_mm"], dtype=np.float32),
        "y": np.asarray(context_npz["axes_y_mm"], dtype=np.float32),
        "z": np.asarray(context_npz["axes_z_mm"], dtype=np.float32),
    }
    config = json.loads((run_root / "phase25_config.json").read_text(encoding="utf-8"))
    return structures, axes_mm, config


def bio_args_from_config(config: Mapping[str, object], *, uptake_scale: float = 1.0) -> SimpleNamespace:
    params = dict(config["bio_parameters"])
    return SimpleNamespace(
        tumor_cytokine_multiplier=float(params["tumor_cytokine_multiplier"]),
        hypoxic_ros_scale=float(params["hypoxic_ros_scale"]),
        hypoxic_cytokine_multiplier=float(params["hypoxic_cytokine_multiplier"]),
        artery_ros_uptake=float(params["artery_ros_uptake"]) * float(uptake_scale),
        artery_cyto_uptake=float(params["artery_cyto_uptake"]) * float(uptake_scale),
        vein_ros_uptake=float(params["vein_ros_uptake"]) * float(uptake_scale),
        vein_cyto_uptake=float(params["vein_cyto_uptake"]) * float(uptake_scale),
    )


def voxel_size_mm_from_axes(axes_mm: Mapping[str, np.ndarray]) -> Tuple[float, float, float]:
    return (
        float(axes_mm["x"][1] - axes_mm["x"][0]),
        float(axes_mm["y"][1] - axes_mm["y"][0]),
        float(axes_mm["z"][1] - axes_mm["z"][0]),
    )


def voxel_volume_cc_from_axes(axes_mm: Mapping[str, np.ndarray]) -> float:
    vx, vy, vz = voxel_size_mm_from_axes(axes_mm)
    return float(vx * vy * vz / 1000.0)


def compute_structure_metrics_local(
    dose: np.ndarray,
    mask: np.ndarray,
    *,
    prescription_gy: float | None,
    voxel_volume_cc: float,
    volume_thresholds_gy: Sequence[float] = (2.0, 5.0, 10.0, 20.0),
) -> Dict[str, float]:
    values = np.asarray(dose[np.asarray(mask, dtype=bool)], dtype=np.float64)
    if values.size == 0:
        return {
            "voxels": 0,
            "volume_cc": 0.0,
            "mean_gy": 0.0,
            "dmax_gy": 0.0,
            "d2_gy": 0.0,
            "d5_gy": 0.0,
            "d95_gy": 0.0,
            "d98_gy": 0.0,
            "v95_pct": 0.0,
            "v100_pct": 0.0,
        }
    metrics = {
        "voxels": int(values.size),
        "volume_cc": float(values.size * float(voxel_volume_cc)),
        "mean_gy": float(np.mean(values)),
        "dmax_gy": float(np.max(values)),
        "d2_gy": float(np.percentile(values, 98.0)),
        "d5_gy": float(np.percentile(values, 95.0)),
        "d95_gy": float(np.percentile(values, 5.0)),
        "d98_gy": float(np.percentile(values, 2.0)),
    }
    if prescription_gy is not None:
        metrics["v95_pct"] = float(100.0 * np.mean(values >= 0.95 * float(prescription_gy)))
        metrics["v100_pct"] = float(100.0 * np.mean(values >= float(prescription_gy)))
    else:
        metrics["v95_pct"] = 0.0
        metrics["v100_pct"] = 0.0
    for threshold in volume_thresholds_gy:
        metrics[f"v{float(threshold):g}gy_pct"] = float(100.0 * np.mean(values >= float(threshold)))
    return metrics


def compute_metrics_for_domain_local(
    dose: np.ndarray,
    *,
    structures: Mapping[str, np.ndarray],
    voxel_volume_cc: float,
    prescription_gy: float,
) -> Dict[str, Dict[str, float]]:
    names = (
        "PTV",
        "GTV",
        "SPINAL_CORD",
        "BRAINSTEM",
        "PAROTID_R",
        "PAROTID_L",
        "THYROID",
        "BRAIN",
        "MANDIBLE",
        "BODY",
    )
    metrics: Dict[str, Dict[str, float]] = {}
    for name in names:
        if name not in structures:
            continue
        metrics[name] = compute_structure_metrics_local(
            dose,
            np.asarray(structures[name], dtype=bool),
            prescription_gy=float(prescription_gy) if name in ("PTV", "GTV") else None,
            voxel_volume_cc=float(voxel_volume_cc),
            volume_thresholds_gy=[float(prescription_gy), 10.0] if name in ("PTV", "GTV") else [5.0, 10.0],
        )
    return metrics


def spherical_union_mask(
    axes_mm: Mapping[str, np.ndarray],
    centers_mm: Sequence[Tuple[float, float, float]],
    radius_mm: float,
) -> np.ndarray:
    x_mm = np.asarray(axes_mm["x"], dtype=np.float32)
    y_mm = np.asarray(axes_mm["y"], dtype=np.float32)
    z_mm = np.asarray(axes_mm["z"], dtype=np.float32)
    mask = np.zeros((x_mm.size, y_mm.size, z_mm.size), dtype=bool)
    if not centers_mm:
        return mask
    margin = float(radius_mm) + max(voxel_size_mm_from_axes(axes_mm))
    radius2 = float(radius_mm) ** 2
    for cx, cy, cz in centers_mm:
        ix = np.flatnonzero((x_mm >= float(cx) - margin) & (x_mm <= float(cx) + margin))
        iy = np.flatnonzero((y_mm >= float(cy) - margin) & (y_mm <= float(cy) + margin))
        iz = np.flatnonzero((z_mm >= float(cz) - margin) & (z_mm <= float(cz) + margin))
        if ix.size == 0 or iy.size == 0 or iz.size == 0:
            continue
        xx = x_mm[ix][:, None, None]
        yy = y_mm[iy][None, :, None]
        zz = z_mm[iz][None, None, :]
        local = ((xx - float(cx)) ** 2 + (yy - float(cy)) ** 2 + (zz - float(cz)) ** 2) <= radius2
        mask[np.ix_(ix, iy, iz)] |= local
    return mask


def build_peak_valley_rois_local(
    structures: Mapping[str, np.ndarray],
    axes_mm: Mapping[str, np.ndarray],
    delivered_spots_mm: Sequence[Tuple[float, float, float]],
    *,
    peak_radius_mm: float,
    valley_exclusion_radius_mm: float,
) -> Tuple[np.ndarray, np.ndarray]:
    ptv_mask = np.asarray(structures["PTV"], dtype=bool)
    peak_mask = spherical_union_mask(axes_mm, delivered_spots_mm, float(peak_radius_mm)) & ptv_mask
    valley_mask = ptv_mask & ~spherical_union_mask(axes_mm, delivered_spots_mm, float(valley_exclusion_radius_mm))
    if int(np.count_nonzero(valley_mask)) == 0:
        relaxed_radius = max(float(peak_radius_mm) * 1.1, float(peak_radius_mm) + 1.0)
        valley_mask = ptv_mask & ~spherical_union_mask(axes_mm, delivered_spots_mm, relaxed_radius)
    if int(np.count_nonzero(valley_mask)) == 0:
        valley_mask = ptv_mask & ~peak_mask
    if int(np.count_nonzero(peak_mask)) == 0 or int(np.count_nonzero(valley_mask)) == 0:
        raise RuntimeError("Peak or valley ROI became empty in Phase 26.")
    return peak_mask, valley_mask


def compute_peak_valley_metrics_local(dose: np.ndarray, peak_mask: np.ndarray, valley_mask: np.ndarray) -> Dict[str, float]:
    peak_values = np.asarray(dose[peak_mask], dtype=np.float64)
    valley_values = np.asarray(dose[valley_mask], dtype=np.float64)
    peak_mean = float(np.mean(peak_values))
    valley_mean = float(np.mean(valley_values))
    return {
        "peak_mean_gy": peak_mean,
        "valley_mean_gy": valley_mean,
        "peak_p90_gy": float(np.percentile(peak_values, 90.0)),
        "valley_p10_gy": float(np.percentile(valley_values, 10.0)),
        "pvdr": float(peak_mean / max(valley_mean, 1.0e-6)),
        "peak_voxels": int(peak_values.size),
        "valley_voxels": int(valley_values.size),
    }


def build_spill_region_masks_local(
    *,
    structures: Mapping[str, np.ndarray],
    axes_mm: Mapping[str, np.ndarray],
    peak_mask: np.ndarray,
    shell_1_mm: float,
    shell_2_mm: float,
    shell_3_mm: float,
    oar_adjacent_mm: float,
) -> Dict[str, np.ndarray]:
    shape = np.asarray(structures["BODY"], dtype=bool).shape
    empty = np.zeros(shape, dtype=bool)
    gtv_mask = np.asarray(structures["GTV"], dtype=bool)
    ptv_mask = np.asarray(structures["PTV"], dtype=bool)
    body_mask = np.asarray(structures["BODY"], dtype=bool)
    outside_gtv = body_mask & ~gtv_mask

    distance_to_gtv_mm = ndimage.distance_transform_edt(~gtv_mask, sampling=voxel_size_mm_from_axes(axes_mm))
    shell_0_5 = outside_gtv & (distance_to_gtv_mm > 0.0) & (distance_to_gtv_mm <= float(shell_1_mm))
    shell_5_15 = outside_gtv & (distance_to_gtv_mm > float(shell_1_mm)) & (distance_to_gtv_mm <= float(shell_2_mm))
    shell_15_30 = outside_gtv & (distance_to_gtv_mm > float(shell_2_mm)) & (distance_to_gtv_mm <= float(shell_3_mm))

    critical_names = ("SPINAL_CORD", "BRAINSTEM", "PAROTID_R", "PAROTID_L", "THYROID", "PARATHYROIDS")
    critical_oar_union = np.zeros(shape, dtype=bool)
    for name in critical_names:
        critical_oar_union |= np.asarray(structures.get(name, empty), dtype=bool)
    distance_to_oar_mm = ndimage.distance_transform_edt(~critical_oar_union, sampling=voxel_size_mm_from_axes(axes_mm))
    return {
        "SPILL_SHELL_0_5": shell_0_5,
        "SPILL_SHELL_5_15": shell_5_15,
        "SPILL_SHELL_15_30": shell_15_30,
        "OUTSIDE_GTV": outside_gtv,
        "OAR_ADJACENT_OUTSIDE_GTV": outside_gtv & (distance_to_oar_mm <= float(oar_adjacent_mm)),
        "PTV_VALLEY_OUTSIDE_GTV": (ptv_mask & ~gtv_mask) & ~np.asarray(peak_mask, dtype=bool),
    }


def compute_region_metrics_local(
    dose: np.ndarray,
    region_masks: Mapping[str, np.ndarray],
    *,
    voxel_volume_cc: float,
    volume_thresholds_gy: Sequence[float] = (2.0, 5.0, 10.0, 20.0),
) -> Dict[str, Dict[str, float]]:
    metrics: Dict[str, Dict[str, float]] = {}
    for name, mask in region_masks.items():
        if int(np.count_nonzero(mask)) == 0:
            continue
        metrics[name] = compute_structure_metrics_local(
            dose,
            np.asarray(mask, dtype=bool),
            prescription_gy=None,
            voxel_volume_cc=float(voxel_volume_cc),
            volume_thresholds_gy=volume_thresholds_gy,
        )
    return metrics


def calculate_gamma_h2ax_proxy_local(
    dose_grid: np.ndarray,
    ros_peak_grid: np.ndarray,
    *,
    alpha: float,
    beta: float,
) -> np.ndarray:
    physical_drive = float(alpha) * dose_grid + float(beta) * dose_grid**2
    ros_scale = max(float(np.percentile(ros_peak_grid, 99.0)), 1.0e-6)
    ros_drive = np.clip(ros_peak_grid / ros_scale, 0.0, 3.0)
    return (1.0 - np.exp(-(physical_drive + 0.45 * ros_drive))).astype(np.float32, copy=False)


def build_emission_tensor(
    dose_grid: np.ndarray,
    *,
    m_type: np.ndarray,
    m_oxygen: np.ndarray,
) -> np.ndarray:
    base_emission = (1.0 - np.exp(-float(LOCKED_GAMMA) * np.asarray(dose_grid, dtype=np.float32))).astype(np.float32)
    return (
        np.asarray([float(EMAX_ROS), float(EMAX_CYTO)], dtype=np.float32)[:, None, None, None]
        * base_emission[None, :, :, :]
        * np.asarray(m_type, dtype=np.float32)
        * np.asarray(m_oxygen, dtype=np.float32)
    ).astype(np.float32, copy=False)


def endpoint_z_scores(values: Sequence[float], *, higher_is_better: bool) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    mean = float(arr.mean()) if arr.size else 0.0
    std = float(arr.std(ddof=0)) if arr.size else 0.0
    if std <= 1.0e-9:
        z = np.zeros_like(arr)
    else:
        z = (arr - mean) / std
    return -z if bool(higher_is_better) else z


def extract_endpoints(
    dose_grid: np.ndarray,
    *,
    structures: Mapping[str, np.ndarray],
    axes_mm: Mapping[str, np.ndarray],
    spots_mm: Sequence[Tuple[float, float, float]],
    voxel_volume_cc: float,
    prescription_gy: float,
) -> Tuple[Dict[str, float], Dict[str, object]]:
    metrics = compute_metrics_for_domain_local(
        dose_grid,
        structures=dict(structures),
        voxel_volume_cc=float(voxel_volume_cc),
        prescription_gy=float(prescription_gy),
    )
    peak_mask, valley_mask = build_peak_valley_rois_local(
        dict(structures),
        dict(axes_mm),
        spots_mm,
        peak_radius_mm=8.0,
        valley_exclusion_radius_mm=14.0,
    )
    pvdr = compute_peak_valley_metrics_local(dose_grid, peak_mask, valley_mask)
    spill_masks = build_spill_region_masks_local(
        structures=dict(structures),
        axes_mm=dict(axes_mm),
        peak_mask=peak_mask,
        shell_1_mm=5.0,
        shell_2_mm=15.0,
        shell_3_mm=30.0,
        oar_adjacent_mm=15.0,
    )
    spill_metrics = compute_region_metrics_local(dose_grid, spill_masks, voxel_volume_cc=float(voxel_volume_cc))
    endpoints = {
        "ptv_d95": float(metrics["PTV"]["d95_gy"]),
        "pvdr": float(pvdr["pvdr"]),
        "spill_shell_0_5_mean": float(spill_metrics.get("SPILL_SHELL_0_5", {}).get("mean_gy", 0.0)),
        "spill_shell_5_15_mean": float(spill_metrics.get("SPILL_SHELL_5_15", {}).get("mean_gy", 0.0)),
        "cord_d2": float(metrics["SPINAL_CORD"]["d2_gy"]),
        "brainstem_d2": float(metrics["BRAINSTEM"]["d2_gy"]),
        "parotid_r_mean": float(metrics["PAROTID_R"]["mean_gy"]),
    }
    supplemental = {
        "gtv_d95": float(metrics["GTV"]["d95_gy"]),
        "thyroid_mean": float(metrics["THYROID"]["mean_gy"]),
        "body_dmax": float(metrics["BODY"]["dmax_gy"]),
        "parotid_l_mean": float(metrics.get("PAROTID_L", {}).get("mean_gy", 0.0)),
        "spill_shell_15_30_mean": float(spill_metrics.get("SPILL_SHELL_15_30", {}).get("mean_gy", 0.0)),
        "outside_gtv_d2": float(spill_metrics.get("OUTSIDE_GTV", {}).get("d2_gy", 0.0)),
        "oar_adjacent_outside_gtv_mean": float(
            spill_metrics.get("OAR_ADJACENT_OUTSIDE_GTV", {}).get("mean_gy", 0.0)
        ),
        "ptv_valley_outside_gtv_mean": float(
            spill_metrics.get("PTV_VALLEY_OUTSIDE_GTV", {}).get("mean_gy", 0.0)
        ),
        "peak_mean": float(pvdr["peak_mean_gy"]),
        "valley_mean": float(pvdr["valley_mean_gy"]),
        "peak_voxels": int(pvdr["peak_voxels"]),
        "valley_voxels": int(pvdr["valley_voxels"]),
        "metrics": metrics,
        "peak_mask": peak_mask,
        "valley_mask": valley_mask,
    }
    return endpoints, supplemental


def summarize_assays(
    *,
    physical_dose: np.ndarray,
    final_survival: np.ndarray,
    peak_concentration_ros: np.ndarray,
    cytokine_final: np.ndarray,
    time_axis: np.ndarray,
    global_history: np.ndarray,
    mask_history: Mapping[str, np.ndarray],
    peak_mask: np.ndarray,
    valley_mask: np.ndarray,
    alpha: float,
    beta: float,
    voxel_size_mm: Tuple[float, float, float],
) -> Dict[str, float]:
    gamma_h2ax_map = calculate_gamma_h2ax_proxy_local(
        physical_dose,
        peak_concentration_ros,
        alpha=float(alpha),
        beta=float(beta),
    )
    tunel_map = (1.0 - np.asarray(final_survival, dtype=np.float32)).astype(np.float32, copy=False)
    _, icd_volume_cm3 = calculate_systemic_immune_penalty(physical_dose, voxel_size_mm)
    return {
        "mean_gammah2ax_peak": float(np.mean(gamma_h2ax_map[peak_mask])),
        "mean_gammah2ax_valley": float(np.mean(gamma_h2ax_map[valley_mask])),
        "mean_tunel_peak": float(np.mean(tunel_map[peak_mask])),
        "mean_tunel_valley": float(np.mean(tunel_map[valley_mask])),
        "mean_cytokine_final_peak": float(np.mean(cytokine_final[peak_mask])),
        "mean_cytokine_final_valley": float(np.mean(cytokine_final[valley_mask])),
        "cytokine_global_auc": float(np.trapz(global_history[1], time_axis)) if time_axis.size else 0.0,
        "cytokine_peak_roi_auc": (
            float(np.trapz(mask_history["peak_roi"][1], time_axis))
            if "peak_roi" in mask_history and time_axis.size
            else 0.0
        ),
        "cytokine_valley_roi_auc": (
            float(np.trapz(mask_history["valley_roi"][1], time_axis))
            if "valley_roi" in mask_history and time_axis.size
            else 0.0
        ),
        "immune_scalar": float(calculate_systemic_immune_penalty(physical_dose, voxel_size_mm)[0]),
        "icd_volume_cm3": float(icd_volume_cm3),
    }


def evaluate_physical_only(
    *,
    physical_dose: np.ndarray,
    structures: Mapping[str, np.ndarray],
    axes_mm: Mapping[str, np.ndarray],
    spots_mm: Sequence[Tuple[float, float, float]],
    voxel_volume_cc: float,
    voxel_size_mm: Tuple[float, float, float],
    prescription_gy: float,
    alpha: float,
    beta: float,
) -> Dict[str, object]:
    endpoints, supplemental = extract_endpoints(
        physical_dose,
        structures=structures,
        axes_mm=axes_mm,
        spots_mm=spots_mm,
        voxel_volume_cc=float(voxel_volume_cc),
        prescription_gy=float(prescription_gy),
    )
    lq_survival = np.exp(-float(alpha) * physical_dose - float(beta) * physical_dose**2).astype(np.float32)
    zero = np.zeros_like(physical_dose, dtype=np.float32)
    assays = summarize_assays(
        physical_dose=physical_dose,
        final_survival=lq_survival,
        peak_concentration_ros=zero,
        cytokine_final=zero,
        time_axis=np.asarray([], dtype=np.float32),
        global_history=np.zeros((2, 0), dtype=np.float32),
        mask_history={},
        peak_mask=supplemental["peak_mask"],
        valley_mask=supplemental["valley_mask"],
        alpha=float(alpha),
        beta=float(beta),
        voxel_size_mm=voxel_size_mm,
    )
    return {"dose_for_endpoints": physical_dose, "endpoints": endpoints, "supplemental": supplemental, "assays": assays}


def evaluate_bystander_mode(
    *,
    physical_dose: np.ndarray,
    structures: Mapping[str, np.ndarray],
    axes_mm: Mapping[str, np.ndarray],
    spots_mm: Sequence[Tuple[float, float, float]],
    config: Mapping[str, object],
    mode: str,
    uptake_scale: float,
    voxel_volume_cc: float,
    voxel_size_mm: Tuple[float, float, float],
    prescription_gy: float,
    history_interval: int,
    progress_interval: int,
    verbose_pde: bool,
) -> Dict[str, object]:
    params = config["bio_parameters"]
    alpha = float(params["alpha"])
    beta = float(params["beta"])
    bio_args = bio_args_from_config(config, uptake_scale=float(uptake_scale))
    uptake_tensor, m_type, m_oxygen, _ = build_anatomical_biology_tensors(bio_args, dict(structures))
    if mode == "bystander_no_sink":
        uptake_tensor = np.zeros_like(uptake_tensor, dtype=np.float32)
    elif mode != "bystander_with_sink":
        raise ValueError(f"Unsupported bystander mode: {mode}")

    endpoint_masks = extract_endpoints(
        physical_dose,
        structures=structures,
        axes_mm=axes_mm,
        spots_mm=spots_mm,
        voxel_volume_cc=float(voxel_volume_cc),
        prescription_gy=float(prescription_gy),
    )[1]
    emission_tensor = build_emission_tensor(physical_dose, m_type=m_type, m_oxygen=m_oxygen)
    observables = solve_multispecies_pde_3d_with_hazard_observables(
        physical_dose,
        voxel_size_mm,
        diffusion_coeffs=(float(D_ROS), float(LOCKED_D_CYTO)),
        decay_coeffs=(float(LAMBDA_ROS), float(LOCKED_LAMBDA_CYTO)),
        emission_emax=(float(EMAX_ROS), float(EMAX_CYTO)),
        emission_gamma_per_gy=float(LOCKED_GAMMA),
        emission_tensor=emission_tensor,
        uptake_tensor=uptake_tensor,
        hazard_weights=(float(W_ROS), float(W_CYTO)),
        history_masks={"peak_roi": endpoint_masks["peak_mask"], "valley_roi": endpoint_masks["valley_mask"]},
        history_interval=int(history_interval),
        steps=int(params["pde_steps"]),
        dt=float(params["pde_dt"]),
        progress_interval=int(progress_interval),
        verbose=bool(verbose_pde),
    )
    lq_survival = np.exp(-alpha * physical_dose - beta * physical_dose**2).astype(np.float32)
    hazard_grid = np.asarray(observables["hazard_grid"], dtype=np.float32)
    final_survival = calculate_phase7_survival(
        lq_survival,
        hazard_grid,
        physical_dose,
        voxel_size_mm,
        float(LOCKED_SCALING_FACTOR),
        weight_immune=float(W_IMMUNE),
        verbose=False,
    )
    effective_dose = calculate_effective_dose(final_survival, alpha=alpha, beta=beta)
    endpoints, supplemental = extract_endpoints(
        effective_dose,
        structures=structures,
        axes_mm=axes_mm,
        spots_mm=spots_mm,
        voxel_volume_cc=float(voxel_volume_cc),
        prescription_gy=float(prescription_gy),
    )
    concentration = np.asarray(observables["concentration"], dtype=np.float32)
    peak_concentration = np.asarray(observables["peak_concentration"], dtype=np.float32)
    mask_history = {
        str(name): np.asarray(values, dtype=np.float32)
        for name, values in dict(observables["mask_mean_history"]).items()
    }
    assays = summarize_assays(
        physical_dose=physical_dose,
        final_survival=final_survival,
        peak_concentration_ros=peak_concentration[0],
        cytokine_final=concentration[1],
        time_axis=np.asarray(observables["time_axis"], dtype=np.float32),
        global_history=np.asarray(observables["global_mean_history"], dtype=np.float32),
        mask_history=mask_history,
        peak_mask=supplemental["peak_mask"],
        valley_mask=supplemental["valley_mask"],
        alpha=alpha,
        beta=beta,
        voxel_size_mm=voxel_size_mm,
    )
    return {
        "dose_for_endpoints": effective_dose,
        "endpoints": endpoints,
        "supplemental": supplemental,
        "assays": assays,
    }


def assign_ranks(endpoint_rows: List[Dict[str, object]]) -> None:
    rows_by_mode: Dict[str, List[Dict[str, object]]] = {}
    for row in endpoint_rows:
        rows_by_mode.setdefault(str(row["mode"]), []).append(row)
    for rows in rows_by_mode.values():
        risk_scores = np.zeros(len(rows), dtype=np.float64)
        for endpoint in PRIMARY_ENDPOINTS:
            values = [float(row[endpoint.key]) for row in rows]
            risk_scores += endpoint_z_scores(values, higher_is_better=endpoint.higher_is_better)
        order = np.argsort(np.argsort(risk_scores)) + 1
        for idx, row in enumerate(rows):
            row["risk_score"] = float(risk_scores[idx])
            row["rank"] = int(order[idx])


def build_rank_shift_rows(endpoint_rows: Sequence[Mapping[str, object]]) -> List[Dict[str, object]]:
    by_plan: Dict[str, Dict[str, Mapping[str, object]]] = {}
    for row in endpoint_rows:
        by_plan.setdefault(str(row["plan_id"]), {})[str(row["mode"])] = row
    rows: List[Dict[str, object]] = []
    for plan_id, modes in sorted(by_plan.items()):
        phys = modes["physical_only"]
        no_sink = modes["bystander_no_sink"]
        with_sink = modes["bystander_with_sink"]
        rows.append(
            {
                "plan_id": plan_id,
                "physical_rank": int(phys["rank"]),
                "no_sink_rank": int(no_sink["rank"]),
                "with_sink_rank": int(with_sink["rank"]),
                "no_sink_vs_physical_shift": int(no_sink["rank"]) - int(phys["rank"]),
                "with_sink_vs_physical_shift": int(with_sink["rank"]) - int(phys["rank"]),
                "with_sink_vs_no_sink_shift": int(with_sink["rank"]) - int(no_sink["rank"]),
                "physical_risk_score": float(phys["risk_score"]),
                "no_sink_risk_score": float(no_sink["risk_score"]),
                "with_sink_risk_score": float(with_sink["risk_score"]),
            }
        )
    return rows


def status(value: float, limit: float, *, lower_is_better: bool = True) -> str:
    if lower_is_better:
        return "pass" if float(value) <= float(limit) else "fail"
    return "pass" if float(value) >= float(limit) else "fail"


def build_oar_reinterpretation_rows(
    endpoint_rows: Sequence[Mapping[str, object]],
    *,
    args: argparse.Namespace,
) -> List[Dict[str, object]]:
    by_plan: Dict[str, Dict[str, Mapping[str, object]]] = {}
    for row in endpoint_rows:
        by_plan.setdefault(str(row["plan_id"]), {})[str(row["mode"])] = row
    specs = [
        ("cord_d2", "Cord D2", float(args.cord_limit_gy)),
        ("brainstem_d2", "Brainstem D2", float(args.brainstem_limit_gy)),
        ("parotid_r_mean", "Parotid R mean", float(args.parotid_r_limit_gy)),
        ("thyroid_mean", "Thyroid mean", float(args.thyroid_limit_gy)),
        ("body_dmax", "Body Dmax", float(args.body_dmax_limit_gy)),
    ]
    rows: List[Dict[str, object]] = []
    for plan_id, modes in sorted(by_plan.items()):
        for key, label, limit in specs:
            values = {
                "physical_only": float(modes["physical_only"][key]),
                "bystander_no_sink": float(modes["bystander_no_sink"][key]),
                "bystander_with_sink": float(modes["bystander_with_sink"][key]),
            }
            statuses = {name: status(value, limit) for name, value in values.items()}
            reinterpretation = "unchanged"
            if statuses["physical_only"] == "pass" and statuses["bystander_with_sink"] == "fail":
                reinterpretation = "biology_adds_failure"
            elif statuses["physical_only"] == "fail" and statuses["bystander_with_sink"] == "pass":
                reinterpretation = "biology_reduces_physical_failure"
            elif statuses["bystander_no_sink"] != statuses["bystander_with_sink"]:
                reinterpretation = "vascular_sink_changes_status"
            elif abs(values["bystander_with_sink"] - values["bystander_no_sink"]) > 0.05 * max(limit, 1.0):
                reinterpretation = "vascular_sink_changes_margin"
            rows.append(
                {
                    "plan_id": plan_id,
                    "metric": key,
                    "label": label,
                    "limit": float(limit),
                    "physical_value": values["physical_only"],
                    "no_sink_value": values["bystander_no_sink"],
                    "with_sink_value": values["bystander_with_sink"],
                    "physical_status": statuses["physical_only"],
                    "no_sink_status": statuses["bystander_no_sink"],
                    "with_sink_status": statuses["bystander_with_sink"],
                    "with_sink_minus_no_sink": values["bystander_with_sink"] - values["bystander_no_sink"],
                    "reinterpretation": reinterpretation,
                }
            )
    return rows


def build_delta_rows(endpoint_rows: Sequence[Mapping[str, object]]) -> List[Dict[str, object]]:
    by_plan: Dict[str, Dict[str, Mapping[str, object]]] = {}
    for row in endpoint_rows:
        by_plan.setdefault(str(row["plan_id"]), {})[str(row["mode"])] = row
    rows: List[Dict[str, object]] = []
    for plan_id, modes in sorted(by_plan.items()):
        for endpoint in PRIMARY_ENDPOINTS:
            phys = float(modes["physical_only"][endpoint.key])
            no_sink = float(modes["bystander_no_sink"][endpoint.key])
            with_sink = float(modes["bystander_with_sink"][endpoint.key])
            rows.append(
                {
                    "plan_id": plan_id,
                    "endpoint": endpoint.key,
                    "label": endpoint.label,
                    "units": endpoint.units,
                    "physical_value": phys,
                    "no_sink_value": no_sink,
                    "with_sink_value": with_sink,
                    "no_sink_minus_physical": no_sink - phys,
                    "with_sink_minus_physical": with_sink - phys,
                    "with_sink_minus_no_sink": with_sink - no_sink,
                    "with_sink_pct_change_from_no_sink": 100.0 * (with_sink - no_sink) / max(abs(no_sink), 1.0e-6),
                }
            )
    return rows


def build_sensitivity_band_rows(
    sensitivity_samples: Sequence[Mapping[str, object]],
    base_lookup: Mapping[Tuple[str, str], Mapping[str, float]],
) -> List[Dict[str, object]]:
    grouped: Dict[Tuple[str, str, str], List[float]] = {}
    for sample in sensitivity_samples:
        plan_id = str(sample["plan_id"])
        metric_type = str(sample["metric_type"])
        for metric, value in dict(sample["values"]).items():
            grouped.setdefault((plan_id, metric_type, str(metric)), []).append(float(value))
    rows: List[Dict[str, object]] = []
    for (plan_id, metric_type, metric), values in sorted(grouped.items()):
        arr = np.asarray(values, dtype=np.float64)
        base = float(base_lookup.get((plan_id, metric_type), {}).get(metric, np.nan))
        rows.append(
            {
                "plan_id": plan_id,
                "metric_type": metric_type,
                "metric": metric,
                "base_with_sink": base,
                "min": float(np.min(arr)),
                "mean": float(np.mean(arr)),
                "max": float(np.max(arr)),
                "lower_delta": float(base - np.min(arr)) if np.isfinite(base) else float("nan"),
                "upper_delta": float(np.max(arr) - base) if np.isfinite(base) else float("nan"),
                "full_width": float(np.max(arr) - np.min(arr)),
                "relative_full_width_pct": float(100.0 * (np.max(arr) - np.min(arr)) / max(abs(base), 1.0e-6))
                if np.isfinite(base)
                else float("nan"),
            }
        )
    return rows


def plot_endpoint_mode_heatmap(out_file: Path, endpoint_rows: Sequence[Mapping[str, object]], *, dpi: int) -> None:
    labels = [endpoint.label for endpoint in PRIMARY_ENDPOINTS]
    by_mode = {mode: [row for row in endpoint_rows if str(row["mode"]) == mode] for mode in MODE_LABELS}
    matrix = np.asarray(
        [[float(np.mean([row[endpoint.key] for row in by_mode[mode]])) for endpoint in PRIMARY_ENDPOINTS] for mode in MODE_LABELS],
        dtype=np.float64,
    )
    risk_matrix = np.zeros_like(matrix)
    for col_idx, endpoint in enumerate(PRIMARY_ENDPOINTS):
        risk_matrix[:, col_idx] = endpoint_z_scores(matrix[:, col_idx], higher_is_better=endpoint.higher_is_better)
    fig, ax = plt.subplots(figsize=(12.0, 4.5), constrained_layout=True)
    image = ax.imshow(risk_matrix, cmap="RdYlBu_r", aspect="auto")
    ax.set_yticks(np.arange(len(MODE_LABELS)))
    ax.set_yticklabels([MODE_LABELS[mode] for mode in MODE_LABELS])
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_title("Phase 26: endpoint risk profile by biological model mode")
    for row_idx in range(matrix.shape[0]):
        for col_idx in range(matrix.shape[1]):
            ax.text(col_idx, row_idx, f"{matrix[row_idx, col_idx]:.2g}", ha="center", va="center", fontsize=8)
    cbar = fig.colorbar(image, ax=ax, shrink=0.9)
    cbar.set_label("Relative endpoint risk")
    fig.savefig(out_file, dpi=int(dpi))
    plt.close(fig)


def plot_rank_shift(out_file: Path, rank_rows: Sequence[Mapping[str, object]], *, dpi: int) -> None:
    x = np.arange(3)
    fig, ax = plt.subplots(figsize=(8.8, 5.2), constrained_layout=True)
    for row in rank_rows:
        y = [float(row["physical_rank"]), float(row["no_sink_rank"]), float(row["with_sink_rank"])]
        ax.plot(x, y, marker="o", linewidth=1.8)
        ax.text(x[-1] + 0.04, y[-1], str(row["plan_id"]), va="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(["Physical", "No sink", "With sink"])
    ax.set_ylabel("Risk rank (1 = lowest)")
    ax.invert_yaxis()
    ax.grid(axis="y", alpha=0.25)
    ax.set_title("Phase 26: plan rank shifts under bystander and vascular sink modelling")
    fig.savefig(out_file, dpi=int(dpi))
    plt.close(fig)


def plot_vascular_delta(out_file: Path, delta_rows: Sequence[Mapping[str, object]], *, dpi: int) -> None:
    endpoint_keys = [endpoint.key for endpoint in PRIMARY_ENDPOINTS]
    means = []
    lowers = []
    uppers = []
    for key in endpoint_keys:
        vals = np.asarray([float(row["with_sink_minus_no_sink"]) for row in delta_rows if str(row["endpoint"]) == key])
        means.append(float(np.mean(vals)))
        lowers.append(float(np.mean(vals) - np.min(vals)))
        uppers.append(float(np.max(vals) - np.mean(vals)))
    fig, ax = plt.subplots(figsize=(11.2, 4.8), constrained_layout=True)
    x = np.arange(len(endpoint_keys))
    ax.bar(x, means, yerr=[lowers, uppers], color="#2ca02c", alpha=0.85, capsize=4)
    ax.axhline(0.0, color="#333333", linewidth=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels([endpoint.label for endpoint in PRIMARY_ENDPOINTS], rotation=25, ha="right")
    ax.set_ylabel("With-sink minus no-sink endpoint value")
    ax.set_title("Phase 26: marginal effect of anatomical vascular sink uptake")
    ax.grid(axis="y", alpha=0.25)
    fig.savefig(out_file, dpi=int(dpi))
    plt.close(fig)


def plot_assay_proxy_ablation(out_file: Path, assay_rows: Sequence[Mapping[str, object]], *, dpi: int) -> None:
    metrics = ["mean_gammah2ax_peak", "mean_tunel_peak", "cytokine_peak_roi_auc", "cytokine_valley_roi_auc"]
    labels = ["gammaH2AX peak", "TUNEL peak", "Cytokine peak AUC", "Cytokine valley AUC"]
    modes = list(MODE_LABELS.keys())
    x = np.arange(len(metrics))
    width = 0.24
    fig, ax = plt.subplots(figsize=(11.0, 4.8), constrained_layout=True)
    for idx, mode in enumerate(modes):
        values = [
            float(np.mean([float(row[metric]) for row in assay_rows if str(row["mode"]) == mode]))
            for metric in metrics
        ]
        ax.bar(x + (idx - 1) * width, values, width=width, label=MODE_LABELS[mode])
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("Mean assay proxy value (a.u.)")
    ax.set_title("Phase 26: assay-like proxy shifts across model modes")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=8)
    fig.savefig(out_file, dpi=int(dpi))
    plt.close(fig)


def plot_sensitivity_bands(out_file: Path, band_rows: Sequence[Mapping[str, object]], *, dpi: int) -> None:
    selected = ["ptv_d95", "pvdr", "brainstem_d2", "parotid_r_mean", "mean_gammah2ax_peak", "cytokine_peak_roi_auc"]
    rows = [row for row in band_rows if str(row["metric"]) in selected]
    metric_labels = {
        "ptv_d95": "PTV D95",
        "pvdr": "PVDR",
        "brainstem_d2": "Brainstem D2",
        "parotid_r_mean": "Parotid R mean",
        "mean_gammah2ax_peak": "gammaH2AX peak",
        "cytokine_peak_roi_auc": "Cytokine peak AUC",
    }
    grouped: Dict[str, List[float]] = {metric: [] for metric in selected}
    for row in rows:
        grouped[str(row["metric"])].append(float(row["relative_full_width_pct"]))
    values = [float(np.mean(grouped[metric])) if grouped[metric] else 0.0 for metric in selected]
    fig, ax = plt.subplots(figsize=(10.8, 4.8), constrained_layout=True)
    x = np.arange(len(selected))
    ax.bar(x, values, color="#9467bd", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels([metric_labels[metric] for metric in selected], rotation=25, ha="right")
    ax.set_ylabel("Mean uptake sensitivity band width (% of base)")
    ax.set_title("Phase 26: Morris-motivated vascular uptake sensitivity bands")
    ax.grid(axis="y", alpha=0.25)
    fig.savefig(out_file, dpi=int(dpi))
    plt.close(fig)


def write_quick_assessment(
    path: Path,
    *,
    rank_rows: Sequence[Mapping[str, object]],
    oar_rows: Sequence[Mapping[str, object]],
    delta_rows: Sequence[Mapping[str, object]],
    band_rows: Sequence[Mapping[str, object]],
) -> None:
    with_sink_rank_shifts = [int(row["with_sink_vs_physical_shift"]) for row in rank_rows]
    no_sink_rank_shifts = [int(row["no_sink_vs_physical_shift"]) for row in rank_rows]
    vascular_rank_shifts = [int(row["with_sink_vs_no_sink_shift"]) for row in rank_rows]
    biology_failures = [row for row in oar_rows if str(row["reinterpretation"]) == "biology_adds_failure"]
    vascular_status_changes = [row for row in oar_rows if str(row["reinterpretation"]) == "vascular_sink_changes_status"]
    vascular_margin_changes = [row for row in oar_rows if str(row["reinterpretation"]) == "vascular_sink_changes_margin"]

    mean_vascular_deltas: Dict[str, float] = {}
    for endpoint in PRIMARY_ENDPOINTS:
        vals = [float(row["with_sink_minus_no_sink"]) for row in delta_rows if str(row["endpoint"]) == endpoint.key]
        mean_vascular_deltas[endpoint.key] = float(np.mean(vals)) if vals else 0.0

    selected_band_metrics = ["brainstem_d2", "parotid_r_mean", "mean_gammah2ax_peak", "cytokine_peak_roi_auc"]
    band_widths = {
        metric: float(
            np.mean([float(row["relative_full_width_pct"]) for row in band_rows if str(row["metric"]) == metric])
        )
        for metric in selected_band_metrics
        if any(str(row["metric"]) == metric for row in band_rows)
    }

    lines = [
        "# Phase 26 quick assessment",
        "",
        "Phase 26 evaluates the same safe-core Phase 25 plans under physical-only scoring, "
        "bystander signalling without vascular sink uptake, and bystander signalling with anatomical vascular sink uptake.",
        "",
        "## Ranking signal",
        f"- Plans with any no-sink rank shift vs physical-only: {sum(shift != 0 for shift in no_sink_rank_shifts)} / {len(rank_rows)}",
        f"- Plans with any with-sink rank shift vs physical-only: {sum(shift != 0 for shift in with_sink_rank_shifts)} / {len(rank_rows)}",
        f"- Plans where adding anatomical sink changed the no-sink rank: {sum(shift != 0 for shift in vascular_rank_shifts)} / {len(rank_rows)}",
        "",
        "## OAR reinterpretation",
        f"- Biology-added OAR failures: {len(biology_failures)}",
        f"- Vascular sink status changes: {len(vascular_status_changes)}",
        f"- Vascular sink margin changes: {len(vascular_margin_changes)}",
        "",
        "## Mean vascular-sink effect, with sink minus no sink",
    ]
    for endpoint in PRIMARY_ENDPOINTS:
        lines.append(f"- {endpoint.label}: {mean_vascular_deltas[endpoint.key]:.4g} {endpoint.units}")
    lines.extend(["", "## Uptake sensitivity band widths"])
    for metric, value in band_widths.items():
        lines.append(f"- {metric}: {value:.2f}% of base with-sink value")
    lines.extend(
        [
            "",
            "Interpretation note: these bands are uptake-scale bands around the anatomical sink model. "
            "They are designed to connect the Phase 26 ablation to the prior Morris result that ROS/cytokine uptake "
            "dominates model variation; they do not replace the original Morris sensitivity analysis.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    run_root = args.phase25_run_root.resolve()
    out_root = args.out_root.resolve() if args.out_root is not None else run_root / "phase26_vascular_sink_ablation"
    out_root.mkdir(parents=True, exist_ok=True)
    figures_dir = out_root / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    print(f"Phase 26 starting from: {run_root}", flush=True)
    structures, axes_mm, config = load_phase25_context(run_root)
    manifest = json.loads((run_root / "phase25_plan_manifest.json").read_text(encoding="utf-8"))
    if int(args.max_plans) > 0:
        manifest = manifest[: int(args.max_plans)]
    print(f"Loaded {len(manifest)} Phase 25 plans.", flush=True)

    params = config["bio_parameters"]
    alpha = float(params["alpha"])
    beta = float(params["beta"])
    prescription_gy = 6.0
    voxel_size_mm = voxel_size_mm_from_axes(axes_mm)
    voxel_volume_cc = voxel_volume_cc_from_axes(axes_mm)

    endpoint_rows: List[Dict[str, object]] = []
    assay_rows: List[Dict[str, object]] = []
    sensitivity_samples: List[Dict[str, object]] = []
    base_lookup: Dict[Tuple[str, str], Dict[str, float]] = {}

    for plan_entry in manifest:
        plan_id = str(plan_entry["plan_id"])
        print(f"Processing {plan_id}...", flush=True)
        summary_json = json.loads(Path(plan_entry["summary_json"]).read_text(encoding="utf-8"))
        spots_mm = [tuple(float(v) for v in row) for row in summary_json["spots_mm"]]
        volumes = load_npz_arrays_with_retry(Path(plan_entry["course_volumes_npz"]))
        physical_dose = np.asarray(volumes["cumulative_physical_dose"], dtype=np.float32)

        mode_results: Dict[str, Dict[str, object]] = {
            "physical_only": evaluate_physical_only(
                physical_dose=physical_dose,
                structures=structures,
                axes_mm=axes_mm,
                spots_mm=spots_mm,
                voxel_volume_cc=float(voxel_volume_cc),
                voxel_size_mm=voxel_size_mm,
                prescription_gy=float(prescription_gy),
                alpha=float(alpha),
                beta=float(beta),
            )
        }
        for mode in ("bystander_no_sink", "bystander_with_sink"):
            print(f"  Evaluating {mode}...", flush=True)
            mode_results[mode] = evaluate_bystander_mode(
                physical_dose=physical_dose,
                structures=structures,
                axes_mm=axes_mm,
                spots_mm=spots_mm,
                config=config,
                mode=mode,
                uptake_scale=1.0,
                voxel_volume_cc=float(voxel_volume_cc),
                voxel_size_mm=voxel_size_mm,
                prescription_gy=float(prescription_gy),
                history_interval=int(args.history_interval),
                progress_interval=int(args.progress_interval),
                verbose_pde=bool(args.verbose_pde),
            )

        for mode, result in mode_results.items():
            row: Dict[str, object] = {"plan_id": plan_id, "mode": mode, "mode_label": MODE_LABELS[mode]}
            row.update({key: float(value) for key, value in dict(result["endpoints"]).items()})
            supplemental = dict(result["supplemental"])
            row.update(
                {
                    "gtv_d95": float(supplemental["gtv_d95"]),
                    "thyroid_mean": float(supplemental["thyroid_mean"]),
                    "body_dmax": float(supplemental["body_dmax"]),
                    "spill_shell_15_30_mean": float(supplemental["spill_shell_15_30_mean"]),
                    "outside_gtv_d2": float(supplemental["outside_gtv_d2"]),
                    "oar_adjacent_outside_gtv_mean": float(supplemental["oar_adjacent_outside_gtv_mean"]),
                    "ptv_valley_outside_gtv_mean": float(supplemental["ptv_valley_outside_gtv_mean"]),
                    "peak_mean": float(supplemental["peak_mean"]),
                    "valley_mean": float(supplemental["valley_mean"]),
                }
            )
            endpoint_rows.append(row)
            assay_row = {"plan_id": plan_id, "mode": mode, "mode_label": MODE_LABELS[mode]}
            assay_row.update({key: float(value) for key, value in dict(result["assays"]).items()})
            assay_rows.append(assay_row)
            if mode == "bystander_with_sink":
                base_lookup[(plan_id, "endpoint")] = dict(result["endpoints"])
                base_lookup[(plan_id, "assay")] = dict(result["assays"])

        for scale in sorted(set(float(v) for v in args.sensitivity_uptake_scales)):
            if abs(scale - 1.0) <= 1.0e-9:
                scaled_result = mode_results["bystander_with_sink"]
            else:
                print(f"  Sensitivity uptake scale {scale:.3g}...", flush=True)
                scaled_result = evaluate_bystander_mode(
                    physical_dose=physical_dose,
                    structures=structures,
                    axes_mm=axes_mm,
                    spots_mm=spots_mm,
                    config=config,
                    mode="bystander_with_sink",
                    uptake_scale=scale,
                    voxel_volume_cc=float(voxel_volume_cc),
                    voxel_size_mm=voxel_size_mm,
                    prescription_gy=float(prescription_gy),
                    history_interval=int(args.history_interval),
                    progress_interval=int(args.progress_interval),
                    verbose_pde=bool(args.verbose_pde),
                )
            sensitivity_samples.append(
                {
                    "plan_id": plan_id,
                    "uptake_scale": float(scale),
                    "metric_type": "endpoint",
                    "values": dict(scaled_result["endpoints"]),
                }
            )
            sensitivity_samples.append(
                {
                    "plan_id": plan_id,
                    "uptake_scale": float(scale),
                    "metric_type": "assay",
                    "values": {key: float(dict(scaled_result["assays"])[key]) for key in ASSAY_METRICS},
                }
            )

    assign_ranks(endpoint_rows)
    rank_rows = build_rank_shift_rows(endpoint_rows)
    delta_rows = build_delta_rows(endpoint_rows)
    oar_rows = build_oar_reinterpretation_rows(endpoint_rows, args=args)
    sensitivity_band_rows = build_sensitivity_band_rows(sensitivity_samples, base_lookup)

    write_csv(out_root / "phase26_endpoint_table.csv", endpoint_rows)
    write_csv(out_root / "phase26_assay_proxy_table.csv", assay_rows)
    write_csv(out_root / "phase26_rank_shift_table.csv", rank_rows)
    write_csv(out_root / "phase26_endpoint_delta_table.csv", delta_rows)
    write_csv(out_root / "phase26_oar_reinterpretation_table.csv", oar_rows)
    sample_rows: List[Dict[str, object]] = []
    for sample in sensitivity_samples:
        row = {
            "plan_id": str(sample["plan_id"]),
            "uptake_scale": float(sample["uptake_scale"]),
            "metric_type": str(sample["metric_type"]),
        }
        row.update({str(key): float(value) for key, value in dict(sample["values"]).items()})
        sample_rows.append(row)
    write_csv(out_root / "phase26_sensitivity_samples.csv", sample_rows)
    write_csv(out_root / "phase26_sensitivity_bands.csv", sensitivity_band_rows)

    write_json(
        out_root / "phase26_summary.json",
        {
            "phase": 26,
            "phase25_run_root": str(run_root),
            "modes": MODE_LABELS,
            "sensitivity_uptake_scales": [float(value) for value in args.sensitivity_uptake_scales],
            "endpoint_rows": endpoint_rows,
            "assay_rows": assay_rows,
            "rank_rows": rank_rows,
            "oar_reinterpretation_rows": oar_rows,
            "sensitivity_band_rows": sensitivity_band_rows,
        },
    )

    plot_endpoint_mode_heatmap(figures_dir / "figure1_phase26_endpoint_mode_heatmap.png", endpoint_rows, dpi=int(args.dpi))
    plot_rank_shift(figures_dir / "figure2_phase26_rank_shift.png", rank_rows, dpi=int(args.dpi))
    plot_vascular_delta(figures_dir / "figure3_phase26_vascular_sink_delta.png", delta_rows, dpi=int(args.dpi))
    plot_assay_proxy_ablation(figures_dir / "figure4_phase26_assay_proxy_ablation.png", assay_rows, dpi=int(args.dpi))
    plot_sensitivity_bands(figures_dir / "figure5_phase26_uptake_sensitivity_bands.png", sensitivity_band_rows, dpi=int(args.dpi))
    write_quick_assessment(
        out_root / "phase26_quick_assessment.md",
        rank_rows=rank_rows,
        oar_rows=oar_rows,
        delta_rows=delta_rows,
        band_rows=sensitivity_band_rows,
    )

    print("=== PHASE 26 VASCULAR SINK ABLATION COMPLETE ===")
    print(f"Output root: {out_root}")
    print(f"Endpoint table: {out_root / 'phase26_endpoint_table.csv'}")
    print(f"Quick assessment: {out_root / 'phase26_quick_assessment.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
