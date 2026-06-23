#!/usr/bin/env python3
"""Phase 29: import a TPS VMAT lattice RTDOSE/RTSTRUCT set and apply the biology model.

This phase is designed for the workflow where the physical plan is created
outside this repository in a clinical TPS (for example Eclipse, RayStation, or
matRad), then exported as DICOM:

1. CT series (optional for this analysis phase)
2. RTSTRUCT
3. RTDOSE
4. RTPLAN (optional metadata only)

The script rasterizes RTSTRUCT contours directly onto the RTDOSE grid, derives
peak/valley/peri-GTV analysis masks from the actual lattice plan, applies the
existing bystander model, and writes physical-vs-biological endpoint tables,
OAR reinterpretation tables, and assay-like proxy outputs.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import pydicom
from pydicom.dataset import Dataset
from scipy import ndimage
from skimage.draw import polygon

from bystander_multispecies_pde_solver import (
    calculate_systemic_immune_penalty,
    calculate_effective_dose,
    calculate_phase7_survival,
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


MODE_LABELS = {
    "physical_only": "Physical-only",
    "bystander_no_sink": "Bystander, no vascular sink",
    "bystander_with_sink": "Bystander, anatomical vascular sink",
}

PRIMARY_ENDPOINT_KEYS = (
    "ptv_d95",
    "pvdr",
    "spill_shell_0_5_mean",
    "spill_shell_5_15_mean",
    "cord_d2",
    "brainstem_d2",
    "parotid_r_mean",
)

CANONICAL_STRUCTURE_ALIASES: Dict[str, Tuple[str, ...]] = {
    "BODY": ("external", "body", "outline"),
    "GTV": ("gtv", "gtvp", "gtvprimary", "gross", "grossdisease", "primarygtv"),
    "PTV": ("ptv", "ptvboost", "ctvboost", "boost", "ctv", "target"),
    "BRAIN": ("brain",),
    "BRAINSTEM": ("brainstem", "stem"),
    "SPINAL_CORD": ("spinalcord", "cord", "spinalcanal", "cordprv"),
    "CHIASM": ("chiasm", "opticchiasm"),
    "OPTIC_NERVE_R": ("opticnervr", "ropticnerve", "rightopticnerve", "rtopticnerve"),
    "OPTIC_NERVE_L": ("opticnervl", "lopticnerve", "leftopticnerve", "ltopticnerve"),
    "EYE_R": ("eyer", "righteye", "rteye"),
    "EYE_L": ("eyel", "lefteye", "lteye"),
    "LENS_R": ("lensr", "rightlens", "rtlens"),
    "LENS_L": ("lensl", "leftlens", "ltlens"),
    "PAROTID_R": ("parotidr", "rightparotid", "rtparotid"),
    "PAROTID_L": ("parotidl", "leftparotid", "ltparotid"),
    "THYROID": ("thyroid",),
    "MANDIBLE": ("mandible", "jaw"),
    "ARTERIES": ("artery", "arteries", "carotid", "internalcarotid", "externalcarotid"),
    "VEINS": ("vein", "veins", "jugular", "venous"),
    "VERTEX": ("vertex", "vertices", "lattice", "sphere", "hotspot", "peak"),
}


@dataclass(frozen=True)
class DoseGeometry:
    image_position_patient: np.ndarray
    column_cosines: np.ndarray
    row_cosines: np.ndarray
    normal: np.ndarray
    row_spacing_mm: float
    column_spacing_mm: float
    frame_offsets_mm: np.ndarray
    source_order_by_axis: Tuple[str, str, str]
    flip_axes: Tuple[bool, bool, bool]
    axes_mm: Dict[str, np.ndarray]
    source_shape: Tuple[int, int, int]


@dataclass(frozen=True)
class PeakComponent:
    label: str
    voxels: int
    volume_cc: float
    max_dose_gy: float
    mean_dose_gy: float
    centroid_x_mm: float
    centroid_y_mm: float
    centroid_z_mm: float


def make_phase28_axes(voxel_mm: float) -> Dict[str, np.ndarray]:
    return {
        "x": np.arange(-90.0, 90.0 + 0.5 * voxel_mm, voxel_mm, dtype=np.float32),
        "y": np.arange(-76.0, 76.0 + 0.5 * voxel_mm, voxel_mm, dtype=np.float32),
        "z": np.arange(0.0, 122.0 + 0.5 * voxel_mm, voxel_mm, dtype=np.float32),
    }


def build_phase28_mesh(axes: Mapping[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    return np.meshgrid(axes["x"], axes["y"], axes["z"], indexing="ij")


def phase28_ellipsoid_mask(
    xg: np.ndarray,
    yg: np.ndarray,
    zg: np.ndarray,
    *,
    center: Tuple[float, float, float],
    radii: Tuple[float, float, float],
) -> np.ndarray:
    cx, cy, cz = center
    rx, ry, rz = radii
    return (((xg - cx) / rx) ** 2 + ((yg - cy) / ry) ** 2 + ((zg - cz) / rz) ** 2) <= 1.0


def phase28_sphere_mask(
    xg: np.ndarray,
    yg: np.ndarray,
    zg: np.ndarray,
    center: Tuple[float, float, float],
    radius_mm: float,
) -> np.ndarray:
    cx, cy, cz = center
    return ((xg - cx) ** 2 + (yg - cy) ** 2 + (zg - cz) ** 2) <= float(radius_mm) ** 2


def phase28_capsule_mask(
    xg: np.ndarray,
    yg: np.ndarray,
    zg: np.ndarray,
    p0: Tuple[float, float, float],
    p1: Tuple[float, float, float],
    radius_mm: float,
) -> np.ndarray:
    p0_arr = np.asarray(p0, dtype=np.float32)
    p1_arr = np.asarray(p1, dtype=np.float32)
    direction = p1_arr - p0_arr
    norm2 = float(np.dot(direction, direction))
    px = xg - float(p0_arr[0])
    py = yg - float(p0_arr[1])
    pz = zg - float(p0_arr[2])
    t = np.clip((px * direction[0] + py * direction[1] + pz * direction[2]) / max(norm2, 1.0e-6), 0.0, 1.0)
    cx = float(p0_arr[0]) + t * direction[0]
    cy = float(p0_arr[1]) + t * direction[1]
    cz = float(p0_arr[2]) + t * direction[2]
    return ((xg - cx) ** 2 + (yg - cy) ** 2 + (zg - cz) ** 2) <= float(radius_mm) ** 2


def build_phase28_structures(axes: Mapping[str, np.ndarray]) -> Dict[str, np.ndarray]:
    xg, yg, zg = build_phase28_mesh(axes)
    body = phase28_ellipsoid_mask(xg, yg, zg, center=(0.0, 0.0, 57.0), radii=(84.0, 70.0, 64.0)) & (zg >= 0.0)
    gtv = phase28_ellipsoid_mask(xg, yg, zg, center=(20.0, 0.0, 38.7), radii=(36.45, 30.0, 15.85)) & body

    cranium_outer = phase28_ellipsoid_mask(xg, yg, zg, center=(0.0, 2.0, 69.0), radii=(56.0, 48.0, 44.0)) & body
    cranium_inner = phase28_ellipsoid_mask(xg, yg, zg, center=(0.0, 2.0, 69.5), radii=(50.0, 42.0, 38.0)) & body
    midface_outer = phase28_ellipsoid_mask(xg, yg, zg, center=(8.0, 10.0, 34.0), radii=(48.0, 30.0, 24.0)) & body
    midface_inner = phase28_ellipsoid_mask(xg, yg, zg, center=(8.0, 8.0, 34.0), radii=(42.0, 24.0, 20.0)) & body
    jaw_outer = phase28_ellipsoid_mask(xg, yg, zg, center=(8.0, -3.0, 16.0), radii=(38.0, 22.0, 14.0)) & body
    jaw_inner = phase28_ellipsoid_mask(xg, yg, zg, center=(8.0, -2.0, 17.0), radii=(31.0, 17.0, 10.0)) & body
    nose = phase28_capsule_mask(xg, yg, zg, (8.0, 20.0, 28.0), (8.0, 38.0, 31.0), 5.5) & body
    cheek_r = phase28_ellipsoid_mask(xg, yg, zg, center=(32.0, 16.0, 25.0), radii=(10.0, 8.0, 10.0)) & body
    cheek_l = phase28_ellipsoid_mask(xg, yg, zg, center=(-16.0, 16.0, 25.0), radii=(10.0, 8.0, 10.0)) & body
    ear_r = phase28_ellipsoid_mask(xg, yg, zg, center=(57.0, 4.0, 33.0), radii=(5.0, 3.0, 10.0)) & body
    ear_l = phase28_ellipsoid_mask(xg, yg, zg, center=(-41.0, 4.0, 33.0), radii=(5.0, 3.0, 10.0)) & body

    head_soft = cranium_outer | midface_outer | jaw_outer | nose | cheek_r | cheek_l | ear_r | ear_l
    skull = (cranium_outer & ~cranium_inner) | (midface_outer & ~midface_inner)
    maxilla = (midface_outer & ~midface_inner) | phase28_ellipsoid_mask(
        xg, yg, zg, center=(8.0, 14.0, 22.0), radii=(30.0, 16.0, 8.0)
    )
    mandible = jaw_outer & ~jaw_inner
    brain = phase28_ellipsoid_mask(xg, yg, zg, center=(0.0, 5.0, 71.0), radii=(48.0, 40.0, 34.0)) & cranium_inner

    return {
        "BODY": body,
        "HEAD_SOFT": head_soft,
        "SKULL": skull,
        "MAXILLA": maxilla,
        "MANDIBLE": mandible,
        "BRAIN": brain,
        "GTV": gtv,
        "BRAINSTEM": phase28_ellipsoid_mask(xg, yg, zg, center=(0.0, -3.0, 78.0), radii=(9.0, 14.0, 12.0)) & body,
        "CHIASM": phase28_ellipsoid_mask(xg, yg, zg, center=(2.0, 16.0, 55.0), radii=(8.0, 4.0, 4.0)) & body,
        "OPTIC_NERVE_R": phase28_capsule_mask(xg, yg, zg, (35.0, 19.0, 18.0), (6.0, 16.0, 53.0), 2.2) & body,
        "OPTIC_NERVE_L": phase28_capsule_mask(xg, yg, zg, (-30.0, 19.0, 18.0), (-2.0, 16.0, 53.0), 2.2) & body,
        "EYE_R": phase28_ellipsoid_mask(xg, yg, zg, center=(38.0, 20.0, 16.0), radii=(12.0, 10.0, 8.0)) & body,
        "EYE_L": phase28_ellipsoid_mask(xg, yg, zg, center=(-32.0, 20.0, 16.0), radii=(12.0, 10.0, 8.0)) & body,
        "LENS_R": phase28_ellipsoid_mask(xg, yg, zg, center=(38.0, 20.0, 7.5), radii=(3.0, 3.0, 2.0)) & body,
        "LENS_L": phase28_ellipsoid_mask(xg, yg, zg, center=(-32.0, 20.0, 7.5), radii=(3.0, 3.0, 2.0)) & body,
    }


def phase28_vertex_centers() -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
    center_x = 20.0
    half_spacing_mm = 30.4 / 2.0
    return ((center_x - half_spacing_mm, 0.0, 38.7), (center_x + half_spacing_mm, 0.0, 38.7))


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


def compute_peak_valley_metrics_local(dose: np.ndarray, peak_mask: np.ndarray, valley_mask: np.ndarray) -> Dict[str, float]:
    peak_values = np.asarray(dose[np.asarray(peak_mask, dtype=bool)], dtype=np.float64)
    valley_values = np.asarray(dose[np.asarray(valley_mask, dtype=bool)], dtype=np.float64)
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
        "mean_gammah2ax_peak": float(np.mean(gamma_h2ax_map[np.asarray(peak_mask, dtype=bool)])),
        "mean_gammah2ax_valley": float(np.mean(gamma_h2ax_map[np.asarray(valley_mask, dtype=bool)])),
        "mean_tunel_peak": float(np.mean(tunel_map[np.asarray(peak_mask, dtype=bool)])),
        "mean_tunel_valley": float(np.mean(tunel_map[np.asarray(valley_mask, dtype=bool)])),
        "mean_cytokine_final_peak": float(np.mean(cytokine_final[np.asarray(peak_mask, dtype=bool)])),
        "mean_cytokine_final_valley": float(np.mean(cytokine_final[np.asarray(valley_mask, dtype=bool)])),
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


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase28-topas-root",
        type=Path,
        default=None,
        help="Use the Phase 28 TOPAS phantom and dose instead of DICOM inputs.",
    )
    parser.add_argument("--dicom-dir", type=Path, default=None, help="Directory containing exported DICOM files.")
    parser.add_argument("--ct-dir", type=Path, default=None, help="Optional CT series directory.")
    parser.add_argument("--rtstruct", type=Path, default=None, help="RTSTRUCT DICOM file.")
    parser.add_argument("--rtdose", type=Path, default=None, help="RTDOSE DICOM file.")
    parser.add_argument("--rtplan", type=Path, default=None, help="Optional RTPLAN DICOM file.")
    parser.add_argument(
        "--out-root",
        type=Path,
        default=root / "runs" / "phase29_vmat_dicom_bio_import",
        help="Output directory for imported-plan biology analysis.",
    )
    parser.add_argument(
        "--target-prescription-gy",
        type=float,
        default=3.5,
        help="Prescription assigned to the peripheral target volume for D95/V95 reporting.",
    )
    parser.add_argument(
        "--peak-threshold-fraction",
        type=float,
        default=0.80,
        help="Peak detection threshold as a fraction of the in-target maximum dose when no explicit vertex ROIs exist.",
    )
    parser.add_argument(
        "--peak-threshold-gy",
        type=float,
        default=0.0,
        help="Optional absolute threshold for peak detection; 0 uses the fractional rule.",
    )
    parser.add_argument("--min-peak-component-voxels", type=int, default=3)
    parser.add_argument("--valley-exclusion-mm", type=float, default=14.0)
    parser.add_argument("--shell-1-mm", type=float, default=5.0)
    parser.add_argument("--shell-2-mm", type=float, default=15.0)
    parser.add_argument("--shell-3-mm", type=float, default=30.0)
    parser.add_argument("--oar-adjacent-mm", type=float, default=15.0)
    parser.add_argument("--alpha", type=float, default=0.03)
    parser.add_argument("--beta", type=float, default=0.003)
    parser.add_argument("--pde-steps", type=int, default=400)
    parser.add_argument("--pde-dt", type=float, default=0.12)
    parser.add_argument("--history-interval", type=int, default=20)
    parser.add_argument("--progress-interval", type=int, default=100)
    parser.add_argument("--verbose-pde", action="store_true")
    parser.add_argument("--tumor-cytokine-multiplier", type=float, default=2.0)
    parser.add_argument("--hypoxic-ros-scale", type=float, default=0.12)
    parser.add_argument("--hypoxic-cytokine-multiplier", type=float, default=2.7)
    parser.add_argument("--artery-ros-uptake", type=float, default=0.05)
    parser.add_argument("--artery-cyto-uptake", type=float, default=0.70)
    parser.add_argument("--vein-ros-uptake", type=float, default=0.05)
    parser.add_argument("--vein-cyto-uptake", type=float, default=0.90)
    parser.add_argument("--hypoxia-core-mm", type=float, default=6.0)
    parser.add_argument(
        "--topas-scale-mode",
        choices=("peak_mean", "peak_max", "ptv_d95", "none"),
        default="peak_mean",
        help="Normalization mode used when --phase28-topas-root is supplied.",
    )
    parser.add_argument(
        "--topas-scale-gy",
        type=float,
        default=15.0,
        help="Target dose in Gy for peak-based Phase 28 TOPAS normalization modes.",
    )
    parser.add_argument(
        "--require-vessels",
        action="store_true",
        help="Fail if artery/vein contours are absent instead of falling back to no-sink equivalence.",
    )
    return parser.parse_args()


def write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        raise ValueError(f"No rows available for {path}")
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


def normalize_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(name).lower())


def discover_dicom_inputs(dicom_dir: Path) -> Dict[str, Path]:
    discovered: Dict[str, List[Path]] = {"CT": [], "RTSTRUCT": [], "RTDOSE": [], "RTPLAN": []}
    for path in sorted(dicom_dir.rglob("*")):
        if not path.is_file():
            continue
        try:
            ds = pydicom.dcmread(path, stop_before_pixels=True, force=True)
        except Exception:
            continue
        modality = str(getattr(ds, "Modality", "")).upper()
        if modality in discovered:
            discovered[modality].append(path)
    resolved: Dict[str, Path] = {}
    for modality, paths in discovered.items():
        if paths:
            resolved[modality] = paths[0]
    return resolved


def resolve_input_paths(args: argparse.Namespace) -> Dict[str, Path | None]:
    if args.phase28_topas_root is not None:
        return {"ct_dir": None, "rtstruct": None, "rtdose": None, "rtplan": None}
    discovered: Dict[str, Path] = {}
    if args.dicom_dir is not None:
        discovered = discover_dicom_inputs(Path(args.dicom_dir))
    ct_dir = Path(args.ct_dir) if args.ct_dir is not None else None
    if ct_dir is None and args.dicom_dir is not None and "CT" in discovered:
        ct_dir = Path(args.dicom_dir)
    rtstruct = Path(args.rtstruct) if args.rtstruct is not None else discovered.get("RTSTRUCT")
    rtdose = Path(args.rtdose) if args.rtdose is not None else discovered.get("RTDOSE")
    rtplan = Path(args.rtplan) if args.rtplan is not None else discovered.get("RTPLAN")
    if rtstruct is None or rtdose is None:
        raise FileNotFoundError("RTSTRUCT and RTDOSE are required. Provide them explicitly or via --dicom-dir.")
    return {"ct_dir": ct_dir, "rtstruct": rtstruct, "rtdose": rtdose, "rtplan": rtplan}


def load_topas_csv_grid(path: Path) -> np.ndarray:
    xbins = ybins = zbins = None
    header_lines = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.startswith("#"):
                break
            header_lines += 1
            stripped = line.strip()
            if stripped.startswith("# X in "):
                xbins = int(stripped.split()[3])
            elif stripped.startswith("# Y in "):
                ybins = int(stripped.split()[3])
            elif stripped.startswith("# Z in "):
                zbins = int(stripped.split()[3])
    if xbins is None or ybins is None or zbins is None:
        raise RuntimeError(f"Unable to parse TOPAS grid dimensions from {path}")
    data = np.loadtxt(path, delimiter=",", comments="#", dtype=np.float32)
    if data.ndim != 2 or data.shape[1] != 4:
        raise RuntimeError(f"Unexpected TOPAS CSV layout in {path}")
    grid = np.zeros((int(xbins), int(ybins), int(zbins)), dtype=np.float32)
    ix = data[:, 0].astype(np.int32)
    iy = data[:, 1].astype(np.int32)
    iz = data[:, 2].astype(np.int32)
    grid[ix, iy, iz] = data[:, 3]
    return grid


def _dominant_axis(vector: np.ndarray, *, tol: float = 1.0e-3) -> Tuple[int, float]:
    idx = int(np.argmax(np.abs(vector)))
    component = float(vector[idx])
    if abs(abs(component) - 1.0) > tol:
        raise ValueError(
            "Only axis-aligned RTDOSE orientations are currently supported. "
            f"Got direction vector {vector.tolist()}."
        )
    return idx, component


def source_array_to_xyz(array_src: np.ndarray, geometry: DoseGeometry) -> np.ndarray:
    source_dim_index = {"frame": 0, "row": 1, "col": 2}
    order = tuple(source_dim_index[name] for name in geometry.source_order_by_axis)
    array_xyz = np.transpose(array_src, axes=order)
    for axis_idx, flip in enumerate(geometry.flip_axes):
        if flip:
            array_xyz = np.flip(array_xyz, axis=axis_idx)
    return np.asarray(array_xyz)


def load_rtdose(path: Path) -> Tuple[np.ndarray, DoseGeometry, Dataset]:
    ds = pydicom.dcmread(path)
    pixel_array = np.asarray(ds.pixel_array, dtype=np.float32)
    if pixel_array.ndim == 2:
        pixel_array = pixel_array[None, :, :]
    dose_src = pixel_array * float(ds.DoseGridScaling)

    image_position = np.asarray(ds.ImagePositionPatient, dtype=np.float64)
    orientation = np.asarray(ds.ImageOrientationPatient, dtype=np.float64)
    column_cosines = orientation[:3]
    row_cosines = orientation[3:]
    normal = np.cross(column_cosines, row_cosines)
    row_spacing_mm = float(ds.PixelSpacing[0])
    column_spacing_mm = float(ds.PixelSpacing[1])
    if hasattr(ds, "GridFrameOffsetVector"):
        frame_offsets_mm = np.asarray(ds.GridFrameOffsetVector, dtype=np.float64)
    else:
        spacing = float(getattr(ds, "SliceThickness", 1.0))
        frame_offsets_mm = np.arange(dose_src.shape[0], dtype=np.float64) * spacing
    if frame_offsets_mm.size != dose_src.shape[0]:
        raise ValueError(
            f"RTDOSE GridFrameOffsetVector length {frame_offsets_mm.size} does not match frame count {dose_src.shape[0]}."
        )

    dim_vectors = {
        "col": column_cosines,
        "row": row_cosines,
        "frame": normal if frame_offsets_mm.size < 2 else normal * float(frame_offsets_mm[-1] - frame_offsets_mm[0]),
    }
    mapped_axes: Dict[str, int] = {}
    coordinate_arrays: Dict[str, np.ndarray] = {}
    counts = {"col": dose_src.shape[2], "row": dose_src.shape[1], "frame": dose_src.shape[0]}
    for dim_name, vector in dim_vectors.items():
        axis_idx, component = _dominant_axis(vector)
        mapped_axes[dim_name] = axis_idx
        if dim_name == "col":
            coords = image_position[axis_idx] + np.arange(counts[dim_name], dtype=np.float64) * column_spacing_mm * component
        elif dim_name == "row":
            coords = image_position[axis_idx] + np.arange(counts[dim_name], dtype=np.float64) * row_spacing_mm * component
        else:
            coords = image_position[axis_idx] + frame_offsets_mm * float(normal[axis_idx])
        coordinate_arrays[dim_name] = np.asarray(coords, dtype=np.float32)
    if len(set(mapped_axes.values())) != 3:
        raise ValueError("RTDOSE axes are not uniquely mappable onto patient x/y/z.")

    axis_names = ("x", "y", "z")
    source_order_by_axis = tuple(
        next(dim_name for dim_name, axis_idx in mapped_axes.items() if axis_idx == target_axis)
        for target_axis in range(3)
    )
    flip_axes: List[bool] = []
    axes_mm: Dict[str, np.ndarray] = {}
    for axis_name, source_dim in zip(axis_names, source_order_by_axis):
        coords = np.asarray(coordinate_arrays[source_dim], dtype=np.float32)
        should_flip = bool(coords[0] > coords[-1])
        flip_axes.append(should_flip)
        axes_mm[axis_name] = coords[::-1].copy() if should_flip else coords.copy()

    geometry = DoseGeometry(
        image_position_patient=image_position,
        column_cosines=np.asarray(column_cosines, dtype=np.float64),
        row_cosines=np.asarray(row_cosines, dtype=np.float64),
        normal=np.asarray(normal, dtype=np.float64),
        row_spacing_mm=float(row_spacing_mm),
        column_spacing_mm=float(column_spacing_mm),
        frame_offsets_mm=np.asarray(frame_offsets_mm, dtype=np.float64),
        source_order_by_axis=source_order_by_axis,
        flip_axes=tuple(flip_axes),
        axes_mm=axes_mm,
        source_shape=tuple(int(v) for v in dose_src.shape),
    )
    dose_xyz = source_array_to_xyz(dose_src, geometry).astype(np.float32, copy=False)
    return dose_xyz, geometry, ds


def patient_points_to_source_indices(points_mm: np.ndarray, geometry: DoseGeometry) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    relative = np.asarray(points_mm, dtype=np.float64) - geometry.image_position_patient[None, :]
    column = np.dot(relative, geometry.column_cosines) / float(geometry.column_spacing_mm)
    row = np.dot(relative, geometry.row_cosines) / float(geometry.row_spacing_mm)
    frame_mm = np.dot(relative, geometry.normal)
    return row, column, frame_mm


def contour_to_frame_index(points_mm: np.ndarray, geometry: DoseGeometry) -> int | None:
    _, _, frame_mm = patient_points_to_source_indices(points_mm, geometry)
    mean_frame_mm = float(np.mean(frame_mm))
    frame_offsets = np.asarray(geometry.frame_offsets_mm, dtype=np.float64)
    frame_idx = int(np.argmin(np.abs(frame_offsets - mean_frame_mm)))
    if frame_offsets.size > 1:
        spacing = float(np.median(np.abs(np.diff(frame_offsets))))
    else:
        spacing = max(float(geometry.row_spacing_mm), float(geometry.column_spacing_mm))
    tolerance = max(1.5, 0.5 * spacing + 0.5)
    if abs(float(frame_offsets[frame_idx]) - mean_frame_mm) > tolerance:
        return None
    return frame_idx


def rasterize_rtstruct(rtstruct_path: Path, geometry: DoseGeometry) -> Tuple[Dict[str, np.ndarray], Dict[str, Dict[str, object]], Dataset]:
    ds = pydicom.dcmread(rtstruct_path)
    roi_number_to_name = {
        int(item.ROINumber): str(item.ROIName)
        for item in getattr(ds, "StructureSetROISequence", [])
    }
    masks: Dict[str, np.ndarray] = {}
    metadata: Dict[str, Dict[str, object]] = {}
    rows = int(geometry.source_shape[1])
    cols = int(geometry.source_shape[2])
    for roi_contour in getattr(ds, "ROIContourSequence", []):
        roi_number = int(getattr(roi_contour, "ReferencedROINumber", -1))
        roi_name = roi_number_to_name.get(roi_number, f"ROI_{roi_number}")
        slice_accumulator: Dict[int, np.ndarray] = {}
        skipped_contours = 0
        total_contours = 0
        for contour in getattr(roi_contour, "ContourSequence", []):
            if str(getattr(contour, "ContourGeometricType", "")).upper() != "CLOSED_PLANAR":
                skipped_contours += 1
                continue
            data = np.asarray(getattr(contour, "ContourData", []), dtype=np.float64)
            if data.size < 9:
                skipped_contours += 1
                continue
            points_mm = data.reshape(-1, 3)
            frame_idx = contour_to_frame_index(points_mm, geometry)
            if frame_idx is None:
                skipped_contours += 1
                continue
            row_idx, col_idx, _ = patient_points_to_source_indices(points_mm, geometry)
            rr, cc = polygon(row_idx, col_idx, shape=(rows, cols))
            plane_mask = np.zeros((rows, cols), dtype=bool)
            plane_mask[rr, cc] = True
            if int(np.count_nonzero(plane_mask)) == 0:
                skipped_contours += 1
                continue
            if frame_idx not in slice_accumulator:
                slice_accumulator[frame_idx] = plane_mask
            else:
                slice_accumulator[frame_idx] = np.logical_xor(slice_accumulator[frame_idx], plane_mask)
            total_contours += 1
        source_mask = np.zeros(geometry.source_shape, dtype=bool)
        for frame_idx, plane_mask in slice_accumulator.items():
            source_mask[frame_idx] = plane_mask
        masks[roi_name] = source_array_to_xyz(source_mask, geometry)
        metadata[roi_name] = {
            "roi_number": roi_number,
            "used_closed_planar_contours": int(total_contours),
            "skipped_contours": int(skipped_contours),
            "voxels": int(np.count_nonzero(masks[roi_name])),
        }
    return masks, metadata, ds


def voxel_size_mm_from_axes(axes_mm: Mapping[str, np.ndarray]) -> Tuple[float, float, float]:
    return (
        float(axes_mm["x"][1] - axes_mm["x"][0]),
        float(axes_mm["y"][1] - axes_mm["y"][0]),
        float(axes_mm["z"][1] - axes_mm["z"][0]),
    )


def voxel_volume_cc_from_axes(axes_mm: Mapping[str, np.ndarray]) -> float:
    dx, dy, dz = voxel_size_mm_from_axes(axes_mm)
    return float(dx * dy * dz / 1000.0)


def union_named_masks(mask_map: Mapping[str, np.ndarray], names: Sequence[str], shape: Tuple[int, int, int]) -> np.ndarray:
    union = np.zeros(shape, dtype=bool)
    for name in names:
        union |= np.asarray(mask_map[name], dtype=bool)
    return union


def _match_score(normalized_name: str, normalized_alias: str) -> int:
    if normalized_name == normalized_alias:
        return 100
    if normalized_name.startswith(normalized_alias) or normalized_name.endswith(normalized_alias):
        return 80
    if normalized_alias in normalized_name:
        return 60
    return 0


def map_canonical_structures(mask_map: Mapping[str, np.ndarray]) -> Tuple[Dict[str, np.ndarray], Dict[str, List[str]]]:
    shape = next(iter(mask_map.values())).shape
    normalized_lookup = {name: normalize_name(name) for name in mask_map}
    matches: Dict[str, List[str]] = {}
    for canonical, aliases in CANONICAL_STRUCTURE_ALIASES.items():
        chosen: List[Tuple[int, str]] = []
        normalized_aliases = tuple(normalize_name(alias) for alias in aliases)
        for name, normalized_name in normalized_lookup.items():
            score = max(_match_score(normalized_name, alias) for alias in normalized_aliases)
            if score > 0:
                chosen.append((score, name))
        if not chosen:
            continue
        best_score = max(score for score, _ in chosen)
        matches[canonical] = sorted(name for score, name in chosen if score == best_score)

    canonical_masks: Dict[str, np.ndarray] = {}
    for canonical, names in matches.items():
        canonical_masks[canonical] = union_named_masks(mask_map, names, shape)

    if "BODY" not in canonical_masks:
        canonical_masks["BODY"] = np.ones(shape, dtype=bool)
        matches["BODY"] = []
    if "PTV" not in canonical_masks:
        if "GTV" in canonical_masks:
            canonical_masks["PTV"] = np.asarray(canonical_masks["GTV"], dtype=bool).copy()
            matches["PTV"] = ["<fallback:GTV>"]
    if "GTV" not in canonical_masks and "PTV" in canonical_masks:
        canonical_masks["GTV"] = np.asarray(canonical_masks["PTV"], dtype=bool).copy()
        matches["GTV"] = ["<fallback:PTV>"]
    return canonical_masks, matches


def derive_hypoxia_mask(gtv_mask: np.ndarray, axes_mm: Mapping[str, np.ndarray], core_mm: float) -> np.ndarray:
    if int(np.count_nonzero(gtv_mask)) == 0:
        return np.zeros_like(gtv_mask, dtype=bool)
    distance_inside = ndimage.distance_transform_edt(np.asarray(gtv_mask, dtype=bool), sampling=voxel_size_mm_from_axes(axes_mm))
    hypoxia_mask = np.asarray(gtv_mask, dtype=bool) & (distance_inside >= float(core_mm))
    if int(np.count_nonzero(hypoxia_mask)) > 0:
        return hypoxia_mask
    target_distance = float(np.percentile(distance_inside[gtv_mask], 75.0))
    hypoxia_mask = np.asarray(gtv_mask, dtype=bool) & (distance_inside >= target_distance)
    if int(np.count_nonzero(hypoxia_mask)) > 0:
        return hypoxia_mask
    max_index = np.unravel_index(int(np.argmax(distance_inside)), distance_inside.shape)
    fallback = np.zeros_like(gtv_mask, dtype=bool)
    fallback[max_index] = True
    return fallback


def build_phase28_ptv_mask(gtv_mask: np.ndarray, body_mask: np.ndarray, axes_mm: Mapping[str, np.ndarray], margin_mm: float) -> np.ndarray:
    distance_to_gtv = ndimage.distance_transform_edt(~np.asarray(gtv_mask, dtype=bool), sampling=voxel_size_mm_from_axes(axes_mm))
    return np.asarray(body_mask, dtype=bool) & ((distance_to_gtv <= float(margin_mm)) | np.asarray(gtv_mask, dtype=bool))


def build_phase28_vertex_mask(axes_mm: Mapping[str, np.ndarray], radius_mm: float) -> np.ndarray:
    xg, yg, zg = build_phase28_mesh(axes_mm)
    mask = np.zeros((axes_mm["x"].size, axes_mm["y"].size, axes_mm["z"].size), dtype=bool)
    for center in phase28_vertex_centers():
        mask |= phase28_sphere_mask(xg, yg, zg, center, float(radius_mm))
    return mask


def scale_topas_phantom_dose(
    dose_grid: np.ndarray,
    *,
    ptv_mask: np.ndarray,
    peak_mask: np.ndarray,
    args: argparse.Namespace,
) -> Tuple[np.ndarray, Dict[str, object]]:
    mode = str(args.topas_scale_mode)
    scaled = np.asarray(dose_grid, dtype=np.float32).copy()
    values = {
        "peak_mean": float(np.mean(scaled[np.asarray(peak_mask, dtype=bool)])),
        "peak_max": float(np.max(scaled[np.asarray(peak_mask, dtype=bool)])),
        "ptv_d95": float(np.percentile(scaled[np.asarray(ptv_mask, dtype=bool)], 5.0)),
    }
    if mode == "none":
        scale_factor = 1.0
        target_gy = None
    elif mode == "ptv_d95":
        target_gy = float(args.target_prescription_gy)
        scale_factor = target_gy / max(values["ptv_d95"], 1.0e-12)
    elif mode == "peak_max":
        target_gy = float(args.topas_scale_gy)
        scale_factor = target_gy / max(values["peak_max"], 1.0e-12)
    else:
        target_gy = float(args.topas_scale_gy)
        scale_factor = target_gy / max(values["peak_mean"], 1.0e-12)
    scaled *= float(scale_factor)
    return scaled, {
        "mode": mode,
        "target_gy": target_gy,
        "scale_factor": float(scale_factor),
        "pre_scale_peak_mean_gy": values["peak_mean"],
        "pre_scale_peak_max_gy": values["peak_max"],
        "pre_scale_ptv_d95_gy": values["ptv_d95"],
        "post_scale_peak_mean_gy": float(np.mean(scaled[np.asarray(peak_mask, dtype=bool)])),
        "post_scale_peak_max_gy": float(np.max(scaled[np.asarray(peak_mask, dtype=bool)])),
        "post_scale_ptv_d95_gy": float(np.percentile(scaled[np.asarray(ptv_mask, dtype=bool)], 5.0)),
    }


def load_phase28_topas_phantom(args: argparse.Namespace) -> Tuple[np.ndarray, Dict[str, np.ndarray], Dict[str, np.ndarray], np.ndarray, Dict[str, Sequence[str]], Dict[str, object]]:
    run_root = Path(args.phase28_topas_root).expanduser().resolve()
    summary_path = run_root / "phase28_topas_tsimagecube_summary.json"
    dose_csv = run_root / "case" / "dosedata.csv"
    if not summary_path.exists():
        raise FileNotFoundError(f"Could not find Phase 28 TsImageCube summary: {summary_path}")
    if not dose_csv.exists():
        raise FileNotFoundError(f"Could not find TOPAS dose CSV: {dose_csv}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    voxel_mm = float(summary["voxel_mm"])
    axes_mm = make_phase28_axes(voxel_mm)
    structures = {key: np.asarray(value, dtype=bool) for key, value in build_phase28_structures(axes_mm).items()}
    structures["PTV"] = build_phase28_ptv_mask(structures["GTV"], structures["BODY"], axes_mm, margin_mm=5.0)
    structures["HYPOXIA"] = derive_hypoxia_mask(structures["GTV"], axes_mm, float(args.hypoxia_core_mm))
    structures["ARTERIES"] = np.zeros_like(structures["BODY"], dtype=bool)
    structures["VEINS"] = np.zeros_like(structures["BODY"], dtype=bool)
    explicit_peak_mask = build_phase28_vertex_mask(axes_mm, radius_mm=5.0) & np.asarray(structures["PTV"], dtype=bool)
    raw_dose = load_topas_csv_grid(dose_csv)
    if raw_dose.shape != tuple(np.asarray(structures["BODY"], dtype=bool).shape):
        raise RuntimeError(
            f"Phase 28 TOPAS dose shape {raw_dose.shape} does not match phantom shape {structures['BODY'].shape}."
        )
    scaled_dose, dose_scaling = scale_topas_phantom_dose(
        raw_dose,
        ptv_mask=structures["PTV"],
        peak_mask=explicit_peak_mask,
        args=args,
    )
    structure_sources = {
        "BODY": ["<phase28 synthetic body>"],
        "GTV": ["<phase28 synthetic gtv>"],
        "PTV": ["<phase28 synthetic ctvboost=gtv+5mm>"],
        "VERTEX": ["<phase28 synthetic vertices>"],
        "BRAIN": ["<phase28 synthetic anatomy>"],
        "BRAINSTEM": ["<phase28 synthetic anatomy>"],
        "CHIASM": ["<phase28 synthetic anatomy>"],
        "OPTIC_NERVE_R": ["<phase28 synthetic anatomy>"],
        "OPTIC_NERVE_L": ["<phase28 synthetic anatomy>"],
    }
    phantom_meta = {
        "source_type": "phase28_topas_phantom",
        "run_root": str(run_root),
        "dose_csv": str(dose_csv),
        "voxel_mm": voxel_mm,
        "grid_shape_xyz": list(map(int, scaled_dose.shape)),
        "dose_scaling": dose_scaling,
    }
    return scaled_dose, structures, axes_mm, explicit_peak_mask, structure_sources, phantom_meta


def centroid_mm(mask: np.ndarray, axes_mm: Mapping[str, np.ndarray]) -> Tuple[float, float, float]:
    indices = np.argwhere(mask)
    center_idx = np.round(indices.mean(axis=0)).astype(int)
    return (
        float(axes_mm["x"][center_idx[0]]),
        float(axes_mm["y"][center_idx[1]]),
        float(axes_mm["z"][center_idx[2]]),
    )


def derive_peak_components(
    *,
    dose_grid: np.ndarray,
    axes_mm: Mapping[str, np.ndarray],
    target_mask: np.ndarray,
    explicit_peak_mask: np.ndarray | None,
    threshold_fraction: float,
    threshold_gy: float,
    min_component_voxels: int,
) -> Tuple[np.ndarray, List[PeakComponent], Dict[str, object]]:
    if explicit_peak_mask is not None and int(np.count_nonzero(explicit_peak_mask)) > 0:
        candidate_mask = np.asarray(explicit_peak_mask, dtype=bool) & np.asarray(target_mask, dtype=bool)
        method = "explicit_vertex_rois"
        threshold_used = float("nan")
    else:
        target_values = np.asarray(dose_grid[np.asarray(target_mask, dtype=bool)], dtype=np.float32)
        if target_values.size == 0:
            raise RuntimeError("Target mask is empty; cannot derive peak components.")
        if float(threshold_gy) > 0.0:
            threshold_used = float(threshold_gy)
        else:
            threshold_used = float(threshold_fraction) * float(np.max(target_values))
        candidate_mask = (np.asarray(dose_grid, dtype=np.float32) >= threshold_used) & np.asarray(target_mask, dtype=bool)
        if int(np.count_nonzero(candidate_mask)) == 0:
            threshold_used = float(np.percentile(target_values, 95.0))
            candidate_mask = (np.asarray(dose_grid, dtype=np.float32) >= threshold_used) & np.asarray(target_mask, dtype=bool)
        method = "dose_threshold"

    labels, count = ndimage.label(candidate_mask)
    voxel_volume_cc = voxel_volume_cc_from_axes(axes_mm)
    components: List[PeakComponent] = []
    peak_mask = np.zeros_like(candidate_mask, dtype=bool)
    for component_idx in range(1, int(count) + 1):
        component_mask = labels == component_idx
        voxels = int(np.count_nonzero(component_mask))
        if voxels < int(min_component_voxels):
            continue
        peak_mask |= component_mask
        cx, cy, cz = centroid_mm(component_mask, axes_mm)
        values = np.asarray(dose_grid[component_mask], dtype=np.float32)
        components.append(
            PeakComponent(
                label=f"peak_{len(components) + 1}",
                voxels=voxels,
                volume_cc=float(voxels * voxel_volume_cc),
                max_dose_gy=float(np.max(values)),
                mean_dose_gy=float(np.mean(values)),
                centroid_x_mm=cx,
                centroid_y_mm=cy,
                centroid_z_mm=cz,
            )
        )
    if int(np.count_nonzero(peak_mask)) == 0:
        raise RuntimeError("Unable to derive any peak components from the imported plan.")
    summary = {
        "method": method,
        "threshold_gy": None if not np.isfinite(threshold_used) else float(threshold_used),
        "component_count": int(len(components)),
        "peak_voxels": int(np.count_nonzero(peak_mask)),
    }
    return peak_mask, components, summary


def build_valley_mask(target_mask: np.ndarray, peak_mask: np.ndarray, axes_mm: Mapping[str, np.ndarray], exclusion_mm: float) -> np.ndarray:
    distance_to_peak = ndimage.distance_transform_edt(~np.asarray(peak_mask, dtype=bool), sampling=voxel_size_mm_from_axes(axes_mm))
    valley_mask = np.asarray(target_mask, dtype=bool) & (distance_to_peak > float(exclusion_mm))
    if int(np.count_nonzero(valley_mask)) > 0:
        return valley_mask
    valley_mask = np.asarray(target_mask, dtype=bool) & ~np.asarray(peak_mask, dtype=bool)
    if int(np.count_nonzero(valley_mask)) > 0:
        return valley_mask
    target_values = np.asarray(distance_to_peak[np.asarray(target_mask, dtype=bool)], dtype=np.float32)
    relaxed = float(np.percentile(target_values, 50.0))
    return np.asarray(target_mask, dtype=bool) & (distance_to_peak >= relaxed)


def safe_metric(metrics: Mapping[str, Mapping[str, float]], structure_name: str, key: str) -> float:
    if structure_name not in metrics:
        return float("nan")
    return float(metrics[structure_name].get(key, float("nan")))


def extract_endpoints_for_masks(
    dose_grid: np.ndarray,
    *,
    structures: Mapping[str, np.ndarray],
    axes_mm: Mapping[str, np.ndarray],
    peak_mask: np.ndarray,
    valley_mask: np.ndarray,
    voxel_volume_cc: float,
    target_prescription_gy: float,
    shell_1_mm: float,
    shell_2_mm: float,
    shell_3_mm: float,
    oar_adjacent_mm: float,
) -> Tuple[Dict[str, float], Dict[str, object]]:
    metrics: Dict[str, Dict[str, float]] = {}
    for name in (
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
        "OPTIC_NERVE_R",
        "OPTIC_NERVE_L",
        "CHIASM",
        "EYE_R",
        "EYE_L",
        "LENS_R",
        "LENS_L",
    ):
        mask = np.asarray(structures.get(name, np.zeros_like(dose_grid, dtype=bool)), dtype=bool)
        if int(np.count_nonzero(mask)) == 0:
            continue
        prescription = float(target_prescription_gy) if name in {"PTV", "GTV"} else None
        metrics[name] = compute_structure_metrics_local(
            dose_grid,
            mask,
            prescription_gy=prescription,
            voxel_volume_cc=float(voxel_volume_cc),
        )

    pvdr = compute_peak_valley_metrics_local(dose_grid, np.asarray(peak_mask, dtype=bool), np.asarray(valley_mask, dtype=bool))
    spill_masks = build_spill_region_masks_local(
        structures=dict(structures),
        axes_mm=dict(axes_mm),
        peak_mask=np.asarray(peak_mask, dtype=bool),
        shell_1_mm=float(shell_1_mm),
        shell_2_mm=float(shell_2_mm),
        shell_3_mm=float(shell_3_mm),
        oar_adjacent_mm=float(oar_adjacent_mm),
    )
    spill_metrics = compute_region_metrics_local(dose_grid, spill_masks, voxel_volume_cc=float(voxel_volume_cc))

    endpoints = {
        "ptv_d95": safe_metric(metrics, "PTV", "d95_gy"),
        "pvdr": float(pvdr["pvdr"]),
        "spill_shell_0_5_mean": float(spill_metrics.get("SPILL_SHELL_0_5", {}).get("mean_gy", float("nan"))),
        "spill_shell_5_15_mean": float(spill_metrics.get("SPILL_SHELL_5_15", {}).get("mean_gy", float("nan"))),
        "cord_d2": safe_metric(metrics, "SPINAL_CORD", "d2_gy"),
        "brainstem_d2": safe_metric(metrics, "BRAINSTEM", "d2_gy"),
        "parotid_r_mean": safe_metric(metrics, "PAROTID_R", "mean_gy"),
    }
    supplemental = {
        "gtv_d95": safe_metric(metrics, "GTV", "d95_gy"),
        "peak_mean": float(pvdr["peak_mean_gy"]),
        "valley_mean": float(pvdr["valley_mean_gy"]),
        "peak_voxels": int(pvdr["peak_voxels"]),
        "valley_voxels": int(pvdr["valley_voxels"]),
        "metrics": metrics,
        "spill_metrics": spill_metrics,
        "body_dmax": safe_metric(metrics, "BODY", "dmax_gy"),
        "brain_dmax": safe_metric(metrics, "BRAIN", "dmax_gy"),
        "optic_nerve_r_dmax": safe_metric(metrics, "OPTIC_NERVE_R", "dmax_gy"),
        "optic_nerve_l_dmax": safe_metric(metrics, "OPTIC_NERVE_L", "dmax_gy"),
        "chiasm_dmax": safe_metric(metrics, "CHIASM", "dmax_gy"),
        "eye_r_dmax": safe_metric(metrics, "EYE_R", "dmax_gy"),
        "eye_l_dmax": safe_metric(metrics, "EYE_L", "dmax_gy"),
        "lens_r_dmax": safe_metric(metrics, "LENS_R", "dmax_gy"),
        "lens_l_dmax": safe_metric(metrics, "LENS_L", "dmax_gy"),
        "parotid_l_mean": safe_metric(metrics, "PAROTID_L", "mean_gy"),
        "thyroid_mean": safe_metric(metrics, "THYROID", "mean_gy"),
        "outside_gtv_d2": float(spill_metrics.get("OUTSIDE_GTV", {}).get("d2_gy", float("nan"))),
        "oar_adjacent_outside_gtv_mean": float(
            spill_metrics.get("OAR_ADJACENT_OUTSIDE_GTV", {}).get("mean_gy", float("nan"))
        ),
    }
    return endpoints, supplemental


def build_plan_assessment_rows(
    mode_results: Mapping[str, Mapping[str, object]],
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]], List[Dict[str, object]]]:
    endpoint_rows: List[Dict[str, object]] = []
    assay_rows: List[Dict[str, object]] = []
    oar_rows: List[Dict[str, object]] = []
    for mode, result in mode_results.items():
        row: Dict[str, object] = {"mode": mode, "mode_label": MODE_LABELS[mode]}
        row.update(result["endpoints"])
        row.update({key: value for key, value in result["supplemental"].items() if isinstance(value, (int, float))})
        endpoint_rows.append(row)

        assay_row: Dict[str, object] = {"mode": mode, "mode_label": MODE_LABELS[mode]}
        assay_row.update(result["assays"])
        assay_rows.append(assay_row)

    physical = mode_results["physical_only"]
    no_sink = mode_results["bystander_no_sink"]
    with_sink = mode_results["bystander_with_sink"]
    reinterpret_specs = (
        ("SPINAL_CORD", "d2_gy", "Cord D2"),
        ("BRAINSTEM", "d2_gy", "Brainstem D2"),
        ("PAROTID_R", "mean_gy", "Parotid R mean"),
        ("PAROTID_L", "mean_gy", "Parotid L mean"),
        ("BRAIN", "dmax_gy", "Brain Dmax"),
        ("CHIASM", "dmax_gy", "Chiasm Dmax"),
        ("OPTIC_NERVE_R", "dmax_gy", "Optic Nerve R Dmax"),
        ("OPTIC_NERVE_L", "dmax_gy", "Optic Nerve L Dmax"),
        ("EYE_R", "dmax_gy", "Eye R Dmax"),
        ("EYE_L", "dmax_gy", "Eye L Dmax"),
        ("LENS_R", "dmax_gy", "Lens R Dmax"),
        ("LENS_L", "dmax_gy", "Lens L Dmax"),
    )
    for structure_name, metric_key, label in reinterpret_specs:
        phys_value = safe_metric(physical["supplemental"]["metrics"], structure_name, metric_key)
        no_sink_value = safe_metric(no_sink["supplemental"]["metrics"], structure_name, metric_key)
        with_sink_value = safe_metric(with_sink["supplemental"]["metrics"], structure_name, metric_key)
        if not (np.isfinite(phys_value) or np.isfinite(no_sink_value) or np.isfinite(with_sink_value)):
            continue
        oar_rows.append(
            {
                "structure": structure_name,
                "label": label,
                "metric": metric_key,
                "physical_value": phys_value,
                "no_sink_value": no_sink_value,
                "with_sink_value": with_sink_value,
                "no_sink_minus_physical": no_sink_value - phys_value if np.isfinite(no_sink_value) and np.isfinite(phys_value) else float("nan"),
                "with_sink_minus_physical": with_sink_value - phys_value if np.isfinite(with_sink_value) and np.isfinite(phys_value) else float("nan"),
                "with_sink_minus_no_sink": with_sink_value - no_sink_value if np.isfinite(with_sink_value) and np.isfinite(no_sink_value) else float("nan"),
            }
        )
    return endpoint_rows, assay_rows, oar_rows


def evaluate_physical_only(
    *,
    dose_grid: np.ndarray,
    structures: Mapping[str, np.ndarray],
    axes_mm: Mapping[str, np.ndarray],
    peak_mask: np.ndarray,
    valley_mask: np.ndarray,
    voxel_volume_cc: float,
    target_prescription_gy: float,
    args: argparse.Namespace,
) -> Dict[str, object]:
    endpoints, supplemental = extract_endpoints_for_masks(
        dose_grid,
        structures=structures,
        axes_mm=axes_mm,
        peak_mask=peak_mask,
        valley_mask=valley_mask,
        voxel_volume_cc=float(voxel_volume_cc),
        target_prescription_gy=float(args.target_prescription_gy),
        shell_1_mm=float(args.shell_1_mm),
        shell_2_mm=float(args.shell_2_mm),
        shell_3_mm=float(args.shell_3_mm),
        oar_adjacent_mm=float(args.oar_adjacent_mm),
    )
    lq_survival = np.exp(-float(args.alpha) * dose_grid - float(args.beta) * dose_grid**2).astype(np.float32)
    zero = np.zeros_like(dose_grid, dtype=np.float32)
    assays = summarize_assays(
        physical_dose=dose_grid,
        final_survival=lq_survival,
        peak_concentration_ros=zero,
        cytokine_final=zero,
        time_axis=np.asarray([], dtype=np.float32),
        global_history=np.zeros((2, 0), dtype=np.float32),
        mask_history={},
        peak_mask=peak_mask,
        valley_mask=valley_mask,
        alpha=float(args.alpha),
        beta=float(args.beta),
        voxel_size_mm=voxel_size_mm_from_axes(axes_mm),
    )
    return {"dose_for_endpoints": dose_grid, "endpoints": endpoints, "supplemental": supplemental, "assays": assays}


def evaluate_biology_mode(
    *,
    physical_dose: np.ndarray,
    structures: Mapping[str, np.ndarray],
    axes_mm: Mapping[str, np.ndarray],
    peak_mask: np.ndarray,
    valley_mask: np.ndarray,
    voxel_volume_cc: float,
    args: argparse.Namespace,
    mode: str,
) -> Dict[str, object]:
    uptake_tensor, m_type, m_oxygen, uptake_meta = build_anatomical_biology_tensors(args, dict(structures))
    if mode == "bystander_no_sink":
        uptake_tensor = np.zeros_like(uptake_tensor, dtype=np.float32)
    elif mode != "bystander_with_sink":
        raise ValueError(f"Unsupported mode: {mode}")

    observables = solve_multispecies_pde_3d_with_hazard_observables(
        physical_dose,
        voxel_size_mm_from_axes(axes_mm),
        diffusion_coeffs=(float(D_ROS), float(LOCKED_D_CYTO)),
        decay_coeffs=(float(LAMBDA_ROS), float(LOCKED_LAMBDA_CYTO)),
        emission_emax=(float(EMAX_ROS), float(EMAX_CYTO)),
        emission_gamma_per_gy=float(LOCKED_GAMMA),
        emission_tensor=build_emission_tensor(physical_dose, m_type=m_type, m_oxygen=m_oxygen),
        uptake_tensor=uptake_tensor,
        hazard_weights=(float(W_ROS), float(W_CYTO)),
        history_masks={"peak_roi": peak_mask, "valley_roi": valley_mask},
        history_interval=int(args.history_interval),
        steps=int(args.pde_steps),
        dt=float(args.pde_dt),
        progress_interval=int(args.progress_interval),
        verbose=bool(args.verbose_pde),
    )

    lq_survival = np.exp(-float(args.alpha) * physical_dose - float(args.beta) * physical_dose**2).astype(np.float32)
    final_survival = calculate_phase7_survival(
        lq_survival,
        np.asarray(observables["hazard_grid"], dtype=np.float32),
        physical_dose,
        voxel_size_mm_from_axes(axes_mm),
        float(LOCKED_SCALING_FACTOR),
        weight_immune=float(W_IMMUNE),
        verbose=False,
    )
    effective_dose = calculate_effective_dose(final_survival, alpha=float(args.alpha), beta=float(args.beta))
    endpoints, supplemental = extract_endpoints_for_masks(
        effective_dose,
        structures=structures,
        axes_mm=axes_mm,
        peak_mask=peak_mask,
        valley_mask=valley_mask,
        voxel_volume_cc=float(voxel_volume_cc),
        target_prescription_gy=float(args.target_prescription_gy),
        shell_1_mm=float(args.shell_1_mm),
        shell_2_mm=float(args.shell_2_mm),
        shell_3_mm=float(args.shell_3_mm),
        oar_adjacent_mm=float(args.oar_adjacent_mm),
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
        peak_mask=peak_mask,
        valley_mask=valley_mask,
        alpha=float(args.alpha),
        beta=float(args.beta),
        voxel_size_mm=voxel_size_mm_from_axes(axes_mm),
    )
    return {
        "dose_for_endpoints": effective_dose,
        "endpoints": endpoints,
        "supplemental": supplemental,
        "assays": assays,
        "uptake_meta": uptake_meta,
    }


def ct_series_summary(ct_dir: Path | None) -> Dict[str, object]:
    if ct_dir is None or not Path(ct_dir).exists():
        return {"available": False}
    slices: List[Dataset] = []
    for path in sorted(Path(ct_dir).rglob("*")):
        if not path.is_file():
            continue
        try:
            ds = pydicom.dcmread(path, stop_before_pixels=True, force=True)
        except Exception:
            continue
        if str(getattr(ds, "Modality", "")).upper() == "CT":
            slices.append(ds)
    if not slices:
        return {"available": False}
    z_positions = sorted(float(getattr(ds, "ImagePositionPatient", [0.0, 0.0, 0.0])[2]) for ds in slices)
    spacing = float(abs(z_positions[1] - z_positions[0])) if len(z_positions) > 1 else float(getattr(slices[0], "SliceThickness", 0.0))
    return {
        "available": True,
        "slice_count": int(len(slices)),
        "series_description": str(getattr(slices[0], "SeriesDescription", "")),
        "pixel_spacing_mm": list(map(float, getattr(slices[0], "PixelSpacing", [0.0, 0.0]))),
        "slice_spacing_mm": float(spacing),
    }


def rtplan_summary(rtplan_path: Path | None) -> Dict[str, object]:
    if rtplan_path is None or not Path(rtplan_path).exists():
        return {"available": False}
    ds = pydicom.dcmread(rtplan_path, stop_before_pixels=True)
    beam_rows: List[Dict[str, object]] = []
    arc_count = 0
    for beam in getattr(ds, "BeamSequence", []):
        cp = list(getattr(beam, "ControlPointSequence", []))
        start_angle = float(getattr(cp[0], "GantryAngle", float("nan"))) if cp else float("nan")
        end_angle = float(getattr(cp[-1], "GantryAngle", float("nan"))) if cp else float("nan")
        is_arc = bool(len(cp) > 1 and np.isfinite(start_angle) and np.isfinite(end_angle) and abs(end_angle - start_angle) > 1.0)
        if is_arc:
            arc_count += 1
        beam_rows.append(
            {
                "beam_number": int(getattr(beam, "BeamNumber", 0)),
                "beam_name": str(getattr(beam, "BeamName", "")),
                "beam_type": str(getattr(beam, "BeamType", "")),
                "radiation_type": str(getattr(beam, "RadiationType", "")),
                "gantry_start_deg": start_angle,
                "gantry_end_deg": end_angle,
                "control_points": int(len(cp)),
                "is_arc": bool(is_arc),
            }
        )
    return {
        "available": True,
        "rtplan_label": str(getattr(ds, "RTPlanLabel", "")),
        "beam_count": int(len(beam_rows)),
        "arc_count": int(arc_count),
        "beams": beam_rows,
    }


def write_quick_assessment(
    path: Path,
    *,
    peak_summary: Mapping[str, object],
    structure_sources: Mapping[str, Sequence[str]],
    endpoint_rows: Sequence[Mapping[str, object]],
    oar_rows: Sequence[Mapping[str, object]],
    vessels_available: bool,
) -> None:
    endpoint_lookup = {str(row["mode"]): row for row in endpoint_rows}
    phys = endpoint_lookup["physical_only"]
    bio = endpoint_lookup["bystander_with_sink"]
    meaningful_oar = sorted(
        (
            row
            for row in oar_rows
            if np.isfinite(float(row["with_sink_minus_physical"]))
        ),
        key=lambda row: abs(float(row["with_sink_minus_physical"])),
        reverse=True,
    )[:4]
    lines = [
        "# Phase 29 Quick Assessment",
        "",
        f"- Peak derivation method: `{peak_summary['method']}`",
        f"- Peak component count: `{peak_summary['component_count']}`",
        f"- Threshold used: `{peak_summary['threshold_gy']}` Gy",
        f"- Vascular contours available: `{vessels_available}`",
        f"- GTV source: `{', '.join(structure_sources.get('GTV', [])) or 'fallback'}`",
        f"- PTV source: `{', '.join(structure_sources.get('PTV', [])) or 'fallback'}`",
        "",
        "## Endpoint shifts",
        "",
        f"- PTV D95: `{float(phys['ptv_d95']):.3f}` -> `{float(bio['ptv_d95']):.3f}` Gy(eq)",
        f"- PVDR: `{float(phys['pvdr']):.3f}` -> `{float(bio['pvdr']):.3f}`",
        f"- Peri-GTV 0-5 mm mean: `{float(phys['spill_shell_0_5_mean']):.3f}` -> `{float(bio['spill_shell_0_5_mean']):.3f}` Gy(eq)",
        f"- Peri-GTV 5-15 mm mean: `{float(phys['spill_shell_5_15_mean']):.3f}` -> `{float(bio['spill_shell_5_15_mean']):.3f}` Gy(eq)",
        "",
        "## Largest OAR reinterpretations",
        "",
    ]
    if meaningful_oar:
        for row in meaningful_oar:
            lines.append(
                f"- {row['label']}: `{float(row['physical_value']):.3f}` -> `{float(row['with_sink_value']):.3f}` "
                f"(delta `{float(row['with_sink_minus_physical']):+.3f}`)"
            )
    else:
        lines.append("- No OAR reinterpretation rows were available from the imported structure set.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    args.out_root.mkdir(parents=True, exist_ok=True)

    paths = resolve_input_paths(args)
    raw_mask_meta: Dict[str, Dict[str, object]] = {}
    dose_ds: Dataset | None = None
    phantom_meta: Dict[str, object] = {}
    if args.phase28_topas_root is not None:
        physical_dose, structures, axes_mm, explicit_peak_mask, structure_sources, phantom_meta = load_phase28_topas_phantom(args)
    else:
        physical_dose, geometry, dose_ds = load_rtdose(Path(paths["rtdose"]))
        raw_masks, raw_mask_meta, _ = rasterize_rtstruct(Path(paths["rtstruct"]), geometry)
        canonical_masks, structure_sources = map_canonical_structures(raw_masks)
        if "GTV" not in canonical_masks or "PTV" not in canonical_masks:
            raise RuntimeError("Imported RTSTRUCT could not be mapped to at least GTV and PTV structures.")

        zero_mask = np.zeros_like(physical_dose, dtype=bool)
        structures = {key: np.asarray(mask, dtype=bool) for key, mask in canonical_masks.items()}
        axes_mm = geometry.axes_mm
        structures["HYPOXIA"] = derive_hypoxia_mask(structures["GTV"], axes_mm, float(args.hypoxia_core_mm))
        structures.setdefault("ARTERIES", zero_mask.copy())
        structures.setdefault("VEINS", zero_mask.copy())
        if args.require_vessels and (
            int(np.count_nonzero(structures["ARTERIES"])) == 0 or int(np.count_nonzero(structures["VEINS"])) == 0
        ):
            raise RuntimeError("Vessel contours were required, but arterial and/or venous contours were not found.")

        explicit_peak_mask = None
        if "VERTEX" in canonical_masks:
            explicit_peak_mask = np.asarray(canonical_masks["VERTEX"], dtype=bool)

    if args.require_vessels and (
        int(np.count_nonzero(structures.get("ARTERIES", np.zeros_like(physical_dose, dtype=bool)))) == 0
        or int(np.count_nonzero(structures.get("VEINS", np.zeros_like(physical_dose, dtype=bool)))) == 0
    ):
        raise RuntimeError("Vessel contours were required, but arterial and/or venous masks were not available.")

    peak_mask, components, peak_summary = derive_peak_components(
        dose_grid=physical_dose,
        axes_mm=axes_mm,
        target_mask=structures["PTV"],
        explicit_peak_mask=explicit_peak_mask,
        threshold_fraction=float(args.peak_threshold_fraction),
        threshold_gy=float(args.peak_threshold_gy),
        min_component_voxels=int(args.min_peak_component_voxels),
    )
    valley_mask = build_valley_mask(structures["PTV"], peak_mask, axes_mm, float(args.valley_exclusion_mm))

    voxel_volume_cc = voxel_volume_cc_from_axes(axes_mm)
    mode_results = {
        "physical_only": evaluate_physical_only(
            dose_grid=physical_dose,
            structures=structures,
            axes_mm=axes_mm,
            peak_mask=peak_mask,
            valley_mask=valley_mask,
            voxel_volume_cc=float(voxel_volume_cc),
            target_prescription_gy=float(args.target_prescription_gy),
            args=args,
        ),
        "bystander_no_sink": evaluate_biology_mode(
            physical_dose=physical_dose,
            structures=structures,
            axes_mm=axes_mm,
            peak_mask=peak_mask,
            valley_mask=valley_mask,
            voxel_volume_cc=float(voxel_volume_cc),
            args=args,
            mode="bystander_no_sink",
        ),
        "bystander_with_sink": evaluate_biology_mode(
            physical_dose=physical_dose,
            structures=structures,
            axes_mm=axes_mm,
            peak_mask=peak_mask,
            valley_mask=valley_mask,
            voxel_volume_cc=float(voxel_volume_cc),
            args=args,
            mode="bystander_with_sink",
        ),
    }

    vessels_available = bool(int(np.count_nonzero(structures["ARTERIES"])) > 0 and int(np.count_nonzero(structures["VEINS"])) > 0)
    if not vessels_available:
        mode_results["bystander_with_sink"]["notes"] = "No artery/vein contours were available; with-sink equals no-sink."

    endpoint_rows, assay_rows, oar_rows = build_plan_assessment_rows(mode_results)
    component_rows = [
        {
            "label": component.label,
            "voxels": int(component.voxels),
            "volume_cc": float(component.volume_cc),
            "max_dose_gy": float(component.max_dose_gy),
            "mean_dose_gy": float(component.mean_dose_gy),
            "centroid_x_mm": float(component.centroid_x_mm),
            "centroid_y_mm": float(component.centroid_y_mm),
            "centroid_z_mm": float(component.centroid_z_mm),
        }
        for component in components
    ]

    endpoint_table = args.out_root / "phase29_endpoint_table.csv"
    assay_table = args.out_root / "phase29_assay_table.csv"
    oar_table = args.out_root / "phase29_oar_reinterpretation.csv"
    component_table = args.out_root / "phase29_peak_component_table.csv"
    structure_map_json = args.out_root / "phase29_structure_map.json"
    plan_summary_json = args.out_root / "phase29_plan_summary.json"
    quick_assessment_md = args.out_root / "phase29_quick_assessment.md"

    write_csv(endpoint_table, endpoint_rows)
    write_csv(assay_table, assay_rows)
    write_csv(oar_table, oar_rows)
    if component_rows:
        write_csv(component_table, component_rows)
    else:
        component_table.write_text("label,voxels,volume_cc,max_dose_gy,mean_dose_gy,centroid_x_mm,centroid_y_mm,centroid_z_mm\n", encoding="utf-8")
    write_json(
        structure_map_json,
        {
            "raw_roi_metadata": raw_mask_meta,
            "canonical_sources": structure_sources,
            "canonical_voxel_counts": {name: int(np.count_nonzero(mask)) for name, mask in structures.items()},
        },
    )
    write_json(
        plan_summary_json,
        {
            "inputs": {key: (None if value is None else str(value)) for key, value in paths.items()},
            "dose_grid_shape_xyz": [int(v) for v in physical_dose.shape],
            "dose_units": ("Gy (scaled from TOPAS scorer)" if args.phase28_topas_root is not None else str(getattr(dose_ds, "DoseUnits", ""))),
            "dose_type": ("DoseToWater" if args.phase28_topas_root is not None else str(getattr(dose_ds, "DoseType", ""))),
            "voxel_size_mm": list(map(float, voxel_size_mm_from_axes(axes_mm))),
            "ct_summary": ct_series_summary(paths["ct_dir"]),
            "rtplan_summary": rtplan_summary(paths["rtplan"]),
            "peak_summary": peak_summary,
            "vessels_available": bool(vessels_available),
            "primary_endpoint_keys": list(PRIMARY_ENDPOINT_KEYS),
            "phantom_meta": phantom_meta,
        },
    )
    write_quick_assessment(
        quick_assessment_md,
        peak_summary=peak_summary,
        structure_sources=structure_sources,
        endpoint_rows=endpoint_rows,
        oar_rows=oar_rows,
        vessels_available=bool(vessels_available),
    )

    print("=== PHASE 29 VMAT DICOM BIOLOGY IMPORT COMPLETE ===")
    print(f"Output root: {args.out_root}")
    print(f"Endpoint table: {endpoint_table}")
    print(f"Assay table: {assay_table}")
    print(f"OAR reinterpretation: {oar_table}")
    print(f"Peak components: {component_table}")
    print(f"Structure map: {structure_map_json}")
    print(f"Plan summary: {plan_summary_json}")
    print(f"Quick assessment: {quick_assessment_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
