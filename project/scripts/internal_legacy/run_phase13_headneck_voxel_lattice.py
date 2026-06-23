#!/usr/bin/env python3
"""Synthetic voxelized head-and-neck lattice-boost workflow.

This script creates an IAEA-style anthropomorphic head-and-neck surrogate,
writes a TOPAS ImageCube deck, runs a multi-source 6 MV polyenergetic lattice
boost plan, computes conventional plan metrics, then re-evaluates the same plan
through the calibrated nonlocal biology model to produce bio-aware DVHs and
effective-dose metrics.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

from analyze_topas_outputs import load_topas_grid
from build_asymmetric_sweep import (
    PHYSICS_PROFILES,
    build_topas_env,
    format_physics_modules,
    has_nonempty_output,
    write_text_with_retries,
)
from bystander_multispecies_pde_solver import (
    build_vessel_network_uptake_tensor,
    calculate_effective_dose,
    calculate_phase7_survival,
    calculate_systemic_immune_penalty,
    run_pde_temporal_integration,
)
from run_linac_6mv_polyenergetic_clinical_sfrt import load_spectrum

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
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

AIR_TAG = np.int16(0)
SOFT_TAG = np.int16(10)
BONE_TAG = np.int16(20)

MATERIAL_TAGS: Dict[int, str] = {
    int(AIR_TAG): "G4_AIR",
    int(SOFT_TAG): "G4_WATER",
    int(BONE_TAG): "G4_BONE_COMPACT_ICRU",
}


@dataclass(frozen=True)
class SourceSpec:
    name: str
    center_mm: Tuple[float, float, float]
    rotation_deg: Tuple[float, float, float]
    cutoff_mm: Tuple[float, float]
    histories: int


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description=(
            "Create a synthetic voxelized head-and-neck lattice-boost phantom, run "
            "a TOPAS 6 MV multi-source plan, and compare physical versus bio-aware "
            "plan metrics."
        )
    )
    parser.add_argument(
        "--template",
        type=Path,
        default=root / "topas" / "headneck_voxel_lattice_template.txt",
        help="TOPAS ImageCube template.",
    )
    parser.add_argument(
        "--spectrum-csv",
        type=Path,
        default=root / "data" / "linac_6mv_representative_spectrum.csv",
        help="Representative 6 MV spectrum as energy_mev,weight CSV.",
    )
    parser.add_argument(
        "--run-root",
        type=Path,
        default=root / "runs" / "linac_6mv_headneck_voxel_lattice_sfrt",
        help="Output root for the phantom, TOPAS case, and analysis artifacts.",
    )
    parser.add_argument(
        "--topas-bin",
        type=str,
        default="/Users/kw/shellScripts/topas",
        help="TOPAS executable.",
    )
    parser.add_argument(
        "--g4-data-dir",
        type=str,
        default="/Applications/GEANT4",
        help="Directory containing Geant4 data folders.",
    )
    parser.add_argument(
        "--physics-profile",
        choices=sorted(PHYSICS_PROFILES),
        default="em_opt4_only",
        help="Named TOPAS modular physics profile.",
    )
    parser.add_argument("--histories", type=int, default=1_000_000, help="Total TOPAS histories.")
    parser.add_argument("--threads", type=int, default=8, help="TOPAS threads.")
    parser.add_argument("--seed", type=int, default=33, help="TOPAS RNG seed.")
    parser.add_argument("--size-x-cm", type=float, default=18.0, help="Phantom left-right size.")
    parser.add_argument("--size-y-cm", type=float, default=24.0, help="Phantom superior-inferior size.")
    parser.add_argument("--size-z-cm", type=float, default=18.0, help="Phantom anterior-posterior size.")
    parser.add_argument("--voxel-mm", type=float, default=2.0, help="Isotropic voxel size.")
    parser.add_argument("--spot-radius-mm", type=float, default=4.0, help="Lattice beamlet radius.")
    parser.add_argument("--base-margin-mm", type=float, default=6.0, help="Margin added to broad-field radius.")
    parser.add_argument(
        "--base-history-fraction",
        type=float,
        default=0.42,
        help="Fraction of histories assigned to the three broad base fields.",
    )
    parser.add_argument(
        "--lattice-spacing-mm",
        nargs=3,
        type=float,
        default=[18.0, 20.0, 18.0],
        help="Nominal lattice spacing in x, y, z within the bulky target.",
    )
    parser.add_argument(
        "--prescription-gy",
        type=float,
        default=6.0,
        help="Physical PTV prescription used to normalize the plan to D95.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.03,
        help="LQ alpha used for the final physical and bio-effective comparison.",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=0.003,
        help="LQ beta used for the final physical and bio-effective comparison.",
    )
    parser.add_argument("--cut-gamma-mm", type=float, default=0.01)
    parser.add_argument("--cut-electron-mm", type=float, default=0.01)
    parser.add_argument("--cut-positron-mm", type=float, default=0.01)
    parser.add_argument("--pde-steps", type=int, default=400, help="Temporal PDE steps.")
    parser.add_argument("--pde-dt", type=float, default=0.12, help="Temporal PDE dt.")
    parser.add_argument(
        "--tumor-cytokine-multiplier",
        type=float,
        default=2.0,
        help="Tumor-specific cytokine emission multiplier.",
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
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip TOPAS if the dose CSV already exists and is non-empty.",
    )
    parser.add_argument(
        "--analyze-only",
        action="store_true",
        help="Skip phantom/TOPAS generation and analyze an existing dose CSV.",
    )
    parser.add_argument("--dpi", type=int, default=250, help="Figure DPI.")
    return parser.parse_args()


def centered_axis_mm(count: int, spacing_mm: float) -> np.ndarray:
    return (np.arange(int(count), dtype=np.float32) - (int(count) - 1) / 2.0) * float(spacing_mm)


def axis_depth_cm(count: int, spacing_mm: float) -> np.ndarray:
    return centered_axis_mm(count, spacing_mm) / 10.0


def ellipsoid_mask(
    x_mm: np.ndarray,
    y_mm: np.ndarray,
    z_mm: np.ndarray,
    *,
    center_mm: Tuple[float, float, float],
    radii_mm: Tuple[float, float, float],
) -> np.ndarray:
    cx, cy, cz = (float(v) for v in center_mm)
    rx, ry, rz = (float(v) for v in radii_mm)
    return (
        ((x_mm[:, None, None] - cx) / rx) ** 2
        + ((y_mm[None, :, None] - cy) / ry) ** 2
        + ((z_mm[None, None, :] - cz) / rz) ** 2
        <= 1.0
    )


def cylinder_along_y_mask(
    x_mm: np.ndarray,
    y_mm: np.ndarray,
    z_mm: np.ndarray,
    *,
    center_x_mm: float,
    center_z_mm: float,
    radius_mm: float,
    y_min_mm: float,
    y_max_mm: float,
) -> np.ndarray:
    radial = (x_mm[:, None, None] - float(center_x_mm)) ** 2 + (z_mm[None, None, :] - float(center_z_mm)) ** 2
    return (radial <= float(radius_mm) ** 2) & (
        (y_mm[None, :, None] >= float(y_min_mm)) & (y_mm[None, :, None] <= float(y_max_mm))
    )


def capped_cylinder_along_y_mask(
    x_mm: np.ndarray,
    y_mm: np.ndarray,
    z_mm: np.ndarray,
    *,
    center_x_mm: float,
    center_z_mm: float,
    radius_x_mm: float,
    radius_z_mm: float,
    y_min_mm: float,
    y_max_mm: float,
) -> np.ndarray:
    radial = (
        ((x_mm[:, None, None] - float(center_x_mm)) / float(radius_x_mm)) ** 2
        + ((z_mm[None, None, :] - float(center_z_mm)) / float(radius_z_mm)) ** 2
    )
    return (radial <= 1.0) & (
        (y_mm[None, :, None] >= float(y_min_mm)) & (y_mm[None, :, None] <= float(y_max_mm))
    )


def build_headneck_phantom(args: argparse.Namespace) -> Dict[str, object]:
    dx = dy = dz = float(args.voxel_mm)
    nx = int(round(float(args.size_x_cm) * 10.0 / dx))
    ny = int(round(float(args.size_y_cm) * 10.0 / dy))
    nz = int(round(float(args.size_z_cm) * 10.0 / dz))
    x_mm = centered_axis_mm(nx, dx)
    y_mm = centered_axis_mm(ny, dy)
    z_mm = centered_axis_mm(nz, dz)

    head = ellipsoid_mask(x_mm, y_mm, z_mm, center_mm=(0.0, 38.0, 0.0), radii_mm=(78.0, 76.0, 70.0))
    neck = ellipsoid_mask(x_mm, y_mm, z_mm, center_mm=(0.0, -34.0, 0.0), radii_mm=(62.0, 80.0, 48.0))
    shoulder_l = ellipsoid_mask(x_mm, y_mm, z_mm, center_mm=(-58.0, -88.0, 0.0), radii_mm=(42.0, 28.0, 52.0))
    shoulder_r = ellipsoid_mask(x_mm, y_mm, z_mm, center_mm=(58.0, -88.0, 0.0), radii_mm=(42.0, 28.0, 52.0))
    body_mask = head | neck | shoulder_l | shoulder_r

    skull_outer = ellipsoid_mask(x_mm, y_mm, z_mm, center_mm=(0.0, 40.0, 0.0), radii_mm=(74.0, 70.0, 64.0))
    skull_inner = ellipsoid_mask(x_mm, y_mm, z_mm, center_mm=(0.0, 40.0, 0.0), radii_mm=(66.0, 62.0, 57.0))
    skull_mask = skull_outer & ~skull_inner

    mandible_arch = capped_cylinder_along_y_mask(
        x_mm, y_mm, z_mm, center_x_mm=0.0, center_z_mm=-8.0, radius_x_mm=44.0, radius_z_mm=24.0, y_min_mm=-8.0, y_max_mm=14.0
    )
    mandible_gap = ellipsoid_mask(x_mm, y_mm, z_mm, center_mm=(0.0, 0.0, 10.0), radii_mm=(36.0, 18.0, 32.0))
    ramus_l = ellipsoid_mask(x_mm, y_mm, z_mm, center_mm=(-42.0, 8.0, 6.0), radii_mm=(8.0, 26.0, 10.0))
    ramus_r = ellipsoid_mask(x_mm, y_mm, z_mm, center_mm=(42.0, 8.0, 6.0), radii_mm=(8.0, 26.0, 10.0))
    mandible_mask = (mandible_arch & ~mandible_gap) | ramus_l | ramus_r

    vertebrae_mask = np.zeros((nx, ny, nz), dtype=bool)
    for center_y in np.arange(-62.0, 42.0, 18.0):
        body = ellipsoid_mask(x_mm, y_mm, z_mm, center_mm=(0.0, float(center_y), 24.0), radii_mm=(12.0, 8.0, 10.0))
        spinous = ellipsoid_mask(x_mm, y_mm, z_mm, center_mm=(0.0, float(center_y), 34.0), radii_mm=(4.0, 8.0, 6.0))
        vertebrae_mask |= body | spinous

    airway_mask = capped_cylinder_along_y_mask(
        x_mm, y_mm, z_mm, center_x_mm=0.0, center_z_mm=-8.0, radius_x_mm=9.0, radius_z_mm=11.0, y_min_mm=-12.0, y_max_mm=34.0
    )
    trachea_mask = cylinder_along_y_mask(
        x_mm, y_mm, z_mm, center_x_mm=0.0, center_z_mm=-7.0, radius_mm=7.0, y_min_mm=-76.0, y_max_mm=-12.0
    )
    oral_cavity = ellipsoid_mask(x_mm, y_mm, z_mm, center_mm=(0.0, 18.0, -18.0), radii_mm=(16.0, 12.0, 10.0))
    airway_mask |= trachea_mask | oral_cavity

    parotid_l = ellipsoid_mask(x_mm, y_mm, z_mm, center_mm=(-50.0, -4.0, -4.0), radii_mm=(16.0, 24.0, 12.0)) & body_mask
    parotid_r = ellipsoid_mask(x_mm, y_mm, z_mm, center_mm=(50.0, -4.0, -4.0), radii_mm=(16.0, 24.0, 12.0)) & body_mask
    spinal_cord = cylinder_along_y_mask(
        x_mm, y_mm, z_mm, center_x_mm=0.0, center_z_mm=22.0, radius_mm=4.5, y_min_mm=-70.0, y_max_mm=56.0
    ) & body_mask
    brainstem = ellipsoid_mask(x_mm, y_mm, z_mm, center_mm=(0.0, 58.0, 20.0), radii_mm=(10.0, 18.0, 12.0)) & body_mask

    gtv_primary = ellipsoid_mask(x_mm, y_mm, z_mm, center_mm=(22.0, 8.0, -4.0), radii_mm=(28.0, 30.0, 24.0))
    gtv_nodal = ellipsoid_mask(x_mm, y_mm, z_mm, center_mm=(34.0, -34.0, 2.0), radii_mm=(24.0, 38.0, 21.0))
    gtv_mask = (gtv_primary | gtv_nodal) & body_mask & ~airway_mask & ~spinal_cord & ~brainstem

    ptv_primary = ellipsoid_mask(x_mm, y_mm, z_mm, center_mm=(22.0, 8.0, -4.0), radii_mm=(32.0, 34.0, 28.0))
    ptv_nodal = ellipsoid_mask(x_mm, y_mm, z_mm, center_mm=(34.0, -34.0, 2.0), radii_mm=(28.0, 42.0, 25.0))
    ptv_mask = (ptv_primary | ptv_nodal) & body_mask & ~airway_mask

    hypoxic_primary = ellipsoid_mask(x_mm, y_mm, z_mm, center_mm=(22.0, 8.0, -4.0), radii_mm=(16.0, 18.0, 14.0))
    hypoxic_nodal = ellipsoid_mask(x_mm, y_mm, z_mm, center_mm=(34.0, -34.0, 2.0), radii_mm=(12.0, 20.0, 11.0))
    hypoxic_mask = (hypoxic_primary | hypoxic_nodal) & gtv_mask

    body_mask &= ~airway_mask

    tag_grid = np.full((nx, ny, nz), AIR_TAG, dtype=np.int16)
    tag_grid[body_mask] = SOFT_TAG
    tag_grid[skull_mask | mandible_mask | vertebrae_mask] = BONE_TAG
    tag_grid[airway_mask & body_mask] = AIR_TAG

    structures = {
        "BODY": body_mask,
        "GTV": gtv_mask,
        "PTV": ptv_mask,
        "HYPOXIA": hypoxic_mask,
        "PAROTID_L": parotid_l,
        "PAROTID_R": parotid_r,
        "SPINAL_CORD": spinal_cord,
        "BRAINSTEM": brainstem,
        "MANDIBLE": mandible_mask & body_mask,
        "AIRWAY": airway_mask,
        "BONE": (skull_mask | mandible_mask | vertebrae_mask) & body_mask,
    }

    voxel_volume_cc = (dx * dy * dz) / 1000.0
    meta = {
        "grid_shape": [int(nx), int(ny), int(nz)],
        "voxel_size_mm": [dx, dy, dz],
        "size_cm": [float(args.size_x_cm), float(args.size_y_cm), float(args.size_z_cm)],
        "voxel_volume_cc": float(voxel_volume_cc),
        "structure_volumes_cc": {
            name: float(np.count_nonzero(mask) * voxel_volume_cc)
            for name, mask in structures.items()
            if name not in {"AIRWAY", "BONE", "HYPOXIA"}
        },
        "anatomical_note": (
            "Synthetic IAEA-style head-and-neck audit surrogate with external contour, "
            "airway, skull/mandible, cervical vertebrae, bilateral parotids, spinal cord, "
            "brainstem, and a bulky right-sided oropharyngeal-nodal target."
        ),
    }
    return {
        "tag_grid": tag_grid,
        "structures": structures,
        "axes_mm": {"x": x_mm, "y": y_mm, "z": z_mm},
        "meta": meta,
    }


def sphere_fits(mask: np.ndarray, center_idx: Tuple[int, int, int], radius_vox: int) -> bool:
    cx, cy, cz = center_idx
    x0 = max(cx - radius_vox, 0)
    x1 = min(cx + radius_vox + 1, mask.shape[0])
    y0 = max(cy - radius_vox, 0)
    y1 = min(cy + radius_vox + 1, mask.shape[1])
    z0 = max(cz - radius_vox, 0)
    z1 = min(cz + radius_vox + 1, mask.shape[2])
    local = mask[x0:x1, y0:y1, z0:z1]
    gx = np.arange(x0, x1) - cx
    gy = np.arange(y0, y1) - cy
    gz = np.arange(z0, z1) - cz
    sphere = (
        gx[:, None, None] ** 2 + gy[None, :, None] ** 2 + gz[None, None, :] ** 2
        <= radius_vox**2
    )
    return bool(np.all(local[sphere]))


def pick_lattice_spots(
    gtv_mask: np.ndarray,
    axes_mm: Dict[str, np.ndarray],
    spacing_mm: Sequence[float],
    *,
    spot_radius_mm: float,
    limit: int = 12,
) -> List[Tuple[float, float, float]]:
    x_mm = axes_mm["x"]
    y_mm = axes_mm["y"]
    z_mm = axes_mm["z"]
    gtv_idx = np.argwhere(gtv_mask)
    centroid_idx = np.round(gtv_idx.mean(axis=0)).astype(int)
    centroid_mm = (float(x_mm[centroid_idx[0]]), float(y_mm[centroid_idx[1]]), float(z_mm[centroid_idx[2]]))

    xs = np.arange(centroid_mm[0] - 1.5 * spacing_mm[0], centroid_mm[0] + 1.51 * spacing_mm[0], spacing_mm[0])
    ys = np.arange(centroid_mm[1] - 1.0 * spacing_mm[1], centroid_mm[1] + 1.01 * spacing_mm[1], spacing_mm[1])
    zs = np.arange(centroid_mm[2] - 1.0 * spacing_mm[2], centroid_mm[2] + 1.01 * spacing_mm[2], spacing_mm[2])

    radius_vox = max(1, int(round(float(spot_radius_mm) / float(x_mm[1] - x_mm[0]))))
    candidates: List[Tuple[float, float, float, float]] = []
    for x0 in xs:
        for y0 in ys:
            for z0 in zs:
                ix = int(np.argmin(np.abs(x_mm - x0)))
                iy = int(np.argmin(np.abs(y_mm - y0)))
                iz = int(np.argmin(np.abs(z_mm - z0)))
                if not gtv_mask[ix, iy, iz]:
                    continue
                if not sphere_fits(gtv_mask, (ix, iy, iz), radius_vox):
                    continue
                dist2 = (float(x_mm[ix]) - centroid_mm[0]) ** 2 + (float(y_mm[iy]) - centroid_mm[1]) ** 2 + (
                    float(z_mm[iz]) - centroid_mm[2]
                ) ** 2
                candidates.append((float(x_mm[ix]), float(y_mm[iy]), float(z_mm[iz]), dist2))

    if not candidates:
        raise RuntimeError("Could not place any lattice spots inside the bulky GTV.")

    ordered = sorted(candidates, key=lambda row: row[3])
    unique_spots: List[Tuple[float, float, float]] = []
    for x0, y0, z0, _ in ordered:
        spot = (x0, y0, z0)
        if spot not in unique_spots:
            unique_spots.append(spot)
        if len(unique_spots) >= int(limit):
            break
    return unique_spots


def histories_from_weights(total_histories: int, weights: Sequence[float]) -> List[int]:
    weights_arr = np.asarray(weights, dtype=np.float64)
    if np.any(weights_arr < 0.0) or float(weights_arr.sum()) <= 0.0:
        raise ValueError("weights must be non-negative and sum to a positive value.")
    weights_arr = weights_arr / float(weights_arr.sum())
    raw = weights_arr * float(total_histories)
    histories = np.floor(raw).astype(int)
    remainder = int(total_histories) - int(histories.sum())
    if remainder > 0:
        order = np.argsort(-(raw - histories))
        histories[order[:remainder]] += 1
    return histories.tolist()


def projected_radius_mm(mask: np.ndarray, axes_a_mm: np.ndarray, axes_b_mm: np.ndarray, centroid_a_mm: float, centroid_b_mm: float, axis_order: Tuple[int, int]) -> float:
    coords = np.argwhere(mask)
    if coords.size == 0:
        raise ValueError("Mask is empty.")
    aa = axes_a_mm[coords[:, axis_order[0]]]
    bb = axes_b_mm[coords[:, axis_order[1]]]
    radii = np.sqrt((aa - float(centroid_a_mm)) ** 2 + (bb - float(centroid_b_mm)) ** 2)
    return float(np.percentile(radii, 99.0))


def build_plan_sources(
    args: argparse.Namespace,
    axes_mm: Dict[str, np.ndarray],
    ptv_mask: np.ndarray,
    spot_centers_mm: Sequence[Tuple[float, float, float]],
) -> Dict[str, object]:
    x_mm = axes_mm["x"]
    y_mm = axes_mm["y"]
    z_mm = axes_mm["z"]
    ptv_idx = np.argwhere(ptv_mask)
    centroid_idx = np.round(ptv_idx.mean(axis=0)).astype(int)
    ptv_centroid_mm = (float(x_mm[centroid_idx[0]]), float(y_mm[centroid_idx[1]]), float(z_mm[centroid_idx[2]]))

    ap_radius = projected_radius_mm(ptv_mask, x_mm, y_mm, ptv_centroid_mm[0], ptv_centroid_mm[1], (0, 1)) + float(args.base_margin_mm)
    lat_radius = projected_radius_mm(ptv_mask, z_mm, y_mm, ptv_centroid_mm[2], ptv_centroid_mm[1], (2, 1)) + float(args.base_margin_mm)
    ap_radius = max(ap_radius, float(getattr(args, "base_min_ap_radius_mm", 0.0)))
    lat_radius = max(lat_radius, float(getattr(args, "base_min_lateral_radius_mm", 0.0)))

    source_plane_margin_mm = 5.0
    x_half_mm = 0.5 * float(args.size_x_cm) * 10.0
    z_half_mm = 0.5 * float(args.size_z_cm) * 10.0
    ap_source_z = -z_half_mm - source_plane_margin_mm
    left_source_x = -x_half_mm - source_plane_margin_mm
    right_source_x = x_half_mm + source_plane_margin_mm

    source_entries: List[Tuple[str, Tuple[float, float, float], Tuple[float, float, float], Tuple[float, float], float]] = [
        ("AP_BASE", (ptv_centroid_mm[0], ptv_centroid_mm[1], ap_source_z), (0.0, 0.0, 0.0), (ap_radius, ap_radius), float(args.base_history_fraction)),
    ]

    spot_fraction = max(0.0, 1.0 - float(args.base_history_fraction))
    if spot_centers_mm:
        ap_scale = float(getattr(args, "spot_ap_weight_scale", 1.0))
        lateral_scale = float(getattr(args, "spot_lateral_weight_scale", 1.0))
        superior_post_scale = float(getattr(args, "superior_posterior_lateral_scale", 1.0))
        superior_threshold_mm = float(getattr(args, "superior_threshold_mm", 0.0))
        posterior_threshold_mm = float(getattr(args, "posterior_threshold_mm", 0.0))
        spot_radius_mm = float(args.spot_radius_mm)
        lateral_radius_scale = float(getattr(args, "lateral_radius_scale", 1.0))
        ap_radius_scale = float(getattr(args, "ap_spot_radius_scale", 1.0))
        for spot_idx, (sx, sy, sz) in enumerate(spot_centers_mm, start=1):
            label = f"SPOT{spot_idx:02d}"
            is_superior = bool(sy >= ptv_centroid_mm[1] + superior_threshold_mm)
            is_posterior = bool(sz >= ptv_centroid_mm[2] + posterior_threshold_mm)
            per_spot_lateral_scale = lateral_scale
            if is_superior or is_posterior:
                per_spot_lateral_scale *= superior_post_scale
            beam_scales = [ap_scale, per_spot_lateral_scale, per_spot_lateral_scale]
            scale_sum = max(1e-6, float(sum(beam_scales)))
            ap_weight = spot_fraction * beam_scales[0] / float(len(spot_centers_mm) * scale_sum)
            lat_left_weight = spot_fraction * beam_scales[1] / float(len(spot_centers_mm) * scale_sum)
            lat_right_weight = spot_fraction * beam_scales[2] / float(len(spot_centers_mm) * scale_sum)
            source_entries.extend(
                [
                    (
                        f"AP_{label}",
                        (sx, sy, ap_source_z),
                        (0.0, 0.0, 0.0),
                        (spot_radius_mm * ap_radius_scale, spot_radius_mm * ap_radius_scale),
                        ap_weight,
                    ),
                    (
                        f"LATL_{label}",
                        (left_source_x, sy, sz),
                        (0.0, 90.0, 0.0),
                        (spot_radius_mm * lateral_radius_scale, spot_radius_mm * lateral_radius_scale),
                        lat_left_weight,
                    ),
                    (
                        f"LATR_{label}",
                        (right_source_x, sy, sz),
                        (0.0, -90.0, 0.0),
                        (spot_radius_mm * lateral_radius_scale, spot_radius_mm * lateral_radius_scale),
                        lat_right_weight,
                    ),
                ]
            )

    history_counts = histories_from_weights(int(args.histories), [row[4] for row in source_entries])
    specs = [
        SourceSpec(
            name=row[0],
            center_mm=(float(row[1][0]), float(row[1][1]), float(row[1][2])),
            rotation_deg=(float(row[2][0]), float(row[2][1]), float(row[2][2])),
            cutoff_mm=(float(row[3][0]), float(row[3][1])),
            histories=int(histories),
        )
        for row, histories in zip(source_entries, history_counts)
    ]
    return {
        "ptv_centroid_mm": [float(v) for v in ptv_centroid_mm],
        "ap_radius_mm": float(ap_radius),
        "lateral_radius_mm": float(lat_radius),
        "sources": specs,
    }


def render_source_block(sources: Sequence[SourceSpec], spectrum_energies: Sequence[float], spectrum_weights: Sequence[float]) -> str:
    lines: List[str] = []
    spectrum_count = len(spectrum_energies)
    spectrum_values = " ".join(f"{float(v):.6f}" for v in spectrum_energies)
    spectrum_weight_values = " ".join(f"{float(v):.8f}" for v in spectrum_weights)

    for spec in sources:
        group_name = f"BeamOrigin_{spec.name}"
        source_name = f"Source_{spec.name}"
        lines.extend(
            [
                f's:Ge/{group_name}/Type = "Group"',
                f's:Ge/{group_name}/Parent = "World"',
                f"d:Ge/{group_name}/TransX = {spec.center_mm[0]:.6f} mm",
                f"d:Ge/{group_name}/TransY = {spec.center_mm[1]:.6f} mm",
                f"d:Ge/{group_name}/TransZ = {spec.center_mm[2]:.6f} mm",
                f"d:Ge/{group_name}/RotX = {spec.rotation_deg[0]:.6f} deg",
                f"d:Ge/{group_name}/RotY = {spec.rotation_deg[1]:.6f} deg",
                f"d:Ge/{group_name}/RotZ = {spec.rotation_deg[2]:.6f} deg",
                f's:So/{source_name}/Type = "Beam"',
                f's:So/{source_name}/Component = "{group_name}"',
                f's:So/{source_name}/BeamParticle = "gamma"',
                f's:So/{source_name}/BeamEnergySpectrumType = "Discrete"',
                f"dv:So/{source_name}/BeamEnergySpectrumValues = {spectrum_count} {spectrum_values} MeV",
                f"uv:So/{source_name}/BeamEnergySpectrumWeights = {spectrum_count} {spectrum_weight_values}",
                f's:So/{source_name}/BeamPositionDistribution = "Flat"',
                f's:So/{source_name}/BeamPositionCutoffShape = "Ellipse"',
                f"d:So/{source_name}/BeamPositionCutoffX = {spec.cutoff_mm[0]:.6f} mm",
                f"d:So/{source_name}/BeamPositionCutoffY = {spec.cutoff_mm[1]:.6f} mm",
                f's:So/{source_name}/BeamAngularDistribution = "None"',
                f"i:So/{source_name}/NumberOfHistoriesInRun = {spec.histories}",
                "",
            ]
        )
    return "\n".join(lines).strip() + "\n"


def write_image_cube(tag_grid_xyz: np.ndarray, out_file: Path) -> None:
    # TOPAS ImageCube expects binary voxels with x varying fastest, then y, then z.
    np.asarray(tag_grid_xyz, dtype=np.int16).transpose(2, 1, 0).tofile(out_file)


def fill_template(template_text: str, replacements: Dict[str, str]) -> str:
    rendered = template_text
    for key, value in replacements.items():
        rendered = rendered.replace(key, value)
    return rendered


def render_case_file(
    args: argparse.Namespace,
    *,
    patient_bin: Path,
    grid_shape: Sequence[int],
    voxel_size_mm: Sequence[float],
    spectrum_energies: Sequence[float],
    spectrum_weights: Sequence[float],
    sources: Sequence[SourceSpec],
) -> str:
    template_text = args.template.read_text(encoding="utf-8")
    world_hlx_cm = max(40.0, 0.5 * float(args.size_x_cm) + 20.0)
    world_hly_cm = max(40.0, 0.5 * float(args.size_y_cm) + 20.0)
    world_hlz_cm = max(40.0, 0.5 * float(args.size_z_cm) + 20.0)
    show_interval = max(1000, int(args.histories) // 10)

    material_tag_values = " ".join(str(int(v)) for v in MATERIAL_TAGS.keys())
    material_name_values = " ".join(f'"{name}"' for name in MATERIAL_TAGS.values())
    patient_input_dir = str(patient_bin.parent.resolve())
    if not patient_input_dir.endswith("/"):
        patient_input_dir = patient_input_dir + "/"

    replacements = {
        "__G4_DATA_DIR__": str(Path(args.g4_data_dir).expanduser()),
        "__PHYSICS_MODULES__": format_physics_modules(str(args.physics_profile)),
        "__CUT_GAMMA_MM__": f"{float(args.cut_gamma_mm):.6f}",
        "__CUT_ELECTRON_MM__": f"{float(args.cut_electron_mm):.6f}",
        "__CUT_POSITRON_MM__": f"{float(args.cut_positron_mm):.6f}",
        "__WORLD_HLX_CM__": f"{world_hlx_cm:.6f}",
        "__WORLD_HLY_CM__": f"{world_hly_cm:.6f}",
        "__WORLD_HLZ_CM__": f"{world_hlz_cm:.6f}",
        "__PATIENT_INPUT_DIR__": patient_input_dir,
        "__PATIENT_INPUT_FILE__": patient_bin.name,
        "__XBINS__": str(int(grid_shape[0])),
        "__YBINS__": str(int(grid_shape[1])),
        "__ZBINS__": str(int(grid_shape[2])),
        "__VOXEL_SIZE_X_MM__": f"{float(voxel_size_mm[0]):.6f}",
        "__VOXEL_SIZE_Y_MM__": f"{float(voxel_size_mm[1]):.6f}",
        "__VOXEL_SIZE_Z_MM__": f"{float(voxel_size_mm[2]):.6f}",
        "__MATERIAL_TAG_COUNT__": str(len(MATERIAL_TAGS)),
        "__MATERIAL_TAG_VALUES__": material_tag_values,
        "__MATERIAL_NAME_VALUES__": material_name_values,
        "__OUTPUT_STEM__": "dosedata",
        "__SOURCE_BLOCK__": render_source_block(sources, spectrum_energies, spectrum_weights),
        "__N_THREADS__": str(int(args.threads)),
        "__SEED__": str(int(args.seed)),
        "__SHOW_HISTORY_INTERVAL__": str(int(show_interval)),
    }
    return fill_template(template_text, replacements)


def run_topas_case(args: argparse.Namespace, case_dir: Path, parameter_file: Path, dose_csv: Path, log_file: Path) -> None:
    result = subprocess.run(
        [str(args.topas_bin), parameter_file.name],
        cwd=str(case_dir),
        capture_output=True,
        text=True,
        env=build_topas_env(str(args.g4_data_dir)),
    )
    combined_log = (result.stdout or "") + ("\n" + result.stderr if result.stderr else "")
    write_text_with_retries(log_file, combined_log)
    if result.returncode != 0 or not has_nonempty_output(dose_csv):
        tail = "\n".join(combined_log.strip().splitlines()[-40:])
        raise RuntimeError(
            "TOPAS run failed for the voxelized head-and-neck lattice plan.\n"
            f"Return code: {result.returncode}\n"
            f"Dose CSV present: {has_nonempty_output(dose_csv)}\n"
            f"Recent log:\n{tail}"
        )


def dose_at_volume_percent(dose_values: np.ndarray, percent_volume: float) -> float:
    return float(np.percentile(dose_values, 100.0 - float(percent_volume)))


def compute_structure_metrics(
    dose_grid: np.ndarray,
    mask: np.ndarray,
    *,
    prescription_gy: float | None,
    voxel_volume_cc: float,
    volume_thresholds_gy: Sequence[float] = (),
) -> Dict[str, float]:
    values = np.asarray(dose_grid[mask], dtype=np.float64)
    if values.size == 0:
        raise ValueError("Cannot compute metrics on an empty structure.")
    metrics: Dict[str, float] = {
        "volume_cc": float(values.size * float(voxel_volume_cc)),
        "mean_gy": float(np.mean(values)),
        "d2_gy": dose_at_volume_percent(values, 2.0),
        "d98_gy": dose_at_volume_percent(values, 98.0),
        "d95_gy": dose_at_volume_percent(values, 95.0),
        "d50_gy": dose_at_volume_percent(values, 50.0),
        "dmax_gy": float(np.max(values)),
    }
    if prescription_gy is not None:
        rx = float(prescription_gy)
        metrics["v95_pct"] = float(np.mean(values >= 0.95 * rx) * 100.0)
        metrics["v100_pct"] = float(np.mean(values >= 1.00 * rx) * 100.0)
        metrics["coverage_ratio"] = float(np.mean(values >= 1.00 * rx))
    for threshold in volume_thresholds_gy:
        label = f"v{int(round(float(threshold)))}_pct"
        metrics[label] = float(np.mean(values >= float(threshold)) * 100.0)
    return metrics


def compute_dvh(values_gy: np.ndarray, dose_axis_gy: np.ndarray) -> np.ndarray:
    return np.array([100.0 * np.mean(values_gy >= dose) for dose in dose_axis_gy], dtype=np.float32)


def metrics_table_rows(structure_metrics: Dict[str, Dict[str, float]], domain_label: str) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for name, metrics in structure_metrics.items():
        row = {"domain": domain_label, "structure": name}
        row.update({key: float(value) for key, value in metrics.items()})
        rows.append(row)
    return rows


def build_biology_tensors(
    args: argparse.Namespace,
    dose_grid: np.ndarray,
    axes_mm: Dict[str, np.ndarray],
    structures: Dict[str, np.ndarray],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[dict]]:
    shape = dose_grid.shape
    m_type = np.ones((2, *shape), dtype=np.float32)
    m_oxygen = np.ones((2, *shape), dtype=np.float32)
    gtv_mask = structures["GTV"]
    hypoxia_mask = structures["HYPOXIA"]
    m_type[1, gtv_mask] = float(args.tumor_cytokine_multiplier)
    m_oxygen[0, hypoxia_mask] = float(args.hypoxic_ros_scale)
    m_oxygen[1, hypoxia_mask] = float(args.hypoxic_cytokine_multiplier)

    y_min = float(np.percentile(axes_mm["y"][np.any(structures["BODY"], axis=(0, 2))], 5))
    y_max = float(np.percentile(axes_mm["y"][np.any(structures["BODY"], axis=(0, 2))], 95))
    vessel_specs = [
        {
            "nodes_mm": [(-18.0, y_min, 4.0), (-18.0, y_max, 4.0)],
            "radius_mm": 3.0,
            "uptake_rates_in_vessel": (0.05, 0.70),
        },
        {
            "nodes_mm": [(18.0, y_min, 4.0), (18.0, y_max, 4.0)],
            "radius_mm": 3.0,
            "uptake_rates_in_vessel": (0.05, 0.70),
        },
        {
            "nodes_mm": [(-28.0, y_min, 8.0), (-28.0, y_max, 8.0)],
            "radius_mm": 4.5,
            "uptake_rates_in_vessel": (0.05, 0.90),
        },
        {
            "nodes_mm": [(28.0, y_min, 8.0), (28.0, y_max, 8.0)],
            "radius_mm": 4.5,
            "uptake_rates_in_vessel": (0.05, 0.90),
        },
    ]
    uptake_tensor, _ = build_vessel_network_uptake_tensor(
        shape,
        (float(args.voxel_mm), float(args.voxel_mm), float(args.voxel_mm)),
        vessel_specs,
        num_species=2,
        dtype=np.float32,
    )
    return uptake_tensor, m_type, m_oxygen, vessel_specs


def save_csv(rows: Sequence[Dict[str, object]], out_file: Path) -> None:
    if not rows:
        raise ValueError("No rows supplied for CSV output.")
    fieldnames: List[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with out_file.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_anatomy(
    out_file: Path,
    axes_mm: Dict[str, np.ndarray],
    structures: Dict[str, np.ndarray],
    spot_centers_mm: Sequence[Tuple[float, float, float]],
    *,
    dpi: int,
) -> None:
    y_index = int(np.argmin(np.abs(axes_mm["y"] - 0.0)))
    z_index = int(np.argmin(np.abs(axes_mm["z"] - 0.0)))
    x_index = int(np.argmin(np.abs(axes_mm["x"] - 20.0)))

    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.8), constrained_layout=True)
    body = structures["BODY"]
    ptv = structures["PTV"]
    gtv = structures["GTV"]
    cord = structures["SPINAL_CORD"]
    par_l = structures["PAROTID_L"]
    par_r = structures["PAROTID_R"]
    brainstem = structures["BRAINSTEM"]

    def draw_panel(ax, base_mask, overlays, axis_a_cm, axis_b_cm, xlabel, ylabel, title):
        ax.imshow(
            base_mask.T,
            origin="lower",
            cmap="Greys",
            extent=[float(axis_a_cm[0]), float(axis_a_cm[-1]), float(axis_b_cm[0]), float(axis_b_cm[-1])],
            alpha=0.95,
        )
        colors = ["tab:red", "tab:orange", "tab:cyan", "tab:green", "tab:purple", "tab:blue"]
        for idx, overlay in enumerate(overlays):
            ax.contour(
                axis_a_cm,
                axis_b_cm,
                overlay.T.astype(float),
                levels=[0.5],
                colors=[colors[idx % len(colors)]],
                linewidths=1.4,
            )
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
    x_cm = axes_mm["x"] / 10.0
    y_cm = axes_mm["y"] / 10.0
    z_cm = axes_mm["z"] / 10.0

    draw_panel(
        axes[0],
        body[:, y_index, :],
        [ptv[:, y_index, :], gtv[:, y_index, :], cord[:, y_index, :], par_l[:, y_index, :], par_r[:, y_index, :]],
        x_cm,
        z_cm,
        "x (cm)",
        "z (cm)",
        "Axial-style slice",
    )
    draw_panel(
        axes[1],
        body[:, :, z_index],
        [ptv[:, :, z_index], gtv[:, :, z_index], par_l[:, :, z_index], par_r[:, :, z_index], brainstem[:, :, z_index]],
        x_cm,
        y_cm,
        "x (cm)",
        "y (cm)",
        "Coronal slice",
    )
    draw_panel(
        axes[2],
        body[x_index, :, :],
        [ptv[x_index, :, :], gtv[x_index, :, :], cord[x_index, :, :], brainstem[x_index, :, :]],
        y_cm,
        z_cm,
        "y (cm)",
        "z (cm)",
        "Sagittal slice",
    )

    for ax in axes[:2]:
        for sx, sy, sz in spot_centers_mm:
            if ax is axes[0] and abs(sy - float(axes_mm["y"][y_index])) <= 0.5 * float(axes_mm["y"][1] - axes_mm["y"][0]):
                ax.scatter([sx / 10.0], [sz / 10.0], c="yellow", s=20, edgecolors="black", linewidths=0.4)
            if ax is axes[1] and abs(sz - float(axes_mm["z"][z_index])) <= 0.5 * float(axes_mm["z"][1] - axes_mm["z"][0]):
                ax.scatter([sx / 10.0], [sy / 10.0], c="yellow", s=20, edgecolors="black", linewidths=0.4)

    fig.suptitle("Synthetic head-and-neck audit surrogate with bulky lattice-eligible target", fontsize=13)
    fig.savefig(out_file, dpi=dpi)
    plt.close(fig)


def plot_dose_slices(
    out_file: Path,
    axes_mm: Dict[str, np.ndarray],
    physical_dose: np.ndarray,
    bio_dose: np.ndarray,
    structures: Dict[str, np.ndarray],
    *,
    dpi: int,
) -> None:
    y_index = int(np.argmin(np.abs(axes_mm["y"] - 0.0)))
    z_index = int(np.argmin(np.abs(axes_mm["z"] - 0.0)))
    x_cm = axes_mm["x"] / 10.0
    y_cm = axes_mm["y"] / 10.0
    z_cm = axes_mm["z"] / 10.0

    fig, axes = plt.subplots(2, 2, figsize=(12.5, 9.0), constrained_layout=True)
    panels = [
        (
            axes[0, 0],
            physical_dose[:, y_index, :],
            structures["PTV"][:, y_index, :],
            structures["SPINAL_CORD"][:, y_index, :],
            x_cm,
            z_cm,
            "Physical dose: axial-style slice",
            "x (cm)",
            "z (cm)",
        ),
        (
            axes[0, 1],
            physical_dose[:, :, z_index],
            structures["PTV"][:, :, z_index],
            structures["SPINAL_CORD"][:, :, z_index],
            x_cm,
            y_cm,
            "Physical dose: coronal slice",
            "x (cm)",
            "y (cm)",
        ),
        (
            axes[1, 0],
            bio_dose[:, y_index, :],
            structures["PTV"][:, y_index, :],
            structures["SPINAL_CORD"][:, y_index, :],
            x_cm,
            z_cm,
            "Bio-effective dose: axial-style slice",
            "x (cm)",
            "z (cm)",
        ),
        (
            axes[1, 1],
            bio_dose[:, :, z_index],
            structures["PTV"][:, :, z_index],
            structures["SPINAL_CORD"][:, :, z_index],
            x_cm,
            y_cm,
            "Bio-effective dose: coronal slice",
            "x (cm)",
            "y (cm)",
        ),
    ]

    for ax, image, ptv_slice, cord_slice, axis_a_cm, axis_b_cm, title, xlabel, ylabel in panels:
        im = ax.imshow(
            image.T,
            origin="lower",
            extent=[float(axis_a_cm[0]), float(axis_a_cm[-1]), float(axis_b_cm[0]), float(axis_b_cm[-1])],
            cmap="inferno",
        )
        ax.contour(axis_a_cm, axis_b_cm, ptv_slice.T.astype(float), levels=[0.5], colors=["cyan"], linewidths=1.2)
        ax.contour(axis_a_cm, axis_b_cm, cord_slice.T.astype(float), levels=[0.5], colors=["white"], linewidths=1.0)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Gy")

    fig.savefig(out_file, dpi=dpi)
    plt.close(fig)


def plot_dvh(
    out_file: Path,
    dose_axis_gy: np.ndarray,
    physical_curves: Dict[str, np.ndarray],
    bio_curves: Dict[str, np.ndarray],
    *,
    dpi: int,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.8), constrained_layout=True)
    structures_left = ["PTV", "GTV", "SPINAL_CORD", "BRAINSTEM"]
    structures_right = ["PAROTID_L", "PAROTID_R", "PTV"]
    colors = {
        "PTV": "tab:red",
        "GTV": "tab:orange",
        "SPINAL_CORD": "tab:blue",
        "BRAINSTEM": "tab:purple",
        "PAROTID_L": "tab:green",
        "PAROTID_R": "tab:olive",
    }

    for structure in structures_left:
        axes[0].plot(dose_axis_gy, physical_curves[structure], color=colors[structure], linestyle="-", label=f"{structure} physical")
        axes[0].plot(dose_axis_gy, bio_curves[structure], color=colors[structure], linestyle="--", label=f"{structure} bio")
    for structure in structures_right:
        axes[1].plot(dose_axis_gy, physical_curves[structure], color=colors[structure], linestyle="-", label=f"{structure} physical")
        axes[1].plot(dose_axis_gy, bio_curves[structure], color=colors[structure], linestyle="--", label=f"{structure} bio")

    axes[0].set_title("Target and serial-organ DVHs")
    axes[1].set_title("Parotid and target DVHs")
    for ax in axes:
        ax.set_xlabel("Dose / Gy")
        ax.set_ylabel("Volume receiving at least dose (%)")
        ax.set_ylim(0.0, 100.0)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)

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
        ("Parotid L mean", physical_metrics["PAROTID_L"]["mean_gy"], bio_metrics["PAROTID_L"]["mean_gy"]),
    ]
    x = np.arange(len(categories))
    width = 0.38
    fig, ax = plt.subplots(figsize=(11.0, 4.8), constrained_layout=True)
    ax.bar(x - width / 2.0, [row[1] for row in categories], width=width, label="Physical", color="tab:blue")
    ax.bar(x + width / 2.0, [row[2] for row in categories], width=width, label="Bio-effective", color="tab:red")
    ax.set_xticks(x)
    ax.set_xticklabels([row[0] for row in categories], rotation=20, ha="right")
    ax.set_ylabel("Gy")
    ax.set_title("Key plan metrics: physical versus bio-effective interpretation")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.savefig(out_file, dpi=dpi)
    plt.close(fig)


def write_markdown_report(
    out_file: Path,
    summary: Dict[str, object],
    physical_metrics: Dict[str, Dict[str, float]],
    bio_metrics: Dict[str, Dict[str, float]],
) -> None:
    lines = [
        "# Voxelized Head-and-Neck Lattice-Boost Summary",
        "",
        "## Phantom",
        f"- Model: {summary['phantom']['anatomical_note']}",
        f"- Grid: `{summary['phantom']['grid_shape']}` voxels at `{summary['phantom']['voxel_size_mm']}` mm",
        f"- PTV volume: `{summary['phantom']['structure_volumes_cc']['PTV']:.1f} cc`",
        f"- GTV volume: `{summary['phantom']['structure_volumes_cc']['GTV']:.1f} cc`",
        "",
        "## Plan",
        f"- Total histories: `{summary['plan']['histories']}`",
        f"- Lattice spots: `{summary['plan']['num_lattice_spots']}`",
        f"- Physical normalization: PTV D95 scaled to `{summary['prescription_gy']:.2f} Gy`",
        f"- Raw-to-clinical scale factor: `{summary['physical_scale_factor']:.4f}`",
        "",
        "## Physical Metrics",
        f"- PTV D95: `{physical_metrics['PTV']['d95_gy']:.2f} Gy` | V95: `{physical_metrics['PTV']['v95_pct']:.1f}%` | D2: `{physical_metrics['PTV']['d2_gy']:.2f} Gy`",
        f"- Spinal cord D2: `{physical_metrics['SPINAL_CORD']['d2_gy']:.2f} Gy`",
        f"- Brainstem D2: `{physical_metrics['BRAINSTEM']['d2_gy']:.2f} Gy`",
        f"- Right parotid mean: `{physical_metrics['PAROTID_R']['mean_gy']:.2f} Gy`",
        "",
        "## Bio-Effective Metrics",
        f"- PTV D95(eq): `{bio_metrics['PTV']['d95_gy']:.2f} Gy` | V95(eq): `{bio_metrics['PTV']['v95_pct']:.1f}%` | D2(eq): `{bio_metrics['PTV']['d2_gy']:.2f} Gy`",
        f"- Spinal cord D2(eq): `{bio_metrics['SPINAL_CORD']['d2_gy']:.2f} Gy`",
        f"- Brainstem D2(eq): `{bio_metrics['BRAINSTEM']['d2_gy']:.2f} Gy`",
        f"- Right parotid mean(eq): `{bio_metrics['PAROTID_R']['mean_gy']:.2f} Gy`",
        "",
        "## Interpretation",
        "- The biology model does not re-optimize the plan by itself; it reinterprets the same plan through nonlocal damage transport, anatomical washout, hypoxia, and immune coupling.",
        "- Differences between physical and bio-effective DVHs therefore indicate hidden biological burden or hidden biological benefit that would be missed by dose-only planning.",
    ]
    write_text_with_retries(out_file, "\n".join(lines) + "\n")


def main() -> int:
    args = parse_args()
    args.run_root.mkdir(parents=True, exist_ok=True)

    case_dir = args.run_root / "case"
    phantom_dir = args.run_root / "phantom"
    analysis_dir = args.run_root / "analysis"
    case_dir.mkdir(parents=True, exist_ok=True)
    phantom_dir.mkdir(parents=True, exist_ok=True)
    analysis_dir.mkdir(parents=True, exist_ok=True)

    phantom = build_headneck_phantom(args)
    tag_grid = phantom["tag_grid"]
    structures = phantom["structures"]
    axes_mm = phantom["axes_mm"]
    phantom_meta = phantom["meta"]

    spot_centers_mm = pick_lattice_spots(
        structures["GTV"],
        axes_mm,
        args.lattice_spacing_mm,
        spot_radius_mm=float(args.spot_radius_mm),
    )
    plan_meta = build_plan_sources(args, axes_mm, structures["PTV"], spot_centers_mm)
    sources = plan_meta["sources"]

    patient_bin = phantom_dir / "synthetic_headneck_tags.bin"
    write_image_cube(tag_grid, patient_bin)
    write_text_with_retries(phantom_dir / "phantom_meta.json", json.dumps(phantom_meta, indent=2))
    write_text_with_retries(
        phantom_dir / "lattice_spots.json",
        json.dumps(
            {
                "spot_centers_mm": [[float(a), float(b), float(c)] for a, b, c in spot_centers_mm],
                "plan_meta": {
                    "ptv_centroid_mm": plan_meta["ptv_centroid_mm"],
                    "ap_radius_mm": plan_meta["ap_radius_mm"],
                    "lateral_radius_mm": plan_meta["lateral_radius_mm"],
                    "num_sources": len(sources),
                },
            },
            indent=2,
        ),
    )

    spectrum_energies, spectrum_weights = load_spectrum(args.spectrum_csv)
    parameter_file = case_dir / "beamline.txt"
    dose_csv = case_dir / "dosedata.csv"
    log_file = case_dir / "topas.log"
    rendered = render_case_file(
        args,
        patient_bin=patient_bin,
        grid_shape=phantom_meta["grid_shape"],
        voxel_size_mm=phantom_meta["voxel_size_mm"],
        spectrum_energies=spectrum_energies,
        spectrum_weights=spectrum_weights,
        sources=sources,
    )
    write_text_with_retries(parameter_file, rendered)

    if not args.analyze_only:
        if args.skip_existing and has_nonempty_output(dose_csv):
            print(f"Reusing existing TOPAS output at {dose_csv}")
        else:
            print("=== RUNNING TOPAS VOXELIZED HEAD-AND-NECK LATTICE PLAN ===")
            run_topas_case(args, case_dir, parameter_file, dose_csv, log_file)

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
        raise RuntimeError("PTV raw D95 is non-positive; cannot normalize the plan.")
    physical_scale_factor = float(args.prescription_gy) / raw_d95
    physical_dose = dose_raw.astype(np.float32) * np.float32(physical_scale_factor)

    lq_survival = np.exp(-float(args.alpha) * physical_dose - float(args.beta) * physical_dose**2).astype(np.float32)
    uptake_tensor, m_type, m_oxygen, vessel_specs = build_biology_tensors(args, physical_dose, axes_mm, structures)
    hazard_grid = run_pde_temporal_integration(
        physical_dose,
        (float(args.voxel_mm), float(args.voxel_mm), float(args.voxel_mm)),
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
        (float(args.voxel_mm), float(args.voxel_mm), float(args.voxel_mm)),
        float(LOCKED_SCALING_FACTOR),
        weight_immune=float(W_IMMUNE),
        verbose=False,
    )
    bioeffective_dose = calculate_effective_dose(final_survival, alpha=float(args.alpha), beta=float(args.beta))
    immune_penalty, icd_volume_cm3 = calculate_systemic_immune_penalty(
        physical_dose,
        (float(args.voxel_mm), float(args.voxel_mm), float(args.voxel_mm)),
    )

    metric_config = {
        "PTV": {"prescription": float(args.prescription_gy), "vxs": [6.0, 10.0]},
        "GTV": {"prescription": float(args.prescription_gy), "vxs": [6.0, 10.0]},
        "SPINAL_CORD": {"prescription": None, "vxs": [5.0, 8.0]},
        "BRAINSTEM": {"prescription": None, "vxs": [5.0, 8.0]},
        "PAROTID_L": {"prescription": None, "vxs": [5.0, 10.0]},
        "PAROTID_R": {"prescription": None, "vxs": [5.0, 10.0]},
        "MANDIBLE": {"prescription": None, "vxs": [5.0, 10.0]},
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
            bioeffective_dose,
            structures[structure_name],
            prescription_gy=config["prescription"],
            voxel_volume_cc=voxel_volume_cc,
            volume_thresholds_gy=config["vxs"],
        )

    dose_axis = np.linspace(0.0, max(float(np.max(physical_dose)), float(np.max(bioeffective_dose))) * 1.05, 350)
    physical_dvhs = {name: compute_dvh(physical_dose[structures[name]], dose_axis) for name in metric_config}
    bio_dvhs = {name: compute_dvh(bioeffective_dose[structures[name]], dose_axis) for name in metric_config}

    dvh_rows: List[Dict[str, object]] = []
    for idx, dose_value in enumerate(dose_axis):
        row: Dict[str, object] = {"dose_gy": float(dose_value)}
        for structure_name in metric_config:
            row[f"{structure_name}_physical_pct"] = float(physical_dvhs[structure_name][idx])
            row[f"{structure_name}_bio_pct"] = float(bio_dvhs[structure_name][idx])
        dvh_rows.append(row)
    save_csv(dvh_rows, analysis_dir / "dvh_curves.csv")
    save_csv(metrics_table_rows(physical_metrics, "physical"), analysis_dir / "physical_plan_metrics.csv")
    save_csv(metrics_table_rows(bio_metrics, "bioeffective"), analysis_dir / "bioeffective_plan_metrics.csv")

    plot_anatomy(analysis_dir / "figure1_headneck_phantom_anatomy.png", axes_mm, structures, spot_centers_mm, dpi=int(args.dpi))
    plot_dose_slices(analysis_dir / "figure2_physical_vs_bioeffective_dose.png", axes_mm, physical_dose, bioeffective_dose, structures, dpi=int(args.dpi))
    plot_dvh(analysis_dir / "figure3_physical_vs_bioeffective_dvhs.png", dose_axis, physical_dvhs, bio_dvhs, dpi=int(args.dpi))
    plot_metric_bars(analysis_dir / "figure4_key_metric_comparison.png", physical_metrics, bio_metrics, dpi=int(args.dpi))

    summary = {
        "phantom": phantom_meta,
        "prescription_gy": float(args.prescription_gy),
        "physical_scale_factor": float(physical_scale_factor),
        "plan": {
            "histories": int(args.histories),
            "num_sources": int(len(sources)),
            "num_lattice_spots": int(len(spot_centers_mm)),
            "spot_centers_mm": [[float(a), float(b), float(c)] for a, b, c in spot_centers_mm],
            "base_history_fraction": float(args.base_history_fraction),
            "ap_radius_mm": float(plan_meta["ap_radius_mm"]),
            "lateral_radius_mm": float(plan_meta["lateral_radius_mm"]),
        },
        "biology": {
            "locked_D_cyto": LOCKED_D_CYTO,
            "locked_lambda_cyto": LOCKED_LAMBDA_CYTO,
            "locked_gamma": LOCKED_GAMMA,
            "locked_scaling_factor": LOCKED_SCALING_FACTOR,
            "immune_scalar": float(immune_penalty),
            "icd_volume_cm3": float(icd_volume_cm3),
            "vessel_specs_mm": vessel_specs,
        },
        "physical_metrics": physical_metrics,
        "bioeffective_metrics": bio_metrics,
    }
    write_text_with_retries(analysis_dir / "phase13_headneck_summary.json", json.dumps(summary, indent=2))
    write_markdown_report(analysis_dir / "phase13_headneck_summary.md", summary, physical_metrics, bio_metrics)

    print("\n=== PHASE 13 HEAD-AND-NECK VOXELIZED LATTICE COMPLETE ===")
    print(f"PTV physical D95 normalized to: {physical_metrics['PTV']['d95_gy']:.2f} Gy")
    print(f"PTV bio-effective D95: {bio_metrics['PTV']['d95_gy']:.2f} Gy")
    print(f"Spinal cord physical D2: {physical_metrics['SPINAL_CORD']['d2_gy']:.2f} Gy")
    print(f"Spinal cord bio-effective D2: {bio_metrics['SPINAL_CORD']['d2_gy']:.2f} Gy")
    print(f"Right parotid physical mean: {physical_metrics['PAROTID_R']['mean_gy']:.2f} Gy")
    print(f"Right parotid bio-effective mean: {bio_metrics['PAROTID_R']['mean_gy']:.2f} Gy")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
