#!/usr/bin/env python3
"""Generate a more anatomically detailed synthetic head-and-neck phantom.

This branch is intended for visual anatomy inspection and future planning work.
It extends the earlier audit-style surrogate with a thyroid/parathyroid complex,
bilateral salivary glands, major arteries and veins, and a richer vascular
network through the neck.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

from build_asymmetric_sweep import write_text_with_retries

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
except Exception as exc:  # pragma: no cover
    raise RuntimeError("matplotlib is required for phantom visualization") from exc


AIR_TAG = np.int16(0)
SOFT_TAG = np.int16(10)
BONE_TAG = np.int16(20)
VESSEL_TAG = np.int16(30)
GLAND_TAG = np.int16(40)
BRAIN_TAG = np.int16(50)


# Approximate bulk mass densities (g/cm^3) used to make the phantom physically heterogeneous.
DENSITY_G_CM3 = {
    "AIR": 0.0012,
    "SOFT_TISSUE": 1.04,
    "BRAIN": 1.04,
    "BLOOD_BRAIN_BARRIER": 1.06,
    "BRAINSTEM": 1.04,
    "SPINAL_CORD": 1.04,
    "PAROTID": 1.03,
    "SUBMANDIBULAR": 1.04,
    "THYROID": 1.05,
    "PARATHYROIDS": 1.05,
    "BLOOD": 1.06,
    "TUMOUR": 1.05,
    "SKULL_CORTICAL_BONE": 1.85,
    "MANDIBLE_MAXILLA_BONE": 1.80,
    "VERTEBRAL_BONE": 1.45,
}


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Generate a detailed synthetic head-and-neck phantom and inspection figures."
    )
    parser.add_argument(
        "--run-root",
        type=Path,
        default=root / "runs" / "detailed_headneck_phantom_v2",
        help="Output directory for phantom artifacts.",
    )
    parser.add_argument("--size-x-cm", type=float, default=20.0, help="Phantom left-right size.")
    parser.add_argument("--size-y-cm", type=float, default=26.0, help="Phantom superior-inferior size.")
    parser.add_argument("--size-z-cm", type=float, default=18.0, help="Phantom anterior-posterior size.")
    parser.add_argument("--voxel-mm", type=float, default=1.5, help="Isotropic voxel size.")
    parser.add_argument("--dpi", type=int, default=260, help="Figure DPI.")
    return parser.parse_args()


def centered_axis_mm(count: int, spacing_mm: float) -> np.ndarray:
    return (np.arange(int(count), dtype=np.float32) - (int(count) - 1) / 2.0) * float(spacing_mm)


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


def add_tube_segment(
    mask: np.ndarray,
    x_mm: np.ndarray,
    y_mm: np.ndarray,
    z_mm: np.ndarray,
    start_mm: Sequence[float],
    end_mm: Sequence[float],
    radius_mm: float,
) -> None:
    start = np.asarray(start_mm, dtype=np.float32)
    end = np.asarray(end_mm, dtype=np.float32)
    segment = end - start
    seg_len2 = float(np.dot(segment, segment))
    if seg_len2 <= 0.0:
        return

    margin = float(radius_mm) + max(float(x_mm[1] - x_mm[0]), float(y_mm[1] - y_mm[0]), float(z_mm[1] - z_mm[0]))
    mins = np.minimum(start, end) - margin
    maxs = np.maximum(start, end) + margin
    ix = np.flatnonzero((x_mm >= mins[0]) & (x_mm <= maxs[0]))
    iy = np.flatnonzero((y_mm >= mins[1]) & (y_mm <= maxs[1]))
    iz = np.flatnonzero((z_mm >= mins[2]) & (z_mm <= maxs[2]))
    if ix.size == 0 or iy.size == 0 or iz.size == 0:
        return

    xx = x_mm[ix][:, None, None]
    yy = y_mm[iy][None, :, None]
    zz = z_mm[iz][None, None, :]
    px = xx - start[0]
    py = yy - start[1]
    pz = zz - start[2]
    t = (px * segment[0] + py * segment[1] + pz * segment[2]) / seg_len2
    t = np.clip(t, 0.0, 1.0)
    closest_x = start[0] + t * segment[0]
    closest_y = start[1] + t * segment[1]
    closest_z = start[2] + t * segment[2]
    dist2 = (xx - closest_x) ** 2 + (yy - closest_y) ** 2 + (zz - closest_z) ** 2
    mask[np.ix_(ix, iy, iz)] |= dist2 <= float(radius_mm) ** 2


def polyline_tube_mask(
    x_mm: np.ndarray,
    y_mm: np.ndarray,
    z_mm: np.ndarray,
    *,
    nodes_mm: Sequence[Sequence[float]],
    radius_mm: float,
) -> np.ndarray:
    mask = np.zeros((x_mm.size, y_mm.size, z_mm.size), dtype=bool)
    nodes = [tuple(float(v) for v in node) for node in nodes_mm]
    for start, end in zip(nodes[:-1], nodes[1:]):
        add_tube_segment(mask, x_mm, y_mm, z_mm, start, end, float(radius_mm))
    return mask


def combine_polylines(
    x_mm: np.ndarray,
    y_mm: np.ndarray,
    z_mm: np.ndarray,
    specs: Sequence[dict],
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    union = np.zeros((x_mm.size, y_mm.size, z_mm.size), dtype=bool)
    individual: Dict[str, np.ndarray] = {}
    for spec in specs:
        mask = polyline_tube_mask(
            x_mm,
            y_mm,
            z_mm,
            nodes_mm=spec["nodes_mm"],
            radius_mm=float(spec["radius_mm"]),
        )
        individual[str(spec["name"])] = mask
        union |= mask
    return union, individual


def build_detailed_headneck_phantom(args: argparse.Namespace) -> Dict[str, object]:
    dx = dy = dz = float(args.voxel_mm)
    nx = int(round(float(args.size_x_cm) * 10.0 / dx))
    ny = int(round(float(args.size_y_cm) * 10.0 / dy))
    nz = int(round(float(args.size_z_cm) * 10.0 / dz))
    x_mm = centered_axis_mm(nx, dx)
    y_mm = centered_axis_mm(ny, dy)
    z_mm = centered_axis_mm(nz, dz)

    head = ellipsoid_mask(x_mm, y_mm, z_mm, center_mm=(0.0, 42.0, 0.0), radii_mm=(80.0, 78.0, 70.0))
    neck = ellipsoid_mask(x_mm, y_mm, z_mm, center_mm=(0.0, -30.0, 0.0), radii_mm=(66.0, 88.0, 50.0))
    shoulder_l = ellipsoid_mask(x_mm, y_mm, z_mm, center_mm=(-64.0, -96.0, 0.0), radii_mm=(46.0, 30.0, 56.0))
    shoulder_r = ellipsoid_mask(x_mm, y_mm, z_mm, center_mm=(64.0, -96.0, 0.0), radii_mm=(46.0, 30.0, 56.0))
    torso_stub = ellipsoid_mask(x_mm, y_mm, z_mm, center_mm=(0.0, -118.0, 0.0), radii_mm=(78.0, 24.0, 62.0))
    body_mask = head | neck | shoulder_l | shoulder_r | torso_stub

    skull_outer = ellipsoid_mask(x_mm, y_mm, z_mm, center_mm=(0.0, 44.0, 0.0), radii_mm=(75.0, 71.0, 64.0))
    skull_inner = ellipsoid_mask(x_mm, y_mm, z_mm, center_mm=(0.0, 44.0, 0.0), radii_mm=(67.0, 63.0, 57.0))
    skull_mask = skull_outer & ~skull_inner

    mandible_arch = capped_cylinder_along_y_mask(
        x_mm, y_mm, z_mm, center_x_mm=0.0, center_z_mm=-9.0, radius_x_mm=46.0, radius_z_mm=25.0, y_min_mm=-8.0, y_max_mm=15.0
    )
    mandible_gap = ellipsoid_mask(x_mm, y_mm, z_mm, center_mm=(0.0, 1.0, 10.0), radii_mm=(38.0, 18.0, 34.0))
    ramus_l = ellipsoid_mask(x_mm, y_mm, z_mm, center_mm=(-43.0, 10.0, 5.0), radii_mm=(8.0, 28.0, 10.0))
    ramus_r = ellipsoid_mask(x_mm, y_mm, z_mm, center_mm=(43.0, 10.0, 5.0), radii_mm=(8.0, 28.0, 10.0))
    maxilla = ellipsoid_mask(x_mm, y_mm, z_mm, center_mm=(0.0, 17.0, -3.0), radii_mm=(34.0, 14.0, 18.0))
    mandible_mask = ((mandible_arch & ~mandible_gap) | ramus_l | ramus_r) & body_mask

    vertebrae_mask = np.zeros((nx, ny, nz), dtype=bool)
    for center_y in np.arange(-76.0, 42.0, 16.0):
        vertebral_body = ellipsoid_mask(x_mm, y_mm, z_mm, center_mm=(0.0, float(center_y), 24.0), radii_mm=(13.0, 8.0, 10.0))
        spinous = ellipsoid_mask(x_mm, y_mm, z_mm, center_mm=(0.0, float(center_y), 35.0), radii_mm=(4.0, 9.0, 6.0))
        transverse_l = ellipsoid_mask(x_mm, y_mm, z_mm, center_mm=(-10.0, float(center_y), 25.0), radii_mm=(4.0, 6.0, 5.0))
        transverse_r = ellipsoid_mask(x_mm, y_mm, z_mm, center_mm=(10.0, float(center_y), 25.0), radii_mm=(4.0, 6.0, 5.0))
        vertebrae_mask |= vertebral_body | spinous | transverse_l | transverse_r

    airway_mask = capped_cylinder_along_y_mask(
        x_mm, y_mm, z_mm, center_x_mm=0.0, center_z_mm=-9.0, radius_x_mm=9.0, radius_z_mm=11.0, y_min_mm=-6.0, y_max_mm=36.0
    )
    trachea_mask = cylinder_along_y_mask(
        x_mm, y_mm, z_mm, center_x_mm=0.0, center_z_mm=-8.0, radius_mm=7.0, y_min_mm=-86.0, y_max_mm=-6.0
    )
    oral_cavity = ellipsoid_mask(x_mm, y_mm, z_mm, center_mm=(0.0, 18.0, -18.0), radii_mm=(17.0, 12.0, 11.0))
    larynx = ellipsoid_mask(x_mm, y_mm, z_mm, center_mm=(0.0, -12.0, -5.0), radii_mm=(13.0, 16.0, 10.0))
    esophagus = capped_cylinder_along_y_mask(
        x_mm, y_mm, z_mm, center_x_mm=0.0, center_z_mm=6.0, radius_x_mm=5.0, radius_z_mm=4.0, y_min_mm=-86.0, y_max_mm=-10.0
    )
    airway_complex = airway_mask | trachea_mask | oral_cavity

    parotid_l = ellipsoid_mask(x_mm, y_mm, z_mm, center_mm=(-50.0, -4.0, -4.0), radii_mm=(16.0, 25.0, 13.0)) & body_mask
    parotid_r = ellipsoid_mask(x_mm, y_mm, z_mm, center_mm=(50.0, -4.0, -4.0), radii_mm=(16.0, 25.0, 13.0)) & body_mask
    submandibular_l = ellipsoid_mask(x_mm, y_mm, z_mm, center_mm=(-31.0, -16.0, -5.0), radii_mm=(12.0, 10.0, 8.0)) & body_mask
    submandibular_r = ellipsoid_mask(x_mm, y_mm, z_mm, center_mm=(31.0, -16.0, -5.0), radii_mm=(12.0, 10.0, 8.0)) & body_mask

    spinal_cord = cylinder_along_y_mask(
        x_mm, y_mm, z_mm, center_x_mm=0.0, center_z_mm=22.5, radius_mm=4.5, y_min_mm=-82.0, y_max_mm=62.0
    ) & body_mask
    brain_mask = ellipsoid_mask(x_mm, y_mm, z_mm, center_mm=(0.0, 46.0, 0.0), radii_mm=(61.0, 58.0, 52.0))
    brain_mask &= skull_inner & body_mask
    brainstem = ellipsoid_mask(x_mm, y_mm, z_mm, center_mm=(0.0, 58.0, 19.0), radii_mm=(10.0, 18.0, 12.0)) & body_mask
    bbb_outer = ellipsoid_mask(x_mm, y_mm, z_mm, center_mm=(0.0, 46.0, 0.0), radii_mm=(63.5, 60.5, 54.5))
    bbb_inner = ellipsoid_mask(x_mm, y_mm, z_mm, center_mm=(0.0, 46.0, 0.0), radii_mm=(61.8, 58.8, 52.8))
    blood_brain_barrier = (bbb_outer & ~bbb_inner) & skull_inner & body_mask

    thyroid_lobe_l = ellipsoid_mask(x_mm, y_mm, z_mm, center_mm=(-18.0, -47.0, -2.0), radii_mm=(10.0, 16.0, 7.0))
    thyroid_lobe_r = ellipsoid_mask(x_mm, y_mm, z_mm, center_mm=(18.0, -47.0, -2.0), radii_mm=(10.0, 16.0, 7.0))
    thyroid_isthmus = ellipsoid_mask(x_mm, y_mm, z_mm, center_mm=(0.0, -48.0, -4.0), radii_mm=(9.0, 5.0, 4.0))
    thyroid_mask = (thyroid_lobe_l | thyroid_lobe_r | thyroid_isthmus) & body_mask & ~trachea_mask

    parathyroid_l_sup = ellipsoid_mask(x_mm, y_mm, z_mm, center_mm=(-18.0, -40.0, 4.0), radii_mm=(3.2, 4.0, 2.8))
    parathyroid_l_inf = ellipsoid_mask(x_mm, y_mm, z_mm, center_mm=(-18.0, -55.0, 4.0), radii_mm=(3.2, 4.0, 2.8))
    parathyroid_r_sup = ellipsoid_mask(x_mm, y_mm, z_mm, center_mm=(18.0, -40.0, 4.0), radii_mm=(3.2, 4.0, 2.8))
    parathyroid_r_inf = ellipsoid_mask(x_mm, y_mm, z_mm, center_mm=(18.0, -55.0, 4.0), radii_mm=(3.2, 4.0, 2.8))
    parathyroid_mask = (parathyroid_l_sup | parathyroid_l_inf | parathyroid_r_sup | parathyroid_r_inf) & body_mask

    tumor_primary = ellipsoid_mask(x_mm, y_mm, z_mm, center_mm=(20.0, 4.0, -4.0), radii_mm=(25.0, 27.0, 22.0))
    tumor_nodal = ellipsoid_mask(x_mm, y_mm, z_mm, center_mm=(34.0, -35.0, 1.0), radii_mm=(21.0, 34.0, 18.0))
    tumor_mask = (tumor_primary | tumor_nodal) & body_mask & ~airway_complex & ~spinal_cord & ~brainstem

    artery_specs = [
        {"name": "common_carotid_l", "nodes_mm": [(-22, -88, 2), (-22, -60, 2), (-21, -36, 3), (-18, -12, 4), (-16, 5, 4)], "radius_mm": 3.3},
        {"name": "common_carotid_r", "nodes_mm": [(22, -88, 2), (22, -60, 2), (21, -36, 3), (18, -12, 4), (16, 5, 4)], "radius_mm": 3.3},
        {"name": "internal_carotid_l", "nodes_mm": [(-16, 5, 4), (-18, 22, 6), (-20, 40, 7), (-18, 60, 8)], "radius_mm": 2.6},
        {"name": "internal_carotid_r", "nodes_mm": [(16, 5, 4), (18, 22, 6), (20, 40, 7), (18, 60, 8)], "radius_mm": 2.6},
        {"name": "external_carotid_l", "nodes_mm": [(-16, 5, 4), (-14, 16, 1), (-11, 28, -2), (-8, 42, -6)], "radius_mm": 2.1},
        {"name": "external_carotid_r", "nodes_mm": [(16, 5, 4), (14, 16, 1), (11, 28, -2), (8, 42, -6)], "radius_mm": 2.1},
        {"name": "vertebral_artery_l", "nodes_mm": [(-12, -86, 18), (-11, -52, 19), (-10, -20, 20), (-8, 12, 21), (-6, 42, 22)], "radius_mm": 1.5},
        {"name": "vertebral_artery_r", "nodes_mm": [(12, -86, 18), (11, -52, 19), (10, -20, 20), (8, 12, 21), (6, 42, 22)], "radius_mm": 1.5},
        {"name": "superior_thyroid_artery_l", "nodes_mm": [(-14, -24, 1), (-16, -32, -1), (-19, -42, -2)], "radius_mm": 1.2},
        {"name": "superior_thyroid_artery_r", "nodes_mm": [(14, -24, 1), (16, -32, -1), (19, -42, -2)], "radius_mm": 1.2},
        {"name": "lingual_facial_branch_l", "nodes_mm": [(-11, 16, 1), (-18, 20, -5), (-28, 12, -12)], "radius_mm": 1.1},
        {"name": "lingual_facial_branch_r", "nodes_mm": [(11, 16, 1), (18, 20, -5), (28, 12, -12)], "radius_mm": 1.1},
    ]

    vein_specs = [
        {"name": "internal_jugular_l", "nodes_mm": [(-31, -90, 3), (-30, -58, 3), (-28, -25, 4), (-26, 8, 5), (-25, 35, 6), (-23, 62, 7)], "radius_mm": 4.4},
        {"name": "internal_jugular_r", "nodes_mm": [(31, -90, 3), (30, -58, 3), (28, -25, 4), (26, 8, 5), (25, 35, 6), (23, 62, 7)], "radius_mm": 4.4},
        {"name": "external_jugular_l", "nodes_mm": [(-41, -72, -2), (-42, -40, -1), (-43, -8, 0), (-44, 18, 1)], "radius_mm": 2.0},
        {"name": "external_jugular_r", "nodes_mm": [(41, -72, -2), (42, -40, -1), (43, -8, 0), (44, 18, 1)], "radius_mm": 2.0},
        {"name": "thyroid_venous_plexus", "nodes_mm": [(-20, -53, -1), (-8, -50, -2), (0, -50, -3), (8, -50, -2), (20, -53, -1)], "radius_mm": 1.8},
        {"name": "middle_thyroid_vein_l", "nodes_mm": [(-20, -47, -1), (-26, -44, 1), (-30, -40, 3)], "radius_mm": 1.3},
        {"name": "middle_thyroid_vein_r", "nodes_mm": [(20, -47, -1), (26, -44, 1), (30, -40, 3)], "radius_mm": 1.3},
        {"name": "anterior_jugular_arch", "nodes_mm": [(-12, -70, -2), (-8, -64, -1), (0, -62, -1), (8, -64, -1), (12, -70, -2)], "radius_mm": 1.4},
    ]

    arteries_mask, artery_individual = combine_polylines(x_mm, y_mm, z_mm, artery_specs)
    veins_mask, vein_individual = combine_polylines(x_mm, y_mm, z_mm, vein_specs)
    arteries_mask &= body_mask & ~airway_complex
    veins_mask &= body_mask & ~airway_complex

    body_mask &= ~airway_complex

    tag_grid = np.full((nx, ny, nz), AIR_TAG, dtype=np.int16)
    tag_grid[body_mask] = SOFT_TAG
    tag_grid[brain_mask | blood_brain_barrier] = BRAIN_TAG
    tag_grid[skull_mask | mandible_mask | maxilla | vertebrae_mask] = BONE_TAG
    tag_grid[thyroid_mask | parotid_l | parotid_r | submandibular_l | submandibular_r | parathyroid_mask] = GLAND_TAG
    tag_grid[arteries_mask | veins_mask] = VESSEL_TAG

    density_grid = np.full((nx, ny, nz), DENSITY_G_CM3["AIR"], dtype=np.float32)
    density_grid[body_mask] = DENSITY_G_CM3["SOFT_TISSUE"]
    density_grid[parotid_l | parotid_r] = DENSITY_G_CM3["PAROTID"]
    density_grid[submandibular_l | submandibular_r] = DENSITY_G_CM3["SUBMANDIBULAR"]
    density_grid[thyroid_mask] = DENSITY_G_CM3["THYROID"]
    density_grid[parathyroid_mask] = DENSITY_G_CM3["PARATHYROIDS"]
    density_grid[brain_mask] = DENSITY_G_CM3["BRAIN"]
    density_grid[blood_brain_barrier] = DENSITY_G_CM3["BLOOD_BRAIN_BARRIER"]
    density_grid[brainstem] = DENSITY_G_CM3["BRAINSTEM"]
    density_grid[spinal_cord] = DENSITY_G_CM3["SPINAL_CORD"]
    density_grid[arteries_mask | veins_mask] = DENSITY_G_CM3["BLOOD"]
    density_grid[tumor_mask] = DENSITY_G_CM3["TUMOUR"]
    density_grid[maxilla | mandible_mask] = DENSITY_G_CM3["MANDIBLE_MAXILLA_BONE"]
    density_grid[skull_mask] = DENSITY_G_CM3["SKULL_CORTICAL_BONE"]
    density_grid[vertebrae_mask] = DENSITY_G_CM3["VERTEBRAL_BONE"]
    density_grid[airway_complex | trachea_mask] = DENSITY_G_CM3["AIR"]

    structures = {
        "BODY": body_mask,
        "SKULL": skull_mask,
        "MANDIBLE": mandible_mask,
        "MAXILLA": maxilla & body_mask,
        "VERTEBRAE": vertebrae_mask & body_mask,
        "AIRWAY": airway_complex,
        "TRACHEA": trachea_mask,
        "LARYNX": larynx & body_mask,
        "ESOPHAGUS": esophagus & body_mask,
        "PAROTID_L": parotid_l,
        "PAROTID_R": parotid_r,
        "SUBMANDIBULAR_L": submandibular_l,
        "SUBMANDIBULAR_R": submandibular_r,
        "SPINAL_CORD": spinal_cord,
        "BRAIN": brain_mask,
        "BLOOD_BRAIN_BARRIER": blood_brain_barrier,
        "BRAINSTEM": brainstem,
        "THYROID": thyroid_mask,
        "PARATHYROIDS": parathyroid_mask,
        "ARTERIES": arteries_mask,
        "VEINS": veins_mask,
        "TUMOUR": tumor_mask,
    }

    structures.update({name.upper(): mask & body_mask for name, mask in artery_individual.items()})
    structures.update({name.upper(): mask & body_mask for name, mask in vein_individual.items()})

    voxel_volume_cc = (dx * dy * dz) / 1000.0
    meta = {
        "grid_shape": [int(nx), int(ny), int(nz)],
        "voxel_size_mm": [dx, dy, dz],
        "size_cm": [float(args.size_x_cm), float(args.size_y_cm), float(args.size_z_cm)],
        "voxel_volume_cc": float(voxel_volume_cc),
        "assigned_densities_g_cm3": DENSITY_G_CM3,
        "structure_volumes_cc": {
            name: float(np.count_nonzero(mask) * voxel_volume_cc)
            for name, mask in structures.items()
            if name in {
                "BODY",
                "THYROID",
                "PARATHYROIDS",
                "PAROTID_L",
                "PAROTID_R",
                "SUBMANDIBULAR_L",
                "SUBMANDIBULAR_R",
                "ARTERIES",
                "VEINS",
                "BRAIN",
                "BLOOD_BRAIN_BARRIER",
                "SPINAL_CORD",
                "BRAINSTEM",
                "TUMOUR",
            }
        },
        "anatomical_note": (
            "Detailed synthetic head-and-neck phantom with intracranial brain and blood-brain "
            "barrier shell, thyroid-parathyroid complex, major carotid/jugular vessels, "
            "vertebral arteries, thyroid vasculature, salivary glands, airway-larynx-trachea "
            "complex, cervical spine, and a bulky right-sided oropharyngeal-nodal tumour surrogate."
        ),
    }
    return {
        "tag_grid": tag_grid,
        "density_grid_g_cm3": density_grid,
        "structures": structures,
        "axes_mm": {"x": x_mm, "y": y_mm, "z": z_mm},
        "meta": meta,
    }


def save_masks_npz(structures: Dict[str, np.ndarray], out_file: Path) -> None:
    np.savez_compressed(out_file, **{name: mask.astype(np.uint8) for name, mask in structures.items()})


def save_density_npz(density_grid_g_cm3: np.ndarray, out_file: Path) -> None:
    np.savez_compressed(out_file, density_g_cm3=density_grid_g_cm3.astype(np.float32))


def draw_projection(
    ax,
    *,
    body_proj: np.ndarray,
    bone_proj: np.ndarray,
    brain_proj: np.ndarray,
    bbb_proj: np.ndarray,
    tumour_proj: np.ndarray,
    thyroid_proj: np.ndarray,
    parathyroid_proj: np.ndarray,
    artery_proj: np.ndarray,
    vein_proj: np.ndarray,
    parotid_proj: np.ndarray,
    submandibular_proj: np.ndarray,
    airway_proj: np.ndarray,
    axis_a_cm: np.ndarray,
    axis_b_cm: np.ndarray,
    title: str,
    xlabel: str,
    ylabel: str,
) -> None:
    extent = [float(axis_a_cm[0]), float(axis_a_cm[-1]), float(axis_b_cm[0]), float(axis_b_cm[-1])]
    ax.imshow(body_proj.T, origin="lower", cmap="Greys", extent=extent, alpha=0.92)
    ax.contour(axis_a_cm, axis_b_cm, bone_proj.T.astype(float), levels=[0.5], colors=["white"], linewidths=0.8)
    ax.contour(axis_a_cm, axis_b_cm, brain_proj.T.astype(float), levels=[0.5], colors=["#8ECAE6"], linewidths=1.0)
    ax.contour(
        axis_a_cm,
        axis_b_cm,
        bbb_proj.T.astype(float),
        levels=[0.5],
        colors=["#00B4D8"],
        linewidths=1.1,
        linestyles=":",
    )
    ax.contour(axis_a_cm, axis_b_cm, airway_proj.T.astype(float), levels=[0.5], colors=["black"], linewidths=1.0)
    ax.contour(axis_a_cm, axis_b_cm, parotid_proj.T.astype(float), levels=[0.5], colors=["#56B870"], linewidths=1.2)
    ax.contour(axis_a_cm, axis_b_cm, submandibular_proj.T.astype(float), levels=[0.5], colors=["#8FD175"], linewidths=1.0)
    ax.contour(axis_a_cm, axis_b_cm, thyroid_proj.T.astype(float), levels=[0.5], colors=["#FF9F1C"], linewidths=1.5)
    ax.contour(axis_a_cm, axis_b_cm, parathyroid_proj.T.astype(float), levels=[0.5], colors=["#FFD166"], linewidths=1.5)
    ax.contour(axis_a_cm, axis_b_cm, artery_proj.T.astype(float), levels=[0.5], colors=["#D62828"], linewidths=1.2)
    ax.contour(axis_a_cm, axis_b_cm, vein_proj.T.astype(float), levels=[0.5], colors=["#277DA1"], linewidths=1.2)
    ax.contour(axis_a_cm, axis_b_cm, tumour_proj.T.astype(float), levels=[0.5], colors=["#C51B8A"], linewidths=1.4, linestyles="--")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)


def plot_density_maps(out_file: Path, phantom: Dict[str, object], *, dpi: int) -> None:
    axes_mm = phantom["axes_mm"]
    structures = phantom["structures"]
    density = phantom["density_grid_g_cm3"]

    x_cm = axes_mm["x"] / 10.0
    y_cm = axes_mm["y"] / 10.0
    z_cm = axes_mm["z"] / 10.0

    ix_mid = int(np.argmin(np.abs(axes_mm["x"] - 0.0)))
    iy_head = int(np.argmin(np.abs(axes_mm["y"] - 42.0)))
    iy_neck = int(np.argmin(np.abs(axes_mm["y"] + 48.0)))
    iz_mid = int(np.argmin(np.abs(axes_mm["z"] - 0.0)))

    density_cor = density[:, :, iz_mid]
    density_ax_head = density[:, iy_head, :]
    density_ax_neck = density[:, iy_neck, :]
    density_sag = density[ix_mid, :, :]

    body_cor = structures["BODY"][:, :, iz_mid]
    body_ax_head = structures["BODY"][:, iy_head, :]
    body_ax_neck = structures["BODY"][:, iy_neck, :]
    body_sag = structures["BODY"][ix_mid, :, :]

    tumour_cor = structures["TUMOUR"][:, :, iz_mid]
    tumour_ax_neck = structures["TUMOUR"][:, iy_neck, :]
    thyroid_cor = structures["THYROID"][:, :, iz_mid]
    thyroid_ax_neck = structures["THYROID"][:, iy_neck, :]
    bbb_cor = structures["BLOOD_BRAIN_BARRIER"][:, :, iz_mid]
    bbb_ax_head = structures["BLOOD_BRAIN_BARRIER"][:, iy_head, :]
    cord_cor = structures["SPINAL_CORD"][:, :, iz_mid]
    cord_sag = structures["SPINAL_CORD"][ix_mid, :, :]

    fig = plt.figure(figsize=(15.5, 11.5), constrained_layout=True)
    gs = fig.add_gridspec(2, 2)
    ax_cor = fig.add_subplot(gs[0, 0])
    ax_ax_head = fig.add_subplot(gs[0, 1])
    ax_ax_neck = fig.add_subplot(gs[1, 0])
    ax_sag = fig.add_subplot(gs[1, 1])

    panels = [
        (ax_cor, density_cor.T, x_cm, y_cm, "Coronal density slice (z = 0 cm)", "x (cm)", "y (cm)", body_cor.T),
        (
            ax_ax_head,
            density_ax_head.T,
            x_cm,
            z_cm,
            "Axial density slice at brain level",
            "x (cm)",
            "z (cm)",
            body_ax_head.T,
        ),
        (
            ax_ax_neck,
            density_ax_neck.T,
            x_cm,
            z_cm,
            "Axial density slice at thyroid / nodal level",
            "x (cm)",
            "z (cm)",
            body_ax_neck.T,
        ),
        (ax_sag, density_sag.T, y_cm, z_cm, "Sagittal density slice (x = 0 cm)", "y (cm)", "z (cm)", body_sag.T),
    ]

    image = None
    for ax, arr, axis_a, axis_b, title, xlabel, ylabel, mask in panels:
        extent = [float(axis_a[0]), float(axis_a[-1]), float(axis_b[0]), float(axis_b[-1])]
        image = ax.imshow(arr, origin="lower", cmap="magma", vmin=0.0, vmax=1.9, extent=extent)
        ax.contour(axis_a, axis_b, mask.astype(float), levels=[0.5], colors=["white"], linewidths=0.8, alpha=0.8)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

    ax_cor.contour(x_cm, y_cm, tumour_cor.T.astype(float), levels=[0.5], colors=["#00E5FF"], linewidths=1.0, linestyles="--")
    ax_cor.contour(x_cm, y_cm, thyroid_cor.T.astype(float), levels=[0.5], colors=["#FFD166"], linewidths=1.0)
    ax_cor.contour(x_cm, y_cm, bbb_cor.T.astype(float), levels=[0.5], colors=["#8ECAE6"], linewidths=0.9, linestyles=":")
    ax_cor.contour(x_cm, y_cm, cord_cor.T.astype(float), levels=[0.5], colors=["#A3A3A3"], linewidths=0.9)
    ax_ax_head.contour(x_cm, z_cm, bbb_ax_head.T.astype(float), levels=[0.5], colors=["#8ECAE6"], linewidths=0.9, linestyles=":")
    ax_ax_neck.contour(x_cm, z_cm, tumour_ax_neck.T.astype(float), levels=[0.5], colors=["#00E5FF"], linewidths=1.0, linestyles="--")
    ax_ax_neck.contour(x_cm, z_cm, thyroid_ax_neck.T.astype(float), levels=[0.5], colors=["#FFD166"], linewidths=1.0)
    ax_sag.contour(y_cm, z_cm, cord_sag.T.astype(float), levels=[0.5], colors=["#A3A3A3"], linewidths=0.9)

    cbar = fig.colorbar(image, ax=[ax_cor, ax_ax_head, ax_ax_neck, ax_sag], shrink=0.92, pad=0.02)
    cbar.set_label("Density (g/cm$^3$)")
    fig.suptitle("Heterogeneous density map for the detailed synthetic head-and-neck phantom", fontsize=14)
    fig.savefig(out_file, dpi=dpi)
    plt.close(fig)


def plot_detailed_phantom(out_file: Path, phantom: Dict[str, object], *, dpi: int) -> None:
    axes_mm = phantom["axes_mm"]
    structures = phantom["structures"]
    x_cm = axes_mm["x"] / 10.0
    y_cm = axes_mm["y"] / 10.0
    z_cm = axes_mm["z"] / 10.0

    body_xy = np.any(structures["BODY"], axis=2)
    body_xz = np.any(structures["BODY"], axis=1)
    body_yz = np.any(structures["BODY"], axis=0)
    bone_xy = np.any(structures["SKULL"] | structures["MANDIBLE"] | structures["VERTEBRAE"], axis=2)
    bone_xz = np.any(structures["SKULL"] | structures["MANDIBLE"] | structures["VERTEBRAE"], axis=1)
    bone_yz = np.any(structures["SKULL"] | structures["MANDIBLE"] | structures["VERTEBRAE"], axis=0)
    brain_xy = np.any(structures["BRAIN"], axis=2)
    brain_xz = np.any(structures["BRAIN"], axis=1)
    brain_yz = np.any(structures["BRAIN"], axis=0)
    bbb_xy = np.any(structures["BLOOD_BRAIN_BARRIER"], axis=2)
    bbb_xz = np.any(structures["BLOOD_BRAIN_BARRIER"], axis=1)
    bbb_yz = np.any(structures["BLOOD_BRAIN_BARRIER"], axis=0)
    tumour_xy = np.any(structures["TUMOUR"], axis=2)
    tumour_xz = np.any(structures["TUMOUR"], axis=1)
    tumour_yz = np.any(structures["TUMOUR"], axis=0)
    thyroid_xy = np.any(structures["THYROID"], axis=2)
    thyroid_xz = np.any(structures["THYROID"], axis=1)
    thyroid_yz = np.any(structures["THYROID"], axis=0)
    parathyroid_xy = np.any(structures["PARATHYROIDS"], axis=2)
    parathyroid_xz = np.any(structures["PARATHYROIDS"], axis=1)
    parathyroid_yz = np.any(structures["PARATHYROIDS"], axis=0)
    artery_xy = np.any(structures["ARTERIES"], axis=2)
    artery_xz = np.any(structures["ARTERIES"], axis=1)
    artery_yz = np.any(structures["ARTERIES"], axis=0)
    vein_xy = np.any(structures["VEINS"], axis=2)
    vein_xz = np.any(structures["VEINS"], axis=1)
    vein_yz = np.any(structures["VEINS"], axis=0)
    parotid_xy = np.any(structures["PAROTID_L"] | structures["PAROTID_R"], axis=2)
    parotid_xz = np.any(structures["PAROTID_L"] | structures["PAROTID_R"], axis=1)
    parotid_yz = np.any(structures["PAROTID_L"] | structures["PAROTID_R"], axis=0)
    submand_xy = np.any(structures["SUBMANDIBULAR_L"] | structures["SUBMANDIBULAR_R"], axis=2)
    submand_xz = np.any(structures["SUBMANDIBULAR_L"] | structures["SUBMANDIBULAR_R"], axis=1)
    submand_yz = np.any(structures["SUBMANDIBULAR_L"] | structures["SUBMANDIBULAR_R"], axis=0)
    airway_xy = np.any(structures["AIRWAY"], axis=2)
    airway_xz = np.any(structures["AIRWAY"], axis=1)
    airway_yz = np.any(structures["AIRWAY"], axis=0)

    fig = plt.figure(figsize=(16.0, 12.0), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.0], width_ratios=[1.0, 1.0])
    ax_cor = fig.add_subplot(gs[0, 0])
    ax_ax = fig.add_subplot(gs[0, 1])
    ax_sag = fig.add_subplot(gs[1, 0])
    ax_zoom = fig.add_subplot(gs[1, 1])

    draw_projection(
        ax_cor,
        body_proj=body_xy,
        bone_proj=bone_xy,
        brain_proj=brain_xy,
        bbb_proj=bbb_xy,
        tumour_proj=tumour_xy,
        thyroid_proj=thyroid_xy,
        parathyroid_proj=parathyroid_xy,
        artery_proj=artery_xy,
        vein_proj=vein_xy,
        parotid_proj=parotid_xy,
        submandibular_proj=submand_xy,
        airway_proj=airway_xy,
        axis_a_cm=x_cm,
        axis_b_cm=y_cm,
        title="Coronal projection",
        xlabel="x (cm)",
        ylabel="y (cm)",
    )

    draw_projection(
        ax_ax,
        body_proj=body_xz,
        bone_proj=bone_xz,
        brain_proj=brain_xz,
        bbb_proj=bbb_xz,
        tumour_proj=tumour_xz,
        thyroid_proj=thyroid_xz,
        parathyroid_proj=parathyroid_xz,
        artery_proj=artery_xz,
        vein_proj=vein_xz,
        parotid_proj=parotid_xz,
        submandibular_proj=submand_xz,
        airway_proj=airway_xz,
        axis_a_cm=x_cm,
        axis_b_cm=z_cm,
        title="Axial-style projection",
        xlabel="x (cm)",
        ylabel="z (cm)",
    )

    draw_projection(
        ax_sag,
        body_proj=body_yz,
        bone_proj=bone_yz,
        brain_proj=brain_yz,
        bbb_proj=bbb_yz,
        tumour_proj=tumour_yz,
        thyroid_proj=thyroid_yz,
        parathyroid_proj=parathyroid_yz,
        artery_proj=artery_yz,
        vein_proj=vein_yz,
        parotid_proj=parotid_yz,
        submandibular_proj=submand_yz,
        airway_proj=airway_yz,
        axis_a_cm=y_cm,
        axis_b_cm=z_cm,
        title="Sagittal projection",
        xlabel="y (cm)",
        ylabel="z (cm)",
    )

    draw_projection(
        ax_zoom,
        body_proj=body_xy,
        bone_proj=bone_xy,
        brain_proj=brain_xy,
        bbb_proj=bbb_xy,
        tumour_proj=tumour_xy,
        thyroid_proj=thyroid_xy,
        parathyroid_proj=parathyroid_xy,
        artery_proj=artery_xy,
        vein_proj=vein_xy,
        parotid_proj=parotid_xy,
        submandibular_proj=submand_xy,
        airway_proj=airway_xy,
        axis_a_cm=x_cm,
        axis_b_cm=y_cm,
        title="Neck vascular and endocrine close-up",
        xlabel="x (cm)",
        ylabel="y (cm)",
    )
    ax_zoom.set_xlim(-5.5, 5.5)
    ax_zoom.set_ylim(-8.5, 1.0)
    ax_zoom.annotate("Thyroid", xy=(2.0, -4.7), xytext=(2.8, -2.5), arrowprops={"arrowstyle": "->", "lw": 0.8}, fontsize=9, color="#FF9F1C")
    ax_zoom.annotate("Parathyroids", xy=(1.9, -4.0), xytext=(-4.8, -1.5), arrowprops={"arrowstyle": "->", "lw": 0.8}, fontsize=9, color="#FFD166")
    ax_zoom.annotate("Carotid sheath", xy=(2.4, -3.5), xytext=(2.8, -7.5), arrowprops={"arrowstyle": "->", "lw": 0.8}, fontsize=9, color="#D62828")
    ax_zoom.annotate("Jugular vein", xy=(-3.1, -4.3), xytext=(-5.2, -7.4), arrowprops={"arrowstyle": "->", "lw": 0.8}, fontsize=9, color="#277DA1")

    legend_handles = [
        Line2D([0], [0], color="white", lw=1.2, label="Bone contour"),
        Line2D([0], [0], color="#8ECAE6", lw=1.2, label="Brain"),
        Line2D([0], [0], color="#00B4D8", lw=1.2, linestyle=":", label="Blood-brain barrier"),
        Line2D([0], [0], color="black", lw=1.2, label="Airway / trachea"),
        Line2D([0], [0], color="#56B870", lw=1.4, label="Parotid glands"),
        Line2D([0], [0], color="#8FD175", lw=1.2, label="Submandibular glands"),
        Line2D([0], [0], color="#FF9F1C", lw=1.6, label="Thyroid"),
        Line2D([0], [0], color="#FFD166", lw=1.6, label="Parathyroids"),
        Line2D([0], [0], color="#D62828", lw=1.4, label="Arteries"),
        Line2D([0], [0], color="#277DA1", lw=1.4, label="Veins"),
        Line2D([0], [0], color="#C51B8A", lw=1.4, linestyle="--", label="Tumour surrogate"),
    ]
    ax_zoom.legend(handles=legend_handles, loc="lower right", fontsize=8)
    fig.suptitle(
        "Detailed synthetic head-and-neck phantom with brain, blood-brain barrier, vascular, endocrine, and glandular anatomy",
        fontsize=14,
    )
    fig.savefig(out_file, dpi=dpi)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    args.run_root.mkdir(parents=True, exist_ok=True)
    phantom_dir = args.run_root / "phantom"
    phantom_dir.mkdir(parents=True, exist_ok=True)

    phantom = build_detailed_headneck_phantom(args)
    structures = phantom["structures"]
    meta = phantom["meta"]
    density_grid = phantom["density_grid_g_cm3"]

    save_masks_npz(structures, phantom_dir / "detailed_headneck_structures.npz")
    save_density_npz(density_grid, phantom_dir / "detailed_headneck_density_map.npz")
    write_text_with_retries(phantom_dir / "detailed_headneck_summary.json", json.dumps(meta, indent=2))
    write_text_with_retries(phantom_dir / "anatomical_note.txt", meta["anatomical_note"] + "\n")
    write_text_with_retries(phantom_dir / "density_assignments.json", json.dumps(DENSITY_G_CM3, indent=2))

    fig_path = args.run_root / "figure1_detailed_headneck_phantom.png"
    plot_detailed_phantom(fig_path, phantom, dpi=int(args.dpi))
    density_fig_path = args.run_root / "figure2_detailed_headneck_density_map.png"
    plot_density_maps(density_fig_path, phantom, dpi=int(args.dpi))

    print(fig_path)
    print(density_fig_path)
    print(phantom_dir / "detailed_headneck_structures.npz")
    print(phantom_dir / "detailed_headneck_density_map.npz")
    print(phantom_dir / "detailed_headneck_summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
