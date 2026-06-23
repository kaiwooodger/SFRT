#!/usr/bin/env python3
"""Phase 32: instantiate the 10 benchmark templates as site-specific synthetic phantoms.

This phase fixes the main limitation of the Phase 31 template library: the
templates are no longer only descriptive geometry rows. Each template is turned
into a voxelized head-and-neck phantom with:

1. site-specific tumour centroid and shape
2. shared but spatially explicit H&N anatomy / OAR atlas
3. template-specific lattice vertices mapped into the instantiated tumour
4. vertex validation against the realised GTV core and nearby OAR clearance
5. TOPAS-ready material tags and per-case geometry exports
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage

from generate_detailed_headneck_topas_phantom import MATERIAL_SPECS, render_materials_include
from run_phase31_publication_package import BENCHMARK_ASSUMPTIONS, BENCHMARK_TEMPLATES


COORDINATE_NOTE = (
    "Coordinate system follows the synthetic benchmark convention: "
    "x = left (+) / right (-), y = anterior (+) / posterior (-), "
    "z = superior (+) / inferior (-)."
)

STRUCTURE_TAG_MAP = {
    "BODY": 10,
    "BRAIN": 11,
    "CHIASM": 11,
    "OPTIC_NERVE_R": 11,
    "OPTIC_NERVE_L": 11,
    "BRAINSTEM": 13,
    "SPINAL_CORD": 14,
    "SKULL": 20,
    "MAXILLA": 21,
    "MANDIBLE": 21,
    "VERTEBRAE": 22,
    "PAROTID_R": 30,
    "PAROTID_L": 30,
    "THYROID": 32,
    "PARATHYROID_R": 33,
    "PARATHYROID_L": 33,
    "ARTERIES": 40,
    "VEINS": 41,
    "GTV": 50,
}

CRITICAL_OAR_NAMES = (
    "ARTERIES",
    "VEINS",
    "SPINAL_CORD",
    "BRAINSTEM",
    "CHIASM",
    "OPTIC_NERVE_R",
    "OPTIC_NERVE_L",
    "EYE_R",
    "EYE_L",
    "LENS_R",
    "LENS_L",
    "COCHLEA_R",
    "COCHLEA_L",
    "BRACHIAL_PLEXUS_R",
    "BRACHIAL_PLEXUS_L",
    "MANDIBLE",
    "MAXILLA",
    "SKULL",
    "SKIN",
)

SITE_SPECS = {
    "case01": {
        "centroid_mm": (-24.0, 18.0, 18.0),
        "site_group": "sinonasal_midface",
        "site_note": "Right midface / maxillary compartment near orbit and skull base.",
    },
    "case02": {
        "centroid_mm": (22.0, 18.0, 18.0),
        "site_group": "sinonasal_midface",
        "site_note": "Left maxillary crescent wrapping toward the orbit.",
    },
    "case03": {
        "centroid_mm": (0.0, -6.0, -4.0),
        "site_group": "oropharynx_bot",
        "site_note": "Bulky midline BOT / oropharyngeal deep-core mass.",
    },
    "case04": {
        "centroid_mm": (0.0, -8.0, -30.0),
        "site_group": "laryngohypopharynx",
        "site_note": "Elongated laryngo-hypopharyngeal target along the cranio-caudal axis.",
    },
    "case05": {
        "centroid_mm": (-14.0, -8.0, 2.0),
        "site_group": "parapharyngeal_deep_space",
        "site_note": "Deep-space parapharyngeal / prestyloid mass close to carotid space.",
    },
    "case06": {
        "centroid_mm": (-36.0, 24.0, -6.0),
        "site_group": "cheek_superficial",
        "site_note": "Superficial buccal / cheek disease with skin and mandibular proximity.",
    },
    "case07": {
        "centroid_mm": (0.0, 16.0, -16.0),
        "site_group": "oral_tongue_floor_mouth",
        "site_note": "Oral tongue / floor-of-mouth horseshoe geometry in the anterior oral cavity.",
    },
    "case08": {
        "centroid_mm": (-28.0, -6.0, -26.0),
        "site_group": "nodal_neck",
        "site_note": "Lateral neck nodal conglomerate spanning upper and mid neck levels.",
    },
    "case09": {
        "centroid_mm": (-38.0, -4.0, 10.0),
        "site_group": "deep_parotid_infratemporal",
        "site_note": "Deep parotid / infratemporal mass with skull-base adjacency.",
    },
    "case10": {
        "centroid_mm": (-4.0, -4.0, -12.0),
        "site_group": "composite_oropharynx_upper_neck",
        "site_note": "Very bulky composite oropharynx + upper-neck target extending across compartments.",
    },
}


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-root",
        type=Path,
        default=root / "runs" / "phase32_site_specific_template_phantoms",
    )
    parser.add_argument("--voxel-mm", type=float, default=2.0)
    parser.add_argument("--gtv-margin-mm", type=float, default=5.0, help="CTVboost margin.")
    parser.add_argument("--vertex-oar-clearance-mm", type=float, default=10.0)
    parser.add_argument("--vertex-search-step-mm", type=float, default=2.0)
    parser.add_argument("--vertex-search-extent-mm", type=float, default=8.0)
    parser.add_argument("--min-center-spacing-mm", type=float, default=16.0)
    parser.add_argument("--dpi", type=int, default=220)
    return parser.parse_args()


def write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        raise ValueError(f"No rows for {path}")
    fieldnames: List[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(str(key))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_markdown(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def write_image_cube(tag_grid_xyz: np.ndarray, out_file: Path) -> None:
    np.asarray(tag_grid_xyz, dtype=np.int16).transpose(2, 1, 0).tofile(out_file)


def make_axes(voxel_mm: float) -> Dict[str, np.ndarray]:
    return {
        "x": np.arange(-90.0, 90.0 + 0.5 * voxel_mm, voxel_mm, dtype=np.float32),
        "y": np.arange(-86.0, 86.0 + 0.5 * voxel_mm, voxel_mm, dtype=np.float32),
        "z": np.arange(-74.0, 90.0 + 0.5 * voxel_mm, voxel_mm, dtype=np.float32),
    }


def mesh(axes: Mapping[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    return np.meshgrid(axes["x"], axes["y"], axes["z"], indexing="ij")


def voxel_spacing_mm(axes: Mapping[str, np.ndarray]) -> Tuple[float, float, float]:
    return tuple(float(values[1] - values[0]) if len(values) > 1 else 1.0 for values in (axes["x"], axes["y"], axes["z"]))


def voxel_volume_cc(axes: Mapping[str, np.ndarray]) -> float:
    dx, dy, dz = voxel_spacing_mm(axes)
    return float(dx * dy * dz / 1000.0)


def ellipsoid_mask(
    xg: np.ndarray,
    yg: np.ndarray,
    zg: np.ndarray,
    *,
    center: Tuple[float, float, float],
    radii: Tuple[float, float, float],
) -> np.ndarray:
    cx, cy, cz = center
    rx, ry, rz = radii
    return (((xg - cx) / max(rx, 1.0e-6)) ** 2 + ((yg - cy) / max(ry, 1.0e-6)) ** 2 + ((zg - cz) / max(rz, 1.0e-6)) ** 2) <= 1.0


def sphere_mask(
    xg: np.ndarray,
    yg: np.ndarray,
    zg: np.ndarray,
    center: Tuple[float, float, float],
    radius_mm: float,
) -> np.ndarray:
    cx, cy, cz = center
    return ((xg - cx) ** 2 + (yg - cy) ** 2 + (zg - cz) ** 2) <= float(radius_mm) ** 2


def capsule_mask(
    xg: np.ndarray,
    yg: np.ndarray,
    zg: np.ndarray,
    p0: Tuple[float, float, float],
    p1: Tuple[float, float, float],
    radius_mm: float,
) -> np.ndarray:
    p0_arr = np.asarray(p0, dtype=np.float32)
    p1_arr = np.asarray(p1, dtype=np.float32)
    v = p1_arr - p0_arr
    vv = float(np.dot(v, v))
    px = xg - float(p0_arr[0])
    py = yg - float(p0_arr[1])
    pz = zg - float(p0_arr[2])
    t = np.clip((px * v[0] + py * v[1] + pz * v[2]) / max(vv, 1.0e-6), 0.0, 1.0)
    cx = float(p0_arr[0]) + t * v[0]
    cy = float(p0_arr[1]) + t * v[1]
    cz = float(p0_arr[2]) + t * v[2]
    return ((xg - cx) ** 2 + (yg - cy) ** 2 + (zg - cz) ** 2) <= float(radius_mm) ** 2


def shell_mask(mask: np.ndarray, sampling: Tuple[float, float, float], width_mm: float) -> np.ndarray:
    return np.asarray(mask, dtype=bool) & (ndimage.distance_transform_edt(mask, sampling=sampling) <= float(width_mm))


def build_base_anatomy(axes: Mapping[str, np.ndarray]) -> Dict[str, np.ndarray]:
    xg, yg, zg = mesh(axes)
    sampling = voxel_spacing_mm(axes)

    cranium_outer = ellipsoid_mask(xg, yg, zg, center=(0.0, 0.0, 44.0), radii=(58.0, 58.0, 38.0))
    cranium_inner = ellipsoid_mask(xg, yg, zg, center=(0.0, -2.0, 45.0), radii=(52.0, 51.0, 33.0))
    midface_outer = ellipsoid_mask(xg, yg, zg, center=(0.0, 28.0, 10.0), radii=(60.0, 34.0, 26.0))
    midface_inner = ellipsoid_mask(xg, yg, zg, center=(0.0, 26.0, 10.0), radii=(52.0, 26.0, 20.0))
    jaw_outer = ellipsoid_mask(xg, yg, zg, center=(0.0, 18.0, -18.0), radii=(40.0, 24.0, 15.0))
    jaw_inner = ellipsoid_mask(xg, yg, zg, center=(0.0, 16.0, -16.0), radii=(33.0, 18.0, 10.0))
    neck = ellipsoid_mask(xg, yg, zg, center=(0.0, -2.0, -22.0), radii=(44.0, 32.0, 46.0))
    shoulders = ellipsoid_mask(xg, yg, zg, center=(0.0, -8.0, -58.0), radii=(68.0, 28.0, 14.0))
    nose = capsule_mask(xg, yg, zg, (0.0, 30.0, 12.0), (0.0, 48.0, 8.0), 7.0)
    cheek_l = ellipsoid_mask(xg, yg, zg, center=(34.0, 24.0, 2.0), radii=(12.0, 10.0, 12.0))
    cheek_r = ellipsoid_mask(xg, yg, zg, center=(-34.0, 24.0, 2.0), radii=(12.0, 10.0, 12.0))
    ear_l = ellipsoid_mask(xg, yg, zg, center=(58.0, -2.0, 10.0), radii=(5.0, 4.0, 10.0))
    ear_r = ellipsoid_mask(xg, yg, zg, center=(-58.0, -2.0, 10.0), radii=(5.0, 4.0, 10.0))

    body = cranium_outer | midface_outer | jaw_outer | neck | shoulders | nose | cheek_l | cheek_r | ear_l | ear_r
    skull = ((cranium_outer & ~cranium_inner) | (midface_outer & ~midface_inner)) & body
    maxilla = (midface_outer & ~midface_inner) | ellipsoid_mask(xg, yg, zg, center=(0.0, 26.0, -2.0), radii=(32.0, 16.0, 8.0))
    mandible = (jaw_outer & ~jaw_inner) & body
    brain = ellipsoid_mask(xg, yg, zg, center=(0.0, -4.0, 46.0), radii=(48.0, 48.0, 30.0)) & cranium_inner
    brainstem = ellipsoid_mask(xg, yg, zg, center=(0.0, -12.0, 16.0), radii=(10.0, 13.0, 12.0)) & body
    spinal_cord = capsule_mask(xg, yg, zg, (0.0, -18.0, 18.0), (0.0, -18.0, -68.0), 4.5) & body
    vertebrae = (capsule_mask(xg, yg, zg, (0.0, -18.0, 20.0), (0.0, -18.0, -68.0), 11.0) & body) & ~spinal_cord

    eye_l = ellipsoid_mask(xg, yg, zg, center=(31.0, 34.0, 16.0), radii=(12.0, 10.0, 8.0)) & body
    eye_r = ellipsoid_mask(xg, yg, zg, center=(-31.0, 34.0, 16.0), radii=(12.0, 10.0, 8.0)) & body
    lens_l = ellipsoid_mask(xg, yg, zg, center=(31.0, 37.0, 15.0), radii=(3.0, 3.0, 2.0)) & body
    lens_r = ellipsoid_mask(xg, yg, zg, center=(-31.0, 37.0, 15.0), radii=(3.0, 3.0, 2.0)) & body
    chiasm = ellipsoid_mask(xg, yg, zg, center=(0.0, 22.0, 24.0), radii=(6.0, 5.0, 4.0)) & body
    optic_l = capsule_mask(xg, yg, zg, (29.0, 30.0, 16.0), (4.0, 22.0, 24.0), 2.2) & body
    optic_r = capsule_mask(xg, yg, zg, (-29.0, 30.0, 16.0), (-4.0, 22.0, 24.0), 2.2) & body
    cochlea_l = ellipsoid_mask(xg, yg, zg, center=(42.0, -4.0, 12.0), radii=(4.0, 4.0, 4.0)) & body
    cochlea_r = ellipsoid_mask(xg, yg, zg, center=(-42.0, -4.0, 12.0), radii=(4.0, 4.0, 4.0)) & body

    oral_cavity = ellipsoid_mask(xg, yg, zg, center=(0.0, 18.0, -10.0), radii=(28.0, 16.0, 10.0)) & body
    airway = capsule_mask(xg, yg, zg, (0.0, 8.0, 8.0), (0.0, 4.0, -18.0), 6.5) & body
    trachea = capsule_mask(xg, yg, zg, (0.0, 2.0, -18.0), (0.0, 0.0, -72.0), 6.0) & body

    parotid_l = ellipsoid_mask(xg, yg, zg, center=(42.0, -2.0, -12.0), radii=(12.0, 16.0, 11.0)) & body
    parotid_r = ellipsoid_mask(xg, yg, zg, center=(-42.0, -2.0, -12.0), radii=(12.0, 16.0, 11.0)) & body
    thyroid_l = ellipsoid_mask(xg, yg, zg, center=(10.0, 6.0, -48.0), radii=(9.0, 7.0, 9.0)) & body
    thyroid_r = ellipsoid_mask(xg, yg, zg, center=(-10.0, 6.0, -48.0), radii=(9.0, 7.0, 9.0)) & body
    parathyroid_l = ellipsoid_mask(xg, yg, zg, center=(12.0, 2.0, -48.0), radii=(2.8, 2.0, 2.4)) & body
    parathyroid_r = ellipsoid_mask(xg, yg, zg, center=(-12.0, 2.0, -48.0), radii=(2.8, 2.0, 2.4)) & body

    artery_l = capsule_mask(xg, yg, zg, (18.0, -8.0, 26.0), (18.0, -10.0, -68.0), 3.4) & body
    artery_r = capsule_mask(xg, yg, zg, (-18.0, -8.0, 26.0), (-18.0, -10.0, -68.0), 3.4) & body
    vein_l = capsule_mask(xg, yg, zg, (27.0, -10.0, 20.0), (27.0, -12.0, -68.0), 4.1) & body
    vein_r = capsule_mask(xg, yg, zg, (-27.0, -10.0, 20.0), (-27.0, -12.0, -68.0), 4.1) & body
    brachial_l = capsule_mask(xg, yg, zg, (34.0, -12.0, -30.0), (32.0, -14.0, -62.0), 3.3) & body
    brachial_r = capsule_mask(xg, yg, zg, (-34.0, -12.0, -30.0), (-32.0, -14.0, -62.0), 3.3) & body

    skin = shell_mask(body, sampling, width_mm=4.0)

    return {
        "BODY": body,
        "SKULL": skull,
        "MAXILLA": maxilla,
        "MANDIBLE": mandible,
        "VERTEBRAE": vertebrae,
        "BRAIN": brain,
        "BRAINSTEM": brainstem,
        "SPINAL_CORD": spinal_cord,
        "CHIASM": chiasm,
        "OPTIC_NERVE_L": optic_l,
        "OPTIC_NERVE_R": optic_r,
        "EYE_L": eye_l,
        "EYE_R": eye_r,
        "LENS_L": lens_l,
        "LENS_R": lens_r,
        "COCHLEA_L": cochlea_l,
        "COCHLEA_R": cochlea_r,
        "PAROTID_L": parotid_l,
        "PAROTID_R": parotid_r,
        "THYROID": thyroid_l | thyroid_r,
        "PARATHYROID_L": parathyroid_l,
        "PARATHYROID_R": parathyroid_r,
        "ORAL_CAVITY": oral_cavity,
        "AIRWAY": airway,
        "TRACHEA": trachea,
        "ARTERIES": artery_l | artery_r,
        "VEINS": vein_l | vein_r,
        "BRACHIAL_PLEXUS_L": brachial_l,
        "BRACHIAL_PLEXUS_R": brachial_r,
        "SKIN": skin,
    }


def radii_from_dimensions_cm(dimensions_cm: Sequence[float]) -> Tuple[float, float, float]:
    return tuple(float(v) * 5.0 for v in dimensions_cm)  # type: ignore[return-value]


def elliptical_cylinder_mask(
    xg: np.ndarray,
    yg: np.ndarray,
    zg: np.ndarray,
    *,
    center: Tuple[float, float, float],
    radii_xy: Tuple[float, float],
    half_height_mm: float,
) -> np.ndarray:
    cx, cy, cz = center
    rx, ry = radii_xy
    core = (((xg - cx) / max(rx, 1.0e-6)) ** 2 + ((yg - cy) / max(ry, 1.0e-6)) ** 2) <= 1.0
    slab = np.abs(zg - cz) <= float(half_height_mm)
    cap_top = ellipsoid_mask(xg, yg, zg, center=(cx, cy, cz + half_height_mm), radii=(rx, ry, min(rx, ry)))
    cap_bottom = ellipsoid_mask(xg, yg, zg, center=(cx, cy, cz - half_height_mm), radii=(rx, ry, min(rx, ry)))
    return (core & slab) | cap_top | cap_bottom


def build_case_tumour(
    template_id: str,
    center_mm: Tuple[float, float, float],
    dimensions_cm: Sequence[float],
    axes: Mapping[str, np.ndarray],
    body: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    xg, yg, zg = mesh(axes)
    rx, ry, rz = radii_from_dimensions_cm(dimensions_cm)
    cx, cy, cz = center_mm
    necrosis = np.zeros_like(body, dtype=bool)

    if template_id == "case01":
        gtv = ellipsoid_mask(xg, yg, zg, center=center_mm, radii=(rx, ry, rz))
    elif template_id == "case02":
        outer = ellipsoid_mask(xg, yg, zg, center=center_mm, radii=(rx, ry, rz))
        inner = ellipsoid_mask(
            xg,
            yg,
            zg,
            center=(cx + 0.32 * rx, cy + 0.18 * ry, cz + 0.10 * rz),
            radii=(0.72 * rx, 0.68 * ry, 0.72 * rz),
        )
        posterior_core = ellipsoid_mask(
            xg,
            yg,
            zg,
            center=(cx - 0.15 * rx, cy - 0.20 * ry, cz - 0.08 * rz),
            radii=(0.70 * rx, 0.60 * ry, 0.60 * rz),
        )
        gtv = (outer & ~inner) | posterior_core
    elif template_id == "case03":
        lobe_l = ellipsoid_mask(xg, yg, zg, center=(cx + 0.28 * rx, cy, cz + 0.10 * rz), radii=(0.55 * rx, 0.58 * ry, 0.52 * rz))
        lobe_r = ellipsoid_mask(xg, yg, zg, center=(cx - 0.28 * rx, cy, cz + 0.10 * rz), radii=(0.55 * rx, 0.58 * ry, 0.52 * rz))
        inferior_core = ellipsoid_mask(xg, yg, zg, center=(cx, cy - 0.15 * ry, cz - 0.20 * rz), radii=(0.58 * rx, 0.54 * ry, 0.52 * rz))
        gtv = lobe_l | lobe_r | inferior_core
    elif template_id == "case04":
        gtv = elliptical_cylinder_mask(xg, yg, zg, center=center_mm, radii_xy=(rx, ry), half_height_mm=0.5 * float(dimensions_cm[0]) * 5.0)
    elif template_id == "case05":
        main = ellipsoid_mask(xg, yg, zg, center=center_mm, radii=(rx, ry, rz))
        superior = ellipsoid_mask(xg, yg, zg, center=(cx - 0.05 * rx, cy - 0.10 * ry, cz + 0.18 * rz), radii=(0.58 * rx, 0.44 * ry, 0.50 * rz))
        inferior = ellipsoid_mask(xg, yg, zg, center=(cx + 0.08 * rx, cy + 0.12 * ry, cz - 0.16 * rz), radii=(0.42 * rx, 0.36 * ry, 0.44 * rz))
        gtv = main | superior | inferior
    elif template_id == "case06":
        outer = ellipsoid_mask(xg, yg, zg, center=center_mm, radii=(rx, ry, rz))
        inner = ellipsoid_mask(xg, yg, zg, center=(cx + 0.12 * rx, cy - 0.22 * ry, cz), radii=(0.72 * rx, 0.58 * ry, 0.70 * rz))
        gtv = outer & ~inner
    elif template_id == "case07":
        wing_l = ellipsoid_mask(xg, yg, zg, center=(cx + 0.30 * rx, cy, cz + 0.10 * rz), radii=(0.46 * rx, 0.54 * ry, 0.42 * rz))
        wing_r = ellipsoid_mask(xg, yg, zg, center=(cx - 0.30 * rx, cy, cz + 0.10 * rz), radii=(0.46 * rx, 0.54 * ry, 0.42 * rz))
        posterior_bridge = ellipsoid_mask(xg, yg, zg, center=(cx, cy - 0.12 * ry, cz - 0.24 * rz), radii=(0.58 * rx, 0.44 * ry, 0.44 * rz))
        anterior_notch = ellipsoid_mask(xg, yg, zg, center=(cx, cy + 0.22 * ry, cz), radii=(0.28 * rx, 0.28 * ry, 0.30 * rz))
        gtv = (wing_l | wing_r | posterior_bridge) & ~anterior_notch
    elif template_id == "case08":
        mass_a = ellipsoid_mask(xg, yg, zg, center=(cx - 0.12 * rx, cy, cz + 0.18 * rz), radii=(0.52 * rx, 0.48 * ry, 0.40 * rz))
        mass_b = ellipsoid_mask(xg, yg, zg, center=(cx + 0.18 * rx, cy + 0.10 * ry, cz + 0.10 * rz), radii=(0.46 * rx, 0.42 * ry, 0.36 * rz))
        mass_c = ellipsoid_mask(xg, yg, zg, center=(cx - 0.05 * rx, cy - 0.18 * ry, cz - 0.18 * rz), radii=(0.42 * rx, 0.40 * ry, 0.34 * rz))
        mass_d = ellipsoid_mask(xg, yg, zg, center=(cx + 0.18 * rx, cy - 0.08 * ry, cz - 0.16 * rz), radii=(0.36 * rx, 0.34 * ry, 0.30 * rz))
        gross = mass_a | mass_b | mass_c | mass_d
        necrosis = ellipsoid_mask(xg, yg, zg, center=(cx, cy, cz), radii=(0.28 * rx, 0.26 * ry, 0.24 * rz))
        gtv = gross & ~necrosis
    elif template_id == "case09":
        main = ellipsoid_mask(xg, yg, zg, center=center_mm, radii=(rx, ry, rz))
        superior_lobe = ellipsoid_mask(xg, yg, zg, center=(cx, cy - 0.10 * ry, cz + 0.22 * rz), radii=(0.55 * rx, 0.45 * ry, 0.46 * rz))
        posterior_lobe = ellipsoid_mask(xg, yg, zg, center=(cx + 0.06 * rx, cy - 0.18 * ry, cz - 0.08 * rz), radii=(0.44 * rx, 0.36 * ry, 0.38 * rz))
        gtv = main | superior_lobe | posterior_lobe
    elif template_id == "case10":
        oropharynx = ellipsoid_mask(xg, yg, zg, center=(cx, cy, cz + 0.10 * rz), radii=(0.58 * rx, 0.52 * ry, 0.48 * rz))
        upper_neck = ellipsoid_mask(xg, yg, zg, center=(cx, cy - 0.06 * ry, cz - 0.22 * rz), radii=(0.50 * rx, 0.42 * ry, 0.44 * rz))
        left_lobe = ellipsoid_mask(xg, yg, zg, center=(cx + 0.22 * rx, cy, cz + 0.18 * rz), radii=(0.34 * rx, 0.30 * ry, 0.30 * rz))
        right_lobe = ellipsoid_mask(xg, yg, zg, center=(cx - 0.22 * rx, cy, cz + 0.18 * rz), radii=(0.34 * rx, 0.30 * ry, 0.30 * rz))
        gtv = oropharynx | upper_neck | left_lobe | right_lobe
    else:
        raise KeyError(f"Unsupported template: {template_id}")

    return np.asarray(gtv, dtype=bool) & np.asarray(body, dtype=bool), np.asarray(necrosis, dtype=bool) & np.asarray(body, dtype=bool)


def expand_mask(mask: np.ndarray, *, sampling: Tuple[float, float, float], margin_mm: float, limit_mask: np.ndarray) -> np.ndarray:
    distance = ndimage.distance_transform_edt(~mask, sampling=sampling)
    return np.asarray(mask, dtype=bool) | ((distance <= float(margin_mm)) & np.asarray(limit_mask, dtype=bool))


def structure_volume_cc(mask: np.ndarray, axes: Mapping[str, np.ndarray]) -> float:
    return float(np.count_nonzero(np.asarray(mask, dtype=bool)) * voxel_volume_cc(axes))


def match_volume_to_target(
    mask: np.ndarray,
    *,
    target_cc: float,
    axes: Mapping[str, np.ndarray],
    limit_mask: np.ndarray,
    tolerance_fraction: float = 0.03,
    max_iterations: int = 40,
) -> np.ndarray:
    current = np.asarray(mask, dtype=bool).copy()
    allowed = np.asarray(limit_mask, dtype=bool)
    target = float(target_cc)
    voxel_cc = voxel_volume_cc(axes)
    structure = ndimage.generate_binary_structure(3, 1)

    def current_cc(arr: np.ndarray) -> float:
        return float(np.count_nonzero(arr) * voxel_cc)

    value = current_cc(current)
    lower = target * (1.0 - tolerance_fraction)
    upper = target * (1.0 + tolerance_fraction)
    if lower <= value <= upper:
        return current

    if value < lower:
        for _ in range(int(max_iterations)):
            expanded = ndimage.binary_dilation(current, structure=structure) & allowed
            if np.array_equal(expanded, current):
                break
            current = np.asarray(expanded, dtype=bool)
            value = current_cc(current)
            if value >= lower:
                break
    else:
        for _ in range(int(max_iterations)):
            eroded = ndimage.binary_erosion(current, structure=structure)
            if not np.any(eroded):
                break
            current = np.asarray(eroded, dtype=bool)
            value = current_cc(current)
            if value <= upper:
                break
    return current


def nearest_index(axis: np.ndarray, value: float) -> int:
    idx = int(np.argmin(np.abs(axis - float(value))))
    return max(0, min(idx, len(axis) - 1))


def generate_shift_vectors(step_mm: float, extent_mm: float) -> List[Tuple[float, float, float]]:
    values = np.arange(-float(extent_mm), float(extent_mm) + 0.5 * float(step_mm), float(step_mm), dtype=np.float32)
    shifts = [(float(x), float(y), float(z)) for x in values for y in values for z in values]
    shifts.sort(key=lambda item: (item[0] ** 2 + item[1] ** 2 + item[2] ** 2, abs(item[0]) + abs(item[1]) + abs(item[2])))
    return shifts


def build_critical_oar_mask(structures: Mapping[str, np.ndarray]) -> np.ndarray:
    mask = np.zeros_like(np.asarray(structures["BODY"], dtype=bool))
    for name in CRITICAL_OAR_NAMES:
        if name in structures:
            mask |= np.asarray(structures[name], dtype=bool)
    return mask


def map_template_vertices_mm(template: Mapping[str, object], center_mm: Tuple[float, float, float]) -> List[Tuple[float, float, float]]:
    vertices: List[Tuple[float, float, float]] = []
    for rel_x_cm, rel_y_cm, rel_z_cm in template["vertex_centres_cm"]:  # type: ignore[index]
        vertices.append(
            (
                float(center_mm[0] + 10.0 * float(rel_x_cm)),
                float(center_mm[1] + 10.0 * float(rel_y_cm)),
                float(center_mm[2] + 10.0 * float(rel_z_cm)),
            )
        )
    return vertices


def validate_and_adjust_vertices(
    *,
    template: Mapping[str, object],
    axes: Mapping[str, np.ndarray],
    structures: Mapping[str, np.ndarray],
    center_mm: Tuple[float, float, float],
    clearance_mm: float,
    min_spacing_mm: float,
    search_step_mm: float,
    search_extent_mm: float,
) -> Tuple[List[Tuple[float, float, float]], List[Dict[str, object]], np.ndarray]:
    spacing = voxel_spacing_mm(axes)
    gtv = np.asarray(structures.get("VERTEX_TARGET", structures["GTV"]), dtype=bool)
    critical_oars = build_critical_oar_mask(structures) & ~gtv
    inside_distance = ndimage.distance_transform_edt(gtv, sampling=spacing)
    oar_distance = ndimage.distance_transform_edt(~critical_oars, sampling=spacing)

    radius_mm = 5.0 * float(template["vertex_diameter_cm_recommended"])
    proposed = map_template_vertices_mm(template, center_mm)
    shifts = generate_shift_vectors(search_step_mm, search_extent_mm)

    accepted: List[Tuple[float, float, float]] = []
    details: List[Dict[str, object]] = []
    peak_mask = np.zeros_like(gtv, dtype=bool)

    for idx, proposed_center in enumerate(proposed, start=1):
        best: Tuple[float, float, float] | None = None
        best_shift = None
        best_score = None
        rejection_reason = "no_valid_shift_found"
        for shift in shifts:
            candidate = (
                float(proposed_center[0] + shift[0]),
                float(proposed_center[1] + shift[1]),
                float(proposed_center[2] + shift[2]),
            )
            ix = nearest_index(np.asarray(axes["x"]), candidate[0])
            iy = nearest_index(np.asarray(axes["y"]), candidate[1])
            iz = nearest_index(np.asarray(axes["z"]), candidate[2])
            if not bool(gtv[ix, iy, iz]):
                rejection_reason = "candidate_center_outside_gtv"
                continue
            if float(inside_distance[ix, iy, iz]) < float(radius_mm):
                rejection_reason = "insufficient_gtv_edge_margin"
                continue
            if float(oar_distance[ix, iy, iz]) < float(radius_mm + clearance_mm):
                rejection_reason = "insufficient_oar_clearance"
                continue
            if any(math.dist(candidate, other) < float(min_spacing_mm) for other in accepted):
                rejection_reason = "insufficient_center_spacing"
                continue
            score = (
                float(shift[0] ** 2 + shift[1] ** 2 + shift[2] ** 2),
                -float(inside_distance[ix, iy, iz]),
                -float(oar_distance[ix, iy, iz]),
            )
            if best_score is None or score < best_score:
                best_score = score
                best = candidate
                best_shift = shift
        if best is not None:
            accepted.append(best)
            peak_mask |= sphere_mask(*mesh(axes), best, radius_mm) & gtv
            ix = nearest_index(np.asarray(axes["x"]), best[0])
            iy = nearest_index(np.asarray(axes["y"]), best[1])
            iz = nearest_index(np.asarray(axes["z"]), best[2])
            details.append(
                {
                    "vertex_id": f"V{idx}",
                    "status": "kept",
                    "proposed_center_mm": [round(v, 3) for v in proposed_center],
                    "adjusted_center_mm": [round(v, 3) for v in best],
                    "shift_mm": [round(float(v), 3) for v in best_shift] if best_shift is not None else [0.0, 0.0, 0.0],
                    "edge_margin_mm": round(float(inside_distance[ix, iy, iz]) - float(radius_mm), 3),
                    "critical_oar_clearance_mm": round(float(oar_distance[ix, iy, iz]) - float(radius_mm), 3),
                }
            )
        else:
            details.append(
                {
                    "vertex_id": f"V{idx}",
                    "status": "pruned",
                    "proposed_center_mm": [round(v, 3) for v in proposed_center],
                    "adjusted_center_mm": None,
                    "shift_mm": None,
                    "reason": rejection_reason,
                }
            )

    return accepted, details, peak_mask


def build_material_tag_grid(structures: Mapping[str, np.ndarray]) -> np.ndarray:
    grid = np.zeros_like(np.asarray(structures["BODY"], dtype=bool), dtype=np.int16)
    grid[np.asarray(structures["BODY"], dtype=bool)] = 10
    for name, tag in STRUCTURE_TAG_MAP.items():
        if name not in structures:
            continue
        grid[np.asarray(structures[name], dtype=bool)] = int(tag)
    grid[np.asarray(structures["AIRWAY"], dtype=bool) | np.asarray(structures["TRACHEA"], dtype=bool)] = 0
    return grid


def min_distance_between_masks_mm(mask_a: np.ndarray, mask_b: np.ndarray, sampling: Tuple[float, float, float]) -> float:
    a = np.asarray(mask_a, dtype=bool)
    b = np.asarray(mask_b, dtype=bool)
    if not np.any(a) or not np.any(b):
        return float("nan")
    distance = ndimage.distance_transform_edt(~b, sampling=sampling)
    return float(np.min(distance[a]))


def select_nearby_oars(structures: Mapping[str, np.ndarray], *, sampling: Tuple[float, float, float], top_n: int = 5) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for name in CRITICAL_OAR_NAMES:
        if name not in structures:
            continue
        dist_mm = min_distance_between_masks_mm(structures["GTV"], structures[name], sampling)
        rows.append({"structure": name, "min_distance_mm": dist_mm})
    rows.sort(key=lambda row: float(row["min_distance_mm"]))
    return rows[:top_n]


def save_context_npz(path: Path, axes: Mapping[str, np.ndarray], structures: Mapping[str, np.ndarray], tag_grid: np.ndarray, vertices_mm: Sequence[Tuple[float, float, float]]) -> None:
    payload: Dict[str, np.ndarray] = {
        "axes_x_mm": np.asarray(axes["x"], dtype=np.float32),
        "axes_y_mm": np.asarray(axes["y"], dtype=np.float32),
        "axes_z_mm": np.asarray(axes["z"], dtype=np.float32),
        "material_tags": np.asarray(tag_grid, dtype=np.int16),
        "vertex_centers_mm": np.asarray(vertices_mm, dtype=np.float32),
    }
    for name, mask in structures.items():
        payload[f"struct_{name}"] = np.asarray(mask, dtype=bool)
    np.savez_compressed(path, **payload)


def build_case_markdown(summary: Mapping[str, object]) -> str:
    lines = [
        f"# {summary['template_id']}: {summary['label']}",
        "",
        f"- Site group: `{summary['site_group']}`",
        f"- Site note: {summary['site_note']}",
        f"- Tumour model: `{summary['tumour_model']}`",
        f"- Estimated GTV: `{float(summary['estimated_gtv_cc']):.1f} cc`",
        f"- Instantiated GTV: `{float(summary['instantiated_gtv_cc']):.1f} cc`",
        f"- Instantiated CTVboost: `{float(summary['ctvboost_cc']):.1f} cc`",
        f"- Proposed vertices: `{int(summary['proposed_vertex_count'])}`",
        f"- Kept vertices: `{int(summary['kept_vertex_count'])}`",
        f"- Coordinate note: {summary['coordinate_note']}",
        "",
        "## Nearby critical anatomy",
        "",
    ]
    for row in summary["nearby_oars"]:  # type: ignore[index]
        lines.append(f"- `{row['structure']}`: `{float(row['min_distance_mm']):.2f} mm` from GTV")
    lines.extend(["", "## Vertex validation", ""])
    for row in summary["vertex_details"]:  # type: ignore[index]
        if row["status"] == "kept":
            lines.append(
                f"- `{row['vertex_id']}` kept at `{row['adjusted_center_mm']}` mm "
                f"(edge margin `{row['edge_margin_mm']}` mm, critical OAR clearance `{row['critical_oar_clearance_mm']}` mm)."
            )
        else:
            lines.append(f"- `{row['vertex_id']}` pruned from `{row['proposed_center_mm']}` mm because `{row['reason']}`.")
    return "\n".join(lines) + "\n"


def plot_case_geometry(
    out_file: Path,
    *,
    axes: Mapping[str, np.ndarray],
    structures: Mapping[str, np.ndarray],
    vertices_mm: Sequence[Tuple[float, float, float]],
    label: str,
    nearby_oars: Sequence[str],
    dpi: int,
) -> None:
    x_cm = np.asarray(axes["x"], dtype=np.float32) / 10.0
    y_cm = np.asarray(axes["y"], dtype=np.float32) / 10.0
    z_cm = np.asarray(axes["z"], dtype=np.float32) / 10.0

    gtv = np.asarray(structures["GTV"], dtype=bool)
    coords = np.argwhere(gtv)
    center_index = np.asarray(np.round(np.mean(coords, axis=0)), dtype=int)
    ix, iy, iz = [int(v) for v in center_index.tolist()]

    fig, axes_plot = plt.subplots(1, 3, figsize=(15.0, 4.8), constrained_layout=True)
    panels = [
        (axes_plot[0], np.asarray(structures["BODY"], dtype=bool)[:, :, iz], gtv[:, :, iz], x_cm, y_cm, "Axial (x/y)", "x (cm)", "y (cm)"),
        (axes_plot[1], np.asarray(structures["BODY"], dtype=bool)[:, iy, :], gtv[:, iy, :], x_cm, z_cm, "Coronal (x/z)", "x (cm)", "z (cm)"),
        (axes_plot[2], np.asarray(structures["BODY"], dtype=bool)[ix, :, :], gtv[ix, :, :], y_cm, z_cm, "Sagittal (y/z)", "y (cm)", "z (cm)"),
    ]

    for ax, body_slice, gtv_slice, axis_a, axis_b, title, xlabel, ylabel in panels:
        ax.imshow(
            body_slice.T.astype(float),
            origin="lower",
            extent=[float(axis_a[0]), float(axis_a[-1]), float(axis_b[0]), float(axis_b[-1])],
            cmap="Greys",
            alpha=0.22,
        )
        ax.contour(axis_a, axis_b, gtv_slice.T.astype(float), levels=[0.5], colors=["#d62728"], linewidths=1.6)
        for name in nearby_oars:
            if name not in structures:
                continue
            if title.startswith("Axial"):
                oar_slice = np.asarray(structures[name], dtype=bool)[:, :, iz]
            elif title.startswith("Coronal"):
                oar_slice = np.asarray(structures[name], dtype=bool)[:, iy, :]
            else:
                oar_slice = np.asarray(structures[name], dtype=bool)[ix, :, :]
            if np.any(oar_slice):
                ax.contour(axis_a, axis_b, oar_slice.T.astype(float), levels=[0.5], linewidths=0.9)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.12)

    if vertices_mm:
        vx = np.asarray([pt[0] for pt in vertices_mm], dtype=np.float32) / 10.0
        vy = np.asarray([pt[1] for pt in vertices_mm], dtype=np.float32) / 10.0
        vz = np.asarray([pt[2] for pt in vertices_mm], dtype=np.float32) / 10.0
        axes_plot[0].scatter(vx, vy, s=54, c="#f0b400", edgecolors="#111111", linewidths=0.7, zorder=5)
        axes_plot[1].scatter(vx, vz, s=54, c="#f0b400", edgecolors="#111111", linewidths=0.7, zorder=5)
        axes_plot[2].scatter(vy, vz, s=54, c="#f0b400", edgecolors="#111111", linewidths=0.7, zorder=5)

    fig.suptitle(label, fontsize=13)
    fig.savefig(out_file, dpi=int(dpi))
    plt.close(fig)


def build_case_summary(
    *,
    template: Mapping[str, object],
    site_spec: Mapping[str, object],
    axes: Mapping[str, np.ndarray],
    structures: Mapping[str, np.ndarray],
    vertex_details: Sequence[Mapping[str, object]],
    kept_vertices: Sequence[Tuple[float, float, float]],
) -> Dict[str, object]:
    sampling = voxel_spacing_mm(axes)
    nearby_oars = select_nearby_oars(structures, sampling=sampling)
    return {
        "template_id": str(template["template_id"]),
        "label": str(template["label"]),
        "site_group": str(site_spec["site_group"]),
        "site_note": str(site_spec["site_note"]),
        "tumour_model": str(template["tumour_model"]),
        "estimated_gtv_cc": float(template["estimated_gtv_cc"]),
        "instantiated_gtv_cc": structure_volume_cc(structures["GTV"], axes),
        "ctvboost_cc": structure_volume_cc(structures["CTVBOOST"], axes),
        "proposed_vertex_count": int(len(template["vertex_centres_cm"])),  # type: ignore[arg-type]
        "kept_vertex_count": int(len(kept_vertices)),
        "coordinate_note": COORDINATE_NOTE,
        "site_centroid_mm": [float(v) for v in site_spec["centroid_mm"]],
        "vertex_diameter_cm_recommended": float(template["vertex_diameter_cm_recommended"]),
        "nearby_oars": nearby_oars,
        "vertex_details": list(vertex_details),
        "structure_volumes_cc": {
            name: round(structure_volume_cc(mask, axes), 4)
            for name, mask in structures.items()
            if np.any(np.asarray(mask, dtype=bool))
            and name
            in {
                "GTV",
                "CTVBOOST",
                "BRAINSTEM",
                "SPINAL_CORD",
                "PAROTID_L",
                "PAROTID_R",
                "THYROID",
                "ARTERIES",
                "VEINS",
                "MANDIBLE",
                "MAXILLA",
                "SKIN",
                "TUMOUR_NECROSIS",
            }
        },
    }


def instantiate_case(
    *,
    template: Mapping[str, object],
    axes: Mapping[str, np.ndarray],
    args: argparse.Namespace,
) -> Tuple[Dict[str, np.ndarray], Dict[str, object], np.ndarray, List[Tuple[float, float, float]]]:
    site_spec = SITE_SPECS[str(template["template_id"])]
    structures = {name: np.asarray(mask, dtype=bool).copy() for name, mask in build_base_anatomy(axes).items()}
    center_mm = tuple(float(v) for v in site_spec["centroid_mm"])
    gtv, necrosis = build_case_tumour(str(template["template_id"]), center_mm, template["dimensions_cm"], axes, structures["BODY"])
    growth_limit = np.asarray(structures["BODY"], dtype=bool) & ~np.asarray(structures["AIRWAY"], dtype=bool) & ~np.asarray(structures["TRACHEA"], dtype=bool)
    if np.any(gtv):
        gtv = match_volume_to_target(
            gtv,
            target_cc=float(template["estimated_gtv_cc"]),
            axes=axes,
            limit_mask=growth_limit,
        )

    sampling = voxel_spacing_mm(axes)
    ctvboost = expand_mask(gtv, sampling=sampling, margin_mm=float(args.gtv_margin_mm), limit_mask=structures["BODY"])
    structures["GTV"] = gtv
    structures["CTVBOOST"] = ctvboost
    structures["PTV"] = ctvboost
    if np.any(necrosis):
        necrosis = np.asarray(necrosis, dtype=bool) & np.asarray(gtv, dtype=bool)
        structures["TUMOUR_NECROSIS"] = necrosis
        structures["VERTEX_TARGET"] = np.asarray(gtv, dtype=bool) & ~necrosis
    else:
        structures["VERTEX_TARGET"] = np.asarray(gtv, dtype=bool)

    kept_vertices, vertex_details, peak_mask = validate_and_adjust_vertices(
        template=template,
        axes=axes,
        structures=structures,
        center_mm=center_mm,
        clearance_mm=float(args.vertex_oar_clearance_mm),
        min_spacing_mm=float(args.min_center_spacing_mm),
        search_step_mm=float(args.vertex_search_step_mm),
        search_extent_mm=float(args.vertex_search_extent_mm),
    )
    structures["VERTEX_UNION"] = peak_mask
    for idx, vertex in enumerate(kept_vertices, start=1):
        structures[f"VERTEX_{idx:02d}"] = sphere_mask(*mesh(axes), vertex, 5.0 * float(template["vertex_diameter_cm_recommended"])) & gtv

    tag_grid = build_material_tag_grid(structures)
    summary = build_case_summary(
        template=template,
        site_spec=site_spec,
        axes=axes,
        structures=structures,
        vertex_details=vertex_details,
        kept_vertices=kept_vertices,
    )
    return structures, summary, tag_grid, kept_vertices


def build_manifest_row(case_dir: Path, summary: Mapping[str, object], kept_vertices: Sequence[Tuple[float, float, float]]) -> Dict[str, object]:
    return {
        "template_id": str(summary["template_id"]),
        "label": str(summary["label"]),
        "site_group": str(summary["site_group"]),
        "estimated_gtv_cc": float(summary["estimated_gtv_cc"]),
        "instantiated_gtv_cc": float(summary["instantiated_gtv_cc"]),
        "proposed_vertex_count": int(summary["proposed_vertex_count"]),
        "kept_vertex_count": int(summary["kept_vertex_count"]),
        "site_centroid_mm": json.dumps(summary["site_centroid_mm"]),
        "kept_vertices_mm": json.dumps([[round(v, 3) for v in point] for point in kept_vertices]),
        "case_dir": str(case_dir),
        "summary_json": str(case_dir / "case_summary.json"),
        "summary_md": str(case_dir / "case_summary.md"),
        "phantom_context_npz": str(case_dir / "phantom_context.npz"),
        "material_tags_npz": str(case_dir / "material_tags.npz"),
        "patient_material_tags_bin": str(case_dir / "patient_material_tags.bin"),
        "materials_include": str(case_dir / "materials.txt"),
        "geometry_figure": str(case_dir / "figure_case_geometry.png"),
    }


def build_quick_assessment(manifest_rows: Sequence[Mapping[str, object]]) -> str:
    kept = [int(row["kept_vertex_count"]) for row in manifest_rows]
    instantiated = [float(row["instantiated_gtv_cc"]) for row in manifest_rows]
    return "\n".join(
        [
            "# Phase 32 quick assessment",
            "",
            "- The 10 Yang-like H&N templates are now instantiated as site-specific voxel phantoms rather than only being described as benchmark rows.",
            f"- Cases instantiated: `{len(manifest_rows)}`.",
            f"- Median instantiated GTV: `{float(np.median(np.asarray(instantiated, dtype=np.float64))):.1f} cc`.",
            f"- Total proposed vertices across the library: `{sum(int(row['proposed_vertex_count']) for row in manifest_rows)}`.",
            f"- Total kept vertices after anatomy-aware validation: `{sum(kept)}`.",
            f"- Cases with at least one pruned vertex: `{sum(1 for row in manifest_rows if int(row['kept_vertex_count']) < int(row['proposed_vertex_count']))}`.",
            "- This library can now be used as a true anatomy-varying synthetic cohort for downstream TOPAS transport or biology analyses.",
        ]
    ) + "\n"


def main() -> int:
    args = parse_args()
    out_root = args.out_root.resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    axes = make_axes(float(args.voxel_mm))
    materials_text = render_materials_include(MATERIAL_SPECS)

    manifest_rows: List[Dict[str, object]] = []
    library_rows: List[Dict[str, object]] = []

    for template in BENCHMARK_TEMPLATES:
        template_id = str(template["template_id"])
        case_dir = out_root / template_id
        case_dir.mkdir(parents=True, exist_ok=True)

        structures, summary, tag_grid, kept_vertices = instantiate_case(template=template, axes=axes, args=args)
        nearby_names = [str(row["structure"]) for row in summary["nearby_oars"]]

        save_context_npz(case_dir / "phantom_context.npz", axes, structures, tag_grid, kept_vertices)
        np.savez_compressed(case_dir / "material_tags.npz", material_tags=np.asarray(tag_grid, dtype=np.int16))
        write_image_cube(np.asarray(tag_grid, dtype=np.int16), case_dir / "patient_material_tags.bin")
        (case_dir / "materials.txt").write_text(materials_text, encoding="utf-8")
        write_json(case_dir / "case_summary.json", summary)
        write_markdown(case_dir / "case_summary.md", build_case_markdown(summary))
        plot_case_geometry(
            case_dir / "figure_case_geometry.png",
            axes=axes,
            structures=structures,
            vertices_mm=kept_vertices,
            label=f"{template_id}: {template['label']}",
            nearby_oars=nearby_names[:4],
            dpi=int(args.dpi),
        )

        manifest_rows.append(build_manifest_row(case_dir, summary, kept_vertices))
        library_rows.append(
            {
                "template_id": template_id,
                "label": str(template["label"]),
                "site_group": str(summary["site_group"]),
                "site_note": str(summary["site_note"]),
                "estimated_gtv_cc": float(summary["estimated_gtv_cc"]),
                "instantiated_gtv_cc": float(summary["instantiated_gtv_cc"]),
                "kept_vertex_count": int(summary["kept_vertex_count"]),
                "proposed_vertex_count": int(summary["proposed_vertex_count"]),
                "site_centroid_mm": json.dumps(summary["site_centroid_mm"]),
                "coordinate_note": COORDINATE_NOTE,
            }
        )

    write_csv(out_root / "phase32_case_manifest.csv", manifest_rows)
    write_json(out_root / "phase32_case_manifest.json", manifest_rows)
    write_csv(out_root / "phase32_site_specific_library.csv", library_rows)
    write_json(
        out_root / "phase32_site_specific_library.json",
        {
            "coordinate_note": COORDINATE_NOTE,
            "benchmark_assumptions": list(BENCHMARK_ASSUMPTIONS),
            "cases": library_rows,
        },
    )
    write_markdown(out_root / "phase32_quick_assessment.md", build_quick_assessment(manifest_rows))

    print("=== PHASE 32 SITE-SPECIFIC TEMPLATE PHANTOMS COMPLETE ===")
    print(f"Output root: {out_root}")
    print(f"Manifest: {out_root / 'phase32_case_manifest.csv'}")
    print(f"Quick assessment: {out_root / 'phase32_quick_assessment.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
