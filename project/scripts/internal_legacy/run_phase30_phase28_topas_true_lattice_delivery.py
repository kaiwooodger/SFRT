#!/usr/bin/env python3
"""Phase 30: build a true TOPAS lattice delivery on the Phase 28 Yang phantom.

This workflow moves the Phase 28 benchmark from an analytical dose surrogate to
Monte Carlo beam transport through the voxelized TsImageCube phantom.

It does this in two TOPAS components:

1. broad coplanar arc-sampled base fields covering CTVboost (GTV + 5 mm)
2. smaller coplanar vertex-focused fields aimed at the two lattice spheres

Because TOPAS dose is linear in source fluence, the script calibrates the
clinically meaningful combined dose offline by solving for the relative base and
spot weights that best match:

- peripheral target coverage (CTVboost/PTV D95 ~= 3.5 Gy)
- mean vertex dose ~= 15 Gy

The resulting calibrated physical dose grid can then be passed to the biology
pipeline as a more meaningful physical input than the earlier single-field
validation run.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
from scipy import ndimage

from build_asymmetric_sweep import (
    PHYSICS_PROFILES,
    build_topas_env,
    format_physics_modules,
    has_nonempty_output,
    write_text_with_retries,
)


@dataclass(frozen=True)
class SourceSpec:
    name: str
    center_mm: Tuple[float, float, float]
    rotation_deg: Tuple[float, float, float]
    cutoff_mm: Tuple[float, float]
    histories: int


def voxel_size_mm_from_axes(axes_mm: Mapping[str, np.ndarray]) -> Tuple[float, float, float]:
    return (
        float(axes_mm["x"][1] - axes_mm["x"][0]),
        float(axes_mm["y"][1] - axes_mm["y"][0]),
        float(axes_mm["z"][1] - axes_mm["z"][0]),
    )


def voxel_volume_cc_from_axes(axes_mm: Mapping[str, np.ndarray]) -> float:
    dx, dy, dz = voxel_size_mm_from_axes(axes_mm)
    return float(dx * dy * dz / 1000.0)


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


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase28-topas-root",
        type=Path,
        default=root / "runs" / "phase28_yang2022_topas_tsimagecube",
        help="Phase 28 TsImageCube export root containing patient_material_tags.bin and materials.txt.",
    )
    parser.add_argument(
        "--template",
        type=Path,
        default=root / "topas" / "headneck_detailed_material_phantom_template.txt",
        help="TOPAS TsImageCube template.",
    )
    parser.add_argument(
        "--spectrum-csv",
        type=Path,
        default=root / "data" / "linac_6mv_representative_spectrum.csv",
        help="Representative 6 MV spectrum as energy_mev,weight CSV.",
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        default=root / "runs" / "phase30_phase28_topas_true_lattice_delivery",
        help="Output root for the Monte Carlo lattice delivery.",
    )
    parser.add_argument("--topas-bin", type=str, default="/Users/kw/shellScripts/topas")
    parser.add_argument("--g4-data-dir", type=str, default="/Applications/GEANT4")
    parser.add_argument("--physics-profile", choices=sorted(PHYSICS_PROFILES), default="em_opt4_only")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=33)
    parser.add_argument("--histories-base", type=int, default=1_000_000, help="TOPAS histories for the base-field component.")
    parser.add_argument("--histories-spot", type=int, default=2_000_000, help="TOPAS histories for the vertex-field component.")
    parser.add_argument("--sad-mm", type=float, default=260.0, help="Source-to-axis distance surrogate for coplanar delivery.")
    parser.add_argument(
        "--base-angles-deg",
        type=float,
        nargs="+",
        default=[0.0, 30.0, 60.0, 90.0, 120.0, 150.0, 180.0, 210.0, 240.0, 270.0, 300.0, 330.0],
        help="Coplanar angle samples for the broad-field component.",
    )
    parser.add_argument(
        "--spot-angles-deg",
        type=float,
        nargs="+",
        default=[0.0, 60.0, 120.0, 180.0, 240.0, 300.0],
        help="Coplanar angle samples for the vertex-focused component.",
    )
    parser.add_argument("--base-margin-mm", type=float, default=4.0, help="Margin added to projected broad-field apertures.")
    parser.add_argument("--spot-margin-mm", type=float, default=1.0, help="Margin added to projected spot apertures.")
    parser.add_argument("--vertex-radius-mm", type=float, default=5.0, help="Vertex radius used for aperture shaping.")
    parser.add_argument("--target-ptv-d95-gy", type=float, default=3.5, help="Coverage target used for final calibration.")
    parser.add_argument("--target-peak-mean-gy", type=float, default=15.0, help="Vertex-mean dose target used for final calibration.")
    parser.add_argument("--ratio-min", type=float, default=0.01, help="Minimum spot-to-base weight ratio searched during calibration.")
    parser.add_argument("--ratio-max", type=float, default=100.0, help="Maximum spot-to-base weight ratio searched during calibration.")
    parser.add_argument("--ratio-count", type=int, default=121, help="Number of searched spot-to-base ratios.")
    parser.add_argument(
        "--dose-smoothing-mm",
        type=float,
        default=6.0,
        help=(
            "Gaussian denoising sigma in mm applied inside the body mask before "
            "calibration and endpoint scoring. Set to 0 to disable."
        ),
    )
    parser.add_argument("--cut-gamma-mm", type=float, default=0.01)
    parser.add_argument("--cut-electron-mm", type=float, default=0.01)
    parser.add_argument("--cut-positron-mm", type=float, default=0.01)
    parser.add_argument("--skip-existing", action="store_true", help="Skip TOPAS component runs if scorer CSVs already exist.")
    parser.add_argument("--analyze-only", action="store_true", help="Skip TOPAS and only recompute calibration from existing component CSVs.")
    return parser.parse_args()


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
    brain = phase28_ellipsoid_mask(xg, yg, zg, center=(0.0, 5.0, 71.0), radii=(48.0, 40.0, 34.0)) & body
    return {
        "BODY": body,
        "GTV": gtv,
        "BRAIN": brain,
        "BRAINSTEM": phase28_ellipsoid_mask(xg, yg, zg, center=(0.0, -3.0, 78.0), radii=(9.0, 14.0, 12.0)) & body,
        "CHIASM": phase28_ellipsoid_mask(xg, yg, zg, center=(2.0, 16.0, 55.0), radii=(8.0, 4.0, 4.0)) & body,
        "OPTIC_NERVE_R": phase28_capsule_mask(xg, yg, zg, (35.0, 19.0, 18.0), (6.0, 16.0, 53.0), 2.2) & body,
        "OPTIC_NERVE_L": phase28_capsule_mask(xg, yg, zg, (-30.0, 19.0, 18.0), (-2.0, 16.0, 53.0), 2.2) & body,
        "EYE_R": phase28_ellipsoid_mask(xg, yg, zg, center=(38.0, 20.0, 16.0), radii=(12.0, 10.0, 8.0)) & body,
        "EYE_L": phase28_ellipsoid_mask(xg, yg, zg, center=(-32.0, 20.0, 16.0), radii=(12.0, 10.0, 8.0)) & body,
        "LENS_R": phase28_ellipsoid_mask(xg, yg, zg, center=(38.0, 20.0, 7.5), radii=(3.0, 3.0, 2.0)) & body,
        "LENS_L": phase28_ellipsoid_mask(xg, yg, zg, center=(-32.0, 20.0, 7.5), radii=(3.0, 3.0, 2.0)) & body,
    }


def build_phase28_ptv_mask(gtv_mask: np.ndarray, body_mask: np.ndarray, axes_mm: Mapping[str, np.ndarray], margin_mm: float) -> np.ndarray:
    distance_to_gtv = ndimage.distance_transform_edt(~np.asarray(gtv_mask, dtype=bool), sampling=voxel_size_mm_from_axes(axes_mm))
    return np.asarray(body_mask, dtype=bool) & ((distance_to_gtv <= float(margin_mm)) | np.asarray(gtv_mask, dtype=bool))


def phase28_vertex_centers() -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
    center_x = 20.0
    half_spacing_mm = 30.4 / 2.0
    return ((center_x - half_spacing_mm, 0.0, 38.7), (center_x + half_spacing_mm, 0.0, 38.7))


def build_phase28_vertex_mask(axes_mm: Mapping[str, np.ndarray], radius_mm: float) -> np.ndarray:
    xg, yg, zg = build_phase28_mesh(axes_mm)
    mask = np.zeros((axes_mm["x"].size, axes_mm["y"].size, axes_mm["z"].size), dtype=bool)
    for center in phase28_vertex_centers():
        mask |= phase28_sphere_mask(xg, yg, zg, center, float(radius_mm))
    return mask


def build_valley_mask(target_mask: np.ndarray, peak_mask: np.ndarray, axes_mm: Mapping[str, np.ndarray], exclusion_mm: float) -> np.ndarray:
    distance_to_peak = ndimage.distance_transform_edt(~np.asarray(peak_mask, dtype=bool), sampling=voxel_size_mm_from_axes(axes_mm))
    valley_mask = np.asarray(target_mask, dtype=bool) & (distance_to_peak > float(exclusion_mm))
    if int(np.count_nonzero(valley_mask)) > 0:
        return valley_mask
    valley_mask = np.asarray(target_mask, dtype=bool) & ~np.asarray(peak_mask, dtype=bool)
    if int(np.count_nonzero(valley_mask)) > 0:
        return valley_mask
    return np.asarray(target_mask, dtype=bool)


def write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        raise ValueError(f"No rows for {path}")
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


def load_spectrum(spectrum_csv: Path) -> Tuple[List[float], List[float]]:
    energies: List[float] = []
    weights: List[float] = []
    with spectrum_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            energies.append(float(row["energy_mev"]))
            weights.append(float(row["weight"]))
    if not energies:
        raise ValueError(f"No spectrum rows found in {spectrum_csv}")
    total = sum(weights)
    normalized = [value / total for value in weights]
    return energies, normalized


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


def load_topas_csv_grid(path: Path) -> np.ndarray:
    xbins = ybins = zbins = None
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.startswith("#"):
                break
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
    grid = np.zeros((int(xbins), int(ybins), int(zbins)), dtype=np.float32)
    grid[data[:, 0].astype(int), data[:, 1].astype(int), data[:, 2].astype(int)] = data[:, 3]
    return grid


def write_dose_csv(path: Path, dose_grid: np.ndarray) -> None:
    nx, ny, nz = dose_grid.shape
    with path.open("w", encoding="utf-8", newline="") as handle:
        handle.write("# Derived combined lattice dose grid\n")
        handle.write(f"# X in {nx} bins\n")
        handle.write(f"# Y in {ny} bins\n")
        handle.write(f"# Z in {nz} bins\n")
        handle.write("# DoseToWater ( Gy ) : CombinedScaled\n")
        for ix in range(nx):
            for iy in range(ny):
                for iz in range(nz):
                    handle.write(f"{ix}, {iy}, {iz}, {float(dose_grid[ix, iy, iz]):.10g}\n")


def smooth_dose_grid(
    dose_grid: np.ndarray,
    *,
    axes_mm: Mapping[str, np.ndarray],
    body_mask: np.ndarray,
    sigma_mm: float,
) -> np.ndarray:
    if float(sigma_mm) <= 0.0:
        return np.asarray(dose_grid, dtype=np.float32)
    voxel_mm = voxel_size_mm_from_axes(axes_mm)
    sigma_vox = tuple(float(sigma_mm) / max(float(step), 1.0e-6) for step in voxel_mm)
    body_float = np.asarray(body_mask, dtype=np.float32)
    numerator = ndimage.gaussian_filter(
        np.asarray(dose_grid, dtype=np.float32) * body_float,
        sigma=sigma_vox,
        mode="nearest",
    )
    denominator = ndimage.gaussian_filter(body_float, sigma=sigma_vox, mode="nearest")
    smoothed = np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator, dtype=np.float32),
        where=denominator > 1.0e-6,
    )
    smoothed *= body_float
    return smoothed.astype(np.float32, copy=False)


def render_source_block(sources: Sequence[SourceSpec], spectrum_energies: Sequence[float], spectrum_weights: Sequence[float]) -> str:
    lines: List[str] = []
    count = len(spectrum_energies)
    values = " ".join(f"{float(v):.6f}" for v in spectrum_energies)
    weights = " ".join(f"{float(v):.8f}" for v in spectrum_weights)
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
                f"dv:So/{source_name}/BeamEnergySpectrumValues = {count} {values} MeV",
                f"uv:So/{source_name}/BeamEnergySpectrumWeights = {count} {weights}",
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


def render_case_file(
    args: argparse.Namespace,
    *,
    patient_input_dir: Path,
    patient_input_file: str,
    materials_file: Path,
    grid_shape: Sequence[int],
    voxel_size_mm: Sequence[float],
    spectrum_energies: Sequence[float],
    spectrum_weights: Sequence[float],
    sources: Sequence[SourceSpec],
) -> str:
    template_text = Path(args.template).read_text(encoding="utf-8")
    input_dir = str(patient_input_dir.resolve())
    if not input_dir.endswith("/"):
        input_dir += "/"
    world_hlx_cm = max(20.0, float(grid_shape[0]) * float(voxel_size_mm[0]) / 20.0 + 5.0)
    world_hly_cm = max(20.0, float(grid_shape[1]) * float(voxel_size_mm[1]) / 20.0 + 5.0)
    world_hlz_cm = max(20.0, float(grid_shape[2]) * float(voxel_size_mm[2]) / 20.0 + 5.0)
    used_tags = [0, 10, 11, 13, 20, 21, 50]
    material_names = [
        "PH28_AIR",
        "PH28_SOFT_TISSUE",
        "PH28_BRAIN",
        "PH28_BRAINSTEM",
        "PH28_SKULL_BONE",
        "PH28_MAXILLOFACIAL_BONE",
        "PH28_TUMOUR",
    ]
    replacements = {
        "__G4_DATA_DIR__": str(args.g4_data_dir),
        "__PHYSICS_MODULES__": format_physics_modules(args.physics_profile),
        "__CUT_GAMMA_MM__": f"{float(args.cut_gamma_mm):.6f}",
        "__CUT_ELECTRON_MM__": f"{float(args.cut_electron_mm):.6f}",
        "__CUT_POSITRON_MM__": f"{float(args.cut_positron_mm):.6f}",
        "__MATERIALS_INCLUDE_FILE__": materials_file.name,
        "__WORLD_HLX_CM__": f"{world_hlx_cm:.6f}",
        "__WORLD_HLY_CM__": f"{world_hly_cm:.6f}",
        "__WORLD_HLZ_CM__": f"{world_hlz_cm:.6f}",
        "__PATIENT_INPUT_DIR__": input_dir,
        "__PATIENT_INPUT_FILE__": patient_input_file,
        "__XBINS__": str(int(grid_shape[0])),
        "__YBINS__": str(int(grid_shape[1])),
        "__ZBINS__": str(int(grid_shape[2])),
        "__VOXEL_SIZE_X_MM__": f"{float(voxel_size_mm[0]):.6f}",
        "__VOXEL_SIZE_Y_MM__": f"{float(voxel_size_mm[1]):.6f}",
        "__VOXEL_SIZE_Z_MM__": f"{float(voxel_size_mm[2]):.6f}",
        "__MATERIAL_TAG_COUNT__": str(len(used_tags)),
        "__MATERIAL_TAG_VALUES__": " ".join(str(tag) for tag in used_tags),
        "__MATERIAL_NAME_VALUES__": " ".join(f'"{name}"' for name in material_names),
        "__OUTPUT_STEM__": "dosedata",
        "__SOURCE_BLOCK__": render_source_block(sources, spectrum_energies, spectrum_weights),
        "__N_THREADS__": str(int(args.threads)),
        "__SEED__": str(int(args.seed)),
        "__SHOW_HISTORY_INTERVAL__": str(max(1, int(sum(spec.histories for spec in sources)) // 10)),
    }
    rendered = template_text
    for key, value in replacements.items():
        rendered = rendered.replace(key, value)
    return rendered


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
        tail = "\n".join(combined_log.strip().splitlines()[-80:])
        raise RuntimeError(
            "TOPAS component run failed for Phase 30.\n"
            f"Return code: {result.returncode}\n"
            f"Dose CSV present: {has_nonempty_output(dose_csv)}\n"
            f"Recent log:\n{tail}"
        )


def mask_centroid_mm(mask: np.ndarray, axes_mm: Mapping[str, np.ndarray]) -> Tuple[float, float, float]:
    coords = np.argwhere(mask)
    centroid_idx = np.round(coords.mean(axis=0)).astype(int)
    return (
        float(axes_mm["x"][centroid_idx[0]]),
        float(axes_mm["y"][centroid_idx[1]]),
        float(axes_mm["z"][centroid_idx[2]]),
    )


def beam_basis(theta_deg: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    theta = math.radians(float(theta_deg))
    beam_dir = np.asarray([math.sin(theta), 0.0, math.cos(theta)], dtype=np.float64)
    aperture_u = np.asarray([math.cos(theta), 0.0, -math.sin(theta)], dtype=np.float64)
    aperture_v = np.asarray([0.0, 1.0, 0.0], dtype=np.float64)
    return beam_dir, aperture_u, aperture_v


def mask_points_mm(mask: np.ndarray, axes_mm: Mapping[str, np.ndarray]) -> np.ndarray:
    idx = np.argwhere(mask)
    return np.column_stack(
        [
            np.asarray(axes_mm["x"], dtype=np.float32)[idx[:, 0]],
            np.asarray(axes_mm["y"], dtype=np.float32)[idx[:, 1]],
            np.asarray(axes_mm["z"], dtype=np.float32)[idx[:, 2]],
        ]
    ).astype(np.float64, copy=False)


def projected_aperture_radii(points_mm: np.ndarray, center_mm: Sequence[float], theta_deg: float, margin_mm: float) -> Tuple[float, float]:
    _, aperture_u, aperture_v = beam_basis(theta_deg)
    rel = np.asarray(points_mm, dtype=np.float64) - np.asarray(center_mm, dtype=np.float64)[None, :]
    u = np.abs(rel @ aperture_u)
    v = np.abs(rel @ aperture_v)
    return float(np.max(u) + float(margin_mm)), float(np.max(v) + float(margin_mm))


def source_center_for_angle(target_mm: Sequence[float], theta_deg: float, sad_mm: float) -> Tuple[float, float, float]:
    beam_dir, _, _ = beam_basis(theta_deg)
    target = np.asarray(target_mm, dtype=np.float64)
    source = target - float(sad_mm) * beam_dir
    return (float(source[0]), float(source[1]), float(source[2]))


def build_base_sources(
    args: argparse.Namespace,
    *,
    ptv_mask: np.ndarray,
    axes_mm: Mapping[str, np.ndarray],
) -> List[SourceSpec]:
    ptv_centroid = mask_centroid_mm(ptv_mask, axes_mm)
    points = mask_points_mm(ptv_mask, axes_mm)
    angles = [float(value) for value in args.base_angles_deg]
    histories = histories_from_weights(int(args.histories_base), [1.0] * len(angles))
    sources: List[SourceSpec] = []
    for angle, hist in zip(angles, histories):
        rad_u, rad_v = projected_aperture_radii(points, ptv_centroid, angle, float(args.base_margin_mm))
        sources.append(
            SourceSpec(
                name=f"BASE_{int(round(angle)) % 360:03d}",
                center_mm=source_center_for_angle(ptv_centroid, angle, float(args.sad_mm)),
                rotation_deg=(0.0, float(angle), 0.0),
                cutoff_mm=(float(rad_u), float(rad_v)),
                histories=int(hist),
            )
        )
    return sources


def build_spot_sources(
    args: argparse.Namespace,
    *,
    axes_mm: Mapping[str, np.ndarray],
) -> List[SourceSpec]:
    xg, yg, zg = build_phase28_mesh(axes_mm)
    angles = [float(value) for value in args.spot_angles_deg]
    weights = [1.0] * (len(angles) * len(phase28_vertex_centers()))
    histories = histories_from_weights(int(args.histories_spot), weights)
    history_iter = iter(histories)
    sources: List[SourceSpec] = []
    for vertex_idx, center in enumerate(phase28_vertex_centers(), start=1):
        vertex_mask = phase28_sphere_mask(xg, yg, zg, center, float(args.vertex_radius_mm))
        points = mask_points_mm(vertex_mask, axes_mm)
        for angle in angles:
            rad_u, rad_v = projected_aperture_radii(points, center, angle, float(args.spot_margin_mm))
            sources.append(
                SourceSpec(
                    name=f"V{vertex_idx}_{int(round(angle)) % 360:03d}",
                    center_mm=source_center_for_angle(center, angle, float(args.sad_mm)),
                    rotation_deg=(0.0, float(angle), 0.0),
                    cutoff_mm=(float(rad_u), float(rad_v)),
                    histories=int(next(history_iter)),
                )
            )
    return sources


def prepare_case_dir(out_dir: Path, *, patient_binary: Path, materials_source: Path) -> Tuple[Path, Path, Path, Path]:
    case_dir = out_dir / "case"
    case_dir.mkdir(parents=True, exist_ok=True)
    materials_file = case_dir / "materials.txt"
    shutil.copyfile(materials_source, materials_file)
    parameter_file = case_dir / "beamline.txt"
    dose_csv = case_dir / "dosedata.csv"
    log_file = case_dir / "topas.log"
    return case_dir, materials_file, parameter_file, dose_csv, log_file


def compute_physical_summary(
    dose_grid: np.ndarray,
    *,
    structures: Mapping[str, np.ndarray],
    peak_mask: np.ndarray,
    valley_mask: np.ndarray,
    axes_mm: Mapping[str, np.ndarray],
    target_ptv_d95_gy: float,
) -> Dict[str, float]:
    voxel_volume_cc = voxel_volume_cc_from_axes(axes_mm)
    metrics = {
        name: compute_structure_metrics_local(
            dose_grid,
            np.asarray(mask, dtype=bool),
            prescription_gy=(float(target_ptv_d95_gy) if name in {"PTV", "GTV"} else None),
            voxel_volume_cc=float(voxel_volume_cc),
        )
        for name, mask in structures.items()
        if int(np.count_nonzero(mask)) > 0
    }
    pvdr_metrics = compute_peak_valley_metrics_local(dose_grid, peak_mask, valley_mask)
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
    return {
        "ptv_d95_gy": float(metrics["PTV"]["d95_gy"]),
        "gtv_d95_gy": float(metrics["GTV"]["d95_gy"]),
        "peak_mean_gy": float(pvdr_metrics["peak_mean_gy"]),
        "peak_p90_gy": float(pvdr_metrics["peak_p90_gy"]),
        "valley_mean_gy": float(pvdr_metrics["valley_mean_gy"]),
        "pvdr": float(pvdr_metrics["pvdr"]),
        "spill_shell_0_5_mean_gy": float(spill_metrics.get("SPILL_SHELL_0_5", {}).get("mean_gy", float("nan"))),
        "spill_shell_5_15_mean_gy": float(spill_metrics.get("SPILL_SHELL_5_15", {}).get("mean_gy", float("nan"))),
        "brainstem_d2_gy": float(metrics.get("BRAINSTEM", {}).get("d2_gy", float("nan"))),
        "brain_dmax_gy": float(metrics.get("BRAIN", {}).get("dmax_gy", float("nan"))),
        "optic_nerve_r_dmax_gy": float(metrics.get("OPTIC_NERVE_R", {}).get("dmax_gy", float("nan"))),
        "optic_nerve_l_dmax_gy": float(metrics.get("OPTIC_NERVE_L", {}).get("dmax_gy", float("nan"))),
        "eye_r_dmax_gy": float(metrics.get("EYE_R", {}).get("dmax_gy", float("nan"))),
        "eye_l_dmax_gy": float(metrics.get("EYE_L", {}).get("dmax_gy", float("nan"))),
    }


def search_component_ratio(
    *,
    base_dose: np.ndarray,
    spot_dose: np.ndarray,
    ptv_mask: np.ndarray,
    peak_mask: np.ndarray,
    args: argparse.Namespace,
) -> Dict[str, object]:
    ratios = np.concatenate(
        [
            np.asarray([0.0], dtype=np.float64),
            np.logspace(math.log10(float(args.ratio_min)), math.log10(float(args.ratio_max)), int(args.ratio_count)),
        ]
    )
    best: Dict[str, object] | None = None
    ptv_mask_bool = np.asarray(ptv_mask, dtype=bool)
    peak_mask_bool = np.asarray(peak_mask, dtype=bool)
    for ratio in ratios:
        combined = base_dose + float(ratio) * spot_dose
        ptv_d95 = float(np.percentile(combined[ptv_mask_bool], 5.0))
        if ptv_d95 <= 0.0:
            continue
        global_scale = float(args.target_ptv_d95_gy) / ptv_d95
        peak_mean = float(np.mean((global_scale * combined)[peak_mask_bool]))
        peak_error = abs(math.log(max(peak_mean, 1.0e-9) / float(args.target_peak_mean_gy)))
        objective = peak_error
        candidate = {
            "spot_to_base_ratio": float(ratio),
            "global_scale": float(global_scale),
            "predicted_ptv_d95_gy": float(global_scale * ptv_d95),
            "predicted_peak_mean_gy": float(peak_mean),
            "objective": float(objective),
        }
        if best is None or float(candidate["objective"]) < float(best["objective"]):
            best = candidate
    if best is None:
        raise RuntimeError("Could not find a valid base/spot combination during calibration.")
    return best


def write_quick_assessment(path: Path, *, summary: Mapping[str, object]) -> None:
    final_metrics = dict(summary["final_metrics"])
    calibration = dict(summary["calibration"])
    lines = [
        "# Phase 30 Quick Assessment",
        "",
        "This is a true TOPAS Monte Carlo lattice-delivery surrogate on the Phase 28 phantom.",
        "The physical dose was built by combining a broad arc-sampled base-field component",
        "with a vertex-focused component and then calibrating their relative weights to the",
        "Yang-style 3.5 Gy peripheral / 15 Gy vertex prescription.",
        "",
        f"- Base component histories: `{summary['histories_base']}`",
        f"- Spot component histories: `{summary['histories_spot']}`",
        f"- Dose smoothing sigma: `{float(summary['dose_smoothing_mm']):.2f} mm`",
        f"- Spot-to-base weight ratio: `{float(calibration['spot_to_base_ratio']):.4f}`",
        f"- Global fluence scale: `{float(calibration['global_scale']):.6g}`",
        "",
        "## Final physical metrics",
        "",
        f"- PTV D95: `{float(final_metrics['ptv_d95_gy']):.3f} Gy`",
        f"- GTV D95: `{float(final_metrics['gtv_d95_gy']):.3f} Gy`",
        f"- Peak mean: `{float(final_metrics['peak_mean_gy']):.3f} Gy`",
        f"- Valley mean: `{float(final_metrics['valley_mean_gy']):.3f} Gy`",
        f"- PVDR: `{float(final_metrics['pvdr']):.3f}`",
        f"- Peri-GTV 0-5 mm mean: `{float(final_metrics['spill_shell_0_5_mean_gy']):.3f} Gy`",
        f"- Peri-GTV 5-15 mm mean: `{float(final_metrics['spill_shell_5_15_mean_gy']):.3f} Gy`",
        "",
        "## Interpretation",
        "",
        "- This is physically more meaningful than the earlier single-field validation run because the dose now comes from actual TOPAS transport of both broad target coverage fields and vertex-directed fields through the full voxel phantom.",
        "- A body-aware Gaussian denoising step is applied before calibration so the final dose reflects the expected Monte Carlo field rather than finite-history per-voxel hit noise.",
        "- It is still not a clinical VMAT replication. It is a coplanar arc-sampled photon lattice-delivery surrogate calibrated against the Yang-style benchmark prescription.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    args.out_root.mkdir(parents=True, exist_ok=True)

    phase28_root = Path(args.phase28_topas_root).expanduser().resolve()
    phase28_summary = json.loads((phase28_root / "phase28_topas_tsimagecube_summary.json").read_text(encoding="utf-8"))
    voxel_mm = float(phase28_summary["voxel_mm"])
    phase28_case_dir = phase28_root / "case"
    patient_binary = phase28_case_dir / "patient_material_tags.bin"
    materials_source = phase28_case_dir / "materials.txt"
    if not patient_binary.exists() or not materials_source.exists():
        raise FileNotFoundError("Phase 28 TsImageCube root is missing patient_material_tags.bin or materials.txt.")

    axes_mm = make_phase28_axes(voxel_mm)
    structures = build_phase28_structures(axes_mm)
    structures["PTV"] = build_phase28_ptv_mask(structures["GTV"], structures["BODY"], axes_mm, margin_mm=5.0)
    peak_mask = build_phase28_vertex_mask(axes_mm, radius_mm=float(args.vertex_radius_mm)) & np.asarray(structures["PTV"], dtype=bool)
    valley_mask = build_valley_mask(structures["PTV"], peak_mask, axes_mm, exclusion_mm=14.0)

    base_sources = build_base_sources(args, ptv_mask=structures["PTV"], axes_mm=axes_mm)
    spot_sources = build_spot_sources(args, axes_mm=axes_mm)
    spectrum_energies, spectrum_weights = load_spectrum(Path(args.spectrum_csv))

    component_specs = [
        ("base_component", base_sources),
        ("spot_component", spot_sources),
    ]
    component_outputs: Dict[str, Dict[str, object]] = {}
    for component_name, sources in component_specs:
        component_root = args.out_root / component_name
        case_dir, materials_file, parameter_file, dose_csv, log_file = prepare_case_dir(
            component_root,
            patient_binary=patient_binary,
            materials_source=materials_source,
        )
        case_text = render_case_file(
            args,
            patient_input_dir=patient_binary.parent,
            patient_input_file=patient_binary.name,
            materials_file=materials_file,
            grid_shape=tuple(int(v) for v in phase28_summary["grid_shape"]),
            voxel_size_mm=(voxel_mm, voxel_mm, voxel_mm),
            spectrum_energies=spectrum_energies,
            spectrum_weights=spectrum_weights,
            sources=sources,
        )
        write_text_with_retries(parameter_file, case_text)
        if not args.analyze_only and not (args.skip_existing and has_nonempty_output(dose_csv)):
            run_topas_case(args, case_dir, parameter_file, dose_csv, log_file)
        if not dose_csv.exists():
            raise FileNotFoundError(f"Expected TOPAS output missing for {component_name}: {dose_csv}")
        component_outputs[component_name] = {
            "root": str(component_root),
            "dose_csv": str(dose_csv),
            "log_file": str(log_file),
            "num_sources": int(len(sources)),
            "sources": [
                {
                    "name": spec.name,
                    "center_mm": list(map(float, spec.center_mm)),
                    "rotation_deg": list(map(float, spec.rotation_deg)),
                    "cutoff_mm": list(map(float, spec.cutoff_mm)),
                    "histories": int(spec.histories),
                }
                for spec in sources
            ],
        }

    base_dose = load_topas_csv_grid(Path(component_outputs["base_component"]["dose_csv"]))
    spot_dose = load_topas_csv_grid(Path(component_outputs["spot_component"]["dose_csv"]))
    body_mask = np.asarray(structures["BODY"], dtype=bool)
    base_processed = smooth_dose_grid(
        base_dose,
        axes_mm=axes_mm,
        body_mask=body_mask,
        sigma_mm=float(args.dose_smoothing_mm),
    )
    spot_processed = smooth_dose_grid(
        spot_dose,
        axes_mm=axes_mm,
        body_mask=body_mask,
        sigma_mm=float(args.dose_smoothing_mm),
    )

    calibration = search_component_ratio(
        base_dose=base_processed,
        spot_dose=spot_processed,
        ptv_mask=structures["PTV"],
        peak_mask=peak_mask,
        args=args,
    )
    combined_processed = base_processed + float(calibration["spot_to_base_ratio"]) * spot_processed
    combined_scaled = float(calibration["global_scale"]) * combined_processed
    scaled_base = float(calibration["global_scale"]) * base_processed
    scaled_spot = float(calibration["global_scale"]) * float(calibration["spot_to_base_ratio"]) * spot_processed

    np.savez_compressed(
        args.out_root / "phase30_component_doses.npz",
        base_raw=base_dose.astype(np.float32),
        spot_raw=spot_dose.astype(np.float32),
        base_processed=base_processed.astype(np.float32),
        spot_processed=spot_processed.astype(np.float32),
        base_scaled=scaled_base.astype(np.float32),
        spot_scaled=scaled_spot.astype(np.float32),
        combined_scaled=combined_scaled.astype(np.float32),
    )
    np.savez_compressed(
        args.out_root / "phase30_combined_physical_dose.npz",
        dose_gy=combined_scaled.astype(np.float32),
    )
    write_dose_csv(args.out_root / "phase30_combined_physical_dose.csv", combined_scaled.astype(np.float32))

    endpoint_rows = []
    for label, dose_grid in (
        ("base_processed_scaled", scaled_base),
        ("spot_processed_scaled", scaled_spot),
        ("combined_processed_scaled", combined_scaled),
    ):
        row = {"component": label}
        row.update(
            compute_physical_summary(
                np.asarray(dose_grid, dtype=np.float32),
                structures=structures,
                peak_mask=peak_mask,
                valley_mask=valley_mask,
                axes_mm=axes_mm,
                target_ptv_d95_gy=float(args.target_ptv_d95_gy),
            )
        )
        endpoint_rows.append(row)
    write_csv(args.out_root / "phase30_physical_endpoint_table.csv", endpoint_rows)

    final_metrics = endpoint_rows[-1]
    plan_summary = {
        "description": "Phase 30 true TOPAS lattice delivery on the Phase 28 Yang phantom",
        "phase28_topas_root": str(phase28_root),
        "voxel_mm": voxel_mm,
        "histories_base": int(args.histories_base),
        "histories_spot": int(args.histories_spot),
        "dose_smoothing_mm": float(args.dose_smoothing_mm),
        "base_angles_deg": [float(v) for v in args.base_angles_deg],
        "spot_angles_deg": [float(v) for v in args.spot_angles_deg],
        "target_ptv_d95_gy": float(args.target_ptv_d95_gy),
        "target_peak_mean_gy": float(args.target_peak_mean_gy),
        "calibration": calibration,
        "final_metrics": final_metrics,
        "component_outputs": component_outputs,
        "paths": {
            "component_doses_npz": str(args.out_root / "phase30_component_doses.npz"),
            "combined_physical_dose_npz": str(args.out_root / "phase30_combined_physical_dose.npz"),
            "combined_physical_dose_csv": str(args.out_root / "phase30_combined_physical_dose.csv"),
            "physical_endpoint_table": str(args.out_root / "phase30_physical_endpoint_table.csv"),
        },
    }
    write_json(args.out_root / "phase30_plan_summary.json", plan_summary)
    write_quick_assessment(args.out_root / "phase30_quick_assessment.md", summary=plan_summary)

    print("=== PHASE 30 TRUE TOPAS LATTICE DELIVERY COMPLETE ===")
    print(f"Output root: {args.out_root}")
    print(f"Combined physical dose NPZ: {args.out_root / 'phase30_combined_physical_dose.npz'}")
    print(f"Combined physical dose CSV: {args.out_root / 'phase30_combined_physical_dose.csv'}")
    print(f"Physical endpoint table: {args.out_root / 'phase30_physical_endpoint_table.csv'}")
    print(f"Plan summary: {args.out_root / 'phase30_plan_summary.json'}")
    print(f"Quick assessment: {args.out_root / 'phase30_quick_assessment.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
