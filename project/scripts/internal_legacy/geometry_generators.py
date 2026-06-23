#!/usr/bin/env python
"""Reusable geometry generators for Phase 9 holdout studies."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Sequence, Tuple

import numpy as np

from analyze_250mev_sfrt_plan import (
    centered_axis_cm,
    choose_reference_depth,
    depth_axis_cm,
    reference_peak_value,
)
from analyze_topas_outputs import load_topas_grid
from bystander_multispecies_pde_solver import (
    build_cylindrical_uptake_tensor,
    build_vessel_network_uptake_tensor,
)
from generate_phase2_multispecies_vessel_valley_figures import build_lattice_with_x_shift


def normalize_single_beam(
    single_beam: np.ndarray,
    z_cm: np.ndarray,
    prescribed_peak_dose_gy: float,
) -> tuple[np.ndarray, Dict[str, float]]:
    ref_idx = choose_reference_depth(
        single_beam,
        z_cm,
        argparse.Namespace(reference_mode="dmax", reference_depth_cm=0.0),
    )
    ref_peak_raw, ref_peak_idx_xy = reference_peak_value(single_beam, ref_idx, center_window_bins=5)
    scale = float(prescribed_peak_dose_gy) / ref_peak_raw
    return single_beam * scale, {
        "reference_depth_cm": float(z_cm[ref_idx]),
        "reference_peak_raw_gy": float(ref_peak_raw),
        "scale_factor": float(scale),
        "reference_peak_index_x": int(ref_peak_idx_xy[0]),
        "reference_peak_index_y": int(ref_peak_idx_xy[1]),
    }


def generate_3d_lattice_geometry(
    *,
    pitch_mm: float,
    csv: Path | None = None,
    prescribed_peak_dose_gy: float = 10.0,
    uniform_dose_floor_fraction: float = 0.015,
    n_beams_x: int = 7,
    n_beams_y: int = 7,
    x_shift_fraction_of_pitch: float = 0.5,
    vessel_radius_mm: float = 1.5,
    vessel_uptake: float = 0.60,
    ros_vessel_uptake: float = 0.05,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, object]]:
    """Build a true-valley 3D lattice from the clinical TOPAS single-beam kernel.

    Returns:
    - dose_grid: lattice dose in project order `(x, y, z)`
    - uptake_tensor: species-specific vascular uptake field
    - oxygen_modifier: uniform oxygen state modifier tensor
    - type_modifier: uniform cell-type modifier tensor
    - metadata: geometry and axis information
    """

    root = Path(__file__).resolve().parents[1]
    csv_path = csv or (
        root / "runs" / "linac_6mv_polyenergetic_clinical_sfrt" / "case" / "dosedata.csv"
    )

    single_beam, header = load_topas_grid(csv_path, retries=5, retry_delay_sec=0.5)
    dx_cm = float(header["dx_cm"])
    dy_cm = float(header["dy_cm"])
    dz_cm = float(header["dz_cm"])
    z_single_cm = depth_axis_cm(single_beam.shape[2], dz_cm)
    normalized_single, normalization = normalize_single_beam(
        single_beam,
        z_single_cm,
        prescribed_peak_dose_gy=float(prescribed_peak_dose_gy),
    )
    normalized_single = normalized_single.astype(np.float32)

    pitch_bins_x = int(round((float(pitch_mm) / 10.0) / dx_cm))
    pitch_bins_y = int(round((float(pitch_mm) / 10.0) / dy_cm))
    x_shift_bins = int(round(float(x_shift_fraction_of_pitch) * float(pitch_bins_x)))

    lattice_dose, _, _, x_centers_idx = build_lattice_with_x_shift(
        normalized_single,
        pitch_bins_x=pitch_bins_x,
        pitch_bins_y=pitch_bins_y,
        n_beams_x=int(n_beams_x),
        n_beams_y=int(n_beams_y),
        x_shift_bins=int(x_shift_bins),
    )
    lattice_dose = lattice_dose.astype(np.float32)
    floor_gy = float(uniform_dose_floor_fraction) * float(prescribed_peak_dose_gy)
    if floor_gy > 0.0:
        lattice_dose = lattice_dose + np.float32(floor_gy)

    voxel_size_mm = (dx_cm * 10.0, dy_cm * 10.0, dz_cm * 10.0)
    uptake_tensor, vessel_mask = build_cylindrical_uptake_tensor(
        lattice_dose.shape,
        voxel_size_mm,
        num_species=2,
        vessel_radius_mm=float(vessel_radius_mm),
        vessel_center_offset_mm=(0.0, 0.0),
        uptake_rates_in_vessel=(float(ros_vessel_uptake), float(vessel_uptake)),
    )
    oxygen_modifier = np.ones((2, *lattice_dose.shape), dtype=np.float32)
    type_modifier = np.ones((2, *lattice_dose.shape), dtype=np.float32)

    metadata: Dict[str, object] = {
        "csv": str(csv_path),
        "pitch_mm": float(pitch_mm),
        "pitch_bins_x": int(pitch_bins_x),
        "pitch_bins_y": int(pitch_bins_y),
        "x_shift_bins": int(x_shift_bins),
        "x_shift_fraction_of_pitch": float(x_shift_fraction_of_pitch),
        "voxel_size_mm": [float(value) for value in voxel_size_mm],
        "dose_grid_shape": [int(value) for value in lattice_dose.shape],
        "x_cm": centered_axis_cm(lattice_dose.shape[0], dx_cm).tolist(),
        "z_cm": depth_axis_cm(lattice_dose.shape[2], dz_cm).tolist(),
        "x_centers_idx": [int(value) for value in x_centers_idx],
        "normalization": normalization,
        "uniform_dose_floor_gy": float(floor_gy),
        "vessel_radius_mm": float(vessel_radius_mm),
        "vessel_uptake": float(vessel_uptake),
        "ros_vessel_uptake": float(ros_vessel_uptake),
        "vessel_mask_centerline_count": int(np.count_nonzero(vessel_mask[:, lattice_dose.shape[1] // 2, 0])),
    }
    return lattice_dose, uptake_tensor, oxygen_modifier, type_modifier, metadata


def _spot_bounds_from_centers_and_kernel(
    centers_x_bins: Sequence[int],
    centers_y_bins: Sequence[int],
    kernel_shape: tuple[int, int, int],
    margin_bins: int = 4,
) -> tuple[int, int]:
    """Return odd lattice dimensions centered on x=y=0 that fit every spot."""

    kernel_left_x = kernel_shape[0] // 2
    kernel_left_y = kernel_shape[1] // 2
    kernel_right_x = kernel_shape[0] - kernel_left_x - 1
    kernel_right_y = kernel_shape[1] - kernel_left_y - 1

    max_extent_x = max(
        max(abs(int(center) - kernel_left_x), abs(int(center) + kernel_right_x))
        for center in centers_x_bins
    )
    max_extent_y = max(
        max(abs(int(center) - kernel_left_y), abs(int(center) + kernel_right_y))
        for center in centers_y_bins
    )

    half_span_x = int(max_extent_x) + int(margin_bins)
    half_span_y = int(max_extent_y) + int(margin_bins)
    return (2 * half_span_x) + 1, (2 * half_span_y) + 1


def build_weighted_spot_lattice(
    kernel: np.ndarray,
    *,
    spot_specs_mm: Sequence[dict],
    voxel_size_mm: Sequence[float],
    margin_bins: int = 4,
) -> tuple[np.ndarray, list[dict]]:
    """Place weighted beam kernels at arbitrary x/y offsets."""

    dx_mm, dy_mm, _ = (float(value) for value in voxel_size_mm)
    centers_x_bins = [int(round(float(spec["x_mm"]) / dx_mm)) for spec in spot_specs_mm]
    centers_y_bins = [int(round(float(spec["y_mm"]) / dy_mm)) for spec in spot_specs_mm]

    out_x, out_y = _spot_bounds_from_centers_and_kernel(
        centers_x_bins,
        centers_y_bins,
        kernel.shape,
        margin_bins=int(margin_bins),
    )
    lattice = np.zeros((int(out_x), int(out_y), int(kernel.shape[2])), dtype=np.float32)
    center_x = lattice.shape[0] // 2
    center_y = lattice.shape[1] // 2
    kernel_center_x = kernel.shape[0] // 2
    kernel_center_y = kernel.shape[1] // 2

    placed_specs: list[dict] = []
    for spec, off_x_bins, off_y_bins in zip(spot_specs_mm, centers_x_bins, centers_y_bins):
        beam_center_x = center_x + int(off_x_bins)
        beam_center_y = center_y + int(off_y_bins)
        start_x = beam_center_x - kernel_center_x
        end_x = start_x + kernel.shape[0]
        start_y = beam_center_y - kernel_center_y
        end_y = start_y + kernel.shape[1]
        weight = float(spec.get("weight", 1.0))
        lattice[start_x:end_x, start_y:end_y, :] += kernel * np.float32(weight)
        placed_specs.append(
            {
                "x_mm": float(spec["x_mm"]),
                "y_mm": float(spec["y_mm"]),
                "weight": float(weight),
                "label": str(spec.get("label", f"spot_{len(placed_specs)}")),
                "center_index_x": int(beam_center_x),
                "center_index_y": int(beam_center_y),
            }
        )

    return lattice, placed_specs


def default_complex_lattice_spot_specs_mm() -> list[dict]:
    """Irregular weighted spot arrangement approximating a plan-like lattice."""

    return [
        {"label": "apex_left", "x_mm": -28.0, "y_mm": 26.0, "weight": 0.94},
        {"label": "apex_mid", "x_mm": 0.0, "y_mm": 30.0, "weight": 0.88},
        {"label": "apex_right", "x_mm": 26.0, "y_mm": 22.0, "weight": 1.06},
        {"label": "mid_left", "x_mm": -32.0, "y_mm": 2.0, "weight": 1.00},
        {"label": "mid_core", "x_mm": -4.0, "y_mm": -4.0, "weight": 1.12},
        {"label": "mid_right", "x_mm": 24.0, "y_mm": 4.0, "weight": 0.92},
        {"label": "base_left", "x_mm": -22.0, "y_mm": -24.0, "weight": 1.04},
        {"label": "base_mid", "x_mm": 6.0, "y_mm": -30.0, "weight": 0.90},
        {"label": "base_right", "x_mm": 30.0, "y_mm": -18.0, "weight": 1.08},
        {"label": "posterior_tail", "x_mm": 0.0, "y_mm": 52.0, "weight": 0.82},
    ]


def default_complex_vessel_specs_mm() -> list[dict]:
    """Curved multi-vessel sink network in centered solver coordinates."""

    return [
        {
            "label": "central_drain",
            "radius_mm": 1.8,
            "uptake_rates_in_vessel": (0.06, 0.72),
            "nodes_mm": [(-2.0, -4.0, -90.0), (0.0, 0.0, -10.0), (1.5, 2.0, 70.0)],
        },
        {
            "label": "left_arc",
            "radius_mm": 1.3,
            "uptake_rates_in_vessel": (0.05, 0.60),
            "nodes_mm": [(-30.0, -12.0, -80.0), (-22.0, -4.0, -20.0), (-16.0, 8.0, 55.0)],
        },
        {
            "label": "right_arc",
            "radius_mm": 1.5,
            "uptake_rates_in_vessel": (0.05, 0.66),
            "nodes_mm": [(22.0, -18.0, -70.0), (18.0, -4.0, -10.0), (12.0, 16.0, 65.0)],
        },
        {
            "label": "superior_branch",
            "radius_mm": 1.1,
            "uptake_rates_in_vessel": (0.04, 0.48),
            "nodes_mm": [(-8.0, 24.0, -35.0), (0.0, 34.0, 10.0), (10.0, 44.0, 75.0)],
        },
    ]


def generate_complex_lattice_plan_geometry(
    *,
    csv: Path | None = None,
    prescribed_peak_dose_gy: float = 10.0,
    uniform_dose_floor_fraction: float = 0.015,
    spot_specs_mm: Sequence[dict] | None = None,
    vessel_specs_mm: Sequence[dict] | None = None,
    ros_default_uptake: float = 0.05,
    cyto_default_uptake: float = 0.60,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, object]]:
    """Build an irregular weighted lattice with a curved multi-vessel sink network."""

    root = Path(__file__).resolve().parents[1]
    csv_path = csv or (
        root / "runs" / "linac_6mv_polyenergetic_clinical_sfrt" / "case" / "dosedata.csv"
    )
    single_beam, header = load_topas_grid(csv_path, retries=5, retry_delay_sec=0.5)
    dx_cm = float(header["dx_cm"])
    dy_cm = float(header["dy_cm"])
    dz_cm = float(header["dz_cm"])
    z_single_cm = depth_axis_cm(single_beam.shape[2], dz_cm)
    normalized_single, normalization = normalize_single_beam(
        single_beam,
        z_single_cm,
        prescribed_peak_dose_gy=float(prescribed_peak_dose_gy),
    )
    normalized_single = normalized_single.astype(np.float32)
    voxel_size_mm = (dx_cm * 10.0, dy_cm * 10.0, dz_cm * 10.0)

    chosen_spots = list(spot_specs_mm or default_complex_lattice_spot_specs_mm())
    lattice_dose, placed_spots = build_weighted_spot_lattice(
        normalized_single,
        spot_specs_mm=chosen_spots,
        voxel_size_mm=voxel_size_mm,
        margin_bins=6,
    )
    floor_gy = float(uniform_dose_floor_fraction) * float(prescribed_peak_dose_gy)
    if floor_gy > 0.0:
        lattice_dose = lattice_dose + np.float32(floor_gy)

    chosen_vessels = list(vessel_specs_mm or default_complex_vessel_specs_mm())
    uptake_tensor, vessel_mask = build_vessel_network_uptake_tensor(
        lattice_dose.shape,
        voxel_size_mm,
        chosen_vessels,
        num_species=2,
        dtype=np.float32,
    )
    if not np.any(uptake_tensor):
        uptake_tensor, vessel_mask = build_cylindrical_uptake_tensor(
            lattice_dose.shape,
            voxel_size_mm,
            num_species=2,
            vessel_radius_mm=1.5,
            vessel_center_offset_mm=(0.0, 0.0),
            uptake_rates_in_vessel=(float(ros_default_uptake), float(cyto_default_uptake)),
            dtype=np.float32,
        )

    oxygen_modifier = np.ones((2, *lattice_dose.shape), dtype=np.float32)
    type_modifier = np.ones((2, *lattice_dose.shape), dtype=np.float32)

    metadata: Dict[str, object] = {
        "csv": str(csv_path),
        "voxel_size_mm": [float(value) for value in voxel_size_mm],
        "dose_grid_shape": [int(value) for value in lattice_dose.shape],
        "x_cm": centered_axis_cm(lattice_dose.shape[0], dx_cm).tolist(),
        "z_cm": depth_axis_cm(lattice_dose.shape[2], dz_cm).tolist(),
        "normalization": normalization,
        "uniform_dose_floor_gy": float(floor_gy),
        "spot_specs_mm": chosen_spots,
        "placed_spots": placed_spots,
        "vessel_specs_mm": chosen_vessels,
        "vessel_voxel_count": int(np.count_nonzero(vessel_mask)),
    }
    return lattice_dose, uptake_tensor, oxygen_modifier, type_modifier, metadata


__all__ = [
    "build_weighted_spot_lattice",
    "default_complex_lattice_spot_specs_mm",
    "default_complex_vessel_specs_mm",
    "generate_3d_lattice_geometry",
    "generate_complex_lattice_plan_geometry",
]
