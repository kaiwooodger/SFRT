#!/usr/bin/env python
"""Geometry generators for train/test bystander-model generalization studies.

The project uses `(x, y, z)` array ordering throughout. These helpers therefore
return dose grids with shape `(nx, ny, nz)` plus matching modifier tensors with
shape `(species, nx, ny, nz)`.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np


def _uniform_modifier_tensors(
    grid_shape: Sequence[int],
    *,
    num_species: int = 2,
    dtype: np.dtype = np.float32,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    nx, ny, nz = (int(value) for value in grid_shape)
    dose_grid = np.zeros((nx, ny, nz), dtype=dtype)
    uptake_tensor = np.zeros((num_species, nx, ny, nz), dtype=dtype)
    oxygen_modifier = np.ones((num_species, nx, ny, nz), dtype=dtype)
    type_modifier = np.ones((num_species, nx, ny, nz), dtype=dtype)
    return dose_grid, uptake_tensor, oxygen_modifier, type_modifier


def generate_half_field_geometry(
    grid_shape: Sequence[int],
    voxel_size_mm: float | Sequence[float],
    *,
    dose_peak_gy: float = 10.0,
    num_species: int = 2,
    irradiated_side: str = "left",
    dtype: np.dtype = np.float32,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate a simple half-field irradiation geometry for calibration.

    The field edge is placed at the geometric center of the x-axis. By default,
    the left half of the domain is irradiated to `dose_peak_gy` while the right
    half remains at 0 Gy.
    """

    if dose_peak_gy < 0.0:
        raise ValueError("dose_peak_gy must be non-negative.")
    if irradiated_side not in {"left", "right"}:
        raise ValueError("irradiated_side must be either 'left' or 'right'.")

    dose_grid, uptake_tensor, oxygen_modifier, type_modifier = _uniform_modifier_tensors(
        grid_shape,
        num_species=num_species,
        dtype=dtype,
    )
    nx = int(dose_grid.shape[0])
    split_idx = nx // 2
    if irradiated_side == "left":
        dose_grid[:split_idx, :, :] = float(dose_peak_gy)
    else:
        dose_grid[split_idx:, :, :] = float(dose_peak_gy)
    return dose_grid, uptake_tensor, oxygen_modifier, type_modifier


def generate_2d_stripe_geometry(
    grid_shape: Sequence[int],
    voxel_size_mm: float | Sequence[float],
    *,
    pitch_mm: float,
    beam_width_mm: float = 10.0,
    dose_peak_gy: float = 10.0,
    num_species: int = 2,
    central_feature: str = "peak",
    dtype: np.dtype = np.float32,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate alternating 2D irradiation stripes across the x-axis.

    Parameters use the project-native `(x, y, z)` ordering. The pattern is
    constant in `y` and `z`, which makes it a clean dimensional holdout between
    half-field calibration and 3D lattice validation.
    """

    if pitch_mm <= 0.0:
        raise ValueError("pitch_mm must be positive.")
    if beam_width_mm <= 0.0:
        raise ValueError("beam_width_mm must be positive.")
    if dose_peak_gy < 0.0:
        raise ValueError("dose_peak_gy must be non-negative.")
    if central_feature not in {"peak", "valley"}:
        raise ValueError("central_feature must be either 'peak' or 'valley'.")

    dose_grid, uptake_tensor, oxygen_modifier, type_modifier = _uniform_modifier_tensors(
        grid_shape,
        num_species=num_species,
        dtype=dtype,
    )

    if np.isscalar(voxel_size_mm):
        dx_mm = float(voxel_size_mm)
    else:
        dx_mm = float(tuple(voxel_size_mm)[0])
    if dx_mm <= 0.0:
        raise ValueError("voxel_size_mm must be positive.")

    pitch_bins = max(1, int(round(float(pitch_mm) / dx_mm)))
    beam_width_bins = max(1, int(round(float(beam_width_mm) / dx_mm)))
    nx = int(dose_grid.shape[0])
    center_x = nx // 2

    phase_shift_bins = 0 if central_feature == "peak" else pitch_bins // 2
    stripe_centers = range(center_x - 4 * pitch_bins, center_x + 5 * pitch_bins, pitch_bins)
    half_width_low = beam_width_bins // 2
    half_width_high = beam_width_bins - half_width_low

    for center in stripe_centers:
        shifted_center = int(center + phase_shift_bins)
        start_x = max(0, shifted_center - half_width_low)
        end_x = min(nx, shifted_center + half_width_high)
        if start_x < end_x:
            dose_grid[start_x:end_x, :, :] = float(dose_peak_gy)

    return dose_grid, uptake_tensor, oxygen_modifier, type_modifier


def generate_2d_parallel_stripe_geometry(
    grid_shape: Sequence[int],
    voxel_size_mm: float | Sequence[float],
    *,
    pitch_mm: float,
    beam_width_mm: float = 10.0,
    dose_peak_gy: float = 10.0,
    leakage: float = 0.0,
    num_species: int = 2,
    dtype: np.dtype = np.float32,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate two infinite parallel Y-Z irradiation planes with a true central valley.

    The project uses `(x, y, z)` ordering, so the returned dose field has shape
    `(nx, ny, nz)`. The two beam centers are placed at `x = -pitch/2` and
    `x = +pitch/2`, leaving `x = 0` as the exact center of the pristine valley.
    """

    if pitch_mm <= 0.0:
        raise ValueError("pitch_mm must be positive.")
    if beam_width_mm <= 0.0:
        raise ValueError("beam_width_mm must be positive.")
    if dose_peak_gy < 0.0:
        raise ValueError("dose_peak_gy must be non-negative.")
    if leakage < 0.0 or leakage > 1.0:
        raise ValueError("leakage must be in the interval [0, 1].")

    dose_grid, uptake_tensor, oxygen_modifier, type_modifier = _uniform_modifier_tensors(
        grid_shape,
        num_species=num_species,
        dtype=dtype,
    )

    if np.isscalar(voxel_size_mm):
        dx_mm = float(voxel_size_mm)
    else:
        dx_mm = float(tuple(voxel_size_mm)[0])
    if dx_mm <= 0.0:
        raise ValueError("voxel_size_mm must be positive.")

    dose_grid.fill(float(dose_peak_gy) * float(leakage))

    nx = int(dose_grid.shape[0])
    x_center = nx // 2
    pitch_bins = max(1, int(round(float(pitch_mm) / dx_mm)))
    half_width_bins = max(1, int(round((float(beam_width_mm) / 2.0) / dx_mm)))

    left_center = x_center - (pitch_bins // 2)
    right_center = x_center + (pitch_bins // 2)

    left_start = max(0, left_center - half_width_bins)
    left_end = min(nx, left_center + half_width_bins)
    right_start = max(0, right_center - half_width_bins)
    right_end = min(nx, right_center + half_width_bins)

    dose_grid[left_start:left_end, :, :] = float(dose_peak_gy)
    dose_grid[right_start:right_end, :, :] = float(dose_peak_gy)

    return dose_grid, uptake_tensor, oxygen_modifier, type_modifier


__all__ = [
    "generate_2d_parallel_stripe_geometry",
    "generate_2d_stripe_geometry",
    "generate_half_field_geometry",
]
