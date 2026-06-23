#!/usr/bin/env python
"""Vectorized 3D reaction-diffusion solver for the bystander-effect model."""

from __future__ import annotations

import time
from typing import Iterable

import numpy as np
from scipy import ndimage


def _voxel_spacing_xyz_mm(voxel_size_mm: float | Iterable[float]) -> tuple[float, float, float]:
    if np.isscalar(voxel_size_mm):
        spacing = float(voxel_size_mm)
        if spacing <= 0.0:
            raise ValueError("voxel_size_mm must be positive.")
        return spacing, spacing, spacing

    values = tuple(float(value) for value in voxel_size_mm)
    if len(values) != 3:
        raise ValueError("voxel_size_mm must be a scalar or an iterable of length 3.")
    if any(value <= 0.0 for value in values):
        raise ValueError("All voxel spacings must be positive.")
    return values


def cfl_stability_limit_3d(voxel_size_mm: float | Iterable[float], diffusion_coeff_max: float) -> float:
    """Return the explicit-Euler CFL time-step limit for 3D diffusion."""
    if diffusion_coeff_max <= 0.0:
        raise ValueError("diffusion_coeff_max must be positive.")
    dx_mm, dy_mm, dz_mm = _voxel_spacing_xyz_mm(voxel_size_mm)
    inverse_spacing_sum = (1.0 / dx_mm**2) + (1.0 / dy_mm**2) + (1.0 / dz_mm**2)
    return 1.0 / (2.0 * float(diffusion_coeff_max) * inverse_spacing_sum)


def anisotropic_laplacian_3d(field: np.ndarray, voxel_size_mm: float | Iterable[float]) -> np.ndarray:
    """Compute a 3D Laplacian with finite differences and ndimage kernels."""
    dx_mm, dy_mm, dz_mm = _voxel_spacing_xyz_mm(voxel_size_mm)
    second_derivative_kernel = np.array([1.0, -2.0, 1.0], dtype=np.float32)

    d2x = ndimage.convolve1d(field, second_derivative_kernel, axis=0, mode="nearest") / (dx_mm**2)
    d2y = ndimage.convolve1d(field, second_derivative_kernel, axis=1, mode="nearest") / (dy_mm**2)
    d2z = ndimage.convolve1d(field, second_derivative_kernel, axis=2, mode="nearest") / (dz_mm**2)
    return d2x + d2y + d2z


def solve_bystander_pde_3d(
    dose_grid: np.ndarray,
    voxel_size_mm: float | Iterable[float],
    *,
    steps: int = 200,
    dt: float = 0.05,
    e_max: float = 1.0,
    gamma: float = 0.35,
    diffusion_coeff: float | np.ndarray = 0.5,
    k_decay: float = 0.05,
    progress_interval: int = 50,
    verbose: bool = True,
) -> np.ndarray:
    """Solve the bystander reaction-diffusion PDE with explicit Euler.

    The model is:
        dC/dt = div(D grad(C)) + E(Dose) - k C

    Notes:
    - `voxel_size_mm` may be a scalar or `(dx, dy, dz)`.
    - `diffusion_coeff` may be a scalar or a 3D array matching `dose_grid`.
    - For spatially varying `diffusion_coeff`, this implementation uses the
      common `D * Laplacian(C)` approximation, which is exact for uniform D and
      adequate for the current water-phantom use case.
    """

    if dose_grid.ndim != 3:
        raise ValueError("dose_grid must be a 3D numpy array.")
    if steps < 1:
        raise ValueError("steps must be at least 1.")
    if dt <= 0.0:
        raise ValueError("dt must be positive.")
    if e_max < 0.0 or gamma < 0.0 or k_decay < 0.0:
        raise ValueError("e_max, gamma, and k_decay must be non-negative.")

    dose_grid = np.asarray(dose_grid, dtype=np.float32)
    diffusion_coeff_array = np.asarray(diffusion_coeff, dtype=np.float32)
    if diffusion_coeff_array.ndim == 0:
        diffusion_coeff_max = float(diffusion_coeff_array)
    else:
        if diffusion_coeff_array.shape != dose_grid.shape:
            raise ValueError("diffusion_coeff must be scalar or the same shape as dose_grid.")
        diffusion_coeff_max = float(np.max(diffusion_coeff_array))
    if diffusion_coeff_max <= 0.0:
        raise ValueError("diffusion_coeff must contain positive values.")

    dt_limit = cfl_stability_limit_3d(voxel_size_mm, diffusion_coeff_max)
    if dt > dt_limit:
        raise ValueError(
            f"Chosen dt={dt:.6f} exceeds the CFL stability limit {dt_limit:.6f}. "
            "Reduce dt or the diffusion coefficient."
        )

    if verbose:
        print("\n--- Initializing 3D PDE Bystander Solver ---")
        print(f"Grid shape: {dose_grid.shape}")
        print(f"Voxel size (mm): {_voxel_spacing_xyz_mm(voxel_size_mm)}")
        print(f"CFL dt limit: {dt_limit:.6f}")
        print(f"Using dt={dt:.6f} for {steps} steps.")

    start_time = time.time()

    source_term = (float(e_max) * (1.0 - np.exp(-float(gamma) * dose_grid))).astype(np.float32)
    if diffusion_coeff_array.ndim == 0:
        diffusion_field = np.full_like(dose_grid, float(diffusion_coeff_array), dtype=np.float32)
    else:
        diffusion_field = diffusion_coeff_array.astype(np.float32, copy=False)

    concentration = np.zeros_like(dose_grid, dtype=np.float32)

    if verbose:
        print("Simulating PDE time steps...")

    for step in range(int(steps)):
        laplacian = anisotropic_laplacian_3d(concentration, voxel_size_mm)
        dcdt = (diffusion_field * laplacian) + source_term - (float(k_decay) * concentration)
        concentration += dcdt * float(dt)

        if verbose and progress_interval > 0 and (step + 1) % int(progress_interval) == 0:
            print(f"  ... Step {step + 1}/{steps} completed.")

    if verbose:
        print(f"PDE solver finished in {time.time() - start_time:.2f} seconds.")

    return concentration


def calculate_pde_multi_effect_survival(
    lq_survival_grid: np.ndarray,
    pde_concentration_grid: np.ndarray,
    *,
    scaling_factor: float = 0.1,
) -> np.ndarray:
    """Convert a stabilized PDE concentration field into a survival penalty."""
    if scaling_factor < 0.0:
        raise ValueError("scaling_factor must be non-negative.")

    lq_survival_grid = np.asarray(lq_survival_grid, dtype=np.float32)
    pde_concentration_grid = np.asarray(pde_concentration_grid, dtype=np.float32)
    if lq_survival_grid.shape != pde_concentration_grid.shape:
        raise ValueError("lq_survival_grid and pde_concentration_grid must have the same shape.")

    return lq_survival_grid * np.exp(-pde_concentration_grid * float(scaling_factor))
