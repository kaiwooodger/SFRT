#!/usr/bin/env python
"""Vectorized multi-species 3D PDE solver with an anatomical uptake sink.

This module keeps the project-native `(x, y, z)` voxel ordering used by the
existing TOPAS and analysis pipeline. The returned concentration tensor is
therefore shaped `(species, x, y, z)`, which is equivalent to the requested
4D multi-channel architecture while remaining drop-in compatible with the
current codebase.
"""

from __future__ import annotations

import time
from typing import Iterable, Sequence

import numpy as np

from bystander_pde_solver import anisotropic_laplacian_3d, cfl_stability_limit_3d


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


def centered_z_offset_from_surface_depth_mm(
    n_voxels: int,
    spacing_mm: float,
    depth_from_surface_mm: float,
) -> float:
    """Convert a physical depth-from-surface into the solver's centered z frame."""

    if int(n_voxels) < 1:
        raise ValueError("n_voxels must be at least 1.")
    if float(spacing_mm) <= 0.0:
        raise ValueError("spacing_mm must be positive.")
    if float(depth_from_surface_mm) < 0.0:
        raise ValueError("depth_from_surface_mm must be non-negative.")

    return float(depth_from_surface_mm) - (float(n_voxels) * float(spacing_mm) / 2.0)


def _infer_num_species(
    uptake_tensor: np.ndarray | None,
    *species_parameters: Sequence[float] | np.ndarray | float,
) -> int:
    if uptake_tensor is not None:
        if uptake_tensor.ndim != 4:
            raise ValueError("uptake_tensor must have shape (species, x, y, z).")
        return int(uptake_tensor.shape[0])

    inferred = 1
    for values in species_parameters:
        arr = np.asarray(values, dtype=np.float32)
        if arr.ndim == 0:
            continue
        if arr.ndim != 1:
            raise ValueError("Species parameters must be scalars or 1D sequences.")
        inferred = max(inferred, int(arr.shape[0]))
    return max(2, inferred)


def _as_species_vector(
    values: Sequence[float] | np.ndarray | float,
    *,
    name: str,
    num_species: int,
) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    if arr.ndim == 0:
        return np.full(num_species, float(arr), dtype=np.float32)
    if arr.ndim != 1 or arr.shape[0] != num_species:
        raise ValueError(f"{name} must be a scalar or a sequence of length {num_species}.")
    return arr.astype(np.float32, copy=False)


def build_cylindrical_uptake_tensor(
    grid_shape: Sequence[int],
    voxel_size_mm: float | Iterable[float],
    *,
    num_species: int = 2,
    vessel_radius_mm: float = 3.0,
    vessel_center_offset_mm: tuple[float, float] = (0.0, 0.0),
    uptake_rates_in_vessel: Sequence[float] | np.ndarray | float = (0.05, 0.60),
    dtype: np.dtype = np.float32,
) -> tuple[np.ndarray, np.ndarray]:
    """Create a species-specific cylindrical uptake sink running along z.

    The cylinder is defined in the lateral `(x, y)` plane and broadcast across
    all depth slices, which represents a blood vessel aligned with the z-axis.
    `vessel_center_offset_mm` is measured from the geometric center of the
    lattice, allowing the sink to be placed in a central valley rather than
    forcing it onto the array midpoint.
    """

    if len(tuple(grid_shape)) != 3:
        raise ValueError("grid_shape must contain exactly three dimensions.")
    if num_species < 1:
        raise ValueError("num_species must be at least 1.")
    if vessel_radius_mm <= 0.0:
        raise ValueError("vessel_radius_mm must be positive.")

    nx, ny, nz = (int(value) for value in grid_shape)
    dx_mm, dy_mm, _ = _voxel_spacing_xyz_mm(voxel_size_mm)
    offset_x_mm, offset_y_mm = (float(value) for value in vessel_center_offset_mm)
    uptake_rates = _as_species_vector(
        uptake_rates_in_vessel,
        name="uptake_rates_in_vessel",
        num_species=num_species,
    )

    x_mm = (np.arange(nx, dtype=np.float32) - (nx - 1) / 2.0) * float(dx_mm)
    y_mm = (np.arange(ny, dtype=np.float32) - (ny - 1) / 2.0) * float(dy_mm)
    x_rel_mm = x_mm[:, None] - offset_x_mm
    y_rel_mm = y_mm[None, :] - offset_y_mm
    vessel_mask_xy = (x_rel_mm**2 + y_rel_mm**2) <= float(vessel_radius_mm) ** 2
    vessel_mask = np.broadcast_to(vessel_mask_xy[:, :, None], (nx, ny, nz))

    uptake_tensor = np.zeros((num_species, nx, ny, nz), dtype=dtype)
    for species_idx, rate in enumerate(uptake_rates):
        uptake_tensor[species_idx, vessel_mask] = float(rate)

    return uptake_tensor, vessel_mask


def build_vessel_network_uptake_tensor(
    grid_shape: Sequence[int],
    voxel_size_mm: float | Iterable[float],
    vessel_specs: Sequence[dict],
    *,
    num_species: int = 2,
    dtype: np.dtype = np.float32,
) -> tuple[np.ndarray, np.ndarray]:
    """Create a multi-vessel uptake tensor from piecewise-linear centerlines.

    Each vessel spec is a dict with:
    - `nodes_mm`: list of `(x_mm, y_mm, z_mm)` points in the centered solver frame
    - `radius_mm`: vessel radius
    - `uptake_rates_in_vessel`: scalar or species-length sequence

    The vessel is rasterized slice-by-slice in z by interpolating the x/y
    centerline through the provided nodes. This keeps memory use modest while
    still supporting curved and depth-limited vascular sinks.
    """

    if len(tuple(grid_shape)) != 3:
        raise ValueError("grid_shape must contain exactly three dimensions.")
    if num_species < 1:
        raise ValueError("num_species must be at least 1.")

    nx, ny, nz = (int(value) for value in grid_shape)
    dx_mm, dy_mm, dz_mm = _voxel_spacing_xyz_mm(voxel_size_mm)
    x_mm = (np.arange(nx, dtype=np.float32) - (nx - 1) / 2.0) * float(dx_mm)
    y_mm = (np.arange(ny, dtype=np.float32) - (ny - 1) / 2.0) * float(dy_mm)
    z_mm = (np.arange(nz, dtype=np.float32) - (nz - 1) / 2.0) * float(dz_mm)

    x_grid_mm = x_mm[:, None]
    y_grid_mm = y_mm[None, :]

    uptake_tensor = np.zeros((num_species, nx, ny, nz), dtype=dtype)
    vessel_union_mask = np.zeros((nx, ny, nz), dtype=bool)

    for spec in vessel_specs:
        nodes = np.asarray(spec["nodes_mm"], dtype=np.float32)
        if nodes.ndim != 2 or nodes.shape[1] != 3 or nodes.shape[0] < 2:
            raise ValueError("Each vessel spec must provide at least two 3D nodes in nodes_mm.")
        radius_mm = float(spec["radius_mm"])
        if radius_mm <= 0.0:
            raise ValueError("Each vessel radius_mm must be positive.")
        uptake_rates = _as_species_vector(
            spec.get("uptake_rates_in_vessel", (0.05, 0.60)),
            name="uptake_rates_in_vessel",
            num_species=num_species,
        )

        order = np.argsort(nodes[:, 2])
        nodes = nodes[order]
        z_nodes = nodes[:, 2]
        x_nodes = nodes[:, 0]
        y_nodes = nodes[:, 1]
        z_min = float(np.min(z_nodes))
        z_max = float(np.max(z_nodes))

        slice_indices = np.flatnonzero((z_mm >= z_min) & (z_mm <= z_max))
        if slice_indices.size == 0:
            continue

        x_interp = np.interp(z_mm[slice_indices], z_nodes, x_nodes)
        y_interp = np.interp(z_mm[slice_indices], z_nodes, y_nodes)
        radius_sq = float(radius_mm) ** 2

        for local_idx, z_idx in enumerate(slice_indices):
            mask_xy = (
                (x_grid_mm - float(x_interp[local_idx])) ** 2
                + (y_grid_mm - float(y_interp[local_idx])) ** 2
            ) <= radius_sq
            if not np.any(mask_xy):
                continue
            vessel_union_mask[:, :, int(z_idx)] |= mask_xy
            for species_idx, rate in enumerate(uptake_rates):
                uptake_tensor[species_idx, :, :, int(z_idx)][mask_xy] = np.maximum(
                    uptake_tensor[species_idx, :, :, int(z_idx)][mask_xy],
                    float(rate),
                )

    return uptake_tensor, vessel_union_mask


def build_state_modifier_tensors(
    grid_shape: Sequence[int],
    voxel_size_mm: float | Iterable[float],
    *,
    num_species: int = 2,
    tumor_radius_mm: float = 15.0,
    tumor_center_offset_mm: tuple[float, float, float] = (0.0, 0.0, 0.0),
    tumor_cytokine_multiplier: float = 2.0,
    hypoxic_radius_mm: float = 5.0,
    hypoxic_center_offset_mm: tuple[float, float, float] | None = None,
    hypoxic_ros_scale: float = 0.10,
    hypoxic_cytokine_multiplier: float = 3.0,
    dtype: np.dtype = np.float32,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Create spatial modifier tensors for cell type and oxygenation.

    Returns `(M_type, M_oxygen, tumor_mask, hypoxic_mask)` using the project
    lattice ordering `(x, y, z)`.
    """

    if len(tuple(grid_shape)) != 3:
        raise ValueError("grid_shape must contain exactly three dimensions.")
    if num_species < 1:
        raise ValueError("num_species must be at least 1.")
    if tumor_radius_mm <= 0.0:
        raise ValueError("tumor_radius_mm must be positive.")
    if hypoxic_radius_mm <= 0.0:
        raise ValueError("hypoxic_radius_mm must be positive.")
    if tumor_cytokine_multiplier < 0.0:
        raise ValueError("tumor_cytokine_multiplier must be non-negative.")
    if hypoxic_ros_scale < 0.0 or hypoxic_cytokine_multiplier < 0.0:
        raise ValueError("Hypoxic emission multipliers must be non-negative.")

    nx, ny, nz = (int(value) for value in grid_shape)
    dx_mm, dy_mm, dz_mm = _voxel_spacing_xyz_mm(voxel_size_mm)

    tumor_offset_x_mm, tumor_offset_y_mm, tumor_offset_z_mm = (
        float(value) for value in tumor_center_offset_mm
    )
    if hypoxic_center_offset_mm is None:
        hypoxic_offset_x_mm, hypoxic_offset_y_mm, hypoxic_offset_z_mm = (
            tumor_offset_x_mm,
            tumor_offset_y_mm,
            tumor_offset_z_mm,
        )
    else:
        hypoxic_offset_x_mm, hypoxic_offset_y_mm, hypoxic_offset_z_mm = (
            float(value) for value in hypoxic_center_offset_mm
        )

    x_mm = (np.arange(nx, dtype=np.float32) - (nx - 1) / 2.0) * float(dx_mm)
    y_mm = (np.arange(ny, dtype=np.float32) - (ny - 1) / 2.0) * float(dy_mm)
    z_mm = (np.arange(nz, dtype=np.float32) - (nz - 1) / 2.0) * float(dz_mm)

    tumor_distance_sq_mm = (
        (x_mm[:, None, None] - tumor_offset_x_mm) ** 2
        + (y_mm[None, :, None] - tumor_offset_y_mm) ** 2
        + (z_mm[None, None, :] - tumor_offset_z_mm) ** 2
    )
    tumor_mask = tumor_distance_sq_mm <= float(tumor_radius_mm) ** 2

    hypoxic_distance_sq_mm = (
        (x_mm[:, None, None] - hypoxic_offset_x_mm) ** 2
        + (y_mm[None, :, None] - hypoxic_offset_y_mm) ** 2
        + (z_mm[None, None, :] - hypoxic_offset_z_mm) ** 2
    )
    hypoxic_mask = hypoxic_distance_sq_mm <= float(hypoxic_radius_mm) ** 2

    type_modifier = np.ones((num_species, nx, ny, nz), dtype=dtype)
    oxygen_modifier = np.ones((num_species, nx, ny, nz), dtype=dtype)

    if num_species > 1:
        type_modifier[1, tumor_mask] = float(tumor_cytokine_multiplier)

    oxygen_modifier[0, hypoxic_mask] = float(hypoxic_ros_scale)
    if num_species > 1:
        oxygen_modifier[1, hypoxic_mask] = float(hypoxic_cytokine_multiplier)

    return type_modifier, oxygen_modifier, tumor_mask, hypoxic_mask


def calculate_state_dependent_emission(
    dose_grid: np.ndarray,
    voxel_size_mm: float | Iterable[float],
    *,
    num_species: int = 2,
    emission_emax: Sequence[float] | np.ndarray | float = (1.5, 0.8),
    emission_gamma_per_gy: float = 0.35,
    tumor_radius_mm: float = 15.0,
    tumor_center_offset_mm: tuple[float, float, float] = (0.0, 0.0, 0.0),
    tumor_cytokine_multiplier: float = 2.0,
    hypoxic_radius_mm: float = 5.0,
    hypoxic_center_offset_mm: tuple[float, float, float] | None = None,
    hypoxic_ros_scale: float = 0.10,
    hypoxic_cytokine_multiplier: float = 3.0,
) -> np.ndarray:
    """Precompute the state-dependent emission tensor E_k(x).

    The model is:
        E_k(x) = Emax_k * (1 - exp(-gamma * Dose(x))) * M_type(x, k) * M_oxygen(x, k)
    """

    dose_grid = np.asarray(dose_grid, dtype=np.float32)
    if dose_grid.ndim != 3:
        raise ValueError("dose_grid must be a 3D numpy array.")
    if emission_gamma_per_gy < 0.0:
        raise ValueError("emission_gamma_per_gy must be non-negative.")

    emax_vector = _as_species_vector(
        emission_emax,
        name="emission_emax",
        num_species=int(num_species),
    )
    base_emission = (1.0 - np.exp(-float(emission_gamma_per_gy) * dose_grid)).astype(np.float32)
    type_modifier, oxygen_modifier, _, _ = build_state_modifier_tensors(
        dose_grid.shape,
        voxel_size_mm,
        num_species=int(num_species),
        tumor_radius_mm=float(tumor_radius_mm),
        tumor_center_offset_mm=tuple(float(value) for value in tumor_center_offset_mm),
        tumor_cytokine_multiplier=float(tumor_cytokine_multiplier),
        hypoxic_radius_mm=float(hypoxic_radius_mm),
        hypoxic_center_offset_mm=(
            None
            if hypoxic_center_offset_mm is None
            else tuple(float(value) for value in hypoxic_center_offset_mm)
        ),
        hypoxic_ros_scale=float(hypoxic_ros_scale),
        hypoxic_cytokine_multiplier=float(hypoxic_cytokine_multiplier),
        dtype=np.float32,
    )
    return (
        emax_vector[:, None, None, None]
        * base_emission[None, :, :, :]
        * type_modifier
        * oxygen_modifier
    ).astype(np.float32, copy=False)


def solve_multispecies_pde_3d(
    dose_grid: np.ndarray,
    voxel_size_mm: float | Iterable[float],
    *,
    steps: int = 400,
    dt: float = 0.15,
    diffusion_coeffs: Sequence[float] | np.ndarray | float = (0.8, 0.4),
    decay_coeffs: Sequence[float] | np.ndarray | float = (0.2, 0.02),
    emission_emax: Sequence[float] | np.ndarray | float = (1.5, 0.8),
    emission_gamma_per_gy: float = 0.35,
    emission_tensor: np.ndarray | None = None,
    state_dependent_emission: bool = False,
    tumor_radius_mm: float = 15.0,
    tumor_center_offset_mm: tuple[float, float, float] = (0.0, 0.0, 0.0),
    tumor_cytokine_multiplier: float = 2.0,
    hypoxic_radius_mm: float = 5.0,
    hypoxic_center_offset_mm: tuple[float, float, float] | None = None,
    hypoxic_ros_scale: float = 0.10,
    hypoxic_cytokine_multiplier: float = 3.0,
    uptake_tensor: np.ndarray | None = None,
    vessel_radius_mm: float = 3.0,
    vessel_center_offset_mm: tuple[float, float] = (0.0, 0.0),
    uptake_rates_in_vessel: Sequence[float] | np.ndarray | float = (0.05, 0.60),
    progress_interval: int = 50,
    verbose: bool = True,
) -> np.ndarray:
    """Solve the multi-species explicit reaction-diffusion PDE.

    The per-species model is:
        dC_k/dt = D_k * Laplacian(C_k) - lambda_k * C_k - u_k(x) * C_k + E_k(x)

    Notes:
    - The returned tensor has shape `(species, x, y, z)`.
    - `uptake_tensor` must match that same shape when provided. If omitted, a
      centered cylindrical vessel sink is synthesized automatically.
    - `emission_tensor` may be passed directly. Otherwise the solver uses either
      a uniform saturated dose-emission model or the state-dependent Phase 4
      modifiers when `state_dependent_emission=True`.
    - The solver intentionally loops only over species; all voxel updates remain
      vectorized over the 3D domain.
    """

    dose_grid = np.asarray(dose_grid, dtype=np.float32)
    if dose_grid.ndim != 3:
        raise ValueError("dose_grid must be a 3D numpy array.")
    if steps < 1:
        raise ValueError("steps must be at least 1.")
    if dt <= 0.0:
        raise ValueError("dt must be positive.")
    if emission_gamma_per_gy < 0.0:
        raise ValueError("emission_gamma_per_gy must be non-negative.")

    num_species = _infer_num_species(
        uptake_tensor,
        diffusion_coeffs,
        decay_coeffs,
        emission_emax,
        uptake_rates_in_vessel,
    )

    diffusion_vector = _as_species_vector(
        diffusion_coeffs,
        name="diffusion_coeffs",
        num_species=num_species,
    )
    decay_vector = _as_species_vector(
        decay_coeffs,
        name="decay_coeffs",
        num_species=num_species,
    )
    emax_vector = _as_species_vector(
        emission_emax,
        name="emission_emax",
        num_species=num_species,
    )

    if np.any(diffusion_vector <= 0.0):
        raise ValueError("All diffusion coefficients must be positive.")
    if np.any(decay_vector < 0.0):
        raise ValueError("All decay coefficients must be non-negative.")
    if np.any(emax_vector < 0.0):
        raise ValueError("All emission_emax values must be non-negative.")

    dt_limit = cfl_stability_limit_3d(voxel_size_mm, float(np.max(diffusion_vector)))
    if dt > dt_limit:
        raise ValueError(
            f"Chosen dt={dt:.6f} exceeds the CFL stability limit {dt_limit:.6f}. "
            "Reduce dt or the maximum diffusion coefficient."
        )

    if uptake_tensor is None:
        uptake_field, _ = build_cylindrical_uptake_tensor(
            dose_grid.shape,
            voxel_size_mm,
            num_species=num_species,
            vessel_radius_mm=vessel_radius_mm,
            vessel_center_offset_mm=vessel_center_offset_mm,
            uptake_rates_in_vessel=uptake_rates_in_vessel,
            dtype=np.float32,
        )
    else:
        uptake_field = np.asarray(uptake_tensor, dtype=np.float32)
        expected_shape = (num_species, *dose_grid.shape)
        if uptake_field.shape != expected_shape:
            raise ValueError(
                f"uptake_tensor must have shape {expected_shape}, got {uptake_field.shape}."
            )
        if np.any(uptake_field < 0.0):
            raise ValueError("uptake_tensor must be non-negative.")

    if verbose:
        print("\n--- Initializing Multi-Species Heterogeneous PDE Solver ---")
        print(f"Grid shape: {dose_grid.shape}")
        print(f"Tensor shape: {(num_species, *dose_grid.shape)}")
        print(f"Voxel size (mm): {_voxel_spacing_xyz_mm(voxel_size_mm)}")
        print(f"CFL dt limit: {dt_limit:.6f}")
        print(f"Using dt={dt:.6f} for {steps} steps.")

    start_time = time.time()
    if emission_tensor is not None:
        source_terms = np.asarray(emission_tensor, dtype=np.float32)
        expected_shape = (num_species, *dose_grid.shape)
        if source_terms.shape != expected_shape:
            raise ValueError(
                f"emission_tensor must have shape {expected_shape}, got {source_terms.shape}."
            )
        if np.any(source_terms < 0.0):
            raise ValueError("emission_tensor must be non-negative.")
        emission_mode = "precomputed_tensor"
    elif state_dependent_emission:
        source_terms = calculate_state_dependent_emission(
            dose_grid,
            voxel_size_mm,
            num_species=num_species,
            emission_emax=emax_vector,
            emission_gamma_per_gy=float(emission_gamma_per_gy),
            tumor_radius_mm=float(tumor_radius_mm),
            tumor_center_offset_mm=tuple(float(value) for value in tumor_center_offset_mm),
            tumor_cytokine_multiplier=float(tumor_cytokine_multiplier),
            hypoxic_radius_mm=float(hypoxic_radius_mm),
            hypoxic_center_offset_mm=(
                None
                if hypoxic_center_offset_mm is None
                else tuple(float(value) for value in hypoxic_center_offset_mm)
            ),
            hypoxic_ros_scale=float(hypoxic_ros_scale),
            hypoxic_cytokine_multiplier=float(hypoxic_cytokine_multiplier),
        )
        emission_mode = "state_dependent"
    else:
        base_emission = (1.0 - np.exp(-float(emission_gamma_per_gy) * dose_grid)).astype(np.float32)
        source_terms = emax_vector[:, None, None, None] * base_emission[None, :, :, :]
        emission_mode = "uniform_saturated"
    concentration = np.zeros((num_species, *dose_grid.shape), dtype=np.float32)

    if verbose:
        print(f"Emission mode: {emission_mode}")
        print(f"Simulating {steps} time steps for {num_species} species...")

    for step in range(int(steps)):
        for species_idx in range(num_species):
            laplacian = anisotropic_laplacian_3d(concentration[species_idx], voxel_size_mm)
            dcdt = (
                diffusion_vector[species_idx] * laplacian
                - decay_vector[species_idx] * concentration[species_idx]
                - uptake_field[species_idx] * concentration[species_idx]
                + source_terms[species_idx]
            )
            concentration[species_idx] += dcdt * float(dt)
            np.maximum(concentration[species_idx], 0.0, out=concentration[species_idx])

        if verbose and progress_interval > 0 and (step + 1) % int(progress_interval) == 0:
            print(f"  ... Step {step + 1}/{steps} completed.")

    if verbose:
        print(f"Solver finished in {time.time() - start_time:.2f} seconds.")

    return concentration


def solve_multispecies_pde_3d_with_hazard(
    dose_grid: np.ndarray,
    voxel_size_mm: float | Iterable[float],
    *,
    steps: int = 400,
    dt: float = 0.15,
    diffusion_coeffs: Sequence[float] | np.ndarray | float = (0.8, 0.4),
    decay_coeffs: Sequence[float] | np.ndarray | float = (0.2, 0.02),
    emission_emax: Sequence[float] | np.ndarray | float = (1.5, 0.8),
    emission_gamma_per_gy: float = 0.35,
    emission_tensor: np.ndarray | None = None,
    state_dependent_emission: bool = False,
    tumor_radius_mm: float = 15.0,
    tumor_center_offset_mm: tuple[float, float, float] = (0.0, 0.0, 0.0),
    tumor_cytokine_multiplier: float = 2.0,
    hypoxic_radius_mm: float = 5.0,
    hypoxic_center_offset_mm: tuple[float, float, float] | None = None,
    hypoxic_ros_scale: float = 0.10,
    hypoxic_cytokine_multiplier: float = 3.0,
    uptake_tensor: np.ndarray | None = None,
    vessel_radius_mm: float = 3.0,
    vessel_center_offset_mm: tuple[float, float] = (0.0, 0.0),
    uptake_rates_in_vessel: Sequence[float] | np.ndarray | float = (0.05, 0.60),
    hazard_weights: Sequence[float] | np.ndarray = (0.40, 0.40),
    progress_interval: int = 50,
    verbose: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Solve the multi-species PDE and integrate a cumulative hazard grid over time."""

    dose_grid = np.asarray(dose_grid, dtype=np.float32)
    if dose_grid.ndim != 3:
        raise ValueError("dose_grid must be a 3D numpy array.")
    if steps < 1:
        raise ValueError("steps must be at least 1.")
    if dt <= 0.0:
        raise ValueError("dt must be positive.")
    if emission_gamma_per_gy < 0.0:
        raise ValueError("emission_gamma_per_gy must be non-negative.")

    num_species = _infer_num_species(
        uptake_tensor,
        diffusion_coeffs,
        decay_coeffs,
        emission_emax,
        uptake_rates_in_vessel,
    )
    if num_species < 2:
        raise ValueError("Phase 7 hazard integration requires at least two species channels.")

    diffusion_vector = _as_species_vector(
        diffusion_coeffs,
        name="diffusion_coeffs",
        num_species=num_species,
    )
    decay_vector = _as_species_vector(
        decay_coeffs,
        name="decay_coeffs",
        num_species=num_species,
    )
    emax_vector = _as_species_vector(
        emission_emax,
        name="emission_emax",
        num_species=num_species,
    )

    if np.any(diffusion_vector <= 0.0):
        raise ValueError("All diffusion coefficients must be positive.")
    if np.any(decay_vector < 0.0):
        raise ValueError("All decay coefficients must be non-negative.")
    if np.any(emax_vector < 0.0):
        raise ValueError("All emission_emax values must be non-negative.")

    dt_limit = cfl_stability_limit_3d(voxel_size_mm, float(np.max(diffusion_vector)))
    if dt > dt_limit:
        raise ValueError(
            f"Chosen dt={dt:.6f} exceeds the CFL stability limit {dt_limit:.6f}. "
            "Reduce dt or the maximum diffusion coefficient."
        )

    if uptake_tensor is None:
        uptake_field, _ = build_cylindrical_uptake_tensor(
            dose_grid.shape,
            voxel_size_mm,
            num_species=num_species,
            vessel_radius_mm=vessel_radius_mm,
            vessel_center_offset_mm=vessel_center_offset_mm,
            uptake_rates_in_vessel=uptake_rates_in_vessel,
            dtype=np.float32,
        )
    else:
        uptake_field = np.asarray(uptake_tensor, dtype=np.float32)
        expected_shape = (num_species, *dose_grid.shape)
        if uptake_field.shape != expected_shape:
            raise ValueError(
                f"uptake_tensor must have shape {expected_shape}, got {uptake_field.shape}."
            )
        if np.any(uptake_field < 0.0):
            raise ValueError("uptake_tensor must be non-negative.")

    if emission_tensor is not None:
        source_terms = np.asarray(emission_tensor, dtype=np.float32)
        expected_shape = (num_species, *dose_grid.shape)
        if source_terms.shape != expected_shape:
            raise ValueError(
                f"emission_tensor must have shape {expected_shape}, got {source_terms.shape}."
            )
        if np.any(source_terms < 0.0):
            raise ValueError("emission_tensor must be non-negative.")
        emission_mode = "precomputed_tensor"
    elif state_dependent_emission:
        source_terms = calculate_state_dependent_emission(
            dose_grid,
            voxel_size_mm,
            num_species=num_species,
            emission_emax=emax_vector,
            emission_gamma_per_gy=float(emission_gamma_per_gy),
            tumor_radius_mm=float(tumor_radius_mm),
            tumor_center_offset_mm=tuple(float(value) for value in tumor_center_offset_mm),
            tumor_cytokine_multiplier=float(tumor_cytokine_multiplier),
            hypoxic_radius_mm=float(hypoxic_radius_mm),
            hypoxic_center_offset_mm=(
                None
                if hypoxic_center_offset_mm is None
                else tuple(float(value) for value in hypoxic_center_offset_mm)
            ),
            hypoxic_ros_scale=float(hypoxic_ros_scale),
            hypoxic_cytokine_multiplier=float(hypoxic_cytokine_multiplier),
        )
        emission_mode = "state_dependent"
    else:
        base_emission = (1.0 - np.exp(-float(emission_gamma_per_gy) * dose_grid)).astype(np.float32)
        source_terms = emax_vector[:, None, None, None] * base_emission[None, :, :, :]
        emission_mode = "uniform_saturated"

    local_weights = _as_species_vector(
        hazard_weights,
        name="hazard_weights",
        num_species=2,
    )
    if np.any(local_weights < 0.0):
        raise ValueError("hazard_weights must be non-negative.")

    concentration = np.zeros((num_species, *dose_grid.shape), dtype=np.float32)
    hazard_grid = np.zeros(dose_grid.shape, dtype=np.float32)

    if verbose:
        print("\n--- Initializing Multi-Species Temporal Hazard PDE Solver ---")
        print(f"Grid shape: {dose_grid.shape}")
        print(f"Tensor shape: {(num_species, *dose_grid.shape)}")
        print(f"Voxel size (mm): {_voxel_spacing_xyz_mm(voxel_size_mm)}")
        print(f"CFL dt limit: {dt_limit:.6f}")
        print(f"Using dt={dt:.6f} for {steps} steps.")
        print(f"Emission mode: {emission_mode}")
        print(
            f"Simulating {steps} time steps and integrating cumulative hazard with "
            f"weights [{float(local_weights[0]):.2f}, {float(local_weights[1]):.2f}]..."
        )

    start_time = time.time()
    for step in range(int(steps)):
        for species_idx in range(num_species):
            laplacian = anisotropic_laplacian_3d(concentration[species_idx], voxel_size_mm)
            dcdt = (
                diffusion_vector[species_idx] * laplacian
                - decay_vector[species_idx] * concentration[species_idx]
                - uptake_field[species_idx] * concentration[species_idx]
                + source_terms[species_idx]
            )
            concentration[species_idx] += dcdt * float(dt)
            np.maximum(concentration[species_idx], 0.0, out=concentration[species_idx])

        instantaneous_stress = (
            float(local_weights[0]) * concentration[0]
            + float(local_weights[1]) * concentration[1]
        )
        hazard_grid += instantaneous_stress * float(dt)

        if verbose and progress_interval > 0 and (step + 1) % int(progress_interval) == 0:
            print(f"  ... Step {step + 1}/{steps} completed.")

    if verbose:
        print(f"Temporal hazard solver finished in {time.time() - start_time:.2f} seconds.")

    return concentration, hazard_grid


def solve_multispecies_pde_3d_with_hazard_observables(
    dose_grid: np.ndarray,
    voxel_size_mm: float | Iterable[float],
    *,
    steps: int = 400,
    dt: float = 0.15,
    diffusion_coeffs: Sequence[float] | np.ndarray | float = (0.8, 0.4),
    decay_coeffs: Sequence[float] | np.ndarray | float = (0.2, 0.02),
    emission_emax: Sequence[float] | np.ndarray | float = (1.5, 0.8),
    emission_gamma_per_gy: float = 0.35,
    emission_tensor: np.ndarray | None = None,
    state_dependent_emission: bool = False,
    tumor_radius_mm: float = 15.0,
    tumor_center_offset_mm: tuple[float, float, float] = (0.0, 0.0, 0.0),
    tumor_cytokine_multiplier: float = 2.0,
    hypoxic_radius_mm: float = 5.0,
    hypoxic_center_offset_mm: tuple[float, float, float] | None = None,
    hypoxic_ros_scale: float = 0.10,
    hypoxic_cytokine_multiplier: float = 3.0,
    uptake_tensor: np.ndarray | None = None,
    vessel_radius_mm: float = 3.0,
    vessel_center_offset_mm: tuple[float, float] = (0.0, 0.0),
    uptake_rates_in_vessel: Sequence[float] | np.ndarray | float = (0.05, 0.60),
    hazard_weights: Sequence[float] | np.ndarray = (0.40, 0.40),
    history_masks: dict[str, np.ndarray] | None = None,
    history_interval: int = 10,
    progress_interval: int = 50,
    verbose: bool = True,
) -> dict[str, object]:
    """Solve the multi-species PDE, integrate hazard, and record assay observables.

    Returns a dictionary containing the final concentration tensor, cumulative
    hazard grid, per-species peak concentration grids, and time-series region
    means suitable for in silico assay plots.
    """

    dose_grid = np.asarray(dose_grid, dtype=np.float32)
    if dose_grid.ndim != 3:
        raise ValueError("dose_grid must be a 3D numpy array.")
    if steps < 1:
        raise ValueError("steps must be at least 1.")
    if dt <= 0.0:
        raise ValueError("dt must be positive.")
    if emission_gamma_per_gy < 0.0:
        raise ValueError("emission_gamma_per_gy must be non-negative.")
    if history_interval < 1:
        raise ValueError("history_interval must be at least 1.")

    num_species = _infer_num_species(
        uptake_tensor,
        diffusion_coeffs,
        decay_coeffs,
        emission_emax,
        uptake_rates_in_vessel,
    )
    if num_species < 2:
        raise ValueError("Hazard-observable integration requires at least two species channels.")

    diffusion_vector = _as_species_vector(
        diffusion_coeffs,
        name="diffusion_coeffs",
        num_species=num_species,
    )
    decay_vector = _as_species_vector(
        decay_coeffs,
        name="decay_coeffs",
        num_species=num_species,
    )
    emax_vector = _as_species_vector(
        emission_emax,
        name="emission_emax",
        num_species=num_species,
    )

    if np.any(diffusion_vector <= 0.0):
        raise ValueError("All diffusion coefficients must be positive.")
    if np.any(decay_vector < 0.0):
        raise ValueError("All decay coefficients must be non-negative.")
    if np.any(emax_vector < 0.0):
        raise ValueError("All emission_emax values must be non-negative.")

    dt_limit = cfl_stability_limit_3d(voxel_size_mm, float(np.max(diffusion_vector)))
    if dt > dt_limit:
        raise ValueError(
            f"Chosen dt={dt:.6f} exceeds the CFL stability limit {dt_limit:.6f}. "
            "Reduce dt or the maximum diffusion coefficient."
        )

    if uptake_tensor is None:
        uptake_field, _ = build_cylindrical_uptake_tensor(
            dose_grid.shape,
            voxel_size_mm,
            num_species=num_species,
            vessel_radius_mm=vessel_radius_mm,
            vessel_center_offset_mm=vessel_center_offset_mm,
            uptake_rates_in_vessel=uptake_rates_in_vessel,
            dtype=np.float32,
        )
    else:
        uptake_field = np.asarray(uptake_tensor, dtype=np.float32)
        expected_shape = (num_species, *dose_grid.shape)
        if uptake_field.shape != expected_shape:
            raise ValueError(
                f"uptake_tensor must have shape {expected_shape}, got {uptake_field.shape}."
            )
        if np.any(uptake_field < 0.0):
            raise ValueError("uptake_tensor must be non-negative.")

    if emission_tensor is not None:
        source_terms = np.asarray(emission_tensor, dtype=np.float32)
        expected_shape = (num_species, *dose_grid.shape)
        if source_terms.shape != expected_shape:
            raise ValueError(
                f"emission_tensor must have shape {expected_shape}, got {source_terms.shape}."
            )
        if np.any(source_terms < 0.0):
            raise ValueError("emission_tensor must be non-negative.")
        emission_mode = "precomputed_tensor"
    elif state_dependent_emission:
        source_terms = calculate_state_dependent_emission(
            dose_grid,
            voxel_size_mm,
            num_species=num_species,
            emission_emax=emax_vector,
            emission_gamma_per_gy=float(emission_gamma_per_gy),
            tumor_radius_mm=float(tumor_radius_mm),
            tumor_center_offset_mm=tuple(float(value) for value in tumor_center_offset_mm),
            tumor_cytokine_multiplier=float(tumor_cytokine_multiplier),
            hypoxic_radius_mm=float(hypoxic_radius_mm),
            hypoxic_center_offset_mm=(
                None
                if hypoxic_center_offset_mm is None
                else tuple(float(value) for value in hypoxic_center_offset_mm)
            ),
            hypoxic_ros_scale=float(hypoxic_ros_scale),
            hypoxic_cytokine_multiplier=float(hypoxic_cytokine_multiplier),
        )
        emission_mode = "state_dependent"
    else:
        base_emission = (1.0 - np.exp(-float(emission_gamma_per_gy) * dose_grid)).astype(np.float32)
        source_terms = emax_vector[:, None, None, None] * base_emission[None, :, :, :]
        emission_mode = "uniform_saturated"

    local_weights = _as_species_vector(
        hazard_weights,
        name="hazard_weights",
        num_species=2,
    )
    if np.any(local_weights < 0.0):
        raise ValueError("hazard_weights must be non-negative.")

    mask_arrays: dict[str, np.ndarray] = {}
    if history_masks:
        for name, mask in history_masks.items():
            mask_array = np.asarray(mask, dtype=bool)
            if mask_array.shape != dose_grid.shape:
                raise ValueError(
                    f"history mask '{name}' must have shape {dose_grid.shape}, got {mask_array.shape}."
                )
            mask_arrays[str(name)] = mask_array

    concentration = np.zeros((num_species, *dose_grid.shape), dtype=np.float32)
    peak_concentration = np.zeros_like(concentration)
    hazard_grid = np.zeros(dose_grid.shape, dtype=np.float32)
    ros_hazard_grid = np.zeros(dose_grid.shape, dtype=np.float32)
    cytokine_hazard_grid = np.zeros(dose_grid.shape, dtype=np.float32)

    time_points: list[float] = []
    global_history: list[np.ndarray] = []
    mask_histories: dict[str, list[np.ndarray]] = {name: [] for name in mask_arrays}

    if verbose:
        print("\n--- Initializing Multi-Species Temporal Hazard PDE Solver (observables) ---")
        print(f"Grid shape: {dose_grid.shape}")
        print(f"Tensor shape: {(num_species, *dose_grid.shape)}")
        print(f"Voxel size (mm): {_voxel_spacing_xyz_mm(voxel_size_mm)}")
        print(f"CFL dt limit: {dt_limit:.6f}")
        print(f"Using dt={dt:.6f} for {steps} steps.")
        print(f"Emission mode: {emission_mode}")
        print(
            f"Simulating {steps} time steps and integrating cumulative hazard with "
            f"weights [{float(local_weights[0]):.2f}, {float(local_weights[1]):.2f}]..."
        )

    start_time = time.time()
    for step in range(int(steps)):
        for species_idx in range(num_species):
            laplacian = anisotropic_laplacian_3d(concentration[species_idx], voxel_size_mm)
            dcdt = (
                diffusion_vector[species_idx] * laplacian
                - decay_vector[species_idx] * concentration[species_idx]
                - uptake_field[species_idx] * concentration[species_idx]
                + source_terms[species_idx]
            )
            concentration[species_idx] += dcdt * float(dt)
            np.maximum(concentration[species_idx], 0.0, out=concentration[species_idx])

        np.maximum(peak_concentration, concentration, out=peak_concentration)

        ros_hazard_grid += float(local_weights[0]) * concentration[0] * float(dt)
        cytokine_hazard_grid += float(local_weights[1]) * concentration[1] * float(dt)
        hazard_grid[:] = ros_hazard_grid + cytokine_hazard_grid

        if (step + 1) % int(history_interval) == 0 or (step + 1) == int(steps):
            time_points.append(float(step + 1) * float(dt))
            global_history.append(
                np.asarray(
                    [float(np.mean(concentration[0])), float(np.mean(concentration[1]))],
                    dtype=np.float32,
                )
            )
            for name, mask_array in mask_arrays.items():
                mask_histories[name].append(
                    np.asarray(
                        [
                            float(np.mean(concentration[0][mask_array])),
                            float(np.mean(concentration[1][mask_array])),
                        ],
                        dtype=np.float32,
                    )
                )

        if verbose and progress_interval > 0 and (step + 1) % int(progress_interval) == 0:
            print(f"  ... Step {step + 1}/{steps} completed.")

    if verbose:
        print(f"Temporal hazard observable solver finished in {time.time() - start_time:.2f} seconds.")

    return {
        "concentration": concentration,
        "peak_concentration": peak_concentration,
        "hazard_grid": hazard_grid,
        "ros_hazard_grid": ros_hazard_grid,
        "cytokine_hazard_grid": cytokine_hazard_grid,
        "time_axis": np.asarray(time_points, dtype=np.float32),
        "global_mean_history": (
            np.stack(global_history, axis=1).astype(np.float32, copy=False)
            if global_history
            else np.zeros((2, 0), dtype=np.float32)
        ),
        "mask_mean_history": {
            name: (
                np.stack(values, axis=1).astype(np.float32, copy=False)
                if values
                else np.zeros((2, 0), dtype=np.float32)
            )
            for name, values in mask_histories.items()
        },
    }


def solve_advanced_multichannel_pde(
    dose_grid: np.ndarray,
    voxel_size_mm: float | Iterable[float],
    *,
    steps: int = 400,
    dt: float = 0.15,
) -> np.ndarray:
    """Compatibility wrapper using the biological defaults from the blueprint."""
    return solve_multispecies_pde_3d(
        dose_grid,
        voxel_size_mm,
        steps=steps,
        dt=dt,
        diffusion_coeffs=(0.8, 0.4),
        decay_coeffs=(0.2, 0.02),
        emission_emax=(1.5, 0.8),
        emission_gamma_per_gy=0.35,
        state_dependent_emission=False,
        vessel_radius_mm=3.0,
        vessel_center_offset_mm=(0.0, 0.0),
        uptake_rates_in_vessel=(0.05, 0.60),
    )


def combine_multispecies_toxicity(
    multispecies_tensor: np.ndarray,
    *,
    species_weights: Sequence[float] | np.ndarray | None = None,
) -> np.ndarray:
    """Collapse a `(species, x, y, z)` tensor into a single toxic field."""
    tensor = np.asarray(multispecies_tensor, dtype=np.float32)
    if tensor.ndim != 4:
        raise ValueError("multispecies_tensor must have shape (species, x, y, z).")

    num_species = int(tensor.shape[0])
    if species_weights is None:
        weights = np.full(num_species, 1.0 / num_species, dtype=np.float32)
    else:
        weights = _as_species_vector(
            species_weights,
            name="species_weights",
            num_species=num_species,
        )
    return np.tensordot(weights, tensor, axes=(0, 0)).astype(np.float32, copy=False)


def calculate_multispecies_multi_effect_survival(
    lq_survival_grid: np.ndarray,
    multispecies_tensor: np.ndarray,
    *,
    species_weights: Sequence[float] | np.ndarray | None = None,
    scaling_factor: float = 0.160891551,
) -> np.ndarray:
    """Apply the weighted multi-species penalty to an LQ survival grid."""
    if scaling_factor < 0.0:
        raise ValueError("scaling_factor must be non-negative.")

    lq_survival_grid = np.asarray(lq_survival_grid, dtype=np.float32)
    toxic_concentration = combine_multispecies_toxicity(
        multispecies_tensor,
        species_weights=species_weights,
    )
    if toxic_concentration.shape != lq_survival_grid.shape:
        raise ValueError("The combined toxic concentration must match lq_survival_grid.")

    return lq_survival_grid * np.exp(-toxic_concentration * float(scaling_factor))


def calculate_systemic_immune_penalty(
    dose_grid: np.ndarray,
    voxel_size_mm: float | Iterable[float],
    *,
    icd_threshold_gy: float = 10.0,
    immune_max_penalty: float = 1.0,
    immune_half_volume_cm3: float = 5.0,
) -> tuple[float, float]:
    """Return the global immune penalty scalar and ICD volume in cm^3."""

    dose_grid = np.asarray(dose_grid, dtype=np.float32)
    if dose_grid.ndim != 3:
        raise ValueError("dose_grid must be a 3D numpy array.")
    if icd_threshold_gy < 0.0:
        raise ValueError("icd_threshold_gy must be non-negative.")
    if immune_max_penalty < 0.0:
        raise ValueError("immune_max_penalty must be non-negative.")
    if immune_half_volume_cm3 <= 0.0:
        raise ValueError("immune_half_volume_cm3 must be positive.")

    dx_mm, dy_mm, dz_mm = _voxel_spacing_xyz_mm(voxel_size_mm)
    voxel_volume_cm3 = (float(dx_mm) * float(dy_mm) * float(dz_mm)) / 1000.0
    icd_voxels = int(np.count_nonzero(dose_grid >= float(icd_threshold_gy)))
    icd_volume_cm3 = float(icd_voxels) * float(voxel_volume_cm3)
    immune_penalty = float(immune_max_penalty) * (
        icd_volume_cm3 / (icd_volume_cm3 + float(immune_half_volume_cm3))
    )
    return float(immune_penalty), float(icd_volume_cm3)


def calculate_phase5_multi_effect_survival(
    lq_survival_grid: np.ndarray,
    multispecies_tensor: np.ndarray,
    dose_grid: np.ndarray,
    voxel_size_mm: float | Iterable[float],
    scaling_factor: float,
    *,
    channel_weights: Sequence[float] | np.ndarray = (0.40, 0.40, 0.20),
    icd_threshold_gy: float = 10.0,
    immune_max_penalty: float = 1.0,
    immune_half_volume_cm3: float = 5.0,
    verbose: bool = True,
) -> np.ndarray:
    """Apply the Phase 5 local-plus-systemic nonlocal penalty model."""

    if scaling_factor < 0.0:
        raise ValueError("scaling_factor must be non-negative.")

    lq_survival_grid = np.asarray(lq_survival_grid, dtype=np.float32)
    tensor = np.asarray(multispecies_tensor, dtype=np.float32)
    dose_grid = np.asarray(dose_grid, dtype=np.float32)
    if tensor.ndim != 4:
        raise ValueError("multispecies_tensor must have shape (species, x, y, z).")
    if tensor.shape[0] < 2:
        raise ValueError("Phase 5 requires at least two local species channels.")
    if lq_survival_grid.shape != dose_grid.shape or lq_survival_grid.shape != tensor.shape[1:]:
        raise ValueError("lq_survival_grid, dose_grid, and multispecies_tensor spatial shapes must match.")

    weights = np.asarray(channel_weights, dtype=np.float32)
    if weights.ndim != 1 or weights.shape[0] != 3:
        raise ValueError("channel_weights must contain exactly three values: [ROS, cytokine, immune].")
    if np.any(weights < 0.0):
        raise ValueError("channel_weights must be non-negative.")
    if not np.isclose(float(np.sum(weights)), 1.0, atol=1e-6):
        raise ValueError("channel_weights must sum to 1.0.")

    ros_penalty = tensor[0]
    cytokine_penalty = tensor[1]
    immune_penalty, icd_volume_cm3 = calculate_systemic_immune_penalty(
        dose_grid,
        voxel_size_mm,
        icd_threshold_gy=float(icd_threshold_gy),
        immune_max_penalty=float(immune_max_penalty),
        immune_half_volume_cm3=float(immune_half_volume_cm3),
    )
    if verbose:
        print(
            f"  -> ICD Volume (>{float(icd_threshold_gy):.2f} Gy): {icd_volume_cm3:.2f} cm^3 | "
            f"P_immune scalar: {immune_penalty:.4f}"
        )

    nonlocal_penalty = (
        float(weights[0]) * ros_penalty
        + float(weights[1]) * cytokine_penalty
        + float(weights[2]) * float(immune_penalty)
    )
    return (lq_survival_grid * np.exp(-nonlocal_penalty * float(scaling_factor))).astype(np.float32, copy=False)


def calculate_phase7_survival(
    lq_survival_grid: np.ndarray,
    hazard_grid: np.ndarray,
    dose_grid: np.ndarray,
    voxel_size_mm: float | Iterable[float],
    scaling_factor: float,
    *,
    weight_immune: float = 0.20,
    icd_threshold_gy: float = 10.0,
    immune_max_penalty: float = 1.0,
    immune_half_volume_cm3: float = 5.0,
    verbose: bool = True,
) -> np.ndarray:
    """Apply the integrated cumulative hazard plus systemic immune penalty."""

    if scaling_factor < 0.0:
        raise ValueError("scaling_factor must be non-negative.")
    if weight_immune < 0.0:
        raise ValueError("weight_immune must be non-negative.")

    lq_survival_grid = np.asarray(lq_survival_grid, dtype=np.float32)
    hazard_grid = np.asarray(hazard_grid, dtype=np.float32)
    dose_grid = np.asarray(dose_grid, dtype=np.float32)
    if lq_survival_grid.shape != hazard_grid.shape or lq_survival_grid.shape != dose_grid.shape:
        raise ValueError("lq_survival_grid, hazard_grid, and dose_grid must share the same shape.")

    immune_penalty, icd_volume_cm3 = calculate_systemic_immune_penalty(
        dose_grid,
        voxel_size_mm,
        icd_threshold_gy=float(icd_threshold_gy),
        immune_max_penalty=float(immune_max_penalty),
        immune_half_volume_cm3=float(immune_half_volume_cm3),
    )
    if verbose:
        print(
            f"  -> ICD Volume (>{float(icd_threshold_gy):.2f} Gy): {icd_volume_cm3:.2f} cm^3 | "
            f"P_immune scalar: {immune_penalty:.4f}"
        )

    nonlocal_penalty = hazard_grid + (float(weight_immune) * float(immune_penalty))
    return (lq_survival_grid * np.exp(-nonlocal_penalty * float(scaling_factor))).astype(np.float32, copy=False)


def run_pde_temporal_integration(
    dose_grid: np.ndarray,
    voxel_size_mm: float | Iterable[float],
    *,
    D_cyto: float,
    lambda_cyto: float,
    gamma: float,
    u_k: np.ndarray | None = None,
    M_oxygen: np.ndarray | None = None,
    M_type: np.ndarray | None = None,
    D_ros: float = 0.8,
    lambda_ros: float = 0.2,
    Emax_ros: float = 1.5,
    Emax_cyto: float = 0.8,
    w_ros: float = 0.40,
    w_cyto: float = 0.40,
    steps: int = 400,
    dt: float = 0.12,
    progress_interval: int = 50,
    verbose: bool = True,
) -> np.ndarray:
    """Convenience wrapper returning only the Phase 7 cumulative hazard grid.

    This keeps the Phase 9 holdout scripts concise while still routing through
    the fully validated multi-species temporal PDE engine.
    """

    dose_grid = np.asarray(dose_grid, dtype=np.float32)
    if dose_grid.ndim != 3:
        raise ValueError("dose_grid must be a 3D numpy array.")

    if M_oxygen is None:
        oxygen_modifier = np.ones((2, *dose_grid.shape), dtype=np.float32)
    else:
        oxygen_modifier = np.asarray(M_oxygen, dtype=np.float32)
    if M_type is None:
        type_modifier = np.ones((2, *dose_grid.shape), dtype=np.float32)
    else:
        type_modifier = np.asarray(M_type, dtype=np.float32)
    expected_modifier_shape = (2, *dose_grid.shape)
    if oxygen_modifier.shape != expected_modifier_shape or type_modifier.shape != expected_modifier_shape:
        raise ValueError(
            f"M_oxygen and M_type must each have shape {expected_modifier_shape}."
        )

    base_emission = (1.0 - np.exp(-float(gamma) * dose_grid)).astype(np.float32)
    emission_tensor = (
        np.asarray([float(Emax_ros), float(Emax_cyto)], dtype=np.float32)[:, None, None, None]
        * base_emission[None, :, :, :]
        * type_modifier
        * oxygen_modifier
    ).astype(np.float32, copy=False)

    _, hazard_grid = solve_multispecies_pde_3d_with_hazard(
        dose_grid=dose_grid,
        voxel_size_mm=voxel_size_mm,
        steps=int(steps),
        dt=float(dt),
        diffusion_coeffs=(float(D_ros), float(D_cyto)),
        decay_coeffs=(float(lambda_ros), float(lambda_cyto)),
        emission_emax=(float(Emax_ros), float(Emax_cyto)),
        emission_gamma_per_gy=float(gamma),
        emission_tensor=emission_tensor,
        uptake_tensor=(None if u_k is None else np.asarray(u_k, dtype=np.float32)),
        hazard_weights=(float(w_ros), float(w_cyto)),
        progress_interval=int(progress_interval),
        verbose=bool(verbose),
    )
    return hazard_grid


def calculate_effective_dose(
    final_survival_grid: np.ndarray,
    *,
    alpha: float = 0.03,
    beta: float = 0.003,
    min_survival: float = 1.0e-10,
) -> np.ndarray:
    """Invert the LQ model to compute biological effective dose in Gy."""

    if alpha <= 0.0:
        raise ValueError("alpha must be positive.")
    if beta <= 0.0:
        raise ValueError("beta must be positive.")
    if min_survival <= 0.0 or min_survival > 1.0:
        raise ValueError("min_survival must be in the interval (0, 1].")

    survival = np.asarray(final_survival_grid, dtype=np.float32)
    safe_survival = np.clip(survival, float(min_survival), 1.0)
    discriminant = (float(alpha) ** 2) - (4.0 * float(beta) * np.log(safe_survival))
    return ((-float(alpha) + np.sqrt(discriminant)) / (2.0 * float(beta))).astype(np.float32, copy=False)


__all__ = [
    "build_cylindrical_uptake_tensor",
    "build_state_modifier_tensors",
    "centered_z_offset_from_surface_depth_mm",
    "combine_multispecies_toxicity",
    "calculate_effective_dose",
    "calculate_phase5_multi_effect_survival",
    "calculate_phase7_survival",
    "calculate_systemic_immune_penalty",
    "calculate_state_dependent_emission",
    "calculate_multispecies_multi_effect_survival",
    "run_pde_temporal_integration",
    "solve_advanced_multichannel_pde",
    "solve_multispecies_pde_3d",
    "solve_multispecies_pde_3d_with_hazard",
    "solve_multispecies_pde_3d_with_hazard_observables",
]
