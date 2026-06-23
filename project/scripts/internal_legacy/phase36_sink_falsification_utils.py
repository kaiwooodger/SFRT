#!/usr/bin/env python3
"""Shared utilities for Phase 36 sink falsification analyses."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
from scipy import ndimage

from run_phase15_detailed_headneck_bioaware import build_anatomical_biology_tensors
from run_phase26_vascular_sink_ablation import (
    PRIMARY_ENDPOINTS,
    bio_args_from_config,
    build_emission_tensor,
    calculate_effective_dose,
    calculate_phase7_survival,
    extract_endpoints,
    summarize_assays,
    solve_multispecies_pde_3d_with_hazard_observables,
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
)


MODEL_LABELS = {
    "physical_only": "Physical-only",
    "no_sink": "Bystander, no vascular sink",
    "true_vessel_sink": "Bystander, anatomical vascular sink",
    "mirror_lr_sink": "Bystander, mirrored left-right vessel sink",
    "shifted_sink_12mm_ap": "Bystander, vessel sink shifted 12 mm anterior",
    "uniform_body_sink_mass_matched": "Bystander, uniform body sink (mass matched)",
    "blurred_vessel_sink_sigma10mm": "Bystander, blurred vessel sink (10 mm sigma)",
    "peri_gtv_shell_sink_5_20mm": "Bystander, peri-GTV shell sink (5-20 mm)",
    "distance_decay_sink_lambda15mm": "Bystander, distance-decay sink (lambda 15 mm)",
}

BIOLOGICAL_MODEL_IDS: Tuple[str, ...] = tuple(
    model_id for model_id in MODEL_LABELS if model_id != "physical_only"
)

PHASE36A_COMPARATOR_IDS: Tuple[str, ...] = (
    "no_sink",
    "mirror_lr_sink",
    "shifted_sink_12mm_ap",
    "uniform_body_sink_mass_matched",
    "blurred_vessel_sink_sigma10mm",
    "peri_gtv_shell_sink_5_20mm",
    "distance_decay_sink_lambda15mm",
)

PHASE36B_MODEL_IDS: Tuple[str, ...] = (
    "no_sink",
    "true_vessel_sink",
    "mirror_lr_sink",
    "uniform_body_sink_mass_matched",
)

ENDPOINT_DIRECTION = {spec.key: bool(spec.higher_is_better) for spec in PRIMARY_ENDPOINTS}
ENDPOINT_LABELS = {spec.key: str(spec.label) for spec in PRIMARY_ENDPOINTS}
ENDPOINT_UNITS = {spec.key: str(spec.units) for spec in PRIMARY_ENDPOINTS}
ENDPOINT_DIRECTION.update(
    {
        "gtv_d95": True,
        "thyroid_mean": False,
        "body_dmax": False,
        "parotid_l_mean": False,
        "spill_shell_15_30_mean": False,
        "outside_gtv_d2": False,
        "oar_adjacent_outside_gtv_mean": False,
        "ptv_valley_outside_gtv_mean": False,
        "peak_mean": True,
        "valley_mean": False,
    }
)
ENDPOINT_LABELS.update(
    {
        "gtv_d95": "GTV D95",
        "thyroid_mean": "Thyroid mean",
        "body_dmax": "Body Dmax",
        "parotid_l_mean": "Parotid L mean",
        "spill_shell_15_30_mean": "Peri-GTV 15-30 mm mean",
        "outside_gtv_d2": "Outside-GTV D2",
        "oar_adjacent_outside_gtv_mean": "OAR-adjacent outside-GTV mean",
        "ptv_valley_outside_gtv_mean": "PTV valley outside-GTV mean",
        "peak_mean": "Peak mean",
        "valley_mean": "Valley mean",
    }
)
ENDPOINT_UNITS.update(
    {
        "gtv_d95": "Gy",
        "thyroid_mean": "Gy",
        "body_dmax": "Gy",
        "parotid_l_mean": "Gy",
        "spill_shell_15_30_mean": "Gy",
        "outside_gtv_d2": "Gy",
        "oar_adjacent_outside_gtv_mean": "Gy",
        "ptv_valley_outside_gtv_mean": "Gy",
        "peak_mean": "Gy",
        "valley_mean": "Gy",
    }
)

COMPARISON_ENDPOINT_KEYS: Tuple[str, ...] = (
    "ptv_d95",
    "pvdr",
    "spill_shell_0_5_mean",
    "spill_shell_5_15_mean",
    "brainstem_d2",
    "parotid_r_mean",
    "outside_gtv_d2",
    "oar_adjacent_outside_gtv_mean",
)

ASSAY_DIRECTION = {
    "mean_gammah2ax_peak": False,
    "mean_gammah2ax_valley": False,
    "mean_tunel_peak": False,
    "mean_tunel_valley": False,
    "mean_cytokine_final_peak": False,
    "mean_cytokine_final_valley": False,
    "cytokine_global_auc": False,
    "cytokine_peak_roi_auc": False,
    "cytokine_valley_roi_auc": False,
    "immune_scalar": False,
    "icd_volume_cm3": False,
}

ASSAY_KEYS_FOR_SUMMARY: Tuple[str, ...] = (
    "mean_gammah2ax_valley",
    "mean_tunel_valley",
    "cytokine_valley_roi_auc",
    "cytokine_global_auc",
)


def load_case_context(path: Path) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Tuple[float, float, float], List[Tuple[float, float, float]]]:
    with np.load(path) as data:
        axes_mm = {
            "x": np.asarray(data["axes_x_mm"], dtype=np.float32),
            "y": np.asarray(data["axes_y_mm"], dtype=np.float32),
            "z": np.asarray(data["axes_z_mm"], dtype=np.float32),
        }
        structures = {
            key.removeprefix("struct_"): np.asarray(data[key], dtype=bool)
            for key in data.files
            if key.startswith("struct_")
        }
        vertices = [
            tuple(float(v) for v in row)
            for row in np.asarray(data["vertex_centers_mm"], dtype=np.float32)
        ]
    voxel_size = (
        float(axes_mm["x"][1] - axes_mm["x"][0]),
        float(axes_mm["y"][1] - axes_mm["y"][0]),
        float(axes_mm["z"][1] - axes_mm["z"][0]),
    )
    return structures, axes_mm, voxel_size, vertices


def load_dose_npz(path: Path) -> np.ndarray:
    with np.load(path) as data:
        return np.asarray(data["dose_gy"], dtype=np.float32)


def add_structure_aliases(structures: Mapping[str, np.ndarray], voxel_size_mm: Tuple[float, float, float]) -> Dict[str, np.ndarray]:
    result = {name: np.asarray(mask, dtype=bool) for name, mask in structures.items()}
    if "PARATHYROIDS" not in result:
        left = np.asarray(result.get("PARATHYROID_L", np.zeros_like(result["BODY"], dtype=bool)), dtype=bool)
        right = np.asarray(result.get("PARATHYROID_R", np.zeros_like(result["BODY"], dtype=bool)), dtype=bool)
        result["PARATHYROIDS"] = left | right
    if "HYPOXIA" not in result:
        vertex_target = np.asarray(result.get("VERTEX_TARGET", result["GTV"]), dtype=bool)
        if np.any(vertex_target):
            distance = ndimage.distance_transform_edt(vertex_target, sampling=voxel_size_mm)
            cutoff = 0.45 * float(np.max(distance))
            hypoxia = vertex_target & (distance >= cutoff)
            if not np.any(hypoxia):
                hypoxia = vertex_target
        else:
            hypoxia = np.zeros_like(result["BODY"], dtype=bool)
        result["HYPOXIA"] = hypoxia
    return result


def voxel_volume_cc(voxel_size_mm: Tuple[float, float, float]) -> float:
    return float(voxel_size_mm[0] * voxel_size_mm[1] * voxel_size_mm[2] / 1000.0)


def build_true_uptake_and_modifiers(
    config: Mapping[str, object],
    structures: Mapping[str, np.ndarray],
    *,
    uptake_scale: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    bio_args = bio_args_from_config(config, uptake_scale=float(uptake_scale))
    uptake_tensor, m_type, m_oxygen, _ = build_anatomical_biology_tensors(bio_args, dict(structures))
    return (
        np.asarray(uptake_tensor, dtype=np.float32),
        np.asarray(m_type, dtype=np.float32),
        np.asarray(m_oxygen, dtype=np.float32),
    )


def _shift_3d_no_wrap(arr: np.ndarray, shift_xyz: Tuple[int, int, int]) -> np.ndarray:
    out = np.zeros_like(arr)
    src_slices = []
    dst_slices = []
    for axis, shift in enumerate(shift_xyz):
        n = arr.shape[axis]
        if shift >= 0:
            src_start, src_end = 0, max(0, n - shift)
            dst_start, dst_end = shift, shift + max(0, n - shift)
        else:
            amount = -shift
            src_start, src_end = amount, n
            dst_start, dst_end = 0, n - amount
        src_slices.append(slice(src_start, src_end))
        dst_slices.append(slice(dst_start, dst_end))
    out[tuple(dst_slices)] = arr[tuple(src_slices)]
    return out


def _renormalize_species_mass(
    tensor: np.ndarray,
    target_mass: np.ndarray,
    *,
    support_mask: np.ndarray | None = None,
) -> np.ndarray:
    out = np.asarray(tensor, dtype=np.float32).copy()
    if support_mask is not None:
        support = np.asarray(support_mask, dtype=bool)
        out[:, ~support] = 0.0
    for species_idx in range(out.shape[0]):
        current = float(np.sum(out[species_idx]))
        target = float(target_mass[species_idx])
        if target <= 0.0 or current <= 1.0e-12:
            out[species_idx] = 0.0
            continue
        out[species_idx] *= float(target / current)
    return out


def _uniform_mass_tensor(target_mass: np.ndarray, support_mask: np.ndarray) -> np.ndarray:
    support = np.asarray(support_mask, dtype=bool)
    out = np.zeros((target_mass.size, *support.shape), dtype=np.float32)
    count = int(np.count_nonzero(support))
    if count == 0:
        return out
    for species_idx, target in enumerate(np.asarray(target_mass, dtype=np.float32)):
        if float(target) <= 0.0:
            continue
        out[species_idx, support] = float(target) / float(count)
    return out


def _weighted_mass_tensor(target_mass: np.ndarray, weights: np.ndarray, support_mask: np.ndarray) -> np.ndarray:
    support = np.asarray(support_mask, dtype=bool)
    base = np.asarray(weights, dtype=np.float32)
    base = np.where(support, np.maximum(base, 0.0), 0.0)
    out = np.zeros((target_mass.size, *support.shape), dtype=np.float32)
    mass = float(np.sum(base))
    if mass <= 1.0e-12:
        return _uniform_mass_tensor(target_mass, support)
    for species_idx, target in enumerate(np.asarray(target_mass, dtype=np.float32)):
        if float(target) <= 0.0:
            continue
        out[species_idx] = base * (float(target) / mass)
    return out


def build_sink_uptake_tensor(
    model_id: str,
    *,
    true_uptake_tensor: np.ndarray,
    structures: Mapping[str, np.ndarray],
    axes_mm: Mapping[str, np.ndarray],
    shift_ap_mm: float = 12.0,
    blur_sigma_mm: float = 10.0,
    shell_inner_mm: float = 5.0,
    shell_outer_mm: float = 20.0,
    decay_lambda_mm: float = 15.0,
) -> Tuple[np.ndarray, Dict[str, object]]:
    true_tensor = np.asarray(true_uptake_tensor, dtype=np.float32)
    body = np.asarray(structures["BODY"], dtype=bool)
    gtv = np.asarray(structures["GTV"], dtype=bool)
    spacing_mm = (
        float(axes_mm["x"][1] - axes_mm["x"][0]),
        float(axes_mm["y"][1] - axes_mm["y"][0]),
        float(axes_mm["z"][1] - axes_mm["z"][0]),
    )
    target_mass = np.asarray(true_tensor.reshape(true_tensor.shape[0], -1).sum(axis=1), dtype=np.float32)

    if model_id == "no_sink":
        uptake = np.zeros_like(true_tensor, dtype=np.float32)
    elif model_id == "true_vessel_sink":
        uptake = true_tensor.copy()
    elif model_id == "mirror_lr_sink":
        mirrored = true_tensor[:, ::-1, :, :]
        uptake = _renormalize_species_mass(mirrored, target_mass, support_mask=body)
    elif model_id == "shifted_sink_12mm_ap":
        shift_vox = int(round(float(shift_ap_mm) / max(spacing_mm[1], 1.0e-6)))
        shifted = np.zeros_like(true_tensor, dtype=np.float32)
        for species_idx in range(true_tensor.shape[0]):
            shifted[species_idx] = _shift_3d_no_wrap(true_tensor[species_idx], (0, shift_vox, 0))
        uptake = _renormalize_species_mass(shifted, target_mass, support_mask=body)
    elif model_id == "uniform_body_sink_mass_matched":
        uptake = _uniform_mass_tensor(target_mass, body)
    elif model_id == "blurred_vessel_sink_sigma10mm":
        sigma_vox = tuple(float(blur_sigma_mm) / max(value, 1.0e-6) for value in spacing_mm)
        blurred = np.zeros_like(true_tensor, dtype=np.float32)
        for species_idx in range(true_tensor.shape[0]):
            blurred[species_idx] = ndimage.gaussian_filter(true_tensor[species_idx], sigma=sigma_vox, mode="constant", cval=0.0)
        uptake = _renormalize_species_mass(blurred, target_mass, support_mask=body)
    elif model_id == "peri_gtv_shell_sink_5_20mm":
        distance = ndimage.distance_transform_edt(~gtv, sampling=spacing_mm)
        shell = body & (~gtv) & (distance >= float(shell_inner_mm)) & (distance <= float(shell_outer_mm))
        if not np.any(shell):
            shell = body & (~gtv)
        uptake = _uniform_mass_tensor(target_mass, shell)
    elif model_id == "distance_decay_sink_lambda15mm":
        distance = ndimage.distance_transform_edt(~gtv, sampling=spacing_mm)
        support = body & (~gtv)
        weights = np.zeros_like(distance, dtype=np.float32)
        weights[support] = np.exp(-distance[support] / max(float(decay_lambda_mm), 1.0e-6)).astype(np.float32)
        uptake = _weighted_mass_tensor(target_mass, weights, support)
    else:
        raise ValueError(f"Unsupported sink model: {model_id}")

    uptake = np.asarray(uptake, dtype=np.float32)
    uptake[:, ~body] = 0.0
    uptake = _renormalize_species_mass(uptake, target_mass, support_mask=body)
    support_voxels = int(np.count_nonzero(np.any(uptake > 0.0, axis=0)))
    meta = {
        "model_id": str(model_id),
        "model_label": str(MODEL_LABELS[model_id]),
        "support_voxels": support_voxels,
        "target_mass_species0": float(target_mass[0]),
        "target_mass_species1": float(target_mass[1]),
    }
    return uptake, meta


def evaluate_custom_sink_model(
    *,
    physical_dose: np.ndarray,
    structures: Mapping[str, np.ndarray],
    axes_mm: Mapping[str, np.ndarray],
    spots_mm: Sequence[Tuple[float, float, float]],
    config: Mapping[str, object],
    uptake_tensor: np.ndarray,
    m_type: np.ndarray,
    m_oxygen: np.ndarray,
    voxel_volume_cc_value: float,
    voxel_size_mm: Tuple[float, float, float],
    prescription_gy: float,
    history_interval: int,
    progress_interval: int,
    verbose_pde: bool,
) -> Dict[str, object]:
    params = config["bio_parameters"]
    alpha = float(params["alpha"])
    beta = float(params["beta"])
    endpoint_masks = extract_endpoints(
        physical_dose,
        structures=structures,
        axes_mm=axes_mm,
        spots_mm=spots_mm,
        voxel_volume_cc=float(voxel_volume_cc_value),
        prescription_gy=float(prescription_gy),
    )[1]
    emission_tensor = build_emission_tensor(physical_dose, m_type=np.asarray(m_type, dtype=np.float32), m_oxygen=np.asarray(m_oxygen, dtype=np.float32))
    observables = solve_multispecies_pde_3d_with_hazard_observables(
        physical_dose,
        voxel_size_mm,
        diffusion_coeffs=(float(D_ROS), float(LOCKED_D_CYTO)),
        decay_coeffs=(float(LAMBDA_ROS), float(LOCKED_LAMBDA_CYTO)),
        emission_emax=(float(EMAX_ROS), float(EMAX_CYTO)),
        emission_gamma_per_gy=float(LOCKED_GAMMA),
        emission_tensor=emission_tensor,
        uptake_tensor=np.asarray(uptake_tensor, dtype=np.float32),
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
        voxel_volume_cc=float(voxel_volume_cc_value),
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


def endpoint_row_from_result(
    *,
    plan_id: str,
    case_label: str,
    site_group: str,
    model_id: str,
    result: Mapping[str, object],
) -> Dict[str, object]:
    endpoints = dict(result["endpoints"])
    supplemental = dict(result["supplemental"])
    row: Dict[str, object] = {
        "plan_id": plan_id,
        "case_label": case_label,
        "site_group": site_group,
        "model_id": model_id,
        "model_label": MODEL_LABELS[model_id],
    }
    row.update({key: float(value) for key, value in endpoints.items()})
    row.update(
        {
            "gtv_d95": float(supplemental["gtv_d95"]),
            "thyroid_mean": float(supplemental["thyroid_mean"]),
            "body_dmax": float(supplemental["body_dmax"]),
            "parotid_l_mean": float(supplemental["parotid_l_mean"]),
            "spill_shell_15_30_mean": float(supplemental["spill_shell_15_30_mean"]),
            "outside_gtv_d2": float(supplemental["outside_gtv_d2"]),
            "oar_adjacent_outside_gtv_mean": float(supplemental["oar_adjacent_outside_gtv_mean"]),
            "ptv_valley_outside_gtv_mean": float(supplemental["ptv_valley_outside_gtv_mean"]),
            "peak_mean": float(supplemental["peak_mean"]),
            "valley_mean": float(supplemental["valley_mean"]),
            "peak_voxels": int(supplemental["peak_voxels"]),
            "valley_voxels": int(supplemental["valley_voxels"]),
        }
    )
    return row


def assay_row_from_result(
    *,
    plan_id: str,
    case_label: str,
    site_group: str,
    model_id: str,
    result: Mapping[str, object],
) -> Dict[str, object]:
    row: Dict[str, object] = {
        "plan_id": plan_id,
        "case_label": case_label,
        "site_group": site_group,
        "model_id": model_id,
        "model_label": MODEL_LABELS[model_id],
    }
    row.update({key: float(value) for key, value in dict(result["assays"]).items()})
    return row


def rows_from_phase34(
    endpoint_rows: Sequence[Mapping[str, str]],
    assay_rows: Sequence[Mapping[str, str]],
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    mode_to_model = {
        "physical_only": "physical_only",
        "bystander_no_sink": "no_sink",
        "bystander_with_sink": "true_vessel_sink",
    }
    endpoint_out: List[Dict[str, object]] = []
    for row in endpoint_rows:
        model_id = mode_to_model[str(row["mode"])]
        converted: Dict[str, object] = {
            "plan_id": str(row["plan_id"]),
            "case_label": str(row["case_label"]),
            "site_group": str(row["site_group"]),
            "model_id": model_id,
            "model_label": MODEL_LABELS[model_id],
        }
        for key, value in row.items():
            if key in {"plan_id", "case_label", "site_group", "mode", "mode_label"}:
                continue
            try:
                converted[key] = float(value)
            except ValueError:
                converted[key] = value
        endpoint_out.append(converted)
    assay_out: List[Dict[str, object]] = []
    for row in assay_rows:
        model_id = mode_to_model[str(row["mode"])]
        converted = {
            "plan_id": str(row["plan_id"]),
            "case_label": str(row["case_label"]),
            "site_group": str(row["site_group"]),
            "model_id": model_id,
            "model_label": MODEL_LABELS[model_id],
        }
        for key, value in row.items():
            if key in {"plan_id", "case_label", "site_group", "mode", "mode_label"}:
                continue
            converted[key] = float(value)
        assay_out.append(converted)
    return endpoint_out, assay_out


def endpoint_value_better(endpoint_key: str, delta_true_minus_comp: float) -> bool:
    higher_is_better = bool(ENDPOINT_DIRECTION.get(endpoint_key, False))
    return float(delta_true_minus_comp) > 0.0 if higher_is_better else float(delta_true_minus_comp) < 0.0


def assay_value_better(assay_key: str, delta_true_minus_comp: float) -> bool:
    higher_is_better = bool(ASSAY_DIRECTION.get(assay_key, False))
    return float(delta_true_minus_comp) > 0.0 if higher_is_better else float(delta_true_minus_comp) < 0.0
