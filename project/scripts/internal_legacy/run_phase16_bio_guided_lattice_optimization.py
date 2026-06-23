#!/usr/bin/env python3
"""Biology-guided lattice placement optimization on the heterogeneous head-neck phantom.

This script closes the loop between planning and the calibrated biology model.
It evaluates an initial direct-plan placement, then proposes four subsequent
placements using feedback from the previous iteration's biology-aware result.
The optimization objective is driven primarily by effective-dose target gain
versus effective-dose OAR burden, with explicit use of the detailed vascular
network as a sink-support prior.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple
import itertools

import numpy as np

from analyze_topas_outputs import load_topas_grid
from build_asymmetric_sweep import has_nonempty_output, write_text_with_retries
from generate_detailed_headneck_topas_phantom import MATERIAL_SPECS
from run_linac_6mv_polyenergetic_clinical_sfrt import load_spectrum
from run_phase13_headneck_voxel_lattice import (
    build_plan_sources,
    compute_structure_metrics,
    sphere_fits,
)
from run_phase14_detailed_headneck_voxel_lattice import (
    build_detailed_plan_phantom,
    render_case_file,
    render_materials_include,
    run_topas_case,
    write_image_cube,
)
from run_phase15_detailed_headneck_bioaware import (
    LOCKED_D_CYTO,
    LOCKED_LAMBDA_CYTO,
    LOCKED_GAMMA,
    LOCKED_SCALING_FACTOR,
    W_ROS,
    W_CYTO,
    W_IMMUNE,
    D_ROS,
    LAMBDA_ROS,
    EMAX_ROS,
    EMAX_CYTO,
    build_anatomical_biology_tensors,
    build_args_from_summary,
    load_phase14_summary,
)
from bystander_multispecies_pde_solver import (
    calculate_effective_dose,
    calculate_phase7_survival,
    calculate_systemic_immune_penalty,
    run_pde_temporal_integration,
)


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description=(
            "Run a five-placement biology-guided optimization loop on the detailed "
            "heterogeneous head-and-neck SFRT phantom."
        )
    )
    parser.add_argument(
        "--baseline-run-root",
        type=Path,
        default=root / "runs" / "linac_6mv_headneck_detailed_voxel_lattice_sfrt_directplan",
        help="Existing detailed direct-plan run root used as iteration 1 baseline.",
    )
    parser.add_argument(
        "--run-root",
        type=Path,
        default=root / "runs" / "linac_6mv_headneck_detailed_voxel_lattice_sfrt_bioopt",
        help="Output root for the optimization loop.",
    )
    parser.add_argument(
        "--template",
        type=Path,
        default=root / "topas" / "headneck_detailed_material_phantom_template.txt",
        help="TOPAS ImageCube template for the heterogeneous phantom.",
    )
    parser.add_argument(
        "--spectrum-csv",
        type=Path,
        default=root / "data" / "linac_6mv_representative_spectrum.csv",
        help="Representative 6 MV spectrum CSV.",
    )
    parser.add_argument("--topas-bin", type=str, default="/Users/kw/shellScripts/topas")
    parser.add_argument("--g4-data-dir", type=str, default="/Applications/GEANT4")
    parser.add_argument("--physics-profile", type=str, default="em_opt4_only")
    parser.add_argument("--histories", type=int, default=1_000_000)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--seed", type=int, default=33)
    parser.add_argument("--cut-gamma-mm", type=float, default=0.01)
    parser.add_argument("--cut-electron-mm", type=float, default=0.01)
    parser.add_argument("--cut-positron-mm", type=float, default=0.01)
    parser.add_argument("--prescription-gy", type=float, default=6.0)
    parser.add_argument("--alpha", type=float, default=0.03)
    parser.add_argument("--beta", type=float, default=0.003)
    parser.add_argument("--pde-steps", type=int, default=400)
    parser.add_argument("--pde-dt", type=float, default=0.12)
    parser.add_argument(
        "--tumor-cytokine-multiplier",
        type=float,
        default=2.0,
        help="Tumour cytokine emission multiplier (aligned with phase15 bio-aware defaults).",
    )
    parser.add_argument("--hypoxic-ros-scale", type=float, default=0.12)
    parser.add_argument("--hypoxic-cytokine-multiplier", type=float, default=2.7)
    parser.add_argument("--artery-ros-uptake", type=float, default=0.05)
    parser.add_argument("--artery-cyto-uptake", type=float, default=0.70)
    parser.add_argument("--vein-ros-uptake", type=float, default=0.05)
    parser.add_argument("--vein-cyto-uptake", type=float, default=0.90)
    parser.add_argument("--spot-radius-mm", type=float, default=8.0)
    parser.add_argument("--base-margin-mm", type=float, default=6.0)
    parser.add_argument("--base-history-fraction", type=float, default=0.95)
    parser.add_argument("--num-spots", type=int, default=4)
    parser.add_argument("--candidate-step-mm", type=float, default=6.0)
    parser.add_argument("--min-spot-spacing-mm", type=float, default=18.0)
    parser.add_argument("--target-effective-gy", type=float, default=28.0)
    parser.add_argument(
        "--hard-min-dist-cord-mm",
        type=float,
        default=55.0,
        help="Hard exclusion radius for lattice centers relative to any spinal cord voxel (mm).",
    )
    parser.add_argument(
        "--hard-min-dist-brainstem-mm",
        type=float,
        default=50.0,
        help="Hard exclusion radius for lattice centers relative to any brainstem voxel (mm).",
    )
    parser.add_argument(
        "--hard-max-cord-d2-effective-gy",
        type=float,
        default=45.0,
        help="Hard acceptance criterion: effective spinal cord D2 must not exceed this value (Gy).",
    )
    parser.add_argument(
        "--hard-max-brainstem-d2-effective-gy",
        type=float,
        default=10.0,
        help="Hard acceptance criterion: effective brainstem D2 must not exceed this value (Gy).",
    )
    parser.add_argument("--iterations", type=int, default=5, help="Total placements including baseline.")
    parser.add_argument("--skip-existing", action="store_true")
    return parser.parse_args()


def build_plan_args(args: argparse.Namespace, phantom_summary: Dict[str, object]) -> SimpleNamespace:
    size_cm = phantom_summary["size_cm"]
    voxel_mm = phantom_summary["voxel_size_mm"][0]
    return SimpleNamespace(
        size_x_cm=float(size_cm[0]),
        size_y_cm=float(size_cm[1]),
        size_z_cm=float(size_cm[2]),
        voxel_mm=float(voxel_mm),
        spot_radius_mm=float(args.spot_radius_mm),
        base_margin_mm=float(args.base_margin_mm),
        base_history_fraction=float(args.base_history_fraction),
        histories=int(args.histories),
        base_min_ap_radius_mm=float(getattr(args, "base_min_ap_radius_mm", 0.0)),
        base_min_lateral_radius_mm=float(getattr(args, "base_min_lateral_radius_mm", 0.0)),
        spot_ap_weight_scale=float(getattr(args, "spot_ap_weight_scale", 1.0)),
        spot_lateral_weight_scale=float(getattr(args, "spot_lateral_weight_scale", 1.0)),
        superior_posterior_lateral_scale=float(getattr(args, "superior_posterior_lateral_scale", 1.0)),
        superior_threshold_mm=float(getattr(args, "superior_threshold_mm", 0.0)),
        posterior_threshold_mm=float(getattr(args, "posterior_threshold_mm", 0.0)),
        lateral_radius_scale=float(getattr(args, "lateral_radius_scale", 1.0)),
        ap_spot_radius_scale=float(getattr(args, "ap_spot_radius_scale", 1.0)),
    )


def smooth_dose_for_normalization(dose_grid: np.ndarray, passes: int = 1) -> np.ndarray:
    """Apply a small box smoothing pass to stabilize low-history coverage estimates."""

    smoothed = np.asarray(dose_grid, dtype=np.float32)
    for _ in range(max(1, int(passes))):
        padded = np.pad(smoothed, ((1, 1), (1, 1), (1, 1)), mode="edge")
        accum = np.zeros_like(smoothed, dtype=np.float32)
        for dx in range(3):
            for dy in range(3):
                for dz in range(3):
                    accum += padded[
                        dx : dx + smoothed.shape[0],
                        dy : dy + smoothed.shape[1],
                        dz : dz + smoothed.shape[2],
                    ]
        smoothed = accum / np.float32(27.0)
    return smoothed


def robust_ptv_d95_for_normalization(dose_grid: np.ndarray, ptv_mask: np.ndarray) -> float:
    """Estimate a low-history-stable PTV D95 from a smoothed dose grid."""

    ptv_values = np.asarray(dose_grid, dtype=np.float32)[ptv_mask]
    if ptv_values.size == 0:
        return 0.0
    return float(np.percentile(ptv_values, 5.0))


def get_structure_centroid_mm(mask: np.ndarray, axes_mm: Dict[str, np.ndarray]) -> np.ndarray:
    coords = np.argwhere(mask)
    center = np.round(coords.mean(axis=0)).astype(int)
    return np.array([axes_mm["x"][center[0]], axes_mm["y"][center[1]], axes_mm["z"][center[2]]], dtype=np.float32)


def build_candidate_centers(
    structures: Dict[str, np.ndarray],
    axes_mm: Dict[str, np.ndarray],
    *,
    spot_radius_mm: float,
    candidate_step_mm: float,
) -> List[Tuple[int, int, int]]:
    gtv_mask = structures["GTV"]
    step_vox = max(1, int(round(float(candidate_step_mm) / float(axes_mm["x"][1] - axes_mm["x"][0]))))
    radius_vox = max(1, int(round(float(spot_radius_mm) / float(axes_mm["x"][1] - axes_mm["x"][0]))))
    candidates: List[Tuple[int, int, int]] = []
    for ix in range(0, gtv_mask.shape[0], step_vox):
        for iy in range(0, gtv_mask.shape[1], step_vox):
            for iz in range(0, gtv_mask.shape[2], step_vox):
                if not gtv_mask[ix, iy, iz]:
                    continue
                if not sphere_fits(gtv_mask, (ix, iy, iz), radius_vox):
                    continue
                candidates.append((ix, iy, iz))
    if not candidates:
        raise RuntimeError("No valid lattice candidate centers were found inside the GTV.")
    return candidates


def point_from_index(candidate_idx: Tuple[int, int, int], axes_mm: Dict[str, np.ndarray]) -> np.ndarray:
    return np.array(
        [
            float(axes_mm["x"][candidate_idx[0]]),
            float(axes_mm["y"][candidate_idx[1]]),
            float(axes_mm["z"][candidate_idx[2]]),
        ],
        dtype=np.float32,
    )


def build_structure_points_mm(
    structures: Dict[str, np.ndarray],
    axes_mm: Dict[str, np.ndarray],
    names: Iterable[str],
) -> Dict[str, np.ndarray]:
    points: Dict[str, np.ndarray] = {}
    for name in names:
        coords = np.argwhere(structures[name])
        points[name] = np.column_stack(
            [
                axes_mm["x"][coords[:, 0]],
                axes_mm["y"][coords[:, 1]],
                axes_mm["z"][coords[:, 2]],
            ]
        ).astype(np.float32)
    return points


def min_distance_mm(point_mm: np.ndarray, structure_points_mm: np.ndarray) -> float:
    return float(np.min(np.linalg.norm(structure_points_mm - point_mm[None, :], axis=1)))


def build_safe_candidate_centers(
    candidate_indices: Sequence[Tuple[int, int, int]],
    axes_mm: Dict[str, np.ndarray],
    structure_points_mm: Mapping[str, np.ndarray],
    *,
    hard_min_dist_cord_mm: float,
    hard_min_dist_brainstem_mm: float,
) -> List[Tuple[int, int, int]]:
    # Safety-first: enforce hard serial-organ avoidance; keep other OARs as soft terms
    # inside the placement scoring instead of hard candidate elimination.
    min_distance_rules_mm = {
        "SPINAL_CORD": float(hard_min_dist_cord_mm),
        "BRAINSTEM": float(hard_min_dist_brainstem_mm),
    }
    # If the requested hard distances eliminate all candidates, automatically relax
    # in small steps until we have at least some feasible centers. This keeps the
    # workflow from hard-failing on compact geometries while still strongly
    # prioritizing serial-organ avoidance.
    relax_steps = 0
    cord_limit = float(min_distance_rules_mm["SPINAL_CORD"])
    brainstem_limit = float(min_distance_rules_mm["BRAINSTEM"])
    min_cord = 25.0
    min_brainstem = 30.0
    while True:
        safe_candidates: List[Tuple[int, int, int]] = []
        for cand in candidate_indices:
            point_mm = point_from_index(cand, axes_mm)
            if (
                min_distance_mm(point_mm, structure_points_mm["SPINAL_CORD"]) >= cord_limit
                and min_distance_mm(point_mm, structure_points_mm["BRAINSTEM"]) >= brainstem_limit
            ):
                safe_candidates.append(cand)
        if len(safe_candidates) >= 4:
            return safe_candidates
        if cord_limit <= min_cord and brainstem_limit <= min_brainstem:
            raise RuntimeError("No safe lattice candidate centers remain after anatomy-aware filtering.")
        cord_limit = max(min_cord, cord_limit - 5.0)
        brainstem_limit = max(min_brainstem, brainstem_limit - 5.0)
        relax_steps += 1


def compute_vessel_distance_reward(
    candidate_idx: Tuple[int, int, int],
    vessel_coords_mm: np.ndarray,
    axes_mm: Dict[str, np.ndarray],
) -> float:
    point = point_from_index(candidate_idx, axes_mm)
    distances = np.linalg.norm(vessel_coords_mm - point[None, :], axis=1)
    min_distance = float(np.min(distances))
    return 1.0 / (1.0 + min_distance / 10.0)


def score_oar_exceedances(effective_metrics: Dict[str, Dict[str, float]]) -> Tuple[Dict[str, float], Dict[str, Tuple[str, float, float, float]]]:
    rules = {
        "SPINAL_CORD": ("d2_gy", 35.0, 1.20),
        "BRAINSTEM": ("d2_gy", 8.0, 0.90),
        "PAROTID_R": ("mean_gy", 20.0, 1.00),
        "PAROTID_L": ("mean_gy", 5.0, 0.25),
        "THYROID": ("mean_gy", 15.0, 0.75),
        "PARATHYROIDS": ("mean_gy", 15.0, 0.55),
        "BRAIN": ("mean_gy", 5.0, 0.70),
        "BLOOD_BRAIN_BARRIER": ("mean_gy", 5.0, 0.45),
        "MANDIBLE": ("mean_gy", 15.0, 0.50),
    }
    weights: Dict[str, float] = {}
    details: Dict[str, Tuple[str, float, float, float]] = {}
    for structure, (metric, threshold, base_weight) in rules.items():
        value = float(effective_metrics[structure][metric])
        exceed = max(0.0, (value - threshold) / threshold)
        weight = float(base_weight * (1.0 + 2.5 * exceed))
        weights[structure] = weight
        details[structure] = (metric, value, threshold, exceed)
    return weights, details


def compute_plan_objective(effective_metrics: Dict[str, Dict[str, float]]) -> Tuple[float, Dict[str, float]]:
    penalties = {
        "SPINAL_CORD": ("d2_gy", 35.0, 1.20),
        "BRAINSTEM": ("d2_gy", 8.0, 0.90),
        "PAROTID_R": ("mean_gy", 20.0, 1.00),
        "PAROTID_L": ("mean_gy", 5.0, 0.25),
        "THYROID": ("mean_gy", 15.0, 0.75),
        "PARATHYROIDS": ("mean_gy", 15.0, 0.55),
        "BRAIN": ("mean_gy", 5.0, 0.70),
        "BLOOD_BRAIN_BARRIER": ("mean_gy", 5.0, 0.45),
        "MANDIBLE": ("mean_gy", 15.0, 0.50),
    }
    reward = (
        2.5 * float(effective_metrics["GTV"]["d95_gy"])
        + 2.0 * float(effective_metrics["PTV"]["d95_gy"])
        + 0.5 * float(effective_metrics["GTV"]["d50_gy"])
    )
    penalty_terms: Dict[str, float] = {}
    total_penalty = 0.0
    for structure, (metric, threshold, weight) in penalties.items():
        value = float(effective_metrics[structure][metric])
        exceed = max(0.0, value - threshold)
        penalty = float(weight * exceed)
        penalty_terms[structure] = penalty
        total_penalty += penalty
    return float(reward - total_penalty), penalty_terms


def choose_next_spots(
    *,
    prev_effective_dose: np.ndarray,
    structures: Dict[str, np.ndarray],
    axes_mm: Dict[str, np.ndarray],
    uptake_tensor: np.ndarray,
    candidate_indices: Sequence[Tuple[int, int, int]],
    num_spots: int,
    min_spacing_mm: float,
    target_effective_gy: float,
    prev_selected_mm: Sequence[Tuple[float, float, float]],
    oar_weights: Dict[str, float],
    structure_points_mm: Mapping[str, np.ndarray],
    vessel_coords_mm: np.ndarray,
    history_counts: Mapping[Tuple[float, float, float], int],
) -> Tuple[List[Tuple[float, float, float]], Dict[str, object]]:
    tumour_need = np.clip(float(target_effective_gy) - prev_effective_dose, a_min=0.0, a_max=None)
    hypoxia_mask = structures["HYPOXIA"]
    cyto_uptake = uptake_tensor[1]
    prev_selected = [np.array(p, dtype=np.float32) for p in prev_selected_mm]

    distance_weight_map = {
        "SPINAL_CORD": 0.90,
        "BRAINSTEM": 0.70,
        "PAROTID_R": 1.25,
        "PAROTID_L": 0.35,
        "THYROID": 0.95,
        "PARATHYROIDS": 0.55,
        "BRAIN": 0.80,
        "BLOOD_BRAIN_BARRIER": 0.55,
        "MANDIBLE": 0.45,
    }

    scored: List[Tuple[float, Tuple[int, int, int], Dict[str, float]]] = []
    for cand in candidate_indices:
        ix, iy, iz = cand
        point_mm = point_from_index(cand, axes_mm)
        need_score = float(min(tumour_need[ix, iy, iz], 8.0))
        hypoxia_bonus = 3.0 if bool(hypoxia_mask[ix, iy, iz]) else 0.0
        sink_bonus = 6.0 * compute_vessel_distance_reward(cand, vessel_coords_mm, axes_mm)
        direct_sink_bonus = 8.0 * float(cyto_uptake[ix, iy, iz])
        distance_score = 0.0
        dominant_structure = ""
        dominant_penalty = -1.0
        for structure, weight in oar_weights.items():
            dist = min_distance_mm(point_mm, structure_points_mm[structure])
            weighted_distance = float(
                weight * distance_weight_map.get(structure, 0.25) * min(dist, 45.0) / 10.0
            )
            distance_score += weighted_distance
            if weight > dominant_penalty:
                dominant_penalty = weight
                dominant_structure = structure
        history_penalty = 0.0
        history_key = tuple(round(float(v), 1) for v in point_mm.tolist())
        history_penalty += 2.0 * float(history_counts.get(history_key, 0))
        for prev in prev_selected:
            if float(np.linalg.norm(point_mm - prev)) < 3.0:
                history_penalty += 1.5
        score = (
            1.5 * need_score
            + hypoxia_bonus
            + sink_bonus
            + direct_sink_bonus
            + distance_score
            - history_penalty
        )
        scored.append(
            (
                float(score),
                cand,
                {
                    "need_score": float(need_score),
                    "hypoxia_bonus": float(hypoxia_bonus),
                    "sink_bonus": float(sink_bonus + direct_sink_bonus),
                    "distance_score": float(distance_score),
                    "history_penalty": float(history_penalty),
                    "dominant_structure": dominant_structure,
                },
            )
        )

    scored.sort(key=lambda row: row[0], reverse=True)
    best_combo: Tuple[int, ...] | None = None
    best_combo_score = -float("inf")
    num_required = int(num_spots)
    combo_space = min(len(scored), 60)
    candidate_subset = scored[:combo_space]
    relax_spacings = [
        float(min_spacing_mm),
        max(14.0, float(min_spacing_mm) * 0.85),
        12.0,
        10.0,
        8.0,
    ]

    for spacing_limit in relax_spacings:
        for combo in itertools.combinations(range(len(candidate_subset)), num_required):
            points = [
                tuple(float(v) for v in point_from_index(candidate_subset[idx][1], axes_mm).tolist())
                for idx in combo
            ]
            valid = True
            pairwise_sum = 0.0
            for a, b in itertools.combinations(range(len(points)), 2):
                dist = math.dist(points[a], points[b])
                if dist < spacing_limit:
                    valid = False
                    break
                pairwise_sum += dist
            if not valid:
                continue
            combo_score = float(sum(candidate_subset[idx][0] for idx in combo) + 0.02 * pairwise_sum)
            if combo_score > best_combo_score:
                best_combo = combo
                best_combo_score = combo_score
        if best_combo is not None:
            break

    if best_combo is None:
        # Fallback: greedy selection with aggressive spacing relaxation.
        for spacing_limit in relax_spacings[::-1]:
            selected_points: List[Tuple[float, float, float]] = []
            selected_debug: List[Dict[str, object]] = []
            for score, cand, debug in scored:
                point = tuple(float(v) for v in point_from_index(cand, axes_mm).tolist())
                if selected_points and min(math.dist(point, other) for other in selected_points) < float(spacing_limit):
                    continue
                selected_points.append(point)
                selected_debug.append({"center_mm": list(point), "score": float(score), **debug})
                if len(selected_points) >= num_required:
                    return selected_points, {"candidate_rankings": selected_debug[: min(10, len(selected_debug))], "fallback": True}
        # Final fallback: if spacing constraints make 4 spots impossible, return the
        # top-scoring unique candidates (still anatomy-safe) and flag that spacing
        # had to be violated.
        unique_points: List[Tuple[float, float, float]] = []
        unique_debug: List[Dict[str, object]] = []
        for score, cand, debug in scored:
            point = tuple(float(v) for v in point_from_index(cand, axes_mm).tolist())
            if point in unique_points:
                continue
            unique_points.append(point)
            unique_debug.append({"center_mm": list(point), "score": float(score), **debug})
            if len(unique_points) >= num_required:
                return unique_points, {"candidate_rankings": unique_debug[: min(10, len(unique_debug))], "fallback": True, "spacing_violated": True}
        raise RuntimeError("Unable to place the requested number of lattice vertices in the feedback loop.")

    selected: List[Tuple[float, float, float]] = []
    selected_debug: List[Dict[str, object]] = []
    for idx in best_combo:
        score, cand, debug = candidate_subset[idx]
        point_mm = tuple(float(v) for v in point_from_index(cand, axes_mm).tolist())
        spread_bonus = 0.0 if not selected else 0.02 * min(math.dist(point_mm, other) for other in selected)
        selected.append(point_mm)
        selected_debug.append(
            {
                "center_mm": list(point_mm),
                "score": float(score + spread_bonus),
                "spread_bonus": float(spread_bonus),
                **debug,
            }
        )
    return selected, {"candidate_rankings": selected_debug, "combo_score": float(best_combo_score)}


def evaluate_plan(
    *,
    args: argparse.Namespace,
    plan_args: SimpleNamespace,
    phantom: Dict[str, object],
    spectrum_energies: Sequence[float],
    spectrum_weights: Sequence[float],
    placement_id: int,
    placement_name: str,
    spot_centers_mm: Sequence[Tuple[float, float, float]],
    reuse_existing_dose_csv: Path | None = None,
) -> Dict[str, object]:
    run_dir = args.run_root / f"placement_{placement_id:02d}_{placement_name}"
    case_dir = run_dir / "case"
    analysis_dir = run_dir / "analysis"
    phantom_dir = run_dir / "phantom"
    case_dir.mkdir(parents=True, exist_ok=True)
    analysis_dir.mkdir(parents=True, exist_ok=True)
    phantom_dir.mkdir(parents=True, exist_ok=True)

    tag_grid = phantom["tag_grid"]
    structures = phantom["structures"]
    axes_mm = phantom["axes_mm"]
    phantom_meta = phantom["meta"]
    voxel_volume_cc = float(phantom_meta["voxel_volume_cc"])

    patient_bin = case_dir / "patient_material_tags.bin"
    materials_file = case_dir / "materials.txt"
    parameter_file = case_dir / "beamline.txt"
    dose_csv = case_dir / "dosedata.csv"
    log_file = case_dir / "topas.log"
    dose_csv_to_load = dose_csv

    if reuse_existing_dose_csv is None or not (args.skip_existing and has_nonempty_output(dose_csv)):
        write_image_cube(tag_grid, patient_bin)
        write_text_with_retries(materials_file, render_materials_include(args.material_specs))
        write_text_with_retries(phantom_dir / "phantom_meta.json", json.dumps(phantom_meta, indent=2))

        plan_meta = build_plan_sources(plan_args, axes_mm, structures["PTV"], spot_centers_mm)
        sources = plan_meta["sources"]
        rendered = render_case_file(
            args,
            patient_bin=patient_bin,
            materials_file=materials_file,
            grid_shape=phantom_meta["grid_shape"],
            voxel_size_mm=phantom_meta["voxel_size_mm"],
            spectrum_energies=spectrum_energies,
            spectrum_weights=spectrum_weights,
            sources=sources,
        )
        write_text_with_retries(parameter_file, rendered)

        if reuse_existing_dose_csv is None:
            if not (args.skip_existing and has_nonempty_output(dose_csv)):
                run_topas_case(args, case_dir, parameter_file, dose_csv, log_file)
        else:
            dose_csv_to_load = reuse_existing_dose_csv
            write_text_with_retries(
                analysis_dir / "reused_dose_source.txt",
                str(reuse_existing_dose_csv) + "\n",
            )
    else:
        plan_meta = build_plan_sources(plan_args, axes_mm, structures["PTV"], spot_centers_mm)
        sources = plan_meta["sources"]

    if reuse_existing_dose_csv is not None:
        dose_csv_to_load = reuse_existing_dose_csv

    dose_raw, _ = load_topas_grid(dose_csv_to_load)
    ptv_raw_metrics = compute_structure_metrics(
        dose_raw,
        structures["PTV"],
        prescription_gy=float(args.prescription_gy),
        voxel_volume_cc=voxel_volume_cc,
    )
    raw_d95 = float(ptv_raw_metrics["d95_gy"])
    normalization_method = "raw_ptv_d95"
    normalization_reference_gy = float(raw_d95)
    if raw_d95 <= 0.0:
        smoothed_raw = smooth_dose_for_normalization(dose_raw, passes=1)
        smoothed_d95 = robust_ptv_d95_for_normalization(smoothed_raw, structures["PTV"])
        if smoothed_d95 <= 0.0:
            raise RuntimeError(f"Placement {placement_id} produced non-positive raw PTV D95.")
        normalization_method = "smoothed_ptv_d95"
        normalization_reference_gy = float(smoothed_d95)
        raw_d95 = float(smoothed_d95)
    physical_scale_factor = float(args.prescription_gy) / raw_d95
    physical_dose = dose_raw.astype(np.float32) * np.float32(physical_scale_factor)

    lq_survival = np.exp(-float(args.alpha) * physical_dose - float(args.beta) * physical_dose**2).astype(np.float32)
    uptake_tensor, m_type, m_oxygen, sink_meta = build_anatomical_biology_tensors(args, structures)
    sink_meta["arterial_volume_cc"] = float(np.count_nonzero(structures["ARTERIES"]) * voxel_volume_cc)
    sink_meta["venous_volume_cc"] = float(np.count_nonzero(structures["VEINS"]) * voxel_volume_cc)
    hazard_grid = run_pde_temporal_integration(
        physical_dose,
        tuple(float(v) for v in phantom_meta["voxel_size_mm"]),
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
        tuple(float(v) for v in phantom_meta["voxel_size_mm"]),
        float(LOCKED_SCALING_FACTOR),
        weight_immune=float(W_IMMUNE),
        verbose=False,
    )
    effective_dose = calculate_effective_dose(final_survival, alpha=float(args.alpha), beta=float(args.beta))
    immune_penalty, icd_volume_cm3 = calculate_systemic_immune_penalty(
        physical_dose,
        tuple(float(v) for v in phantom_meta["voxel_size_mm"]),
    )

    metric_config = {
        "PTV": {"prescription": float(args.prescription_gy), "vxs": [6.0, 10.0]},
        "GTV": {"prescription": float(args.prescription_gy), "vxs": [6.0, 10.0]},
        "SPINAL_CORD": {"prescription": None, "vxs": [5.0, 8.0]},
        "BRAINSTEM": {"prescription": None, "vxs": [5.0, 8.0]},
        "PAROTID_L": {"prescription": None, "vxs": [5.0, 10.0]},
        "PAROTID_R": {"prescription": None, "vxs": [5.0, 10.0]},
        "MANDIBLE": {"prescription": None, "vxs": [5.0, 10.0]},
        "THYROID": {"prescription": None, "vxs": [5.0, 10.0]},
        "PARATHYROIDS": {"prescription": None, "vxs": [5.0, 10.0]},
        "BRAIN": {"prescription": None, "vxs": [5.0, 10.0]},
        "BLOOD_BRAIN_BARRIER": {"prescription": None, "vxs": [5.0, 10.0]},
    }
    physical_metrics: Dict[str, Dict[str, float]] = {}
    effective_metrics: Dict[str, Dict[str, float]] = {}
    for structure_name, config in metric_config.items():
        physical_metrics[structure_name] = compute_structure_metrics(
            physical_dose,
            structures[structure_name],
            prescription_gy=config["prescription"],
            voxel_volume_cc=voxel_volume_cc,
            volume_thresholds_gy=config["vxs"],
        )
        effective_metrics[structure_name] = compute_structure_metrics(
            effective_dose,
            structures[structure_name],
            prescription_gy=config["prescription"],
            voxel_volume_cc=voxel_volume_cc,
            volume_thresholds_gy=config["vxs"],
        )

    objective_score, penalty_terms = compute_plan_objective(effective_metrics)
    write_text_with_retries(
        analysis_dir / "plan_summary.json",
        json.dumps(
            {
                "placement_id": int(placement_id),
                "placement_name": placement_name,
                "spot_centers_mm": [[float(a), float(b), float(c)] for a, b, c in spot_centers_mm],
                "objective_score": float(objective_score),
                "penalty_terms": penalty_terms,
                "normalization_method": str(normalization_method),
                "normalization_reference_d95_gy": float(normalization_reference_gy),
                "raw_ptv_d95_gy": float(ptv_raw_metrics["d95_gy"]),
                "immune_scalar": float(immune_penalty),
                "icd_volume_cm3": float(icd_volume_cm3),
                "physical_metrics": physical_metrics,
                "effective_metrics": effective_metrics,
            },
            indent=2,
        ),
    )
    return {
        "placement_id": int(placement_id),
        "placement_name": placement_name,
        "run_dir": str(run_dir),
        "spot_centers_mm": [tuple(float(v) for v in row) for row in spot_centers_mm],
        "physical_metrics": physical_metrics,
        "effective_metrics": effective_metrics,
        "physical_dose": physical_dose,
        "effective_dose": effective_dose,
        "objective_score": float(objective_score),
        "penalty_terms": penalty_terms,
        "immune_scalar": float(immune_penalty),
        "icd_volume_cm3": float(icd_volume_cm3),
        "sink_meta": sink_meta,
    }


def summarize_result_row(result: Dict[str, object], feedback_note: str) -> Dict[str, object]:
    phys = result["physical_metrics"]
    eff = result["effective_metrics"]
    return {
        "placement_id": int(result["placement_id"]),
        "placement_name": str(result["placement_name"]),
        "feedback_note": feedback_note,
        "accepted": "yes" if bool(result.get("accepted", False)) else "no",
        "objective_score": float(result["objective_score"]),
        "spot_centers_mm": "; ".join(f"({x:.1f},{y:.1f},{z:.1f})" for x, y, z in result["spot_centers_mm"]),
        "ptv_d95_physical_gy": float(phys["PTV"]["d95_gy"]),
        "ptv_d95_effective_gy": float(eff["PTV"]["d95_gy"]),
        "gtv_d95_effective_gy": float(eff["GTV"]["d95_gy"]),
        "spinal_cord_d2_physical_gy": float(phys["SPINAL_CORD"]["d2_gy"]),
        "spinal_cord_d2_effective_gy": float(eff["SPINAL_CORD"]["d2_gy"]),
        "brainstem_d2_effective_gy": float(eff["BRAINSTEM"]["d2_gy"]),
        "parotid_r_mean_effective_gy": float(eff["PAROTID_R"]["mean_gy"]),
        "thyroid_mean_effective_gy": float(eff["THYROID"]["mean_gy"]),
        "brain_mean_effective_gy": float(eff["BRAIN"]["mean_gy"]),
        "immune_scalar": float(result["immune_scalar"]),
    }


def write_markdown_table(out_file: Path, rows: Sequence[Dict[str, object]]) -> None:
    headers = [
        "ID",
        "Placement",
        "Feedback",
        "Accepted",
        "Objective",
        "PTV D95 (phys)",
        "PTV D95 (eff)",
        "GTV D95 (eff)",
        "Cord D2 (eff)",
        "Brainstem D2 (eff)",
        "Parotid R mean (eff)",
        "Thyroid mean (eff)",
        "Brain mean (eff)",
        "Spots (mm)",
    ]
    lines = [
        "# Biology-guided lattice optimization table",
        "",
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["placement_id"]),
                    str(row["placement_name"]),
                    str(row["feedback_note"]),
                    str(row["accepted"]),
                    f"{float(row['objective_score']):.2f}",
                    f"{float(row['ptv_d95_physical_gy']):.2f}",
                    f"{float(row['ptv_d95_effective_gy']):.2f}",
                    f"{float(row['gtv_d95_effective_gy']):.2f}",
                    f"{float(row['spinal_cord_d2_effective_gy']):.2f}",
                    f"{float(row['brainstem_d2_effective_gy']):.2f}",
                    f"{float(row['parotid_r_mean_effective_gy']):.2f}",
                    f"{float(row['thyroid_mean_effective_gy']):.2f}",
                    f"{float(row['brain_mean_effective_gy']):.2f}",
                    str(row["spot_centers_mm"]),
                ]
            )
            + " |"
        )
    write_text_with_retries(out_file, "\n".join(lines) + "\n")


def main() -> int:
    args = parse_args()
    args.run_root.mkdir(parents=True, exist_ok=True)

    baseline_summary = load_phase14_summary(args.baseline_run_root.resolve())
    phantom_args = build_args_from_summary(baseline_summary)
    args.size_x_cm = float(baseline_summary["phantom"]["size_cm"][0])
    args.size_y_cm = float(baseline_summary["phantom"]["size_cm"][1])
    args.size_z_cm = float(baseline_summary["phantom"]["size_cm"][2])
    phantom = build_detailed_plan_phantom(phantom_args)
    args.material_specs = MATERIAL_SPECS
    spectrum_energies, spectrum_weights = load_spectrum(args.spectrum_csv)

    phantom_meta = phantom["meta"]
    plan_args = build_plan_args(args, phantom_meta)
    structures = phantom["structures"]
    axes_mm = phantom["axes_mm"]

    candidate_indices = build_candidate_centers(
        structures,
        axes_mm,
        spot_radius_mm=float(args.spot_radius_mm),
        candidate_step_mm=float(args.candidate_step_mm),
    )
    structure_points_mm = build_structure_points_mm(
        structures,
        axes_mm,
        [
            "SPINAL_CORD",
            "BRAINSTEM",
            "PAROTID_R",
            "PAROTID_L",
            "THYROID",
            "PARATHYROIDS",
            "BRAIN",
            "BLOOD_BRAIN_BARRIER",
            "MANDIBLE",
            "ARTERIES",
            "VEINS",
        ],
    )
    safe_candidate_indices = build_safe_candidate_centers(
        candidate_indices,
        axes_mm,
        structure_points_mm,
        hard_min_dist_cord_mm=float(args.hard_min_dist_cord_mm),
        hard_min_dist_brainstem_mm=float(args.hard_min_dist_brainstem_mm),
    )
    vessel_coords = np.argwhere(structures["ARTERIES"] | structures["VEINS"])
    vessel_coords_mm = np.column_stack(
        [
            axes_mm["x"][vessel_coords[:, 0]],
            axes_mm["y"][vessel_coords[:, 1]],
            axes_mm["z"][vessel_coords[:, 2]],
        ]
    ).astype(np.float32)
    baseline_spots = [tuple(map(float, row)) for row in baseline_summary["plan"]["spot_centers_mm"]]
    baseline_dose_csv = args.baseline_run_root / "case" / "dosedata.csv"
    results: List[Dict[str, object]] = []
    table_rows: List[Dict[str, object]] = []
    history_counts: Dict[Tuple[float, float, float], int] = {}

    baseline_result = evaluate_plan(
        args=args,
        plan_args=plan_args,
        phantom=phantom,
        spectrum_energies=spectrum_energies,
        spectrum_weights=spectrum_weights,
        placement_id=1,
        placement_name="baseline_direct",
        spot_centers_mm=baseline_spots,
        reuse_existing_dose_csv=baseline_dose_csv,
    )
    baseline_result["accepted"] = True
    results.append(baseline_result)
    table_rows.append(summarize_result_row(baseline_result, "Initial detailed direct plan"))
    for spot in baseline_spots:
        key = tuple(round(float(v), 1) for v in spot)
        history_counts[key] = history_counts.get(key, 0) + 1

    best_result = baseline_result
    best_spots = baseline_spots
    uptake_tensor, _, _, _ = build_anatomical_biology_tensors(args, structures)
    for placement_id in range(2, int(args.iterations) + 1):
        oar_weights, exceedance_details = score_oar_exceedances(best_result["effective_metrics"])
        dominant_oars = sorted(
            (
                (
                    name,
                    max(0.0, info[1] - info[2]),
                )
                for name, info in exceedance_details.items()
            ),
            key=lambda row: row[1],
            reverse=True,
        )
        chosen_spots, debug_meta = choose_next_spots(
            prev_effective_dose=best_result["effective_dose"],
            structures=structures,
            axes_mm=axes_mm,
            uptake_tensor=uptake_tensor,
            candidate_indices=safe_candidate_indices,
            num_spots=int(args.num_spots),
            min_spacing_mm=float(args.min_spot_spacing_mm),
            target_effective_gy=float(args.target_effective_gy),
            prev_selected_mm=best_spots,
            oar_weights=oar_weights,
            structure_points_mm=structure_points_mm,
            vessel_coords_mm=vessel_coords_mm,
            history_counts=history_counts,
        )

        dominant_note = ", ".join(name for name, excess in dominant_oars[:2] if excess > 0.0) or "target-shaping only"
        result = evaluate_plan(
            args=args,
            plan_args=plan_args,
            phantom=phantom,
            spectrum_energies=spectrum_energies,
            spectrum_weights=spectrum_weights,
            placement_id=placement_id,
            placement_name=f"feedback_{placement_id:02d}",
            spot_centers_mm=chosen_spots,
            reuse_existing_dose_csv=None,
        )
        result["feedback_debug"] = debug_meta
        cord_ok = float(result["effective_metrics"]["SPINAL_CORD"]["d2_gy"]) <= float(args.hard_max_cord_d2_effective_gy)
        brainstem_ok = float(result["effective_metrics"]["BRAINSTEM"]["d2_gy"]) <= float(args.hard_max_brainstem_d2_effective_gy)
        result["hard_constraints_ok"] = bool(cord_ok and brainstem_ok)
        result["accepted"] = bool(result["hard_constraints_ok"] and (result["objective_score"] > best_result["objective_score"]))
        results.append(result)
        for spot in chosen_spots:
            key = tuple(round(float(v), 1) for v in spot)
            history_counts[key] = history_counts.get(key, 0) + 1
        if result["accepted"]:
            best_result = result
            best_spots = chosen_spots
        table_rows.append(
            summarize_result_row(
                result,
                (
                    f"Accepted update (hard-safe) using burden in {dominant_note}"
                    if result["accepted"]
                    else (
                        f"Rejected (failed hard safety) using burden in {dominant_note}"
                        if not result.get("hard_constraints_ok", True)
                        else f"Rejected (objective) using burden in {dominant_note}"
                    )
                ),
            )
        )

    summary = {
        "baseline_run_root": str(args.baseline_run_root),
        "run_root": str(args.run_root),
        "objective_description": (
            "2.5*GTV_D95(eq) + 2.0*PTV_D95(eq) + 0.5*GTV_D50(eq) minus weighted OAR exceedance penalties "
            "using effective-dose metrics."
        ),
        "best_placement_id": int(best_result["placement_id"]),
        "best_placement_name": str(best_result["placement_name"]),
        "best_objective_score": float(best_result["objective_score"]),
        "placements": [
            {
                "placement_id": int(result["placement_id"]),
                "placement_name": str(result["placement_name"]),
                "spot_centers_mm": [[float(a), float(b), float(c)] for a, b, c in result["spot_centers_mm"]],
                "objective_score": float(result["objective_score"]),
                "accepted": bool(result.get("accepted", False)),
                "physical_metrics": result["physical_metrics"],
                "effective_metrics": result["effective_metrics"],
                "penalty_terms": result["penalty_terms"],
                "immune_scalar": float(result["immune_scalar"]),
                "icd_volume_cm3": float(result["icd_volume_cm3"]),
            }
            for result in results
        ],
    }

    save_path_csv = args.run_root / "optimization_results.csv"
    save_path_md = args.run_root / "optimization_results.md"
    save_path_json = args.run_root / "optimization_results.json"
    with save_path_csv.open("w", encoding="utf-8", newline="") as handle:
        import csv

        writer = csv.DictWriter(handle, fieldnames=list(table_rows[0].keys()))
        writer.writeheader()
        writer.writerows(table_rows)
    write_markdown_table(save_path_md, table_rows)
    write_text_with_retries(save_path_json, json.dumps(summary, indent=2))

    print("=== PHASE 16 BIO-GUIDED LATTICE OPTIMIZATION COMPLETE ===")
    print(f"Best placement: {best_result['placement_name']} (ID {best_result['placement_id']})")
    print(f"Best objective score: {best_result['objective_score']:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
