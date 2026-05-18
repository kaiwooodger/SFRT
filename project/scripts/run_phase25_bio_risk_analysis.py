#!/usr/bin/env python3
"""Phase 25: protocol-constrained random-plan biological risk analysis."""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from build_asymmetric_sweep import write_text_with_retries
from generate_detailed_headneck_topas_phantom import MATERIAL_SPECS
from run_linac_6mv_polyenergetic_clinical_sfrt import load_spectrum
from run_phase14_detailed_headneck_voxel_lattice import build_detailed_plan_phantom
from run_phase15_detailed_headneck_bioaware import build_args_from_summary, load_phase14_summary
from run_phase16_bio_guided_lattice_optimization import build_plan_args, evaluate_plan, point_from_index
from run_phase17_fraction_aware_bio_optimization import (
    build_clinical_gtv_core_candidate_centers,
    check_cumulative_constraints,
    compute_cumulative_course_summary,
)


@dataclass(frozen=True)
class EndpointSpec:
    key: str
    label: str
    units: str
    higher_is_better: bool


PRIMARY_ENDPOINTS: Tuple[EndpointSpec, ...] = (
    EndpointSpec("ptv_d95", "PTV D95", "Gy", True),
    EndpointSpec("pvdr", "PVDR", "ratio", True),
    EndpointSpec("spill_shell_0_5_mean", "Peri-GTV 0-5 mm", "Gy", False),
    EndpointSpec("spill_shell_5_15_mean", "Peri-GTV 5-15 mm", "Gy", False),
    EndpointSpec("cord_d2", "Cord D2", "Gy", False),
    EndpointSpec("brainstem_d2", "Brainstem D2", "Gy", False),
    EndpointSpec("parotid_r_mean", "Parotid R mean", "Gy", False),
)


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description=(
            "Generate five independent protocol-constrained random lattice plans, "
            "run each as a repeated five-fraction course, and summarize clinically "
            "interpretable biological risk-analysis endpoints."
        )
    )
    parser.add_argument("--phase-number", type=int, default=25)
    parser.add_argument(
        "--phase-description",
        type=str,
        default=(
            "Protocol-constrained random-plan biological risk analysis across five "
            "independent lattice courses with aligned physical-versus-biological endpoints."
        ),
    )
    parser.add_argument(
        "--baseline-run-root",
        type=Path,
        default=root / "runs" / "linac_6mv_headneck_detailed_voxel_lattice_sfrt_directplan",
    )
    parser.add_argument(
        "--run-root",
        type=Path,
        default=root / "runs" / "linac_6mv_headneck_detailed_voxel_lattice_sfrt_phase25_bio_risk_analysis",
    )
    parser.add_argument(
        "--template",
        type=Path,
        default=root / "topas" / "headneck_detailed_material_phantom_template.txt",
    )
    parser.add_argument(
        "--spectrum-csv",
        type=Path,
        default=root / "data" / "linac_6mv_representative_spectrum.csv",
    )
    parser.add_argument("--topas-bin", type=str, default="/Users/kw/shellScripts/topas")
    parser.add_argument("--g4-data-dir", type=str, default="/Applications/GEANT4")
    parser.add_argument("--physics-profile", type=str, default="em_opt4_only")
    parser.add_argument("--histories", type=int, default=100_000)
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
    parser.add_argument("--tumor-cytokine-multiplier", type=float, default=2.0)
    parser.add_argument("--hypoxic-ros-scale", type=float, default=0.12)
    parser.add_argument("--hypoxic-cytokine-multiplier", type=float, default=2.7)
    parser.add_argument("--artery-ros-uptake", type=float, default=0.05)
    parser.add_argument("--artery-cyto-uptake", type=float, default=0.70)
    parser.add_argument("--vein-ros-uptake", type=float, default=0.05)
    parser.add_argument("--vein-cyto-uptake", type=float, default=0.90)
    parser.add_argument("--fractions", type=int, default=5)
    parser.add_argument("--plan-count", type=int, default=5)
    parser.add_argument(
        "--plan-library-json",
        type=Path,
        default=None,
        help=(
            "Optional explicit plan-library JSON. When provided, Phase 25 uses these fixed "
            "plans instead of generating a random protocol-constrained library."
        ),
    )
    parser.add_argument("--num-spots", type=int, default=4)
    parser.add_argument("--spot-radius-mm", type=float, default=7.5)
    parser.add_argument("--base-margin-mm", type=float, default=6.0)
    parser.add_argument("--base-history-fraction", type=float, default=0.95)
    parser.add_argument("--base-min-ap-radius-mm", type=float, default=0.0)
    parser.add_argument("--base-min-lateral-radius-mm", type=float, default=0.0)
    parser.add_argument("--spot-ap-weight-scale", type=float, default=1.0)
    parser.add_argument("--spot-lateral-weight-scale", type=float, default=1.0)
    parser.add_argument("--superior-posterior-lateral-scale", type=float, default=1.0)
    parser.add_argument("--superior-threshold-mm", type=float, default=0.0)
    parser.add_argument("--posterior-threshold-mm", type=float, default=0.0)
    parser.add_argument("--lateral-radius-scale", type=float, default=1.0)
    parser.add_argument("--ap-spot-radius-scale", type=float, default=1.0)
    parser.add_argument("--candidate-step-mm", type=float, default=6.0)
    parser.add_argument("--min-spot-spacing-mm", type=float, default=18.0)
    parser.add_argument("--peak-radius-mm", type=float, default=8.0)
    parser.add_argument("--valley-exclusion-radius-mm", type=float, default=14.0)
    parser.add_argument("--spill-shell-1-mm", type=float, default=5.0)
    parser.add_argument("--spill-shell-2-mm", type=float, default=15.0)
    parser.add_argument("--spill-shell-3-mm", type=float, default=30.0)
    parser.add_argument("--spill-oar-adjacent-mm", type=float, default=15.0)
    parser.add_argument("--hard-cumulative-cord-d2-eff-gy", type=float, default=85.0)
    parser.add_argument("--hard-cumulative-brainstem-d2-eff-gy", type=float, default=30.0)
    parser.add_argument("--hard-cumulative-parotid-r-mean-eff-gy", type=float, default=60.0)
    parser.add_argument("--hard-cumulative-thyroid-mean-eff-gy", type=float, default=50.0)
    parser.add_argument("--hard-cumulative-body-dmax-phys-gy", type=float, default=400.0)
    parser.add_argument("--disable-body-hotspot-hard-constraint", action="store_true")
    parser.add_argument("--clinical-gtv-contraction-mm", type=float, default=5.0)
    parser.add_argument("--clinical-oar-clearance-mm", type=float, default=15.0)
    parser.add_argument("--clinical-inplane-pitch-mm", type=float, default=60.0)
    parser.add_argument("--clinical-layer-spacing-mm", type=float, default=30.0)
    parser.add_argument("--clinical-grid-max-snap-mm", type=float, default=18.0)
    parser.add_argument("--random-plan-attempts", type=int, default=6000)
    parser.add_argument("--random-library-size", type=int, default=120)
    parser.add_argument("--random-top-fraction", type=float, default=0.35)
    parser.add_argument("--random-plan-min-separation-mm", type=float, default=12.0)
    parser.add_argument("--allow-vertex-fallback", action="store_true")
    parser.add_argument("--dpi", type=int, default=180)
    parser.add_argument("--skip-existing", action="store_true")
    return parser.parse_args()


def write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        raise ValueError(f"No rows available for CSV output: {path}")
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: object) -> None:
    write_text_with_retries(path, json.dumps(payload, indent=2))


def placement_key(points_mm: Sequence[Tuple[float, float, float]]) -> Tuple[Tuple[float, float, float], ...]:
    return tuple(sorted(tuple(round(float(v), 2) for v in point) for point in points_mm))


def pairwise_distances(points_mm: Sequence[Tuple[float, float, float]]) -> List[float]:
    distances: List[float] = []
    for idx in range(len(points_mm)):
        for jdx in range(idx + 1, len(points_mm)):
            distances.append(float(math.dist(points_mm[idx], points_mm[jdx])))
    return distances


def spacing_valid(points_mm: Sequence[Tuple[float, float, float]], min_spacing_mm: float) -> bool:
    return all(distance >= float(min_spacing_mm) for distance in pairwise_distances(points_mm))


def plan_distance_mm(
    left: Sequence[Tuple[float, float, float]],
    right: Sequence[Tuple[float, float, float]],
) -> float:
    if not left or not right:
        return 0.0
    left_to_right = [min(math.dist(lpt, rpt) for rpt in right) for lpt in left]
    right_to_left = [min(math.dist(rpt, lpt) for lpt in left) for rpt in right]
    return float(0.5 * (np.mean(left_to_right) + np.mean(right_to_left)))


def compute_random_protocol_heuristic(
    points_mm: Sequence[Tuple[float, float, float]],
    *,
    clinical_inplane_pitch_mm: float,
    clinical_layer_spacing_mm: float,
) -> Dict[str, float]:
    pairwise = pairwise_distances(points_mm)
    if not pairwise:
        return {
            "heuristic_score": 0.0,
            "mean_pairwise_mm": 0.0,
            "min_pairwise_mm": 0.0,
            "same_layer_pitch_score": 0.0,
            "adjacent_layer_score": 0.0,
            "layer_count": float(len(points_mm)),
        }
    same_layer_pitch = 0.0
    adjacent_layer = 0.0
    for idx in range(len(points_mm)):
        for jdx in range(idx + 1, len(points_mm)):
            dx = float(points_mm[idx][0] - points_mm[jdx][0])
            dy = float(points_mm[idx][1] - points_mm[jdx][1])
            dz = abs(float(points_mm[idx][2] - points_mm[jdx][2]))
            inplane = math.hypot(dx, dy)
            if dz <= 0.45 * float(clinical_layer_spacing_mm):
                same_layer_pitch += math.exp(-((inplane - float(clinical_inplane_pitch_mm)) / 20.0) ** 2)
            else:
                adjacent_layer += math.exp(-((dz - float(clinical_layer_spacing_mm)) / 12.0) ** 2) * math.exp(
                    -((inplane - float(clinical_inplane_pitch_mm) / math.sqrt(2.0)) / 22.0) ** 2
                )
    z_values = np.asarray([point[2] for point in points_mm], dtype=np.float32)
    layer_count = float(len({round(float(z) / max(float(clinical_layer_spacing_mm), 1.0), 1) for z in z_values}))
    heuristic = (
        0.10 * float(np.mean(pairwise))
        + 0.20 * float(np.min(pairwise))
        + 6.0 * float(same_layer_pitch)
        + 4.0 * float(adjacent_layer)
        + 1.25 * float(layer_count)
    )
    return {
        "heuristic_score": float(heuristic),
        "mean_pairwise_mm": float(np.mean(pairwise)),
        "min_pairwise_mm": float(np.min(pairwise)),
        "same_layer_pitch_score": float(same_layer_pitch),
        "adjacent_layer_score": float(adjacent_layer),
        "layer_count": float(layer_count),
    }


def build_random_plan_library(
    *,
    args: argparse.Namespace,
    candidate_indices: Sequence[Tuple[int, int, int]],
    axes_mm: Dict[str, np.ndarray],
) -> List[Dict[str, object]]:
    rng = np.random.default_rng(int(args.seed))
    candidate_points = [tuple(float(v) for v in point_from_index(cand, axes_mm).tolist()) for cand in candidate_indices]
    if len(candidate_points) < 2:
        raise RuntimeError("Phase 25 found too few clinically feasible centers to build a random plan library.")

    target_vertex_counts = [int(args.num_spots)]
    if bool(args.allow_vertex_fallback):
        target_vertex_counts.extend([count for count in range(int(args.num_spots) - 1, 1, -1)])
    unique_rows: Dict[Tuple[Tuple[float, float, float], ...], Dict[str, object]] = {}
    attempts = max(int(args.random_plan_attempts), int(args.plan_count) * 400)

    for _ in range(attempts):
        vertex_count = int(rng.choice(target_vertex_counts))
        if len(candidate_points) < vertex_count:
            continue
        choice = rng.choice(len(candidate_points), size=vertex_count, replace=False)
        points = [candidate_points[int(idx)] for idx in choice]
        if not spacing_valid(points, float(args.min_spot_spacing_mm)):
            continue
        key = placement_key(points)
        if key in unique_rows:
            continue
        heuristic = compute_random_protocol_heuristic(
            points,
            clinical_inplane_pitch_mm=float(args.clinical_inplane_pitch_mm),
            clinical_layer_spacing_mm=float(args.clinical_layer_spacing_mm),
        )
        unique_rows[key] = {
            "spots_mm": [tuple(float(v) for v in point) for point in points],
            "heuristic_score": float(heuristic["heuristic_score"]),
            "layout_debug": {
                "strategy": "phase25_random_protocol",
                "vertex_count": int(vertex_count),
                **heuristic,
            },
        }
        if len(unique_rows) >= int(args.random_library_size):
            break

    library = sorted(unique_rows.values(), key=lambda row: float(row["heuristic_score"]), reverse=True)
    if len(library) < int(args.plan_count):
        raise RuntimeError(
            f"Phase 25 only built {len(library)} unique protocol-constrained random plans; "
            f"need at least {int(args.plan_count)}."
        )
    return library


def select_independent_plans(
    *,
    args: argparse.Namespace,
    library: Sequence[Dict[str, object]],
) -> List[Dict[str, object]]:
    rng = np.random.default_rng(int(args.seed) + 25_001)
    library_rows = list(library)
    top_count = max(int(args.plan_count) * 3, int(math.ceil(float(args.random_top_fraction) * len(library_rows))))
    top_rows = library_rows[: min(len(library_rows), max(top_count, int(args.plan_count)))]
    selected: List[Dict[str, object]] = []
    remaining = list(top_rows)
    while remaining and len(selected) < int(args.plan_count):
        if not selected:
            weights = np.asarray([float(row["heuristic_score"]) for row in remaining], dtype=np.float64)
            weights = np.maximum(weights - float(np.min(weights)) + 1.0e-3, 1.0e-3)
            weights /= float(np.sum(weights))
            chosen_idx = int(rng.choice(len(remaining), p=weights))
            selected.append(remaining.pop(chosen_idx))
            continue

        scored_options: List[Tuple[float, int]] = []
        for idx, row in enumerate(remaining):
            min_sep = min(plan_distance_mm(row["spots_mm"], chosen["spots_mm"]) for chosen in selected)
            if min_sep < float(args.random_plan_min_separation_mm):
                continue
            score = float(row["heuristic_score"]) + 0.30 * float(min_sep)
            scored_options.append((score, idx))
        if not scored_options:
            break
        scores = np.asarray([row[0] for row in scored_options], dtype=np.float64)
        scores = np.maximum(scores - float(np.min(scores)) + 1.0e-3, 1.0e-3)
        scores /= float(np.sum(scores))
        chosen_pair = scored_options[int(rng.choice(len(scored_options), p=scores))]
        selected.append(remaining.pop(int(chosen_pair[1])))

    if len(selected) < int(args.plan_count):
        for row in top_rows:
            if row in selected:
                continue
            selected.append(row)
            if len(selected) >= int(args.plan_count):
                break
    return selected[: int(args.plan_count)]


def load_explicit_plan_library(
    *,
    args: argparse.Namespace,
) -> List[Dict[str, object]]:
    if args.plan_library_json is None:
        raise ValueError("plan_library_json is required for explicit plan-library loading.")
    payload = json.loads(Path(args.plan_library_json).read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        rows = payload.get("plans", [])
    else:
        rows = payload
    if not isinstance(rows, list) or not rows:
        raise RuntimeError(f"Explicit plan library is empty: {args.plan_library_json}")

    library: List[Dict[str, object]] = []
    for idx, row in enumerate(rows, start=1):
        raw_spots = row.get("spots_mm") or row.get("vertices")
        if not raw_spots:
            raise RuntimeError(f"Explicit plan {idx} is missing 'spots_mm'/'vertices'.")
        spots_mm: List[Tuple[float, float, float]] = []
        for spot in raw_spots:
            if isinstance(spot, Mapping):
                spot_tuple = (
                    float(spot["x"]),
                    float(spot["y"]),
                    float(spot["z"]),
                )
            else:
                spot_tuple = tuple(float(v) for v in spot[:3])
            spots_mm.append(spot_tuple)
        heuristic = compute_random_protocol_heuristic(
            spots_mm,
            clinical_inplane_pitch_mm=float(args.clinical_inplane_pitch_mm),
            clinical_layer_spacing_mm=float(args.clinical_layer_spacing_mm),
        )
        library.append(
            {
                "spots_mm": spots_mm,
                "heuristic_score": float(row.get("heuristic_score", heuristic["heuristic_score"])),
                "layout_debug": {
                    "strategy": str(row.get("strategy", "phase25_explicit_safe_core_library")),
                    "vertex_count": int(len(spots_mm)),
                    "source_plan_id": str(row.get("plan_id", f"library_plan_{idx:02d}")),
                    **heuristic,
                    **dict(row.get("layout_debug") or {}),
                },
            }
        )
    if len(library) < int(args.plan_count):
        raise RuntimeError(
            f"Explicit plan library only contains {len(library)} plans; need at least {int(args.plan_count)}."
        )
    return library[: int(args.plan_count)]


def extract_primary_endpoint_values(course_summary: Mapping[str, object], *, domain: str) -> Dict[str, float]:
    if domain == "physical":
        metrics = course_summary["cumulative_physical_metrics"]
        pvdr = course_summary["pvdr_physical"]
        spill = course_summary["spill_physical_metrics"]
    elif domain == "biological":
        metrics = course_summary["cumulative_effective_metrics"]
        pvdr = course_summary["pvdr_effective"]
        spill = course_summary["spill_effective_metrics"]
    else:
        raise ValueError(f"Unknown domain: {domain}")
    return {
        "ptv_d95": float(metrics["PTV"]["d95_gy"]),
        "pvdr": float(pvdr["pvdr"]),
        "spill_shell_0_5_mean": float(spill.get("SPILL_SHELL_0_5", {}).get("mean_gy", 0.0)),
        "spill_shell_5_15_mean": float(spill.get("SPILL_SHELL_5_15", {}).get("mean_gy", 0.0)),
        "cord_d2": float(metrics["SPINAL_CORD"]["d2_gy"]),
        "brainstem_d2": float(metrics["BRAINSTEM"]["d2_gy"]),
        "parotid_r_mean": float(metrics["PAROTID_R"]["mean_gy"]),
    }


def compute_fraction_progression(
    *,
    args: argparse.Namespace,
    structures: Dict[str, np.ndarray],
    axes_mm: Dict[str, np.ndarray],
    voxel_volume_cc: float,
    voxel_size_mm: Tuple[float, float, float],
    single_fraction_result: Dict[str, object],
    fractions: int,
) -> Tuple[List[Dict[str, object]], int | None]:
    rows: List[Dict[str, object]] = []
    first_bio_failure_fraction: int | None = None
    for fraction_idx in range(1, int(fractions) + 1):
        repeated_sequence = [single_fraction_result] * int(fraction_idx)
        cumulative_physical = np.asarray(single_fraction_result["physical_dose"], dtype=np.float32) * np.float32(fraction_idx)
        summary = compute_cumulative_course_summary(
            args=args,
            structures=structures,
            axes_mm=axes_mm,
            voxel_volume_cc=float(voxel_volume_cc),
            voxel_size_mm=voxel_size_mm,
            cumulative_physical_dose=cumulative_physical,
            accepted_sequence=repeated_sequence,
        )
        constraints_ok, constraint_details = check_cumulative_constraints(args, summary)
        bio_values = extract_primary_endpoint_values(summary, domain="biological")
        phys_values = extract_primary_endpoint_values(summary, domain="physical")
        failed_constraints = [
            name
            for name, detail in constraint_details.items()
            if bool(detail.get("enabled", True)) and not bool(detail.get("passed", False))
        ]
        if first_bio_failure_fraction is None and failed_constraints:
            first_bio_failure_fraction = int(fraction_idx)
        row = {
            "fraction": int(fraction_idx),
            "bio_constraints_ok": "yes" if constraints_ok else "no",
            "bio_failed_constraints": "; ".join(failed_constraints),
        }
        for endpoint in PRIMARY_ENDPOINTS:
            row[f"{endpoint.key}_phys"] = float(phys_values[endpoint.key])
            row[f"{endpoint.key}_bio"] = float(bio_values[endpoint.key])
        row["thyroid_mean_phys"] = float(summary["cumulative_physical_metrics"]["THYROID"]["mean_gy"])
        row["thyroid_mean_bio"] = float(summary["cumulative_effective_metrics"]["THYROID"]["mean_gy"])
        rows.append(row)
    return rows, first_bio_failure_fraction


def endpoint_z_scores(values: Sequence[float], *, higher_is_better: bool) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return arr
    mean = float(arr.mean())
    std = float(arr.std(ddof=0))
    if std <= 1.0e-9:
        z = np.zeros_like(arr)
    else:
        z = (arr - mean) / std
    return -z if higher_is_better else z


def build_assessment_rows(
    plan_summaries: Sequence[Mapping[str, object]],
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    physical_rows: List[Dict[str, object]] = []
    biological_rows: List[Dict[str, object]] = []
    for summary in plan_summaries:
        physical_rows.append({endpoint.key: float(summary["physical_endpoints"][endpoint.key]) for endpoint in PRIMARY_ENDPOINTS})
        biological_rows.append({endpoint.key: float(summary["biological_endpoints"][endpoint.key]) for endpoint in PRIMARY_ENDPOINTS})
    physical_scores = np.zeros(len(plan_summaries), dtype=np.float64)
    biological_scores = np.zeros(len(plan_summaries), dtype=np.float64)
    for endpoint in PRIMARY_ENDPOINTS:
        phys = [row[endpoint.key] for row in physical_rows]
        bio = [row[endpoint.key] for row in biological_rows]
        physical_scores += endpoint_z_scores(phys, higher_is_better=endpoint.higher_is_better)
        biological_scores += endpoint_z_scores(bio, higher_is_better=endpoint.higher_is_better)

    phys_rank = np.argsort(np.argsort(physical_scores)) + 1
    bio_rank = np.argsort(np.argsort(biological_scores)) + 1
    ranking_rows: List[Dict[str, object]] = []
    for idx, summary in enumerate(plan_summaries):
        ranking_rows.append(
            {
                "plan_id": str(summary["plan_id"]),
                "physical_risk_score": float(physical_scores[idx]),
                "biological_risk_score": float(biological_scores[idx]),
                "physical_rank": int(phys_rank[idx]),
                "biological_rank": int(bio_rank[idx]),
                "rank_shift": int(int(bio_rank[idx]) - int(phys_rank[idx])),
            }
        )
    return ranking_rows, biological_rows


def plot_biological_heatmap(
    out_file: Path,
    plan_summaries: Sequence[Mapping[str, object]],
    *,
    dpi: int,
) -> None:
    values = np.asarray(
        [
            [float(summary["biological_endpoints"][endpoint.key]) for endpoint in PRIMARY_ENDPOINTS]
            for summary in plan_summaries
        ],
        dtype=np.float64,
    )
    risk_values = np.zeros_like(values)
    for col_idx, endpoint in enumerate(PRIMARY_ENDPOINTS):
        risk_values[:, col_idx] = endpoint_z_scores(values[:, col_idx], higher_is_better=endpoint.higher_is_better)

    fig, ax = plt.subplots(figsize=(11.0, 4.8), constrained_layout=True)
    image = ax.imshow(risk_values, cmap="RdYlBu_r", aspect="auto")
    ax.set_xticks(np.arange(len(PRIMARY_ENDPOINTS)))
    ax.set_xticklabels([f"{endpoint.label}\n({endpoint.units})" for endpoint in PRIMARY_ENDPOINTS], rotation=25, ha="right")
    ax.set_yticks(np.arange(len(plan_summaries)))
    ax.set_yticklabels([str(summary["plan_id"]) for summary in plan_summaries])
    ax.set_title("Phase 25: biological endpoint heatmap across independent lattice courses")
    for row_idx in range(values.shape[0]):
        for col_idx in range(values.shape[1]):
            ax.text(col_idx, row_idx, f"{values[row_idx, col_idx]:.1f}", ha="center", va="center", fontsize=8)
    cbar = fig.colorbar(image, ax=ax, shrink=0.92)
    cbar.set_label("Relative biological risk (within-endpoint z score)")
    fig.savefig(out_file, dpi=int(dpi))
    plt.close(fig)


def plot_domain_pairs(
    out_file: Path,
    plan_summaries: Sequence[Mapping[str, object]],
    *,
    dpi: int,
) -> None:
    figure, axes = plt.subplots(2, 4, figsize=(14.5, 8.2), constrained_layout=True)
    axes_list = list(axes.ravel())
    plan_labels = [str(summary["plan_id"]) for summary in plan_summaries]
    x_positions = np.arange(len(plan_summaries))
    colors = {"physical": "#1f77b4", "biological": "#d62728"}
    for axis, endpoint in zip(axes_list, PRIMARY_ENDPOINTS):
        phys = [float(summary["physical_endpoints"][endpoint.key]) for summary in plan_summaries]
        bio = [float(summary["biological_endpoints"][endpoint.key]) for summary in plan_summaries]
        axis.plot(x_positions, phys, marker="o", color=colors["physical"], linewidth=1.8, label="Physical only")
        axis.plot(x_positions, bio, marker="s", color=colors["biological"], linewidth=1.8, label="Bio-aware")
        axis.set_title(f"{endpoint.label} ({endpoint.units})")
        axis.set_xticks(x_positions)
        axis.set_xticklabels(plan_labels, rotation=20, ha="right")
        axis.grid(alpha=0.25)
    axes_list[-1].axis("off")
    handles, labels = axes_list[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", ncol=2, frameon=False)
    figure.suptitle("Phase 25: physical-versus-biological endpoint reinterpretation", fontsize=15)
    figure.savefig(out_file, dpi=int(dpi))
    plt.close(figure)


def plot_delta_heatmap(
    out_file: Path,
    plan_summaries: Sequence[Mapping[str, object]],
    *,
    dpi: int,
) -> None:
    deltas = np.asarray(
        [
            [
                100.0
                * (
                    float(summary["biological_endpoints"][endpoint.key]) - float(summary["physical_endpoints"][endpoint.key])
                )
                / max(abs(float(summary["physical_endpoints"][endpoint.key])), 1.0e-6)
                for endpoint in PRIMARY_ENDPOINTS
            ]
            for summary in plan_summaries
        ],
        dtype=np.float64,
    )
    vmax = max(5.0, float(np.max(np.abs(deltas))))
    fig, ax = plt.subplots(figsize=(11.0, 4.8), constrained_layout=True)
    image = ax.imshow(deltas, cmap="coolwarm", aspect="auto", vmin=-vmax, vmax=vmax)
    ax.set_xticks(np.arange(len(PRIMARY_ENDPOINTS)))
    ax.set_xticklabels([endpoint.label for endpoint in PRIMARY_ENDPOINTS], rotation=25, ha="right")
    ax.set_yticks(np.arange(len(plan_summaries)))
    ax.set_yticklabels([str(summary["plan_id"]) for summary in plan_summaries])
    ax.set_title("Phase 25: percent shift from physical-only to bio-aware endpoints")
    for row_idx in range(deltas.shape[0]):
        for col_idx in range(deltas.shape[1]):
            ax.text(col_idx, row_idx, f"{deltas[row_idx, col_idx]:+.0f}%", ha="center", va="center", fontsize=8)
    cbar = fig.colorbar(image, ax=ax, shrink=0.92)
    cbar.set_label("Percent change relative to physical-only endpoint")
    fig.savefig(out_file, dpi=int(dpi))
    plt.close(fig)


def plot_rank_and_failure_summary(
    out_file: Path,
    plan_summaries: Sequence[Mapping[str, object]],
    ranking_rows: Sequence[Mapping[str, object]],
    *,
    fractions: int,
    dpi: int,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.8), constrained_layout=True)
    left, right = axes
    x_positions = np.arange(len(ranking_rows))
    for idx, row in enumerate(ranking_rows):
        left.plot([0, 1], [float(row["physical_rank"]), float(row["biological_rank"])], color="#7f7f7f", linewidth=1.6)
        left.scatter([0], [float(row["physical_rank"])], color="#1f77b4", s=50)
        left.scatter([1], [float(row["biological_rank"])], color="#d62728", s=50)
        left.text(1.05, float(row["biological_rank"]), str(row["plan_id"]), va="center", fontsize=8)
    left.set_xlim(-0.25, 1.35)
    left.set_xticks([0, 1])
    left.set_xticklabels(["Physical only", "Bio-aware"])
    left.set_ylabel("Risk rank (1 = lowest)")
    left.set_title("Plan ranking shift once biology is included")
    left.invert_yaxis()
    left.grid(alpha=0.25, axis="y")

    def failure_fraction(summary: Mapping[str, object]) -> float:
        top_level = summary.get("first_bio_failure_fraction")
        if top_level is not None:
            return float(top_level)
        supplemental = summary.get("supplemental", {})
        if isinstance(supplemental, Mapping):
            nested = supplemental.get("first_bio_failure_fraction")
            if nested is not None:
                return float(nested)
        return float(fractions) + 0.5

    failure_values = [
        failure_fraction(summary)
        for summary in plan_summaries
    ]
    bar_colors = ["#d62728" if value <= float(fractions) else "#2ca02c" for value in failure_values]
    right.bar(x_positions, failure_values, color=bar_colors)
    right.set_xticks(x_positions)
    right.set_xticklabels([str(summary["plan_id"]) for summary in plan_summaries], rotation=20, ha="right")
    right.set_ylim(0.0, float(fractions) + 1.0)
    right.set_ylabel("Fraction")
    right.set_title("First biological constraint failure")
    right.axhline(float(fractions), color="#555555", linestyle="--", linewidth=1.0)
    for idx, value in enumerate(failure_values):
        label = f">{int(fractions)}" if value > float(fractions) else str(int(value))
        right.text(idx, value + 0.08, label, ha="center", va="bottom", fontsize=8)

    fig.savefig(out_file, dpi=int(dpi))
    plt.close(fig)


def save_phantom_context(path: Path, structures: Mapping[str, np.ndarray], axes_mm: Mapping[str, np.ndarray]) -> None:
    payload: Dict[str, np.ndarray] = {
        "axes_x_mm": np.asarray(axes_mm["x"], dtype=np.float32),
        "axes_y_mm": np.asarray(axes_mm["y"], dtype=np.float32),
        "axes_z_mm": np.asarray(axes_mm["z"], dtype=np.float32),
    }
    for name, mask in structures.items():
        payload[f"struct_{name}"] = np.asarray(mask, dtype=bool)
    np.savez_compressed(path, **payload)


def build_plan_output_rows(
    *,
    plan_id: str,
    spots_mm: Sequence[Tuple[float, float, float]],
    heuristic_score: float,
    physical_endpoints: Mapping[str, float],
    biological_endpoints: Mapping[str, float],
    supplemental: Mapping[str, float | int | None | str],
) -> Dict[str, object]:
    row: Dict[str, object] = {
        "plan_id": str(plan_id),
        "heuristic_score": float(heuristic_score),
        "spots_mm": "; ".join(f"({x:.1f},{y:.1f},{z:.1f})" for x, y, z in spots_mm),
        "vertex_count": int(len(spots_mm)),
    }
    for endpoint in PRIMARY_ENDPOINTS:
        row[f"{endpoint.key}_phys"] = float(physical_endpoints[endpoint.key])
        row[f"{endpoint.key}_bio"] = float(biological_endpoints[endpoint.key])
    row.update(supplemental)
    return row


def main() -> int:
    args = parse_args()
    args.run_root.mkdir(parents=True, exist_ok=True)
    args._cached_template_text = Path(args.template).read_text(encoding="utf-8")

    baseline_summary = load_phase14_summary(args.baseline_run_root.resolve())
    phantom_args = build_args_from_summary(baseline_summary)
    args.size_x_cm = float(baseline_summary["phantom"]["size_cm"][0])
    args.size_y_cm = float(baseline_summary["phantom"]["size_cm"][1])
    args.size_z_cm = float(baseline_summary["phantom"]["size_cm"][2])
    args.material_specs = MATERIAL_SPECS

    phantom = build_detailed_plan_phantom(phantom_args)
    structures = phantom["structures"]
    axes_mm = phantom["axes_mm"]
    phantom_meta = phantom["meta"]
    voxel_volume_cc = float(phantom_meta["voxel_volume_cc"])
    voxel_size_mm = tuple(float(v) for v in phantom_meta["voxel_size_mm"])

    plan_args = build_plan_args(args, phantom_meta)
    spectrum_energies, spectrum_weights = load_spectrum(args.spectrum_csv)
    candidate_indices, candidate_pool_debug = build_clinical_gtv_core_candidate_centers(
        structures=structures,
        axes_mm=axes_mm,
        spot_radius_mm=float(args.spot_radius_mm),
        candidate_step_mm=float(args.candidate_step_mm),
        contraction_mm=float(args.clinical_gtv_contraction_mm),
        oar_clearance_mm=float(args.clinical_oar_clearance_mm),
        min_pool_size=max(8, int(args.num_spots) * 2),
    )

    if args.plan_library_json is not None:
        plan_library = load_explicit_plan_library(args=args)
        selected_plans = list(plan_library[: int(args.plan_count)])
        library_mode = "explicit fixed safe-core library"
    else:
        plan_library = build_random_plan_library(args=args, candidate_indices=candidate_indices, axes_mm=axes_mm)
        selected_plans = select_independent_plans(args=args, library=plan_library)
        library_mode = "random protocol-constrained library"

    figures_dir = args.run_root / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    plans_dir = args.run_root / "plans"
    plans_dir.mkdir(parents=True, exist_ok=True)
    save_phantom_context(args.run_root / "phase25_phantom_context.npz", structures, axes_mm)

    library_rows: List[Dict[str, object]] = []
    for idx, row in enumerate(plan_library, start=1):
        debug = dict(row.get("layout_debug") or {})
        library_rows.append(
            {
                "library_rank": int(idx),
                "heuristic_score": float(row["heuristic_score"]),
                "vertex_count": int(len(row["spots_mm"])),
                "spots_mm": "; ".join(f"({x:.1f},{y:.1f},{z:.1f})" for x, y, z in row["spots_mm"]),
                "mean_pairwise_mm": float(debug.get("mean_pairwise_mm", 0.0)),
                "min_pairwise_mm": float(debug.get("min_pairwise_mm", 0.0)),
                "same_layer_pitch_score": float(debug.get("same_layer_pitch_score", 0.0)),
                "adjacent_layer_score": float(debug.get("adjacent_layer_score", 0.0)),
            }
        )
    write_csv(args.run_root / "phase25_random_plan_library.csv", library_rows)

    manifest_rows: List[Dict[str, object]] = []
    plan_summaries: List[Dict[str, object]] = []
    combined_rows: List[Dict[str, object]] = []

    for plan_idx, plan_candidate in enumerate(selected_plans, start=1):
        plan_id = f"plan{plan_idx:02d}"
        spots_mm = [tuple(float(v) for v in spot) for spot in plan_candidate["spots_mm"]]
        placement_strategy = str((plan_candidate.get("layout_debug") or {}).get("strategy", "phase25_random_protocol"))
        placement_name = f"{plan_id}_{placement_strategy}"
        single_fraction_result = evaluate_plan(
            args=args,
            plan_args=plan_args,
            phantom=phantom,
            spectrum_energies=spectrum_energies,
            spectrum_weights=spectrum_weights,
            placement_id=int(plan_idx),
            placement_name=placement_name,
            spot_centers_mm=spots_mm,
            reuse_existing_dose_csv=None,
        )

        final_cumulative_physical = np.asarray(single_fraction_result["physical_dose"], dtype=np.float32) * np.float32(args.fractions)
        repeated_sequence = [single_fraction_result] * int(args.fractions)
        course_summary = compute_cumulative_course_summary(
            args=args,
            structures=structures,
            axes_mm=axes_mm,
            voxel_volume_cc=voxel_volume_cc,
            voxel_size_mm=voxel_size_mm,
            cumulative_physical_dose=final_cumulative_physical,
            accepted_sequence=repeated_sequence,
        )
        final_constraints_ok, final_constraint_details = check_cumulative_constraints(args, course_summary)
        physical_endpoints = extract_primary_endpoint_values(course_summary, domain="physical")
        biological_endpoints = extract_primary_endpoint_values(course_summary, domain="biological")
        fraction_rows, first_bio_failure_fraction = compute_fraction_progression(
            args=args,
            structures=structures,
            axes_mm=axes_mm,
            voxel_volume_cc=voxel_volume_cc,
            voxel_size_mm=voxel_size_mm,
            single_fraction_result=single_fraction_result,
            fractions=int(args.fractions),
        )

        plan_dir = plans_dir / plan_id
        plan_dir.mkdir(parents=True, exist_ok=True)
        write_csv(plan_dir / "fraction_progression.csv", fraction_rows)
        np.savez_compressed(
            plan_dir / "course_volumes.npz",
            cumulative_physical_dose=np.asarray(course_summary["cumulative_physical_dose"], dtype=np.float32),
            cumulative_effective_dose=np.asarray(course_summary["cumulative_effective_dose"], dtype=np.float32),
            single_fraction_physical_dose=np.asarray(single_fraction_result["physical_dose"], dtype=np.float32),
            single_fraction_effective_dose=np.asarray(single_fraction_result["effective_dose"], dtype=np.float32),
            spot_centers_mm=np.asarray(spots_mm, dtype=np.float32),
        )

        supplemental = {
            "gtv_d95_phys": float(course_summary["cumulative_physical_metrics"]["GTV"]["d95_gy"]),
            "gtv_d95_bio": float(course_summary["cumulative_effective_metrics"]["GTV"]["d95_gy"]),
            "thyroid_mean_phys": float(course_summary["cumulative_physical_metrics"]["THYROID"]["mean_gy"]),
            "thyroid_mean_bio": float(course_summary["cumulative_effective_metrics"]["THYROID"]["mean_gy"]),
            "body_dmax_phys": float(course_summary["cumulative_physical_metrics"]["BODY"]["dmax_gy"]),
            "first_bio_failure_fraction": (
                int(first_bio_failure_fraction) if first_bio_failure_fraction is not None else None
            ),
            "final_constraints_ok": "yes" if final_constraints_ok else "no",
            "final_failed_constraints": "; ".join(
                [
                    name
                    for name, detail in final_constraint_details.items()
                    if bool(detail.get("enabled", True)) and not bool(detail.get("passed", False))
                ]
            ),
        }
        row = build_plan_output_rows(
            plan_id=plan_id,
            spots_mm=spots_mm,
            heuristic_score=float(plan_candidate["heuristic_score"]),
            physical_endpoints=physical_endpoints,
            biological_endpoints=biological_endpoints,
            supplemental=supplemental,
        )
        combined_rows.append(row)

        summary_payload = {
            "plan_id": plan_id,
            "placement_name": placement_name,
            "heuristic_score": float(plan_candidate["heuristic_score"]),
            "layout_debug": plan_candidate.get("layout_debug", {}),
            "spots_mm": [[float(a), float(b), float(c)] for a, b, c in spots_mm],
            "fractions": int(args.fractions),
            "first_bio_failure_fraction": (
                int(first_bio_failure_fraction) if first_bio_failure_fraction is not None else None
            ),
            "physical_endpoints": physical_endpoints,
            "biological_endpoints": biological_endpoints,
            "supplemental": supplemental,
            "final_constraint_details": final_constraint_details,
        }
        write_json(plan_dir / "phase25_plan_summary.json", summary_payload)
        manifest_rows.append(
            {
                "plan_id": plan_id,
                "plan_dir": str(plan_dir),
                "course_volumes_npz": str(plan_dir / "course_volumes.npz"),
                "summary_json": str(plan_dir / "phase25_plan_summary.json"),
            }
        )
        plan_summaries.append(summary_payload)

    ranking_rows, _ = build_assessment_rows(plan_summaries)
    rank_lookup = {str(row["plan_id"]): row for row in ranking_rows}
    for row in combined_rows:
        row.update(
            {
                "physical_risk_score": float(rank_lookup[str(row["plan_id"])]["physical_risk_score"]),
                "biological_risk_score": float(rank_lookup[str(row["plan_id"])]["biological_risk_score"]),
                "physical_rank": int(rank_lookup[str(row["plan_id"])]["physical_rank"]),
                "biological_rank": int(rank_lookup[str(row["plan_id"])]["biological_rank"]),
                "rank_shift": int(rank_lookup[str(row["plan_id"])]["rank_shift"]),
            }
        )

    write_csv(args.run_root / "phase25_endpoint_table.csv", combined_rows)
    write_csv(args.run_root / "phase25_plan_manifest.csv", manifest_rows)
    write_json(args.run_root / "phase25_plan_manifest.json", manifest_rows)
    write_json(
        args.run_root / "phase25_config.json",
        {
            "phase_number": int(args.phase_number),
            "phase_description": str(args.phase_description),
            "plan_count": int(args.plan_count),
            "fractions": int(args.fractions),
            "baseline_run_root": str(args.baseline_run_root.resolve()),
            "run_root": str(args.run_root.resolve()),
            "seed": int(args.seed),
            "histories": int(args.histories),
            "library_mode": str(library_mode),
            "plan_library_json": (
                str(args.plan_library_json.resolve()) if args.plan_library_json is not None else None
            ),
            "primary_endpoints": [
                {
                    "key": endpoint.key,
                    "label": endpoint.label,
                    "units": endpoint.units,
                    "higher_is_better": bool(endpoint.higher_is_better),
                }
                for endpoint in PRIMARY_ENDPOINTS
            ],
            "candidate_pool_debug": candidate_pool_debug,
            "bio_parameters": {
                "alpha": float(args.alpha),
                "beta": float(args.beta),
                "pde_steps": int(args.pde_steps),
                "pde_dt": float(args.pde_dt),
                "tumor_cytokine_multiplier": float(args.tumor_cytokine_multiplier),
                "hypoxic_ros_scale": float(args.hypoxic_ros_scale),
                "hypoxic_cytokine_multiplier": float(args.hypoxic_cytokine_multiplier),
                "artery_ros_uptake": float(args.artery_ros_uptake),
                "artery_cyto_uptake": float(args.artery_cyto_uptake),
                "vein_ros_uptake": float(args.vein_ros_uptake),
                "vein_cyto_uptake": float(args.vein_cyto_uptake),
            },
        },
    )

    write_json(args.run_root / "phase25_plan_summaries.json", plan_summaries)

    plot_biological_heatmap(figures_dir / "figure1_phase25_biological_endpoint_heatmap.png", plan_summaries, dpi=int(args.dpi))
    plot_domain_pairs(figures_dir / "figure2_phase25_physical_vs_biological_pairs.png", plan_summaries, dpi=int(args.dpi))
    plot_delta_heatmap(figures_dir / "figure3_phase25_endpoint_delta_heatmap.png", plan_summaries, dpi=int(args.dpi))
    plot_rank_and_failure_summary(
        figures_dir / "figure4_phase25_rank_shift_and_failure.png",
        plan_summaries,
        ranking_rows,
        fractions=int(args.fractions),
        dpi=int(args.dpi),
    )

    rank_shift_count = int(sum(1 for row in ranking_rows if int(row["rank_shift"]) != 0))
    median_plan_distance = 0.0
    if len(selected_plans) > 1:
        distances = []
        for idx in range(len(selected_plans)):
            for jdx in range(idx + 1, len(selected_plans)):
                distances.append(plan_distance_mm(selected_plans[idx]["spots_mm"], selected_plans[jdx]["spots_mm"]))
        if distances:
            median_plan_distance = float(np.median(np.asarray(distances, dtype=np.float64)))
    assessment_lines = [
        "# Phase 25 Quick Assessment",
        "",
        f"- Evaluated {len(selected_plans)} plans from the {library_mode}.",
        f"- Median inter-plan geometric separation: {median_plan_distance:.2f} mm.",
        f"- Plans with a physical-versus-biological rank shift: {rank_shift_count} / {len(ranking_rows)}.",
        "- This design is well matched to the current model because it compares broad plan-level biological liability rather than tiny optimizer score differences.",
        "- The most important readout is whether the bio-aware figures and endpoint table separate plans differently from the physical-only view.",
        "- If the bio-aware rankings and spill/OAR endpoints diverge from the physical-only rankings, the method is doing exactly what we want as a risk-analysis layer.",
    ]
    write_text_with_retries(args.run_root / "phase25_quick_assessment.md", "\n".join(assessment_lines) + "\n")

    print("=== PHASE 25 BIOLOGICAL RISK ANALYSIS COMPLETE ===")
    print(f"Independent plans evaluated: {len(plan_summaries)}")
    print(f"Run root: {args.run_root}")
    print(f"Endpoint table: {args.run_root / 'phase25_endpoint_table.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
