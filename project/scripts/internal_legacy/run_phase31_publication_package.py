#!/usr/bin/env python3
"""Phase 31: publication-oriented data package for the lattice risk-analysis study.

This phase converts the current project outputs into a manuscript-facing package:

1. freezes a supplementary library of 10 synthetic H&N lattice benchmark templates
2. repeats the Phase 30 TOPAS Yang-style photon benchmark across seeds/history levels
3. runs the biology model directly on the Phase 30 TOPAS dose
4. builds manuscript tables with locked endpoints plus assay proxies
5. estimates effect sizes, confidence bands, and rank robustness under uncertainty
6. writes a reproducibility manifest and a discussion draft framed as risk analysis
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
from scipy import ndimage

from bystander_multispecies_pde_solver import calculate_systemic_immune_penalty
from generate_phase11c_insilico_assays import calculate_gamma_h2ax_proxy
from run_phase28_yang2022_sinonasal_benchmark import (
    build_peak_valley_masks,
    build_structures,
    compute_effective_and_assays,
    make_axes,
    mesh,
    pvdr_metrics,
    sphere_mask,
    structure_metrics,
    vertex_centers,
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
    EndpointSpec("spill_shell_0_5_mean", "Peri-GTV 0-5 mm mean", "Gy", False),
    EndpointSpec("spill_shell_5_15_mean", "Peri-GTV 5-15 mm mean", "Gy", False),
    EndpointSpec("cord_d2", "Cord D2", "Gy", False),
    EndpointSpec("brainstem_d2", "Brainstem D2", "Gy", False),
    EndpointSpec("parotid_r_mean", "Parotid R mean", "Gy", False),
)


def endpoint_z_scores(values: Sequence[float], *, higher_is_better: bool) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    mean = float(arr.mean()) if arr.size else 0.0
    std = float(arr.std(ddof=0)) if arr.size else 0.0
    if std <= 1.0e-9:
        z = np.zeros_like(arr)
    else:
        z = (arr - mean) / std
    return -z if bool(higher_is_better) else z


ASSAY_KEYS: Tuple[str, ...] = (
    "mean_gammah2ax_peak",
    "mean_gammah2ax_valley",
    "mean_tunel_peak",
    "mean_tunel_valley",
    "cytokine_global_auc",
    "cytokine_peak_roi_auc",
    "cytokine_valley_roi_auc",
    "immune_scalar",
    "icd_volume_cm3",
)

PHASE30_REPEAT_METRIC_MAP = {
    "peripheral_target_d95_gy": "ptv_d95",
    "pvdr": "pvdr",
    "spill_shell_0_5_mean_gy": "spill_shell_0_5_mean",
    "spill_shell_5_15_mean_gy": "spill_shell_5_15_mean",
    "cord_d2_gy": "cord_d2",
    "brainstem_d2_gy": "brainstem_d2",
    "parotid_r_mean_gy": "parotid_r_mean",
}


BENCHMARK_ASSUMPTIONS: Tuple[str, ...] = (
    "Lattice boost subplan only: 15 Gy to vertices and 3.5 Gy to CTVboost = GTV + 5 mm in 1 fraction.",
    "Photon concept: 2 full VMAT arcs as the default Yang-like delivery concept.",
    "Vertex rule: keep each sphere wholly intratumoral; if a sphere would breach GTV, move or delete it.",
    "H&N pruning rule: if a vertex competes with carotid, jugular, spinal cord, brainstem, optic structures, cochlea, mandible, skull base, or skin, delete it rather than force symmetry.",
    "PET guidance is a secondary selector only and does not override OAR safety.",
    "These 10 geometries are synthetic benchmark templates, not published patient datasets.",
)


BENCHMARK_TEMPLATES: Tuple[Mapping[str, object], ...] = (
    {
        "template_id": "case01",
        "label": "Right sinonasal / maxillary bulky ellipsoid",
        "tumour_model": "ellipsoid",
        "dimensions_cm": (5.8, 4.8, 4.5),
        "estimated_gtv_cc": 66.0,
        "shape_note": "compact central mass in maxillary-sinonasal compartment",
        "vertex_diameter_cm_recommended": 1.0,
        "vertex_diameter_cm_min": 1.0,
        "vertex_diameter_cm_max": 1.0,
        "vertex_centres_cm": [(-1.4, 0.0, 0.8), (1.4, 0.0, -0.8)],
        "rationale": "Yang-like two-sphere layout for a compact midface tumour; keeps hot spots central and away from orbit/skull-base interfaces.",
        "use_when": "Tumour is mostly solid and not wrapping the orbit.",
        "avoid_when": "",
    },
    {
        "template_id": "case02",
        "label": "Left maxillary sinus crescent abutting orbit",
        "tumour_model": "crescent / banana",
        "dimensions_cm": (7.2, 5.0, 4.6),
        "estimated_gtv_cc": 87.0,
        "shape_note": "posterior-inferior bias to spare the orbital limb",
        "vertex_diameter_cm_recommended": 1.0,
        "vertex_diameter_cm_min": 1.0,
        "vertex_diameter_cm_max": 1.0,
        "vertex_centres_cm": [(-1.5, -0.6, 0.7), (1.2, 0.4, -0.8)],
        "rationale": "Non-symmetric two-sphere layout for a crescent wrapped toward the orbit.",
        "use_when": "There is enough posterior-inferior solid core.",
        "avoid_when": "Do not place a third vertex toward the superior-anterior orbital limb.",
    },
    {
        "template_id": "case03",
        "label": "Base-of-tongue / oropharynx bilobed central mass",
        "tumour_model": "bilobed",
        "dimensions_cm": (7.0, 5.8, 5.2),
        "estimated_gtv_cc": 111.0,
        "shape_note": "bulky midline BOT tumour with deep muscular core",
        "vertex_diameter_cm_recommended": 1.0,
        "vertex_diameter_cm_min": 1.0,
        "vertex_diameter_cm_max": 1.0,
        "vertex_centres_cm": [(-1.7, 0.0, 0.9), (1.7, 0.0, 0.9), (0.0, -0.8, -1.4)],
        "rationale": "Triangular central pattern avoiding lateral drift toward carotid spaces.",
        "use_when": "Bulky midline BOT tumour with enough deep core.",
        "avoid_when": "",
    },
    {
        "template_id": "case04",
        "label": "Laryngo-hypopharyngeal elongated cylinder",
        "tumour_model": "elongated cylinder",
        "dimensions_cm": (8.6, 4.2, 4.0),
        "estimated_gtv_cc": 76.0,
        "shape_note": "narrow geometry with central cranio-caudal axis",
        "vertex_diameter_cm_recommended": 1.0,
        "vertex_diameter_cm_min": 1.0,
        "vertex_diameter_cm_max": 1.0,
        "vertex_centres_cm": [(0.0, -0.4, 1.6), (0.0, 0.3, -1.6)],
        "rationale": "Superior-inferior stacking rather than lateral pairing in a narrow geometry.",
        "use_when": "Central elongated disease with limited safe lateral room.",
        "avoid_when": "Do not force lateral paired vertices near carotid/cord interfaces.",
    },
    {
        "template_id": "case05",
        "label": "Parapharyngeal / prestyloid deep-space bulky mass",
        "tumour_model": "asymmetric ellipsoid",
        "dimensions_cm": (7.5, 5.6, 5.2),
        "estimated_gtv_cc": 114.0,
        "shape_note": "central deep-space tripod layout",
        "vertex_diameter_cm_recommended": 1.0,
        "vertex_diameter_cm_min": 1.0,
        "vertex_diameter_cm_max": 1.0,
        "vertex_centres_cm": [(-1.5, -0.5, 1.2), (1.5, -0.4, 0.8), (0.0, 0.8, -1.3)],
        "rationale": "Tripod in the medial solid core while staying away from skull-base and carotid-adjacent extremes.",
        "use_when": "Lesion has enough medial solid core to support 3 vertices.",
        "avoid_when": "",
    },
    {
        "template_id": "case06",
        "label": "Buccal mucosa / cheek infiltrative crescent",
        "tumour_model": "superficial crescent",
        "dimensions_cm": (8.8, 6.0, 3.6),
        "estimated_gtv_cc": 100.0,
        "shape_note": "deep soft-tissue core only with superficial risk",
        "vertex_diameter_cm_recommended": 0.8,
        "vertex_diameter_cm_min": 0.8,
        "vertex_diameter_cm_max": 1.0,
        "vertex_centres_cm": [(-1.2, -0.4, 0.8), (1.2, 0.6, -0.7)],
        "rationale": "Two small deep-core vertices for crowded superficial cheek anatomy.",
        "use_when": "There is enough deep soft-tissue bulk away from skin and mandible.",
        "avoid_when": "Stay off skin and mandibular cortex.",
    },
    {
        "template_id": "case07",
        "label": "Oral tongue / floor-of-mouth horseshoe",
        "tumour_model": "horseshoe / wraparound",
        "dimensions_cm": (7.4, 6.4, 3.8),
        "estimated_gtv_cc": 94.0,
        "shape_note": "two superior-lateral deep vertices plus one posterior core vertex",
        "vertex_diameter_cm_recommended": 0.8,
        "vertex_diameter_cm_min": 0.8,
        "vertex_diameter_cm_max": 1.0,
        "vertex_centres_cm": [(-1.5, 0.0, 0.8), (1.5, 0.0, 0.8), (0.0, -0.9, -1.8)],
        "rationale": "Three-vertex deep tongue-body pattern with inferior-posterior core anchoring.",
        "use_when": "There is preserved deep tongue bulk.",
        "avoid_when": "Avoid anterior superficial floor-of-mouth and lingual cortex interfaces.",
    },
    {
        "template_id": "case08",
        "label": "Bulky nodal conglomerate level II-IV with central necrosis",
        "tumour_model": "irregular nodal mass",
        "dimensions_cm": (9.0, 6.6, 5.4),
        "estimated_gtv_cc": 168.0,
        "shape_note": "4-point deep-core lattice around, not inside, necrotic centre",
        "vertex_diameter_cm_recommended": 1.0,
        "vertex_diameter_cm_min": 1.0,
        "vertex_diameter_cm_max": 1.0,
        "vertex_centres_cm": [(-1.8, 0.0, 1.2), (1.8, 0.0, 1.2), (-0.8, -1.0, -1.2), (0.9, 1.0, -1.3)],
        "rationale": "Four viable-rim vertices around an irregular necrotic core.",
        "use_when": "Viable rim remains after shifting the centroid away from liquefactive centre.",
        "avoid_when": "Do not place a vertex into fully liquefactive central necrosis.",
    },
    {
        "template_id": "case09",
        "label": "Deep parotid / infratemporal fossa bulky mass",
        "tumour_model": "irregular ellipsoid",
        "dimensions_cm": (8.0, 6.0, 5.8),
        "estimated_gtv_cc": 146.0,
        "shape_note": "deep posterior-inferior bias away from skull-base foramina",
        "vertex_diameter_cm_recommended": 1.0,
        "vertex_diameter_cm_min": 1.0,
        "vertex_diameter_cm_max": 1.0,
        "vertex_centres_cm": [(-1.4, -0.6, 1.0), (1.4, -0.3, 0.9), (0.0, 0.9, -1.3)],
        "rationale": "Three deep-biased vertices in preserved soft-tissue bulk.",
        "use_when": "Mass preserves a deep soft-tissue core.",
        "avoid_when": "Stay off superficial parotid skin edge and skull-base foramina.",
    },
    {
        "template_id": "case10",
        "label": "Composite oropharynx + upper-neck very bulky tumour",
        "tumour_model": "giant composite volume",
        "dimensions_cm": (10.8, 8.0, 7.2),
        "estimated_gtv_cc": 326.0,
        "shape_note": "sparse 5-point central array for very large volume",
        "vertex_diameter_cm_recommended": 1.2,
        "vertex_diameter_cm_min": 1.2,
        "vertex_diameter_cm_max": 1.2,
        "vertex_centres_cm": [(0.0, 0.0, 0.0), (-2.2, 0.0, 1.4), (2.2, 0.0, 1.4), (-0.8, -1.8, -1.6), (1.0, 1.9, -1.7)],
        "rationale": "Sparse central five-point array borrowing from larger-volume lattice planning without automatically escalating to 6-8 vertices in H&N.",
        "use_when": "There is truly safe central bulk after OAR pruning.",
        "avoid_when": "Do not escalate to 6-8 vertices automatically in crowded H&N anatomy.",
    },
)


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase25-run-root",
        type=Path,
        default=root / "runs" / "linac_6mv_headneck_detailed_voxel_lattice_sfrt_phase25_safe_core_plan_library_1e5",
    )
    parser.add_argument(
        "--phase26-run-root",
        type=Path,
        default=root / "runs" / "linac_6mv_headneck_detailed_voxel_lattice_sfrt_phase25_safe_core_plan_library_1e5" / "phase26_vascular_sink_ablation",
    )
    parser.add_argument(
        "--phase28-run-root",
        type=Path,
        default=root / "runs" / "phase28_yang2022_sinonasal_benchmark",
    )
    parser.add_argument(
        "--phase30-run-root",
        type=Path,
        default=root / "runs" / "phase30_phase28_topas_true_lattice_delivery",
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        default=root / "runs" / "phase31_publication_package",
    )
    parser.add_argument("--phase30-repeat-seeds", type=int, nargs="+", default=[11, 22, 33])
    parser.add_argument("--phase30-repeat-history-scales", type=float, nargs="+", default=[0.5, 1.0, 2.0])
    parser.add_argument("--phase30-repeat-threads", type=int, default=4)
    parser.add_argument("--dose-smoothing-mm", type=float, default=6.0)
    parser.add_argument("--bootstrap-samples", type=int, default=4000)
    parser.add_argument("--bootstrap-seed", type=int, default=31)
    parser.add_argument("--history-interval", type=int, default=20)
    parser.add_argument("--progress-interval", type=int, default=100)
    parser.add_argument("--verbose-pde", action="store_true")
    parser.add_argument("--skip-phase30-repeats", action="store_true")
    return parser.parse_args()


def write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        raise ValueError(f"No rows for CSV output: {path}")
    fieldnames: List[str] = []
    for row in rows:
        for key in row.keys():
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


def load_csv_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def pairwise_distances_cm(points_cm: Sequence[Tuple[float, float, float]]) -> List[float]:
    distances: List[float] = []
    for a, b in itertools.combinations(points_cm, 2):
        distances.append(float(math.dist(a, b)))
    return distances


def scalar_summary(arr: Sequence[float]) -> Dict[str, float]:
    values = np.asarray(arr, dtype=np.float64)
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values, ddof=0)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "p2_5": float(np.percentile(values, 2.5)),
        "p97_5": float(np.percentile(values, 97.5)),
    }


def bootstrap_interval(
    values: Sequence[float],
    *,
    rng: np.random.Generator,
    samples: int,
    reducer=np.mean,
) -> Tuple[float, float]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return float("nan"), float("nan")
    draws = np.empty(int(samples), dtype=np.float64)
    for idx in range(int(samples)):
        sample = arr[rng.integers(0, arr.size, size=arr.size)]
        draws[idx] = float(reducer(sample))
    return float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def paired_cohens_dz(values: Sequence[float]) -> float:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size < 2:
        return float("nan")
    sd = float(np.std(arr, ddof=1))
    if sd <= 1.0e-9:
        return 0.0
    return float(np.mean(arr) / sd)


def build_template_rows() -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for template in BENCHMARK_TEMPLATES:
        centres = [tuple(map(float, point)) for point in template["vertex_centres_cm"]]
        spacings = pairwise_distances_cm(centres)
        rows.append(
            {
                "template_id": str(template["template_id"]),
                "label": str(template["label"]),
                "tumour_model": str(template["tumour_model"]),
                "dimensions_cm": " x ".join(f"{float(v):.1f}" for v in template["dimensions_cm"]),
                "estimated_gtv_cc": float(template["estimated_gtv_cc"]),
                "vertex_count": int(len(centres)),
                "vertex_diameter_cm_recommended": float(template["vertex_diameter_cm_recommended"]),
                "vertex_diameter_cm_min": float(template["vertex_diameter_cm_min"]),
                "vertex_diameter_cm_max": float(template["vertex_diameter_cm_max"]),
                "min_center_to_center_spacing_cm": float(min(spacings)) if spacings else float("nan"),
                "mean_center_to_center_spacing_cm": float(np.mean(spacings)) if spacings else float("nan"),
                "all_spacings_cm": json.dumps([round(v, 3) for v in spacings]),
                "vertex_centres_cm": json.dumps(centres),
                "shape_note": str(template["shape_note"]),
                "rationale": str(template["rationale"]),
                "use_when": str(template["use_when"]),
                "avoid_when": str(template["avoid_when"]),
            }
        )
    return rows


def write_template_markdown(out_file: Path, rows: Sequence[Mapping[str, object]]) -> None:
    lines = [
        "# Phase 31 synthetic H&N benchmark template library",
        "",
        "These templates are synthetic geometry benchmarks derived from the Yang-style lattice planning rules supplied in the study notes.",
        "",
        "## Common assumptions",
        "",
    ]
    for assumption in BENCHMARK_ASSUMPTIONS:
        lines.append(f"- {assumption}")
    lines.extend(["", "## Template summary", ""])
    for row in rows:
        lines.extend(
            [
                f"### {row['template_id']}: {row['label']}",
                f"- Tumour model: `{row['tumour_model']}`",
                f"- Bounding dimensions: `{row['dimensions_cm']} cm`",
                f"- Estimated GTV: `{float(row['estimated_gtv_cc']):.1f} cc`",
                f"- Vertices: `{int(row['vertex_count'])}`",
                f"- Recommended diameter: `{float(row['vertex_diameter_cm_recommended']):.1f} cm`",
                f"- Spacing summary: `{row['all_spacings_cm']}`",
                f"- Coordinates: `{row['vertex_centres_cm']}`",
                f"- Intent: {row['rationale']}",
            ]
        )
        if str(row["avoid_when"]).strip():
            lines.append(f"- Caution: {row['avoid_when']}")
        lines.append("")
    out_file.write_text("\n".join(lines) + "\n", encoding="utf-8")


def shell_distances(
    structures: Mapping[str, np.ndarray],
    axes: Mapping[str, np.ndarray],
) -> Tuple[np.ndarray, np.ndarray]:
    sampling = tuple(float(axes[key][1] - axes[key][0]) for key in ("x", "y", "z"))
    distance = ndimage.distance_transform_edt(~np.asarray(structures["GTV"], dtype=bool), sampling=sampling)
    outside_gtv = np.asarray(structures["BODY"], dtype=bool) & ~np.asarray(structures["GTV"], dtype=bool)
    return distance, outside_gtv


def summarize_phase28_like_mode(
    dose: np.ndarray,
    *,
    axes: Mapping[str, np.ndarray],
    structures: Mapping[str, np.ndarray],
    peak_mask: np.ndarray,
    valley_mask: np.ndarray,
    voxel_cc: float,
) -> Dict[str, float]:
    ctv = structure_metrics(dose, np.asarray(structures["CTVBOOST"], dtype=bool), float(voxel_cc), prescription=3.5)
    cord = structure_metrics(dose, np.asarray(structures["SPINAL_CORD"], dtype=bool), float(voxel_cc))
    brainstem = structure_metrics(dose, np.asarray(structures["BRAINSTEM"], dtype=bool), float(voxel_cc))
    parotid_r = structure_metrics(dose, np.asarray(structures["PAROTID_R"], dtype=bool), float(voxel_cc))
    gtv = structure_metrics(dose, np.asarray(structures["GTV"], dtype=bool), float(voxel_cc), prescription=3.5)
    vertices = []
    xg, yg, zg = mesh(axes)
    for center in vertex_centers():
        vertices.append(sphere_mask(xg, yg, zg, center, 5.0))
    pv = pvdr_metrics(dose, peak_mask, valley_mask, vertices)
    distance_to_gtv, outside_gtv = shell_distances(structures, axes)
    shell_0_5 = outside_gtv & (distance_to_gtv > 0.0) & (distance_to_gtv <= 5.0)
    shell_5_15 = outside_gtv & (distance_to_gtv > 5.0) & (distance_to_gtv <= 15.0)
    return {
        "peripheral_target_d95_gy": float(ctv["d95"]),
        "gtv_d95_gy": float(gtv["d95"]),
        "pvdr": float(pv["pvdr_mean"]),
        "peak_mean_gy": float(pv["peak_mean"]),
        "valley_mean_gy": float(pv["valley_mean"]),
        "spill_shell_0_5_mean_gy": float(np.mean(dose[shell_0_5])) if np.count_nonzero(shell_0_5) else 0.0,
        "spill_shell_5_15_mean_gy": float(np.mean(dose[shell_5_15])) if np.count_nonzero(shell_5_15) else 0.0,
        "cord_d2_gy": float(cord["d2"]),
        "brainstem_d2_gy": float(brainstem["d2"]),
        "parotid_r_mean_gy": float(parotid_r["mean"]),
    }


def summarize_phase28_physical_assays(
    physical_dose: np.ndarray,
    *,
    alpha: float,
    beta: float,
    peak_mask: np.ndarray,
    valley_mask: np.ndarray,
    voxel_size_mm: Tuple[float, float, float],
) -> Dict[str, float]:
    zero = np.zeros_like(physical_dose, dtype=np.float32)
    gamma = calculate_gamma_h2ax_proxy(physical_dose, zero, alpha=float(alpha), beta=float(beta))
    final_survival = np.exp(-float(alpha) * physical_dose - float(beta) * physical_dose**2).astype(np.float32)
    tunel = (1.0 - final_survival).astype(np.float32, copy=False)
    immune_scalar, icd_volume_cc = calculate_systemic_immune_penalty(physical_dose, voxel_size_mm)
    return {
        "mean_gammah2ax_peak": float(np.mean(gamma[peak_mask])),
        "mean_gammah2ax_valley": float(np.mean(gamma[valley_mask])),
        "mean_tunel_peak": float(np.mean(tunel[peak_mask])),
        "mean_tunel_valley": float(np.mean(tunel[valley_mask])),
        "mean_cytokine_final_peak": 0.0,
        "mean_cytokine_final_valley": 0.0,
        "cytokine_global_auc": 0.0,
        "cytokine_peak_roi_auc": 0.0,
        "cytokine_valley_roi_auc": 0.0,
        "immune_scalar": float(immune_scalar),
        "icd_volume_cm3": float(icd_volume_cc),
    }


def normalize_phase28_assays(assays: Mapping[str, float]) -> Dict[str, float]:
    return {
        "mean_gammah2ax_peak": float(assays["gammah2ax_peak_mean"]),
        "mean_gammah2ax_valley": float(assays["gammah2ax_valley_mean"]),
        "mean_tunel_peak": float(assays["tunel_peak_mean"]),
        "mean_tunel_valley": float(assays["tunel_valley_mean"]),
        "cytokine_global_auc": float(assays["cytokine_global_auc"]),
        "cytokine_peak_roi_auc": float(assays["cytokine_peak_auc"]),
        "cytokine_valley_roi_auc": float(assays["cytokine_valley_auc"]),
        "immune_scalar": float(assays["immune_scalar"]),
        "icd_volume_cm3": float(assays["icd_volume_cc"]),
    }


def load_phase30_combined_dose(run_root: Path) -> Tuple[np.ndarray, Mapping[str, object]]:
    summary = json.loads((run_root / "phase30_plan_summary.json").read_text(encoding="utf-8"))
    dose_npz = np.load(run_root / "phase30_combined_physical_dose.npz")
    return np.asarray(dose_npz["dose_gy"], dtype=np.float32), summary


def analyze_phase30_physical_repeat(run_root: Path) -> Dict[str, float]:
    dose, summary = load_phase30_combined_dose(run_root)
    voxel_mm = float(summary["voxel_mm"])
    axes = make_axes(voxel_mm)
    structures = build_structures(axes)
    peak_mask, valley_mask = build_peak_valley_masks(axes, structures, vertex_centers())
    voxel_cc = float(voxel_mm**3 / 1000.0)
    metrics = summarize_phase28_like_mode(
        dose,
        axes=axes,
        structures=structures,
        peak_mask=peak_mask,
        valley_mask=valley_mask,
        voxel_cc=voxel_cc,
    )
    metrics.update(
        {
            "seed": int(summary["component_outputs"]["base_component"]["sources"][0]["histories"])  # placeholder overwritten by caller
        }
    )
    return metrics


def build_phase30_bio_args(config: Mapping[str, object], *, voxel_mm: float, history_interval: int, progress_interval: int, verbose_pde: bool) -> SimpleNamespace:
    params = dict(config["bio_parameters"])
    return SimpleNamespace(
        voxel_mm=float(voxel_mm),
        pde_steps=int(params["pde_steps"]),
        pde_dt=float(params["pde_dt"]),
        alpha=float(params["alpha"]),
        beta=float(params["beta"]),
        history_interval=int(history_interval),
        progress_interval=int(progress_interval),
        verbose_pde=bool(verbose_pde),
    )


def evaluate_phase30_biology(
    *,
    phase30_run_root: Path,
    phase25_config: Mapping[str, object],
    history_interval: int,
    progress_interval: int,
    verbose_pde: bool,
) -> List[Dict[str, object]]:
    dose, summary = load_phase30_combined_dose(phase30_run_root)
    voxel_mm = float(summary["voxel_mm"])
    axes = make_axes(voxel_mm)
    structures = build_structures(axes)
    voxel_cc = float(voxel_mm**3 / 1000.0)
    peak_mask, valley_mask = build_peak_valley_masks(axes, structures, vertex_centers())
    vertex_voxels = int(np.count_nonzero(peak_mask))
    bio_args = build_phase30_bio_args(
        phase25_config,
        voxel_mm=voxel_mm,
        history_interval=int(history_interval),
        progress_interval=int(progress_interval),
        verbose_pde=bool(verbose_pde),
    )
    rows: List[Dict[str, object]] = []

    physical_metrics = summarize_phase28_like_mode(
        dose,
        axes=axes,
        structures=structures,
        peak_mask=peak_mask,
        valley_mask=valley_mask,
        voxel_cc=voxel_cc,
    )
    physical_assays = summarize_phase28_physical_assays(
        dose,
        alpha=float(bio_args.alpha),
        beta=float(bio_args.beta),
        peak_mask=peak_mask,
        valley_mask=valley_mask,
        voxel_size_mm=(voxel_mm, voxel_mm, voxel_mm),
    )
    row = {
        "dataset": "phase30_yang_topas_photon",
        "mode": "physical_only",
        "mode_label": "Physical-only",
        "target_label": "CTVBOOST D95",
        "vertex_voxels": vertex_voxels,
    }
    row.update(physical_metrics)
    row.update(physical_assays)
    rows.append(row)

    for mode_name, with_sink in (("bystander_no_sink", False), ("bystander_with_sink", True)):
        deff, assays, vessel = compute_effective_and_assays(
            dose,
            args=bio_args,
            axes=axes,
            structures=structures,
            peak_mask=peak_mask,
            valley_mask=valley_mask,
            with_sink=with_sink,
        )
        metrics = summarize_phase28_like_mode(
            deff,
            axes=axes,
            structures=structures,
            peak_mask=peak_mask,
            valley_mask=valley_mask,
            voxel_cc=voxel_cc,
        )
        row = {
            "dataset": "phase30_yang_topas_photon",
            "mode": mode_name,
            "mode_label": "Bystander, no vascular sink" if not with_sink else "Bystander, anatomical vascular sink",
            "target_label": "CTVBOOST D95",
            "vertex_voxels": vertex_voxels,
            "vessel_voxels": int(np.count_nonzero(vessel)),
        }
        row.update(metrics)
        row.update(normalize_phase28_assays(assays))
        rows.append(row)
    return rows


def run_phase30_repeats(
    args: argparse.Namespace,
    *,
    repo_root: Path,
) -> List[Dict[str, object]]:
    if bool(args.skip_phase30_repeats):
        return []
    base_summary = json.loads((args.phase30_run_root / "phase30_plan_summary.json").read_text(encoding="utf-8"))
    base_histories = int(base_summary["histories_base"])
    spot_histories = int(base_summary["histories_spot"])
    phase30_script = repo_root / "scripts" / "run_phase30_phase28_topas_true_lattice_delivery.py"
    repeat_root = args.out_root / "phase30_repeats"
    repeat_root.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, object]] = []
    for seed in [int(v) for v in args.phase30_repeat_seeds]:
        for scale in [float(v) for v in args.phase30_repeat_history_scales]:
            scale_label = str(scale).replace(".", "p")
            run_root = repeat_root / f"seed{seed}_hist{scale_label}"
            summary_file = run_root / "phase30_plan_summary.json"
            if not summary_file.exists():
                cmd = [
                    sys.executable,
                    str(phase30_script),
                    "--out-root",
                    str(run_root),
                    "--seed",
                    str(seed),
                    "--histories-base",
                    str(max(100, int(round(base_histories * scale)))),
                    "--histories-spot",
                    str(max(100, int(round(spot_histories * scale)))),
                    "--threads",
                    str(int(args.phase30_repeat_threads)),
                    "--dose-smoothing-mm",
                    str(float(args.dose_smoothing_mm)),
                ]
                result = subprocess.run(
                    cmd,
                    cwd=str(repo_root),
                    capture_output=True,
                    text=True,
                )
                (run_root / "phase31_repeat_stdout.log").write_text((result.stdout or "") + "\n", encoding="utf-8")
                (run_root / "phase31_repeat_stderr.log").write_text((result.stderr or "") + "\n", encoding="utf-8")
                if result.returncode != 0:
                    raise RuntimeError(
                        "Phase 30 repeat failed.\n"
                        f"Command: {' '.join(cmd)}\n"
                        f"Return code: {result.returncode}\n"
                        f"Stderr tail:\n{result.stderr[-2000:] if result.stderr else ''}"
                    )
            metrics = analyze_phase30_physical_repeat(run_root)
            metrics.update(
                {
                    "seed": int(seed),
                    "history_scale": float(scale),
                    "histories_base": int(max(100, int(round(base_histories * scale)))),
                    "histories_spot": int(max(100, int(round(spot_histories * scale)))),
                    "run_root": str(run_root),
                }
            )
            rows.append(metrics)
    return rows


def build_repeat_band_rows(repeat_rows: Sequence[Mapping[str, object]]) -> List[Dict[str, object]]:
    if not repeat_rows:
        return []
    metrics = [
        "peripheral_target_d95_gy",
        "pvdr",
        "spill_shell_0_5_mean_gy",
        "spill_shell_5_15_mean_gy",
        "cord_d2_gy",
        "brainstem_d2_gy",
        "parotid_r_mean_gy",
        "peak_mean_gy",
        "valley_mean_gy",
    ]
    rows: List[Dict[str, object]] = []
    for key in metrics:
        values = [float(row[key]) for row in repeat_rows]
        summary = scalar_summary(values)
        rows.append(
            {
                "metric": key,
                "n_repeats": int(len(values)),
                "mean": summary["mean"],
                "std": summary["std"],
                "min": summary["min"],
                "max": summary["max"],
                "p2_5": summary["p2_5"],
                "p97_5": summary["p97_5"],
                "coefficient_of_variation_pct": float(100.0 * summary["std"] / max(abs(summary["mean"]), 1.0e-6)),
            }
        )
    return rows


def build_biology_sigma_lookup(phase26_run_root: Path) -> Dict[str, float]:
    sample_path = phase26_run_root / "phase26_sensitivity_samples.csv"
    try:
        sample_rows = load_csv_rows(sample_path)
    except (OSError, TimeoutError) as exc:
        fallback_path = Path(__file__).resolve().parents[1] / "public_results" / "phase35_uncertainty" / "phase35_sink_delta_noise_table.csv"
        if fallback_path.exists():
            fallback_rows = load_csv_rows(fallback_path)
            sigmas: Dict[str, float] = {}
            for endpoint in PRIMARY_ENDPOINTS:
                values = [
                    float(row["bio_sigma_for_delta"])
                    for row in fallback_rows
                    if str(row.get("endpoint", "")) == endpoint.key and str(row.get("bio_sigma_for_delta", "")).strip() not in {"", "nan", "NaN"}
                ]
                sigmas[endpoint.key] = float(np.mean(values)) if values else 0.0
            if any(value > 0.0 for value in sigmas.values()):
                print(
                    "Warning: phase26_sensitivity_samples.csv was unreadable at "
                    f"{sample_path}; falling back to bundled historical Phase 35 "
                    f"bio_sigma values from {fallback_path}.",
                    file=sys.stderr,
                )
                return sigmas
        raise RuntimeError(
            "Unable to build biology sigma lookup because the Phase 26 sensitivity "
            f"samples file was unreadable at {sample_path} and no readable bundled "
            "Phase 35 fallback was available."
        ) from exc
    grouped: Dict[str, Dict[str, List[float]]] = {}
    for row in sample_rows:
        if row["metric_type"] != "endpoint":
            continue
        plan_id = str(row["plan_id"])
        metric_map = grouped.setdefault(plan_id, {})
        for endpoint in PRIMARY_ENDPOINTS:
            metric_map.setdefault(endpoint.key, []).append(float(row[endpoint.key]))
    sigmas: Dict[str, float] = {}
    for endpoint in PRIMARY_ENDPOINTS:
        per_plan_sd = []
        for metric_map in grouped.values():
            values = metric_map.get(endpoint.key, [])
            if len(values) >= 2:
                per_plan_sd.append(float(np.std(np.asarray(values, dtype=np.float64), ddof=0)))
        sigmas[endpoint.key] = float(np.mean(per_plan_sd)) if per_plan_sd else 0.0
    return sigmas


def build_mc_sigma_lookup(repeat_band_rows: Sequence[Mapping[str, object]]) -> Dict[str, float]:
    sigmas: Dict[str, float] = {}
    for row in repeat_band_rows:
        mapped = PHASE30_REPEAT_METRIC_MAP.get(str(row["metric"]))
        if mapped is not None:
            sigmas[mapped] = float(row["std"])
    return sigmas


def simulate_rank_robustness(
    endpoint_rows: Sequence[Mapping[str, object]],
    *,
    mc_sigma: Mapping[str, float],
    bio_sigma: Mapping[str, float],
    rng: np.random.Generator,
    samples: int,
) -> List[Dict[str, object]]:
    nominal_by_mode: Dict[str, List[Dict[str, object]]] = {}
    for row in endpoint_rows:
        nominal_by_mode.setdefault(str(row["mode"]), []).append({key: row[key] for key in row})
    counts: Dict[Tuple[str, str], Dict[str, float]] = {}
    for mode, rows in nominal_by_mode.items():
        for row in rows:
            counts[(str(row["plan_id"]), mode)] = {
                "same_rank": 0.0,
                "top_rank": 0.0,
                "rank_sum": 0.0,
                "rank_sq_sum": 0.0,
                "nominal_rank": float(row["rank"]),
            }
    for _ in range(int(samples)):
        sim_rows_by_mode: Dict[str, List[Dict[str, object]]] = {}
        for mode, rows in nominal_by_mode.items():
            sim_rows: List[Dict[str, object]] = []
            for row in rows:
                sim_row = {"plan_id": row["plan_id"], "mode": mode}
                for endpoint in PRIMARY_ENDPOINTS:
                    sigma = float(mc_sigma.get(endpoint.key, 0.0))
                    if mode == "bystander_with_sink":
                        sigma = math.sqrt(sigma**2 + float(bio_sigma.get(endpoint.key, 0.0)) ** 2)
                    value = float(row[endpoint.key])
                    if sigma > 0.0:
                        value = max(0.0, float(rng.normal(value, sigma)))
                    sim_row[endpoint.key] = value
                sim_rows.append(sim_row)
            risk_scores = np.zeros(len(sim_rows), dtype=np.float64)
            for endpoint in PRIMARY_ENDPOINTS:
                values = [float(sim_row[endpoint.key]) for sim_row in sim_rows]
                risk_scores += endpoint_z_scores(values, higher_is_better=endpoint.higher_is_better)
            order = np.argsort(np.argsort(risk_scores)) + 1
            for idx, sim_row in enumerate(sim_rows):
                sim_row["rank"] = int(order[idx])
            sim_rows_by_mode[mode] = sim_rows
        for mode, sim_rows in sim_rows_by_mode.items():
            for sim_row in sim_rows:
                key = (str(sim_row["plan_id"]), mode)
                bucket = counts[key]
                rank = float(sim_row["rank"])
                bucket["same_rank"] += float(rank == bucket["nominal_rank"])
                bucket["top_rank"] += float(rank == 1.0)
                bucket["rank_sum"] += rank
                bucket["rank_sq_sum"] += rank**2
    rows: List[Dict[str, object]] = []
    for (plan_id, mode), bucket in sorted(counts.items()):
        mean_rank = float(bucket["rank_sum"] / max(int(samples), 1))
        mean_sq = float(bucket["rank_sq_sum"] / max(int(samples), 1))
        rows.append(
            {
                "plan_id": plan_id,
                "mode": mode,
                "nominal_rank": int(bucket["nominal_rank"]),
                "rank_retention_probability": float(bucket["same_rank"] / max(int(samples), 1)),
                "top_rank_probability": float(bucket["top_rank"] / max(int(samples), 1)),
                "mean_simulated_rank": mean_rank,
                "rank_sd": float(max(mean_sq - mean_rank**2, 0.0) ** 0.5),
            }
        )
    return rows


def build_rank_effect_rows(
    rank_rows: Sequence[Mapping[str, object]],
    *,
    rng: np.random.Generator,
    bootstrap_samples: int,
) -> List[Dict[str, object]]:
    specs = (
        ("no_sink_vs_physical_shift", "No-sink rank minus physical rank"),
        ("with_sink_vs_physical_shift", "With-sink rank minus physical rank"),
        ("with_sink_vs_no_sink_shift", "With-sink rank minus no-sink rank"),
    )
    rows: List[Dict[str, object]] = []
    for key, label in specs:
        values = np.asarray([float(row[key]) for row in rank_rows], dtype=np.float64)
        abs_values = np.abs(values)
        signed_ci = bootstrap_interval(values, rng=rng, samples=int(bootstrap_samples), reducer=np.mean)
        abs_ci = bootstrap_interval(abs_values, rng=rng, samples=int(bootstrap_samples), reducer=np.mean)
        rows.append(
            {
                "metric": key,
                "label": label,
                "n": int(values.size),
                "mean_shift": float(np.mean(values)),
                "mean_shift_ci_lower": signed_ci[0],
                "mean_shift_ci_upper": signed_ci[1],
                "mean_absolute_shift": float(np.mean(abs_values)),
                "mean_absolute_shift_ci_lower": abs_ci[0],
                "mean_absolute_shift_ci_upper": abs_ci[1],
                "fraction_nonzero": float(np.mean(abs_values > 0.0)),
                "cohens_dz": float(paired_cohens_dz(values)),
            }
        )
    return rows


def build_sink_delta_effect_rows(
    delta_rows: Sequence[Mapping[str, object]],
    assay_rows: Sequence[Mapping[str, object]],
    *,
    rng: np.random.Generator,
    bootstrap_samples: int,
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for endpoint in PRIMARY_ENDPOINTS:
        values = np.asarray(
            [float(row["with_sink_minus_no_sink"]) for row in delta_rows if str(row["endpoint"]) == endpoint.key],
            dtype=np.float64,
        )
        ci = bootstrap_interval(values, rng=rng, samples=int(bootstrap_samples), reducer=np.mean)
        rows.append(
            {
                "metric_type": "endpoint",
                "metric": endpoint.key,
                "label": endpoint.label,
                "units": endpoint.units,
                "mean_with_sink_minus_no_sink": float(np.mean(values)),
                "ci_lower": ci[0],
                "ci_upper": ci[1],
                "cohens_dz": float(paired_cohens_dz(values)),
            }
        )
    assay_by_plan_mode: Dict[Tuple[str, str], Mapping[str, str]] = {
        (str(row["plan_id"]), str(row["mode"])): row for row in assay_rows
    }
    assay_metrics = (
        "mean_gammah2ax_peak",
        "mean_gammah2ax_valley",
        "mean_tunel_peak",
        "mean_tunel_valley",
        "cytokine_peak_roi_auc",
        "cytokine_valley_roi_auc",
        "immune_scalar",
    )
    for key in assay_metrics:
        values = []
        for plan_id in sorted({str(row["plan_id"]) for row in assay_rows}):
            no_sink = assay_by_plan_mode[(plan_id, "bystander_no_sink")]
            with_sink = assay_by_plan_mode[(plan_id, "bystander_with_sink")]
            values.append(float(with_sink[key]) - float(no_sink[key]))
        arr = np.asarray(values, dtype=np.float64)
        ci = bootstrap_interval(arr, rng=rng, samples=int(bootstrap_samples), reducer=np.mean)
        rows.append(
            {
                "metric_type": "assay",
                "metric": key,
                "label": key,
                "units": "a.u.",
                "mean_with_sink_minus_no_sink": float(np.mean(arr)),
                "ci_lower": ci[0],
                "ci_upper": ci[1],
                "cohens_dz": float(paired_cohens_dz(arr)),
            }
        )
    return rows


def build_phase25_mode_summary_rows(
    endpoint_rows: Sequence[Mapping[str, object]],
    assay_rows: Sequence[Mapping[str, object]],
) -> List[Dict[str, object]]:
    assay_lookup: Dict[Tuple[str, str], Mapping[str, object]] = {
        (str(row["plan_id"]), str(row["mode"])): row for row in assay_rows
    }
    rows: List[Dict[str, object]] = []
    for mode in ("physical_only", "bystander_no_sink", "bystander_with_sink"):
        mode_endpoints = [row for row in endpoint_rows if str(row["mode"]) == mode]
        combined_rows = [assay_lookup[(str(row["plan_id"]), mode)] for row in mode_endpoints]
        row = {
            "dataset": "phase25_safe_core_mean",
            "mode": mode,
            "mode_label": str(mode_endpoints[0]["mode_label"]),
            "target_label": "PTV D95",
            "n_cases": int(len(mode_endpoints)),
            "peripheral_target_d95_gy": float(np.mean([float(r["ptv_d95"]) for r in mode_endpoints])),
            "gtv_d95_gy": float(np.mean([float(r["gtv_d95"]) for r in mode_endpoints])),
            "pvdr": float(np.mean([float(r["pvdr"]) for r in mode_endpoints])),
            "peak_mean_gy": float(np.mean([float(r["peak_mean"]) for r in mode_endpoints])),
            "valley_mean_gy": float(np.mean([float(r["valley_mean"]) for r in mode_endpoints])),
            "spill_shell_0_5_mean_gy": float(np.mean([float(r["spill_shell_0_5_mean"]) for r in mode_endpoints])),
            "spill_shell_5_15_mean_gy": float(np.mean([float(r["spill_shell_5_15_mean"]) for r in mode_endpoints])),
            "cord_d2_gy": float(np.mean([float(r["cord_d2"]) for r in mode_endpoints])),
            "brainstem_d2_gy": float(np.mean([float(r["brainstem_d2"]) for r in mode_endpoints])),
            "parotid_r_mean_gy": float(np.mean([float(r["parotid_r_mean"]) for r in mode_endpoints])),
        }
        for key in ASSAY_KEYS:
            row[key] = float(np.mean([float(r[key]) for r in combined_rows]))
        rows.append(row)
    return rows


def build_reproducibility_manifest(
    *,
    args: argparse.Namespace,
    repeat_rows: Sequence[Mapping[str, object]],
    template_rows: Sequence[Mapping[str, object]],
) -> Dict[str, object]:
    root = Path(__file__).resolve().parents[1]
    return {
        "phase": 31,
        "description": "Publication-oriented package for lattice biological risk-analysis study.",
        "script": str(Path(__file__).resolve()),
        "repo_root": str(root),
        "input_roots": {
            "phase25_run_root": str(args.phase25_run_root.resolve()),
            "phase26_run_root": str(args.phase26_run_root.resolve()),
            "phase28_run_root": str(args.phase28_run_root.resolve()),
            "phase30_run_root": str(args.phase30_run_root.resolve()),
        },
        "output_root": str(args.out_root.resolve()),
        "phase30_repeat_settings": {
            "seeds": [int(v) for v in args.phase30_repeat_seeds],
            "history_scales": [float(v) for v in args.phase30_repeat_history_scales],
            "threads": int(args.phase30_repeat_threads),
            "dose_smoothing_mm": float(args.dose_smoothing_mm),
        },
        "bootstrap": {
            "samples": int(args.bootstrap_samples),
            "seed": int(args.bootstrap_seed),
        },
        "benchmark_template_count": int(len(template_rows)),
        "benchmark_templates_are_synthetic": True,
        "repeat_runs": [
            {
                "seed": int(row["seed"]),
                "history_scale": float(row["history_scale"]),
                "histories_base": int(row["histories_base"]),
                "histories_spot": int(row["histories_spot"]),
                "run_root": str(row["run_root"]),
            }
            for row in repeat_rows
        ],
        "script_dependencies": [
            str(root / "scripts" / "run_phase25_bio_risk_analysis.py"),
            str(root / "scripts" / "run_phase26_vascular_sink_ablation.py"),
            str(root / "scripts" / "run_phase28_yang2022_sinonasal_benchmark.py"),
            str(root / "scripts" / "run_phase30_phase28_topas_true_lattice_delivery.py"),
            str(root / "scripts" / "run_phase31_publication_package.py"),
        ],
    }


def write_discussion_draft(
    out_file: Path,
    *,
    rank_effect_rows: Sequence[Mapping[str, object]],
    sink_effect_rows: Sequence[Mapping[str, object]],
    phase30_bio_rows: Sequence[Mapping[str, object]],
    repeat_band_rows: Sequence[Mapping[str, object]],
) -> None:
    rank_lookup = {str(row["metric"]): row for row in rank_effect_rows}
    sink_lookup = {str(row["metric"]): row for row in sink_effect_rows}
    phase30_lookup = {str(row["mode"]): row for row in phase30_bio_rows}
    repeat_lookup = {str(row["metric"]): row for row in repeat_band_rows}
    lines = [
        "# Phase 31 discussion draft",
        "",
        "This study should be framed as a biological risk-analysis investigation rather than a clinical plan-optimization paper.",
        "The strongest result is that fixed, protocol-constrained lattice plans can change rank once non-local bystander burden is considered, even before any claim is made about deliverable TPS optimization.",
        "",
        "In the safe-core library, rank shifts persisted across modelling modes, with a mean absolute with-sink-versus-physical shift of "
        f"`{float(rank_lookup['with_sink_vs_physical_shift']['mean_absolute_shift']):.2f}` ranks "
        f"(95% bootstrap CI `{float(rank_lookup['with_sink_vs_physical_shift']['mean_absolute_shift_ci_lower']):.2f}` to "
        f"`{float(rank_lookup['with_sink_vs_physical_shift']['mean_absolute_shift_ci_upper']):.2f}`).",
        "This supports the claim that the model changes plan interpretation rather than simply rescaling endpoint magnitudes.",
        "",
        "The vascular sink should be described as a mechanistic modifier of diffusible non-local burden rather than a dominant driver of direct peak damage.",
        f"For example, the mean with-sink-minus-no-sink delta for peri-GTV 0-5 mm spill was `{float(sink_lookup['spill_shell_0_5_mean']['mean_with_sink_minus_no_sink']):.3f} Gy`, "
        f"while the corresponding delta for peak gammaH2AX was `{float(sink_lookup['mean_gammah2ax_peak']['mean_with_sink_minus_no_sink']):.4f}` a.u.",
        "That pattern is consistent with sink uptake acting mainly on the propagated biological field.",
        "",
        "The Yang-style benchmark remains important as an external anchor. The analytical benchmark already showed biological valley fill-in, and the direct TOPAS-derived photon case now supports the same qualitative conclusion.",
        f"In the TOPAS case, physical PVDR was `{float(phase30_lookup['physical_only']['pvdr']):.3f}` and the with-sink biological PVDR was `{float(phase30_lookup['bystander_with_sink']['pvdr']):.3f}`, "
        f"with biological peri-GTV 0-5 mm mean `{float(phase30_lookup['bystander_with_sink']['spill_shell_0_5_mean_gy']):.3f} Gy`.",
        "This strengthens the interpretation that the biological signal is not solely an artefact of the analytical surrogate.",
        "",
    ]
    if repeat_lookup:
        lines.extend(
            [
                "The repeated Phase 30 TOPAS runs provide a practical uncertainty band for the physical benchmark. "
                f"The repeated-run coefficient of variation for PVDR was `{float(repeat_lookup['pvdr']['coefficient_of_variation_pct']):.2f}%`, "
                f"and for peri-GTV 0-5 mm spill it was `{float(repeat_lookup['spill_shell_0_5_mean_gy']['coefficient_of_variation_pct']):.2f}%`.",
                "These bands should be used to justify that downstream biological reinterpretation is being judged against a measured Monte Carlo noise scale rather than against an unrealistically exact physical baseline.",
                "",
            ]
        )
    else:
        lines.extend(
            [
                "The final publication build should include the repeated Phase 30 TOPAS runs so the physical benchmark is accompanied by an explicit Monte Carlo uncertainty band.",
                "",
            ]
        )
    lines.extend(
        [
            "The Discussion should explicitly avoid claiming clinical VMAT equivalence, deliverable patient-specific planning, or experimental validation of the assay proxies.",
            "Instead, it should claim that protocol-constrained lattice plans may carry hidden biological liability outside the tumour core, and that the current framework provides a structured way to quantify that liability and compare plans on that basis.",
            "",
            "A fair concluding statement is that this is a benchmark-anchored, hypothesis-generating biological risk-analysis framework for lattice RT, suitable for retrospective plan reinterpretation and for prioritizing which candidate plans warrant deeper physics or experimental follow-up.",
        ]
    )
    out_file.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_quick_assessment(
    out_file: Path,
    *,
    rank_effect_rows: Sequence[Mapping[str, object]],
    phase30_bio_rows: Sequence[Mapping[str, object]],
    repeat_rows: Sequence[Mapping[str, object]],
) -> None:
    rank_lookup = {str(row["metric"]): row for row in rank_effect_rows}
    phase30_lookup = {str(row["mode"]): row for row in phase30_bio_rows}
    lines = [
        "# Phase 31 quick assessment",
        "",
        f"- Frozen synthetic benchmark library: `{len(BENCHMARK_TEMPLATES)}` H&N lattice templates.",
        f"- Phase 30 repeated TOPAS runs: `{len(repeat_rows)}` combinations of seed/history level.",
        f"- Mean absolute with-sink vs physical rank shift: `{float(rank_lookup['with_sink_vs_physical_shift']['mean_absolute_shift']):.2f}`.",
        f"- Phase 30 physical PVDR: `{float(phase30_lookup['physical_only']['pvdr']):.3f}`.",
        f"- Phase 30 with-sink biological PVDR: `{float(phase30_lookup['bystander_with_sink']['pvdr']):.3f}`.",
        f"- Phase 30 with-sink cytokine valley AUC: `{float(phase30_lookup['bystander_with_sink']['cytokine_valley_roi_auc']):.2f}`.",
        "",
        "Interpretation: the current data are strongest as a benchmark-anchored biological risk-analysis package, not as a clinical TPS-equivalent optimization result.",
    ]
    out_file.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    out_root = args.out_root.resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(int(args.bootstrap_seed))

    template_rows = build_template_rows()
    write_csv(out_root / "phase31_benchmark_template_library.csv", template_rows)
    write_json(out_root / "phase31_benchmark_template_library.json", list(template_rows))
    write_template_markdown(out_root / "phase31_benchmark_template_library.md", template_rows)

    repeat_rows = run_phase30_repeats(args, repo_root=Path(__file__).resolve().parents[1])
    if repeat_rows:
        write_csv(out_root / "phase31_phase30_repeat_summary.csv", repeat_rows)
    repeat_band_rows = build_repeat_band_rows(repeat_rows)
    if repeat_band_rows:
        write_csv(out_root / "phase31_phase30_repeat_bands.csv", repeat_band_rows)

    phase25_config = json.loads((args.phase25_run_root / "phase25_config.json").read_text(encoding="utf-8"))
    phase30_bio_rows = evaluate_phase30_biology(
        phase30_run_root=args.phase30_run_root.resolve(),
        phase25_config=phase25_config,
        history_interval=int(args.history_interval),
        progress_interval=int(args.progress_interval),
        verbose_pde=bool(args.verbose_pde),
    )
    write_csv(out_root / "phase31_phase30_biology_table.csv", phase30_bio_rows)
    write_json(out_root / "phase31_phase30_biology_table.json", phase30_bio_rows)

    endpoint_rows = load_csv_rows(args.phase26_run_root / "phase26_endpoint_table.csv")
    assay_rows = load_csv_rows(args.phase26_run_root / "phase26_assay_proxy_table.csv")
    rank_rows = load_csv_rows(args.phase26_run_root / "phase26_rank_shift_table.csv")
    delta_rows = load_csv_rows(args.phase26_run_root / "phase26_endpoint_delta_table.csv")

    manuscript_rows = build_phase25_mode_summary_rows(endpoint_rows, assay_rows)
    manuscript_rows.extend(phase30_bio_rows)
    write_csv(out_root / "phase31_manuscript_master_table.csv", manuscript_rows)

    rank_effect_rows = build_rank_effect_rows(rank_rows, rng=rng, bootstrap_samples=int(args.bootstrap_samples))
    write_csv(out_root / "phase31_rank_effect_sizes.csv", rank_effect_rows)

    sink_effect_rows = build_sink_delta_effect_rows(
        delta_rows,
        assay_rows,
        rng=rng,
        bootstrap_samples=int(args.bootstrap_samples),
    )
    write_csv(out_root / "phase31_sink_delta_effect_sizes.csv", sink_effect_rows)

    mc_sigma = build_mc_sigma_lookup(repeat_band_rows)
    bio_sigma = build_biology_sigma_lookup(args.phase26_run_root.resolve())
    rank_robustness_rows = simulate_rank_robustness(
        endpoint_rows,
        mc_sigma=mc_sigma,
        bio_sigma=bio_sigma,
        rng=rng,
        samples=int(args.bootstrap_samples),
    )
    write_csv(out_root / "phase31_rank_robustness_table.csv", rank_robustness_rows)

    manifest = build_reproducibility_manifest(
        args=args,
        repeat_rows=repeat_rows,
        template_rows=template_rows,
    )
    write_json(out_root / "phase31_reproducibility_manifest.json", manifest)

    write_discussion_draft(
        out_root / "phase31_discussion_draft.md",
        rank_effect_rows=rank_effect_rows,
        sink_effect_rows=sink_effect_rows,
        phase30_bio_rows=phase30_bio_rows,
        repeat_band_rows=repeat_band_rows,
    )
    write_quick_assessment(
        out_root / "phase31_quick_assessment.md",
        rank_effect_rows=rank_effect_rows,
        phase30_bio_rows=phase30_bio_rows,
        repeat_rows=repeat_rows,
    )

    print("=== PHASE 31 PUBLICATION PACKAGE COMPLETE ===")
    print(f"Output root: {out_root}")
    print(f"Benchmark template library: {out_root / 'phase31_benchmark_template_library.csv'}")
    print(f"Phase 30 repeat summary: {out_root / 'phase31_phase30_repeat_summary.csv'}")
    print(f"Phase 30 biology table: {out_root / 'phase31_phase30_biology_table.csv'}")
    print(f"Master manuscript table: {out_root / 'phase31_manuscript_master_table.csv'}")
    print(f"Rank robustness table: {out_root / 'phase31_rank_robustness_table.csv'}")
    print(f"Reproducibility manifest: {out_root / 'phase31_reproducibility_manifest.json'}")
    print(f"Discussion draft: {out_root / 'phase31_discussion_draft.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
