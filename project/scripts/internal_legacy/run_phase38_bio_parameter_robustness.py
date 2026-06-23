#!/usr/bin/env python3
"""Phase 38: biological parameter robustness and sink-strength sweep for the Phase 33 cohort."""

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

from phase37_bio_model_params import BioModelParams, bio_model_params_from_config
from run_phase26_vascular_sink_ablation import (
    PRIMARY_ENDPOINTS,
    assign_ranks,
    evaluate_bystander_mode,
    evaluate_physical_only,
)
from run_phase34_phase32_bio_cohort import (
    add_structure_aliases,
    endpoint_row_from_result,
    harmonize_context_to_dose_shape,
    load_case_context,
    load_csv_rows,
    load_dose_from_manifest_row,
    resolve_context_path,
    voxel_size_mm_from_axes,
)


def first_existing(candidates: Sequence[Path]) -> Path:
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def resolve_project_path(
    path_str: str,
    *,
    phase33_data_root: Path,
    phase32_data_root: Path | None = None,
) -> Path:
    if path_str.startswith("PROJECT_ROOT/"):
        relative = Path(path_str.removeprefix("PROJECT_ROOT/"))
        parts = relative.parts
        if len(parts) >= 3 and parts[0] == "runs":
            run_name = parts[1]
            tail = Path(*parts[2:])
            if run_name == "phase33_phase32_topas_cohort":
                return phase33_data_root / tail
            if run_name == "phase32_site_specific_template_phantoms" and phase32_data_root is not None:
                return phase32_data_root / tail
        return relative
    return Path(path_str)


def default_run_root(name: str) -> Path:
    project_root = Path(__file__).resolve().parents[1]
    return first_existing(
        [
            project_root / "runs" / name,
            Path("/Users/kw/Documents/Playground/vhee_topas/runs") / name,
            Path("/Users/kw/Desktop/SFRT_repo_main_update/project/runs") / name,
        ]
    )


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase32-root", type=Path, default=default_run_root("phase32_site_specific_template_phantoms"))
    parser.add_argument("--phase33-root", type=Path, default=default_run_root("phase33_phase32_topas_cohort"))
    parser.add_argument(
        "--phase33-manifest-root",
        type=Path,
        default=first_existing(
            [
                project_root / "public_results" / "phase33_34_cohort",
                Path("/Users/kw/Desktop/SFRT_repo_main_update/project/runs/phase33_phase32_topas_cohort"),
                Path("/Users/kw/Desktop/SFRT_Submission_reproducibility_bundle_ready/project/runs/phase33_phase32_topas_cohort"),
            ]
        ),
    )
    parser.add_argument(
        "--phase25-config",
        type=Path,
        default=first_existing(
            [
                project_root / "public_results" / "phase25_safe_core" / "phase25_config.json",
                default_run_root("linac_6mv_headneck_detailed_voxel_lattice_sfrt_phase25_safe_core_plan_library_1e5")
                / "phase25_config.json",
            ]
        ),
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        default=project_root / "public_results" / "phase38_bio_parameter_robustness",
    )
    parser.add_argument("--samples", type=int, default=16)
    parser.add_argument("--seed", type=int, default=38)
    parser.add_argument("--history-interval", type=int, default=20)
    parser.add_argument("--progress-interval", type=int, default=100)
    parser.add_argument("--brainstem-limit-gy", type=float, default=30.0)
    parser.add_argument("--sink-scales", type=float, nargs="+", default=[0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0])
    parser.add_argument("--dpi", type=int, default=260)
    parser.add_argument("--only-case-ids", type=str, nargs="*", default=[])
    return parser.parse_args()


def write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        raise ValueError(f"No rows supplied for {path}")
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


def geometric_sample(rng: np.random.Generator, baseline: float, lo_factor: float, hi_factor: float) -> float:
    return float(baseline * math.exp(rng.uniform(math.log(lo_factor), math.log(hi_factor))))


def linear_sample(rng: np.random.Generator, baseline: float, lo_factor: float, hi_factor: float) -> float:
    return float(baseline * rng.uniform(lo_factor, hi_factor))


def sample_bio_params(rng: np.random.Generator, base: BioModelParams) -> BioModelParams:
    ros_share = float(rng.uniform(0.35, 0.65))
    total_weight = float(base.weight_ros + base.weight_cyto)
    uptake_scale = float(rng.uniform(0.5, 1.5))
    return base.with_updates(
        gamma=linear_sample(rng, base.gamma, 0.75, 1.25),
        scaling_factor=geometric_sample(rng, base.scaling_factor, 0.5, 2.0),
        diffusion_ros=linear_sample(rng, base.diffusion_ros, 0.75, 1.35),
        diffusion_cyto=linear_sample(rng, base.diffusion_cyto, 0.75, 1.35),
        decay_ros=geometric_sample(rng, base.decay_ros, 0.5, 2.0),
        decay_cyto=geometric_sample(rng, base.decay_cyto, 0.5, 2.0),
        emax_ros=linear_sample(rng, base.emax_ros, 0.75, 1.35),
        emax_cyto=linear_sample(rng, base.emax_cyto, 0.75, 1.35),
        weight_ros=float(total_weight * ros_share),
        weight_cyto=float(total_weight * (1.0 - ros_share)),
        hypoxic_ros_scale=linear_sample(rng, base.hypoxic_ros_scale, 0.5, 1.5),
        hypoxic_cytokine_multiplier=linear_sample(rng, base.hypoxic_cytokine_multiplier, 0.7, 1.3),
        artery_ros_uptake=float(base.artery_ros_uptake * uptake_scale),
        artery_cyto_uptake=float(base.artery_cyto_uptake * uptake_scale),
        vein_ros_uptake=float(base.vein_ros_uptake * uptake_scale),
        vein_cyto_uptake=float(base.vein_cyto_uptake * uptake_scale),
        weight_immune=0.0,
    )


def build_parameter_range_rows(base: BioModelParams) -> List[Dict[str, object]]:
    return [
        {"parameter": "gamma", "baseline": base.gamma, "lower": base.gamma * 0.75, "upper": base.gamma * 1.25, "sampling": "linear"},
        {"parameter": "s", "baseline": base.scaling_factor, "lower": base.scaling_factor * 0.5, "upper": base.scaling_factor * 2.0, "sampling": "log"},
        {"parameter": "D_ROS", "baseline": base.diffusion_ros, "lower": base.diffusion_ros * 0.75, "upper": base.diffusion_ros * 1.35, "sampling": "linear"},
        {"parameter": "D_cyto", "baseline": base.diffusion_cyto, "lower": base.diffusion_cyto * 0.75, "upper": base.diffusion_cyto * 1.35, "sampling": "linear"},
        {"parameter": "lambda_ROS", "baseline": base.decay_ros, "lower": base.decay_ros * 0.5, "upper": base.decay_ros * 2.0, "sampling": "log"},
        {"parameter": "lambda_cyto", "baseline": base.decay_cyto, "lower": base.decay_cyto * 0.5, "upper": base.decay_cyto * 2.0, "sampling": "log"},
        {"parameter": "Emax_ROS", "baseline": base.emax_ros, "lower": base.emax_ros * 0.75, "upper": base.emax_ros * 1.35, "sampling": "linear"},
        {"parameter": "Emax_cyto", "baseline": base.emax_cyto, "lower": base.emax_cyto * 0.75, "upper": base.emax_cyto * 1.35, "sampling": "linear"},
        {"parameter": "w_ROS share", "baseline": base.weight_ros / (base.weight_ros + base.weight_cyto), "lower": 0.35, "upper": 0.65, "sampling": "linear"},
        {"parameter": "Hypoxic ROS scale", "baseline": base.hypoxic_ros_scale, "lower": base.hypoxic_ros_scale * 0.5, "upper": base.hypoxic_ros_scale * 1.5, "sampling": "linear"},
        {"parameter": "Hypoxic cytokine multiplier", "baseline": base.hypoxic_cytokine_multiplier, "lower": base.hypoxic_cytokine_multiplier * 0.7, "upper": base.hypoxic_cytokine_multiplier * 1.3, "sampling": "linear"},
        {"parameter": "Global vascular uptake scale", "baseline": 1.0, "lower": 0.5, "upper": 1.5, "sampling": "linear"},
    ]


def load_cases(
    phase33_root: Path,
    phase33_manifest_root: Path,
    phase32_root: Path,
    *,
    selected_case_ids: set[str] | None = None,
) -> List[Dict[str, object]]:
    case_rows = load_csv_rows(phase33_manifest_root / "phase33_case_manifest.csv")
    selected = [row for row in case_rows if str(row["status"]) == "completed"]
    if selected_case_ids:
        selected = [row for row in selected if str(row["case_id"]) in selected_case_ids]
    phase32_manifest = {str(row["template_id"]): row for row in load_csv_rows(phase32_root / "phase32_case_manifest.csv")}
    loaded: List[Dict[str, object]] = []
    for row in selected:
        phase32_row = phase32_manifest[str(row["case_id"])]
        context_path = resolve_context_path(
            phase33_row=row,
            phase32_row=phase32_row,
            phase33_data_root=phase33_root,
            phase32_data_root=phase32_root,
        )
        structures, axes_mm, voxel_size_mm, vertices_mm = load_case_context(context_path)
        structures = add_structure_aliases(structures)
        dose = load_dose_from_manifest_row(
            row,
            phase33_data_root=phase33_root,
            phase32_data_root=phase32_root,
        )
        structures, axes_mm = harmonize_context_to_dose_shape(
            structures=structures,
            axes_mm=axes_mm,
            dose_shape=tuple(int(v) for v in dose.shape),
        )
        voxel_size_mm = voxel_size_mm_from_axes(axes_mm)
        loaded.append(
            {
                "case_id": str(row["case_id"]),
                "case_label": str(row["case_label"]),
                "site_group": str(phase32_row.get("site_group", "")),
                "dose": dose,
                "structures": structures,
                "axes_mm": axes_mm,
                "voxel_size_mm": voxel_size_mm,
                "voxel_volume_cc": float(voxel_size_mm[0] * voxel_size_mm[1] * voxel_size_mm[2] / 1000.0),
                "vertices_mm": vertices_mm,
            }
        )
    return loaded


def evaluate_sample(
    *,
    sample_id: str,
    cases: Sequence[Mapping[str, object]],
    config: Mapping[str, object],
    bio_params: BioModelParams,
    history_interval: int,
    progress_interval: int,
) -> List[Dict[str, object]]:
    endpoint_rows: List[Dict[str, object]] = []
    for case in cases:
        physical = evaluate_physical_only(
            physical_dose=np.asarray(case["dose"], dtype=np.float32),
            structures=case["structures"],
            axes_mm=case["axes_mm"],
            spots_mm=case["vertices_mm"],
            voxel_volume_cc=float(case["voxel_volume_cc"]),
            voxel_size_mm=tuple(case["voxel_size_mm"]),
            prescription_gy=float(config.get("prescription_gy", 3.5)),
            alpha=float(bio_params.alpha),
            beta=float(bio_params.beta),
        )
        no_sink = evaluate_bystander_mode(
            physical_dose=np.asarray(case["dose"], dtype=np.float32),
            structures=case["structures"],
            axes_mm=case["axes_mm"],
            spots_mm=case["vertices_mm"],
            config=config,
            mode="bystander_no_sink",
            uptake_scale=1.0,
            voxel_volume_cc=float(case["voxel_volume_cc"]),
            voxel_size_mm=tuple(case["voxel_size_mm"]),
            prescription_gy=float(config.get("prescription_gy", 3.5)),
            history_interval=int(history_interval),
            progress_interval=int(progress_interval),
            verbose_pde=False,
            bio_params=bio_params,
        )
        with_sink = evaluate_bystander_mode(
            physical_dose=np.asarray(case["dose"], dtype=np.float32),
            structures=case["structures"],
            axes_mm=case["axes_mm"],
            spots_mm=case["vertices_mm"],
            config=config,
            mode="bystander_with_sink",
            uptake_scale=1.0,
            voxel_volume_cc=float(case["voxel_volume_cc"]),
            voxel_size_mm=tuple(case["voxel_size_mm"]),
            prescription_gy=float(config.get("prescription_gy", 3.5)),
            history_interval=int(history_interval),
            progress_interval=int(progress_interval),
            verbose_pde=False,
            bio_params=bio_params,
        )
        for mode, result in (
            ("physical_only", physical),
            ("bystander_no_sink", no_sink),
            ("bystander_with_sink", with_sink),
        ):
            row = endpoint_row_from_result(
                case_id=str(case["case_id"]),
                case_label=str(case["case_label"]),
                site_group=str(case["site_group"]),
                mode=mode,
                result=result,
            )
            row["sample_id"] = sample_id
            endpoint_rows.append(row)
    assign_ranks(endpoint_rows)
    return endpoint_rows


def summarize_rank_stability(
    sample_rank_rows: Sequence[Mapping[str, object]],
    baseline_rank_lookup: Mapping[str, int],
) -> List[Dict[str, object]]:
    grouped: Dict[str, List[int]] = {}
    for row in sample_rank_rows:
        grouped.setdefault(str(row["plan_id"]), []).append(int(row["with_sink_rank"]))
    rows: List[Dict[str, object]] = []
    for plan_id, ranks in sorted(grouped.items()):
        arr = np.asarray(ranks, dtype=int)
        baseline = int(baseline_rank_lookup[plan_id])
        rows.append(
            {
                "plan_id": plan_id,
                "baseline_with_sink_rank": baseline,
                "rank_mean": float(np.mean(arr)),
                "rank_median": float(np.median(arr)),
                "rank_min": int(np.min(arr)),
                "rank_max": int(np.max(arr)),
                "prob_same_rank": float(np.mean(arr == baseline)),
                "prob_top3": float(np.mean(arr <= 3)),
                "prob_bottom3": float(np.mean(arr >= 8)),
            }
        )
    return rows


def summarize_brainstem_flags(
    sample_endpoint_rows: Sequence[Mapping[str, object]],
    *,
    brainstem_limit_gy: float,
) -> List[Dict[str, object]]:
    by_sample_plan: Dict[Tuple[str, str], Dict[str, Mapping[str, object]]] = {}
    for row in sample_endpoint_rows:
        key = (str(row["sample_id"]), str(row["plan_id"]))
        by_sample_plan.setdefault(key, {})[str(row["mode"])] = row
    grouped: Dict[str, List[int]] = {}
    baseline_flag: Dict[str, int] = {}
    for (sample_id, plan_id), modes in sorted(by_sample_plan.items()):
        physical = float(modes["physical_only"]["brainstem_d2"])
        with_sink = float(modes["bystander_with_sink"]["brainstem_d2"])
        flag = int(physical <= float(brainstem_limit_gy) and with_sink > float(brainstem_limit_gy))
        grouped.setdefault(plan_id, []).append(flag)
        if sample_id == "baseline":
            baseline_flag[plan_id] = flag
    rows: List[Dict[str, object]] = []
    for plan_id, values in sorted(grouped.items()):
        arr = np.asarray(values, dtype=int)
        rows.append(
            {
                "plan_id": plan_id,
                "baseline_biology_adds_brainstem_failure": int(baseline_flag.get(plan_id, 0)),
                "prob_biology_adds_brainstem_failure": float(np.mean(arr)),
                "n_positive_samples": int(np.sum(arr)),
                "n_samples": int(arr.size),
            }
        )
    return rows


def summarize_endpoint_deltas(
    sample_endpoint_rows: Sequence[Mapping[str, object]],
) -> List[Dict[str, object]]:
    by_sample_plan: Dict[Tuple[str, str], Dict[str, Mapping[str, object]]] = {}
    for row in sample_endpoint_rows:
        key = (str(row["sample_id"]), str(row["plan_id"]))
        by_sample_plan.setdefault(key, {})[str(row["mode"])] = row
    rows: List[Dict[str, object]] = []
    for endpoint in PRIMARY_ENDPOINTS:
        deltas: List[float] = []
        sign_matches: List[bool] = []
        baseline_by_case: Dict[str, float] = {}
        for (sample_id, plan_id), modes in sorted(by_sample_plan.items()):
            delta = float(modes["bystander_with_sink"][endpoint.key]) - float(modes["bystander_no_sink"][endpoint.key])
            deltas.append(delta)
            if sample_id == "baseline":
                baseline_by_case[plan_id] = delta
        for (sample_id, plan_id), modes in sorted(by_sample_plan.items()):
            delta = float(modes["bystander_with_sink"][endpoint.key]) - float(modes["bystander_no_sink"][endpoint.key])
            baseline_delta = float(baseline_by_case.get(plan_id, 0.0))
            sign_matches.append(bool(np.sign(delta) == np.sign(baseline_delta)))
        arr = np.asarray(deltas, dtype=np.float64)
        baseline = float(np.mean(list(baseline_by_case.values()))) if baseline_by_case else 0.0
        rows.append(
            {
                "metric": endpoint.key,
                "label": endpoint.label,
                "baseline_delta": float(baseline),
                "mean_delta": float(np.mean(arr)) if arr.size else 0.0,
                "p2_5_delta": float(np.percentile(arr, 2.5)) if arr.size else 0.0,
                "p97_5_delta": float(np.percentile(arr, 97.5)) if arr.size else 0.0,
                "prob_same_sign_as_baseline": float(np.mean(sign_matches)) if sign_matches else 0.0,
            }
        )
    return rows


def build_sample_endpoint_delta_rows(
    sample_endpoint_rows: Sequence[Mapping[str, object]],
) -> List[Dict[str, object]]:
    by_sample_plan: Dict[Tuple[str, str], Dict[str, Mapping[str, object]]] = {}
    for row in sample_endpoint_rows:
        key = (str(row["sample_id"]), str(row["plan_id"]))
        by_sample_plan.setdefault(key, {})[str(row["mode"])] = row

    rows: List[Dict[str, object]] = []
    for (sample_id, plan_id), modes in sorted(by_sample_plan.items()):
        for endpoint in PRIMARY_ENDPOINTS:
            physical_value = float(modes["physical_only"][endpoint.key])
            no_sink_value = float(modes["bystander_no_sink"][endpoint.key])
            with_sink_value = float(modes["bystander_with_sink"][endpoint.key])
            rows.append(
                {
                    "sample_id": sample_id,
                    "plan_id": plan_id,
                    "metric": endpoint.key,
                    "label": endpoint.label,
                    "physical_value": physical_value,
                    "no_sink_value": no_sink_value,
                    "with_sink_value": with_sink_value,
                    "delta": with_sink_value - no_sink_value,
                }
            )
    return rows


def summarize_endpoint_delta_case_stability(
    sample_endpoint_rows: Sequence[Mapping[str, object]],
) -> List[Dict[str, object]]:
    by_sample_plan: Dict[Tuple[str, str], Dict[str, Mapping[str, object]]] = {}
    for row in sample_endpoint_rows:
        key = (str(row["sample_id"]), str(row["plan_id"]))
        by_sample_plan.setdefault(key, {})[str(row["mode"])] = row

    grouped: Dict[Tuple[str, str], List[float]] = {}
    baseline_lookup: Dict[Tuple[str, str], float] = {}
    for (sample_id, plan_id), modes in sorted(by_sample_plan.items()):
        for endpoint in PRIMARY_ENDPOINTS:
            key = (plan_id, endpoint.key)
            delta = float(modes["bystander_with_sink"][endpoint.key]) - float(modes["bystander_no_sink"][endpoint.key])
            grouped.setdefault(key, []).append(delta)
            if sample_id == "baseline":
                baseline_lookup[key] = delta

    rows: List[Dict[str, object]] = []
    for endpoint in PRIMARY_ENDPOINTS:
        for plan_id, _metric in sorted(key for key in grouped if key[1] == endpoint.key):
            key = (plan_id, endpoint.key)
            arr = np.asarray(grouped[key], dtype=np.float64)
            baseline = float(baseline_lookup.get(key, 0.0))
            rows.append(
                {
                    "plan_id": plan_id,
                    "metric": endpoint.key,
                    "label": endpoint.label,
                    "baseline_delta": baseline,
                    "mean_delta": float(np.mean(arr)) if arr.size else 0.0,
                    "min_delta": float(np.min(arr)) if arr.size else 0.0,
                    "max_delta": float(np.max(arr)) if arr.size else 0.0,
                    "p2_5_delta": float(np.percentile(arr, 2.5)) if arr.size else 0.0,
                    "p97_5_delta": float(np.percentile(arr, 97.5)) if arr.size else 0.0,
                    "prob_same_sign_as_baseline": float(np.mean(np.sign(arr) == np.sign(baseline))) if arr.size else 0.0,
                }
            )
    return rows


def build_rank_rows(endpoint_rows: Sequence[Mapping[str, object]]) -> List[Dict[str, object]]:
    by_sample: Dict[str, Dict[str, Dict[str, Mapping[str, object]]]] = {}
    for row in endpoint_rows:
        by_sample.setdefault(str(row["sample_id"]), {}).setdefault(str(row["plan_id"]), {})[str(row["mode"])] = row
    rows: List[Dict[str, object]] = []
    for sample_id, sample_payload in sorted(by_sample.items()):
        for plan_id, modes in sorted(sample_payload.items()):
            rows.append(
                {
                    "sample_id": sample_id,
                    "plan_id": plan_id,
                    "physical_rank": int(modes["physical_only"]["rank"]),
                    "no_sink_rank": int(modes["bystander_no_sink"]["rank"]),
                    "with_sink_rank": int(modes["bystander_with_sink"]["rank"]),
                    "with_sink_vs_physical_shift": int(modes["bystander_with_sink"]["rank"]) - int(modes["physical_only"]["rank"]),
                    "with_sink_vs_no_sink_shift": int(modes["bystander_with_sink"]["rank"]) - int(modes["bystander_no_sink"]["rank"]),
                }
            )
    return rows


def summarize_rank_shift_persistence(
    sample_rank_rows: Sequence[Mapping[str, object]],
) -> List[Dict[str, object]]:
    grouped: Dict[str, List[Mapping[str, object]]] = {}
    for row in sample_rank_rows:
        grouped.setdefault(str(row["plan_id"]), []).append(row)

    rows: List[Dict[str, object]] = []
    for plan_id, entries in sorted(grouped.items()):
        baseline = next(row for row in entries if str(row["sample_id"]) == "baseline")
        baseline_phys = int(baseline["with_sink_vs_physical_shift"])
        baseline_nosink = int(baseline["with_sink_vs_no_sink_shift"])
        phys_arr = np.asarray([int(row["with_sink_vs_physical_shift"]) for row in entries], dtype=int)
        nosink_arr = np.asarray([int(row["with_sink_vs_no_sink_shift"]) for row in entries], dtype=int)
        rows.append(
            {
                "plan_id": plan_id,
                "baseline_with_sink_vs_physical_shift": baseline_phys,
                "baseline_with_sink_vs_no_sink_shift": baseline_nosink,
                "prob_nonzero_with_sink_vs_physical_shift": float(np.mean(phys_arr != 0)),
                "prob_same_sign_with_sink_vs_physical_shift": float(np.mean(np.sign(phys_arr) == np.sign(baseline_phys))),
                "prob_nonzero_with_sink_vs_no_sink_shift": float(np.mean(nosink_arr != 0)),
                "prob_same_sign_with_sink_vs_no_sink_shift": float(np.mean(np.sign(nosink_arr) == np.sign(baseline_nosink))),
            }
        )
    return rows


def safe_corr(x: np.ndarray, y: np.ndarray) -> float:
    if x.size != y.size or x.size == 0:
        return 0.0
    if float(np.std(x)) == 0.0 or float(np.std(y)) == 0.0:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def summarize_parameter_dominance(
    sample_rows: Sequence[Mapping[str, object]],
    sample_rank_rows: Sequence[Mapping[str, object]],
    sample_endpoint_delta_rows: Sequence[Mapping[str, object]],
    *,
    brainstem_limit_gy: float,
) -> List[Dict[str, object]]:
    if not sample_rows:
        return []
    sample_order = [str(row["sample_id"]) for row in sample_rows]
    rank_by_sample: Dict[str, List[Mapping[str, object]]] = {}
    for row in sample_rank_rows:
        rank_by_sample.setdefault(str(row["sample_id"]), []).append(row)

    brainstem_by_sample: Dict[str, List[int]] = {sample_id: [] for sample_id in sample_order}
    endpoint_by_sample: Dict[Tuple[str, str], List[float]] = {}
    for row in sample_endpoint_delta_rows:
        endpoint_by_sample.setdefault((str(row["sample_id"]), str(row["metric"])), []).append(float(row["delta"]))

    for row in sample_endpoint_delta_rows:
        if str(row["metric"]) == "brainstem_d2":
            sample_id = str(row["sample_id"])
            baseline_physical = float(row["physical_value"])
            with_sink = float(row["with_sink_value"])
            flag = int(baseline_physical <= float(brainstem_limit_gy) and with_sink > float(brainstem_limit_gy))
            brainstem_by_sample.setdefault(sample_id, []).append(flag)

    outcomes = {
        "rank_changes_vs_physical": np.asarray(
            [sum(int(row["with_sink_vs_physical_shift"]) != 0 for row in rank_by_sample.get(sample_id, [])) for sample_id in sample_order],
            dtype=float,
        ),
        "rank_changes_vs_no_sink": np.asarray(
            [sum(int(row["with_sink_vs_no_sink_shift"]) != 0 for row in rank_by_sample.get(sample_id, [])) for sample_id in sample_order],
            dtype=float,
        ),
        "brainstem_flag_count": np.asarray(
            [sum(brainstem_by_sample.get(sample_id, [])) for sample_id in sample_order],
            dtype=float,
        ),
        "mean_abs_pvdr_delta": np.asarray(
            [float(np.mean(np.abs(endpoint_by_sample.get((sample_id, "pvdr"), [0.0])))) for sample_id in sample_order],
            dtype=float,
        ),
    }

    excluded = {"sample_id"}
    rows: List[Dict[str, object]] = []
    for key in sample_rows[0].keys():
        if key in excluded:
            continue
        values = np.asarray([float(row[key]) for row in sample_rows], dtype=float)
        row = {"parameter": key}
        max_abs = 0.0
        dominant_outcome = ""
        for outcome_name, outcome_values in outcomes.items():
            corr = safe_corr(values, outcome_values)
            row[f"corr_{outcome_name}"] = corr
            if abs(corr) > max_abs:
                max_abs = abs(corr)
                dominant_outcome = outcome_name
        row["max_abs_corr"] = max_abs
        row["dominant_outcome"] = dominant_outcome if dominant_outcome else "none"
        rows.append(row)
    rows.sort(key=lambda item: float(item["max_abs_corr"]), reverse=True)
    return rows


def summarize_sink_strength(
    *,
    cases: Sequence[Mapping[str, object]],
    config: Mapping[str, object],
    base_params: BioModelParams,
    sink_scales: Sequence[float],
    history_interval: int,
    progress_interval: int,
) -> List[Dict[str, object]]:
    no_sink_by_case: Dict[str, Mapping[str, object]] = {}
    for case in cases:
        result = evaluate_bystander_mode(
            physical_dose=np.asarray(case["dose"], dtype=np.float32),
            structures=case["structures"],
            axes_mm=case["axes_mm"],
            spots_mm=case["vertices_mm"],
            config=config,
            mode="bystander_no_sink",
            uptake_scale=1.0,
            voxel_volume_cc=float(case["voxel_volume_cc"]),
            voxel_size_mm=tuple(case["voxel_size_mm"]),
            prescription_gy=float(config.get("prescription_gy", 3.5)),
            history_interval=int(history_interval),
            progress_interval=int(progress_interval),
            verbose_pde=False,
            bio_params=base_params,
        )
        no_sink_by_case[str(case["case_id"])] = result

    rows: List[Dict[str, object]] = []
    tracked = {
        "pvdr": "PVDR",
        "spill_shell_0_5_mean": "Peri-GTV 0-5 mm mean",
        "brainstem_d2": "Brainstem D2",
        "parotid_r_mean": "Right parotid mean",
    }
    for scale in sink_scales:
        scale_params = base_params.with_updates(
            artery_ros_uptake=float(base_params.artery_ros_uptake * scale),
            artery_cyto_uptake=float(base_params.artery_cyto_uptake * scale),
            vein_ros_uptake=float(base_params.vein_ros_uptake * scale),
            vein_cyto_uptake=float(base_params.vein_cyto_uptake * scale),
        )
        deltas: Dict[str, List[float]] = {key: [] for key in tracked}
        for case in cases:
            with_sink = evaluate_bystander_mode(
                physical_dose=np.asarray(case["dose"], dtype=np.float32),
                structures=case["structures"],
                axes_mm=case["axes_mm"],
                spots_mm=case["vertices_mm"],
                config=config,
                mode="bystander_with_sink",
                uptake_scale=1.0,
                voxel_volume_cc=float(case["voxel_volume_cc"]),
                voxel_size_mm=tuple(case["voxel_size_mm"]),
                prescription_gy=float(config.get("prescription_gy", 3.5)),
                history_interval=int(history_interval),
                progress_interval=int(progress_interval),
                verbose_pde=False,
                bio_params=scale_params,
            )
            no_sink = no_sink_by_case[str(case["case_id"])]
            for key in tracked:
                deltas[key].append(float(with_sink["endpoints"][key]) - float(no_sink["endpoints"][key]))
        for key, label in tracked.items():
            arr = np.asarray(deltas[key], dtype=np.float64)
            rows.append(
                {
                    "uptake_scale": float(scale),
                    "metric": key,
                    "label": label,
                    "mean_delta": float(np.mean(arr)) if arr.size else 0.0,
                    "min_delta": float(np.min(arr)) if arr.size else 0.0,
                    "max_delta": float(np.max(arr)) if arr.size else 0.0,
                }
            )
    return rows


def plot_robustness_figure(
    out_file: Path,
    rank_rows: Sequence[Mapping[str, object]],
    brainstem_rows: Sequence[Mapping[str, object]],
    sink_rows: Sequence[Mapping[str, object]],
    *,
    dpi: int,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.4), constrained_layout=True)

    rank_rows = list(rank_rows)
    x = np.arange(len(rank_rows))
    rank_means = np.asarray([float(row["rank_mean"]) for row in rank_rows], dtype=float)
    rank_medians = np.asarray([float(row["rank_median"]) for row in rank_rows], dtype=float)
    rank_mins = np.asarray([float(row["rank_min"]) for row in rank_rows], dtype=float)
    rank_maxs = np.asarray([float(row["rank_max"]) for row in rank_rows], dtype=float)
    baseline = np.asarray([float(row["baseline_with_sink_rank"]) for row in rank_rows], dtype=float)
    axes[0].vlines(x, rank_mins, rank_maxs, color="#7aa6d8", linewidth=2.0)
    axes[0].scatter(x, rank_medians, color="#1f77b4", s=42, label="Median sampled rank", zorder=3)
    axes[0].scatter(x, baseline, color="#111111", marker="D", s=32, label="Baseline rank", zorder=4)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([str(row["plan_id"]).replace("case", "") for row in rank_rows])
    axes[0].invert_yaxis()
    axes[0].set_ylabel("With-sink risk rank (1 = least risky)")
    axes[0].set_title("(a) Rank stability under biology-parameter variation", fontsize=11, fontweight="bold")
    axes[0].legend(frameon=False, fontsize=8, loc="upper right")

    brainstem_rows = list(brainstem_rows)
    bx = np.arange(len(brainstem_rows))
    axes[1].bar(
        bx,
        [float(row["prob_biology_adds_brainstem_failure"]) for row in brainstem_rows],
        color="#d35400",
        edgecolor="black",
        linewidth=0.5,
    )
    axes[1].set_xticks(bx)
    axes[1].set_xticklabels([str(row["plan_id"]).replace("case", "") for row in brainstem_rows])
    axes[1].set_ylim(0.0, 1.0)
    axes[1].set_ylabel("Probability of biology-added brainstem flag")
    axes[1].set_title("(b) Brainstem-flag persistence", fontsize=11, fontweight="bold")

    sink_rows = list(sink_rows)
    tracked = [
        ("pvdr", "#34495e"),
        ("spill_shell_0_5_mean", "#16a085"),
        ("brainstem_d2", "#8e44ad"),
    ]
    for metric, color in tracked:
        rows = [row for row in sink_rows if str(row["metric"]) == metric]
        rows = sorted(rows, key=lambda row: float(row["uptake_scale"]))
        axes[2].plot(
            [float(row["uptake_scale"]) for row in rows],
            [float(row["mean_delta"]) for row in rows],
            marker="o",
            linewidth=1.8,
            color=color,
            label=str(rows[0]["label"]) if rows else metric,
        )
    axes[2].axhline(0.0, color="#555555", linewidth=0.8, linestyle="--")
    axes[2].set_xlabel("Global vascular uptake scale")
    axes[2].set_ylabel("Mean with-sink minus no-sink delta")
    axes[2].set_title("(c) Sink-strength sweep", fontsize=11, fontweight="bold")
    axes[2].legend(frameon=False, fontsize=8, loc="best")

    fig.savefig(out_file, dpi=int(dpi), bbox_inches="tight")
    plt.close(fig)


def build_quick_assessment(
    rank_rows: Sequence[Mapping[str, object]],
    brainstem_rows: Sequence[Mapping[str, object]],
    delta_rows: Sequence[Mapping[str, object]],
    dominance_rows: Sequence[Mapping[str, object]] | None = None,
) -> str:
    mean_rank_same = float(np.mean([float(row["prob_same_rank"]) for row in rank_rows])) if rank_rows else 0.0
    mean_brainstem = (
        float(np.mean([float(row["prob_biology_adds_brainstem_failure"]) for row in brainstem_rows])) if brainstem_rows else 0.0
    )
    pvdr = next((row for row in delta_rows if str(row["metric"]) == "pvdr"), None)
    lines = [
        "# Phase 38 quick assessment",
        "",
        f"- Mean probability of retaining the same with-sink rank under biological parameter variation: `{mean_rank_same:.3f}`.",
        f"- Mean probability of preserving a biology-added brainstem flag across the sampled parameter set: `{mean_brainstem:.3f}`.",
    ]
    if pvdr is not None:
        lines.append(
            f"- Baseline with-sink vs no-sink PVDR delta remained sign-consistent in `{float(pvdr['prob_same_sign_as_baseline']):.3f}` of sampled scenarios."
        )
    if dominance_rows:
        top = list(dominance_rows)[0]
        lines.append(
            f"- Largest sampled parameter-outcome correlation magnitude was `{float(top['max_abs_corr']):.3f}` for `{top['parameter']}` against `{top['dominant_outcome']}`."
        )
    return "\n".join(lines) + "\n"


def build_cohort_overview_rows(
    rank_rows: Sequence[Mapping[str, object]],
    rank_summary_rows: Sequence[Mapping[str, object]],
    brainstem_rows: Sequence[Mapping[str, object]],
    endpoint_delta_rows: Sequence[Mapping[str, object]],
) -> List[Dict[str, object]]:
    baseline_rank_rows = [row for row in rank_rows if str(row["sample_id"]) == "baseline"]
    rows: List[Dict[str, object]] = [
        {
            "metric": "with_sink vs physical rank shifts",
            "full_cohort_result": f"{sum(int(row['with_sink_vs_physical_shift']) != 0 for row in baseline_rank_rows)}/{len(baseline_rank_rows)}",
            "robustness_interpretation": "ranking changes occur in main model",
        },
        {
            "metric": "no_sink vs physical rank shifts",
            "full_cohort_result": f"{sum(int(row['physical_rank']) != int(row['no_sink_rank']) for row in baseline_rank_rows)}/{len(baseline_rank_rows)}",
            "robustness_interpretation": "non-local biology alone drives much of the effect",
        },
        {
            "metric": "with_sink vs no-sink rank shifts",
            "full_cohort_result": f"{sum(int(row['with_sink_vs_no_sink_shift']) != 0 for row in baseline_rank_rows)}/{len(baseline_rank_rows)}",
            "robustness_interpretation": "sink modifies interpretation in a subset",
        },
        {
            "metric": "biology-added brainstem flags",
            "full_cohort_result": f"{sum(int(row['baseline_biology_adds_brainstem_failure']) != 0 for row in brainstem_rows)}/{len(brainstem_rows)}",
            "robustness_interpretation": "endpoint-level OAR reinterpretation present",
        },
        {
            "metric": "same-rank probability",
            "full_cohort_result": f"{float(np.mean([float(row['prob_same_rank']) for row in rank_summary_rows])):.3f}",
            "robustness_interpretation": "ranking robustness test",
        },
        {
            "metric": "endpoint sign stability",
            "full_cohort_result": f"{float(np.mean([float(row['prob_same_sign_as_baseline']) for row in endpoint_delta_rows])):.3f}",
            "robustness_interpretation": "endpoint-direction robustness test",
        },
        {
            "metric": "PVDR sign stability",
            "full_cohort_result": f"{float(next(row for row in endpoint_delta_rows if str(row['metric']) == 'pvdr')['prob_same_sign_as_baseline']):.3f}",
            "robustness_interpretation": "PVDR robustness test",
        },
    ]
    return rows


def main() -> int:
    args = parse_args()
    out_root = args.out_root.resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    config = json.loads(args.phase25_config.read_text(encoding="utf-8"))
    base_params = bio_model_params_from_config(config, include_immune=False)
    selected_case_ids = set(str(case_id) for case_id in args.only_case_ids) if args.only_case_ids else None
    cases = load_cases(
        args.phase33_root.resolve(),
        args.phase33_manifest_root.resolve(),
        args.phase32_root.resolve(),
        selected_case_ids=selected_case_ids,
    )
    rng = np.random.default_rng(int(args.seed))

    sample_rows: List[Dict[str, object]] = [
        {"sample_id": "baseline", **base_params.as_dict()},
    ]
    sample_endpoint_rows = evaluate_sample(
        sample_id="baseline",
        cases=cases,
        config=config,
        bio_params=base_params,
        history_interval=int(args.history_interval),
        progress_interval=int(args.progress_interval),
    )

    for idx in range(1, int(args.samples) + 1):
        sample_id = f"sample_{idx:03d}"
        sampled = sample_bio_params(rng, base_params)
        sample_rows.append({"sample_id": sample_id, **sampled.as_dict()})
        sample_endpoint_rows.extend(
            evaluate_sample(
                sample_id=sample_id,
                cases=cases,
                config=config,
                bio_params=sampled,
                history_interval=int(args.history_interval),
                progress_interval=int(args.progress_interval),
            )
        )

    rank_rows = build_rank_rows(sample_endpoint_rows)
    baseline_rank_lookup = {
        str(row["plan_id"]): int(row["with_sink_rank"])
        for row in rank_rows
        if str(row["sample_id"]) == "baseline"
    }
    rank_summary_rows = summarize_rank_stability(rank_rows, baseline_rank_lookup)
    rank_shift_persistence_rows = summarize_rank_shift_persistence(rank_rows)
    brainstem_rows = summarize_brainstem_flags(sample_endpoint_rows, brainstem_limit_gy=float(args.brainstem_limit_gy))
    sample_endpoint_delta_rows = build_sample_endpoint_delta_rows(sample_endpoint_rows)
    endpoint_delta_rows = summarize_endpoint_deltas(sample_endpoint_rows)
    endpoint_delta_case_rows = summarize_endpoint_delta_case_stability(sample_endpoint_rows)
    sink_strength_rows = summarize_sink_strength(
        cases=cases,
        config=config,
        base_params=base_params,
        sink_scales=[float(value) for value in args.sink_scales],
        history_interval=int(args.history_interval),
        progress_interval=int(args.progress_interval),
    )
    parameter_dominance_rows = summarize_parameter_dominance(
        sample_rows,
        rank_rows,
        sample_endpoint_delta_rows,
        brainstem_limit_gy=float(args.brainstem_limit_gy),
    )
    cohort_overview_rows = build_cohort_overview_rows(
        rank_rows,
        rank_summary_rows,
        brainstem_rows,
        endpoint_delta_rows,
    )

    write_csv(out_root / "phase38_parameter_ranges.csv", build_parameter_range_rows(base_params))
    write_csv(out_root / "phase38_sample_manifest.csv", sample_rows)
    write_csv(out_root / "phase38_sample_endpoint_rows.csv", sample_endpoint_rows)
    write_csv(out_root / "phase38_sample_rank_rows.csv", rank_rows)
    write_csv(out_root / "phase38_rank_stability.csv", rank_summary_rows)
    write_csv(out_root / "phase38_rank_shift_persistence.csv", rank_shift_persistence_rows)
    write_csv(out_root / "phase38_brainstem_flag_stability.csv", brainstem_rows)
    write_csv(out_root / "phase38_sample_endpoint_delta_rows.csv", sample_endpoint_delta_rows)
    write_csv(out_root / "phase38_endpoint_delta_summary.csv", endpoint_delta_rows)
    write_csv(out_root / "phase38_endpoint_delta_case_stability.csv", endpoint_delta_case_rows)
    write_csv(out_root / "phase38_sink_strength_summary.csv", sink_strength_rows)
    write_csv(out_root / "phase38_parameter_dominance.csv", parameter_dominance_rows)
    write_csv(out_root / "phase38_cohort_overview.csv", cohort_overview_rows)
    (out_root / "phase38_quick_assessment.md").write_text(
        build_quick_assessment(rank_summary_rows, brainstem_rows, endpoint_delta_rows, parameter_dominance_rows),
        encoding="utf-8",
    )
    plot_robustness_figure(
        out_root / "figure_phase38_bio_parameter_robustness.png",
        rank_summary_rows,
        brainstem_rows,
        sink_strength_rows,
        dpi=int(args.dpi),
    )
    write_json(
        out_root / "phase38_manifest.json",
        {
            "phase33_root": str(args.phase33_root.resolve()),
            "phase32_root": str(args.phase32_root.resolve()),
            "phase25_config": str(args.phase25_config.resolve()),
            "samples": int(args.samples),
            "seed": int(args.seed),
            "sink_scales": [float(value) for value in args.sink_scales],
            "case_ids": [str(case["case_id"]) for case in cases],
            "base_params": base_params.as_dict(),
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
