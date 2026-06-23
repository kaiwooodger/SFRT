#!/usr/bin/env python3
"""Phase 37B: uncertainty overlay for stronger true-vs-surrogate sink comparisons on the repeated subset."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np

from phase37_sink_falsification_utils import (
    COMPARISON_ENDPOINT_KEYS,
    ENDPOINT_DIRECTION,
    ENDPOINT_LABELS,
    ENDPOINT_UNITS,
    MODEL_LABELS,
    PHASE37B_MODEL_IDS,
    add_structure_aliases,
    assay_row_from_result,
    build_sink_uptake_tensor,
    build_true_uptake_and_modifiers,
    endpoint_row_from_result,
    endpoint_value_better,
    evaluate_custom_sink_model,
    load_case_context,
    load_dose_npz,
    rows_from_phase34,
    stable_seed_from_key,
    voxel_volume_cc,
)
from run_phase26_vascular_sink_ablation import PRIMARY_ENDPOINTS, endpoint_z_scores
from run_phase31_publication_package import (
    bootstrap_interval,
    build_biology_sigma_lookup,
    load_csv_rows,
    scalar_summary,
    write_csv,
    write_json,
)


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase35-root",
        type=Path,
        default=root / "runs" / "phase35_subset_repeat_uncertainty",
    )
    parser.add_argument(
        "--phase37a-root",
        type=Path,
        default=root / "runs" / "phase37a_vessel_falsification_cohort",
    )
    parser.add_argument(
        "--phase25-run-root",
        type=Path,
        default=root / "runs" / "linac_6mv_headneck_detailed_voxel_lattice_sfrt_phase25_safe_core_plan_library_1e5",
    )
    parser.add_argument(
        "--phase26-run-root",
        type=Path,
        default=root
        / "runs"
        / "linac_6mv_headneck_detailed_voxel_lattice_sfrt_phase25_safe_core_plan_library_1e5"
        / "phase26_vascular_sink_ablation",
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        default=root / "runs" / "phase37b_vessel_falsification_uncertainty",
    )
    parser.add_argument("--only-case-ids", type=str, nargs="*", default=[])
    parser.add_argument("--history-interval", type=int, default=20)
    parser.add_argument("--progress-interval", type=int, default=100)
    parser.add_argument("--rank-sim-samples", type=int, default=2000)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=361)
    parser.add_argument("--verbose-pde", action="store_true")
    return parser.parse_args()


def assign_model_ranks(rows: List[Dict[str, object]], *, group_keys: Sequence[str]) -> None:
    groups: Dict[Tuple[str, ...], List[Dict[str, object]]] = {}
    for row in rows:
        groups.setdefault(tuple(str(row[key]) for key in group_keys), []).append(row)
    for group_rows in groups.values():
        risk_scores = np.zeros(len(group_rows), dtype=np.float64)
        for endpoint in PRIMARY_ENDPOINTS:
            values = [float(row[endpoint.key]) for row in group_rows]
            risk_scores += endpoint_z_scores(values, higher_is_better=endpoint.higher_is_better)
        order = np.argsort(np.argsort(risk_scores)) + 1
        for idx, row in enumerate(group_rows):
            row["model_risk_score"] = float(risk_scores[idx])
            row["model_rank_within_case"] = int(order[idx])


def build_baseline_subset_model_rows(phase37a_root: Path, *, case_ids: Sequence[str]) -> List[Dict[str, object]]:
    rows = [
        row
        for row in load_csv_rows(phase37a_root / "phase37a_endpoint_table.csv")
        if str(row["plan_id"]) in set(case_ids) and str(row["model_id"]) in set(PHASE37B_MODEL_IDS)
    ]
    converted: List[Dict[str, object]] = []
    for row in rows:
        converted_row: Dict[str, object] = {
            "plan_id": str(row["plan_id"]),
            "case_label": str(row["case_label"]),
            "site_group": str(row["site_group"]),
            "model_id": str(row["model_id"]),
            "model_label": str(row["model_label"]),
        }
        for key, value in row.items():
            if key in converted_row:
                continue
            try:
                converted_row[key] = float(value)
            except ValueError:
                converted_row[key] = value
        converted.append(converted_row)
    assign_model_ranks(converted, group_keys=("plan_id",))
    return converted


def load_repeat_rows_for_existing_models(
    phase35_root: Path,
    *,
    case_ids: Sequence[str],
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]], List[Mapping[str, str]]]:
    repeat_manifest = load_csv_rows(phase35_root / "phase35_repeat_manifest.csv")
    endpoint_rows: List[Dict[str, object]] = []
    assay_rows: List[Dict[str, object]] = []
    for repeat in repeat_manifest:
        repeat_id = str(repeat["repeat_id"])
        phase34_root = Path(str(repeat["phase34_root"]))
        endpoint_raw = [
            row
            for row in load_csv_rows(phase34_root / "phase34_endpoint_table.csv")
            if str(row["plan_id"]) in set(case_ids)
        ]
        assay_raw = [
            row
            for row in load_csv_rows(phase34_root / "phase34_assay_proxy_table.csv")
            if str(row["plan_id"]) in set(case_ids)
        ]
        endpoint_base, assay_base = rows_from_phase34(endpoint_raw, assay_raw)
        for row in endpoint_base:
            if str(row["model_id"]) not in {"no_sink", "true_vessel_sink"}:
                continue
            row["repeat_id"] = repeat_id
            row["seed"] = int(repeat["seed"])
            row["history_scale"] = float(repeat["history_scale"])
            endpoint_rows.append(row)
        for row in assay_base:
            if str(row["model_id"]) not in {"no_sink", "true_vessel_sink"}:
                continue
            row["repeat_id"] = repeat_id
            row["seed"] = int(repeat["seed"])
            row["history_scale"] = float(repeat["history_scale"])
            assay_rows.append(row)
    return endpoint_rows, assay_rows, repeat_manifest


def evaluate_repeat_surrogates(
    *,
    repeat_manifest: Sequence[Mapping[str, str]],
    phase25_config: Mapping[str, object],
    case_ids: Sequence[str],
    history_interval: int,
    progress_interval: int,
    verbose_pde: bool,
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    endpoint_rows: List[Dict[str, object]] = []
    assay_rows: List[Dict[str, object]] = []
    for repeat in repeat_manifest:
        repeat_id = str(repeat["repeat_id"])
        phase33_root = Path(str(repeat["phase33_root"]))
        phase33_rows = [
            row
            for row in load_csv_rows(phase33_root / "phase33_case_manifest.csv")
            if str(row["status"]) == "completed" and str(row["case_id"]) in set(case_ids)
        ]
        for row in phase33_rows:
            case_id = str(row["case_id"])
            case_label = str(row["case_label"])
            context_path = Path(str(row["phase33_context_npz"]))
            structures, axes_mm, voxel_size_mm, vertices_mm = load_case_context(context_path)
            structures = add_structure_aliases(structures, voxel_size_mm)
            dose = load_dose_npz(Path(str(row["combined_dose_npz"])))
            voxel_cc_value = voxel_volume_cc(voxel_size_mm)
            true_uptake, m_type, m_oxygen = build_true_uptake_and_modifiers(phase25_config, structures)
            for model_id in (
                "ap_flip_sink",
                "si_flip_sink",
                "local_dropout_sink_20mm",
                "randomized_displacement_sink",
                "uniform_body_sink_mass_matched",
            ):
                random_seed = stable_seed_from_key(f"phase37b:{case_id}:{model_id}")
                uptake_tensor, _ = build_sink_uptake_tensor(
                    model_id,
                    true_uptake_tensor=true_uptake,
                    structures=structures,
                    axes_mm=axes_mm,
                    random_seed=random_seed,
                )
                result = evaluate_custom_sink_model(
                    physical_dose=dose,
                    structures=structures,
                    axes_mm=axes_mm,
                    spots_mm=vertices_mm,
                    config=phase25_config,
                    uptake_tensor=uptake_tensor,
                    m_type=m_type,
                    m_oxygen=m_oxygen,
                    voxel_volume_cc_value=voxel_cc_value,
                    voxel_size_mm=voxel_size_mm,
                    prescription_gy=3.5,
                    history_interval=int(history_interval),
                    progress_interval=int(progress_interval),
                    verbose_pde=bool(verbose_pde),
                )
                endpoint_row = endpoint_row_from_result(
                    plan_id=case_id,
                    case_label=case_label,
                    site_group="",
                    model_id=model_id,
                    result=result,
                )
                assay_row = assay_row_from_result(
                    plan_id=case_id,
                    case_label=case_label,
                    site_group="",
                    model_id=model_id,
                    result=result,
                )
                endpoint_row["repeat_id"] = repeat_id
                endpoint_row["seed"] = int(repeat["seed"])
                endpoint_row["history_scale"] = float(repeat["history_scale"])
                assay_row["repeat_id"] = repeat_id
                assay_row["seed"] = int(repeat["seed"])
                assay_row["history_scale"] = float(repeat["history_scale"])
                endpoint_rows.append(endpoint_row)
                assay_rows.append(assay_row)
    return endpoint_rows, assay_rows


def build_true_vs_surrogate_noise_rows(
    endpoint_rows: Sequence[Mapping[str, object]],
    *,
    bio_sigma_lookup: Mapping[str, float],
    rng: np.random.Generator,
    bootstrap_samples: int,
) -> List[Dict[str, object]]:
    lookup = {
        (str(row["repeat_id"]), str(row["plan_id"]), str(row["model_id"])): row for row in endpoint_rows
    }
    repeat_ids = sorted({str(row["repeat_id"]) for row in endpoint_rows})
    plan_ids = sorted({str(row["plan_id"]) for row in endpoint_rows})
    comparator_ids = (
        "no_sink",
        "ap_flip_sink",
        "si_flip_sink",
        "local_dropout_sink_20mm",
        "randomized_displacement_sink",
        "uniform_body_sink_mass_matched",
    )
    rows: List[Dict[str, object]] = []
    for plan_id in plan_ids:
        case_label = next(str(row["case_label"]) for row in endpoint_rows if str(row["plan_id"]) == plan_id)
        for comparator_id in comparator_ids:
            for endpoint_key in COMPARISON_ENDPOINT_KEYS:
                deltas: List[float] = []
                for repeat_id in repeat_ids:
                    true_row = lookup[(repeat_id, plan_id, "true_vessel_sink")]
                    comp_row = lookup[(repeat_id, plan_id, comparator_id)]
                    deltas.append(float(true_row[endpoint_key]) - float(comp_row[endpoint_key]))
                summary = scalar_summary(deltas)
                ci = bootstrap_interval(deltas, rng=rng, samples=int(bootstrap_samples), reducer=np.mean)
                mc_sigma = float(summary["std"])
                bio_sigma = float(bio_sigma_lookup.get(endpoint_key, 0.0))
                extra_sigma = bio_sigma if comparator_id == "no_sink" else math.sqrt(2.0) * bio_sigma
                combined_sigma = math.sqrt(mc_sigma**2 + extra_sigma**2)
                combined_band_95 = 1.96 * combined_sigma
                mean_delta = float(summary["mean"])
                rows.append(
                    {
                        "plan_id": plan_id,
                        "case_label": case_label,
                        "comparator_id": comparator_id,
                        "comparator_label": MODEL_LABELS[comparator_id],
                        "metric": endpoint_key,
                        "label": ENDPOINT_LABELS[endpoint_key],
                        "units": ENDPOINT_UNITS[endpoint_key],
                        "n_repeats": int(len(deltas)),
                        "mean_true_minus_comparator": mean_delta,
                        "std_repeat_delta": mc_sigma,
                        "min_delta": summary["min"],
                        "max_delta": summary["max"],
                        "p2_5_delta": summary["p2_5"],
                        "p97_5_delta": summary["p97_5"],
                        "mean_delta_ci_lower": ci[0],
                        "mean_delta_ci_upper": ci[1],
                        "bio_sigma_for_delta": extra_sigma,
                        "combined_sigma": combined_sigma,
                        "combined_95pct_noise_band": combined_band_95,
                        "abs_mean_over_combined_sigma": float(abs(mean_delta) / max(combined_sigma, 1.0e-9)),
                        "favors_true_sink": bool(endpoint_value_better(endpoint_key, mean_delta)),
                        "exceeds_1sigma_noise_band": bool(abs(mean_delta) > combined_sigma),
                        "exceeds_95pct_noise_band": bool(abs(mean_delta) > combined_band_95),
                        "mean_delta_ci_excludes_zero": bool((ci[0] > 0.0) or (ci[1] < 0.0)),
                    }
                )
    return rows


def build_mc_sigma_lookup(
    endpoint_rows: Sequence[Mapping[str, object]],
) -> Dict[Tuple[str, str, str], float]:
    grouped: Dict[Tuple[str, str, str], List[float]] = {}
    for row in endpoint_rows:
        plan_id = str(row["plan_id"])
        model_id = str(row["model_id"])
        for endpoint in PRIMARY_ENDPOINTS:
            grouped.setdefault((plan_id, model_id, endpoint.key), []).append(float(row[endpoint.key]))
    return {
        key: float(np.std(np.asarray(values, dtype=np.float64), ddof=0))
        for key, values in grouped.items()
    }


def build_empirical_rank_rows(
    repeated_rank_rows: Sequence[Mapping[str, object]],
    baseline_rank_lookup: Mapping[Tuple[str, str], Mapping[str, object]],
) -> List[Dict[str, object]]:
    grouped: Dict[Tuple[str, str], List[Mapping[str, object]]] = {}
    for row in repeated_rank_rows:
        grouped.setdefault((str(row["plan_id"]), str(row["comparator_id"])), []).append(row)
    out: List[Dict[str, object]] = []
    for key, rows in sorted(grouped.items()):
        baseline = baseline_rank_lookup[key]
        nominal = float(baseline["true_rank_minus_comparator_rank"])
        values = np.asarray([float(row["true_rank_minus_comparator_rank"]) for row in rows], dtype=np.float64)
        if nominal < 0.0:
            same_direction = float(np.mean(values < 0.0))
        elif nominal > 0.0:
            same_direction = float(np.mean(values > 0.0))
        else:
            same_direction = float(np.mean(values == 0.0))
        out.append(
            {
                "plan_id": key[0],
                "comparator_id": key[1],
                "comparator_label": MODEL_LABELS[key[1]],
                "nominal_true_rank_minus_comparator_rank": nominal,
                "n_repeats": int(values.size),
                "mean_empirical_rank_delta": float(np.mean(values)),
                "sd_empirical_rank_delta": float(np.std(values, ddof=0)),
                "fraction_true_better": float(np.mean(values < 0.0)),
                "fraction_equal_rank": float(np.mean(values == 0.0)),
                "fraction_true_worse": float(np.mean(values > 0.0)),
                "probability_same_direction_as_nominal": same_direction,
            }
        )
    return out


def simulate_rank_noise(
    baseline_model_rows: Sequence[Mapping[str, object]],
    *,
    mc_sigma_lookup: Mapping[Tuple[str, str, str], float],
    bio_sigma_lookup: Mapping[str, float],
    rng: np.random.Generator,
    samples: int,
) -> List[Dict[str, object]]:
    by_plan: Dict[str, List[Mapping[str, object]]] = {}
    for row in baseline_model_rows:
        by_plan.setdefault(str(row["plan_id"]), []).append(row)
    rows: List[Dict[str, object]] = []
    for plan_id, model_rows in sorted(by_plan.items()):
        if {str(row["model_id"]) for row in model_rows} != set(PHASE37B_MODEL_IDS):
            continue
        nominal_rank_by_model = {str(row["model_id"]): int(row["model_rank_within_case"]) for row in model_rows}
        shift_samples: Dict[str, List[float]] = {
            cid: []
            for cid in (
                "no_sink",
                "ap_flip_sink",
                "si_flip_sink",
                "local_dropout_sink_20mm",
                "randomized_displacement_sink",
                "uniform_body_sink_mass_matched",
            )
        }
        for _ in range(int(samples)):
            simulated_rows: List[Dict[str, object]] = []
            for row in model_rows:
                model_id = str(row["model_id"])
                sim_row = {"plan_id": plan_id, "model_id": model_id}
                for endpoint in PRIMARY_ENDPOINTS:
                    sigma = float(mc_sigma_lookup.get((plan_id, model_id, endpoint.key), 0.0))
                    if model_id != "no_sink":
                        sigma = math.sqrt(sigma**2 + float(bio_sigma_lookup.get(endpoint.key, 0.0)) ** 2)
                    value = float(row[endpoint.key])
                    if sigma > 0.0:
                        value = max(0.0, float(rng.normal(value, sigma)))
                    sim_row[endpoint.key] = value
                simulated_rows.append(sim_row)
            risk_scores = np.zeros(len(simulated_rows), dtype=np.float64)
            for endpoint in PRIMARY_ENDPOINTS:
                values = [float(sim_row[endpoint.key]) for sim_row in simulated_rows]
                risk_scores += endpoint_z_scores(values, higher_is_better=endpoint.higher_is_better)
            order = np.argsort(np.argsort(risk_scores)) + 1
            rank_lookup = {str(simulated_rows[idx]["model_id"]): int(order[idx]) for idx in range(len(simulated_rows))}
            for comparator_id in shift_samples:
                shift_samples[comparator_id].append(float(rank_lookup["true_vessel_sink"] - rank_lookup[comparator_id]))
        for comparator_id, values_list in shift_samples.items():
            values = np.asarray(values_list, dtype=np.float64)
            nominal_shift = float(nominal_rank_by_model["true_vessel_sink"] - nominal_rank_by_model[comparator_id])
            ci = (float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5)))
            if nominal_shift < 0.0:
                same_direction = float(np.mean(values < 0.0))
            elif nominal_shift > 0.0:
                same_direction = float(np.mean(values > 0.0))
            else:
                same_direction = float(np.mean(values == 0.0))
            rows.append(
                {
                    "plan_id": plan_id,
                    "comparator_id": comparator_id,
                    "comparator_label": MODEL_LABELS[comparator_id],
                    "nominal_true_rank_minus_comparator_rank": nominal_shift,
                    "simulated_mean_rank_delta": float(np.mean(values)),
                    "simulated_rank_delta_sd": float(np.std(values, ddof=0)),
                    "simulated_rank_delta_ci_lower": ci[0],
                    "simulated_rank_delta_ci_upper": ci[1],
                    "simulated_probability_same_direction_as_nominal": same_direction,
                    "simulated_probability_true_better": float(np.mean(values < 0.0)),
                    "noise_qualified_rank_change": bool(
                        nominal_shift != 0.0
                        and ((ci[0] > 0.0) or (ci[1] < 0.0))
                        and same_direction >= 0.80
                    ),
                }
            )
    return rows


def build_baseline_rank_pair_rows(model_rows: Sequence[Mapping[str, object]]) -> List[Dict[str, object]]:
    by_plan: Dict[str, Dict[str, Mapping[str, object]]] = {}
    for row in model_rows:
        by_plan.setdefault(str(row["plan_id"]), {})[str(row["model_id"])] = row
    rows: List[Dict[str, object]] = []
    for plan_id, models in sorted(by_plan.items()):
        true_rank = int(models["true_vessel_sink"]["model_rank_within_case"])
        for comparator_id in (
            "no_sink",
            "ap_flip_sink",
            "si_flip_sink",
            "local_dropout_sink_20mm",
            "randomized_displacement_sink",
            "uniform_body_sink_mass_matched",
        ):
            comp_rank = int(models[comparator_id]["model_rank_within_case"])
            rows.append(
                {
                    "plan_id": plan_id,
                    "comparator_id": comparator_id,
                    "comparator_label": MODEL_LABELS[comparator_id],
                    "true_rank": int(true_rank),
                    "comparator_rank": int(comp_rank),
                    "true_rank_minus_comparator_rank": int(true_rank - comp_rank),
                }
            )
    return rows


def build_repeated_rank_pair_rows(model_rows: Sequence[Mapping[str, object]]) -> List[Dict[str, object]]:
    by_group: Dict[Tuple[str, str], Dict[str, Mapping[str, object]]] = {}
    for row in model_rows:
        by_group.setdefault((str(row["repeat_id"]), str(row["plan_id"])), {})[str(row["model_id"])] = row
    rows: List[Dict[str, object]] = []
    for (repeat_id, plan_id), models in sorted(by_group.items()):
        true_rank = int(models["true_vessel_sink"]["model_rank_within_case"])
        for comparator_id in (
            "no_sink",
            "ap_flip_sink",
            "si_flip_sink",
            "local_dropout_sink_20mm",
            "randomized_displacement_sink",
            "uniform_body_sink_mass_matched",
        ):
            comp_rank = int(models[comparator_id]["model_rank_within_case"])
            rows.append(
                {
                    "repeat_id": repeat_id,
                    "plan_id": plan_id,
                    "comparator_id": comparator_id,
                    "comparator_label": MODEL_LABELS[comparator_id],
                    "true_rank": int(true_rank),
                    "comparator_rank": int(comp_rank),
                    "true_rank_minus_comparator_rank": int(true_rank - comp_rank),
                }
            )
    return rows


def build_quick_assessment(
    noise_rows: Sequence[Mapping[str, object]],
    rank_rows: Sequence[Mapping[str, object]],
    case_ids: Sequence[str],
    repeat_count: int,
) -> str:
    endpoint_hits = sum(1 for row in noise_rows if bool(row["exceeds_95pct_noise_band"]))
    rank_hits = sum(1 for row in rank_rows if bool(row["noise_qualified_rank_change"]))
    lines = [
        "# Phase 37B quick assessment",
        "",
        f"- Repeated subset cases: `{', '.join(case_ids)}`.",
        f"- Repeat combinations executed: `{repeat_count}`.",
        f"- True-vs-surrogate endpoint deltas exceeding the 95% combined MC+uptake band: `{endpoint_hits}` / `{len(noise_rows)}`.",
        f"- True-vs-surrogate rank comparisons with noise-qualified rank changes: `{rank_hits}` / `{len(rank_rows)}`.",
        "- Interpretation: this package tests whether the anatomical sink remains distinguishable from stronger spatial falsifications after repeated transport and uptake-sensitivity uncertainty are both considered.",
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    out_root = args.out_root.resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(int(args.bootstrap_seed))

    case_ids = list(args.only_case_ids)
    if not case_ids:
        case_ids = [
            str(value)
            for value in json.loads(
                (args.phase35_root / "phase35_reproducibility_manifest.json").read_text(encoding="utf-8")
            )["selected_case_ids"]
        ]

    phase25_config = json.loads((args.phase25_run_root / "phase25_config.json").read_text(encoding="utf-8"))
    existing_endpoint_rows, existing_assay_rows, repeat_manifest = load_repeat_rows_for_existing_models(
        args.phase35_root.resolve(),
        case_ids=case_ids,
    )
    surrogate_endpoint_rows, surrogate_assay_rows = evaluate_repeat_surrogates(
        repeat_manifest=repeat_manifest,
        phase25_config=phase25_config,
        case_ids=case_ids,
        history_interval=int(args.history_interval),
        progress_interval=int(args.progress_interval),
        verbose_pde=bool(args.verbose_pde),
    )

    endpoint_rows = existing_endpoint_rows + surrogate_endpoint_rows
    assay_rows = existing_assay_rows + surrogate_assay_rows
    assign_model_ranks(endpoint_rows, group_keys=("repeat_id", "plan_id"))
    repeated_rank_rows = build_repeated_rank_pair_rows(endpoint_rows)

    baseline_model_rows = build_baseline_subset_model_rows(args.phase37a_root.resolve(), case_ids=case_ids)
    baseline_rank_rows = build_baseline_rank_pair_rows(baseline_model_rows)
    baseline_rank_lookup = {(str(row["plan_id"]), str(row["comparator_id"])): row for row in baseline_rank_rows}

    bio_sigma_lookup = build_biology_sigma_lookup(args.phase26_run_root.resolve())
    noise_rows = build_true_vs_surrogate_noise_rows(
        endpoint_rows,
        bio_sigma_lookup=bio_sigma_lookup,
        rng=rng,
        bootstrap_samples=int(args.bootstrap_samples),
    )
    empirical_rank_rows = build_empirical_rank_rows(
        repeated_rank_rows,
        baseline_rank_lookup=baseline_rank_lookup,
    )
    mc_sigma_lookup = build_mc_sigma_lookup(endpoint_rows)
    simulated_rank_rows = simulate_rank_noise(
        baseline_model_rows,
        mc_sigma_lookup=mc_sigma_lookup,
        bio_sigma_lookup=bio_sigma_lookup,
        rng=rng,
        samples=int(args.rank_sim_samples),
    )

    write_csv(out_root / "phase37b_endpoint_table.csv", endpoint_rows)
    write_csv(out_root / "phase37b_assay_table.csv", assay_rows)
    write_csv(out_root / "phase37b_baseline_subset_model_ranks.csv", baseline_model_rows)
    write_csv(out_root / "phase37b_repeat_model_rank_pairs.csv", repeated_rank_rows)
    write_csv(out_root / "phase37b_true_vs_surrogate_noise_table.csv", noise_rows)
    write_csv(out_root / "phase37b_true_vs_surrogate_empirical_rank_summary.csv", empirical_rank_rows)
    write_csv(out_root / "phase37b_true_vs_surrogate_rank_noise.csv", simulated_rank_rows)
    write_json(
        out_root / "phase37b_reproducibility_manifest.json",
        {
            "phase": "37B",
            "description": "Repeated-subset uncertainty overlay for stronger true-vs-surrogate sink comparisons.",
            "phase35_root": str(args.phase35_root.resolve()),
            "phase37a_root": str(args.phase37a_root.resolve()),
            "phase25_run_root": str(args.phase25_run_root.resolve()),
            "phase26_run_root": str(args.phase26_run_root.resolve()),
            "selected_case_ids": list(case_ids),
            "model_ids": list(PHASE37B_MODEL_IDS),
            "repeat_count": int(len(repeat_manifest)),
        },
    )
    (out_root / "phase37b_quick_assessment.md").write_text(
        build_quick_assessment(
            noise_rows,
            simulated_rank_rows,
            case_ids=case_ids,
            repeat_count=len(repeat_manifest),
        ),
        encoding="utf-8",
    )

    print("=== PHASE 37B VESSEL FALSIFICATION UNCERTAINTY COMPLETE ===")
    print(f"Output root: {out_root}")
    print(f"Noise table: {out_root / 'phase37b_true_vs_surrogate_noise_table.csv'}")
    print(f"Rank noise table: {out_root / 'phase37b_true_vs_surrogate_rank_noise.csv'}")
    print(f"Quick assessment: {out_root / 'phase37b_quick_assessment.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
