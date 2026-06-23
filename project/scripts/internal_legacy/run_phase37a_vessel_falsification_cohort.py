#!/usr/bin/env python3
"""Phase 37A: stronger vessel falsification and surrogate sink comparison on the 10-case cohort."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np

from phase37_sink_falsification_utils import (
    COMPARISON_ENDPOINT_KEYS,
    ASSAY_KEYS_FOR_SUMMARY,
    ENDPOINT_DIRECTION,
    ENDPOINT_LABELS,
    ENDPOINT_UNITS,
    MODEL_LABELS,
    PHASE37A_COMPARATOR_IDS,
    add_structure_aliases,
    assay_row_from_result,
    assay_value_better,
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
from run_phase31_publication_package import bootstrap_interval, load_csv_rows, write_csv, write_json


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase32-root",
        type=Path,
        default=root / "runs" / "phase32_site_specific_template_phantoms",
    )
    parser.add_argument(
        "--phase33-root",
        type=Path,
        default=root / "runs" / "phase33_phase32_topas_cohort",
    )
    parser.add_argument(
        "--phase34-root",
        type=Path,
        default=root / "runs" / "phase33_phase32_topas_cohort" / "phase34_bio_cohort",
    )
    parser.add_argument(
        "--phase25-run-root",
        type=Path,
        default=root / "runs" / "linac_6mv_headneck_detailed_voxel_lattice_sfrt_phase25_safe_core_plan_library_1e5",
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        default=root / "runs" / "phase37a_vessel_falsification_cohort",
    )
    parser.add_argument("--only-case-ids", type=str, nargs="*", default=[])
    parser.add_argument("--max-cases", type=int, default=0)
    parser.add_argument("--history-interval", type=int, default=20)
    parser.add_argument("--progress-interval", type=int, default=100)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=36)
    parser.add_argument("--verbose-pde", action="store_true")
    return parser.parse_args()


def assign_model_ranks_within_case(endpoint_rows: List[Dict[str, object]]) -> None:
    rows_by_plan: Dict[str, List[Dict[str, object]]] = {}
    for row in endpoint_rows:
        if str(row["model_id"]) == "physical_only":
            row["model_risk_score"] = ""
            row["model_rank_within_case"] = ""
            continue
        rows_by_plan.setdefault(str(row["plan_id"]), []).append(row)
    for rows in rows_by_plan.values():
        risk_scores = np.zeros(len(rows), dtype=np.float64)
        for endpoint in PRIMARY_ENDPOINTS:
            values = [float(row[endpoint.key]) for row in rows]
            risk_scores += endpoint_z_scores(values, higher_is_better=endpoint.higher_is_better)
        order = np.argsort(np.argsort(risk_scores)) + 1
        for idx, row in enumerate(rows):
            row["model_risk_score"] = float(risk_scores[idx])
            row["model_rank_within_case"] = int(order[idx])


def build_true_vs_comparator_delta_rows(
    endpoint_rows: Sequence[Mapping[str, object]],
    assay_rows: Sequence[Mapping[str, object]],
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    endpoint_lookup = {
        (str(row["plan_id"]), str(row["model_id"])): row for row in endpoint_rows
    }
    assay_lookup = {
        (str(row["plan_id"]), str(row["model_id"])): row for row in assay_rows
    }
    plan_ids = sorted({str(row["plan_id"]) for row in endpoint_rows})
    endpoint_delta_rows: List[Dict[str, object]] = []
    assay_delta_rows: List[Dict[str, object]] = []
    for plan_id in plan_ids:
        true_row = endpoint_lookup[(plan_id, "true_vessel_sink")]
        case_label = str(true_row["case_label"])
        site_group = str(true_row["site_group"])
        for comparator_id in PHASE37A_COMPARATOR_IDS:
            comp_row = endpoint_lookup[(plan_id, comparator_id)]
            for endpoint_key in COMPARISON_ENDPOINT_KEYS:
                delta = float(true_row[endpoint_key]) - float(comp_row[endpoint_key])
                endpoint_delta_rows.append(
                    {
                        "plan_id": plan_id,
                        "case_label": case_label,
                        "site_group": site_group,
                        "true_model_id": "true_vessel_sink",
                        "true_model_label": MODEL_LABELS["true_vessel_sink"],
                        "comparator_id": comparator_id,
                        "comparator_label": MODEL_LABELS[comparator_id],
                        "metric": endpoint_key,
                        "label": ENDPOINT_LABELS[endpoint_key],
                        "units": ENDPOINT_UNITS[endpoint_key],
                        "true_value": float(true_row[endpoint_key]),
                        "comparator_value": float(comp_row[endpoint_key]),
                        "true_minus_comparator": delta,
                        "favors_true_sink": bool(endpoint_value_better(endpoint_key, delta)),
                    }
                )
            true_assay = assay_lookup[(plan_id, "true_vessel_sink")]
            comp_assay = assay_lookup[(plan_id, comparator_id)]
            for assay_key in ASSAY_KEYS_FOR_SUMMARY:
                delta = float(true_assay[assay_key]) - float(comp_assay[assay_key])
                assay_delta_rows.append(
                    {
                        "plan_id": plan_id,
                        "case_label": case_label,
                        "site_group": site_group,
                        "true_model_id": "true_vessel_sink",
                        "true_model_label": MODEL_LABELS["true_vessel_sink"],
                        "comparator_id": comparator_id,
                        "comparator_label": MODEL_LABELS[comparator_id],
                        "metric": assay_key,
                        "label": assay_key,
                        "units": "a.u.",
                        "true_value": float(true_assay[assay_key]),
                        "comparator_value": float(comp_assay[assay_key]),
                        "true_minus_comparator": delta,
                        "favors_true_sink": bool(assay_value_better(assay_key, delta)),
                    }
                )
    return endpoint_delta_rows, assay_delta_rows


def summarize_delta_rows(
    delta_rows: Sequence[Mapping[str, object]],
    *,
    rng: np.random.Generator,
    bootstrap_samples: int,
) -> List[Dict[str, object]]:
    grouped: Dict[Tuple[str, str], List[Mapping[str, object]]] = {}
    for row in delta_rows:
        grouped.setdefault((str(row["comparator_id"]), str(row["metric"])), []).append(row)
    out: List[Dict[str, object]] = []
    for (comparator_id, metric), rows in sorted(grouped.items()):
        values = [float(row["true_minus_comparator"]) for row in rows]
        ci = bootstrap_interval(values, rng=rng, samples=int(bootstrap_samples), reducer=np.mean)
        out.append(
            {
                "comparator_id": comparator_id,
                "comparator_label": MODEL_LABELS[comparator_id],
                "metric": metric,
                "label": str(rows[0]["label"]),
                "units": str(rows[0]["units"]),
                "n_cases": int(len(rows)),
                "mean_true_minus_comparator": float(np.mean(values)),
                "median_true_minus_comparator": float(np.median(values)),
                "ci_lower": ci[0],
                "ci_upper": ci[1],
                "cases_favoring_true_sink": int(sum(bool(row["favors_true_sink"]) for row in rows)),
                "fraction_favoring_true_sink": float(np.mean([bool(row["favors_true_sink"]) for row in rows])),
                "ci_excludes_zero": bool((ci[0] > 0.0) or (ci[1] < 0.0)),
            }
        )
    return out


def build_quick_assessment(
    endpoint_summary_rows: Sequence[Mapping[str, object]],
    endpoint_rows: Sequence[Mapping[str, object]],
) -> str:
    key_comparators = {
        "ap_flip_sink",
        "si_flip_sink",
        "local_dropout_sink_20mm",
        "randomized_displacement_sink",
        "uniform_body_sink_mass_matched",
    }
    key_metrics = {"pvdr", "spill_shell_0_5_mean", "spill_shell_5_15_mean", "parotid_r_mean", "brainstem_d2"}
    key_hits = [
        row
        for row in endpoint_summary_rows
        if str(row["comparator_id"]) in key_comparators and str(row["metric"]) in key_metrics and bool(row["ci_excludes_zero"])
    ]
    lines = [
        "# Phase 37A quick assessment",
        "",
        f"- Cases analyzed: `{len({str(row['plan_id']) for row in endpoint_rows})}`.",
        f"- Biological sink models evaluated per case: `{len({str(row['model_id']) for row in endpoint_rows if str(row['model_id']) != 'physical_only'})}`.",
        f"- Key true-vs-falsified comparator endpoint summaries with bootstrap CI excluding zero: `{len(key_hits)}`.",
        "- Interpretation: if the anatomical sink remains systematically different from AP/SI flips, local dropout, randomized displacement, or uniform washout surrogates, that supports an anatomy-specific rather than generic-washout explanation of the sink effect.",
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    out_root = args.out_root.resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(int(args.bootstrap_seed))

    phase25_config = json.loads((args.phase25_run_root / "phase25_config.json").read_text(encoding="utf-8"))
    phase32_manifest = {str(row["template_id"]): row for row in load_csv_rows(args.phase32_root / "phase32_case_manifest.csv")}
    phase33_rows = [row for row in load_csv_rows(args.phase33_root / "phase33_case_manifest.csv") if str(row["status"]) == "completed"]
    if args.only_case_ids:
        allowed = set(args.only_case_ids)
        phase33_rows = [row for row in phase33_rows if str(row["case_id"]) in allowed]
    if int(args.max_cases) > 0:
        phase33_rows = phase33_rows[: int(args.max_cases)]

    base_endpoint_rows_raw = [
        row
        for row in load_csv_rows(args.phase34_root / "phase34_endpoint_table.csv")
        if not args.only_case_ids or str(row["plan_id"]) in set(args.only_case_ids)
    ]
    base_assay_rows_raw = [
        row
        for row in load_csv_rows(args.phase34_root / "phase34_assay_proxy_table.csv")
        if not args.only_case_ids or str(row["plan_id"]) in set(args.only_case_ids)
    ]
    endpoint_rows, assay_rows = rows_from_phase34(base_endpoint_rows_raw, base_assay_rows_raw)
    model_meta_rows: List[Dict[str, object]] = []

    for row in phase33_rows:
        case_id = str(row["case_id"])
        phase32_row = phase32_manifest[case_id]
        case_label = str(row["case_label"])
        site_group = str(phase32_row["site_group"])
        context_path = Path(str(row.get("phase33_context_npz") or phase32_row["phantom_context_npz"]))
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
            "blurred_vessel_sink_sigma10mm",
            "peri_gtv_shell_sink_5_20mm",
            "distance_decay_sink_lambda15mm",
        ):
            random_seed = stable_seed_from_key(f"phase37a:{case_id}:{model_id}")
            uptake_tensor, meta = build_sink_uptake_tensor(
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
                history_interval=int(args.history_interval),
                progress_interval=int(args.progress_interval),
                verbose_pde=bool(args.verbose_pde),
            )
            endpoint_row = endpoint_row_from_result(
                plan_id=case_id,
                case_label=case_label,
                site_group=site_group,
                model_id=model_id,
                result=result,
            )
            assay_row = assay_row_from_result(
                plan_id=case_id,
                case_label=case_label,
                site_group=site_group,
                model_id=model_id,
                result=result,
            )
            endpoint_rows.append(endpoint_row)
            assay_rows.append(assay_row)
            model_meta_rows.append(
                {
                    "plan_id": case_id,
                    "case_label": case_label,
                    "site_group": site_group,
                    **meta,
                }
            )

    assign_model_ranks_within_case(endpoint_rows)
    endpoint_delta_rows, assay_delta_rows = build_true_vs_comparator_delta_rows(endpoint_rows, assay_rows)
    endpoint_summary_rows = summarize_delta_rows(
        endpoint_delta_rows,
        rng=rng,
        bootstrap_samples=int(args.bootstrap_samples),
    )
    assay_summary_rows = summarize_delta_rows(
        assay_delta_rows,
        rng=rng,
        bootstrap_samples=int(args.bootstrap_samples),
    )

    write_csv(out_root / "phase37a_endpoint_table.csv", endpoint_rows)
    write_csv(out_root / "phase37a_assay_table.csv", assay_rows)
    write_csv(out_root / "phase37a_model_metadata.csv", model_meta_rows)
    write_csv(out_root / "phase37a_true_vs_surrogate_endpoint_deltas.csv", endpoint_delta_rows)
    write_csv(out_root / "phase37a_true_vs_surrogate_assay_deltas.csv", assay_delta_rows)
    write_csv(out_root / "phase37a_falsification_endpoint_summary.csv", endpoint_summary_rows)
    write_csv(out_root / "phase37a_falsification_assay_summary.csv", assay_summary_rows)
    write_json(
        out_root / "phase37a_reproducibility_manifest.json",
        {
            "phase": "37A",
            "description": "Stronger vessel falsification and surrogate sink comparison on the Phase 33/34 cohort.",
            "phase32_root": str(args.phase32_root.resolve()),
            "phase33_root": str(args.phase33_root.resolve()),
            "phase34_root": str(args.phase34_root.resolve()),
            "phase25_run_root": str(args.phase25_run_root.resolve()),
            "selected_case_ids": [str(row["case_id"]) for row in phase33_rows],
            "sink_models": list(MODEL_LABELS.keys()),
        },
    )
    (out_root / "phase37a_quick_assessment.md").write_text(
        build_quick_assessment(endpoint_summary_rows, endpoint_rows),
        encoding="utf-8",
    )

    print("=== PHASE 37A VESSEL FALSIFICATION COHORT COMPLETE ===")
    print(f"Output root: {out_root}")
    print(f"Endpoint table: {out_root / 'phase37a_endpoint_table.csv'}")
    print(f"Endpoint summary: {out_root / 'phase37a_falsification_endpoint_summary.csv'}")
    print(f"Quick assessment: {out_root / 'phase37a_quick_assessment.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
