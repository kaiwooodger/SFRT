#!/usr/bin/env python3
"""Phase 40: small smoothing-kernel sensitivity on representative cohort cases.

This phase reuses saved Phase 33 base/spot component dose CSVs, reapplies a
small set of body-masked Gaussian smoothing kernels, recalibrates the
base/spot ratio for each kernel, and then reruns the Phase 34 biology model on
selected cases. It is intended as a lightweight stress test of whether the main
revision conclusion survives modest smoothing-kernel changes:

1. non-local biology changes endpoint interpretation relative to physical-only
2. the anatomical vascular sink remains a secondary modifier relative to the
   larger physical-to-biology reinterpretation
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np

from run_phase26_vascular_sink_ablation import (
    PRIMARY_ENDPOINTS,
    assign_ranks,
    build_oar_reinterpretation_rows,
    build_rank_shift_rows,
    evaluate_bystander_mode,
    evaluate_physical_only,
    write_csv,
    write_json,
)
from run_phase30_phase28_topas_true_lattice_delivery import (
    load_topas_csv_grid,
    search_component_ratio,
    smooth_dose_grid,
)
from run_phase33_phase32_topas_cohort import sphere_union_mask
from run_phase34_phase32_bio_cohort import (
    add_structure_aliases,
    harmonize_context_to_dose_shape,
    load_case_context,
)


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase32-root",
        type=Path,
        default=root / "public_results" / "phase32_site_specific_cohort_regenerated",
    )
    parser.add_argument(
        "--phase33-root",
        type=Path,
        default=Path("/Users/kw/Documents/Playground/vhee_topas/runs/phase33_phase32_topas_cohort"),
    )
    parser.add_argument(
        "--phase25-config",
        type=Path,
        default=root / "public_results" / "phase25_safe_core" / "phase25_config.json",
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        default=root / "public_results" / "revision_checks_20260617" / "step10_smoothing_kernel_sensitivity",
    )
    parser.add_argument("--case-ids", type=str, nargs="+", default=["case03", "case04", "case06"])
    parser.add_argument("--sigma-mm", type=float, nargs="+", default=[2.0, 4.0, 6.0])
    parser.add_argument("--vertex-radius-mm", type=float, default=5.0)
    parser.add_argument("--ratio-min", type=float, default=0.01)
    parser.add_argument("--ratio-max", type=float, default=100.0)
    parser.add_argument("--ratio-count", type=int, default=121)
    parser.add_argument("--target-ptv-d95-gy", type=float, default=3.5)
    parser.add_argument("--target-peak-mean-gy", type=float, default=15.0)
    parser.add_argument("--history-interval", type=int, default=20)
    parser.add_argument("--progress-interval", type=int, default=100)
    parser.add_argument("--brainstem-limit-gy", type=float, default=30.0)
    parser.add_argument("--cord-limit-gy", type=float, default=85.0)
    parser.add_argument("--parotid-r-limit-gy", type=float, default=60.0)
    parser.add_argument("--thyroid-limit-gy", type=float, default=50.0)
    parser.add_argument("--body-dmax-limit-gy", type=float, default=400.0)
    parser.add_argument("--verbose-pde", action="store_true")
    return parser.parse_args()


def load_json(path: Path) -> Mapping[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def endpoint_row_from_result(
    *,
    case_id: str,
    sigma_mm: float,
    mode: str,
    result: Mapping[str, object],
) -> Dict[str, object]:
    endpoints = dict(result["endpoints"])
    supplemental = dict(result["supplemental"])
    row: Dict[str, object] = {
        "plan_id": case_id,
        "sigma_mm": float(sigma_mm),
        "mode": mode,
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


def smoothing_args(args: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        ratio_min=float(args.ratio_min),
        ratio_max=float(args.ratio_max),
        ratio_count=int(args.ratio_count),
        target_ptv_d95_gy=float(args.target_ptv_d95_gy),
        target_peak_mean_gy=float(args.target_peak_mean_gy),
    )


def main() -> int:
    args = parse_args()
    out_root = args.out_root.resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    config = load_json(args.phase25_config.resolve())
    endpoint_rows: List[Dict[str, object]] = []
    summary_rows: List[Dict[str, object]] = []

    for case_id in [str(v) for v in args.case_ids]:
        case_root = args.phase33_root.resolve() / case_id
        case_summary = load_json(case_root / "phase33_case_summary.json")
        base_csv = Path(str(case_summary["component_outputs"]["base_component"]["dose_csv"]))
        spot_csv = Path(str(case_summary["component_outputs"]["spot_component"]["dose_csv"]))
        base_raw = np.asarray(load_topas_csv_grid(base_csv), dtype=np.float32)
        spot_raw = np.asarray(load_topas_csv_grid(spot_csv), dtype=np.float32)

        structures, axes_mm, voxel_size_mm, vertices_mm = load_case_context(args.phase32_root.resolve() / case_id / "phantom_context.npz")
        structures = add_structure_aliases(structures)
        structures, axes_mm = harmonize_context_to_dose_shape(
            structures=structures,
            axes_mm=axes_mm,
            dose_shape=tuple(int(v) for v in base_raw.shape),
        )
        voxel_size_mm = (
            float(axes_mm["x"][1] - axes_mm["x"][0]),
            float(axes_mm["y"][1] - axes_mm["y"][0]),
            float(axes_mm["z"][1] - axes_mm["z"][0]),
        )
        voxel_volume_cc = float(voxel_size_mm[0] * voxel_size_mm[1] * voxel_size_mm[2] / 1000.0)
        body_mask = np.asarray(structures["BODY"], dtype=bool)
        peak_mask = sphere_union_mask(axes_mm, vertices_mm, float(args.vertex_radius_mm)) & np.asarray(structures["PTV"], dtype=bool)

        for sigma_mm in [float(v) for v in args.sigma_mm]:
            base_processed = smooth_dose_grid(base_raw, axes_mm=axes_mm, body_mask=body_mask, sigma_mm=sigma_mm)
            spot_processed = smooth_dose_grid(spot_raw, axes_mm=axes_mm, body_mask=body_mask, sigma_mm=sigma_mm)
            calibration = search_component_ratio(
                base_dose=base_processed,
                spot_dose=spot_processed,
                ptv_mask=np.asarray(structures["PTV"], dtype=bool),
                peak_mask=peak_mask,
                args=smoothing_args(args),
            )
            combined = float(calibration["global_scale"]) * (
                base_processed + float(calibration["spot_to_base_ratio"]) * spot_processed
            )

            physical = evaluate_physical_only(
                physical_dose=combined,
                structures=structures,
                axes_mm=axes_mm,
                spots_mm=vertices_mm,
                voxel_volume_cc=voxel_volume_cc,
                voxel_size_mm=voxel_size_mm,
                prescription_gy=float(config.get("prescription_gy", 3.5)),
                alpha=float(config["bio_parameters"]["alpha"]),
                beta=float(config["bio_parameters"]["beta"]),
            )
            no_sink = evaluate_bystander_mode(
                physical_dose=combined,
                structures=structures,
                axes_mm=axes_mm,
                spots_mm=vertices_mm,
                config=config,
                mode="bystander_no_sink",
                uptake_scale=1.0,
                voxel_volume_cc=voxel_volume_cc,
                voxel_size_mm=voxel_size_mm,
                prescription_gy=float(config.get("prescription_gy", 3.5)),
                history_interval=int(args.history_interval),
                progress_interval=int(args.progress_interval),
                verbose_pde=bool(args.verbose_pde),
            )
            with_sink = evaluate_bystander_mode(
                physical_dose=combined,
                structures=structures,
                axes_mm=axes_mm,
                spots_mm=vertices_mm,
                config=config,
                mode="bystander_with_sink",
                uptake_scale=1.0,
                voxel_volume_cc=voxel_volume_cc,
                voxel_size_mm=voxel_size_mm,
                prescription_gy=float(config.get("prescription_gy", 3.5)),
                history_interval=int(args.history_interval),
                progress_interval=int(args.progress_interval),
                verbose_pde=bool(args.verbose_pde),
            )

            case_rows = [
                endpoint_row_from_result(case_id=case_id, sigma_mm=sigma_mm, mode="physical_only", result=physical),
                endpoint_row_from_result(case_id=case_id, sigma_mm=sigma_mm, mode="bystander_no_sink", result=no_sink),
                endpoint_row_from_result(case_id=case_id, sigma_mm=sigma_mm, mode="bystander_with_sink", result=with_sink),
            ]
            endpoint_rows.extend(case_rows)

            case_rank_rows = [{k: v for k, v in row.items() if k != "sigma_mm"} for row in case_rows]
            assign_ranks(case_rank_rows)
            oar_rows = build_oar_reinterpretation_rows(case_rank_rows, args=args)
            rank_rows = build_rank_shift_rows(case_rank_rows)
            biology_adds_failure = any(str(row["reinterpretation"]) == "biology_adds_failure" for row in oar_rows)

            phys_ep = dict(physical["endpoints"])
            no_sink_ep = dict(no_sink["endpoints"])
            with_sink_ep = dict(with_sink["endpoints"])
            mean_abs_bio_shift = float(np.mean([abs(float(no_sink_ep[ep.key]) - float(phys_ep[ep.key])) for ep in PRIMARY_ENDPOINTS]))
            mean_abs_sink_shift = float(np.mean([abs(float(with_sink_ep[ep.key]) - float(no_sink_ep[ep.key])) for ep in PRIMARY_ENDPOINTS]))
            sink_to_bio_ratio = float(mean_abs_sink_shift / max(mean_abs_bio_shift, 1.0e-9))
            brainstem_row = next((row for row in oar_rows if str(row["metric"]) == "brainstem_d2"), None)
            summary_rows.append(
                {
                    "case_id": case_id,
                    "sigma_mm": sigma_mm,
                    "spot_to_base_ratio": float(calibration["spot_to_base_ratio"]),
                    "global_scale": float(calibration["global_scale"]),
                    "predicted_ptv_d95_gy": float(calibration["predicted_ptv_d95_gy"]),
                    "predicted_peak_mean_gy": float(calibration["predicted_peak_mean_gy"]),
                    "physical_brainstem_d2": float(phys_ep["brainstem_d2"]),
                    "no_sink_brainstem_d2": float(no_sink_ep["brainstem_d2"]),
                    "with_sink_brainstem_d2": float(with_sink_ep["brainstem_d2"]),
                    "physical_pvdr": float(phys_ep["pvdr"]),
                    "no_sink_pvdr": float(no_sink_ep["pvdr"]),
                    "with_sink_pvdr": float(with_sink_ep["pvdr"]),
                    "physical_spill_0_5": float(phys_ep["spill_shell_0_5_mean"]),
                    "no_sink_spill_0_5": float(no_sink_ep["spill_shell_0_5_mean"]),
                    "with_sink_spill_0_5": float(with_sink_ep["spill_shell_0_5_mean"]),
                    "mean_abs_bio_shift": mean_abs_bio_shift,
                    "mean_abs_sink_shift": mean_abs_sink_shift,
                    "sink_to_bio_ratio": sink_to_bio_ratio,
                    "biology_adds_failure": bool(biology_adds_failure),
                    "brainstem_reinterpretation": str(brainstem_row["reinterpretation"]) if brainstem_row is not None else "",
                    "subset_with_sink_vs_physical_shift": int(rank_rows[0]["with_sink_vs_physical_shift"]) if len(rank_rows) == 1 else "",
                    "subset_with_sink_vs_no_sink_shift": int(rank_rows[0]["with_sink_vs_no_sink_shift"]) if len(rank_rows) == 1 else "",
                }
            )

    write_csv(out_root / "phase40_endpoint_rows.csv", endpoint_rows)
    write_csv(out_root / "phase40_summary_rows.csv", summary_rows)

    grouped_by_sigma: Dict[float, List[Mapping[str, object]]] = {}
    for row in summary_rows:
        grouped_by_sigma.setdefault(float(row["sigma_mm"]), []).append(row)

    sigma_rows: List[Dict[str, object]] = []
    lines: List[str] = [
        "# Phase 40 smoothing-kernel sensitivity",
        "",
        f"- Cases tested: `{', '.join(str(v) for v in args.case_ids)}`",
        f"- Smoothing kernels: `{', '.join(f'{float(v):.1f} mm' for v in args.sigma_mm)}`",
        "",
    ]
    for sigma_mm in sorted(grouped_by_sigma):
        rows = grouped_by_sigma[sigma_mm]
        n_fail = sum(bool(row["biology_adds_failure"]) for row in rows)
        mean_bio = float(np.mean([float(row["mean_abs_bio_shift"]) for row in rows]))
        mean_sink = float(np.mean([float(row["mean_abs_sink_shift"]) for row in rows]))
        mean_ratio = float(np.mean([float(row["sink_to_bio_ratio"]) for row in rows]))
        sigma_rows.append(
            {
                "sigma_mm": sigma_mm,
                "cases_tested": int(len(rows)),
                "biology_added_failures": int(n_fail),
                "mean_abs_bio_shift": mean_bio,
                "mean_abs_sink_shift": mean_sink,
                "mean_sink_to_bio_ratio": mean_ratio,
                "primary_conclusion_survives": bool((n_fail >= 2) and (mean_sink < mean_bio)),
            }
        )
        lines.extend(
            [
                f"## Sigma {sigma_mm:.1f} mm",
                "",
                f"- Biology-added brainstem failures: `{n_fail}` / `{len(rows)}`",
                f"- Mean |no-sink minus physical| across primary endpoints: `{mean_bio:.3f}`",
                f"- Mean |with-sink minus no-sink| across primary endpoints: `{mean_sink:.3f}`",
                f"- Mean sink-to-biology shift ratio: `{mean_ratio:.3f}`",
                f"- Conclusion at this kernel: `{'survives' if (n_fail >= 2 and mean_sink < mean_bio) else 'does not survive cleanly'}`",
                "",
            ]
        )
    write_csv(out_root / "phase40_sigma_summary.csv", sigma_rows)
    (out_root / "phase40_quick_assessment.md").write_text("\n".join(lines), encoding="utf-8")
    write_json(
        out_root / "phase40_manifest.json",
        {
            "phase32_root": str(args.phase32_root.resolve()),
            "phase33_root": str(args.phase33_root.resolve()),
            "phase25_config": str(args.phase25_config.resolve()),
            "case_ids": [str(v) for v in args.case_ids],
            "sigma_mm": [float(v) for v in args.sigma_mm],
        },
    )

    print("=== PHASE 40 SMOOTHING-KERNEL SENSITIVITY COMPLETE ===")
    print(f"Output root: {out_root}")
    print(f"Case-level summary: {out_root / 'phase40_summary_rows.csv'}")
    print(f"Sigma summary: {out_root / 'phase40_sigma_summary.csv'}")
    print(f"Quick assessment: {out_root / 'phase40_quick_assessment.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
