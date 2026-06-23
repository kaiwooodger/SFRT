#!/usr/bin/env python3
"""Phase 34: biology analysis package for the Phase 33 Phase 32 cohort."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np
from scipy import ndimage

from run_phase26_vascular_sink_ablation import (
    MODE_LABELS,
    assign_ranks,
    build_delta_rows,
    build_oar_reinterpretation_rows,
    build_rank_shift_rows,
    evaluate_bystander_mode,
    evaluate_physical_only,
    write_csv,
    write_json,
)
from run_phase30_phase28_topas_true_lattice_delivery import load_topas_csv_grid


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


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase32-root",
        type=Path,
        default=first_existing(
            [
                root / "public_results" / "phase32_site_specific_cohort",
                Path("/Users/kw/Desktop/SFRT_repo_main_update/project/runs/phase32_site_specific_template_phantoms"),
                Path("/Users/kw/Documents/Playground/vhee_topas/runs/phase32_site_specific_template_phantoms"),
            ]
        ),
    )
    parser.add_argument(
        "--phase33-root",
        type=Path,
        default=first_existing(
            [
                Path("/Users/kw/Documents/Playground/vhee_topas/runs/paper_refresh_hihist_10x/phase33_phase32_topas_cohort_hihist"),
                Path("/Users/kw/Documents/Playground/vhee_topas/runs/phase33_phase32_topas_cohort"),
                Path("/Users/kw/Desktop/SFRT_repo_main_update/project/runs/phase33_phase32_topas_cohort"),
            ]
        ),
    )
    parser.add_argument(
        "--phase33-manifest-root",
        type=Path,
        default=first_existing(
            [
                root / "public_results" / "phase33_34_cohort",
                Path("/Users/kw/Desktop/SFRT_repo_main_update/project/runs/phase33_phase32_topas_cohort"),
                Path("/Users/kw/Desktop/SFRT_Submission_reproducibility_bundle_ready/project/runs/phase33_phase32_topas_cohort"),
            ]
        ),
    )
    parser.add_argument(
        "--phase25-run-root",
        type=Path,
        default=first_existing(
            [
                root / "public_results" / "phase25_safe_core",
                Path("/Users/kw/Desktop/SFRT_repo_main_update/project/public_results/phase25_safe_core"),
                root / "runs" / "linac_6mv_headneck_detailed_voxel_lattice_sfrt_phase25_safe_core_plan_library_1e5",
            ]
        ),
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        default=None,
    )
    parser.add_argument("--history-interval", type=int, default=20)
    parser.add_argument("--progress-interval", type=int, default=100)
    parser.add_argument("--cord-limit-gy", type=float, default=85.0)
    parser.add_argument("--brainstem-limit-gy", type=float, default=30.0)
    parser.add_argument("--parotid-r-limit-gy", type=float, default=60.0)
    parser.add_argument("--thyroid-limit-gy", type=float, default=50.0)
    parser.add_argument("--body-dmax-limit-gy", type=float, default=400.0)
    parser.add_argument("--only-case-ids", type=str, nargs="*", default=[])
    parser.add_argument("--max-cases", type=int, default=0)
    parser.add_argument("--verbose-pde", action="store_true")
    return parser.parse_args()


def load_csv_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


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
        vertices = [tuple(float(v) for v in row) for row in np.asarray(data["vertex_centers_mm"], dtype=np.float32)]
    voxel_size = (
        float(axes_mm["x"][1] - axes_mm["x"][0]),
        float(axes_mm["y"][1] - axes_mm["y"][0]),
        float(axes_mm["z"][1] - axes_mm["z"][0]),
    )
    return structures, axes_mm, voxel_size, vertices


def load_dose_npz(path: Path) -> np.ndarray:
    with np.load(path) as data:
        return np.asarray(data["dose_gy"], dtype=np.float32)


def load_dose_csv(path: Path) -> np.ndarray:
    return np.asarray(load_topas_csv_grid(path), dtype=np.float32)


def voxel_size_mm_from_axes(axes_mm: Mapping[str, np.ndarray]) -> Tuple[float, float, float]:
    return (
        float(axes_mm["x"][1] - axes_mm["x"][0]),
        float(axes_mm["y"][1] - axes_mm["y"][0]),
        float(axes_mm["z"][1] - axes_mm["z"][0]),
    )


def resolve_context_path(
    *,
    phase33_row: Mapping[str, str],
    phase32_row: Mapping[str, str],
    phase33_data_root: Path,
    phase32_data_root: Path,
) -> Path:
    candidates: List[Path] = []
    phase33_context = str(phase33_row.get("phase33_context_npz", "")).strip()
    if phase33_context:
        candidates.append(
            resolve_project_path(
                phase33_context,
                phase33_data_root=phase33_data_root,
                phase32_data_root=phase32_data_root,
            )
        )
    phase32_context = str(phase32_row.get("phantom_context_npz", "")).strip()
    if phase32_context:
        candidates.append(
            resolve_project_path(
                phase32_context,
                phase33_data_root=phase33_data_root,
                phase32_data_root=phase32_data_root,
            )
        )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    if not candidates:
        raise FileNotFoundError(f"No context path available for case {phase33_row.get('case_id', '?')}")
    return candidates[0]


def load_dose_from_manifest_row(
    row: Mapping[str, str],
    *,
    phase33_data_root: Path,
    phase32_data_root: Path,
) -> np.ndarray:
    dose_npz = str(row.get("combined_dose_npz", "")).strip()
    if dose_npz:
        dose_npz_path = resolve_project_path(
            dose_npz,
            phase33_data_root=phase33_data_root,
            phase32_data_root=phase32_data_root,
        )
        if dose_npz_path.exists():
            return load_dose_npz(dose_npz_path)
    dose_csv = str(row.get("combined_dose_csv", "")).strip()
    if dose_csv:
        dose_csv_path = resolve_project_path(
            dose_csv,
            phase33_data_root=phase33_data_root,
            phase32_data_root=phase32_data_root,
        )
        if dose_csv_path.exists():
            return load_dose_csv(dose_csv_path)
    raise FileNotFoundError(f"No readable combined dose found for case {row.get('case_id', '?')}")


def harmonize_context_to_dose_shape(
    *,
    structures: Mapping[str, np.ndarray],
    axes_mm: Mapping[str, np.ndarray],
    dose_shape: Tuple[int, int, int],
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    current_shape = tuple(int(v) for v in np.asarray(structures["BODY"]).shape)
    if current_shape == tuple(int(v) for v in dose_shape):
        return (
            {name: np.asarray(mask, dtype=bool) for name, mask in structures.items()},
            {axis: np.asarray(values, dtype=np.float32).copy() for axis, values in axes_mm.items()},
        )
    if current_shape[:2] == tuple(int(v) for v in dose_shape[:2]) and current_shape[2] + 1 == int(dose_shape[2]):
        padded_structures = {
            name: np.pad(np.asarray(mask, dtype=bool), ((0, 0), (0, 0), (0, 1)), mode="constant", constant_values=False)
            for name, mask in structures.items()
        }
        padded_axes = {axis: np.asarray(values, dtype=np.float32).copy() for axis, values in axes_mm.items()}
        z_mm = padded_axes["z"]
        dz = float(z_mm[1] - z_mm[0]) if z_mm.size > 1 else 1.0
        padded_axes["z"] = np.concatenate([z_mm, np.asarray([float(z_mm[-1]) + dz], dtype=np.float32)])
        return padded_structures, padded_axes
    raise ValueError(f"Context shape {current_shape} does not match dose shape {tuple(int(v) for v in dose_shape)}")


def add_structure_aliases(structures: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    result = {name: np.asarray(mask, dtype=bool) for name, mask in structures.items()}
    if "PARATHYROIDS" not in result:
        left = np.asarray(result.get("PARATHYROID_L", np.zeros_like(result["BODY"], dtype=bool)), dtype=bool)
        right = np.asarray(result.get("PARATHYROID_R", np.zeros_like(result["BODY"], dtype=bool)), dtype=bool)
        result["PARATHYROIDS"] = left | right
    if "HYPOXIA" not in result:
        vertex_target = np.asarray(result.get("VERTEX_TARGET", result["GTV"]), dtype=bool)
        if np.any(vertex_target):
            distance = ndimage.distance_transform_edt(vertex_target)
            cutoff = 0.45 * float(np.max(distance))
            hypoxia = vertex_target & (distance >= cutoff)
            if not np.any(hypoxia):
                hypoxia = vertex_target
        else:
            hypoxia = np.zeros_like(result["BODY"], dtype=bool)
        result["HYPOXIA"] = hypoxia
    return result


def endpoint_row_from_result(
    *,
    case_id: str,
    case_label: str,
    site_group: str,
    mode: str,
    result: Mapping[str, object],
) -> Dict[str, object]:
    endpoints = dict(result["endpoints"])
    supplemental = dict(result["supplemental"])
    row: Dict[str, object] = {
        "plan_id": case_id,
        "case_label": case_label,
        "site_group": site_group,
        "mode": mode,
        "mode_label": MODE_LABELS[mode],
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
    case_id: str,
    case_label: str,
    site_group: str,
    mode: str,
    result: Mapping[str, object],
) -> Dict[str, object]:
    row: Dict[str, object] = {
        "plan_id": case_id,
        "case_label": case_label,
        "site_group": site_group,
        "mode": mode,
        "mode_label": MODE_LABELS[mode],
    }
    retained = {
        key: float(value)
        for key, value in dict(result["assays"]).items()
        if str(key)
        in {
            "mean_gammah2ax_peak",
            "mean_gammah2ax_valley",
            "mean_tunel_peak",
            "mean_tunel_valley",
            "cytokine_global_auc",
            "cytokine_peak_roi_auc",
            "cytokine_valley_roi_auc",
        }
    }
    row.update(retained)
    return row


def build_quick_assessment(endpoint_rows: Sequence[Mapping[str, object]], rank_rows: Sequence[Mapping[str, object]], oar_rows: Sequence[Mapping[str, object]]) -> str:
    bio_added = sum(1 for row in oar_rows if str(row["reinterpretation"]) == "biology_adds_failure")
    rank_shift = sum(1 for row in rank_rows if int(row["with_sink_vs_physical_shift"]) != 0)
    lines = [
        "# Phase 34 quick assessment",
        "",
        f"- Cases analyzed: `{len({str(row['plan_id']) for row in endpoint_rows})}`.",
        f"- Endpoint rows written: `{len(endpoint_rows)}`.",
        f"- Cases with with-sink rank shift vs physical-only: `{rank_shift}`.",
        f"- Biology-added OAR failures: `{bio_added}`.",
        "- This package is intended to be run locally after the Phase 33 cohort transport so the full biology cohort does not need to be executed through chat.",
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    out_root = args.out_root.resolve() if args.out_root is not None else args.phase33_root.resolve() / "phase34_bio_cohort"
    out_root.mkdir(parents=True, exist_ok=True)

    phase25_config = json.loads((args.phase25_run_root / "phase25_config.json").read_text(encoding="utf-8"))
    case_rows = load_csv_rows(args.phase33_manifest_root / "phase33_case_manifest.csv")
    selected = [row for row in case_rows if str(row["status"]) == "completed"]
    if args.only_case_ids:
        allowed = set(args.only_case_ids)
        selected = [row for row in selected if str(row["case_id"]) in allowed]
    if int(args.max_cases) > 0:
        selected = selected[: int(args.max_cases)]

    phase32_manifest = {str(row["template_id"]): row for row in load_csv_rows(args.phase32_root / "phase32_case_manifest.csv")}

    endpoint_rows: List[Dict[str, object]] = []
    assay_rows: List[Dict[str, object]] = []

    for row in selected:
        case_id = str(row["case_id"])
        phase32_row = phase32_manifest[case_id]
        case_label = str(row["case_label"])
        site_group = str(phase32_row["site_group"])
        context_path = resolve_context_path(
            phase33_row=row,
            phase32_row=phase32_row,
            phase33_data_root=args.phase33_root,
            phase32_data_root=args.phase32_root,
        )
        structures, axes_mm, voxel_size_mm, vertices_mm = load_case_context(context_path)
        structures = add_structure_aliases(structures)
        dose = load_dose_from_manifest_row(
            row,
            phase33_data_root=args.phase33_root,
            phase32_data_root=args.phase32_root,
        )
        structures, axes_mm = harmonize_context_to_dose_shape(
            structures=structures,
            axes_mm=axes_mm,
            dose_shape=tuple(int(v) for v in dose.shape),
        )
        voxel_size_mm = voxel_size_mm_from_axes(axes_mm)
        voxel_volume_cc = float(voxel_size_mm[0] * voxel_size_mm[1] * voxel_size_mm[2] / 1000.0)

        physical = evaluate_physical_only(
            physical_dose=dose,
            structures=structures,
            axes_mm=axes_mm,
            spots_mm=vertices_mm,
            voxel_volume_cc=voxel_volume_cc,
            voxel_size_mm=voxel_size_mm,
            prescription_gy=float(phase25_config.get("prescription_gy", 3.5)),
            alpha=float(phase25_config["bio_parameters"]["alpha"]),
            beta=float(phase25_config["bio_parameters"]["beta"]),
        )
        no_sink = evaluate_bystander_mode(
            physical_dose=dose,
            structures=structures,
            axes_mm=axes_mm,
            spots_mm=vertices_mm,
            config=phase25_config,
            mode="bystander_no_sink",
            uptake_scale=1.0,
            voxel_volume_cc=voxel_volume_cc,
            voxel_size_mm=voxel_size_mm,
            prescription_gy=float(phase25_config.get("prescription_gy", 3.5)),
            history_interval=int(args.history_interval),
            progress_interval=int(args.progress_interval),
            verbose_pde=bool(args.verbose_pde),
        )
        with_sink = evaluate_bystander_mode(
            physical_dose=dose,
            structures=structures,
            axes_mm=axes_mm,
            spots_mm=vertices_mm,
            config=phase25_config,
            mode="bystander_with_sink",
            uptake_scale=1.0,
            voxel_volume_cc=voxel_volume_cc,
            voxel_size_mm=voxel_size_mm,
            prescription_gy=float(phase25_config.get("prescription_gy", 3.5)),
            history_interval=int(args.history_interval),
            progress_interval=int(args.progress_interval),
            verbose_pde=bool(args.verbose_pde),
        )

        for mode, result in (
            ("physical_only", physical),
            ("bystander_no_sink", no_sink),
            ("bystander_with_sink", with_sink),
        ):
            endpoint_rows.append(
                endpoint_row_from_result(
                    case_id=case_id,
                    case_label=case_label,
                    site_group=site_group,
                    mode=mode,
                    result=result,
                )
            )
            assay_rows.append(
                assay_row_from_result(
                    case_id=case_id,
                    case_label=case_label,
                    site_group=site_group,
                    mode=mode,
                    result=result,
                )
            )

    assign_ranks(endpoint_rows)
    rank_rows = build_rank_shift_rows(endpoint_rows)
    oar_rows = build_oar_reinterpretation_rows(endpoint_rows, args=args)
    delta_rows = build_delta_rows(endpoint_rows)

    write_csv(out_root / "phase34_endpoint_table.csv", endpoint_rows)
    write_csv(out_root / "phase34_assay_proxy_table.csv", assay_rows)
    write_csv(out_root / "phase34_rank_shift_table.csv", rank_rows)
    write_csv(out_root / "phase34_oar_reinterpretation.csv", oar_rows)
    write_csv(out_root / "phase34_endpoint_delta_table.csv", delta_rows)
    write_json(
        out_root / "phase34_manifest.json",
        {
            "phase32_root": str(args.phase32_root.resolve()),
            "phase33_root": str(args.phase33_root.resolve()),
            "phase25_run_root": str(args.phase25_run_root.resolve()),
            "cases": [str(row["case_id"]) for row in selected],
        },
    )
    (out_root / "phase34_quick_assessment.md").write_text(
        build_quick_assessment(endpoint_rows, rank_rows, oar_rows),
        encoding="utf-8",
    )

    print("=== PHASE 34 PHASE 32 BIO COHORT PACKAGE READY ===")
    print(f"Output root: {out_root}")
    print(f"Endpoint table: {out_root / 'phase34_endpoint_table.csv'}")
    print(f"Assay table: {out_root / 'phase34_assay_proxy_table.csv'}")
    print(f"Rank shifts: {out_root / 'phase34_rank_shift_table.csv'}")
    print(f"Quick assessment: {out_root / 'phase34_quick_assessment.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
