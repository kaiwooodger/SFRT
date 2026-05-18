#!/usr/bin/env python3
"""Phase 33: TOPAS transport package for the Phase 32 site-specific cohort.

This script turns the Phase 32 site-specific synthetic phantoms into a local-run
TOPAS cohort package. For each case with at least one validated vertex it:

1. builds a broad PTV coverage component
2. builds a vertex-focused component
3. runs both through TOPAS (or reuses existing scorer CSVs)
4. calibrates the spot/base ratio to target ~3.5 Gy PTV D95 and ~15 Gy peak mean
5. writes a per-case physical dose package ready for downstream biology analysis
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np

from build_asymmetric_sweep import format_physics_modules, has_nonempty_output
from generate_detailed_headneck_topas_phantom import MATERIAL_SPECS
from run_phase26_vascular_sink_ablation import extract_endpoints
from run_phase30_phase28_topas_true_lattice_delivery import (
    SourceSpec,
    histories_from_weights,
    load_spectrum,
    load_topas_csv_grid,
    mask_centroid_mm,
    mask_points_mm,
    projected_aperture_radii,
    run_topas_case,
    search_component_ratio,
    smooth_dose_grid,
    source_center_for_angle,
    write_dose_csv,
)


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase32-root",
        type=Path,
        default=root / "runs" / "phase32_site_specific_template_phantoms",
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        default=root / "runs" / "phase33_phase32_topas_cohort",
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
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=33)
    parser.add_argument("--histories-base", type=int, default=12000)
    parser.add_argument("--histories-spot", type=int, default=24000)
    parser.add_argument("--sad-mm", type=float, default=260.0)
    parser.add_argument("--base-angles-deg", type=float, nargs="+", default=[0.0, 30.0, 60.0, 90.0, 120.0, 150.0, 180.0, 210.0, 240.0, 270.0, 300.0, 330.0])
    parser.add_argument("--spot-angles-deg", type=float, nargs="+", default=[0.0, 60.0, 120.0, 180.0, 240.0, 300.0])
    parser.add_argument("--base-margin-mm", type=float, default=4.0)
    parser.add_argument("--spot-margin-mm", type=float, default=1.0)
    parser.add_argument("--vertex-radius-mm", type=float, default=5.0)
    parser.add_argument("--target-ptv-d95-gy", type=float, default=3.5)
    parser.add_argument("--target-peak-mean-gy", type=float, default=15.0)
    parser.add_argument("--ratio-min", type=float, default=0.01)
    parser.add_argument("--ratio-max", type=float, default=100.0)
    parser.add_argument("--ratio-count", type=int, default=121)
    parser.add_argument("--dose-smoothing-mm", type=float, default=6.0)
    parser.add_argument("--cut-gamma-mm", type=float, default=0.01)
    parser.add_argument("--cut-electron-mm", type=float, default=0.01)
    parser.add_argument("--cut-positron-mm", type=float, default=0.01)
    parser.add_argument("--prescription-gy", type=float, default=3.5)
    parser.add_argument("--only-case-ids", type=str, nargs="*", default=[])
    parser.add_argument("--max-cases", type=int, default=0)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--analyze-only", action="store_true")
    return parser.parse_args()


def write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        raise ValueError(f"No rows for {path}")
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


def load_case_manifest(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def load_case_context(path: Path) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Tuple[float, float, float]]:
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
    voxel_size = (
        float(axes_mm["x"][1] - axes_mm["x"][0]),
        float(axes_mm["y"][1] - axes_mm["y"][0]),
        float(axes_mm["z"][1] - axes_mm["z"][0]),
    )
    return structures, axes_mm, voxel_size


def load_material_tags(path: Path) -> np.ndarray:
    with np.load(path) as data:
        return np.asarray(data["material_tags"], dtype=np.int16)


def parse_vertices(text: str) -> List[Tuple[float, float, float]]:
    if not str(text).strip():
        return []
    values = json.loads(text)
    return [tuple(float(v) for v in row) for row in values]


def ensure_structure_aliases(structures: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    result = {name: np.asarray(mask, dtype=bool) for name, mask in structures.items()}
    if "PARATHYROIDS" not in result:
        left = np.asarray(result.get("PARATHYROID_L", np.zeros_like(result["BODY"], dtype=bool)), dtype=bool)
        right = np.asarray(result.get("PARATHYROID_R", np.zeros_like(result["BODY"], dtype=bool)), dtype=bool)
        result["PARATHYROIDS"] = left | right
    return result


def write_image_cube(tag_grid_xyz: np.ndarray, out_file: Path) -> None:
    np.asarray(tag_grid_xyz, dtype=np.int16).transpose(2, 1, 0).tofile(out_file)


def prepare_topas_ready_context(
    *,
    structures: Mapping[str, np.ndarray],
    axes_mm: Mapping[str, np.ndarray],
    material_tags: np.ndarray,
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], np.ndarray, bool]:
    structures_out = {name: np.asarray(mask, dtype=bool) for name, mask in structures.items()}
    axes_out = {axis: np.asarray(values, dtype=np.float32).copy() for axis, values in axes_mm.items()}
    tags_out = np.asarray(material_tags, dtype=np.int16).copy()
    padding_applied = False
    if int(tags_out.size) % 2 == 1:
        tags_out = np.pad(tags_out, ((0, 0), (0, 0), (0, 1)), mode="constant", constant_values=0)
        structures_out = {
            name: np.pad(mask, ((0, 0), (0, 0), (0, 1)), mode="constant", constant_values=False)
            for name, mask in structures_out.items()
        }
        z_mm = axes_out["z"]
        dz = float(z_mm[1] - z_mm[0]) if z_mm.size > 1 else 1.0
        axes_out["z"] = np.concatenate([z_mm, np.asarray([float(z_mm[-1]) + dz], dtype=np.float32)])
        padding_applied = True
    return structures_out, axes_out, tags_out, padding_applied


def save_case_context(
    path: Path,
    *,
    structures: Mapping[str, np.ndarray],
    axes_mm: Mapping[str, np.ndarray],
    vertices_mm: Sequence[Tuple[float, float, float]],
) -> None:
    payload: Dict[str, np.ndarray] = {
        "axes_x_mm": np.asarray(axes_mm["x"], dtype=np.float32),
        "axes_y_mm": np.asarray(axes_mm["y"], dtype=np.float32),
        "axes_z_mm": np.asarray(axes_mm["z"], dtype=np.float32),
        "vertex_centers_mm": np.asarray(vertices_mm, dtype=np.float32),
    }
    for name, mask in structures.items():
        payload[f"struct_{name}"] = np.asarray(mask, dtype=bool)
    np.savez_compressed(path, **payload)


def render_source_block(sources: Sequence[SourceSpec], spectrum_energies: Sequence[float], spectrum_weights: Sequence[float]) -> str:
    spectrum_count = len(spectrum_energies)
    spectrum_values = " ".join(f"{float(v):.6f}" for v in spectrum_energies)
    spectrum_weight_values = " ".join(f"{float(v):.8f}" for v in spectrum_weights)
    blocks: List[str] = []
    for source in sources:
        group_name = f"BeamOrigin_{source.name}"
        source_name = f"Source_{source.name}"
        blocks.extend(
            [
                f's:Ge/{group_name}/Type = "Group"',
                f's:Ge/{group_name}/Parent = "World"',
                f"d:Ge/{group_name}/TransX = {source.center_mm[0]:.6f} mm",
                f"d:Ge/{group_name}/TransY = {source.center_mm[1]:.6f} mm",
                f"d:Ge/{group_name}/TransZ = {source.center_mm[2]:.6f} mm",
                f"d:Ge/{group_name}/RotX = {source.rotation_deg[0]:.6f} deg",
                f"d:Ge/{group_name}/RotY = {source.rotation_deg[1]:.6f} deg",
                f"d:Ge/{group_name}/RotZ = {source.rotation_deg[2]:.6f} deg",
                f's:So/{source_name}/Type = "Beam"',
                f's:So/{source_name}/Component = "{group_name}"',
                f's:So/{source_name}/BeamParticle = "gamma"',
                f's:So/{source_name}/BeamEnergySpectrumType = "Discrete"',
                f"dv:So/{source_name}/BeamEnergySpectrumValues = {spectrum_count} {spectrum_values} MeV",
                f"uv:So/{source_name}/BeamEnergySpectrumWeights = {spectrum_count} {spectrum_weight_values}",
                f's:So/{source_name}/BeamPositionDistribution = "Flat"',
                f's:So/{source_name}/BeamPositionCutoffShape = "Ellipse"',
                f"d:So/{source_name}/BeamPositionCutoffX = {source.cutoff_mm[0]:.6f} mm",
                f"d:So/{source_name}/BeamPositionCutoffY = {source.cutoff_mm[1]:.6f} mm",
                f's:So/{source_name}/BeamAngularDistribution = "None"',
                f"i:So/{source_name}/NumberOfHistoriesInRun = {source.histories}",
                "",
            ]
        )
    return "\n".join(blocks)


def render_case_file(
    args: argparse.Namespace,
    *,
    patient_input_dir: Path,
    patient_input_file: str,
    materials_file: Path,
    grid_shape: Sequence[int],
    voxel_size_mm: Sequence[float],
    spectrum_energies: Sequence[float],
    spectrum_weights: Sequence[float],
    sources: Sequence[SourceSpec],
) -> str:
    template_text = Path(args.template).read_text(encoding="utf-8")
    input_dir = str(patient_input_dir.resolve())
    if not input_dir.endswith("/"):
        input_dir += "/"
    world_hlx_cm = max(20.0, float(grid_shape[0]) * float(voxel_size_mm[0]) / 20.0 + 5.0)
    world_hly_cm = max(20.0, float(grid_shape[1]) * float(voxel_size_mm[1]) / 20.0 + 5.0)
    world_hlz_cm = max(20.0, float(grid_shape[2]) * float(voxel_size_mm[2]) / 20.0 + 5.0)
    replacements = {
        "__G4_DATA_DIR__": str(args.g4_data_dir),
        "__PHYSICS_MODULES__": format_physics_modules(str(args.physics_profile)),
        "__CUT_GAMMA_MM__": f"{float(args.cut_gamma_mm):.6f}",
        "__CUT_ELECTRON_MM__": f"{float(args.cut_electron_mm):.6f}",
        "__CUT_POSITRON_MM__": f"{float(args.cut_positron_mm):.6f}",
        "__MATERIALS_INCLUDE_FILE__": materials_file.name,
        "__WORLD_HLX_CM__": f"{world_hlx_cm:.6f}",
        "__WORLD_HLY_CM__": f"{world_hly_cm:.6f}",
        "__WORLD_HLZ_CM__": f"{world_hlz_cm:.6f}",
        "__PATIENT_INPUT_DIR__": input_dir,
        "__PATIENT_INPUT_FILE__": patient_input_file,
        "__XBINS__": str(int(grid_shape[0])),
        "__YBINS__": str(int(grid_shape[1])),
        "__ZBINS__": str(int(grid_shape[2])),
        "__VOXEL_SIZE_X_MM__": f"{float(voxel_size_mm[0]):.6f}",
        "__VOXEL_SIZE_Y_MM__": f"{float(voxel_size_mm[1]):.6f}",
        "__VOXEL_SIZE_Z_MM__": f"{float(voxel_size_mm[2]):.6f}",
        "__MATERIAL_TAG_COUNT__": str(len(MATERIAL_SPECS)),
        "__MATERIAL_TAG_VALUES__": " ".join(str(int(spec.tag)) for spec in MATERIAL_SPECS),
        "__MATERIAL_NAME_VALUES__": " ".join(f'"{spec.name}"' for spec in MATERIAL_SPECS),
        "__OUTPUT_STEM__": "dosedata",
        "__SOURCE_BLOCK__": render_source_block(sources, spectrum_energies, spectrum_weights),
        "__N_THREADS__": str(int(args.threads)),
        "__SEED__": str(int(args.seed)),
        "__SHOW_HISTORY_INTERVAL__": str(max(1, int(sum(spec.histories for spec in sources)) // 10)),
    }
    rendered = template_text
    for key, value in replacements.items():
        rendered = rendered.replace(key, value)
    return rendered


def prepare_case_dir(out_dir: Path, *, materials_source: Path) -> Tuple[Path, Path, Path, Path]:
    case_dir = out_dir / "case"
    case_dir.mkdir(parents=True, exist_ok=True)
    materials_file = case_dir / "materials.txt"
    shutil.copyfile(materials_source, materials_file)
    parameter_file = case_dir / "beamline.txt"
    dose_csv = case_dir / "dosedata.csv"
    log_file = case_dir / "topas.log"
    return case_dir, materials_file, parameter_file, dose_csv, log_file


def sphere_union_mask(axes_mm: Mapping[str, np.ndarray], centers_mm: Sequence[Tuple[float, float, float]], radius_mm: float) -> np.ndarray:
    x_mm = np.asarray(axes_mm["x"], dtype=np.float32)
    y_mm = np.asarray(axes_mm["y"], dtype=np.float32)
    z_mm = np.asarray(axes_mm["z"], dtype=np.float32)
    mask = np.zeros((x_mm.size, y_mm.size, z_mm.size), dtype=bool)
    if not centers_mm:
        return mask
    margin = float(radius_mm) + max(float(x_mm[1] - x_mm[0]), float(y_mm[1] - y_mm[0]), float(z_mm[1] - z_mm[0]))
    radius2 = float(radius_mm) ** 2
    for cx, cy, cz in centers_mm:
        ix = np.flatnonzero((x_mm >= float(cx) - margin) & (x_mm <= float(cx) + margin))
        iy = np.flatnonzero((y_mm >= float(cy) - margin) & (y_mm <= float(cy) + margin))
        iz = np.flatnonzero((z_mm >= float(cz) - margin) & (z_mm <= float(cz) + margin))
        if ix.size == 0 or iy.size == 0 or iz.size == 0:
            continue
        xx = x_mm[ix][:, None, None]
        yy = y_mm[iy][None, :, None]
        zz = z_mm[iz][None, None, :]
        local = ((xx - float(cx)) ** 2 + (yy - float(cy)) ** 2 + (zz - float(cz)) ** 2) <= radius2
        mask[np.ix_(ix, iy, iz)] |= local
    return mask


def build_base_sources(args: argparse.Namespace, *, ptv_mask: np.ndarray, axes_mm: Mapping[str, np.ndarray]) -> List[SourceSpec]:
    ptv_centroid = mask_centroid_mm(ptv_mask, axes_mm)
    points = mask_points_mm(ptv_mask, axes_mm)
    angles = [float(value) for value in args.base_angles_deg]
    histories = histories_from_weights(int(args.histories_base), [1.0] * len(angles))
    sources: List[SourceSpec] = []
    for angle, hist in zip(angles, histories):
        rad_u, rad_v = projected_aperture_radii(points, ptv_centroid, angle, float(args.base_margin_mm))
        sources.append(
            SourceSpec(
                name=f"BASE_{int(round(angle)) % 360:03d}",
                center_mm=source_center_for_angle(ptv_centroid, angle, float(args.sad_mm)),
                rotation_deg=(0.0, float(angle), 0.0),
                cutoff_mm=(float(rad_u), float(rad_v)),
                histories=int(hist),
            )
        )
    return sources


def build_spot_sources(
    args: argparse.Namespace,
    *,
    axes_mm: Mapping[str, np.ndarray],
    vertices_mm: Sequence[Tuple[float, float, float]],
) -> List[SourceSpec]:
    weights = [1.0] * (len(args.spot_angles_deg) * len(vertices_mm))
    histories = histories_from_weights(int(args.histories_spot), weights)
    history_iter = iter(histories)
    sources: List[SourceSpec] = []
    for vertex_idx, center in enumerate(vertices_mm, start=1):
        vertex_mask = sphere_union_mask(axes_mm, [center], float(args.vertex_radius_mm))
        points = mask_points_mm(vertex_mask, axes_mm)
        for angle in [float(value) for value in args.spot_angles_deg]:
            rad_u, rad_v = projected_aperture_radii(points, center, angle, float(args.spot_margin_mm))
            sources.append(
                SourceSpec(
                    name=f"V{vertex_idx}_{int(round(angle)) % 360:03d}",
                    center_mm=source_center_for_angle(center, angle, float(args.sad_mm)),
                    rotation_deg=(0.0, float(angle), 0.0),
                    cutoff_mm=(float(rad_u), float(rad_v)),
                    histories=int(next(history_iter)),
                )
            )
    return sources


def summarize_physical_case(
    dose_grid: np.ndarray,
    *,
    structures: Mapping[str, np.ndarray],
    axes_mm: Mapping[str, np.ndarray],
    vertices_mm: Sequence[Tuple[float, float, float]],
    prescription_gy: float,
) -> Dict[str, float]:
    voxel_cc = float((axes_mm["x"][1] - axes_mm["x"][0]) * (axes_mm["y"][1] - axes_mm["y"][0]) * (axes_mm["z"][1] - axes_mm["z"][0]) / 1000.0)
    endpoints, supplemental = extract_endpoints(
        dose_grid,
        structures=structures,
        axes_mm=axes_mm,
        spots_mm=vertices_mm,
        voxel_volume_cc=voxel_cc,
        prescription_gy=float(prescription_gy),
    )
    summary = {
        "ptv_d95_gy": float(endpoints["ptv_d95"]),
        "gtv_d95_gy": float(supplemental["gtv_d95"]),
        "peak_mean_gy": float(supplemental["peak_mean"]),
        "valley_mean_gy": float(supplemental["valley_mean"]),
        "pvdr": float(endpoints["pvdr"]),
        "spill_shell_0_5_mean_gy": float(endpoints["spill_shell_0_5_mean"]),
        "spill_shell_5_15_mean_gy": float(endpoints["spill_shell_5_15_mean"]),
        "cord_d2_gy": float(endpoints["cord_d2"]),
        "brainstem_d2_gy": float(endpoints["brainstem_d2"]),
        "parotid_r_mean_gy": float(endpoints["parotid_r_mean"]),
        "thyroid_mean_gy": float(supplemental["thyroid_mean"]),
        "body_dmax_gy": float(supplemental["body_dmax"]),
    }
    return summary


def case_rows_from_summary(case_id: str, case_label: str, component_label: str, metrics: Mapping[str, float]) -> Dict[str, object]:
    row: Dict[str, object] = {
        "case_id": case_id,
        "case_label": case_label,
        "component": component_label,
    }
    row.update({key: float(value) for key, value in metrics.items()})
    return row


def build_quick_assessment(rows: Sequence[Mapping[str, object]]) -> str:
    delivered = [row for row in rows if str(row["status"]) == "completed"]
    skipped = [row for row in rows if str(row["status"]) != "completed"]
    lines = [
        "# Phase 33 quick assessment",
        "",
        f"- Cohort cases requested: `{len(rows)}`.",
        f"- Cases completed with calibrated physical dose: `{len(delivered)}`.",
        f"- Cases skipped or deferred: `{len(skipped)}`.",
    ]
    if delivered:
        pvdr = np.asarray([float(row["final_pvdr"]) for row in delivered], dtype=np.float64)
        lines.extend(
            [
                f"- Median final PVDR: `{float(np.median(pvdr)):.3f}`.",
                f"- Mean final PTV D95: `{float(np.mean([float(row['final_ptv_d95_gy']) for row in delivered])):.3f} Gy`.",
            ]
        )
    lines.append("- This package is designed to be run locally so the 10-case TOPAS cohort does not have to be executed through the chat session.")
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    out_root = args.out_root.resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    manifest_rows = load_case_manifest(args.phase32_root / "phase32_case_manifest.csv")
    selected = [row for row in manifest_rows if not args.only_case_ids or str(row["template_id"]) in set(args.only_case_ids)]
    if int(args.max_cases) > 0:
        selected = selected[: int(args.max_cases)]

    spectrum_energies, spectrum_weights = load_spectrum(Path(args.spectrum_csv))

    cohort_rows: List[Dict[str, object]] = []
    endpoint_rows: List[Dict[str, object]] = []

    for row in selected:
        case_id = str(row["template_id"])
        case_label = str(row["label"])
        case_root = out_root / case_id
        case_root.mkdir(parents=True, exist_ok=True)

        vertices_mm = parse_vertices(str(row["kept_vertices_mm"]))
        if not vertices_mm:
            cohort_rows.append(
                {
                    "case_id": case_id,
                    "case_label": case_label,
                    "status": "skipped_no_vertices",
                    "run_root": str(case_root),
                }
            )
            continue

        structures, axes_mm, voxel_size_mm = load_case_context(Path(str(row["phantom_context_npz"])))
        structures = ensure_structure_aliases(structures)
        material_tags = load_material_tags(Path(str(row["material_tags_npz"])))
        structures, axes_mm, material_tags, padding_applied = prepare_topas_ready_context(
            structures=structures,
            axes_mm=axes_mm,
            material_tags=material_tags,
        )
        voxel_size_mm = (
            float(axes_mm["x"][1] - axes_mm["x"][0]),
            float(axes_mm["y"][1] - axes_mm["y"][0]),
            float(axes_mm["z"][1] - axes_mm["z"][0]),
        )
        patient_binary = case_root / "phase33_patient_material_tags.bin"
        write_image_cube(material_tags, patient_binary)
        np.savez_compressed(case_root / "phase33_material_tags.npz", material_tags=material_tags.astype(np.int16))
        save_case_context(
            case_root / "phase33_context.npz",
            structures=structures,
            axes_mm=axes_mm,
            vertices_mm=vertices_mm,
        )
        materials_source = Path(str(row["materials_include"]))
        grid_shape = tuple(int(v) for v in np.asarray(structures["BODY"]).shape)

        base_sources = build_base_sources(args, ptv_mask=np.asarray(structures["PTV"], dtype=bool), axes_mm=axes_mm)
        spot_sources = build_spot_sources(args, axes_mm=axes_mm, vertices_mm=vertices_mm)
        component_specs = [("base_component", base_sources), ("spot_component", spot_sources)]

        component_outputs: Dict[str, Dict[str, object]] = {}
        for component_name, sources in component_specs:
            component_root = case_root / component_name
            case_dir, materials_file, parameter_file, dose_csv, log_file = prepare_case_dir(component_root, materials_source=materials_source)
            if not args.analyze_only:
                case_text = render_case_file(
                    args,
                    patient_input_dir=patient_binary.parent,
                    patient_input_file=patient_binary.name,
                    materials_file=materials_file,
                    grid_shape=grid_shape,
                    voxel_size_mm=voxel_size_mm,
                    spectrum_energies=spectrum_energies,
                    spectrum_weights=spectrum_weights,
                    sources=sources,
                )
                parameter_file.write_text(case_text, encoding="utf-8")
                if not (args.skip_existing and has_nonempty_output(dose_csv)):
                    run_topas_case(args, case_dir, parameter_file, dose_csv, log_file)
            if not dose_csv.exists():
                raise FileNotFoundError(f"Expected TOPAS output missing for {case_id} {component_name}: {dose_csv}")
            component_outputs[component_name] = {
                "root": str(component_root),
                "dose_csv": str(dose_csv),
                "log_file": str(log_file),
                "num_sources": int(len(sources)),
            }

        base_dose = load_topas_csv_grid(Path(component_outputs["base_component"]["dose_csv"]))
        spot_dose = load_topas_csv_grid(Path(component_outputs["spot_component"]["dose_csv"]))
        body_mask = np.asarray(structures["BODY"], dtype=bool)
        base_processed = smooth_dose_grid(base_dose, axes_mm=axes_mm, body_mask=body_mask, sigma_mm=float(args.dose_smoothing_mm))
        spot_processed = smooth_dose_grid(spot_dose, axes_mm=axes_mm, body_mask=body_mask, sigma_mm=float(args.dose_smoothing_mm))
        peak_mask = sphere_union_mask(axes_mm, vertices_mm, float(args.vertex_radius_mm)) & np.asarray(structures["PTV"], dtype=bool)

        calibration = search_component_ratio(
            base_dose=base_processed,
            spot_dose=spot_processed,
            ptv_mask=np.asarray(structures["PTV"], dtype=bool),
            peak_mask=peak_mask,
            args=args,
        )
        combined_processed = base_processed + float(calibration["spot_to_base_ratio"]) * spot_processed
        combined_scaled = float(calibration["global_scale"]) * combined_processed
        scaled_base = float(calibration["global_scale"]) * base_processed
        scaled_spot = float(calibration["global_scale"]) * float(calibration["spot_to_base_ratio"]) * spot_processed

        np.savez_compressed(
            case_root / "phase33_component_doses.npz",
            base_raw=base_dose.astype(np.float32),
            spot_raw=spot_dose.astype(np.float32),
            base_processed=base_processed.astype(np.float32),
            spot_processed=spot_processed.astype(np.float32),
            base_scaled=scaled_base.astype(np.float32),
            spot_scaled=scaled_spot.astype(np.float32),
            combined_scaled=combined_scaled.astype(np.float32),
        )
        np.savez_compressed(case_root / "phase33_combined_physical_dose.npz", dose_gy=combined_scaled.astype(np.float32))
        write_dose_csv(case_root / "phase33_combined_physical_dose.csv", combined_scaled.astype(np.float32))

        for label, dose_grid in (
            ("base_processed_scaled", scaled_base),
            ("spot_processed_scaled", scaled_spot),
            ("combined_processed_scaled", combined_scaled),
        ):
            endpoint_rows.append(
                case_rows_from_summary(
                    case_id,
                    case_label,
                    label,
                    summarize_physical_case(
                        dose_grid,
                        structures=structures,
                        axes_mm=axes_mm,
                        vertices_mm=vertices_mm,
                        prescription_gy=float(args.prescription_gy),
                    ),
                )
            )

        final_metrics = summarize_physical_case(
            combined_scaled,
            structures=structures,
            axes_mm=axes_mm,
            vertices_mm=vertices_mm,
            prescription_gy=float(args.prescription_gy),
        )
        summary = {
            "case_id": case_id,
            "case_label": case_label,
            "phase32_case_root": str(Path(str(row["case_dir"]))),
            "phase33_context_npz": str(case_root / "phase33_context.npz"),
            "phase33_patient_material_tags_bin": str(patient_binary),
            "phase33_material_tags_npz": str(case_root / "phase33_material_tags.npz"),
            "topas_padding_applied": bool(padding_applied),
            "topas_grid_shape": [int(v) for v in grid_shape],
            "histories_base": int(args.histories_base),
            "histories_spot": int(args.histories_spot),
            "dose_smoothing_mm": float(args.dose_smoothing_mm),
            "voxel_mm": float(voxel_size_mm[0]),
            "kept_vertices_mm": [[float(a), float(b), float(c)] for a, b, c in vertices_mm],
            "component_outputs": component_outputs,
            "calibration": calibration,
            "final_metrics": final_metrics,
            "output_files": {
                "component_doses_npz": str(case_root / "phase33_component_doses.npz"),
                "combined_physical_dose_npz": str(case_root / "phase33_combined_physical_dose.npz"),
                "combined_physical_dose_csv": str(case_root / "phase33_combined_physical_dose.csv"),
            },
        }
        write_json(case_root / "phase33_case_summary.json", summary)
        cohort_rows.append(
            {
                "case_id": case_id,
                "case_label": case_label,
                "status": "completed",
                "run_root": str(case_root),
                "summary_json": str(case_root / "phase33_case_summary.json"),
                "phase33_context_npz": str(case_root / "phase33_context.npz"),
                "phase33_patient_material_tags_bin": str(patient_binary),
                "combined_dose_npz": str(case_root / "phase33_combined_physical_dose.npz"),
                "combined_dose_csv": str(case_root / "phase33_combined_physical_dose.csv"),
                "final_ptv_d95_gy": float(final_metrics["ptv_d95_gy"]),
                "final_peak_mean_gy": float(final_metrics["peak_mean_gy"]),
                "final_pvdr": float(final_metrics["pvdr"]),
                "kept_vertex_count": int(len(vertices_mm)),
                "topas_padding_applied": bool(padding_applied),
            }
        )

    write_csv(out_root / "phase33_case_manifest.csv", cohort_rows)
    write_csv(out_root / "phase33_physical_endpoint_table.csv", endpoint_rows)
    write_json(out_root / "phase33_case_manifest.json", cohort_rows)
    (out_root / "phase33_quick_assessment.md").write_text(build_quick_assessment(cohort_rows), encoding="utf-8")

    print("=== PHASE 33 PHASE 32 TOPAS COHORT PACKAGE READY ===")
    print(f"Output root: {out_root}")
    print(f"Case manifest: {out_root / 'phase33_case_manifest.csv'}")
    print(f"Endpoint table: {out_root / 'phase33_physical_endpoint_table.csv'}")
    print(f"Quick assessment: {out_root / 'phase33_quick_assessment.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
