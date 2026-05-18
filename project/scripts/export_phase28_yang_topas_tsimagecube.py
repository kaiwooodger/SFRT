#!/usr/bin/env python3
"""Export the Phase 28 Yang benchmark phantom as a TOPAS TsImageCube case."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Dict, Sequence

import numpy as np

from build_asymmetric_sweep import PHYSICS_PROFILES, build_topas_env, format_physics_modules, has_nonempty_output, write_text_with_retries
from generate_detailed_headneck_topas_phantom import SourceSpec, fill_template, render_source_block
from run_linac_6mv_polyenergetic_clinical_sfrt import load_spectrum
from render_phase28_geometry_3d_figures import (
    TOPAS_TAG_SPECS,
    build_density_from_tags,
    build_phase28_material_tag_grid,
    build_structures,
    make_axes,
    render_topas_materials_include,
)


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--template",
        type=Path,
        default=root / "topas" / "headneck_detailed_material_phantom_template.txt",
        help="TOPAS TsImageCube template.",
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        default=root / "runs" / "phase28_yang2022_topas_tsimagecube",
        help="Output root for the TOPAS-ready Yang benchmark phantom.",
    )
    parser.add_argument(
        "--spectrum-csv",
        type=Path,
        default=root / "data" / "linac_6mv_representative_spectrum.csv",
    )
    parser.add_argument(
        "--topas-bin",
        type=str,
        default="/Users/kw/shellScripts/topas",
    )
    parser.add_argument(
        "--g4-data-dir",
        type=str,
        default="/Applications/GEANT4",
    )
    parser.add_argument(
        "--physics-profile",
        choices=sorted(PHYSICS_PROFILES),
        default="em_opt4_only",
    )
    parser.add_argument("--voxel-mm", type=float, default=2.0)
    parser.add_argument("--field-radius-mm", type=float, default=85.0)
    parser.add_argument("--source-z-mm", type=float, default=-260.0)
    parser.add_argument("--histories", type=int, default=2000)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--seed", type=int, default=33)
    parser.add_argument("--cut-gamma-mm", type=float, default=0.01)
    parser.add_argument("--cut-electron-mm", type=float, default=0.01)
    parser.add_argument("--cut-positron-mm", type=float, default=0.01)
    parser.add_argument("--run-topas", action="store_true", help="Run a small TOPAS validation field after export.")
    return parser.parse_args()


def write_image_cube(tag_grid_xyz: np.ndarray, out_file: Path) -> None:
    np.asarray(tag_grid_xyz, dtype=np.int16).transpose(2, 1, 0).tofile(out_file)


def render_case_file(
    args: argparse.Namespace,
    *,
    patient_bin: Path,
    materials_file: Path,
    grid_shape: Sequence[int],
    spectrum_energies: Sequence[float],
    spectrum_weights: Sequence[float],
    source: SourceSpec,
) -> str:
    template_text = args.template.read_text(encoding="utf-8")
    input_dir = str(patient_bin.parent.resolve())
    if not input_dir.endswith("/"):
        input_dir += "/"
    world_hlx_cm = max(20.0, float(grid_shape[0]) * float(args.voxel_mm) / 20.0 + 5.0)
    world_hly_cm = max(20.0, float(grid_shape[1]) * float(args.voxel_mm) / 20.0 + 5.0)
    world_hlz_cm = max(20.0, float(grid_shape[2]) * float(args.voxel_mm) / 20.0 + 5.0)
    used_tags = [int(tag) for tag in np.unique(np.fromfile(patient_bin, dtype=np.int16))]
    used_specs = [TOPAS_TAG_SPECS[int(tag)] for tag in used_tags]
    replacements = {
        "__G4_DATA_DIR__": str(args.g4_data_dir),
        "__PHYSICS_MODULES__": format_physics_modules(args.physics_profile),
        "__CUT_GAMMA_MM__": f"{float(args.cut_gamma_mm):.6f}",
        "__CUT_ELECTRON_MM__": f"{float(args.cut_electron_mm):.6f}",
        "__CUT_POSITRON_MM__": f"{float(args.cut_positron_mm):.6f}",
        "__MATERIALS_INCLUDE_FILE__": materials_file.name,
        "__WORLD_HLX_CM__": f"{world_hlx_cm:.6f}",
        "__WORLD_HLY_CM__": f"{world_hly_cm:.6f}",
        "__WORLD_HLZ_CM__": f"{world_hlz_cm:.6f}",
        "__PATIENT_INPUT_DIR__": input_dir,
        "__PATIENT_INPUT_FILE__": patient_bin.name,
        "__XBINS__": str(int(grid_shape[0])),
        "__YBINS__": str(int(grid_shape[1])),
        "__ZBINS__": str(int(grid_shape[2])),
        "__VOXEL_SIZE_X_MM__": f"{float(args.voxel_mm):.6f}",
        "__VOXEL_SIZE_Y_MM__": f"{float(args.voxel_mm):.6f}",
        "__VOXEL_SIZE_Z_MM__": f"{float(args.voxel_mm):.6f}",
        "__MATERIAL_TAG_COUNT__": str(len(used_specs)),
        "__MATERIAL_TAG_VALUES__": " ".join(str(int(tag)) for tag in used_tags),
        "__MATERIAL_NAME_VALUES__": " ".join(f'"{spec["name"]}"' for spec in used_specs),
        "__OUTPUT_STEM__": "dosedata",
        "__SOURCE_BLOCK__": render_source_block(spectrum_energies, spectrum_weights, source),
        "__N_THREADS__": str(int(args.threads)),
        "__SEED__": str(int(args.seed)),
        "__SHOW_HISTORY_INTERVAL__": str(max(1, int(args.histories) // 10)),
    }
    return fill_template(template_text, replacements)


def run_topas_case(args: argparse.Namespace, case_dir: Path, parameter_file: Path, dose_csv: Path, log_file: Path) -> None:
    result = subprocess.run(
        [str(args.topas_bin), parameter_file.name],
        cwd=str(case_dir),
        capture_output=True,
        text=True,
        env=build_topas_env(str(args.g4_data_dir)),
    )
    combined_log = (result.stdout or "") + ("\n" + result.stderr if result.stderr else "")
    write_text_with_retries(log_file, combined_log)
    if result.returncode != 0 or not has_nonempty_output(dose_csv):
        tail = "\n".join(combined_log.strip().splitlines()[-80:])
        raise RuntimeError(
            "TOPAS validation run failed for the Phase 28 Yang TsImageCube export.\n"
            f"Return code: {result.returncode}\n"
            f"Dose CSV present: {has_nonempty_output(dose_csv)}\n"
            f"Recent log:\n{tail}"
        )


def main() -> int:
    args = parse_args()
    args.out_root.mkdir(parents=True, exist_ok=True)
    case_dir = args.out_root / "case"
    case_dir.mkdir(parents=True, exist_ok=True)

    axes = make_axes(float(args.voxel_mm))
    structures = build_structures(axes)
    tag_grid = build_phase28_material_tag_grid(structures)
    density_grid = build_density_from_tags(tag_grid)
    used_tags = [int(tag) for tag in np.unique(tag_grid)]
    used_specs = [TOPAS_TAG_SPECS[int(tag)] for tag in used_tags]

    patient_bin = case_dir / "patient_material_tags.bin"
    density_npz = args.out_root / "phase28_topas_density_from_tags.npz"
    materials_file = case_dir / "materials.txt"
    parameter_file = case_dir / "beamline.txt"
    dose_csv = case_dir / "dosedata.csv"
    log_file = case_dir / "topas.log"

    write_image_cube(tag_grid, patient_bin)
    np.savez_compressed(density_npz, density_g_cm3=density_grid.astype(np.float32))
    write_text_with_retries(materials_file, render_topas_materials_include(used_tags))

    spectrum_energies, spectrum_weights = load_spectrum(args.spectrum_csv)
    source = SourceSpec(
        name="phase28_ap_validation",
        center_mm=(0.0, 0.0, float(args.source_z_mm)),
        rotation_deg=(0.0, 0.0, 0.0),
        cutoff_mm=(float(args.field_radius_mm), float(args.field_radius_mm)),
        histories=int(args.histories),
    )
    case_text = render_case_file(
        args,
        patient_bin=patient_bin,
        materials_file=materials_file,
        grid_shape=tag_grid.shape,
        spectrum_energies=spectrum_energies,
        spectrum_weights=spectrum_weights,
        source=source,
    )
    write_text_with_retries(parameter_file, case_text)

    summary = {
        "description": "Phase 28 Yang benchmark TsImageCube export",
        "voxel_mm": float(args.voxel_mm),
        "grid_shape": [int(v) for v in tag_grid.shape],
        "tag_values": used_tags,
        "materials": [
            {
                "tag": int(tag),
                "name": str(spec["name"]),
                "base_material": str(spec["base_material"]),
                "density_g_cm3": float(spec["density_g_cm3"]),
                "voxel_count": int(np.count_nonzero(tag_grid == int(tag))),
            }
            for tag, spec in zip(used_tags, used_specs)
        ],
        "paths": {
            "patient_material_tags_bin": str(patient_bin),
            "materials_include": str(materials_file),
            "beamline_file": str(parameter_file),
            "density_npz": str(density_npz),
        },
        "topas_validation_run_requested": bool(args.run_topas),
    }
    write_text_with_retries(args.out_root / "phase28_topas_tsimagecube_summary.json", json.dumps(summary, indent=2))

    if args.run_topas:
        run_topas_case(args, case_dir, parameter_file, dose_csv, log_file)

    print("=== PHASE 28 YANG TOPAS TSIMAGECUBE EXPORT COMPLETE ===")
    print(f"Output root: {args.out_root}")
    print(f"ImageCube binary: {patient_bin}")
    print(f"Materials include: {materials_file}")
    print(f"Beamline file: {parameter_file}")
    print(f"Summary: {args.out_root / 'phase28_topas_tsimagecube_summary.json'}")
    if args.run_topas:
        print(f"Validation dose CSV: {dose_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
