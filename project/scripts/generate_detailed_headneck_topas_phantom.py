#!/usr/bin/env python3
"""Convert the detailed synthetic head-and-neck phantom into a TOPAS ImageCube case.

This script reuses the richer anatomical phantom and creates:
1. a per-voxel material-tag binary for TsImageCube,
2. a custom TOPAS material-definition include with explicit densities,
3. a ready-to-run validation beamline deck,
4. inspection figures and summary tables for the materialized phantom.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

from build_asymmetric_sweep import PHYSICS_PROFILES, build_topas_env, format_physics_modules, has_nonempty_output, write_text_with_retries
from generate_detailed_headneck_phantom import build_detailed_headneck_phantom
from run_linac_6mv_polyenergetic_clinical_sfrt import load_spectrum

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.lines import Line2D
except Exception as exc:  # pragma: no cover
    raise RuntimeError("matplotlib is required for phantom visualization") from exc


@dataclass(frozen=True)
class MaterialSpec:
    tag: int
    name: str
    base_material: str
    density_g_cm3: float
    color: str
    description: str


@dataclass(frozen=True)
class SourceSpec:
    name: str
    center_mm: Tuple[float, float, float]
    rotation_deg: Tuple[float, float, float]
    cutoff_mm: Tuple[float, float]
    histories: int


MATERIAL_SPECS: List[MaterialSpec] = [
    MaterialSpec(0, "HN_AIR", "G4_AIR", 0.0012, "black", "Outside phantom and airway"),
    MaterialSpec(10, "HN_SOFT_TISSUE", "G4_TISSUE_SOFT_ICRP", 1.04, "lightpink", "Generic soft tissue"),
    MaterialSpec(11, "HN_BRAIN", "G4_TISSUE_SOFT_ICRP", 1.04, "skyblue", "Intracranial brain"),
    MaterialSpec(12, "HN_BBB", "G4_TISSUE_SOFT_ICRP", 1.06, "cyan", "Blood-brain barrier shell"),
    MaterialSpec(13, "HN_BRAINSTEM", "G4_TISSUE_SOFT_ICRP", 1.04, "dodgerblue", "Brainstem"),
    MaterialSpec(14, "HN_SPINAL_CORD", "G4_TISSUE_SOFT_ICRP", 1.04, "lightslategray", "Spinal cord"),
    MaterialSpec(20, "HN_SKULL_BONE", "G4_BONE_CORTICAL_ICRP", 1.85, "white", "Skull cortical bone"),
    MaterialSpec(21, "HN_MANDIBLE_MAXILLA_BONE", "G4_BONE_COMPACT_ICRU", 1.80, "lightgray", "Mandible and maxilla bone"),
    MaterialSpec(22, "HN_VERTEBRAL_BONE", "G4_BONE_CORTICAL_ICRP", 1.45, "gray", "Cervical vertebral bone"),
    MaterialSpec(30, "HN_PAROTID", "G4_TISSUE_SOFT_ICRP", 1.03, "green", "Parotid gland"),
    MaterialSpec(31, "HN_SUBMANDIBULAR", "G4_TISSUE_SOFT_ICRP", 1.04, "limegreen", "Submandibular gland"),
    MaterialSpec(32, "HN_THYROID", "G4_TISSUE_SOFT_ICRP", 1.05, "orange", "Thyroid"),
    MaterialSpec(33, "HN_PARATHYROID", "G4_TISSUE_SOFT_ICRP", 1.05, "yellow", "Parathyroids"),
    MaterialSpec(40, "HN_ARTERIAL_BLOOD", "G4_TISSUE_SOFT_ICRP", 1.06, "red", "Arterial blood pool"),
    MaterialSpec(41, "HN_VENOUS_BLOOD", "G4_TISSUE_SOFT_ICRP", 1.06, "blue", "Venous blood pool"),
    MaterialSpec(50, "HN_TUMOUR", "G4_TISSUE_SOFT_ICRP", 1.05, "magenta", "Tumour surrogate"),
]

TAG_TO_SPEC = {spec.tag: spec for spec in MATERIAL_SPECS}


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Create a TOPAS-ready heterogeneous head-and-neck ImageCube phantom.")
    parser.add_argument(
        "--template",
        type=Path,
        default=root / "topas" / "headneck_detailed_material_phantom_template.txt",
        help="TOPAS ImageCube template for the detailed heterogeneous phantom.",
    )
    parser.add_argument(
        "--spectrum-csv",
        type=Path,
        default=root / "data" / "linac_6mv_representative_spectrum.csv",
        help="Representative 6 MV spectrum as energy_mev,weight CSV.",
    )
    parser.add_argument(
        "--run-root",
        type=Path,
        default=root / "runs" / "detailed_headneck_topas_material_phantom",
        help="Output root for the TOPAS-ready phantom.",
    )
    parser.add_argument(
        "--topas-bin",
        type=str,
        default="/Users/kw/shellScripts/topas",
        help="TOPAS executable.",
    )
    parser.add_argument(
        "--g4-data-dir",
        type=str,
        default="/Applications/GEANT4",
        help="Directory containing Geant4 data folders.",
    )
    parser.add_argument(
        "--physics-profile",
        choices=sorted(PHYSICS_PROFILES),
        default="em_opt4_only",
        help="Named TOPAS modular physics profile.",
    )
    parser.add_argument("--histories", type=int, default=1000, help="Validation TOPAS histories.")
    parser.add_argument("--threads", type=int, default=2, help="TOPAS threads.")
    parser.add_argument("--seed", type=int, default=33, help="TOPAS RNG seed.")
    parser.add_argument("--size-x-cm", type=float, default=20.0, help="Phantom left-right size.")
    parser.add_argument("--size-y-cm", type=float, default=26.0, help="Phantom superior-inferior size.")
    parser.add_argument("--size-z-cm", type=float, default=18.0, help="Phantom anterior-posterior size.")
    parser.add_argument("--voxel-mm", type=float, default=1.5, help="Isotropic voxel size.")
    parser.add_argument("--field-radius-mm", type=float, default=90.0, help="Broad AP validation field radius.")
    parser.add_argument("--source-z-mm", type=float, default=-260.0, help="Source plane z position.")
    parser.add_argument("--cut-gamma-mm", type=float, default=0.01)
    parser.add_argument("--cut-electron-mm", type=float, default=0.01)
    parser.add_argument("--cut-positron-mm", type=float, default=0.01)
    parser.add_argument("--dpi", type=int, default=260, help="Figure DPI.")
    parser.add_argument(
        "--run-topas",
        action="store_true",
        help="Run a small validation dose calculation after generating the TOPAS case.",
    )
    return parser.parse_args()


def write_image_cube(tag_grid_xyz: np.ndarray, out_file: Path) -> None:
    np.asarray(tag_grid_xyz, dtype=np.int16).transpose(2, 1, 0).tofile(out_file)


def build_material_tag_grid(structures: Dict[str, np.ndarray]) -> np.ndarray:
    shape = structures["BODY"].shape
    grid = np.zeros(shape, dtype=np.int16)

    grid[structures["BODY"]] = 10
    grid[structures["PAROTID_L"] | structures["PAROTID_R"]] = 30
    grid[structures["SUBMANDIBULAR_L"] | structures["SUBMANDIBULAR_R"]] = 31
    grid[structures["THYROID"]] = 32
    grid[structures["PARATHYROIDS"]] = 33
    grid[structures["BRAIN"]] = 11
    grid[structures["BLOOD_BRAIN_BARRIER"]] = 12
    grid[structures["BRAINSTEM"]] = 13
    grid[structures["SPINAL_CORD"]] = 14
    grid[structures["ARTERIES"]] = 40
    grid[structures["VEINS"]] = 41
    grid[structures["TUMOUR"]] = 50
    grid[structures["MAXILLA"] | structures["MANDIBLE"]] = 21
    grid[structures["SKULL"]] = 20
    grid[structures["VERTEBRAE"]] = 22
    grid[structures["AIRWAY"] | structures["TRACHEA"]] = 0

    return grid


def build_density_from_tags(tag_grid: np.ndarray) -> np.ndarray:
    density = np.zeros(tag_grid.shape, dtype=np.float32)
    for tag, spec in TAG_TO_SPEC.items():
        density[tag_grid == int(tag)] = float(spec.density_g_cm3)
    return density


def render_materials_include(specs: Sequence[MaterialSpec]) -> str:
    lines: List[str] = ["# Custom heterogeneous materials for the detailed head-and-neck phantom", ""]
    for spec in specs:
        lines.extend(
            [
                f's:Ma/{spec.name}/BaseMaterial = "{spec.base_material}"',
                f"d:Ma/{spec.name}/Density = {spec.density_g_cm3:.6f} g/cm3",
                f's:Ma/{spec.name}/DefaultColor = "{spec.color}"',
                "",
            ]
        )
    return "\n".join(lines).strip() + "\n"


def render_source_block(spectrum_energies: Sequence[float], spectrum_weights: Sequence[float], source: SourceSpec) -> str:
    spectrum_count = len(spectrum_energies)
    spectrum_values = " ".join(f"{float(v):.6f}" for v in spectrum_energies)
    spectrum_weight_values = " ".join(f"{float(v):.8f}" for v in spectrum_weights)
    group_name = f"BeamOrigin_{source.name}"
    source_name = f"Source_{source.name}"
    return "\n".join(
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


def fill_template(template_text: str, replacements: Dict[str, str]) -> str:
    rendered = template_text
    for key, value in replacements.items():
        rendered = rendered.replace(key, value)
    return rendered


def render_case_file(
    args: argparse.Namespace,
    *,
    patient_bin: Path,
    materials_file: Path,
    grid_shape: Sequence[int],
    voxel_size_mm: Sequence[float],
    spectrum_energies: Sequence[float],
    spectrum_weights: Sequence[float],
    source: SourceSpec,
) -> str:
    template_text = args.template.read_text(encoding="utf-8")
    world_hlx_cm = max(40.0, 0.5 * float(args.size_x_cm) + 20.0)
    world_hly_cm = max(40.0, 0.5 * float(args.size_y_cm) + 20.0)
    world_hlz_cm = max(40.0, 0.5 * float(args.size_z_cm) + 20.0)
    show_interval = max(1000, int(args.histories) // 10)

    material_tag_values = " ".join(str(int(spec.tag)) for spec in MATERIAL_SPECS)
    material_name_values = " ".join(f'"{spec.name}"' for spec in MATERIAL_SPECS)
    patient_input_dir = str(patient_bin.parent.resolve())
    if not patient_input_dir.endswith("/"):
        patient_input_dir += "/"

    replacements = {
        "__G4_DATA_DIR__": str(Path(args.g4_data_dir).expanduser()),
        "__PHYSICS_MODULES__": format_physics_modules(str(args.physics_profile)),
        "__CUT_GAMMA_MM__": f"{float(args.cut_gamma_mm):.6f}",
        "__CUT_ELECTRON_MM__": f"{float(args.cut_electron_mm):.6f}",
        "__CUT_POSITRON_MM__": f"{float(args.cut_positron_mm):.6f}",
        "__MATERIALS_INCLUDE_FILE__": materials_file.name,
        "__WORLD_HLX_CM__": f"{world_hlx_cm:.6f}",
        "__WORLD_HLY_CM__": f"{world_hly_cm:.6f}",
        "__WORLD_HLZ_CM__": f"{world_hlz_cm:.6f}",
        "__PATIENT_INPUT_DIR__": patient_input_dir,
        "__PATIENT_INPUT_FILE__": patient_bin.name,
        "__XBINS__": str(int(grid_shape[0])),
        "__YBINS__": str(int(grid_shape[1])),
        "__ZBINS__": str(int(grid_shape[2])),
        "__VOXEL_SIZE_X_MM__": f"{float(voxel_size_mm[0]):.6f}",
        "__VOXEL_SIZE_Y_MM__": f"{float(voxel_size_mm[1]):.6f}",
        "__VOXEL_SIZE_Z_MM__": f"{float(voxel_size_mm[2]):.6f}",
        "__MATERIAL_TAG_COUNT__": str(len(MATERIAL_SPECS)),
        "__MATERIAL_TAG_VALUES__": material_tag_values,
        "__MATERIAL_NAME_VALUES__": material_name_values,
        "__OUTPUT_STEM__": "dosedata",
        "__SOURCE_BLOCK__": render_source_block(spectrum_energies, spectrum_weights, source),
        "__N_THREADS__": str(int(args.threads)),
        "__SEED__": str(int(args.seed)),
        "__SHOW_HISTORY_INTERVAL__": str(int(show_interval)),
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
            "TOPAS validation run failed for the detailed heterogeneous head-and-neck phantom.\n"
            f"Return code: {result.returncode}\n"
            f"Dose CSV present: {has_nonempty_output(dose_csv)}\n"
            f"Recent log:\n{tail}"
        )


def plot_material_map(out_file: Path, phantom: Dict[str, object], tag_grid: np.ndarray, *, dpi: int) -> None:
    axes_mm = phantom["axes_mm"]
    structures = phantom["structures"]
    x_cm = axes_mm["x"] / 10.0
    y_cm = axes_mm["y"] / 10.0
    z_cm = axes_mm["z"] / 10.0

    iz_mid = int(np.argmin(np.abs(axes_mm["z"] - 0.0)))
    iy_head = int(np.argmin(np.abs(axes_mm["y"] - 42.0)))
    iy_neck = int(np.argmin(np.abs(axes_mm["y"] + 48.0)))
    ix_mid = int(np.argmin(np.abs(axes_mm["x"] - 0.0)))

    slices = [
        (
            "Coronal material slice (z = 0 cm)",
            tag_grid[:, :, iz_mid],
            structures["BODY"][:, :, iz_mid],
            x_cm,
            y_cm,
            "x (cm)",
            "y (cm)",
        ),
        (
            "Axial material slice at brain level",
            tag_grid[:, iy_head, :],
            structures["BODY"][:, iy_head, :],
            x_cm,
            z_cm,
            "x (cm)",
            "z (cm)",
        ),
        (
            "Axial material slice at thyroid / nodal level",
            tag_grid[:, iy_neck, :],
            structures["BODY"][:, iy_neck, :],
            x_cm,
            z_cm,
            "x (cm)",
            "z (cm)",
        ),
        (
            "Sagittal material slice (x = 0 cm)",
            tag_grid[ix_mid, :, :],
            structures["BODY"][ix_mid, :, :],
            y_cm,
            z_cm,
            "y (cm)",
            "z (cm)",
        ),
    ]

    colors = [spec.color for spec in MATERIAL_SPECS]
    cmap = ListedColormap(colors)
    ordered_tags = [spec.tag for spec in MATERIAL_SPECS]
    index_map = {tag: idx for idx, tag in enumerate(ordered_tags)}
    indexed_grid = np.vectorize(index_map.get)(tag_grid)

    fig, axes = plt.subplots(2, 2, figsize=(15.5, 11.5), constrained_layout=True)
    axes_list = axes.ravel()
    for ax, (title, image, body_slice, axis_a, axis_b, xlabel, ylabel) in zip(axes_list, slices):
        indexed_image = np.vectorize(index_map.get)(image)
        extent = [float(axis_a[0]), float(axis_a[-1]), float(axis_b[0]), float(axis_b[-1])]
        ax.imshow(indexed_image.T, origin="lower", cmap=cmap, vmin=0, vmax=len(ordered_tags) - 1, extent=extent)
        ax.contour(axis_a, axis_b, body_slice.T.astype(float), levels=[0.5], colors=["white"], linewidths=0.4, alpha=0.35)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

    legend_handles = [Line2D([0], [0], color=spec.color, lw=4, label=f"{spec.name} ({spec.density_g_cm3:.3f} g/cm$^3$)") for spec in MATERIAL_SPECS]
    fig.legend(handles=legend_handles, loc="lower center", ncol=2, fontsize=8, frameon=True)
    fig.suptitle("TOPAS material-tag phantom with explicit per-structure densities", fontsize=14)
    fig.savefig(out_file, dpi=dpi)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    args.run_root.mkdir(parents=True, exist_ok=True)
    case_dir = args.run_root / "case"
    case_dir.mkdir(parents=True, exist_ok=True)
    phantom_dir = args.run_root / "phantom"
    phantom_dir.mkdir(parents=True, exist_ok=True)

    phantom = build_detailed_headneck_phantom(args)
    structures = phantom["structures"]
    axes_mm = phantom["axes_mm"]
    meta = dict(phantom["meta"])

    tag_grid = build_material_tag_grid(structures)
    density_from_tags = build_density_from_tags(tag_grid)
    source = SourceSpec(
        name="ap_validation",
        center_mm=(0.0, 0.0, float(args.source_z_mm)),
        rotation_deg=(0.0, 0.0, 0.0),
        cutoff_mm=(float(args.field_radius_mm), float(args.field_radius_mm)),
        histories=int(args.histories),
    )

    patient_bin = case_dir / "patient_material_tags.bin"
    materials_file = case_dir / "materials.txt"
    parameter_file = case_dir / "beamline.txt"
    dose_csv = case_dir / "dosedata.csv"
    log_file = case_dir / "topas.log"

    write_image_cube(tag_grid, patient_bin)
    write_text_with_retries(materials_file, render_materials_include(MATERIAL_SPECS))

    spectrum_energies, spectrum_weights = load_spectrum(args.spectrum_csv)
    case_text = render_case_file(
        args,
        patient_bin=patient_bin,
        materials_file=materials_file,
        grid_shape=tag_grid.shape,
        voxel_size_mm=(float(args.voxel_mm), float(args.voxel_mm), float(args.voxel_mm)),
        spectrum_energies=spectrum_energies,
        spectrum_weights=spectrum_weights,
        source=source,
    )
    write_text_with_retries(parameter_file, case_text)

    np.savez_compressed(phantom_dir / "topas_material_tag_grid.npz", material_tags=tag_grid.astype(np.int16))
    np.savez_compressed(phantom_dir / "topas_material_density_from_tags.npz", density_g_cm3=density_from_tags.astype(np.float32))

    material_summary = {
        "grid_shape": [int(v) for v in tag_grid.shape],
        "voxel_size_mm": [float(args.voxel_mm), float(args.voxel_mm), float(args.voxel_mm)],
        "materials": [
            {
                "tag": int(spec.tag),
                "name": spec.name,
                "base_material": spec.base_material,
                "density_g_cm3": float(spec.density_g_cm3),
                "description": spec.description,
                "voxel_count": int(np.count_nonzero(tag_grid == spec.tag)),
            }
            for spec in MATERIAL_SPECS
        ],
        "source": {
            "name": source.name,
            "center_mm": [float(v) for v in source.center_mm],
            "rotation_deg": [float(v) for v in source.rotation_deg],
            "cutoff_mm": [float(v) for v in source.cutoff_mm],
            "histories": int(source.histories),
        },
        "anatomical_note": meta["anatomical_note"],
    }
    write_text_with_retries(phantom_dir / "topas_material_summary.json", json.dumps(material_summary, indent=2))

    figure_path = args.run_root / "figure1_topas_material_phantom.png"
    plot_material_map(figure_path, phantom, tag_grid, dpi=int(args.dpi))

    if args.run_topas:
        run_topas_case(args, case_dir, parameter_file, dose_csv, log_file)

    print(figure_path)
    print(parameter_file)
    print(materials_file)
    print(patient_bin)
    print(phantom_dir / "topas_material_summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
