#!/usr/bin/env python3
"""Run the legacy SFRT plan on the detailed heterogeneous head-and-neck phantom.

This workflow keeps the earlier published/simple SFRT source geometry fixed and
applies it directly to the new heterogeneous head-and-neck ImageCube phantom
with explicit materials and densities. It then compares physical dose metrics
against the older simpler phantom run to quantify the impact of anatomy and
heterogeneity on planning metrics.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

from analyze_topas_outputs import load_topas_grid
from build_asymmetric_sweep import PHYSICS_PROFILES, build_topas_env, format_physics_modules, has_nonempty_output, write_text_with_retries
from generate_detailed_headneck_phantom import build_detailed_headneck_phantom, ellipsoid_mask
from generate_detailed_headneck_topas_phantom import MATERIAL_SPECS, build_material_tag_grid, render_materials_include
from run_linac_6mv_polyenergetic_clinical_sfrt import load_spectrum
from run_phase13_headneck_voxel_lattice import (
    build_plan_sources,
    compute_dvh,
    compute_structure_metrics,
    pick_lattice_spots,
    save_csv,
)

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception as exc:  # pragma: no cover
    raise RuntimeError("matplotlib is required for figure generation") from exc


@dataclass(frozen=True)
class SourceSpec:
    name: str
    center_mm: Tuple[float, float, float]
    rotation_deg: Tuple[float, float, float]
    cutoff_mm: Tuple[float, float]
    histories: int


COMMON_STRUCTURE_ORDER = ["PTV", "GTV", "SPINAL_CORD", "BRAINSTEM", "PAROTID_L", "PAROTID_R", "MANDIBLE"]
NEW_ONLY_STRUCTURE_ORDER = ["THYROID", "PARATHYROIDS", "BRAIN", "BLOOD_BRAIN_BARRIER"]


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description=(
            "Apply the legacy AP-base lattice-boost plan to the detailed heterogeneous "
            "head-and-neck phantom and compare physical dose metrics to the older "
            "simpler phantom run."
        )
    )
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
        "--legacy-source-csv",
        type=Path,
        default=root / "runs" / "linac_6mv_headneck_voxel_lattice_sfrt_apbase" / "analysis" / "phase13_plan_sources_full.csv",
        help="Exact source list from the older simpler phantom plan.",
    )
    parser.add_argument(
        "--legacy-analysis-dir",
        type=Path,
        default=root / "runs" / "linac_6mv_headneck_voxel_lattice_sfrt_apbase" / "analysis",
        help="Analysis directory from the older simpler phantom run for comparison.",
    )
    parser.add_argument(
        "--run-root",
        type=Path,
        default=root / "runs" / "linac_6mv_headneck_detailed_voxel_lattice_sfrt_apbase",
        help="Output root for the detailed phantom comparison run.",
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
    parser.add_argument("--histories", type=int, default=1_000_000, help="Total TOPAS histories.")
    parser.add_argument("--threads", type=int, default=8, help="TOPAS threads.")
    parser.add_argument("--seed", type=int, default=33, help="TOPAS RNG seed.")
    parser.add_argument(
        "--plan-mode",
        choices=("legacy", "direct"),
        default="legacy",
        help="Use the legacy source list verbatim or re-place lattice spots directly on the detailed phantom.",
    )
    parser.add_argument("--size-x-cm", type=float, default=20.0, help="Phantom left-right size.")
    parser.add_argument("--size-y-cm", type=float, default=26.0, help="Phantom superior-inferior size.")
    parser.add_argument("--size-z-cm", type=float, default=18.0, help="Phantom anterior-posterior size.")
    parser.add_argument("--voxel-mm", type=float, default=1.5, help="Isotropic voxel size.")
    parser.add_argument("--spot-radius-mm", type=float, default=8.0, help="Direct-plan lattice beamlet radius.")
    parser.add_argument("--base-margin-mm", type=float, default=6.0, help="Margin added to broad-field radius in direct planning mode.")
    parser.add_argument(
        "--base-history-fraction",
        type=float,
        default=0.95,
        help="Fraction of histories assigned to the broad AP base field in direct planning mode.",
    )
    parser.add_argument(
        "--lattice-spacing-mm",
        nargs=3,
        type=float,
        default=[18.0, 20.0, 18.0],
        help="Nominal lattice spacing in x, y, z for direct planning mode.",
    )
    parser.add_argument(
        "--spot-limit",
        type=int,
        default=7,
        help="Maximum number of lattice vertices in direct planning mode.",
    )
    parser.add_argument(
        "--prescription-gy",
        type=float,
        default=6.0,
        help="Physical PTV prescription used to normalize the plan to D95.",
    )
    parser.add_argument("--cut-gamma-mm", type=float, default=0.01)
    parser.add_argument("--cut-electron-mm", type=float, default=0.01)
    parser.add_argument("--cut-positron-mm", type=float, default=0.01)
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip TOPAS if the dose CSV already exists and is non-empty.",
    )
    parser.add_argument(
        "--analyze-only",
        action="store_true",
        help="Skip TOPAS generation/run and analyze an existing detailed-phantom dose CSV.",
    )
    parser.add_argument("--dpi", type=int, default=260, help="Figure DPI.")
    return parser.parse_args()


def write_image_cube(tag_grid_xyz: np.ndarray, out_file: Path) -> None:
    np.asarray(tag_grid_xyz, dtype=np.int16).transpose(2, 1, 0).tofile(out_file)


def load_legacy_sources(csv_path: Path, total_histories: int) -> List[SourceSpec]:
    rows: List[Dict[str, str]] = []
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    if not rows:
        raise RuntimeError(f"No source rows found in {csv_path}")
    legacy_total = sum(int(float(row["histories"])) for row in rows)
    if legacy_total <= 0:
        raise RuntimeError(f"Legacy source file has non-positive total histories: {csv_path}")

    scaled = np.array([int(float(row["histories"])) for row in rows], dtype=np.float64) * (float(total_histories) / float(legacy_total))
    histories = np.floor(scaled).astype(int)
    remainder = int(total_histories) - int(histories.sum())
    if remainder > 0:
        order = np.argsort(-(scaled - histories))
        histories[order[:remainder]] += 1

    sources: List[SourceSpec] = []
    for row, hist in zip(rows, histories.tolist()):
        sources.append(
            SourceSpec(
                name=row["source_name"],
                center_mm=(float(row["trans_x_mm"]), float(row["trans_y_mm"]), float(row["trans_z_mm"])),
                rotation_deg=(float(row["rot_x_deg"]), float(row["rot_y_deg"]), float(row["rot_z_deg"])),
                cutoff_mm=(float(row["cutoff_x_mm"]), float(row["cutoff_y_mm"])),
                histories=int(hist),
            )
        )
    return sources


def render_source_block(sources: Sequence[SourceSpec], spectrum_energies: Sequence[float], spectrum_weights: Sequence[float]) -> str:
    lines: List[str] = []
    spectrum_count = len(spectrum_energies)
    spectrum_values = " ".join(f"{float(v):.6f}" for v in spectrum_energies)
    spectrum_weight_values = " ".join(f"{float(v):.8f}" for v in spectrum_weights)

    for spec in sources:
        group_name = f"BeamOrigin_{spec.name}"
        source_name = f"Source_{spec.name}"
        lines.extend(
            [
                f's:Ge/{group_name}/Type = "Group"',
                f's:Ge/{group_name}/Parent = "World"',
                f"d:Ge/{group_name}/TransX = {spec.center_mm[0]:.6f} mm",
                f"d:Ge/{group_name}/TransY = {spec.center_mm[1]:.6f} mm",
                f"d:Ge/{group_name}/TransZ = {spec.center_mm[2]:.6f} mm",
                f"d:Ge/{group_name}/RotX = {spec.rotation_deg[0]:.6f} deg",
                f"d:Ge/{group_name}/RotY = {spec.rotation_deg[1]:.6f} deg",
                f"d:Ge/{group_name}/RotZ = {spec.rotation_deg[2]:.6f} deg",
                f's:So/{source_name}/Type = "Beam"',
                f's:So/{source_name}/Component = "{group_name}"',
                f's:So/{source_name}/BeamParticle = "gamma"',
                f's:So/{source_name}/BeamEnergySpectrumType = "Discrete"',
                f"dv:So/{source_name}/BeamEnergySpectrumValues = {spectrum_count} {spectrum_values} MeV",
                f"uv:So/{source_name}/BeamEnergySpectrumWeights = {spectrum_count} {spectrum_weight_values}",
                f's:So/{source_name}/BeamPositionDistribution = "Flat"',
                f's:So/{source_name}/BeamPositionCutoffShape = "Ellipse"',
                f"d:So/{source_name}/BeamPositionCutoffX = {spec.cutoff_mm[0]:.6f} mm",
                f"d:So/{source_name}/BeamPositionCutoffY = {spec.cutoff_mm[1]:.6f} mm",
                f's:So/{source_name}/BeamAngularDistribution = "None"',
                f"i:So/{source_name}/NumberOfHistoriesInRun = {spec.histories}",
                "",
            ]
        )
    return "\n".join(lines).strip() + "\n"


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
    sources: Sequence[SourceSpec],
) -> str:
    template_text = getattr(args, "_cached_template_text", None)
    if template_text is None:
        template_path = Path(args.template)
        template_text = template_path.read_text(encoding="utf-8")
        try:
            setattr(args, "_cached_template_text", template_text)
        except Exception:
            pass
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
        "__SOURCE_BLOCK__": render_source_block(sources, spectrum_energies, spectrum_weights),
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
            "TOPAS run failed for the detailed heterogeneous head-and-neck SFRT plan.\n"
            f"Return code: {result.returncode}\n"
            f"Dose CSV present: {has_nonempty_output(dose_csv)}\n"
            f"Recent log:\n{tail}"
        )


def build_detailed_plan_phantom(args: argparse.Namespace) -> Dict[str, object]:
    phantom = build_detailed_headneck_phantom(args)
    structures = dict(phantom["structures"])
    axes_mm = phantom["axes_mm"]
    x_mm = axes_mm["x"]
    y_mm = axes_mm["y"]
    z_mm = axes_mm["z"]

    gtv_mask = structures["TUMOUR"]
    ptv_primary = ellipsoid_mask(x_mm, y_mm, z_mm, center_mm=(20.0, 4.0, -4.0), radii_mm=(29.0, 31.0, 26.0))
    ptv_nodal = ellipsoid_mask(x_mm, y_mm, z_mm, center_mm=(34.0, -35.0, 1.0), radii_mm=(25.0, 38.0, 22.0))
    ptv_mask = (ptv_primary | ptv_nodal) & structures["BODY"] & ~structures["AIRWAY"]
    hypoxic_primary = ellipsoid_mask(x_mm, y_mm, z_mm, center_mm=(20.0, 4.0, -4.0), radii_mm=(14.0, 16.0, 13.0))
    hypoxic_nodal = ellipsoid_mask(x_mm, y_mm, z_mm, center_mm=(34.0, -35.0, 1.0), radii_mm=(10.0, 18.0, 10.0))
    hypoxic_mask = (hypoxic_primary | hypoxic_nodal) & gtv_mask

    structures["GTV"] = gtv_mask
    structures["PTV"] = ptv_mask
    structures["HYPOXIA"] = hypoxic_mask
    structures["MANDIBLE"] = structures["MANDIBLE"]
    structures["BONE"] = structures["SKULL"] | structures["MANDIBLE"] | structures["MAXILLA"] | structures["VERTEBRAE"]

    tag_grid = build_material_tag_grid(structures)

    meta = dict(phantom["meta"])
    voxel_volume_cc = float(meta["voxel_volume_cc"])
    meta["structure_volumes_cc"] = dict(meta["structure_volumes_cc"])
    meta["structure_volumes_cc"]["GTV"] = float(np.count_nonzero(gtv_mask) * voxel_volume_cc)
    meta["structure_volumes_cc"]["PTV"] = float(np.count_nonzero(ptv_mask) * voxel_volume_cc)
    meta["structure_volumes_cc"]["MANDIBLE"] = float(np.count_nonzero(structures["MANDIBLE"]) * voxel_volume_cc)
    meta["anatomical_note"] = (
        "Detailed heterogeneous head-and-neck phantom with explicit brain, BBB, thyroid-parathyroid complex, "
        "arterial/venous network, salivary glands, cervical spine, and bulky right-sided oropharyngeal-nodal target."
    )

    return {
        "tag_grid": tag_grid,
        "structures": structures,
        "axes_mm": axes_mm,
        "meta": meta,
    }


def load_metrics_csv(csv_path: Path) -> Dict[str, Dict[str, float]]:
    metrics: Dict[str, Dict[str, float]] = {}
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            structure = row["structure"]
            metrics[structure] = {}
            for key, value in row.items():
                if key in {"domain", "structure"} or value == "":
                    continue
                metrics[structure][key] = float(value)
    return metrics


def load_dvh_csv(csv_path: Path) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    dose_axis = np.array([float(row["dose_gy"]) for row in rows], dtype=np.float32)
    structures: Dict[str, np.ndarray] = {}
    if not rows:
        return dose_axis, structures
    for key in rows[0].keys():
        if key.endswith("_physical_pct"):
            name = key[: -len("_physical_pct")]
            structures[name] = np.array([float(row[key]) for row in rows], dtype=np.float32)
    return dose_axis, structures


def plot_dose_slices(out_file: Path, axes_mm: Dict[str, np.ndarray], physical_dose: np.ndarray, structures: Dict[str, np.ndarray], *, dpi: int) -> None:
    y_index = int(np.argmin(np.abs(axes_mm["y"] - 0.0)))
    z_index = int(np.argmin(np.abs(axes_mm["z"] - 0.0)))
    x_cm = axes_mm["x"] / 10.0
    y_cm = axes_mm["y"] / 10.0
    z_cm = axes_mm["z"] / 10.0

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.0), constrained_layout=True)
    panels = [
        (
            axes[0],
            physical_dose[:, y_index, :],
            structures["PTV"][:, y_index, :],
            structures["SPINAL_CORD"][:, y_index, :],
            x_cm,
            z_cm,
            "Detailed phantom physical dose: axial-style slice",
            "x (cm)",
            "z (cm)",
        ),
        (
            axes[1],
            physical_dose[:, :, z_index],
            structures["PTV"][:, :, z_index],
            structures["SPINAL_CORD"][:, :, z_index],
            x_cm,
            y_cm,
            "Detailed phantom physical dose: coronal slice",
            "x (cm)",
            "y (cm)",
        ),
    ]

    for ax, image, ptv_slice, cord_slice, axis_a_cm, axis_b_cm, title, xlabel, ylabel in panels:
        im = ax.imshow(
            image.T,
            origin="lower",
            extent=[float(axis_a_cm[0]), float(axis_a_cm[-1]), float(axis_b_cm[0]), float(axis_b_cm[-1])],
            cmap="inferno",
        )
        ax.contour(axis_a_cm, axis_b_cm, ptv_slice.T.astype(float), levels=[0.5], colors=["cyan"], linewidths=1.2)
        ax.contour(axis_a_cm, axis_b_cm, cord_slice.T.astype(float), levels=[0.5], colors=["white"], linewidths=1.0)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Gy")

    fig.savefig(out_file, dpi=dpi)
    plt.close(fig)


def plot_dvh_comparison(
    out_file: Path,
    dose_axis_new: np.ndarray,
    new_curves: Dict[str, np.ndarray],
    dose_axis_old: np.ndarray,
    old_curves: Dict[str, np.ndarray],
    *,
    dpi: int,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.9), constrained_layout=True)
    structures_left = ["PTV", "GTV", "SPINAL_CORD", "BRAINSTEM"]
    structures_right = ["PAROTID_L", "PAROTID_R", "MANDIBLE"]
    colors = {
        "PTV": "tab:red",
        "GTV": "tab:orange",
        "SPINAL_CORD": "tab:blue",
        "BRAINSTEM": "tab:purple",
        "PAROTID_L": "tab:green",
        "PAROTID_R": "tab:olive",
        "MANDIBLE": "tab:brown",
    }

    for structure in structures_left:
        axes[0].plot(dose_axis_old, old_curves[structure], color=colors[structure], linestyle="--", label=f"{structure} simple")
        axes[0].plot(dose_axis_new, new_curves[structure], color=colors[structure], linestyle="-", label=f"{structure} detailed")
    for structure in structures_right:
        axes[1].plot(dose_axis_old, old_curves[structure], color=colors[structure], linestyle="--", label=f"{structure} simple")
        axes[1].plot(dose_axis_new, new_curves[structure], color=colors[structure], linestyle="-", label=f"{structure} detailed")

    axes[0].set_title("Target and serial-organ DVHs")
    axes[1].set_title("Parotid and mandible DVHs")
    for ax in axes:
        ax.set_xlabel("Dose / Gy")
        ax.set_ylabel("Volume receiving at least dose (%)")
        ax.set_ylim(0.0, 100.0)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8, ncol=2)

    fig.savefig(out_file, dpi=dpi)
    plt.close(fig)


def plot_metric_comparison(
    out_file: Path,
    old_metrics: Dict[str, Dict[str, float]],
    new_metrics: Dict[str, Dict[str, float]],
    *,
    dpi: int,
) -> None:
    categories = [
        ("PTV D95", old_metrics["PTV"]["d95_gy"], new_metrics["PTV"]["d95_gy"]),
        ("PTV D2", old_metrics["PTV"]["d2_gy"], new_metrics["PTV"]["d2_gy"]),
        ("PTV mean", old_metrics["PTV"]["mean_gy"], new_metrics["PTV"]["mean_gy"]),
        ("Cord D2", old_metrics["SPINAL_CORD"]["d2_gy"], new_metrics["SPINAL_CORD"]["d2_gy"]),
        ("Brainstem D2", old_metrics["BRAINSTEM"]["d2_gy"], new_metrics["BRAINSTEM"]["d2_gy"]),
        ("Parotid R mean", old_metrics["PAROTID_R"]["mean_gy"], new_metrics["PAROTID_R"]["mean_gy"]),
        ("Parotid L mean", old_metrics["PAROTID_L"]["mean_gy"], new_metrics["PAROTID_L"]["mean_gy"]),
        ("Mandible mean", old_metrics["MANDIBLE"]["mean_gy"], new_metrics["MANDIBLE"]["mean_gy"]),
    ]
    x = np.arange(len(categories))
    width = 0.38
    fig, ax = plt.subplots(figsize=(12.5, 4.9), constrained_layout=True)
    ax.bar(x - width / 2.0, [row[1] for row in categories], width=width, label="Simple phantom", color="tab:blue")
    ax.bar(x + width / 2.0, [row[2] for row in categories], width=width, label="Detailed heterogeneous phantom", color="tab:red")
    ax.set_xticks(x)
    ax.set_xticklabels([row[0] for row in categories], rotation=20, ha="right")
    ax.set_ylabel("Gy")
    ax.set_title("Dose-metric comparison: simple versus detailed heterogeneous phantom")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.savefig(out_file, dpi=dpi)
    plt.close(fig)


def metrics_rows(domain_label: str, metrics: Dict[str, Dict[str, float]]) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for structure, struct_metrics in metrics.items():
        row: Dict[str, object] = {"domain": domain_label, "structure": structure}
        row.update({key: float(value) for key, value in struct_metrics.items()})
        rows.append(row)
    return rows


def build_comparison_rows(old_metrics: Dict[str, Dict[str, float]], new_metrics: Dict[str, Dict[str, float]]) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    metric_keys = ["mean_gy", "d2_gy", "d95_gy", "d98_gy", "v95_pct", "v100_pct", "coverage_ratio", "v5_pct", "v8_pct", "v10_pct"]
    for structure in COMMON_STRUCTURE_ORDER:
        for key in metric_keys:
            if key not in old_metrics.get(structure, {}) or key not in new_metrics.get(structure, {}):
                continue
            old_value = float(old_metrics[structure][key])
            new_value = float(new_metrics[structure][key])
            rows.append(
                {
                    "structure": structure,
                    "metric": key,
                    "simple_phantom": old_value,
                    "detailed_phantom": new_value,
                    "delta_abs": new_value - old_value,
                    "delta_pct_of_simple": ((new_value - old_value) / old_value * 100.0) if abs(old_value) > 1e-9 else np.nan,
                }
            )
    return rows


def write_markdown_report(
    out_file: Path,
    summary: Dict[str, object],
    old_metrics: Dict[str, Dict[str, float]],
    new_metrics: Dict[str, Dict[str, float]],
    extra_metrics: Dict[str, Dict[str, float]],
) -> None:
    lines = [
        "# Detailed Heterogeneous Phantom SFRT Comparison",
        "",
        "## Setup",
        f"- Planning mode: `{summary['plan']['mode']}`",
        f"- Legacy source geometry reference: `{summary['legacy_source_csv']}`",
        f"- Total histories: `{summary['plan']['histories']}`",
        f"- Prescription normalization: PTV D95 scaled to `{summary['prescription_gy']:.2f} Gy`",
        f"- Detailed phantom grid: `{summary['phantom']['grid_shape']}` at `{summary['phantom']['voxel_size_mm']}` mm",
        f"- Detailed phantom note: {summary['phantom']['anatomical_note']}",
        "",
        "## Common-Structure Comparison",
        f"- PTV D95: simple `{old_metrics['PTV']['d95_gy']:.2f} Gy` -> detailed `{new_metrics['PTV']['d95_gy']:.2f} Gy`",
        f"- PTV mean: simple `{old_metrics['PTV']['mean_gy']:.2f} Gy` -> detailed `{new_metrics['PTV']['mean_gy']:.2f} Gy`",
        f"- Spinal cord D2: simple `{old_metrics['SPINAL_CORD']['d2_gy']:.2f} Gy` -> detailed `{new_metrics['SPINAL_CORD']['d2_gy']:.2f} Gy`",
        f"- Brainstem D2: simple `{old_metrics['BRAINSTEM']['d2_gy']:.2f} Gy` -> detailed `{new_metrics['BRAINSTEM']['d2_gy']:.2f} Gy`",
        f"- Right parotid mean: simple `{old_metrics['PAROTID_R']['mean_gy']:.2f} Gy` -> detailed `{new_metrics['PAROTID_R']['mean_gy']:.2f} Gy`",
        f"- Left parotid mean: simple `{old_metrics['PAROTID_L']['mean_gy']:.2f} Gy` -> detailed `{new_metrics['PAROTID_L']['mean_gy']:.2f} Gy`",
        "",
        "## New Structures Available In The Detailed Phantom",
    ]
    for structure in NEW_ONLY_STRUCTURE_ORDER:
        if structure not in extra_metrics:
            continue
        metrics = extra_metrics[structure]
        lines.append(
            f"- {structure}: mean `{metrics['mean_gy']:.2f} Gy`, D2 `{metrics['d2_gy']:.2f} Gy`, D95 `{metrics['d95_gy']:.2f} Gy`"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "- This comparison isolates the phantom effect by keeping the old source geometry fixed and changing only the underlying anatomy/material model.",
            "- Any shift in D95, D2, or mean dose therefore reflects the influence of heterogeneous materials and richer anatomical detail rather than a different optimizer output.",
        ]
    )
    write_text_with_retries(out_file, "\n".join(lines) + "\n")


def main() -> int:
    args = parse_args()
    args.run_root.mkdir(parents=True, exist_ok=True)

    case_dir = args.run_root / "case"
    phantom_dir = args.run_root / "phantom"
    analysis_dir = args.run_root / "analysis"
    case_dir.mkdir(parents=True, exist_ok=True)
    phantom_dir.mkdir(parents=True, exist_ok=True)
    analysis_dir.mkdir(parents=True, exist_ok=True)

    phantom = build_detailed_plan_phantom(args)
    tag_grid = phantom["tag_grid"]
    structures = phantom["structures"]
    axes_mm = phantom["axes_mm"]
    phantom_meta = phantom["meta"]
    voxel_volume_cc = float(phantom_meta["voxel_volume_cc"])

    patient_bin = case_dir / "patient_material_tags.bin"
    materials_file = case_dir / "materials.txt"
    parameter_file = case_dir / "beamline.txt"
    dose_csv = case_dir / "dosedata.csv"
    log_file = case_dir / "topas.log"

    write_image_cube(tag_grid, patient_bin)
    write_text_with_retries(materials_file, render_materials_include(MATERIAL_SPECS))
    write_text_with_retries(phantom_dir / "phantom_meta.json", json.dumps(phantom_meta, indent=2))

    plan_meta: Dict[str, object]
    if args.plan_mode == "direct":
        spot_centers_mm = pick_lattice_spots(
            structures["GTV"],
            axes_mm,
            args.lattice_spacing_mm,
            spot_radius_mm=float(args.spot_radius_mm),
            limit=int(args.spot_limit),
        )
        plan_meta = build_plan_sources(args, axes_mm, structures["PTV"], spot_centers_mm)
        plan_meta["spot_centers_mm"] = [[float(a), float(b), float(c)] for a, b, c in spot_centers_mm]
        sources = plan_meta["sources"]
    else:
        sources = load_legacy_sources(args.legacy_source_csv, int(args.histories))
        plan_meta = {
            "ptv_centroid_mm": None,
            "ap_radius_mm": None,
            "lateral_radius_mm": None,
            "spot_centers_mm": None,
        }

    spectrum_energies, spectrum_weights = load_spectrum(args.spectrum_csv)
    rendered = render_case_file(
        args,
        patient_bin=patient_bin,
        materials_file=materials_file,
        grid_shape=phantom_meta["grid_shape"],
        voxel_size_mm=phantom_meta["voxel_size_mm"],
        spectrum_energies=spectrum_energies,
        spectrum_weights=spectrum_weights,
        sources=sources,
    )
    write_text_with_retries(parameter_file, rendered)

    if not args.analyze_only:
        if args.skip_existing and has_nonempty_output(dose_csv):
            print(f"Reusing existing TOPAS output at {dose_csv}")
        else:
            print("=== RUNNING DETAILED HETEROGENEOUS HEAD-AND-NECK SFRT PLAN ===")
            run_topas_case(args, case_dir, parameter_file, dose_csv, log_file)

    dose_raw, _ = load_topas_grid(dose_csv)
    ptv_raw_metrics = compute_structure_metrics(
        dose_raw,
        structures["PTV"],
        prescription_gy=float(args.prescription_gy),
        voxel_volume_cc=voxel_volume_cc,
    )
    raw_d95 = float(ptv_raw_metrics["d95_gy"])
    if raw_d95 <= 0.0:
        raise RuntimeError("Detailed-phantom raw PTV D95 is non-positive; cannot normalize the plan.")
    physical_scale_factor = float(args.prescription_gy) / raw_d95
    physical_dose = dose_raw.astype(np.float32) * np.float32(physical_scale_factor)

    common_metric_config = {
        "PTV": {"prescription": float(args.prescription_gy), "vxs": [6.0, 10.0]},
        "GTV": {"prescription": float(args.prescription_gy), "vxs": [6.0, 10.0]},
        "SPINAL_CORD": {"prescription": None, "vxs": [5.0, 8.0]},
        "BRAINSTEM": {"prescription": None, "vxs": [5.0, 8.0]},
        "PAROTID_L": {"prescription": None, "vxs": [5.0, 10.0]},
        "PAROTID_R": {"prescription": None, "vxs": [5.0, 10.0]},
        "MANDIBLE": {"prescription": None, "vxs": [5.0, 10.0]},
    }
    new_only_metric_config = {
        "THYROID": {"prescription": None, "vxs": [5.0, 10.0]},
        "PARATHYROIDS": {"prescription": None, "vxs": [5.0, 10.0]},
        "BRAIN": {"prescription": None, "vxs": [5.0, 10.0]},
        "BLOOD_BRAIN_BARRIER": {"prescription": None, "vxs": [5.0, 10.0]},
    }

    new_metrics: Dict[str, Dict[str, float]] = {}
    for structure_name, config in common_metric_config.items():
        new_metrics[structure_name] = compute_structure_metrics(
            physical_dose,
            structures[structure_name],
            prescription_gy=config["prescription"],
            voxel_volume_cc=voxel_volume_cc,
            volume_thresholds_gy=config["vxs"],
        )

    extra_metrics: Dict[str, Dict[str, float]] = {}
    for structure_name, config in new_only_metric_config.items():
        extra_metrics[structure_name] = compute_structure_metrics(
            physical_dose,
            structures[structure_name],
            prescription_gy=config["prescription"],
            voxel_volume_cc=voxel_volume_cc,
            volume_thresholds_gy=config["vxs"],
        )

    old_metrics = load_metrics_csv(args.legacy_analysis_dir / "physical_plan_metrics.csv")
    old_dvh_axis, old_dvh_curves = load_dvh_csv(args.legacy_analysis_dir / "dvh_curves.csv")

    dose_axis = np.linspace(0.0, max(float(np.max(physical_dose)), float(np.max(old_dvh_axis))) * 1.05, 350)
    new_dvhs = {name: compute_dvh(physical_dose[structures[name]], dose_axis) for name in common_metric_config}

    dvh_rows: List[Dict[str, object]] = []
    for idx, dose_value in enumerate(dose_axis):
        row: Dict[str, object] = {"dose_gy": float(dose_value)}
        for structure_name in common_metric_config:
            row[f"{structure_name}_detailed_pct"] = float(new_dvhs[structure_name][idx])
        dvh_rows.append(row)

    comparison_rows = build_comparison_rows(old_metrics, new_metrics)
    save_csv(metrics_rows("detailed_physical", new_metrics), analysis_dir / "detailed_physical_plan_metrics.csv")
    save_csv(metrics_rows("detailed_new_only", extra_metrics), analysis_dir / "detailed_new_structure_metrics.csv")
    save_csv(dvh_rows, analysis_dir / "detailed_dvh_curves.csv")
    save_csv(comparison_rows, analysis_dir / "simple_vs_detailed_comparison_metrics.csv")

    plot_dose_slices(analysis_dir / "figure1_detailed_phantom_physical_dose.png", axes_mm, physical_dose, structures, dpi=int(args.dpi))
    plot_dvh_comparison(
        analysis_dir / "figure2_simple_vs_detailed_dvhs.png",
        dose_axis,
        new_dvhs,
        old_dvh_axis,
        old_dvh_curves,
        dpi=int(args.dpi),
    )
    plot_metric_comparison(
        analysis_dir / "figure3_simple_vs_detailed_metric_bars.png",
        old_metrics,
        new_metrics,
        dpi=int(args.dpi),
    )

    summary = {
        "legacy_source_csv": str(args.legacy_source_csv),
        "legacy_analysis_dir": str(args.legacy_analysis_dir),
        "prescription_gy": float(args.prescription_gy),
        "physical_scale_factor": float(physical_scale_factor),
        "plan": {
            "mode": args.plan_mode,
            "histories": int(args.histories),
            "num_sources": int(len(sources)),
            "source_names": [spec.name for spec in sources],
            "num_lattice_spots": int(len(plan_meta.get("spot_centers_mm", []) or [])),
            "spot_centers_mm": plan_meta.get("spot_centers_mm"),
            "ap_radius_mm": plan_meta.get("ap_radius_mm"),
            "lateral_radius_mm": plan_meta.get("lateral_radius_mm"),
        },
        "phantom": phantom_meta,
        "new_metrics": new_metrics,
        "extra_metrics": extra_metrics,
    }
    write_text_with_retries(analysis_dir / "phase14_detailed_headneck_summary.json", json.dumps(summary, indent=2))
    write_markdown_report(analysis_dir / "phase14_detailed_headneck_summary.md", summary, old_metrics, new_metrics, extra_metrics)

    print("\n=== PHASE 14 DETAILED HETEROGENEOUS PHANTOM COMPARISON COMPLETE ===")
    print(f"PTV detailed physical D95 normalized to: {new_metrics['PTV']['d95_gy']:.2f} Gy")
    print(f"Spinal cord detailed physical D2: {new_metrics['SPINAL_CORD']['d2_gy']:.2f} Gy")
    print(f"Right parotid detailed physical mean: {new_metrics['PAROTID_R']['mean_gy']:.2f} Gy")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
