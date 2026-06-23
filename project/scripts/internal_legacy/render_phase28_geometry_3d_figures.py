#!/usr/bin/env python3
"""Render clean 3D geometry figures for the Phase 28 Yang benchmark."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Iterable, Mapping, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from skimage.measure import marching_cubes


TUMOR_COLOR = "#c94f4f"
VERTEX_COLORS = ("#f4b000", "#00a6a6")
OAR_STYLES = {
    "BRAINSTEM": {"color": "#2e8b57", "alpha": 0.18, "label": "Brainstem"},
    "CHIASM": {"color": "#8b5fbf", "alpha": 0.30, "label": "Chiasm"},
    "OPTIC_NERVE_R": {"color": "#2b6cb0", "alpha": 0.32, "label": "Optic Nerve R"},
    "OPTIC_NERVE_L": {"color": "#4f87c2", "alpha": 0.32, "label": "Optic Nerve L"},
    "EYE_R": {"color": "#5dade2", "alpha": 0.10, "label": "Eye R"},
    "EYE_L": {"color": "#a8d8ff", "alpha": 0.10, "label": "Eye L"},
    "LENS_R": {"color": "#f9e27d", "alpha": 0.65, "label": "Lens R"},
    "LENS_L": {"color": "#ffe8a3", "alpha": 0.65, "label": "Lens L"},
}
NEARBY_OARS = tuple(OAR_STYLES.keys())
ANATOMY_STYLES = {
    "HEAD_SOFT": {"color": "#d7b59a", "alpha": 0.08, "label": "Soft Tissue Face"},
    "SKULL": {"color": "#f5f1e8", "alpha": 0.16, "label": "Skull"},
    "MAXILLA": {"color": "#ebe4d3", "alpha": 0.18, "label": "Maxilla"},
    "MANDIBLE": {"color": "#ddd5c2", "alpha": 0.20, "label": "Mandible"},
    "BRAIN": {"color": "#7fc3e2", "alpha": 0.12, "label": "Brain"},
}
TOPAS_TAG_SPECS = {
    0: {
        "name": "PH28_AIR",
        "base_material": "G4_AIR",
        "density_g_cm3": 0.0012,
        "label": "Air",
        "color": "#101418",
    },
    10: {
        "name": "PH28_SOFT_TISSUE",
        "base_material": "G4_TISSUE_SOFT_ICRP",
        "density_g_cm3": 1.04,
        "label": "Soft tissue / eyes / lenses",
        "color": "#c8a684",
    },
    11: {
        "name": "PH28_BRAIN",
        "base_material": "G4_TISSUE_SOFT_ICRP",
        "density_g_cm3": 1.04,
        "label": "Brain / optic apparatus",
        "color": "#62c6ff",
    },
    13: {
        "name": "PH28_BRAINSTEM",
        "base_material": "G4_TISSUE_SOFT_ICRP",
        "density_g_cm3": 1.04,
        "label": "Brainstem",
        "color": "#6adf8c",
    },
    20: {
        "name": "PH28_SKULL_BONE",
        "base_material": "G4_BONE_CORTICAL_ICRP",
        "density_g_cm3": 1.85,
        "label": "Skull bone",
        "color": "#f6f0e6",
    },
    21: {
        "name": "PH28_MAXILLOFACIAL_BONE",
        "base_material": "G4_BONE_COMPACT_ICRU",
        "density_g_cm3": 1.80,
        "label": "Maxilla / mandible",
        "color": "#ece1cf",
    },
    50: {
        "name": "PH28_TUMOUR",
        "base_material": "G4_TISSUE_SOFT_ICRP",
        "density_g_cm3": 1.05,
        "label": "Tumour",
        "color": "#ff5d5d",
    },
}
STRUCTURE_TO_TOPAS_TAG = {
    "BODY": 10,
    "HEAD_SOFT": 10,
    "EYE_R": 10,
    "EYE_L": 10,
    "LENS_R": 10,
    "LENS_L": 10,
    "BRAIN": 11,
    "CHIASM": 11,
    "OPTIC_NERVE_R": 11,
    "OPTIC_NERVE_L": 11,
    "BRAINSTEM": 13,
    "SKULL": 20,
    "MAXILLA": 21,
    "MANDIBLE": 21,
    "GTV": 50,
}


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--outdir",
        type=Path,
        default=root / "runs" / "phase28_yang2022_sinonasal_benchmark" / "figures",
    )
    parser.add_argument("--voxel-mm", type=float, default=2.0)
    parser.add_argument("--dpi", type=int, default=280)
    return parser.parse_args()


def make_axes(voxel_mm: float) -> Dict[str, np.ndarray]:
    return {
        "x": np.arange(-90.0, 90.0 + 0.5 * voxel_mm, voxel_mm, dtype=np.float32),
        "y": np.arange(-76.0, 76.0 + 0.5 * voxel_mm, voxel_mm, dtype=np.float32),
        "z": np.arange(0.0, 122.0 + 0.5 * voxel_mm, voxel_mm, dtype=np.float32),
    }


def write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        raise ValueError(f"No rows for {path}")
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def mesh(axes: Mapping[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    return np.meshgrid(axes["x"], axes["y"], axes["z"], indexing="ij")


def ellipsoid_mask(
    xg: np.ndarray,
    yg: np.ndarray,
    zg: np.ndarray,
    *,
    center: Tuple[float, float, float],
    radii: Tuple[float, float, float],
) -> np.ndarray:
    cx, cy, cz = center
    rx, ry, rz = radii
    return (((xg - cx) / rx) ** 2 + ((yg - cy) / ry) ** 2 + ((zg - cz) / rz) ** 2) <= 1.0


def sphere_mask(
    xg: np.ndarray,
    yg: np.ndarray,
    zg: np.ndarray,
    center: Tuple[float, float, float],
    radius_mm: float,
) -> np.ndarray:
    cx, cy, cz = center
    return ((xg - cx) ** 2 + (yg - cy) ** 2 + (zg - cz) ** 2) <= float(radius_mm) ** 2


def capsule_mask(
    xg: np.ndarray,
    yg: np.ndarray,
    zg: np.ndarray,
    p0: Tuple[float, float, float],
    p1: Tuple[float, float, float],
    radius_mm: float,
) -> np.ndarray:
    p0_arr = np.asarray(p0, dtype=np.float32)
    p1_arr = np.asarray(p1, dtype=np.float32)
    v = p1_arr - p0_arr
    vv = float(np.dot(v, v))
    px = xg - float(p0_arr[0])
    py = yg - float(p0_arr[1])
    pz = zg - float(p0_arr[2])
    t = np.clip((px * v[0] + py * v[1] + pz * v[2]) / max(vv, 1.0e-6), 0.0, 1.0)
    cx = float(p0_arr[0]) + t * v[0]
    cy = float(p0_arr[1]) + t * v[1]
    cz = float(p0_arr[2]) + t * v[2]
    return ((xg - cx) ** 2 + (yg - cy) ** 2 + (zg - cz) ** 2) <= float(radius_mm) ** 2


def build_structures(axes: Mapping[str, np.ndarray]) -> Dict[str, np.ndarray]:
    xg, yg, zg = mesh(axes)
    body = ellipsoid_mask(xg, yg, zg, center=(0.0, 0.0, 57.0), radii=(84.0, 70.0, 64.0)) & (zg >= 0.0)
    gtv = ellipsoid_mask(xg, yg, zg, center=(20.0, 0.0, 38.7), radii=(36.45, 30.0, 15.85)) & body

    cranium_outer = ellipsoid_mask(xg, yg, zg, center=(0.0, 2.0, 69.0), radii=(56.0, 48.0, 44.0)) & body
    cranium_inner = ellipsoid_mask(xg, yg, zg, center=(0.0, 2.0, 69.5), radii=(50.0, 42.0, 38.0)) & body
    midface_outer = ellipsoid_mask(xg, yg, zg, center=(8.0, 10.0, 34.0), radii=(48.0, 30.0, 24.0)) & body
    midface_inner = ellipsoid_mask(xg, yg, zg, center=(8.0, 8.0, 34.0), radii=(42.0, 24.0, 20.0)) & body
    jaw_outer = ellipsoid_mask(xg, yg, zg, center=(8.0, -3.0, 16.0), radii=(38.0, 22.0, 14.0)) & body
    jaw_inner = ellipsoid_mask(xg, yg, zg, center=(8.0, -2.0, 17.0), radii=(31.0, 17.0, 10.0)) & body
    nose = capsule_mask(xg, yg, zg, (8.0, 20.0, 28.0), (8.0, 38.0, 31.0), 5.5) & body
    cheek_r = ellipsoid_mask(xg, yg, zg, center=(32.0, 16.0, 25.0), radii=(10.0, 8.0, 10.0)) & body
    cheek_l = ellipsoid_mask(xg, yg, zg, center=(-16.0, 16.0, 25.0), radii=(10.0, 8.0, 10.0)) & body
    ear_r = ellipsoid_mask(xg, yg, zg, center=(57.0, 4.0, 33.0), radii=(5.0, 3.0, 10.0)) & body
    ear_l = ellipsoid_mask(xg, yg, zg, center=(-41.0, 4.0, 33.0), radii=(5.0, 3.0, 10.0)) & body

    head_soft = cranium_outer | midface_outer | jaw_outer | nose | cheek_r | cheek_l | ear_r | ear_l
    skull = (cranium_outer & ~cranium_inner) | (midface_outer & ~midface_inner)
    maxilla = (midface_outer & ~midface_inner) | ellipsoid_mask(xg, yg, zg, center=(8.0, 14.0, 22.0), radii=(30.0, 16.0, 8.0))
    mandible = jaw_outer & ~jaw_inner
    brain = ellipsoid_mask(xg, yg, zg, center=(0.0, 5.0, 71.0), radii=(48.0, 40.0, 34.0)) & cranium_inner

    structures = {
        "BODY": body,
        "HEAD_SOFT": head_soft,
        "SKULL": skull,
        "MAXILLA": maxilla,
        "MANDIBLE": mandible,
        "BRAIN": brain,
        "GTV": gtv,
        "BRAINSTEM": ellipsoid_mask(xg, yg, zg, center=(0.0, -3.0, 78.0), radii=(9.0, 14.0, 12.0)) & body,
        "CHIASM": ellipsoid_mask(xg, yg, zg, center=(2.0, 16.0, 55.0), radii=(8.0, 4.0, 4.0)) & body,
        "OPTIC_NERVE_R": capsule_mask(xg, yg, zg, (35.0, 19.0, 18.0), (6.0, 16.0, 53.0), 2.2) & body,
        "OPTIC_NERVE_L": capsule_mask(xg, yg, zg, (-30.0, 19.0, 18.0), (-2.0, 16.0, 53.0), 2.2) & body,
        "EYE_R": ellipsoid_mask(xg, yg, zg, center=(38.0, 20.0, 16.0), radii=(12.0, 10.0, 8.0)) & body,
        "EYE_L": ellipsoid_mask(xg, yg, zg, center=(-32.0, 20.0, 16.0), radii=(12.0, 10.0, 8.0)) & body,
        "LENS_R": ellipsoid_mask(xg, yg, zg, center=(38.0, 20.0, 7.5), radii=(3.0, 3.0, 2.0)) & body,
        "LENS_L": ellipsoid_mask(xg, yg, zg, center=(-32.0, 20.0, 7.5), radii=(3.0, 3.0, 2.0)) & body,
    }
    return structures


def build_phase28_material_tag_grid(structures: Mapping[str, np.ndarray]) -> np.ndarray:
    shape = np.asarray(structures["BODY"], dtype=bool).shape
    grid = np.zeros(shape, dtype=np.int16)
    for structure_name, tag in STRUCTURE_TO_TOPAS_TAG.items():
        if structure_name not in structures:
            continue
        grid[np.asarray(structures[structure_name], dtype=bool)] = int(tag)
    return grid


def build_density_from_tags(tag_grid: np.ndarray) -> np.ndarray:
    density = np.zeros(tag_grid.shape, dtype=np.float32)
    for tag, spec in TOPAS_TAG_SPECS.items():
        density[tag_grid == int(tag)] = float(spec["density_g_cm3"])
    return density


def render_topas_materials_include(used_tags: Sequence[int]) -> str:
    lines = ["# Phase 28 Yang benchmark TOPAS materials", ""]
    for tag in used_tags:
        spec = TOPAS_TAG_SPECS[int(tag)]
        lines.extend(
            [
                f"# Tag {int(tag)}: {spec['label']}",
                f's:Ma/{spec["name"]}/BaseMaterial = "{spec["base_material"]}"',
                f'd:Ma/{spec["name"]}/Density = {float(spec["density_g_cm3"]):.6f} g/cm3',
                "",
            ]
        )
    return "\n".join(lines).strip() + "\n"


def vertex_centers() -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
    center_x = 20.0
    half_spacing_mm = 30.4 / 2.0
    return ((center_x - half_spacing_mm, 0.0, 38.7), (center_x + half_spacing_mm, 0.0, 38.7))


def voxel_spacing_mm(axes: Mapping[str, np.ndarray]) -> Tuple[float, float, float]:
    return tuple(float(axis[1] - axis[0]) for axis in (axes["x"], axes["y"], axes["z"]))


def mask_to_mesh(mask: np.ndarray, axes: Mapping[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    spacing = voxel_spacing_mm(axes)
    verts, faces, _, _ = marching_cubes(mask.astype(np.float32), level=0.5, spacing=spacing)
    origin = np.array([float(axes["x"][0]), float(axes["y"][0]), float(axes["z"][0])], dtype=np.float32)
    return verts + origin, faces


def add_mask_surface(
    ax,
    mask: np.ndarray,
    axes: Mapping[str, np.ndarray],
    *,
    color: str,
    alpha: float,
    edge_alpha: float = 0.08,
    linewidth: float = 0.16,
) -> None:
    if not np.any(mask):
        return
    verts, faces = mask_to_mesh(mask, axes)
    mesh = Poly3DCollection(verts[faces], alpha=alpha)
    rgba = list(matplotlib.colors.to_rgba(color))
    edge_rgba = list(matplotlib.colors.to_rgba("#111111"))
    rgba[3] = alpha
    edge_rgba[3] = edge_alpha
    mesh.set_facecolor(tuple(rgba))
    mesh.set_edgecolor(tuple(edge_rgba))
    mesh.set_linewidth(linewidth)
    ax.add_collection3d(mesh)


def add_vertex_sphere(
    ax,
    center: Sequence[float],
    radius_mm: float,
    *,
    color: str,
    alpha: float = 0.95,
) -> None:
    u = np.linspace(0.0, 2.0 * np.pi, 30)
    v = np.linspace(0.0, np.pi, 22)
    xs = center[0] + radius_mm * np.outer(np.cos(u), np.sin(v))
    ys = center[1] + radius_mm * np.outer(np.sin(u), np.sin(v))
    zs = center[2] + radius_mm * np.outer(np.ones_like(u), np.cos(v))
    ax.plot_surface(xs, ys, zs, color=color, alpha=alpha, linewidth=0.0, shade=True)


def bounds_from_masks(
    axes: Mapping[str, np.ndarray],
    masks: Iterable[np.ndarray],
    *,
    margin_mm: float,
) -> Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]:
    mins = []
    maxs = []
    for mask in masks:
        idx = np.argwhere(mask)
        if idx.size == 0:
            continue
        mins.append(
            (
                float(axes["x"][idx[:, 0].min()]),
                float(axes["y"][idx[:, 1].min()]),
                float(axes["z"][idx[:, 2].min()]),
            )
        )
        maxs.append(
            (
                float(axes["x"][idx[:, 0].max()]),
                float(axes["y"][idx[:, 1].max()]),
                float(axes["z"][idx[:, 2].max()]),
            )
        )
    min_xyz = np.min(np.asarray(mins, dtype=np.float32), axis=0) - float(margin_mm)
    max_xyz = np.max(np.asarray(maxs, dtype=np.float32), axis=0) + float(margin_mm)
    return (
        (float(min_xyz[0]), float(max_xyz[0])),
        (float(min_xyz[1]), float(max_xyz[1])),
        (float(min_xyz[2]), float(max_xyz[2])),
    )


def apply_bounds(ax, bounds: Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]) -> None:
    (xmin, xmax), (ymin, ymax), (zmin, zmax) = bounds
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_zlim(zmin, zmax)
    ax.set_box_aspect((xmax - xmin, ymax - ymin, zmax - zmin))


def style_axes(ax, *, minimal: bool) -> None:
    ax.grid(False)
    if minimal:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_zlabel("")
    else:
        ax.set_xlabel("x (mm)", labelpad=9)
        ax.set_ylabel("y (mm)", labelpad=9)
        ax.set_zlabel("z (mm)", labelpad=6)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_facecolor((0.98, 0.98, 0.98, 1.0))
        axis.pane.set_edgecolor((0.86, 0.86, 0.86, 1.0))


def style_dark_axes(ax) -> None:
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_zlabel("")
    ax.set_facecolor("#0f1117")
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_facecolor((0.07, 0.08, 0.11, 1.0))
        axis.pane.set_edgecolor((0.78, 0.82, 0.90, 0.06))
        try:
            axis.line.set_color((0.78, 0.82, 0.90, 0.16))
        except Exception:
            pass


def style_publication_axes(ax) -> None:
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_zlabel("")
    ax.set_facecolor("white")
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_facecolor((1.0, 1.0, 1.0, 1.0))
        axis.pane.set_edgecolor((0.75, 0.75, 0.75, 0.35))
        try:
            axis.line.set_color((0.55, 0.55, 0.55, 0.45))
        except Exception:
            pass


def build_cinematic_display_structures(
    axes: Mapping[str, np.ndarray],
    structures: Mapping[str, np.ndarray],
) -> Dict[str, np.ndarray]:
    xg, yg, zg = mesh(axes)
    shell_window = (
        (xg > -4.0)
        & (yg > -14.0)
        & (zg > 4.0)
        & (zg < 92.0)
        & ((xg + 0.55 * yg) > 2.0)
    )
    jaw_window = (
        (xg > -2.0)
        & (yg > -18.0)
        & (zg > 2.0)
        & (zg < 42.0)
        & ((xg + 0.35 * yg) > 8.0)
    )
    brain_window = (
        (xg > 6.0)
        & (yg > -8.0)
        & (zg > 40.0)
        & (zg < 90.0)
        & ((xg + 0.30 * yg) > 14.0)
    )
    display = dict(structures)
    display["HEAD_SOFT_CUT"] = structures["HEAD_SOFT"] & ~shell_window
    display["SKULL_CUT"] = structures["SKULL"] & ~shell_window
    display["MAXILLA_CUT"] = structures["MAXILLA"] & ~(shell_window | jaw_window)
    display["MANDIBLE_CUT"] = structures["MANDIBLE"] & ~jaw_window
    display["BRAIN_CUT"] = structures["BRAIN"] & ~brain_window
    return display


def plot_vertices_figure(
    axes: Mapping[str, np.ndarray],
    structures: Mapping[str, np.ndarray],
    out_file: Path,
    *,
    dpi: int,
) -> None:
    bounds = bounds_from_masks(axes, (structures["GTV"],), margin_mm=10.0)
    fig = plt.figure(figsize=(8.5, 7.2))
    ax = fig.add_subplot(111, projection="3d")
    add_mask_surface(ax, structures["GTV"], axes, color=TUMOR_COLOR, alpha=0.33, edge_alpha=0.10, linewidth=0.20)
    for idx, center in enumerate(vertex_centers(), start=1):
        add_vertex_sphere(ax, center, 5.0, color=VERTEX_COLORS[idx - 1], alpha=0.96)
        ax.text(center[0], center[1], center[2] + 7.0, f"V{idx}", fontsize=10, ha="center", va="bottom")
    apply_bounds(ax, bounds)
    style_axes(ax, minimal=False)
    ax.view_init(elev=21, azim=-58)
    ax.set_title("Yang 2022 Benchmark Tumour with Lattice Vertices", pad=18)
    legend_handles = [
        Patch(facecolor=TUMOR_COLOR, edgecolor="none", alpha=0.33, label="GTV"),
        Patch(facecolor=VERTEX_COLORS[0], edgecolor="none", alpha=0.96, label="Vertex 1"),
        Patch(facecolor=VERTEX_COLORS[1], edgecolor="none", alpha=0.96, label="Vertex 2"),
    ]
    ax.legend(handles=legend_handles, loc="upper right", frameon=False)
    fig.tight_layout()
    fig.savefig(out_file, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_oar_figure(
    axes: Mapping[str, np.ndarray],
    structures: Mapping[str, np.ndarray],
    out_file: Path,
    *,
    dpi: int,
) -> None:
    masks = [structures["GTV"], *(structures[name] for name in NEARBY_OARS)]
    bounds = bounds_from_masks(axes, masks, margin_mm=12.0)
    fig = plt.figure(figsize=(10.3, 7.6))
    ax = fig.add_subplot(111, projection="3d")
    add_mask_surface(ax, structures["GTV"], axes, color=TUMOR_COLOR, alpha=0.18, edge_alpha=0.05, linewidth=0.14)
    for name in NEARBY_OARS:
        style = OAR_STYLES[name]
        add_mask_surface(ax, structures[name], axes, color=style["color"], alpha=style["alpha"], edge_alpha=0.06, linewidth=0.14)
    for idx, center in enumerate(vertex_centers()):
        add_vertex_sphere(ax, center, 5.0, color=VERTEX_COLORS[idx], alpha=0.92)
    apply_bounds(ax, bounds)
    style_axes(ax, minimal=False)
    ax.view_init(elev=22, azim=-60)
    ax.set_title("Tumour, Lattice Vertices, and Nearby Skull-Base OARs", pad=18)
    legend_handles = [Patch(facecolor=TUMOR_COLOR, edgecolor="none", alpha=0.18, label="GTV")]
    legend_handles.extend(
        Patch(facecolor=OAR_STYLES[name]["color"], edgecolor="none", alpha=max(0.35, OAR_STYLES[name]["alpha"]), label=OAR_STYLES[name]["label"])
        for name in NEARBY_OARS
    )
    legend_handles.extend(
        [
            Patch(facecolor=VERTEX_COLORS[0], edgecolor="none", alpha=0.92, label="Vertex 1"),
            Patch(facecolor=VERTEX_COLORS[1], edgecolor="none", alpha=0.92, label="Vertex 2"),
        ]
    )
    ax.legend(handles=legend_handles, loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    fig.tight_layout()
    fig.savefig(out_file, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def draw_manuscript_scene(ax, axes: Mapping[str, np.ndarray], structures: Mapping[str, np.ndarray], bounds) -> None:
    add_mask_surface(ax, structures["GTV"], axes, color=TUMOR_COLOR, alpha=0.14, edge_alpha=0.03, linewidth=0.10)
    for name in ("BRAINSTEM", "CHIASM", "OPTIC_NERVE_R", "OPTIC_NERVE_L"):
        style = OAR_STYLES[name]
        add_mask_surface(ax, structures[name], axes, color=style["color"], alpha=style["alpha"], edge_alpha=0.05, linewidth=0.12)
    for name in ("EYE_R", "EYE_L"):
        style = OAR_STYLES[name]
        add_mask_surface(ax, structures[name], axes, color=style["color"], alpha=style["alpha"], edge_alpha=0.03, linewidth=0.08)
    for name in ("LENS_R", "LENS_L"):
        style = OAR_STYLES[name]
        add_mask_surface(ax, structures[name], axes, color=style["color"], alpha=style["alpha"], edge_alpha=0.08, linewidth=0.12)
    for idx, center in enumerate(vertex_centers()):
        add_vertex_sphere(ax, center, 5.0, color=VERTEX_COLORS[idx], alpha=0.97)
    apply_bounds(ax, bounds)
    style_axes(ax, minimal=True)


def draw_lab_phantom_scene(ax, axes: Mapping[str, np.ndarray], structures: Mapping[str, np.ndarray], bounds) -> None:
    for name in ("HEAD_SOFT", "SKULL", "MAXILLA", "MANDIBLE", "BRAIN"):
        style = ANATOMY_STYLES[name]
        add_mask_surface(ax, structures[name], axes, color=style["color"], alpha=style["alpha"], edge_alpha=0.03, linewidth=0.10)
    add_mask_surface(ax, structures["GTV"], axes, color=TUMOR_COLOR, alpha=0.20, edge_alpha=0.04, linewidth=0.12)
    for name in ("BRAINSTEM", "CHIASM", "OPTIC_NERVE_R", "OPTIC_NERVE_L"):
        style = OAR_STYLES[name]
        add_mask_surface(ax, structures[name], axes, color=style["color"], alpha=max(style["alpha"], 0.22), edge_alpha=0.04, linewidth=0.10)
    for name in ("EYE_R", "EYE_L"):
        style = OAR_STYLES[name]
        add_mask_surface(ax, structures[name], axes, color=style["color"], alpha=0.16, edge_alpha=0.02, linewidth=0.08)
    for name in ("LENS_R", "LENS_L"):
        style = OAR_STYLES[name]
        add_mask_surface(ax, structures[name], axes, color=style["color"], alpha=0.72, edge_alpha=0.04, linewidth=0.10)
    for idx, center in enumerate(vertex_centers()):
        add_vertex_sphere(ax, center, 5.0, color=VERTEX_COLORS[idx], alpha=0.97)
    apply_bounds(ax, bounds)
    style_axes(ax, minimal=True)


def plot_manuscript_multi_view(
    axes: Mapping[str, np.ndarray],
    structures: Mapping[str, np.ndarray],
    out_file: Path,
    *,
    dpi: int,
) -> None:
    masks = [structures["GTV"], *(structures[name] for name in NEARBY_OARS)]
    bounds = bounds_from_masks(axes, masks, margin_mm=10.0)
    fig = plt.figure(figsize=(15.4, 5.0))
    views = (
        ("A  Oblique", 22, -58),
        ("B  Coronal", 4, -90),
        ("C  Superior", 88, -90),
    )
    for idx, (title, elev, azim) in enumerate(views, start=1):
        ax = fig.add_subplot(1, 3, idx, projection="3d")
        draw_manuscript_scene(ax, axes, structures, bounds)
        ax.view_init(elev=elev, azim=azim)
        ax.set_title(title, fontsize=12, pad=10)
    legend_handles = [
        Patch(facecolor=TUMOR_COLOR, edgecolor="none", alpha=0.18, label="GTV"),
        Patch(facecolor=VERTEX_COLORS[0], edgecolor="none", alpha=0.96, label="Vertices"),
        Patch(facecolor=OAR_STYLES["BRAINSTEM"]["color"], edgecolor="none", alpha=0.25, label="Brainstem"),
        Patch(facecolor=OAR_STYLES["CHIASM"]["color"], edgecolor="none", alpha=0.30, label="Chiasm"),
        Patch(facecolor=OAR_STYLES["OPTIC_NERVE_R"]["color"], edgecolor="none", alpha=0.32, label="Optic Nerves"),
        Patch(facecolor=OAR_STYLES["EYE_R"]["color"], edgecolor="none", alpha=0.18, label="Eyes"),
        Patch(facecolor=OAR_STYLES["LENS_R"]["color"], edgecolor="none", alpha=0.65, label="Lenses"),
    ]
    fig.legend(handles=legend_handles, ncol=7, loc="lower center", bbox_to_anchor=(0.5, -0.01), frameon=False)
    fig.suptitle("Yang 2022 Benchmark: Transparent 3D Geometry Views", y=0.97, fontsize=18)
    fig.tight_layout(rect=(0.0, 0.07, 1.0, 0.95))
    fig.savefig(out_file, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_labstyle_phantom_oblique(
    axes: Mapping[str, np.ndarray],
    structures: Mapping[str, np.ndarray],
    out_file: Path,
    *,
    dpi: int,
) -> None:
    masks = [
        structures["HEAD_SOFT"],
        structures["SKULL"],
        structures["BRAIN"],
        structures["GTV"],
        *(structures[name] for name in NEARBY_OARS),
    ]
    bounds = bounds_from_masks(axes, masks, margin_mm=10.0)
    fig = plt.figure(figsize=(9.4, 8.0))
    fig.patch.set_facecolor("#f6f4ef")
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor("#f6f4ef")
    draw_lab_phantom_scene(ax, axes, structures, bounds)
    ax.view_init(elev=20, azim=-58)
    ax.set_title("Yang 2022 Benchmark Phantom-Style Anatomy Render", pad=16)
    legend_handles = [
        Patch(facecolor=ANATOMY_STYLES["HEAD_SOFT"]["color"], edgecolor="none", alpha=0.25, label="Soft Tissue Face"),
        Patch(facecolor=ANATOMY_STYLES["SKULL"]["color"], edgecolor="none", alpha=0.30, label="Skull"),
        Patch(facecolor=ANATOMY_STYLES["BRAIN"]["color"], edgecolor="none", alpha=0.22, label="Brain"),
        Patch(facecolor=OAR_STYLES["BRAINSTEM"]["color"], edgecolor="none", alpha=0.28, label="Brainstem"),
        Patch(facecolor=OAR_STYLES["CHIASM"]["color"], edgecolor="none", alpha=0.28, label="Chiasm"),
        Patch(facecolor=OAR_STYLES["OPTIC_NERVE_R"]["color"], edgecolor="none", alpha=0.28, label="Optic Nerves"),
        Patch(facecolor=OAR_STYLES["EYE_R"]["color"], edgecolor="none", alpha=0.20, label="Eyes"),
        Patch(facecolor=TUMOR_COLOR, edgecolor="none", alpha=0.24, label="Tumour"),
        Patch(facecolor=VERTEX_COLORS[0], edgecolor="none", alpha=0.96, label="Vertices"),
    ]
    ax.legend(handles=legend_handles, loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    fig.tight_layout()
    fig.savefig(out_file, dpi=dpi, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def plot_labstyle_phantom_multi_view(
    axes: Mapping[str, np.ndarray],
    structures: Mapping[str, np.ndarray],
    out_file: Path,
    *,
    dpi: int,
) -> None:
    masks = [
        structures["HEAD_SOFT"],
        structures["SKULL"],
        structures["BRAIN"],
        structures["GTV"],
        *(structures[name] for name in NEARBY_OARS),
    ]
    bounds = bounds_from_masks(axes, masks, margin_mm=10.0)
    fig = plt.figure(figsize=(15.8, 5.4))
    fig.patch.set_facecolor("#f6f4ef")
    views = (
        ("A  Lab Oblique", 20, -58),
        ("B  Coronal", 3, -90),
        ("C  Superior", 88, -90),
    )
    for idx, (title, elev, azim) in enumerate(views, start=1):
        ax = fig.add_subplot(1, 3, idx, projection="3d")
        ax.set_facecolor("#f6f4ef")
        draw_lab_phantom_scene(ax, axes, structures, bounds)
        ax.view_init(elev=elev, azim=azim)
        ax.set_title(title, fontsize=12, pad=8)
    legend_handles = [
        Patch(facecolor=ANATOMY_STYLES["HEAD_SOFT"]["color"], edgecolor="none", alpha=0.25, label="Soft Tissue Face"),
        Patch(facecolor=ANATOMY_STYLES["SKULL"]["color"], edgecolor="none", alpha=0.30, label="Skull"),
        Patch(facecolor=ANATOMY_STYLES["MAXILLA"]["color"], edgecolor="none", alpha=0.28, label="Maxilla / Mandible"),
        Patch(facecolor=ANATOMY_STYLES["BRAIN"]["color"], edgecolor="none", alpha=0.22, label="Brain"),
        Patch(facecolor=TUMOR_COLOR, edgecolor="none", alpha=0.24, label="Tumour"),
        Patch(facecolor=VERTEX_COLORS[0], edgecolor="none", alpha=0.96, label="Vertices"),
        Patch(facecolor=OAR_STYLES["BRAINSTEM"]["color"], edgecolor="none", alpha=0.28, label="Brainstem"),
        Patch(facecolor=OAR_STYLES["CHIASM"]["color"], edgecolor="none", alpha=0.28, label="Chiasm"),
        Patch(facecolor=OAR_STYLES["OPTIC_NERVE_R"]["color"], edgecolor="none", alpha=0.28, label="Optic Nerves"),
        Patch(facecolor=OAR_STYLES["EYE_R"]["color"], edgecolor="none", alpha=0.20, label="Eyes / Lenses"),
    ]
    fig.legend(handles=legend_handles, ncol=5, loc="lower center", bbox_to_anchor=(0.5, -0.01), frameon=False)
    fig.suptitle("Yang 2022 Benchmark: Anatomy-Rich Transparent Phantom Views", y=0.98, fontsize=18)
    fig.tight_layout(rect=(0.0, 0.08, 1.0, 0.94))
    fig.savefig(out_file, dpi=dpi, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def plot_cinematic_cutaway_phantom(
    axes: Mapping[str, np.ndarray],
    structures: Mapping[str, np.ndarray],
    out_file: Path,
    *,
    dpi: int,
) -> None:
    display = build_cinematic_display_structures(axes, structures)
    masks = [
        display["HEAD_SOFT_CUT"],
        display["SKULL_CUT"],
        display["BRAIN_CUT"],
        structures["GTV"],
        *(structures[name] for name in NEARBY_OARS),
    ]
    bounds = bounds_from_masks(axes, masks, margin_mm=8.0)
    fig = plt.figure(figsize=(9.8, 8.2))
    fig.patch.set_facecolor("#0b0e14")
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor("#0b0e14")

    add_mask_surface(
        ax,
        display["HEAD_SOFT_CUT"],
        axes,
        color="#c8a684",
        alpha=0.05,
        edge_alpha=0.025,
        linewidth=0.08,
    )
    add_mask_surface(
        ax,
        display["SKULL_CUT"],
        axes,
        color="#f6f0e6",
        alpha=0.14,
        edge_alpha=0.11,
        linewidth=0.16,
    )
    add_mask_surface(
        ax,
        display["MAXILLA_CUT"],
        axes,
        color="#ece1cf",
        alpha=0.15,
        edge_alpha=0.09,
        linewidth=0.14,
    )
    add_mask_surface(
        ax,
        display["MANDIBLE_CUT"],
        axes,
        color="#ddd0b8",
        alpha=0.16,
        edge_alpha=0.10,
        linewidth=0.14,
    )
    add_mask_surface(
        ax,
        display["BRAIN_CUT"],
        axes,
        color="#62c6ff",
        alpha=0.18,
        edge_alpha=0.03,
        linewidth=0.08,
    )
    add_mask_surface(
        ax,
        structures["BRAINSTEM"],
        axes,
        color="#6adf8c",
        alpha=0.46,
        edge_alpha=0.06,
        linewidth=0.10,
    )
    add_mask_surface(
        ax,
        structures["CHIASM"],
        axes,
        color="#c593ff",
        alpha=0.60,
        edge_alpha=0.06,
        linewidth=0.10,
    )
    add_mask_surface(
        ax,
        structures["OPTIC_NERVE_R"],
        axes,
        color="#5fa8ff",
        alpha=0.48,
        edge_alpha=0.05,
        linewidth=0.09,
    )
    add_mask_surface(
        ax,
        structures["OPTIC_NERVE_L"],
        axes,
        color="#88bcff",
        alpha=0.44,
        edge_alpha=0.05,
        linewidth=0.09,
    )
    add_mask_surface(
        ax,
        structures["EYE_R"],
        axes,
        color="#9fd8ff",
        alpha=0.14,
        edge_alpha=0.02,
        linewidth=0.07,
    )
    add_mask_surface(
        ax,
        structures["EYE_L"],
        axes,
        color="#d1edff",
        alpha=0.12,
        edge_alpha=0.02,
        linewidth=0.07,
    )
    add_mask_surface(
        ax,
        structures["LENS_R"],
        axes,
        color="#ffe98a",
        alpha=0.92,
        edge_alpha=0.10,
        linewidth=0.10,
    )
    add_mask_surface(
        ax,
        structures["LENS_L"],
        axes,
        color="#fff2b5",
        alpha=0.90,
        edge_alpha=0.10,
        linewidth=0.10,
    )
    add_mask_surface(
        ax,
        structures["GTV"],
        axes,
        color="#ff5d5d",
        alpha=0.34,
        edge_alpha=0.08,
        linewidth=0.14,
    )
    for idx, center in enumerate(vertex_centers()):
        add_vertex_sphere(ax, center, 5.0, color=VERTEX_COLORS[idx], alpha=0.99)

    apply_bounds(ax, bounds)
    style_dark_axes(ax)
    ax.view_init(elev=18, azim=-61)

    label_specs = [
        ("Tumour", (34.0, 11.0, 30.0), "#ff9a9a"),
        ("V1", (5.0, 3.0, 42.0), "#ffd24a"),
        ("V2", (34.0, 13.0, 35.0), "#42d7d7"),
        ("Brainstem", (2.0, -5.0, 81.0), "#88f0aa"),
        ("Chiasm", (5.0, 18.0, 58.0), "#dfb8ff"),
    ]
    for text, coords, color in label_specs:
        ax.text(*coords, text, color=color, fontsize=11, fontweight="bold")

    ax.set_title("Yang 2022 Benchmark: Cinematic Cutaway Phantom", color="white", pad=16, fontsize=20)
    legend_handles = [
        Patch(facecolor="#f6f0e6", edgecolor="none", alpha=0.28, label="Cutaway Skull"),
        Patch(facecolor="#62c6ff", edgecolor="none", alpha=0.24, label="Brain"),
        Patch(facecolor="#6adf8c", edgecolor="none", alpha=0.42, label="Brainstem"),
        Patch(facecolor="#c593ff", edgecolor="none", alpha=0.58, label="Chiasm"),
        Patch(facecolor="#5fa8ff", edgecolor="none", alpha=0.42, label="Optic Pathway"),
        Patch(facecolor="#ff5d5d", edgecolor="none", alpha=0.34, label="Tumour"),
        Patch(facecolor=VERTEX_COLORS[0], edgecolor="none", alpha=0.98, label="Lattice Vertices"),
    ]
    legend = ax.legend(
        handles=legend_handles,
        loc="lower right",
        bbox_to_anchor=(1.03, 0.02),
        frameon=False,
        fontsize=11,
    )
    for text in legend.get_texts():
        text.set_color("white")

    fig.tight_layout()
    fig.savefig(out_file, dpi=dpi, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def plot_topas_density_tagged_cutaway(
    axes: Mapping[str, np.ndarray],
    structures: Mapping[str, np.ndarray],
    out_file: Path,
    *,
    dpi: int,
) -> None:
    display = build_cinematic_display_structures(axes, structures)
    masks = [
        display["HEAD_SOFT_CUT"],
        display["SKULL_CUT"],
        display["BRAIN_CUT"],
        structures["GTV"],
        *(structures[name] for name in NEARBY_OARS),
    ]
    bounds = bounds_from_masks(axes, masks, margin_mm=8.0)
    fig = plt.figure(figsize=(11.6, 8.4))
    fig.patch.set_facecolor("#0b0e14")
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor("#0b0e14")

    add_mask_surface(ax, display["HEAD_SOFT_CUT"], axes, color="#c8a684", alpha=0.05, edge_alpha=0.02, linewidth=0.08)
    add_mask_surface(ax, display["SKULL_CUT"], axes, color="#f6f0e6", alpha=0.15, edge_alpha=0.10, linewidth=0.16)
    add_mask_surface(ax, display["MAXILLA_CUT"], axes, color="#ece1cf", alpha=0.16, edge_alpha=0.08, linewidth=0.14)
    add_mask_surface(ax, display["MANDIBLE_CUT"], axes, color="#ddd0b8", alpha=0.17, edge_alpha=0.09, linewidth=0.14)
    add_mask_surface(ax, display["BRAIN_CUT"], axes, color="#62c6ff", alpha=0.20, edge_alpha=0.03, linewidth=0.08)
    add_mask_surface(ax, structures["BRAINSTEM"], axes, color="#6adf8c", alpha=0.48, edge_alpha=0.06, linewidth=0.10)
    add_mask_surface(ax, structures["CHIASM"], axes, color="#c593ff", alpha=0.62, edge_alpha=0.06, linewidth=0.10)
    add_mask_surface(ax, structures["OPTIC_NERVE_R"], axes, color="#5fa8ff", alpha=0.48, edge_alpha=0.05, linewidth=0.09)
    add_mask_surface(ax, structures["OPTIC_NERVE_L"], axes, color="#88bcff", alpha=0.45, edge_alpha=0.05, linewidth=0.09)
    add_mask_surface(ax, structures["EYE_R"], axes, color="#9fd8ff", alpha=0.16, edge_alpha=0.02, linewidth=0.07)
    add_mask_surface(ax, structures["EYE_L"], axes, color="#d1edff", alpha=0.14, edge_alpha=0.02, linewidth=0.07)
    add_mask_surface(ax, structures["LENS_R"], axes, color="#ffe98a", alpha=0.92, edge_alpha=0.10, linewidth=0.10)
    add_mask_surface(ax, structures["LENS_L"], axes, color="#fff2b5", alpha=0.90, edge_alpha=0.10, linewidth=0.10)
    add_mask_surface(ax, structures["GTV"], axes, color="#ff5d5d", alpha=0.36, edge_alpha=0.08, linewidth=0.14)
    for idx, center in enumerate(vertex_centers()):
        add_vertex_sphere(ax, center, 5.0, color=VERTEX_COLORS[idx], alpha=0.99)

    apply_bounds(ax, bounds)
    style_dark_axes(ax)
    ax.view_init(elev=18, azim=-61)
    ax.set_title("Yang 2022 Benchmark: TOPAS Density-Tagged Phantom", color="white", pad=16, fontsize=20)

    label_specs = [
        ("Tag 50", (35.0, 13.0, 30.0), "#ff9a9a"),
        ("Tag 20", (-18.0, -5.0, 78.0), "#f6f0e6"),
        ("Tag 11", (-8.0, -3.0, 66.0), "#7edaff"),
        ("Tag 13", (4.0, -5.0, 81.0), "#88f0aa"),
    ]
    for text, coords, color in label_specs:
        ax.text(*coords, text, color=color, fontsize=11, fontweight="bold")

    used_tags = [10, 11, 13, 20, 21, 50]
    legend_handles = [
        Patch(
            facecolor=TOPAS_TAG_SPECS[tag]["color"],
            edgecolor="none",
            alpha=(0.28 if tag in {20, 21} else 0.45 if tag in {11, 13, 50} else 0.22),
            label=(
                f"Tag {tag} | {TOPAS_TAG_SPECS[tag]['name']} | "
                f"{TOPAS_TAG_SPECS[tag]['density_g_cm3']:.2f} g/cm3"
            ),
        )
        for tag in used_tags
    ]
    legend = ax.legend(
        handles=legend_handles,
        loc="lower right",
        bbox_to_anchor=(1.18, 0.02),
        frameon=False,
        fontsize=10,
    )
    for text in legend.get_texts():
        text.set_color("white")

    fig.text(
        0.67,
        0.12,
        "Vertices are planning hotspots, not material tags.",
        color="#c6cbd4",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(out_file, dpi=dpi, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def plot_topas_density_publication_sidepane(
    axes: Mapping[str, np.ndarray],
    structures: Mapping[str, np.ndarray],
    out_file: Path,
    *,
    dpi: int,
) -> None:
    display = build_cinematic_display_structures(axes, structures)
    masks = [
        display["HEAD_SOFT_CUT"],
        display["SKULL_CUT"],
        display["BRAIN_CUT"],
        structures["GTV"],
        *(structures[name] for name in NEARBY_OARS),
    ]
    bounds = bounds_from_masks(axes, masks, margin_mm=8.0)
    fig = plt.figure(figsize=(13.4, 8.0), constrained_layout=False)
    grid = fig.add_gridspec(1, 2, width_ratios=(3.6, 1.35), wspace=0.02)
    ax = fig.add_subplot(grid[0, 0], projection="3d")
    panel = fig.add_subplot(grid[0, 1])
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    panel.set_facecolor("#fbfbfb")
    panel.axis("off")

    add_mask_surface(ax, display["HEAD_SOFT_CUT"], axes, color="#d5b89d", alpha=0.05, edge_alpha=0.015, linewidth=0.06)
    add_mask_surface(ax, display["SKULL_CUT"], axes, color="#f0ebe2", alpha=0.17, edge_alpha=0.08, linewidth=0.12)
    add_mask_surface(ax, display["MAXILLA_CUT"], axes, color="#e9e1d4", alpha=0.19, edge_alpha=0.07, linewidth=0.10)
    add_mask_surface(ax, display["MANDIBLE_CUT"], axes, color="#ddd2bf", alpha=0.18, edge_alpha=0.07, linewidth=0.10)
    add_mask_surface(ax, display["BRAIN_CUT"], axes, color="#7ecaf2", alpha=0.18, edge_alpha=0.03, linewidth=0.07)
    add_mask_surface(ax, structures["BRAINSTEM"], axes, color="#64cb84", alpha=0.40, edge_alpha=0.05, linewidth=0.08)
    add_mask_surface(ax, structures["CHIASM"], axes, color="#b88ae8", alpha=0.48, edge_alpha=0.05, linewidth=0.08)
    add_mask_surface(ax, structures["OPTIC_NERVE_R"], axes, color="#5c96e6", alpha=0.38, edge_alpha=0.05, linewidth=0.07)
    add_mask_surface(ax, structures["OPTIC_NERVE_L"], axes, color="#90baf3", alpha=0.34, edge_alpha=0.05, linewidth=0.07)
    add_mask_surface(ax, structures["EYE_R"], axes, color="#b8dff9", alpha=0.13, edge_alpha=0.02, linewidth=0.05)
    add_mask_surface(ax, structures["EYE_L"], axes, color="#dcedfb", alpha=0.11, edge_alpha=0.02, linewidth=0.05)
    add_mask_surface(ax, structures["LENS_R"], axes, color="#ffe89a", alpha=0.88, edge_alpha=0.10, linewidth=0.07)
    add_mask_surface(ax, structures["LENS_L"], axes, color="#fff0bf", alpha=0.84, edge_alpha=0.10, linewidth=0.07)
    add_mask_surface(ax, structures["GTV"], axes, color="#f05a5a", alpha=0.34, edge_alpha=0.09, linewidth=0.10)
    for idx, center in enumerate(vertex_centers()):
        add_vertex_sphere(ax, center, 5.0, color=VERTEX_COLORS[idx], alpha=0.98)

    apply_bounds(ax, bounds)
    style_publication_axes(ax)
    ax.view_init(elev=18, azim=-61)
    ax.set_title("Phase 28 Synthetic Yang Phantom with TOPAS Density Tags", pad=16, fontsize=18)

    panel.text(
        0.0,
        0.98,
        "TOPAS Density Tags",
        fontsize=16,
        fontweight="bold",
        va="top",
        ha="left",
        color="#111111",
        transform=panel.transAxes,
    )
    panel.text(
        0.0,
        0.92,
        "Tag legend is separated from the 3D panel for manuscript use.\nVertices remain planning hotspots rather than material classes.",
        fontsize=10.5,
        va="top",
        ha="left",
        color="#444444",
        linespacing=1.35,
        transform=panel.transAxes,
    )

    used_tags = [10, 11, 13, 20, 21, 50]
    legend_handles = [
        Patch(
            facecolor=TOPAS_TAG_SPECS[tag]["color"],
            edgecolor="#555555",
            linewidth=0.5,
            alpha=(0.85 if tag in {50, 13, 11} else 0.72),
            label=(
                f"Tag {tag}\n"
                f"{TOPAS_TAG_SPECS[tag]['name']}\n"
                f"{TOPAS_TAG_SPECS[tag]['density_g_cm3']:.2f} g/cm3"
            ),
        )
        for tag in used_tags
    ]
    legend = panel.legend(
        handles=legend_handles,
        loc="upper left",
        bbox_to_anchor=(0.0, 0.76),
        frameon=False,
        fontsize=10.0,
        handlelength=1.4,
        handleheight=1.2,
        labelspacing=1.1,
        borderaxespad=0.0,
    )
    for text in legend.get_texts():
        text.set_color("#222222")

    panel.text(
        0.0,
        0.22,
        "Included classes",
        fontsize=12,
        fontweight="bold",
        ha="left",
        va="top",
        color="#111111",
        transform=panel.transAxes,
    )
    panel.text(
        0.0,
        0.18,
        "Soft tissue, brain, brainstem, skull, maxillofacial bone, and tumour are written as material-tag voxels for the TsImageCube export.",
        fontsize=10.0,
        ha="left",
        va="top",
        color="#444444",
        linespacing=1.35,
        transform=panel.transAxes,
    )
    panel.text(
        0.0,
        0.08,
        "Phase 28 benchmark geometry\nYang-style 2-vertex sinonasal lattice",
        fontsize=10.0,
        ha="left",
        va="top",
        color="#666666",
        linespacing=1.35,
        transform=panel.transAxes,
    )

    fig.savefig(out_file, dpi=dpi, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def main() -> int:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    out_root = args.outdir.parent
    axes = make_axes(float(args.voxel_mm))
    structures = build_structures(axes)
    tag_grid = build_phase28_material_tag_grid(structures)
    density_grid = build_density_from_tags(tag_grid)
    used_tags = [int(tag) for tag in np.unique(tag_grid)]

    manifest_rows = []
    voxel_volume_cc = (float(args.voxel_mm) ** 3) / 1000.0
    for tag in used_tags:
        spec = TOPAS_TAG_SPECS[int(tag)]
        voxel_count = int(np.count_nonzero(tag_grid == int(tag)))
        manifest_rows.append(
            {
                "tag": int(tag),
                "name": str(spec["name"]),
                "label": str(spec["label"]),
                "base_material": str(spec["base_material"]),
                "density_g_cm3": float(spec["density_g_cm3"]),
                "voxel_count": voxel_count,
                "volume_cc": float(voxel_count * voxel_volume_cc),
            }
        )

    np.savez_compressed(
        out_root / "phase28_topas_material_tag_grid.npz",
        material_tags=tag_grid.astype(np.int16),
        density_g_cm3=density_grid.astype(np.float32),
    )
    write_csv(out_root / "phase28_topas_material_manifest.csv", manifest_rows)
    write_json(out_root / "phase28_topas_material_manifest.json", manifest_rows)
    (out_root / "phase28_topas_materials_include.txt").write_text(
        render_topas_materials_include(used_tags),
        encoding="utf-8",
    )

    outputs = (
        args.outdir / "figure6_phase28_gtv_with_vertices_3d.png",
        args.outdir / "figure7_phase28_gtv_with_nearby_oars_3d.png",
        args.outdir / "figure8_phase28_manuscript_multi_view_3d.png",
        args.outdir / "figure9_phase28_labstyle_phantom_oblique_3d.png",
        args.outdir / "figure10_phase28_labstyle_phantom_multi_view_3d.png",
        args.outdir / "figure11_phase28_cinematic_cutaway_phantom_3d.png",
        args.outdir / "figure12_phase28_topas_density_tagged_cutaway_3d.png",
        args.outdir / "figure13_phase28_topas_density_publication_sidepane.png",
    )
    plot_vertices_figure(axes, structures, outputs[0], dpi=int(args.dpi))
    plot_oar_figure(axes, structures, outputs[1], dpi=int(args.dpi))
    plot_manuscript_multi_view(axes, structures, outputs[2], dpi=int(args.dpi))
    plot_labstyle_phantom_oblique(axes, structures, outputs[3], dpi=int(args.dpi))
    plot_labstyle_phantom_multi_view(axes, structures, outputs[4], dpi=int(args.dpi))
    plot_cinematic_cutaway_phantom(axes, structures, outputs[5], dpi=int(args.dpi))
    plot_topas_density_tagged_cutaway(axes, structures, outputs[6], dpi=int(args.dpi))
    plot_topas_density_publication_sidepane(axes, structures, outputs[7], dpi=int(args.dpi))

    for output in outputs:
        print(output)
    print(out_root / "phase28_topas_material_manifest.csv")
    print(out_root / "phase28_topas_material_manifest.json")
    print(out_root / "phase28_topas_material_tag_grid.npz")
    print(out_root / "phase28_topas_materials_include.txt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
