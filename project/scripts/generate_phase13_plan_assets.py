#!/usr/bin/env python3
"""Generate manuscript-facing treatment-plan assets for the Phase 13 voxel run."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List

import numpy as np

from build_asymmetric_sweep import write_text_with_retries
from run_phase13_headneck_voxel_lattice import build_headneck_phantom

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patheffects as pe
    from matplotlib.patches import Circle
except Exception as exc:  # pragma: no cover
    raise RuntimeError("matplotlib is required for plan asset generation") from exc


GE_TRANSLATION_RE = re.compile(r"^d:Ge/BeamOrigin_(?P<name>[^/]+)/(?P<field>Trans[XYZ]) = (?P<value>[-0-9.]+) mm$")
GE_ROTATION_RE = re.compile(r"^d:Ge/BeamOrigin_(?P<name>[^/]+)/(?P<field>Rot[XYZ]) = (?P<value>[-0-9.]+) deg$")
SOURCE_CUTOFF_RE = re.compile(r"^d:So/Source_(?P<name>[^/]+)/(?P<field>BeamPositionCutoff[XY]) = (?P<value>[-0-9.]+) mm$")
SOURCE_HIST_RE = re.compile(r"^i:So/Source_(?P<name>[^/]+)/NumberOfHistoriesInRun = (?P<value>[0-9]+)$")


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Create a clean treatment-plan table and layout figure from a Phase 13 run."
    )
    parser.add_argument(
        "--run-root",
        type=Path,
        default=root / "runs" / "linac_6mv_headneck_voxel_lattice_sfrt_apbase",
        help="Completed Phase 13 run root.",
    )
    parser.add_argument("--dpi", type=int, default=250, help="Figure DPI.")
    return parser.parse_args()


def save_csv(rows: List[Dict[str, object]], out_file: Path) -> None:
    if not rows:
        raise ValueError("No rows supplied for CSV output.")
    fieldnames: List[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with out_file.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_beamline_sources(beamline_file: Path) -> List[Dict[str, object]]:
    by_name: Dict[str, Dict[str, object]] = {}
    for raw_line in beamline_file.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        match = GE_TRANSLATION_RE.match(line)
        if match:
            record = by_name.setdefault(match.group("name"), {"name": match.group("name")})
            record[match.group("field")] = float(match.group("value"))
            continue
        match = GE_ROTATION_RE.match(line)
        if match:
            record = by_name.setdefault(match.group("name"), {"name": match.group("name")})
            record[match.group("field")] = float(match.group("value"))
            continue
        match = SOURCE_CUTOFF_RE.match(line)
        if match:
            record = by_name.setdefault(match.group("name"), {"name": match.group("name")})
            record[match.group("field")] = float(match.group("value"))
            continue
        match = SOURCE_HIST_RE.match(line)
        if match:
            record = by_name.setdefault(match.group("name"), {"name": match.group("name")})
            record["histories"] = int(match.group("value"))

    rows: List[Dict[str, object]] = []
    total_histories = sum(int(record.get("histories", 0)) for record in by_name.values())
    for name, record in sorted(by_name.items()):
        if name == "AP_BASE":
            delivery = "AP"
            source_kind = "base"
            spot_label = ""
        else:
            delivery, spot_label = name.split("_", 1)
            source_kind = "spot"
        row = {
            "source_name": name,
            "kind": source_kind,
            "delivery": delivery,
            "spot_label": spot_label,
            "trans_x_mm": float(record.get("TransX", 0.0)),
            "trans_y_mm": float(record.get("TransY", 0.0)),
            "trans_z_mm": float(record.get("TransZ", 0.0)),
            "rot_x_deg": float(record.get("RotX", 0.0)),
            "rot_y_deg": float(record.get("RotY", 0.0)),
            "rot_z_deg": float(record.get("RotZ", 0.0)),
            "cutoff_x_mm": float(record.get("BeamPositionCutoffX", 0.0)),
            "cutoff_y_mm": float(record.get("BeamPositionCutoffY", 0.0)),
            "histories": int(record.get("histories", 0)),
            "history_pct": 100.0 * float(record.get("histories", 0)) / float(total_histories or 1),
        }
        rows.append(row)
    return rows


def build_spot_summary(source_rows: List[Dict[str, object]], lattice_spots: Dict[str, object]) -> List[Dict[str, object]]:
    spot_centers = lattice_spots["spot_centers_mm"]
    grouped: Dict[str, Dict[str, object]] = {}
    for row in source_rows:
        if row["kind"] != "spot":
            continue
        label = str(row["spot_label"])
        grouped.setdefault(label, {"spot_label": label, "delivery_histories": {}})
        grouped[label]["delivery_histories"][str(row["delivery"])] = int(row["histories"])
        grouped[label]["cutoff_mm"] = float(row["cutoff_x_mm"])
        if row["delivery"] == "AP":
            grouped[label]["spot_center_mm"] = [float(row["trans_x_mm"]), float(row["trans_y_mm"]), None]
        elif row["delivery"] == "LATL":
            grouped[label]["spot_center_mm"] = [None, float(row["trans_y_mm"]), float(row["trans_z_mm"])]

    summary_rows: List[Dict[str, object]] = [
        {
            "entry": "AP_BASE",
            "kind": "broad_base",
            "delivery": "AP",
            "center_mm": lattice_spots["plan_meta"]["ptv_centroid_mm"],
            "aperture_mm": float(lattice_spots["plan_meta"]["ap_radius_mm"]),
            "histories": next(row["histories"] for row in source_rows if row["source_name"] == "AP_BASE"),
            "notes": "Broad anterior compensatory field covering the bulky target envelope.",
        }
    ]

    for idx, center in enumerate(spot_centers, start=1):
        label = f"SPOT{idx:02d}"
        grouped_row = grouped[label]
        total_histories = int(sum(grouped_row["delivery_histories"].values()))
        summary_rows.append(
            {
                "entry": label,
                "kind": "lattice_vertex",
                "delivery": "AP + left lateral + right lateral",
                "center_mm": [float(center[0]), float(center[1]), float(center[2])],
                "aperture_mm": float(grouped_row["cutoff_mm"]),
                "histories": total_histories,
                "notes": (
                    f"Per direction histories: AP={grouped_row['delivery_histories'].get('AP', 0)}, "
                    f"LATL={grouped_row['delivery_histories'].get('LATL', 0)}, "
                    f"LATR={grouped_row['delivery_histories'].get('LATR', 0)}"
                ),
            }
        )
    return summary_rows


def write_markdown_table(summary_rows: List[Dict[str, object]], out_file: Path) -> None:
    lines = [
        "# SFRT Plan Summary Table",
        "",
        "| Entry | Type | Delivery | Center (mm) | Aperture radius (mm) | Histories | Notes |",
        "| --- | --- | --- | --- | ---: | ---: | --- |",
    ]
    for row in summary_rows:
        center = row["center_mm"]
        center_text = f"({center[0]:.1f}, {center[1]:.1f}, {center[2]:.1f})"
        lines.append(
            f"| {row['entry']} | {row['kind']} | {row['delivery']} | {center_text} | "
            f"{float(row['aperture_mm']):.1f} | {int(row['histories'])} | {row['notes']} |"
        )
    write_text_with_retries(out_file, "\n".join(lines) + "\n")


def write_latex_table(summary_rows: List[Dict[str, object]], out_file: Path) -> None:
    lines = [
        "\\begin{table}[!tbp]",
        "    \\centering",
        "    \\caption{Spatially fractionated treatment plan used for the voxelized head-and-neck lattice-boost study. The plan comprised one broad anterior compensatory field and seven orthogonal lattice vertices delivered from anterior, left-lateral, and right-lateral directions.}",
        "    \\label{tab:phase13_sfrt_plan}",
        "    \\small",
        "    \\begin{tabular}{llllrr}",
        "        \\toprule",
        "        Entry & Type & Delivery & Center (mm) & Radius (mm) & Histories \\\\",
        "        \\midrule",
    ]
    for row in summary_rows:
        center = row["center_mm"]
        center_text = f"({center[0]:.1f}, {center[1]:.1f}, {center[2]:.1f})"
        row_type = str(row["kind"]).replace("_", "\\_")
        delivery = str(row["delivery"]).replace("+", "$+$")
        lines.append(
            f"        {row['entry']} & {row_type} & {delivery} & {center_text} & "
            f"{float(row['aperture_mm']):.1f} & {int(row['histories'])} \\\\"
        )
    lines.extend(
        [
            "        \\bottomrule",
            "    \\end{tabular}",
            "\\end{table}",
        ]
    )
    write_text_with_retries(out_file, "\n".join(lines) + "\n")


def write_treatment_planning_subsection(
    summary_rows: List[Dict[str, object]],
    summary_json: Dict[str, object],
    out_file: Path,
) -> None:
    base_row = summary_rows[0]
    spot_rows = summary_rows[1:]
    base_histories = int(base_row["histories"])
    total_histories = int(summary_json["plan"]["histories"])
    spot_histories = int(sum(int(row["histories"]) for row in spot_rows))
    centroid = summary_json["plan"]["spot_centers_mm"][0]
    ptv_centroid = summary_json["plan"]["spot_centers_mm"]
    paragraph = (
        "\\subsection{Treatment Planning}\n"
        "A synthetic spatially fractionated treatment plan was constructed directly in TOPAS for the voxelized head-and-neck audit surrogate shown in Fig.~\\ref{fig:phase13_plan_layout}. "
        f"The delivered plan used a total of {total_histories:,} primary histories and combined one broad anterior compensatory field with seven lattice vertices positioned inside the bulky right-sided target volume. "
        f"The broad anterior field was centered on the PTV centroid at ({float(base_row['center_mm'][0]):.1f}, {float(base_row['center_mm'][1]):.1f}, {float(base_row['center_mm'][2]):.1f}) mm with an elliptical aperture radius of {float(base_row['aperture_mm']):.1f} mm and carried {base_histories:,} histories ({100.0 * base_histories / total_histories:.1f}\\% of the plan weight). "
        f"Each lattice vertex used an {float(spot_rows[0]['aperture_mm']):.1f} mm radius and was delivered from three orthogonal directions (anterior, left lateral, and right lateral), with the seven vertices sharing the remaining {spot_histories:,} histories ({100.0 * spot_histories / total_histories:.1f}\\%). "
        "This geometry was chosen to mimic an SFRT lattice-boost strategy in which a clinically necessary low-gradient base dose is retained while discrete high-intensity subvolumes are embedded within the gross disease. "
        "The final source coordinates, aperture sizes, and history allocations are listed in Table~\\ref{tab:phase13_sfrt_plan}.\n"
    )
    write_text_with_retries(out_file, paragraph)


def plot_plan_layout(
    out_file: Path,
    *,
    axes_mm: Dict[str, np.ndarray],
    structures: Dict[str, np.ndarray],
    lattice_spots: Dict[str, object],
    source_rows: List[Dict[str, object]],
    dpi: int,
) -> None:
    x_cm = axes_mm["x"] / 10.0
    y_cm = axes_mm["y"] / 10.0
    z_cm = axes_mm["z"] / 10.0

    ptv = structures["PTV"]
    gtv = structures["GTV"]
    body = structures["BODY"]
    ptv_xy = np.any(ptv, axis=2)
    gtv_xy = np.any(gtv, axis=2)
    body_xy = np.any(body, axis=2)
    ptv_xz = np.any(ptv, axis=1)
    gtv_xz = np.any(gtv, axis=1)
    body_xz = np.any(body, axis=1)
    ptv_yz = np.any(ptv, axis=0)
    gtv_yz = np.any(gtv, axis=0)
    body_yz = np.any(body, axis=0)

    spot_centers = np.asarray(lattice_spots["spot_centers_mm"], dtype=np.float32)
    centroid = np.asarray(lattice_spots["plan_meta"]["ptv_centroid_mm"], dtype=np.float32)
    ap_radius_cm = float(lattice_spots["plan_meta"]["ap_radius_mm"]) / 10.0

    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.4), constrained_layout=True)

    def add_projection(ax, base_proj, ptv_proj, gtv_proj, axis_a_cm, axis_b_cm, title, xlabel, ylabel):
        ax.imshow(
            base_proj.T,
            origin="lower",
            cmap="Greys",
            extent=[float(axis_a_cm[0]), float(axis_a_cm[-1]), float(axis_b_cm[0]), float(axis_b_cm[-1])],
            alpha=0.9,
        )
        ax.contour(axis_a_cm, axis_b_cm, ptv_proj.T.astype(float), levels=[0.5], colors=["tab:cyan"], linewidths=1.5)
        ax.contour(axis_a_cm, axis_b_cm, gtv_proj.T.astype(float), levels=[0.5], colors=["tab:red"], linewidths=1.5)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

    add_projection(axes[0], body_xy, ptv_xy, gtv_xy, x_cm, y_cm, "Coronal projection", "x (cm)", "y (cm)")
    add_projection(axes[1], body_xz, ptv_xz, gtv_xz, x_cm, z_cm, "Axial-style projection", "x (cm)", "z (cm)")
    add_projection(axes[2], body_yz, ptv_yz, gtv_yz, y_cm, z_cm, "Sagittal projection", "y (cm)", "z (cm)")

    axes[0].scatter(spot_centers[:, 0] / 10.0, spot_centers[:, 1] / 10.0, c="yellow", s=42, edgecolors="black", linewidths=0.5, zorder=5)
    axes[1].scatter(spot_centers[:, 0] / 10.0, spot_centers[:, 2] / 10.0, c="yellow", s=42, edgecolors="black", linewidths=0.5, zorder=5)
    axes[2].scatter(spot_centers[:, 1] / 10.0, spot_centers[:, 2] / 10.0, c="yellow", s=42, edgecolors="black", linewidths=0.5, zorder=5)

    outline = [pe.withStroke(linewidth=1.5, foreground="black")]
    for idx, (sx, sy, sz) in enumerate(spot_centers, start=1):
        axes[0].annotate(str(idx), (sx / 10.0, sy / 10.0), xytext=(3, 3), textcoords="offset points", fontsize=8, color="black")
        axial_text = axes[1].annotate(
            str(idx),
            (sx / 10.0, sz / 10.0),
            xytext=(3, 3),
            textcoords="offset points",
            fontsize=8,
            color="white",
        )
        axial_text.set_path_effects(outline)
        axes[2].annotate(str(idx), (sy / 10.0, sz / 10.0), xytext=(3, 3), textcoords="offset points", fontsize=8, color="black")

    ap_circle = Circle((centroid[0] / 10.0, centroid[1] / 10.0), radius=ap_radius_cm, fill=False, linestyle="--", linewidth=1.4, color="tab:blue")
    axes[0].add_patch(ap_circle)
    axes[0].annotate(
        "AP broad base field",
        xy=(centroid[0] / 10.0 + ap_radius_cm, centroid[1] / 10.0),
        xytext=(10, 10),
        textcoords="offset points",
        fontsize=9,
        color="tab:blue",
    )

    axes[1].annotate(
        "Orthogonal lattice delivery\nAP + left lateral + right lateral",
        xy=(spot_centers[0, 0] / 10.0, spot_centers[0, 2] / 10.0),
        xytext=(-35, 25),
        textcoords="offset points",
        arrowprops={"arrowstyle": "->", "lw": 1.0},
        fontsize=9,
    )

    handles = [
        plt.Line2D([0], [0], color="tab:cyan", lw=1.8, label="PTV contour"),
        plt.Line2D([0], [0], color="tab:red", lw=1.8, label="GTV contour"),
        plt.Line2D([0], [0], marker="o", linestyle="", markersize=7, markerfacecolor="yellow", markeredgecolor="black", label="Lattice spot center"),
        plt.Line2D([0], [0], color="tab:blue", lw=1.4, linestyle="--", label="AP base-field footprint"),
    ]
    axes[2].legend(handles=handles, loc="lower right", fontsize=8)
    fig.suptitle("Spatially fractionated treatment plan: lattice-boost geometry in the voxelized head-and-neck phantom", fontsize=13)
    fig.savefig(out_file, dpi=dpi)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    analysis_dir = args.run_root / "analysis"
    phantom_dir = args.run_root / "phantom"
    case_dir = args.run_root / "case"

    summary = json.loads((analysis_dir / "phase13_headneck_summary.json").read_text(encoding="utf-8"))
    phantom_args = SimpleNamespace(
        size_x_cm=float(summary["phantom"]["size_cm"][0]),
        size_y_cm=float(summary["phantom"]["size_cm"][1]),
        size_z_cm=float(summary["phantom"]["size_cm"][2]),
        voxel_mm=float(summary["phantom"]["voxel_size_mm"][0]),
    )
    phantom = build_headneck_phantom(phantom_args)
    lattice_spots = json.loads((phantom_dir / "lattice_spots.json").read_text(encoding="utf-8"))
    source_rows = parse_beamline_sources(case_dir / "beamline.txt")
    summary_rows = build_spot_summary(source_rows, lattice_spots)

    save_csv(source_rows, analysis_dir / "phase13_plan_sources_full.csv")
    save_csv(summary_rows, analysis_dir / "phase13_plan_spot_summary.csv")
    write_markdown_table(summary_rows, analysis_dir / "phase13_plan_spot_summary.md")
    write_latex_table(summary_rows, analysis_dir / "phase13_plan_spot_summary.tex")
    write_treatment_planning_subsection(summary_rows, summary, analysis_dir / "phase13_treatment_planning_subsection.tex")
    plot_plan_layout(
        analysis_dir / "figure5_sfrt_plan_layout.png",
        axes_mm=phantom["axes_mm"],
        structures=phantom["structures"],
        lattice_spots=lattice_spots,
        source_rows=source_rows,
        dpi=int(args.dpi),
    )
    print(analysis_dir / "phase13_plan_sources_full.csv")
    print(analysis_dir / "phase13_plan_spot_summary.csv")
    print(analysis_dir / "phase13_plan_spot_summary.md")
    print(analysis_dir / "phase13_plan_spot_summary.tex")
    print(analysis_dir / "phase13_treatment_planning_subsection.tex")
    print(analysis_dir / "figure5_sfrt_plan_layout.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
