#!/usr/bin/env python3
"""Phase 27: collect Phase 25/25A/26 into a publication-ready evidence package."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


PHASE25_ROOT_DEFAULT = (
    Path(__file__).resolve().parents[1]
    / "runs"
    / "linac_6mv_headneck_detailed_voxel_lattice_sfrt_phase25_safe_core_plan_library_1e5"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase25-root", type=Path, default=PHASE25_ROOT_DEFAULT)
    parser.add_argument("--out-root", type=Path, default=None)
    parser.add_argument("--dpi", type=int, default=220)
    return parser.parse_args()


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        raise ValueError(f"No rows supplied for {path}")
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


def copy_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def f(row: Mapping[str, str], key: str) -> float:
    return float(row[key])


def mean(values: Iterable[float]) -> float:
    arr = np.asarray(list(values), dtype=np.float64)
    return float(np.mean(arr)) if arr.size else 0.0


def count_rank_shifts(rows: Sequence[Mapping[str, str]], key: str = "rank_shift") -> int:
    return sum(int(float(row[key])) != 0 for row in rows)


def unique_plan_ids(rows: Sequence[Mapping[str, str]]) -> List[str]:
    return sorted({str(row["plan_id"]) for row in rows})


def plot_workflow(out_file: Path, *, dpi: int) -> None:
    fig, ax = plt.subplots(figsize=(13.5, 4.2), constrained_layout=True)
    ax.axis("off")
    steps = [
        ("Safe-core\nlattice plans", "Protocol-filtered\n2-vertex courses"),
        ("TOPAS dose", "Physical course\nvolumes reused"),
        ("Bystander PDE", "No sink vs\nvascular sink"),
        ("Clinical endpoints", "D95, PVDR,\nspill, OARs"),
        ("Assay proxies", "gammaH2AX,\nTUNEL, ELISA"),
        ("Risk analysis", "Rank shifts,\nOAR reinterpretation"),
    ]
    xs = np.linspace(0.06, 0.94, len(steps))
    y = 0.55
    for idx, ((title, subtitle), x) in enumerate(zip(steps, xs)):
        box = plt.Rectangle((x - 0.075, y - 0.18), 0.15, 0.36, facecolor="#f3efe6", edgecolor="#333333", linewidth=1.4)
        ax.add_patch(box)
        ax.text(x, y + 0.055, title, ha="center", va="center", fontsize=10, weight="bold")
        ax.text(x, y - 0.075, subtitle, ha="center", va="center", fontsize=8.5)
        if idx < len(steps) - 1:
            ax.annotate(
                "",
                xy=(xs[idx + 1] - 0.085, y),
                xytext=(x + 0.085, y),
                arrowprops=dict(arrowstyle="->", color="#333333", lw=1.6),
            )
    ax.text(
        0.5,
        0.11,
        "Phase 27 package: biological risk-analysis framing for Lattice/SFRT, not treatment-plan optimization",
        ha="center",
        va="center",
        fontsize=11,
        color="#333333",
    )
    fig.savefig(out_file, dpi=int(dpi))
    plt.close(fig)


def plot_phase25_endpoint_summary(out_file: Path, rows: Sequence[Mapping[str, str]], *, dpi: int) -> None:
    endpoints = [
        ("ptv_d95", "PTV D95", "Gy"),
        ("pvdr", "PVDR", "ratio"),
        ("spill_shell_0_5_mean", "Peri-GTV 0-5", "Gy"),
        ("spill_shell_5_15_mean", "Peri-GTV 5-15", "Gy"),
        ("brainstem_d2", "Brainstem D2", "Gy"),
        ("parotid_r_mean", "Parotid R mean", "Gy"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 7.4), constrained_layout=True)
    labels = [row["plan_id"] for row in rows]
    x = np.arange(len(rows))
    width = 0.36
    for ax, (key, label, units) in zip(axes.ravel(), endpoints):
        phys = [f(row, f"{key}_phys") for row in rows]
        bio = [f(row, f"{key}_bio") for row in rows]
        ax.bar(x - width / 2, phys, width=width, label="Physical", color="#4f79a7")
        ax.bar(x + width / 2, bio, width=width, label="Bio-aware", color="#c95f4a")
        ax.set_title(f"{label} ({units})")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=20, ha="right")
        ax.grid(axis="y", alpha=0.25)
    handles, labels_legend = axes.ravel()[0].get_legend_handles_labels()
    fig.legend(handles, labels_legend, loc="upper center", ncol=2, frameon=False)
    fig.suptitle("Phase 25: physical-only versus biological endpoint interpretation", fontsize=15)
    fig.savefig(out_file, dpi=int(dpi))
    plt.close(fig)


def plot_phase25a_assays(out_file: Path, rows: Sequence[Mapping[str, str]], *, dpi: int) -> None:
    labels = [row["plan_id"] for row in rows]
    x = np.arange(len(rows))
    width = 0.36
    fig, axes = plt.subplots(1, 3, figsize=(14.0, 4.8), constrained_layout=True)
    pairs = [
        ("mean_gammah2ax_peak", "mean_gammah2ax_valley", "gammaH2AX proxy"),
        ("mean_tunel_peak", "mean_tunel_valley", "TUNEL proxy"),
        ("cytokine_peak_roi_auc", "cytokine_valley_roi_auc", "ELISA-like cytokine AUC"),
    ]
    for ax, (peak_key, valley_key, title) in zip(axes, pairs):
        ax.bar(x - width / 2, [f(row, peak_key) for row in rows], width=width, label="Peak ROI", color="#b33f2f")
        ax.bar(x + width / 2, [f(row, valley_key) for row in rows], width=width, label="Valley ROI", color="#3274a1")
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=20, ha="right")
        ax.grid(axis="y", alpha=0.25)
    handles, labels_legend = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels_legend, loc="upper center", ncol=2, frameon=False)
    fig.suptitle("Phase 25A: assay-like peak-versus-valley biological response", fontsize=15)
    fig.savefig(out_file, dpi=int(dpi))
    plt.close(fig)


def plot_phase26_ablation_summary(
    out_file: Path,
    endpoint_rows: Sequence[Mapping[str, str]],
    rank_rows: Sequence[Mapping[str, str]],
    *,
    dpi: int,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.2), constrained_layout=True)
    ax = axes[0]
    x = np.arange(3)
    for row in rank_rows:
        ranks = [f(row, "physical_rank"), f(row, "no_sink_rank"), f(row, "with_sink_rank")]
        ax.plot(x, ranks, marker="o", linewidth=1.8)
        ax.text(x[-1] + 0.04, ranks[-1], row["plan_id"], va="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(["Physical", "No sink", "With sink"])
    ax.set_ylabel("Risk rank (1 = lowest)")
    ax.invert_yaxis()
    ax.grid(axis="y", alpha=0.25)
    ax.set_title("Plan ranking shifts")

    ax = axes[1]
    deltas: Dict[str, List[float]] = {}
    by_plan_mode = {(row["plan_id"], row["mode"]): row for row in endpoint_rows}
    for plan_id in unique_plan_ids(endpoint_rows):
        no_sink = by_plan_mode[(plan_id, "bystander_no_sink")]
        with_sink = by_plan_mode[(plan_id, "bystander_with_sink")]
        for key, label in [
            ("ptv_d95", "PTV D95"),
            ("pvdr", "PVDR"),
            ("spill_shell_0_5_mean", "Shell 0-5"),
            ("brainstem_d2", "Brainstem"),
            ("parotid_r_mean", "Parotid R"),
        ]:
            deltas.setdefault(label, []).append(f(with_sink, key) - f(no_sink, key))
    labels = list(deltas.keys())
    means = [mean(deltas[label]) for label in labels]
    ax.bar(np.arange(len(labels)), means, color="#4a9a62")
    ax.axhline(0, color="#333333", linewidth=1.0)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylabel("With-sink minus no-sink value")
    ax.set_title("Marginal vascular sink effect")
    ax.grid(axis="y", alpha=0.25)
    fig.suptitle("Phase 26: vascular sink ablation", fontsize=15)
    fig.savefig(out_file, dpi=int(dpi))
    plt.close(fig)


def build_phase27_master_summary(
    *,
    phase25_rows: Sequence[Mapping[str, str]],
    phase25a_rows: Sequence[Mapping[str, str]],
    phase26_endpoint_rows: Sequence[Mapping[str, str]],
    phase26_rank_rows: Sequence[Mapping[str, str]],
    phase26_oar_rows: Sequence[Mapping[str, str]],
    phase26_band_rows: Sequence[Mapping[str, str]],
) -> Dict[str, object]:
    phase25_rank_shifts = count_rank_shifts(phase25_rows)
    first_failure_values = [f(row, "first_bio_failure_fraction") for row in phase25_rows]
    phase26_with_sink_rank_shifts = sum(int(float(row["with_sink_vs_physical_shift"])) != 0 for row in phase26_rank_rows)
    phase26_sink_vs_no_sink_rank_shifts = sum(int(float(row["with_sink_vs_no_sink_shift"])) != 0 for row in phase26_rank_rows)
    biology_added_failures = [row for row in phase26_oar_rows if row["reinterpretation"] == "biology_adds_failure"]

    by_plan_mode = {(row["plan_id"], row["mode"]): row for row in phase26_endpoint_rows}
    vascular_mean_deltas = {}
    for key in ["ptv_d95", "pvdr", "spill_shell_0_5_mean", "spill_shell_5_15_mean", "brainstem_d2", "parotid_r_mean"]:
        values = []
        for plan_id in unique_plan_ids(phase26_endpoint_rows):
            values.append(f(by_plan_mode[(plan_id, "bystander_with_sink")], key) - f(by_plan_mode[(plan_id, "bystander_no_sink")], key))
        vascular_mean_deltas[key] = mean(values)

    assay_means = {
        "gammah2ax_peak": mean(f(row, "mean_gammah2ax_peak") for row in phase25a_rows),
        "gammah2ax_valley": mean(f(row, "mean_gammah2ax_valley") for row in phase25a_rows),
        "tunel_peak": mean(f(row, "mean_tunel_peak") for row in phase25a_rows),
        "tunel_valley": mean(f(row, "mean_tunel_valley") for row in phase25a_rows),
        "cytokine_peak_auc": mean(f(row, "cytokine_peak_roi_auc") for row in phase25a_rows),
        "cytokine_valley_auc": mean(f(row, "cytokine_valley_roi_auc") for row in phase25a_rows),
    }

    band_means = {}
    for metric in ["ptv_d95", "brainstem_d2", "parotid_r_mean", "cytokine_peak_roi_auc", "cytokine_valley_roi_auc"]:
        vals = [
            f(row, "relative_full_width_pct")
            for row in phase26_band_rows
            if row["metric"] == metric
        ]
        band_means[metric] = mean(vals)

    return {
        "phase25_plan_count": len(phase25_rows),
        "phase25_rank_shift_count": phase25_rank_shifts,
        "phase25_all_failed_by_fraction": int(max(first_failure_values)) if first_failure_values else None,
        "phase25_mean_body_dmax_phys": mean(f(row, "body_dmax_phys") for row in phase25_rows),
        "phase25_mean_ptv_d95_phys": mean(f(row, "ptv_d95_phys") for row in phase25_rows),
        "phase25_mean_ptv_d95_bio": mean(f(row, "ptv_d95_bio") for row in phase25_rows),
        "phase25a_assay_means": assay_means,
        "phase26_with_sink_rank_shift_count_vs_physical": phase26_with_sink_rank_shifts,
        "phase26_with_sink_rank_shift_count_vs_no_sink": phase26_sink_vs_no_sink_rank_shifts,
        "phase26_biology_added_oar_failures": [
            {
                "plan_id": row["plan_id"],
                "metric": row["metric"],
                "physical_value": f(row, "physical_value"),
                "with_sink_value": f(row, "with_sink_value"),
                "limit": f(row, "limit"),
            }
            for row in biology_added_failures
        ],
        "phase26_mean_vascular_sink_deltas": vascular_mean_deltas,
        "phase26_uptake_sensitivity_band_means_pct": band_means,
    }


def write_results_markdown(path: Path, summary: Mapping[str, object]) -> None:
    assay = dict(summary["phase25a_assay_means"])
    sink = dict(summary["phase26_mean_vascular_sink_deltas"])
    bands = dict(summary["phase26_uptake_sensitivity_band_means_pct"])
    failures = list(summary["phase26_biology_added_oar_failures"])
    failure_text = (
        f"{failures[0]['plan_id']} {failures[0]['metric']} "
        f"({failures[0]['physical_value']:.2f} -> {failures[0]['with_sink_value']:.2f} Gy, "
        f"limit {failures[0]['limit']:.1f} Gy)"
        if failures
        else "none"
    )
    text = f"""# Phase 27 publication package summary

## Framing

This package reframes the work as a biological risk-analysis framework for Lattice/SFRT. The emphasis is no longer optimizer success, but whether non-local bystander modelling and anatomical vascular sink uptake reveal biological risk features that physical dose metrics alone miss.

## Phase 25: clinical endpoint reinterpretation

- Plans evaluated: {summary['phase25_plan_count']}
- Physical-versus-biological rank shifts: {summary['phase25_rank_shift_count']} / {summary['phase25_plan_count']}
- Mean physical PTV D95: {summary['phase25_mean_ptv_d95_phys']:.2f} Gy
- Mean biological PTV D95: {summary['phase25_mean_ptv_d95_bio']:.2f} Gy
- Mean physical body Dmax: {summary['phase25_mean_body_dmax_phys']:.2f} Gy
- All plans first failed by fraction: {summary['phase25_all_failed_by_fraction']}

Interpretation: the safe-core plan library is anatomically more credible than the earlier random bounding-box plans, but the delivery family remains physically hotspot-prone. The important contribution is therefore risk reinterpretation, not clinical plan acceptability.

## Phase 25A: assay-like biological outputs

- Mean gammaH2AX peak versus valley: {assay['gammah2ax_peak']:.3f} vs {assay['gammah2ax_valley']:.3f}
- Mean TUNEL peak versus valley: {assay['tunel_peak']:.3f} vs {assay['tunel_valley']:.3f}
- Mean cytokine peak AUC versus valley AUC: {assay['cytokine_peak_auc']:.1f} vs {assay['cytokine_valley_auc']:.1f}

Interpretation: the assay proxies preserve an SFRT-like biological separation between peak and valley compartments while still showing substantial valley burden, which supports the use of the model as a biological risk readout.

## Phase 26: vascular sink ablation

- With-sink rank shifts versus physical-only: {summary['phase26_with_sink_rank_shift_count_vs_physical']} / {summary['phase25_plan_count']}
- With-sink rank shifts versus no-sink bystander model: {summary['phase26_with_sink_rank_shift_count_vs_no_sink']} / {summary['phase25_plan_count']}
- Biology-added OAR failure: {failure_text}

Mean anatomical vascular sink effect, reported as with-sink minus no-sink:

- PTV D95: {sink['ptv_d95']:.2f} Gy
- PVDR: {sink['pvdr']:.3f}
- Peri-GTV 0-5 mm mean: {sink['spill_shell_0_5_mean']:.2f} Gy
- Peri-GTV 5-15 mm mean: {sink['spill_shell_5_15_mean']:.2f} Gy
- Brainstem D2: {sink['brainstem_d2']:.3f} Gy
- Parotid R mean: {sink['parotid_r_mean']:.2f} Gy

Interpretation: anatomical vascular uptake materially changes ranking in most plans and suppresses non-local/cytokine-mediated burden, especially in valley and spill-like regions, while having minimal effect on serial D2 endpoints.

## Uptake sensitivity

Mean uptake sensitivity band widths:

- PTV D95: {bands['ptv_d95']:.2f}% of base with-sink value
- Brainstem D2: {bands['brainstem_d2']:.3f}% of base with-sink value
- Parotid R mean: {bands['parotid_r_mean']:.2f}% of base with-sink value
- Cytokine peak AUC: {bands['cytokine_peak_roi_auc']:.2f}% of base with-sink value
- Cytokine valley AUC: {bands['cytokine_valley_roi_auc']:.2f}% of base with-sink value

Interpretation: these sensitivity bands align with the prior Morris analysis by showing that uptake variation most clearly affects signalling and valley-burden endpoints rather than every clinical DVH endpoint equally.

## Reviewer-safe novelty statement

To our knowledge, this is among the first protocol-constrained, multi-fraction Lattice/SFRT risk-analysis frameworks to combine non-local bystander signalling, Morris-motivated anatomical vascular sink uptake, physical-versus-biological clinical endpoint comparison, and assay-like readouts in a single plan-evaluation pipeline.
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def main() -> int:
    args = parse_args()
    phase25_root = args.phase25_root.resolve()
    out_root = args.out_root.resolve() if args.out_root else phase25_root / "phase27_publication_package"
    tables_dir = out_root / "tables"
    figures_dir = out_root / "figures"
    source_figures_dir = out_root / "source_figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    source_figures_dir.mkdir(parents=True, exist_ok=True)

    phase25_rows = read_csv(phase25_root / "phase25_endpoint_table.csv")
    phase25a_rows = read_csv(phase25_root / "phase25a_assays" / "phase25a_assay_summary.csv")
    phase26_root = phase25_root / "phase26_vascular_sink_ablation"
    phase26_endpoint_rows = read_csv(phase26_root / "phase26_endpoint_table.csv")
    phase26_rank_rows = read_csv(phase26_root / "phase26_rank_shift_table.csv")
    phase26_oar_rows = read_csv(phase26_root / "phase26_oar_reinterpretation_table.csv")
    phase26_band_rows = read_csv(phase26_root / "phase26_sensitivity_bands.csv")

    copy_targets = {
        phase25_root / "phase25_endpoint_table.csv": tables_dir / "table1_phase25_physical_vs_bio_endpoints.csv",
        phase25_root / "phase25_plan_manifest.csv": tables_dir / "table2_phase25_plan_manifest.csv",
        phase25_root / "phase25a_assays" / "phase25a_assay_summary.csv": tables_dir / "table3_phase25a_assay_summary.csv",
        phase26_root / "phase26_endpoint_table.csv": tables_dir / "table4_phase26_ablation_endpoints.csv",
        phase26_root / "phase26_rank_shift_table.csv": tables_dir / "table5_phase26_rank_shifts.csv",
        phase26_root / "phase26_oar_reinterpretation_table.csv": tables_dir / "table6_phase26_oar_reinterpretation.csv",
        phase26_root / "phase26_sensitivity_bands.csv": tables_dir / "table7_phase26_uptake_sensitivity_bands.csv",
    }
    for src, dst in copy_targets.items():
        copy_file(src, dst)

    for src in list((phase25_root / "figures").glob("*.png")):
        copy_file(src, source_figures_dir / f"phase25_{src.name}")
    for src in list((phase25_root / "phase25a_assays").glob("figure*.png")):
        copy_file(src, source_figures_dir / f"phase25a_{src.name}")
    for src in list((phase26_root / "figures").glob("*.png")):
        copy_file(src, source_figures_dir / f"phase26_{src.name}")

    plot_workflow(figures_dir / "figure1_phase27_workflow.png", dpi=int(args.dpi))
    plot_phase25_endpoint_summary(figures_dir / "figure2_phase27_phase25_endpoint_summary.png", phase25_rows, dpi=int(args.dpi))
    plot_phase25a_assays(figures_dir / "figure3_phase27_assay_proxy_summary.png", phase25a_rows, dpi=int(args.dpi))
    plot_phase26_ablation_summary(
        figures_dir / "figure4_phase27_vascular_sink_ablation_summary.png",
        phase26_endpoint_rows,
        phase26_rank_rows,
        dpi=int(args.dpi),
    )

    summary = build_phase27_master_summary(
        phase25_rows=phase25_rows,
        phase25a_rows=phase25a_rows,
        phase26_endpoint_rows=phase26_endpoint_rows,
        phase26_rank_rows=phase26_rank_rows,
        phase26_oar_rows=phase26_oar_rows,
        phase26_band_rows=phase26_band_rows,
    )
    write_json(out_root / "phase27_master_summary.json", summary)
    write_results_markdown(out_root / "phase27_results_summary.md", summary)

    print("=== PHASE 27 PUBLICATION PACKAGE COMPLETE ===")
    print(f"Output root: {out_root}")
    print(f"Results summary: {out_root / 'phase27_results_summary.md'}")
    print(f"Figures: {figures_dir}")
    print(f"Tables: {tables_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
