#!/usr/bin/env python3
"""Render manuscript figures and text from the Phase 31 publication package."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase31-root",
        type=Path,
        default=root / "runs" / "phase31_publication_package",
        help="Completed Phase 31 publication package root.",
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        default=None,
        help="Output root for manuscript assets. Defaults to <phase31-root>/manuscript_assets.",
    )
    parser.add_argument("--dpi", type=int, default=220)
    return parser.parse_args()


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        raise ValueError(f"No rows for {path}")
    fieldnames: List[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(str(key))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def to_float(value: object) -> float:
    text = str(value).strip()
    if text == "" or text.lower() == "nan":
        return float("nan")
    return float(text)


def find_row(rows: Sequence[Mapping[str, object]], key: str, value: str) -> Mapping[str, object]:
    for row in rows:
        if str(row[key]) == value:
            return row
    raise KeyError(f"Could not find row with {key}={value}")


def load_inputs(phase31_root: Path) -> Dict[str, object]:
    manifest = json.loads((phase31_root / "phase31_reproducibility_manifest.json").read_text(encoding="utf-8"))
    phase26_root = Path(manifest["input_roots"]["phase26_run_root"])
    return {
        "manifest": manifest,
        "discussion_draft": (phase31_root / "phase31_discussion_draft.md").read_text(encoding="utf-8").strip(),
        "benchmark": read_csv(phase31_root / "phase31_benchmark_template_library.csv"),
        "master": read_csv(phase31_root / "phase31_manuscript_master_table.csv"),
        "phase30_bio": read_csv(phase31_root / "phase31_phase30_biology_table.csv"),
        "phase30_repeat_summary": read_csv(phase31_root / "phase31_phase30_repeat_summary.csv"),
        "phase30_repeat_bands": read_csv(phase31_root / "phase31_phase30_repeat_bands.csv"),
        "rank_effects": read_csv(phase31_root / "phase31_rank_effect_sizes.csv"),
        "rank_robustness": read_csv(phase31_root / "phase31_rank_robustness_table.csv"),
        "sink_effects": read_csv(phase31_root / "phase31_sink_delta_effect_sizes.csv"),
        "phase26_rank_shift": read_csv(phase26_root / "phase26_rank_shift_table.csv"),
        "phase26_assays": read_csv(phase26_root / "phase26_assay_proxy_table.csv"),
        "phase26_endpoints": read_csv(phase26_root / "phase26_endpoint_table.csv"),
    }


def benchmark_summary(rows: Sequence[Mapping[str, str]]) -> Dict[str, float]:
    gtv = np.asarray([to_float(row["estimated_gtv_cc"]) for row in rows], dtype=np.float64)
    count = np.asarray([to_float(row["vertex_count"]) for row in rows], dtype=np.float64)
    diam = np.asarray([to_float(row["vertex_diameter_cm_recommended"]) for row in rows], dtype=np.float64)
    spacing = np.asarray([to_float(row["mean_center_to_center_spacing_cm"]) for row in rows], dtype=np.float64)
    return {
        "n_templates": float(len(rows)),
        "gtv_min": float(np.min(gtv)),
        "gtv_max": float(np.max(gtv)),
        "gtv_median": float(np.median(gtv)),
        "vertex_min": float(np.min(count)),
        "vertex_max": float(np.max(count)),
        "diam_min": float(np.min(diam)),
        "diam_max": float(np.max(diam)),
        "spacing_mean": float(np.mean(spacing)),
        "spacing_min": float(np.min(spacing)),
        "spacing_max": float(np.max(spacing)),
    }


def plot_benchmark_library(out_file: Path, rows: Sequence[Mapping[str, str]], *, dpi: int) -> None:
    gtv = np.asarray([to_float(row["estimated_gtv_cc"]) for row in rows], dtype=np.float64)
    spacing = np.asarray([to_float(row["mean_center_to_center_spacing_cm"]) for row in rows], dtype=np.float64)
    diameter = np.asarray([to_float(row["vertex_diameter_cm_recommended"]) for row in rows], dtype=np.float64)
    vertex_count = np.asarray([to_float(row["vertex_count"]) for row in rows], dtype=np.float64)
    labels = [str(row["template_id"]) for row in rows]

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.4), constrained_layout=True)

    scatter = axes[0].scatter(
        gtv,
        spacing,
        s=140.0 + 80.0 * (diameter - np.min(diameter) + 0.2),
        c=vertex_count,
        cmap="YlOrRd",
        edgecolors="#222222",
        linewidths=0.7,
        alpha=0.92,
    )
    for x, y, label in zip(gtv, spacing, labels):
        axes[0].annotate(label, (x, y), xytext=(5, 5), textcoords="offset points", fontsize=8)
    axes[0].set_xlabel("Estimated GTV (cc)")
    axes[0].set_ylabel("Mean vertex spacing (cm)")
    axes[0].set_title("Synthetic H&N benchmark lattice templates")
    axes[0].grid(alpha=0.25)
    cbar = fig.colorbar(scatter, ax=axes[0], shrink=0.9)
    cbar.set_label("Vertex count")

    order = np.argsort(gtv)
    axes[1].bar(
        np.arange(len(rows)),
        gtv[order],
        color="#3C6E71",
        edgecolor="#1B1B1B",
        linewidth=0.6,
    )
    axes[1].scatter(
        np.arange(len(rows)),
        10.0 * vertex_count[order],
        color="#E07A5F",
        label="10 x vertex count",
        zorder=3,
    )
    axes[1].set_xticks(np.arange(len(rows)))
    axes[1].set_xticklabels([labels[idx] for idx in order], rotation=25, ha="right")
    axes[1].set_ylabel("GTV (cc)")
    axes[1].set_title("Template size and lattice complexity")
    axes[1].grid(axis="y", alpha=0.25)
    axes[1].legend(frameon=False, fontsize=8)

    fig.savefig(out_file, dpi=int(dpi))
    plt.close(fig)


def plot_rank_reinterpretation(
    out_file: Path,
    rank_shift_rows: Sequence[Mapping[str, str]],
    rank_robustness_rows: Sequence[Mapping[str, str]],
    *,
    dpi: int,
) -> None:
    plans = [str(row["plan_id"]) for row in rank_shift_rows]
    physical = np.asarray([to_float(row["physical_rank"]) for row in rank_shift_rows], dtype=np.float64)
    no_sink = np.asarray([to_float(row["no_sink_rank"]) for row in rank_shift_rows], dtype=np.float64)
    with_sink = np.asarray([to_float(row["with_sink_rank"]) for row in rank_shift_rows], dtype=np.float64)

    fig, axes = plt.subplots(1, 2, figsize=(13.8, 5.8), constrained_layout=True)

    x = np.asarray([0, 1, 2], dtype=np.float64)
    mode_labels = ["Physical", "No sink", "With sink"]
    colors = plt.cm.tab10(np.linspace(0.0, 1.0, len(plans)))
    for idx, plan in enumerate(plans):
        y = np.asarray([physical[idx], no_sink[idx], with_sink[idx]], dtype=np.float64)
        axes[0].plot(x, y, marker="o", color=colors[idx], linewidth=2.0, label=plan)
        axes[0].text(x[-1] + 0.05, y[-1], plan, color=colors[idx], va="center", fontsize=8)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(mode_labels)
    axes[0].set_ylim(len(plans) + 0.5, 0.5)
    axes[0].set_ylabel("Plan rank (1 = lowest risk)")
    axes[0].set_title("Plan rank reinterpretation across model modes")
    axes[0].grid(alpha=0.25)

    mode_order = ["physical_only", "bystander_no_sink", "bystander_with_sink"]
    robust_matrix = np.full((len(plans), len(mode_order)), np.nan, dtype=np.float64)
    for row in rank_robustness_rows:
        plan = str(row["plan_id"])
        mode = str(row["mode"])
        robust_matrix[plans.index(plan), mode_order.index(mode)] = to_float(row["rank_retention_probability"])
    im = axes[1].imshow(robust_matrix, cmap="YlGnBu", vmin=0.0, vmax=1.0, aspect="auto")
    axes[1].set_xticks(np.arange(len(mode_order)))
    axes[1].set_xticklabels(["Physical", "No sink", "With sink"])
    axes[1].set_yticks(np.arange(len(plans)))
    axes[1].set_yticklabels(plans)
    axes[1].set_title("Rank-retention probability under uncertainty")
    for i in range(robust_matrix.shape[0]):
        for j in range(robust_matrix.shape[1]):
            axes[1].text(j, i, f"{robust_matrix[i, j]:.2f}", ha="center", va="center", fontsize=8, color="#0B0B0B")
    cbar = fig.colorbar(im, ax=axes[1], shrink=0.9)
    cbar.set_label("Probability of retaining nominal rank")

    fig.savefig(out_file, dpi=int(dpi))
    plt.close(fig)


def plot_sink_effects(out_file: Path, rows: Sequence[Mapping[str, str]], *, dpi: int) -> None:
    wanted = [
        ("endpoint", "pvdr", "PVDR"),
        ("endpoint", "spill_shell_0_5_mean", "Peri-GTV 0-5 mm"),
        ("endpoint", "spill_shell_5_15_mean", "Peri-GTV 5-15 mm"),
        ("endpoint", "brainstem_d2", "Brainstem D2"),
        ("endpoint", "parotid_r_mean", "Parotid R mean"),
        ("assay", "mean_tunel_valley", "TUNEL valley"),
        ("assay", "cytokine_peak_roi_auc", "Cytokine peak AUC"),
        ("assay", "cytokine_valley_roi_auc", "Cytokine valley AUC"),
    ]
    selected = []
    for metric_type, key, label in wanted:
        row = next(r for r in rows if str(r["metric_type"]) == metric_type and str(r["metric"]) == key)
        selected.append(
            {
                "label": label,
                "mean": to_float(row["mean_with_sink_minus_no_sink"]),
                "lo": to_float(row["ci_lower"]),
                "hi": to_float(row["ci_upper"]),
                "metric_type": metric_type,
            }
        )

    fig, axes = plt.subplots(1, 2, figsize=(13.8, 5.6), constrained_layout=True)
    for ax, metric_type, title in zip(
        axes,
        ("endpoint", "assay"),
        ("Primary endpoint shifts", "Assay-proxy shifts"),
    ):
        subset = [row for row in selected if row["metric_type"] == metric_type]
        ypos = np.arange(len(subset))
        means = np.asarray([row["mean"] for row in subset], dtype=np.float64)
        xerr = np.vstack(
            [
                means - np.asarray([row["lo"] for row in subset], dtype=np.float64),
                np.asarray([row["hi"] for row in subset], dtype=np.float64) - means,
            ]
        )
        ax.errorbar(means, ypos, xerr=xerr, fmt="o", color="#0B3954", ecolor="#087E8B", capsize=3, linewidth=1.5)
        ax.axvline(0.0, color="#555555", linestyle="--", linewidth=1.0)
        ax.set_yticks(ypos)
        ax.set_yticklabels([row["label"] for row in subset])
        ax.set_title(title)
        ax.grid(axis="x", alpha=0.25)
        ax.set_xlabel("With-sink minus no-sink")

    fig.savefig(out_file, dpi=int(dpi))
    plt.close(fig)


def plot_phase30_topas_reinterpretation(out_file: Path, rows: Sequence[Mapping[str, str]], *, dpi: int) -> None:
    mode_order = ["physical_only", "bystander_no_sink", "bystander_with_sink"]
    mode_labels = ["Physical", "No sink", "With sink"]
    row_lookup = {str(row["mode"]): row for row in rows}
    metrics = [
        ("pvdr", "PVDR"),
        ("spill_shell_0_5_mean_gy", "Peri-GTV 0-5 mm"),
        ("parotid_r_mean_gy", "Parotid R mean"),
        ("brainstem_d2_gy", "Brainstem D2"),
        ("mean_gammah2ax_peak", "gammaH2AX peak"),
        ("cytokine_valley_roi_auc", "Cytokine valley AUC"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(14.5, 8.0), constrained_layout=True)
    for ax, (key, label) in zip(axes.ravel(), metrics):
        values = [to_float(row_lookup[mode][key]) for mode in mode_order]
        bars = ax.bar(np.arange(len(mode_order)), values, color=["#4C78A8", "#F58518", "#54A24B"])
        ax.set_xticks(np.arange(len(mode_order)))
        ax.set_xticklabels(mode_labels, rotation=15, ha="right")
        ax.set_title(label)
        ax.grid(axis="y", alpha=0.25)
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2.0, bar.get_height(), f"{value:.2f}", ha="center", va="bottom", fontsize=8)
    fig.suptitle("Direct biological reinterpretation of the TOPAS-derived Yang-style photon benchmark", fontsize=13)
    fig.savefig(out_file, dpi=int(dpi))
    plt.close(fig)


def plot_phase30_repeat_uncertainty(
    out_file: Path,
    repeat_rows: Sequence[Mapping[str, str]],
    repeat_bands: Sequence[Mapping[str, str]],
    *,
    dpi: int,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 5.2), constrained_layout=True)
    seeds = sorted({int(to_float(row["seed"])) for row in repeat_rows})
    colors = plt.cm.Set2(np.linspace(0.0, 1.0, len(seeds)))
    metrics = [
        ("pvdr", "PVDR"),
        ("spill_shell_0_5_mean_gy", "Peri-GTV 0-5 mm"),
        ("parotid_r_mean_gy", "Parotid R mean"),
    ]
    for ax, (key, title) in zip(axes[:2], metrics[:2]):
        for color, seed in zip(colors, seeds):
            seed_rows = [row for row in repeat_rows if int(to_float(row["seed"])) == seed]
            seed_rows.sort(key=lambda row: to_float(row["history_scale"]))
            x = [to_float(row["history_scale"]) for row in seed_rows]
            y = [to_float(row[key]) for row in seed_rows]
            ax.plot(x, y, marker="o", linewidth=2.0, color=color, label=f"Seed {seed}")
        ax.set_xlabel("History scale")
        ax.set_title(title)
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("Value")
    axes[0].legend(frameon=False, fontsize=8)

    cv_rows = [row for row in repeat_bands if str(row["metric"]) in [m[0] for m in metrics]]
    cv_labels = [next(label for key, label in metrics if key == str(row["metric"])) for row in cv_rows]
    cv_values = [to_float(row["coefficient_of_variation_pct"]) for row in cv_rows]
    axes[2].bar(np.arange(len(cv_rows)), cv_values, color=["#4C78A8", "#F58518", "#E45756"])
    axes[2].set_xticks(np.arange(len(cv_rows)))
    axes[2].set_xticklabels(cv_labels, rotation=20, ha="right")
    axes[2].set_ylabel("Coefficient of variation (%)")
    axes[2].set_title("Repeated TOPAS uncertainty bands")
    axes[2].grid(axis="y", alpha=0.25)

    fig.savefig(out_file, dpi=int(dpi))
    plt.close(fig)


def build_manuscript_table(master_rows: Sequence[Mapping[str, str]]) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    selected = [
        ("phase25_safe_core_mean", "physical_only"),
        ("phase25_safe_core_mean", "bystander_no_sink"),
        ("phase25_safe_core_mean", "bystander_with_sink"),
        ("phase30_yang_topas_photon", "physical_only"),
        ("phase30_yang_topas_photon", "bystander_no_sink"),
        ("phase30_yang_topas_photon", "bystander_with_sink"),
    ]
    for dataset, mode in selected:
        row = find_row(master_rows, "dataset", dataset)
        candidate_rows = [r for r in master_rows if str(r["dataset"]) == dataset and str(r["mode"]) == mode]
        row = candidate_rows[0]
        rows.append(
            {
                "dataset": str(row["dataset"]),
                "mode": str(row["mode"]),
                "mode_label": str(row["mode_label"]),
                "target_label": str(row["target_label"]),
                "peripheral_target_d95_gy": to_float(row["peripheral_target_d95_gy"]),
                "pvdr": to_float(row["pvdr"]),
                "spill_shell_0_5_mean_gy": to_float(row["spill_shell_0_5_mean_gy"]),
                "spill_shell_5_15_mean_gy": to_float(row["spill_shell_5_15_mean_gy"]),
                "cord_d2_gy": to_float(row["cord_d2_gy"]),
                "brainstem_d2_gy": to_float(row["brainstem_d2_gy"]),
                "parotid_r_mean_gy": to_float(row["parotid_r_mean_gy"]),
                "mean_gammah2ax_peak": to_float(row["mean_gammah2ax_peak"]),
                "mean_gammah2ax_valley": to_float(row["mean_gammah2ax_valley"]),
                "mean_tunel_peak": to_float(row["mean_tunel_peak"]),
                "mean_tunel_valley": to_float(row["mean_tunel_valley"]),
                "cytokine_peak_roi_auc": to_float(row["cytokine_peak_roi_auc"]),
                "cytokine_valley_roi_auc": to_float(row["cytokine_valley_roi_auc"]),
                "immune_scalar": to_float(row["immune_scalar"]),
            }
        )
    return rows


def build_abstract(
    benchmark_stats: Mapping[str, float],
    rank_effect_rows: Sequence[Mapping[str, str]],
    sink_effect_rows: Sequence[Mapping[str, str]],
    phase30_rows: Sequence[Mapping[str, str]],
    repeat_band_rows: Sequence[Mapping[str, str]],
) -> str:
    rank_row = find_row(rank_effect_rows, "metric", "with_sink_vs_physical_shift")
    sink_spill = next(row for row in sink_effect_rows if str(row["metric"]) == "spill_shell_0_5_mean")
    sink_parotid = next(row for row in sink_effect_rows if str(row["metric"]) == "parotid_r_mean")
    sink_pvdr = next(row for row in sink_effect_rows if str(row["metric"]) == "pvdr")
    phase30_phys = find_row(phase30_rows, "mode", "physical_only")
    phase30_sink = find_row(phase30_rows, "mode", "bystander_with_sink")
    repeat_pvdr = next(row for row in repeat_band_rows if str(row["metric"]) == "pvdr")
    repeat_spill = next(row for row in repeat_band_rows if str(row["metric"]) == "spill_shell_0_5_mean_gy")
    return (
        "Purpose: To convert a protocol-constrained lattice radiotherapy modelling workflow into a manuscript-ready "
        "biological risk-analysis package anchored to synthetic head-and-neck benchmarks and a TOPAS-derived sinonasal photon case.\n\n"
        f"Methods: We froze a supplementary library of {int(benchmark_stats['n_templates'])} synthetic H&N Yang-like lattice templates "
        f"(estimated GTV range {benchmark_stats['gtv_min']:.0f}-{benchmark_stats['gtv_max']:.0f} cc; "
        f"{benchmark_stats['vertex_min']:.0f}-{benchmark_stats['vertex_max']:.0f} vertices; recommended diameters "
        f"{benchmark_stats['diam_min']:.1f}-{benchmark_stats['diam_max']:.1f} cm; mean spacing {benchmark_stats['spacing_mean']:.2f} cm). "
        "We then summarized a fixed safe-core plan library under physical-only scoring, bystander signalling without vascular sink uptake, "
        "and bystander signalling with vascular sink uptake; derived manuscript endpoint and assay tables; and repeated the Phase 30 TOPAS "
        "Yang-style photon benchmark across multiple seeds and history levels before applying the biology model directly to that TOPAS dose.\n\n"
        f"Results: Biological modelling changed plan ordering in the safe-core library, with a mean absolute with-sink-versus-physical "
        f"rank shift of {to_float(rank_row['mean_absolute_shift']):.2f} (95% bootstrap CI {to_float(rank_row['mean_absolute_shift_ci_lower']):.2f}-"
        f"{to_float(rank_row['mean_absolute_shift_ci_upper']):.2f}). Adding vascular sink uptake reduced peri-GTV 0-5 mm burden by "
        f"{abs(to_float(sink_spill['mean_with_sink_minus_no_sink'])):.3f} Gy and parotid mean by {abs(to_float(sink_parotid['mean_with_sink_minus_no_sink'])):.3f} Gy, "
        f"while increasing PVDR by {to_float(sink_pvdr['mean_with_sink_minus_no_sink']):.3f}. In the TOPAS-derived Yang-style photon benchmark, "
        f"physical PVDR was {to_float(phase30_phys['pvdr']):.3f} and with-sink biological PVDR was {to_float(phase30_sink['pvdr']):.3f}, "
        f"with with-sink cytokine valley AUC {to_float(phase30_sink['cytokine_valley_roi_auc']):.1f}. Repeated TOPAS runs yielded a PVDR coefficient "
        f"of variation of {to_float(repeat_pvdr['coefficient_of_variation_pct']):.2f}% and a peri-GTV 0-5 mm spill coefficient of variation of "
        f"{to_float(repeat_spill['coefficient_of_variation_pct']):.2f}%.\n\n"
        "Conclusion: The resulting manuscript package supports a benchmark-anchored, hypothesis-generating biological risk-analysis framework "
        "for lattice RT. It does not establish TPS-equivalent optimization, but it does show that protocol-constrained lattice plans can be "
        "reinterpreted biologically in ways not captured by physical dose metrics alone."
    )


def build_results_text(
    benchmark_rows: Sequence[Mapping[str, str]],
    rank_effect_rows: Sequence[Mapping[str, str]],
    rank_robustness_rows: Sequence[Mapping[str, str]],
    sink_effect_rows: Sequence[Mapping[str, str]],
    phase30_rows: Sequence[Mapping[str, str]],
    repeat_band_rows: Sequence[Mapping[str, str]],
) -> str:
    stats = benchmark_summary(benchmark_rows)
    rank_row = find_row(rank_effect_rows, "metric", "with_sink_vs_physical_shift")
    no_sink_row = find_row(rank_effect_rows, "metric", "no_sink_vs_physical_shift")
    sink_spill = next(row for row in sink_effect_rows if str(row["metric"]) == "spill_shell_0_5_mean")
    sink_spill_2 = next(row for row in sink_effect_rows if str(row["metric"]) == "spill_shell_5_15_mean")
    sink_parotid = next(row for row in sink_effect_rows if str(row["metric"]) == "parotid_r_mean")
    sink_pvdr = next(row for row in sink_effect_rows if str(row["metric"]) == "pvdr")
    sink_cyto = next(row for row in sink_effect_rows if str(row["metric"]) == "cytokine_valley_roi_auc")
    phase30_phys = find_row(phase30_rows, "mode", "physical_only")
    phase30_no_sink = find_row(phase30_rows, "mode", "bystander_no_sink")
    phase30_sink = find_row(phase30_rows, "mode", "bystander_with_sink")
    repeat_pvdr = next(row for row in repeat_band_rows if str(row["metric"]) == "pvdr")
    repeat_spill = next(row for row in repeat_band_rows if str(row["metric"]) == "spill_shell_0_5_mean_gy")
    repeat_parotid = next(row for row in repeat_band_rows if str(row["metric"]) == "parotid_r_mean_gy")
    top_phys = find_row(rank_robustness_rows, "mode", "physical_only")
    top_sink_plan5 = next(row for row in rank_robustness_rows if str(row["plan_id"]) == "plan05" and str(row["mode"]) == "bystander_with_sink")

    lines = [
        "## Results",
        "",
        "### Synthetic benchmark library",
        "",
        f"The supplementary benchmark library comprised {int(stats['n_templates'])} synthetic head-and-neck lattice templates spanning an estimated GTV range of "
        f"{stats['gtv_min']:.0f}-{stats['gtv_max']:.0f} cc (median {stats['gtv_median']:.0f} cc). Recommended lattice complexity ranged from "
        f"{stats['vertex_min']:.0f} to {stats['vertex_max']:.0f} vertices with recommended vertex diameters of {stats['diam_min']:.1f}-{stats['diam_max']:.1f} cm. "
        f"Mean centre-to-centre spacing remained close to the Yang-style geometric prior, averaging {stats['spacing_mean']:.2f} cm "
        f"(range {stats['spacing_min']:.2f}-{stats['spacing_max']:.2f} cm). Together, these templates provide a structured geometry supplement for bulky sinonasal, oral cavity, "
        "oropharyngeal, deep-space, and nodal lattice scenarios without claiming patient-level anatomy.",
        "",
        "### Biological reinterpretation of the safe-core library",
        "",
        f"In the fixed safe-core library, biological modelling altered plan ranking relative to physical-only scoring. The mean absolute no-sink-versus-physical "
        f"rank shift was {to_float(no_sink_row['mean_absolute_shift']):.2f}, and the mean absolute with-sink-versus-physical rank shift was "
        f"{to_float(rank_row['mean_absolute_shift']):.2f} (95% bootstrap CI {to_float(rank_row['mean_absolute_shift_ci_lower']):.2f}-"
        f"{to_float(rank_row['mean_absolute_shift_ci_upper']):.2f}). These rank shifts indicate that the model changes comparative plan interpretation rather than simply rescaling all plans uniformly.",
        "",
        "Rank robustness under combined Monte Carlo and biology uncertainty was modest rather than dominant. Across modes, rank-retention probabilities typically fell between "
        f"{min(to_float(row['rank_retention_probability']) for row in rank_robustness_rows):.2f} and "
        f"{max(to_float(row['rank_retention_probability']) for row in rank_robustness_rows):.2f}. "
        f"For the nominally best physical plan, top-rank retention remained {to_float(top_phys['top_rank_probability']):.2f}, while the nominally best with-sink plan "
        f"(plan05) retained top rank with probability {to_float(top_sink_plan5['top_rank_probability']):.2f}. This supports a hypothesis-generating interpretation rather than a claim of stable winner-take-all optimization.",
        "",
        "### Vascular sink ablation and assay-proxy shifts",
        "",
        f"Adding anatomical vascular sink uptake consistently shifted off-target burden downward. Relative to the no-sink model, the with-sink model reduced peri-GTV 0-5 mm mean dose by "
        f"{abs(to_float(sink_spill['mean_with_sink_minus_no_sink'])):.3f} Gy (95% CI {abs(to_float(sink_spill['ci_upper'])):.3f}-{abs(to_float(sink_spill['ci_lower'])):.3f} Gy) "
        f"and peri-GTV 5-15 mm mean dose by {abs(to_float(sink_spill_2['mean_with_sink_minus_no_sink'])):.3f} Gy. Parotid mean fell by "
        f"{abs(to_float(sink_parotid['mean_with_sink_minus_no_sink'])):.3f} Gy, while PVDR increased by {to_float(sink_pvdr['mean_with_sink_minus_no_sink']):.3f}. "
        f"At the assay-proxy level, the largest mean change was seen for cytokine valley AUC ({abs(to_float(sink_cyto['mean_with_sink_minus_no_sink'])):.1f} a.u. reduction), "
        "whereas peak gammaH2AX changed minimally. This pattern is consistent with vascular sink uptake acting mainly on diffusible non-local signalling rather than on direct peak injury.",
        "",
        "### Direct TOPAS-derived benchmark reinterpretation",
        "",
        f"When the biology model was applied directly to the TOPAS-derived Yang-style photon case, physical PVDR was {to_float(phase30_phys['pvdr']):.3f}. "
        f"This fell to {to_float(phase30_no_sink['pvdr']):.3f} without sink uptake and to {to_float(phase30_sink['pvdr']):.3f} with sink uptake, indicating biological valley fill-in even after Monte Carlo transport. "
        f"Peri-GTV 0-5 mm mean increased from {to_float(phase30_phys['spill_shell_0_5_mean_gy']):.2f} Gy physically to {to_float(phase30_sink['spill_shell_0_5_mean_gy']):.2f} Gy biologically, "
        f"and parotid mean increased from {to_float(phase30_phys['parotid_r_mean_gy']):.2f} Gy to {to_float(phase30_sink['parotid_r_mean_gy']):.2f} Gy(eq). "
        f"With-sink cytokine valley AUC reached {to_float(phase30_sink['cytokine_valley_roi_auc']):.1f}, again supporting a non-local burden outside the physical hot spots.",
        "",
        "### Repeated TOPAS uncertainty bands",
        "",
        f"Nine repeated Phase 30 TOPAS runs were generated across three random seeds and three history levels. Peripheral target D95 remained numerically stable "
        f"(coefficient of variation {to_float(next(row for row in repeat_band_rows if str(row['metric']) == 'peripheral_target_d95_gy')['coefficient_of_variation_pct']):.4f}%), "
        f"whereas PVDR showed a coefficient of variation of {to_float(repeat_pvdr['coefficient_of_variation_pct']):.2f}%. "
        f"Peri-GTV 0-5 mm spill varied by {to_float(repeat_spill['coefficient_of_variation_pct']):.2f}% and parotid mean by {to_float(repeat_parotid['coefficient_of_variation_pct']):.2f}%. "
        "These repeated-run bands provide a practical Monte Carlo noise scale against which downstream biological reinterpretation can be judged.",
    ]
    return "\n".join(lines) + "\n"


def build_figure_legends() -> str:
    return (
        "**Figure 1.** Synthetic head-and-neck benchmark lattice library. Left: estimated GTV versus mean centre-to-centre spacing for the 10 frozen synthetic benchmark templates; point colour indicates vertex count and point size scales with recommended vertex diameter. Right: template size distribution with overlaid lattice complexity.\n\n"
        "**Figure 2.** Rank reinterpretation and uncertainty in the fixed safe-core plan library. Left: slopegraph of plan ranks across physical-only, bystander without vascular sink uptake, and bystander with anatomical vascular sink uptake. Right: heatmap of nominal-rank retention probability under combined Monte Carlo and biology uncertainty.\n\n"
        "**Figure 3.** Mean with-sink minus no-sink effect sizes with bootstrap confidence intervals. Left: primary endpoint shifts. Right: assay-proxy shifts. Negative values indicate lower burden after adding anatomical vascular sink uptake.\n\n"
        "**Figure 4.** Direct biological reinterpretation of the Phase 30 TOPAS-derived Yang-style photon benchmark. Bars show physical-only, no-sink, and with-sink values for physical/biological selectivity, spill, OAR, and assay-proxy endpoints.\n\n"
        "**Figure 5.** Repeated TOPAS uncertainty bands for the Phase 30 photon benchmark. Left and middle: metric trajectories across history scale for each seed. Right: coefficient of variation for selected benchmark metrics."
    )


def build_manuscript_draft(abstract_text: str, results_text: str, discussion_text: str, legends_text: str) -> str:
    results_body = results_text.strip()
    if results_body.startswith("## Results"):
        results_body = results_body[len("## Results") :].lstrip()
    discussion_body = discussion_text.strip()
    if discussion_body.startswith("# "):
        discussion_lines = discussion_body.splitlines()
        discussion_body = "\n".join(discussion_lines[1:]).lstrip()
    if not discussion_body.startswith("## Discussion"):
        discussion_body = "## Discussion\n\n" + discussion_body
    return (
        "# Phase 31 Manuscript Draft\n\n"
        "## Abstract\n\n"
        f"{abstract_text.strip()}\n\n"
        "## Results\n\n"
        f"{results_body}\n\n"
        f"{discussion_body}\n\n"
        "## Figure Legends\n\n"
        f"{legends_text.strip()}\n"
    )


def write_markdown(path: Path, text: str) -> None:
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    phase31_root = args.phase31_root.resolve()
    out_root = args.out_root.resolve() if args.out_root is not None else phase31_root / "manuscript_assets"
    figures_dir = out_root / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    data = load_inputs(phase31_root)
    benchmark_stats = benchmark_summary(data["benchmark"])

    plot_benchmark_library(figures_dir / "figure1_benchmark_template_library.png", data["benchmark"], dpi=int(args.dpi))
    plot_rank_reinterpretation(
        figures_dir / "figure2_rank_reinterpretation_and_robustness.png",
        data["phase26_rank_shift"],
        data["rank_robustness"],
        dpi=int(args.dpi),
    )
    plot_sink_effects(figures_dir / "figure3_vascular_sink_effect_sizes.png", data["sink_effects"], dpi=int(args.dpi))
    plot_phase30_topas_reinterpretation(
        figures_dir / "figure4_phase30_topas_biological_reinterpretation.png",
        data["phase30_bio"],
        dpi=int(args.dpi),
    )
    plot_phase30_repeat_uncertainty(
        figures_dir / "figure5_phase30_repeat_uncertainty.png",
        data["phase30_repeat_summary"],
        data["phase30_repeat_bands"],
        dpi=int(args.dpi),
    )

    manuscript_table = build_manuscript_table(data["master"])
    write_csv(out_root / "phase31_manuscript_table_locked_endpoints.csv", manuscript_table)

    abstract_text = build_abstract(
        benchmark_stats,
        data["rank_effects"],
        data["sink_effects"],
        data["phase30_bio"],
        data["phase30_repeat_bands"],
    )
    results_text = build_results_text(
        data["benchmark"],
        data["rank_effects"],
        data["rank_robustness"],
        data["sink_effects"],
        data["phase30_bio"],
        data["phase30_repeat_bands"],
    )
    legends_text = build_figure_legends()
    manuscript_draft_text = build_manuscript_draft(
        abstract_text,
        results_text,
        str(data["discussion_draft"]),
        legends_text,
    )

    write_markdown(out_root / "phase31_manuscript_abstract.md", abstract_text)
    write_markdown(out_root / "phase31_manuscript_results.md", results_text)
    write_markdown(out_root / "phase31_manuscript_figure_legends.md", legends_text)
    write_markdown(out_root / "phase31_manuscript_draft.md", manuscript_draft_text)

    summary = {
        "phase31_root": str(phase31_root),
        "output_root": str(out_root),
        "figures": {
            "figure1_benchmark_template_library": str(figures_dir / "figure1_benchmark_template_library.png"),
            "figure2_rank_reinterpretation_and_robustness": str(figures_dir / "figure2_rank_reinterpretation_and_robustness.png"),
            "figure3_vascular_sink_effect_sizes": str(figures_dir / "figure3_vascular_sink_effect_sizes.png"),
            "figure4_phase30_topas_biological_reinterpretation": str(figures_dir / "figure4_phase30_topas_biological_reinterpretation.png"),
            "figure5_phase30_repeat_uncertainty": str(figures_dir / "figure5_phase30_repeat_uncertainty.png"),
        },
        "text_outputs": {
            "abstract": str(out_root / "phase31_manuscript_abstract.md"),
            "results": str(out_root / "phase31_manuscript_results.md"),
            "manuscript_draft": str(out_root / "phase31_manuscript_draft.md"),
            "figure_legends": str(out_root / "phase31_manuscript_figure_legends.md"),
            "locked_table": str(out_root / "phase31_manuscript_table_locked_endpoints.csv"),
        },
    }
    (out_root / "phase31_manuscript_assets_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("=== PHASE 31 MANUSCRIPT ASSETS COMPLETE ===")
    print(f"Output root: {out_root}")
    print(f"Abstract: {out_root / 'phase31_manuscript_abstract.md'}")
    print(f"Results: {out_root / 'phase31_manuscript_results.md'}")
    print(f"Draft: {out_root / 'phase31_manuscript_draft.md'}")
    print(f"Figure legends: {out_root / 'phase31_manuscript_figure_legends.md'}")
    print(f"Locked table: {out_root / 'phase31_manuscript_table_locked_endpoints.csv'}")
    print(f"Figures dir: {figures_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
