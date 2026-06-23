#!/usr/bin/env python3
"""Build a publication-style figure atlas covering the biological risk-analysis workflow from Phase 25 to Phase 37."""

from __future__ import annotations

import argparse
import shutil
import textwrap
from pathlib import Path
from typing import Dict, List, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


plt.style.use("seaborn-v0_8-whitegrid")


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-root",
        type=Path,
        default=root / "runs" / "phase25_to_phase37_publication_figure_atlas",
    )
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def wrap(text: str, width: int = 30) -> str:
    return "\n".join(textwrap.wrap(str(text), width=width))


def add_manifest_row(
    rows: List[Dict[str, object]],
    *,
    figure_id: str,
    phase: str,
    title: str,
    file_path: Path,
    what_it_shows: str,
    caption: str,
) -> None:
    rows.append(
        {
            "figure_id": figure_id,
            "phase": phase,
            "title": title,
            "file_path": str(file_path.resolve()),
            "what_it_shows": what_it_shows,
            "caption": caption,
        }
    )


def copy_existing(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def compose_phase28_benchmark(geometry_src: Path, bio_src: Path, dst: Path, *, dpi: int) -> None:
    img_a = plt.imread(str(geometry_src))
    img_b = plt.imread(str(bio_src))
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.5), constrained_layout=True)
    for ax, img, panel_title in zip(
        axes,
        (img_a, img_b),
        ("A. Yang-style benchmark geometry", "B. Biological reinterpretation"),
    ):
        ax.imshow(img)
        ax.axis("off")
        ax.set_title(panel_title, fontsize=12, fontweight="bold")
    fig.suptitle("Phase 28: benchmarked geometry and biological interpretation", fontsize=14, fontweight="bold")
    fig.savefig(dst, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_phase30_decomposition(table_path: Path, dst: Path, *, dpi: int) -> Dict[str, float]:
    df = pd.read_csv(table_path).set_index("component")
    order = ["base_processed_scaled", "spot_processed_scaled", "combined_processed_scaled"]
    labels = ["Base field", "Vertex boost", "Combined"]
    metrics = [
        ("ptv_d95_gy", "PTV D95 (Gy)", 3.5),
        ("peak_mean_gy", "Peak mean (Gy)", 15.0),
        ("valley_mean_gy", "Valley mean (Gy)", None),
        ("pvdr", "PVDR", None),
        ("spill_shell_0_5_mean_gy", "Peri-GTV 0-5 mm mean (Gy)", None),
        ("brainstem_d2_gy", "Brainstem D2 (Gy)", None),
    ]
    colors = ["#7f8c8d", "#d35400", "#1f77b4"]
    fig, axes = plt.subplots(2, 3, figsize=(14, 8.5), constrained_layout=True)
    for ax, (col, title, target) in zip(axes.flat, metrics):
        values = [float(df.loc[key, col]) for key in order]
        bars = ax.bar(labels, values, color=colors, edgecolor="black", linewidth=0.6)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.tick_params(axis="x", labelrotation=20)
        if target is not None:
            ax.axhline(target, color="#c0392b", linestyle="--", linewidth=1.2, label="Target")
            ax.legend(frameon=False, fontsize=8, loc="best")
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2.0, value, f"{value:.2f}", ha="center", va="bottom", fontsize=8)
    fig.suptitle("Phase 30: TOPAS-derived photon lattice delivery decomposition", fontsize=14, fontweight="bold")
    fig.savefig(dst, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    combined = df.loc["combined_processed_scaled"]
    return {
        "ptv_d95_gy": float(combined["ptv_d95_gy"]),
        "peak_mean_gy": float(combined["peak_mean_gy"]),
        "pvdr": float(combined["pvdr"]),
        "brainstem_d2_gy": float(combined["brainstem_d2_gy"]),
    }


def plot_phase32_montage(manifest_path: Path, dst: Path, *, dpi: int) -> Dict[str, float]:
    df = pd.read_csv(manifest_path).sort_values("template_id")
    fig, axes = plt.subplots(2, 5, figsize=(18, 8.5), constrained_layout=True)
    for ax, (_, row) in zip(axes.flat, df.iterrows()):
        img = plt.imread(str(row["geometry_figure"]))
        ax.imshow(img)
        ax.axis("off")
        title = f"{row['template_id']}: {wrap(row['label'], 20)}"
        subtitle = f"GTV {row['instantiated_gtv_cc']:.1f} cc | vertices {int(row['kept_vertex_count'])}/{int(row['proposed_vertex_count'])}"
        ax.set_title(f"{title}\n{subtitle}", fontsize=8, fontweight="bold")
    fig.suptitle("Phase 32: site-specific synthetic H&N benchmark cohort", fontsize=14, fontweight="bold")
    fig.savefig(dst, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return {
        "n_cases": float(len(df)),
        "median_gtv_cc": float(df["instantiated_gtv_cc"].median()),
        "kept_vertices": float(df["kept_vertex_count"].sum()),
    }


def plot_phase33_physical_cohort(table_path: Path, dst: Path, *, dpi: int) -> Dict[str, float]:
    df = pd.read_csv(table_path)
    df = df[df["component"] == "combined_processed_scaled"].copy()
    metrics = [
        ("ptv_d95_gy", "PTV D95 (Gy)", 3.5),
        ("pvdr", "PVDR", None),
        ("spill_shell_0_5_mean_gy", "Peri-GTV 0-5 mm mean (Gy)", None),
        ("brainstem_d2_gy", "Brainstem D2 (Gy)", None),
        ("parotid_r_mean_gy", "Parotid-R mean (Gy)", None),
    ]
    fig, axes = plt.subplots(1, 5, figsize=(17, 5.5), constrained_layout=True)
    rng = np.random.default_rng(33)
    for ax, (col, title, target) in zip(axes, metrics):
        values = df[col].to_numpy(dtype=float)
        jitter = rng.normal(loc=1.0, scale=0.04, size=values.size)
        ax.boxplot(values, positions=[1.0], widths=0.25, patch_artist=True, boxprops={"facecolor": "#dfe6e9"})
        ax.scatter(jitter, values, color="#1f77b4", edgecolor="black", linewidth=0.5, s=38, zorder=3)
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.set_xlim(0.65, 1.35)
        ax.set_xticks([])
        if target is not None:
            ax.axhline(target, color="#c0392b", linestyle="--", linewidth=1.2)
    fig.suptitle("Phase 33: physical TOPAS cohort summary across the 10 site-specific cases", fontsize=14, fontweight="bold")
    fig.savefig(dst, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return {
        "n_cases": float(len(df)),
        "median_pvdr": float(df["pvdr"].median()),
        "mean_ptv_d95_gy": float(df["ptv_d95_gy"].mean()),
    }


def plot_phase34_biology_reinterpretation(endpoint_path: Path, rank_path: Path, dst: Path, *, dpi: int) -> Dict[str, float]:
    df = pd.read_csv(endpoint_path)
    rank_df = pd.read_csv(rank_path)
    metric_info = [
        ("pvdr", "PVDR"),
        ("spill_shell_0_5_mean", "Peri-GTV 0-5"),
        ("brainstem_d2", "Brainstem D2"),
        ("parotid_r_mean", "Parotid-R"),
        ("ptv_d95", "PTV D95"),
    ]
    mode_order = ["physical_only", "bystander_no_sink", "bystander_with_sink"]
    mode_labels = ["Physical", "No sink", "With sink"]
    mean_table = (
        df.groupby("mode")[[col for col, _ in metric_info]].mean().reindex(mode_order)
    )
    physical_row = mean_table.loc["physical_only"]
    ratio_table = mean_table.divide(physical_row.replace(0.0, np.nan))

    fig = plt.figure(figsize=(15, 7), constrained_layout=True)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.05, 0.95])
    ax0 = fig.add_subplot(gs[0, 0])
    heat = ax0.imshow(ratio_table.to_numpy(dtype=float), cmap="coolwarm", aspect="auto", vmin=0.7, vmax=3.2)
    ax0.set_xticks(np.arange(len(metric_info)))
    ax0.set_xticklabels([label for _, label in metric_info], rotation=25, ha="right")
    ax0.set_yticks(np.arange(len(mode_labels)))
    ax0.set_yticklabels(mode_labels)
    ax0.set_title("Cohort-mean endpoint ratio relative to physical-only", fontsize=11, fontweight="bold")
    for i in range(ratio_table.shape[0]):
        for j in range(ratio_table.shape[1]):
            actual = mean_table.iloc[i, j]
            ratio = ratio_table.iloc[i, j]
            ax0.text(j, i, f"{actual:.2f}\n({ratio:.2f}x)", ha="center", va="center", fontsize=8)
    cbar = fig.colorbar(heat, ax=ax0, fraction=0.046, pad=0.04)
    cbar.set_label("Ratio vs physical-only")

    ax1 = fig.add_subplot(gs[0, 1])
    x = np.array([0, 1, 2], dtype=float)
    for _, row in rank_df.sort_values("physical_rank").iterrows():
        y = np.array([row["physical_rank"], row["no_sink_rank"], row["with_sink_rank"]], dtype=float)
        ax1.plot(x, y, marker="o", linewidth=1.6, alpha=0.8)
        ax1.text(x[-1] + 0.05, y[-1], str(row["plan_id"]), fontsize=8, va="center")
    ax1.set_xticks(x)
    ax1.set_xticklabels(["Physical", "No sink", "With sink"])
    ax1.invert_yaxis()
    ax1.set_ylabel("Risk rank (1 = least risky)")
    ax1.set_title("Case-wise rank reinterpretation across biological modes", fontsize=11, fontweight="bold")
    ax1.set_xlim(-0.1, 2.45)

    fig.suptitle("Phase 34: biological reinterpretation of the 10-case TOPAS cohort", fontsize=14, fontweight="bold")
    fig.savefig(dst, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    changed = int(np.sum(rank_df["with_sink_vs_physical_shift"] != 0))
    return {
        "changed_rank_cases": float(changed),
        "n_cases": float(len(rank_df)),
    }


def plot_phase35_uncertainty(noise_path: Path, rank_path: Path, dst: Path, *, dpi: int) -> Dict[str, float]:
    noise_df = pd.read_csv(noise_path)
    rank_df = pd.read_csv(rank_path)
    endpoint_summary = (
        noise_df.groupby("label")["exceeds_95pct_noise_band"].mean().sort_values()
    )
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), constrained_layout=True)
    axes[0].barh(endpoint_summary.index, endpoint_summary.values, color="#1f77b4")
    axes[0].set_xlim(0, 1)
    axes[0].set_xlabel("Fraction of case-endpoints above 95% noise band")
    axes[0].set_title("Endpoint-level robustness of with-sink vs no-sink deltas", fontsize=11, fontweight="bold")

    rank_df = rank_df.sort_values("plan_id")
    axes[1].bar(rank_df["plan_id"], rank_df["probability_same_direction_as_nominal"], color="#16a085")
    axes[1].axhline(0.8, color="#c0392b", linestyle="--", linewidth=1.2, label="0.8 heuristic")
    axes[1].set_ylim(0, 1)
    axes[1].set_ylabel("Probability of preserving nominal shift direction")
    axes[1].set_title("Rank-direction stability across repeated subset runs", fontsize=11, fontweight="bold")
    axes[1].legend(frameon=False, fontsize=8, loc="upper right")

    fig.suptitle("Phase 35: uncertainty overlay for sink effects on the repeated 6-case subset", fontsize=14, fontweight="bold")
    fig.savefig(dst, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return {
        "n_exceed95": float(noise_df["exceeds_95pct_noise_band"].sum()),
        "n_rows": float(len(noise_df)),
    }


def plot_phase36_falsification(summary_path: Path, noise_path: Path, dst: Path, *, dpi: int) -> Dict[str, float]:
    summary_df = pd.read_csv(summary_path)
    noise_df = pd.read_csv(noise_path)
    comparators = ["no_sink", "mirror_lr_sink", "uniform_body_sink_mass_matched"]
    summary_stats = (
        summary_df[summary_df["comparator_id"].isin(comparators)]
        .groupby("comparator_label")["ci_excludes_zero"]
        .sum()
        .sort_values(ascending=False)
    )
    noise_stats = (
        noise_df[noise_df["comparator_id"].isin(comparators)]
        .groupby("comparator_label")["exceeds_95pct_noise_band"]
        .mean()
        .sort_values(ascending=False)
    )
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.5), constrained_layout=True)
    axes[0].bar(summary_stats.index, summary_stats.values, color=["#1f77b4", "#95a5a6", "#d35400"])
    axes[0].set_ylabel("Endpoint summaries with CI excluding zero (out of 8)")
    axes[0].set_title("Baseline falsification strength", fontsize=11, fontweight="bold")
    axes[0].tick_params(axis="x", rotation=22)
    axes[1].bar(noise_stats.index, noise_stats.values, color=["#1f77b4", "#95a5a6", "#d35400"])
    axes[1].set_ylim(0, 1)
    axes[1].set_ylabel("Fraction of repeated endpoint deltas above 95% noise band")
    axes[1].set_title("Uncertainty-qualified comparator separation", fontsize=11, fontweight="bold")
    axes[1].tick_params(axis="x", rotation=22)
    fig.suptitle("Phase 36: first sink falsification controls", fontsize=14, fontweight="bold")
    fig.savefig(dst, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return {
        "mirror_hits": float(summary_stats.get("Bystander, mirrored left-right vessel sink", 0.0)),
        "uniform_frac": float(noise_stats.get("Bystander, uniform body sink (mass matched)", 0.0)),
    }


def plot_phase37_falsification(summary_path: Path, noise_path: Path, rank_path: Path, dst: Path, *, dpi: int) -> Dict[str, float]:
    summary_df = pd.read_csv(summary_path)
    noise_df = pd.read_csv(noise_path)
    rank_df = pd.read_csv(rank_path)
    comparator_order = [
        "no_sink",
        "uniform_body_sink_mass_matched",
        "local_dropout_sink_20mm",
        "si_flip_sink",
        "ap_flip_sink",
        "randomized_displacement_sink",
    ]
    label_lookup = {row["comparator_id"]: row["comparator_label"] for _, row in summary_df.drop_duplicates("comparator_id").iterrows()}
    labels = [wrap(label_lookup[cid].replace("Bystander, ", ""), 18) for cid in comparator_order]
    summary_stats = (
        summary_df[summary_df["comparator_id"].isin(comparator_order)]
        .groupby("comparator_id")["ci_excludes_zero"]
        .sum()
        .reindex(comparator_order)
    )
    noise_stats = (
        noise_df[noise_df["comparator_id"].isin(comparator_order)]
        .groupby("comparator_id")["exceeds_95pct_noise_band"]
        .mean()
        .reindex(comparator_order)
    )
    rank_stats = (
        rank_df[rank_df["comparator_id"].isin(comparator_order)]
        .groupby("comparator_id")["noise_qualified_rank_change"]
        .sum()
        .reindex(comparator_order)
    )
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), constrained_layout=True)
    colors = ["#34495e", "#d35400", "#16a085", "#8e44ad", "#2980b9", "#7f8c8d"]
    axes[0].bar(labels, summary_stats.values, color=colors)
    axes[0].set_ylabel("Endpoint summaries with CI excluding zero (out of 8)")
    axes[0].set_title("Baseline separation from the true sink", fontsize=11, fontweight="bold")
    axes[0].tick_params(axis="x", rotation=25)
    axes[1].bar(labels, noise_stats.values, color=colors)
    axes[1].set_ylim(0, 1)
    axes[1].set_ylabel("Fraction above 95% noise band")
    axes[1].set_title("Repeated-subset endpoint robustness", fontsize=11, fontweight="bold")
    axes[1].tick_params(axis="x", rotation=25)
    axes[2].bar(labels, rank_stats.values, color=colors)
    axes[2].set_ylabel("Noise-qualified rank changes (out of 6 cases)")
    axes[2].set_title("Rank-level falsification impact", fontsize=11, fontweight="bold")
    axes[2].tick_params(axis="x", rotation=25)
    fig.suptitle("Phase 37: stronger spatial falsification controls for the vascular sink", fontsize=14, fontweight="bold")
    fig.savefig(dst, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return {
        "n_exceed95": float(noise_df["exceeds_95pct_noise_band"].sum()),
        "n_rows": float(len(noise_df)),
        "n_rank_changes": float(rank_df["noise_qualified_rank_change"].sum()),
    }


def write_manifest_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    pd.DataFrame(rows).to_csv(path, index=False)


def write_markdown(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    lines = ["# Phase 25-37 Publication Figure Atlas", ""]
    for row in rows:
        lines.extend(
            [
                f"## {row['figure_id']}. {row['title']}",
                "",
                f"Phase: `{row['phase']}`",
                "",
                f"What it shows: {row['what_it_shows']}",
                "",
                f"Caption: {row['caption']}",
                "",
                f"![{row['title']}]({row['file_path']})",
                "",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    out_root = args.out_root.resolve()
    fig_dir = out_root / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    manifest_rows: List[Dict[str, object]] = []

    # Figure 1: Phase 25
    src = root / "runs" / "linac_6mv_headneck_detailed_voxel_lattice_sfrt_phase25_safe_core_plan_library_1e5" / "figures" / "figure1_phase25_biological_endpoint_heatmap.png"
    dst = fig_dir / "figure01_phase25_safe_core_biological_heatmap.png"
    copy_existing(src, dst)
    add_manifest_row(
        manifest_rows,
        figure_id="Figure 1",
        phase="Phase 25",
        title="Safe-core biological endpoint heatmap",
        file_path=dst,
        what_it_shows="Biological risk-analysis of the fixed safe-core plan library across the locked endpoint set after the project was first reframed away from adaptive optimization.",
        caption="Heatmap summarizing how the Phase 25 biological risk-analysis model re-scored the safe-core lattice plan library. This figure introduced the core question of the later phases: whether physically similar lattice plans remain biologically distinguishable once non-local burden is included.",
    )

    # Figure 2: Phase 26
    src = root / "runs" / "linac_6mv_headneck_detailed_voxel_lattice_sfrt_phase25_safe_core_plan_library_1e5" / "phase26_vascular_sink_ablation" / "figures" / "figure3_phase26_vascular_sink_delta.png"
    dst = fig_dir / "figure02_phase26_vascular_sink_ablation.png"
    copy_existing(src, dst)
    add_manifest_row(
        manifest_rows,
        figure_id="Figure 2",
        phase="Phase 26",
        title="Vascular sink ablation deltas",
        file_path=dst,
        what_it_shows="Endpoint changes induced by adding the anatomical vascular sink term relative to the no-sink biological model.",
        caption="Ablation plot from Phase 26 showing that the vascular sink operates as a secondary modifier of biological burden rather than the dominant driver of plan reinterpretation. The direction and magnitude of the deltas motivated the later falsification analyses.",
    )

    # Figure 3: Phase 28
    dst = fig_dir / "figure03_phase28_benchmark_geometry_and_biology.png"
    compose_phase28_benchmark(
        root / "runs" / "phase28_yang2022_sinonasal_benchmark" / "figures" / "figure13_phase28_topas_density_publication_sidepane.png",
        root / "runs" / "phase28_yang2022_sinonasal_benchmark" / "figures" / "figure4_phase28_biological_predictions.png",
        dst,
        dpi=args.dpi,
    )
    add_manifest_row(
        manifest_rows,
        figure_id="Figure 3",
        phase="Phase 28",
        title="Yang-style benchmark geometry and biological interpretation",
        file_path=dst,
        what_it_shows="The synthetic Yang-style sinonasal benchmark phantom alongside the biological readout used to reinterpret the benchmarked lattice plan.",
        caption="Phase 28 anchored the workflow to a Yang-style sinonasal benchmark case. The left panel shows the benchmark phantom geometry, and the right panel shows the biological predictions demonstrating that physical peak-valley contrast can be biologically filled in by non-local effects.",
    )

    # Figure 4: Phase 30
    phase30_metrics = plot_phase30_decomposition(
        root / "runs" / "phase30_phase28_topas_true_lattice_delivery" / "phase30_physical_endpoint_table.csv",
        fig_dir / "figure04_phase30_topas_delivery_decomposition.png",
        dpi=args.dpi,
    )
    add_manifest_row(
        manifest_rows,
        figure_id="Figure 4",
        phase="Phase 30",
        title="TOPAS-derived photon lattice delivery decomposition",
        file_path=fig_dir / "figure04_phase30_topas_delivery_decomposition.png",
        what_it_shows="How the broad base field and focused vertex boost combine to produce the final TOPAS-derived photon lattice dose used as a physical anchor for the biology model.",
        caption=(
            "Phase 30 decomposed the benchmark photon delivery into a base component, a vertex-boost component, and their calibrated combination. "
            f"The final combined plan achieved PTV D95 {phase30_metrics['ptv_d95_gy']:.2f} Gy, peak mean {phase30_metrics['peak_mean_gy']:.2f} Gy, PVDR {phase30_metrics['pvdr']:.2f}, "
            f"and brainstem D2 {phase30_metrics['brainstem_d2_gy']:.2f} Gy."
        ),
    )

    # Figure 5: Phase 31
    src = root / "runs" / "phase31_publication_package" / "manuscript_assets" / "figures" / "figure2_rank_reinterpretation_and_robustness.png"
    dst = fig_dir / "figure05_phase31_rank_reinterpretation_and_robustness.png"
    copy_existing(src, dst)
    add_manifest_row(
        manifest_rows,
        figure_id="Figure 5",
        phase="Phase 31",
        title="Rank reinterpretation and robustness synthesis",
        file_path=dst,
        what_it_shows="Manuscript-stage synthesis of how biological reinterpretation changed plan ordering and how stable those changes remained under repeat analyses.",
        caption="Phase 31 consolidated the early biological risk-analysis workflow into a publication-oriented synthesis, emphasizing that the most important signal was plan reinterpretation rather than physical-plan optimization.",
    )

    # Figure 6: Phase 32
    phase32_stats = plot_phase32_montage(
        root / "runs" / "phase32_site_specific_template_phantoms" / "phase32_case_manifest.csv",
        fig_dir / "figure06_phase32_site_specific_phantom_montage.png",
        dpi=args.dpi,
    )
    add_manifest_row(
        manifest_rows,
        figure_id="Figure 6",
        phase="Phase 32",
        title="Site-specific synthetic H&N cohort montage",
        file_path=fig_dir / "figure06_phase32_site_specific_phantom_montage.png",
        what_it_shows="The full 10-case site-specific synthetic cohort, replacing the earlier single-phantom assumption with distinct anatomical sites and vertex-pruned geometries.",
        caption=(
            f"Phase 32 instantiated {int(phase32_stats['n_cases'])} separate site-specific synthetic phantoms spanning sinonasal, oropharyngeal, parapharyngeal, cheek, nodal, and composite bulky head-and-neck geometries. "
            f"The median instantiated GTV was {phase32_stats['median_gtv_cc']:.1f} cc, with {int(phase32_stats['kept_vertices'])} kept vertices across the cohort after anatomy-aware pruning."
        ),
    )

    # Figure 7: Phase 33
    phase33_stats = plot_phase33_physical_cohort(
        root / "runs" / "phase33_phase32_topas_cohort" / "phase33_physical_endpoint_table.csv",
        fig_dir / "figure07_phase33_physical_cohort_summary.png",
        dpi=args.dpi,
    )
    add_manifest_row(
        manifest_rows,
        figure_id="Figure 7",
        phase="Phase 33",
        title="Physical TOPAS cohort summary",
        file_path=fig_dir / "figure07_phase33_physical_cohort_summary.png",
        what_it_shows="Distribution of key physical endpoints across the 10 site-specific cases after TOPAS-based photon lattice delivery was generated for each phantom.",
        caption=(
            f"Phase 33 established the physical anchor for the synthetic cohort. Across {int(phase33_stats['n_cases'])} completed cases, the mean PTV D95 was {phase33_stats['mean_ptv_d95_gy']:.2f} Gy and the median PVDR was {phase33_stats['median_pvdr']:.2f}, while substantial spread remained in spill and OAR-adjacent physical dose."
        ),
    )

    # Figure 8: Phase 34
    phase34_stats = plot_phase34_biology_reinterpretation(
        root / "runs" / "phase33_phase32_topas_cohort" / "phase34_bio_cohort" / "phase34_endpoint_table.csv",
        root / "runs" / "phase33_phase32_topas_cohort" / "phase34_bio_cohort" / "phase34_rank_shift_table.csv",
        fig_dir / "figure08_phase34_biological_reinterpretation.png",
        dpi=args.dpi,
    )
    add_manifest_row(
        manifest_rows,
        figure_id="Figure 8",
        phase="Phase 34",
        title="Biological reinterpretation of the 10-case cohort",
        file_path=fig_dir / "figure08_phase34_biological_reinterpretation.png",
        what_it_shows="How the same physical cohort was re-scored when the bystander model was applied with and without vascular sink uptake, including case-wise rank shifts.",
        caption=(
            f"Phase 34 marked the full cohort-level biological reinterpretation step. Biology changed the final ranking in {int(phase34_stats['changed_rank_cases'])}/{int(phase34_stats['n_cases'])} cases, and cohort-mean spill, parotid, and brainstem burden rose markedly relative to the physical-only view."
        ),
    )

    # Figure 9: Phase 35
    phase35_stats = plot_phase35_uncertainty(
        root / "runs" / "phase35_subset_repeat_uncertainty" / "phase35_sink_delta_noise_table.csv",
        root / "runs" / "phase35_subset_repeat_uncertainty" / "phase35_empirical_rank_repeat_summary.csv",
        fig_dir / "figure09_phase35_uncertainty_robustness.png",
        dpi=args.dpi,
    )
    add_manifest_row(
        manifest_rows,
        figure_id="Figure 9",
        phase="Phase 35",
        title="Repeated-subset uncertainty robustness",
        file_path=fig_dir / "figure09_phase35_uncertainty_robustness.png",
        what_it_shows="Endpoint-level and rank-direction robustness of the sink effect after repeated transport and biology solves on the 6-case uncertainty subset.",
        caption=(
            f"Phase 35 showed that {int(phase35_stats['n_exceed95'])}/{int(phase35_stats['n_rows'])} case-endpoint sink deltas exceeded the combined Monte Carlo plus uptake-sensitivity 95% noise band, even though rank-direction stability was weaker than endpoint-level stability."
        ),
    )

    # Figure 10: Phase 36
    phase36_stats = plot_phase36_falsification(
        root / "runs" / "phase36a_vessel_falsification_cohort" / "phase36a_falsification_endpoint_summary.csv",
        root / "runs" / "phase36b_vessel_falsification_uncertainty" / "phase36b_true_vs_surrogate_noise_table.csv",
        fig_dir / "figure10_phase36_falsification_controls.png",
        dpi=args.dpi,
    )
    add_manifest_row(
        manifest_rows,
        figure_id="Figure 10",
        phase="Phase 36",
        title="First falsification controls for the vascular sink",
        file_path=fig_dir / "figure10_phase36_falsification_controls.png",
        what_it_shows="Comparison of the true anatomical sink against no sink, mirrored sink, and mass-matched uniform washout in both baseline and uncertainty-qualified analyses.",
        caption=(
            "Phase 36 tested whether the sink effect could be explained away by simpler controls. "
            f"The mirrored left-right sink remained essentially null, while the uniform body sink retained a strong repeated-subset signal (fraction above 95% noise band {phase36_stats['uniform_frac']:.2f}), "
            f"highlighting that the sink was not reducible to a trivial mirror artefact."
        ),
    )

    # Figure 11: Phase 37
    phase37_stats = plot_phase37_falsification(
        root / "runs" / "phase37a_vessel_falsification_cohort" / "phase37a_falsification_endpoint_summary.csv",
        root / "runs" / "phase37b_vessel_falsification_uncertainty" / "phase37b_true_vs_surrogate_noise_table.csv",
        root / "runs" / "phase37b_vessel_falsification_uncertainty" / "phase37b_true_vs_surrogate_rank_noise.csv",
        fig_dir / "figure11_phase37_stronger_falsification_controls.png",
        dpi=args.dpi,
    )
    add_manifest_row(
        manifest_rows,
        figure_id="Figure 11",
        phase="Phase 37",
        title="Stronger spatial falsification controls",
        file_path=fig_dir / "figure11_phase37_stronger_falsification_controls.png",
        what_it_shows="Performance of the true anatomical sink against stronger spatial falsifications: AP flip, SI flip, local vessel dropout, randomized displacement, uniform washout, and no-sink control.",
        caption=(
            f"Phase 37 strengthened the novelty claim by showing that {int(phase37_stats['n_exceed95'])}/{int(phase37_stats['n_rows'])} repeated case-endpoint differences remained larger than the combined noise band when the true sink was compared with stronger spatial falsifications. "
            f"Only {int(phase37_stats['n_rank_changes'])} noise-qualified rank changes remained, indicating that the sink behaves mainly as a robust endpoint modulator rather than a broad rank-flipping driver."
        ),
    )

    write_manifest_csv(out_root / "phase25_to_phase37_figure_manifest.csv", manifest_rows)
    write_markdown(out_root / "phase25_to_phase37_figure_captions.md", manifest_rows)

    print("=== PHASE 25-37 PUBLICATION FIGURE ATLAS COMPLETE ===")
    print(f"Output root: {out_root}")
    print(f"Manifest: {out_root / 'phase25_to_phase37_figure_manifest.csv'}")
    print(f"Captions: {out_root / 'phase25_to_phase37_figure_captions.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
