#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import sys
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont


ROOT = Path("PROJECT_ROOT")
DESKTOP = Path("/Users/kw/Desktop")
RUNS = ROOT / "runs"

plt.style.use("seaborn-v0_8-whitegrid")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-root",
        type=Path,
        default=DESKTOP / "PMB_SFRT_figures_from_source_clean",
    )
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def wrap(text: str, width: int) -> str:
    return "\n".join(textwrap.wrap(str(text), width=width))


def add_panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        0.01,
        0.99,
        label,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=16,
        fontweight="bold",
        bbox={"facecolor": "white", "edgecolor": "black", "boxstyle": "square,pad=0.28", "linewidth": 1.0},
        zorder=10,
    )


def save_figure(fig: plt.Figure, path: Path, *, dpi: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight", pad_inches=0.03, facecolor="white")
    plt.close(fig)


def trim_white(img: Image.Image, *, pad: int = 8, threshold: int = 248) -> Image.Image:
    arr = np.asarray(img.convert("RGB"))
    keep = np.any(arr < threshold, axis=2)
    ys, xs = np.where(keep)
    if len(xs) == 0 or len(ys) == 0:
        return img
    left = max(0, int(xs.min()) - pad)
    right = min(img.width, int(xs.max()) + pad + 1)
    top = max(0, int(ys.min()) - pad)
    bottom = min(img.height, int(ys.max()) + pad + 1)
    return img.crop((left, top, right, bottom))


def export_pdf(paths: list[Path], pdf_path: Path) -> None:
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    images = [Image.open(path).convert("RGB") for path in paths]
    try:
        first, rest = images[0], images[1:]
        first.save(pdf_path, save_all=True, append_images=rest, resolution=300.0)
    finally:
        for img in images:
            img.close()


def build_workflow(dst: Path, *, dpi: int) -> None:
    fig, ax = plt.subplots(figsize=(13.5, 7.6))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    boxes = {
        "geometry": (0.06, 0.70, 0.21, 0.13, "Synthetic anatomy and\nlattice geometry"),
        "transport": (0.37, 0.70, 0.23, 0.13, "Monte Carlo photon transport\nand physical dose $D(x)$"),
        "source": (0.70, 0.70, 0.23, 0.13, "Dose-driven source terms\nROS-like and cytokine-like"),
        "cal": (0.06, 0.40, 0.21, 0.13, "One-dimensional calibration\nwith locked parameters"),
        "bio": (0.37, 0.40, 0.23, 0.13, "Reaction-diffusion transport\nand temporal hazard"),
        "sink": (0.70, 0.40, 0.23, 0.13, "Anatomy-aware vascular uptake\nand falsification"),
        "reinterpret": (0.23, 0.11, 0.23, 0.13, "Survival and effective-dose\nreinterpretation"),
        "endpoints": (0.57, 0.11, 0.24, 0.13, "Endpoint extraction:\nPVDR, spill, OAR burden,\nand assay-like readouts"),
    }
    for x, y, w, h, text in boxes.values():
        ax.add_patch(
            FancyBboxPatch(
                (x, y),
                w,
                h,
                boxstyle="round,pad=0.012,rounding_size=0.02",
                linewidth=1.4,
                edgecolor="black",
                facecolor="#f7f7f7",
            )
        )
        ax.text(x + w / 2.0, y + h / 2.0, text, ha="center", va="center", fontsize=13)
    arrows = [
        ((0.27, 0.765), (0.37, 0.765)),
        ((0.60, 0.765), (0.70, 0.765)),
        ((0.165, 0.70), (0.165, 0.53)),
        ((0.485, 0.70), (0.485, 0.53)),
        ((0.815, 0.70), (0.815, 0.53)),
        ((0.27, 0.465), (0.37, 0.465)),
        ((0.60, 0.465), (0.70, 0.465)),
        ((0.48, 0.40), (0.34, 0.24)),
        ((0.81, 0.40), (0.69, 0.24)),
        ((0.46, 0.175), (0.57, 0.175)),
    ]
    for start, end in arrows:
        ax.add_patch(
            FancyArrowPatch(
                start,
                end,
                arrowstyle="Simple,head_length=10,head_width=10,tail_width=1.1",
                mutation_scale=1.0,
                color="black",
            )
        )
    save_figure(fig, dst, dpi=dpi)


def build_calibration_transfer(dst: Path, *, dpi: int) -> None:
    half_df = pd.read_csv(
        RUNS
        / "linac_6mv_polyenergetic_clinical_sfrt"
        / "analysis_phase8_geometry_generalization"
        / "half_field_profile.csv"
    )
    stripe_df = pd.read_csv(
        RUNS
        / "linac_6mv_polyenergetic_clinical_sfrt"
        / "analysis_phase8_geometry_generalization"
        / "stripe_validation_profiles.csv"
    )
    holdout_df = pd.read_csv(
        RUNS
        / "linac_6mv_polyenergetic_clinical_sfrt"
        / "analysis_phase9_holdout_3d_lattice"
        / "phase9_holdout_3d_lattice_metrics.csv"
    )

    fig = plt.figure(figsize=(13.5, 9.0), constrained_layout=True)
    outer = fig.add_gridspec(2, 2, height_ratios=[1.0, 0.72], width_ratios=[1.0, 1.0])

    # (a) Half-field calibration
    gs_a = outer[0, 0].subgridspec(2, 1, hspace=0.08)
    ax_a1 = fig.add_subplot(gs_a[0, 0])
    ax_a2 = fig.add_subplot(gs_a[1, 0], sharex=ax_a1)
    add_panel_label(ax_a1, "(a)")

    ax_a1.plot(half_df["x_mm"], half_df["dose_gy"], color="#34495e", linewidth=2.2)
    ax_a1.axvline(0.0, color="#7f8c8d", linestyle="--", linewidth=1.0)
    ax_a1.set_ylabel("Dose (Gy)")
    ax_a1.tick_params(axis="x", labelbottom=False)

    ax_a2.plot(half_df["x_mm"], half_df["survival_lq"], color="#1f77b4", linewidth=1.8)
    ax_a2.plot(half_df["x_mm"], half_df["survival_total"], color="#d62728", linewidth=2.2)
    ax_a2.axvline(0.0, color="#7f8c8d", linestyle="--", linewidth=1.0)
    ax_a2.scatter([2.0, 10.0], [0.70, 0.95], s=20, color="#d62728", zorder=4)
    ax_a2.text(3.2, 0.73, "0.70", fontsize=8, color="#444444")
    ax_a2.text(11.5, 0.96, "0.95", fontsize=8, color="#444444")
    ax_a2.set_xlabel("Distance from field edge x (mm)")
    ax_a2.set_ylabel("Cell survival")
    ax_a2.set_ylim(0, 1.02)

    # (b) Stripe holdout
    gs_b = outer[0, 1].subgridspec(3, 1, hspace=0.10)
    b_axes = [fig.add_subplot(gs_b[i, 0]) for i in range(3)]
    add_panel_label(b_axes[0], "(b)")
    for ax, pitch in zip(b_axes, [20.0, 30.0, 40.0]):
        sub = stripe_df[np.isclose(stripe_df["pitch_mm"], pitch)].copy()
        ax.plot(sub["x_mm"], sub["dose_gy"] / sub["dose_gy"].max(), color="#1f77b4", linewidth=1.8)
        ax.plot(sub["x_mm"], sub["survival_total"], color="#d62728", linewidth=2.0)
        ax.set_ylim(0, 1.05)
        ax.text(0.98, 0.88, f"{int(pitch)} mm", transform=ax.transAxes, ha="right", fontsize=8, color="#444444")
        if ax is not b_axes[-1]:
            ax.tick_params(axis="x", labelbottom=False)
        else:
            ax.set_xlabel("Lateral position x (mm)")
        ax.set_ylabel("Survival")

    # (c) 3D holdout
    ax_c = fig.add_subplot(outer[1, :])
    add_panel_label(ax_c, "(c)")
    ax_c.plot(holdout_df["pitch_mm"], holdout_df["valley_effective_dose_gy"], color="#d62728", marker="o", linewidth=2.4)
    for _, row in holdout_df.iterrows():
        ax_c.text(
            row["pitch_mm"],
            row["valley_effective_dose_gy"] + 0.12,
            f"SF={row['valley_survival_total']:.3f}",
            ha="center",
            fontsize=8,
        )
    ax_c.set_xlabel("Lattice pitch (mm)")
    ax_c.set_ylabel("Effective dose (Gy)")

    save_figure(fig, dst, dpi=dpi)


def build_synthetic_cohort(dst: Path, *, dpi: int) -> None:
    with open(RUNS / "phase32_site_specific_template_phantoms" / "phase32_case_manifest.csv", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    cols = 2
    rows_n = 5
    cell_w = 1280
    cell_h = 420
    label_h = 50
    outer_pad = 28
    gap_x = 24
    gap_y = 18

    canvas_w = outer_pad * 2 + cols * cell_w + (cols - 1) * gap_x
    canvas_h = outer_pad * 2 + rows_n * (cell_h + label_h) + (rows_n - 1) * gap_y
    canvas = Image.new("RGB", (canvas_w, canvas_h), "white")
    font_path = font_manager.findfont(font_manager.FontProperties(family="DejaVu Sans"))
    font = ImageFont.truetype(font_path, 24)
    draw = ImageDraw.Draw(canvas)

    for idx, row in enumerate(rows):
        src = Image.open(row["geometry_figure"]).convert("RGB")
        src = src.crop((0, 90, src.width, src.height))
        src = trim_white(src, pad=4)
        tile = Image.new("RGB", (cell_w, cell_h), "white")
        scaled = src.copy()
        scaled.thumbnail((cell_w, cell_h), Image.Resampling.LANCZOS)
        paste_x = (cell_w - scaled.width) // 2
        paste_y = (cell_h - scaled.height) // 2
        tile.paste(scaled, (paste_x, paste_y))

        row_i = idx // cols
        col_i = idx % cols
        x0 = outer_pad + col_i * (cell_w + gap_x)
        y0 = outer_pad + row_i * (cell_h + label_h + gap_y)
        canvas.paste(tile, (x0, y0))

        label = f"{row['template_id']}  |  {float(row['instantiated_gtv_cc']):.1f} cc  |  {row['kept_vertex_count']}/{row['proposed_vertex_count']} vertices"
        bbox = draw.textbbox((0, 0), label, font=font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]
        label_x = x0 + (cell_w - text_w) // 2
        label_y = y0 + cell_h + (label_h - text_h) // 2 - 2
        draw.text((label_x, label_y), label, font=font, fill="black")

    dst.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(dst, dpi=(dpi, dpi))


def build_cohort_reinterpretation(dst: Path, *, dpi: int) -> None:
    endpoint_df = pd.read_csv(RUNS / "phase33_phase32_topas_cohort" / "phase34_bio_cohort" / "phase34_endpoint_table.csv")
    rank_df = pd.read_csv(RUNS / "phase33_phase32_topas_cohort" / "phase34_bio_cohort" / "phase34_rank_shift_table.csv")
    metric_info = [
        ("pvdr", "PVDR"),
        ("spill_shell_0_5_mean", "Peri-GTV 0-5"),
        ("brainstem_d2", "Brainstem D2"),
        ("parotid_r_mean", "Parotid R"),
        ("ptv_d95", "PTV D95"),
    ]
    mode_order = ["physical_only", "bystander_no_sink", "bystander_with_sink"]
    mode_labels = ["Physical", "No sink", "With sink"]
    mean_table = endpoint_df.groupby("mode")[[m for m, _ in metric_info]].mean().reindex(mode_order)
    ratio_table = mean_table.divide(mean_table.loc["physical_only"].replace(0.0, np.nan))

    fig = plt.figure(figsize=(13.5, 6.4), constrained_layout=True)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 0.95])
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])
    add_panel_label(ax0, "(a)")
    add_panel_label(ax1, "(b)")

    heat = ax0.imshow(ratio_table.to_numpy(dtype=float), cmap="coolwarm", aspect="auto", vmin=0.75, vmax=3.2)
    ax0.set_xticks(np.arange(len(metric_info)))
    ax0.set_xticklabels([label for _, label in metric_info], rotation=25, ha="right")
    ax0.set_yticks(np.arange(len(mode_labels)))
    ax0.set_yticklabels(mode_labels)
    for i in range(ratio_table.shape[0]):
        for j in range(ratio_table.shape[1]):
            ax0.text(j, i, f"{mean_table.iloc[i, j]:.2f}\n({ratio_table.iloc[i, j]:.2f}x)", ha="center", va="center", fontsize=8)
    # scale bar without legend box
    cbar = fig.colorbar(heat, ax=ax0, fraction=0.046, pad=0.03)
    cbar.set_label("Ratio vs physical")

    x = np.array([0, 1, 2], dtype=float)
    for _, row in rank_df.sort_values("physical_rank").iterrows():
        y = np.array([row["physical_rank"], row["no_sink_rank"], row["with_sink_rank"]], dtype=float)
        ax1.plot(x, y, marker="o", linewidth=1.6, alpha=0.82)
        ax1.text(2.05, y[-1], row["plan_id"], fontsize=8, va="center")
    ax1.set_xticks(x)
    ax1.set_xticklabels(["Physical", "No sink", "With sink"])
    ax1.invert_yaxis()
    ax1.set_ylabel("Risk rank (1 = least risky)")
    ax1.set_xlim(-0.1, 2.38)

    save_figure(fig, dst, dpi=dpi)


def build_uncertainty_sensitivity(dst: Path, *, dpi: int) -> None:
    unc_df = pd.read_csv(
        RUNS / "linac_6mv_polyenergetic_clinical_sfrt" / "analysis_phase10_complex_lattice_plan" / "phase10_uncertainty_band.csv"
    )
    sens_df = pd.read_csv(
        RUNS / "linac_6mv_polyenergetic_clinical_sfrt" / "analysis_phase10_complex_lattice_plan" / "phase10_sensitivity_metrics.csv"
    )
    noise_df = pd.read_csv(RUNS / "phase35_subset_repeat_uncertainty" / "phase35_sink_delta_noise_table.csv")
    rank_df = pd.read_csv(RUNS / "phase35_subset_repeat_uncertainty" / "phase35_sink_rank_noise_assessment.csv")

    fig = plt.figure(figsize=(12.4, 10.2), constrained_layout=True)
    outer = fig.add_gridspec(3, 2, height_ratios=[1.15, 0.82, 0.86], hspace=0.10, wspace=0.16)

    ax_a = fig.add_subplot(outer[0, :])
    add_panel_label(ax_a, "(a)")
    ax_a.fill_between(unc_df["x_cm"], unc_df["lower_survival"], unc_df["upper_survival"], color="#f4b6b6", alpha=0.55)
    ax_a.plot(unc_df["x_cm"], unc_df["baseline_survival"], color="#1f77b4", linewidth=2.2)
    ax_a.set_xlabel("x (cm)")
    ax_a.set_ylabel("Final survival fraction")
    ax_a.set_ylim(0.0, 1.0)
    ax_a.grid(True, axis="y", alpha=0.25)
    ax_a.grid(False, axis="x")

    ax_b1 = fig.add_subplot(outer[1, 0])
    ax_b2 = fig.add_subplot(outer[1, 1])
    add_panel_label(ax_b1, "(b)")
    add_panel_label(ax_b2, "(c)")

    sens_sorted = sens_df.sort_values("max_abs_delta_center_survival", ascending=True)
    y1 = np.arange(len(sens_sorted), dtype=float)
    ax_b1.hlines(y1, 0.0, sens_sorted["max_abs_delta_center_survival"], color="#9ecae1", linewidth=2.0)
    ax_b1.scatter(sens_sorted["max_abs_delta_center_survival"], y1, s=48, color="#1f77b4", zorder=3)
    ax_b1.set_yticks(y1)
    ax_b1.set_yticklabels(sens_sorted["parameter"])
    ax_b1.set_xlabel("Max |delta center survival|")
    ax_b1.set_title("Local sensitivity ranking", fontsize=10)
    ax_b1.grid(True, axis="x", alpha=0.25)
    ax_b1.grid(False, axis="y")

    sens_sorted2 = sens_df.sort_values("max_abs_delta_valley_eud_shift_gy", ascending=True)
    y2 = np.arange(len(sens_sorted2), dtype=float)
    ax_b2.hlines(y2, 0.0, sens_sorted2["max_abs_delta_valley_eud_shift_gy"], color="#f2b6a0", linewidth=2.0)
    ax_b2.scatter(sens_sorted2["max_abs_delta_valley_eud_shift_gy"], y2, s=48, color="#d62728", zorder=3)
    ax_b2.set_yticks(y2)
    ax_b2.set_yticklabels(sens_sorted2["parameter"])
    ax_b2.set_xlabel("Max |delta valley dose lift| (Gy)")
    ax_b2.set_title("Valley sensitivity ranking", fontsize=10)
    ax_b2.grid(True, axis="x", alpha=0.25)
    ax_b2.grid(False, axis="y")

    ax_c1 = fig.add_subplot(outer[2, 0])
    ax_c2 = fig.add_subplot(outer[2, 1])
    add_panel_label(ax_c1, "(d)")
    add_panel_label(ax_c2, "(e)")

    endpoint_summary = noise_df.groupby("label")["exceeds_95pct_noise_band"].mean().sort_values(ascending=True)
    ax_c1.barh(endpoint_summary.index, endpoint_summary.values, color="#3182bd")
    ax_c1.set_xlim(0, 1)
    ax_c1.set_xlabel("Fraction above 95% noise band")
    ax_c1.set_title("Endpoint robustness", fontsize=10)
    ax_c1.grid(True, axis="x", alpha=0.25)
    ax_c1.grid(False, axis="y")

    rank_vals = rank_df["simulated_probability_same_direction_as_nominal"].to_numpy(dtype=float)
    ax_c2.bar(rank_df["plan_id"], rank_vals, color="#16a085")
    ax_c2.axhline(0.8, color="#c0392b", linestyle="--", linewidth=1.0)
    ax_c2.text(len(rank_vals) - 0.35, 0.815, "0.8 heuristic", color="#c0392b", fontsize=8, ha="right", va="bottom")
    ax_c2.set_ylim(0, 1)
    ax_c2.set_ylabel("Probability of preserving\nnominal shift direction")
    ax_c2.set_title("Rank-direction stability", fontsize=10)
    ax_c2.tick_params(axis="x", rotation=18)
    ax_c2.grid(True, axis="y", alpha=0.25)
    ax_c2.grid(False, axis="x")

    save_figure(fig, dst, dpi=dpi)


def build_sink_falsification(dst: Path, *, dpi: int) -> None:
    summary_df = pd.read_csv(RUNS / "phase37a_vessel_falsification_cohort" / "phase37a_falsification_endpoint_summary.csv")
    noise_df = pd.read_csv(RUNS / "phase37b_vessel_falsification_uncertainty" / "phase37b_true_vs_surrogate_noise_table.csv")
    rank_df = pd.read_csv(RUNS / "phase37b_vessel_falsification_uncertainty" / "phase37b_true_vs_surrogate_rank_noise.csv")

    comparator_order = [
        "no_sink",
        "uniform_body_sink_mass_matched",
        "local_dropout_sink_20mm",
        "si_flip_sink",
        "ap_flip_sink",
        "randomized_displacement_sink",
    ]
    label_lookup = {
        row["comparator_id"]: row["comparator_label"].replace("Bystander, ", "").replace(" vessel sink", "")
        for _, row in summary_df.drop_duplicates("comparator_id").iterrows()
    }
    labels = [wrap(label_lookup[c], 16) for c in comparator_order]

    summary_stats = summary_df[summary_df["comparator_id"].isin(comparator_order)].groupby("comparator_id")["ci_excludes_zero"].sum().reindex(comparator_order)
    noise_stats = noise_df[noise_df["comparator_id"].isin(comparator_order)].groupby("comparator_id")["exceeds_95pct_noise_band"].mean().reindex(comparator_order)
    rank_stats = rank_df[rank_df["comparator_id"].isin(comparator_order)].groupby("comparator_id")["noise_qualified_rank_change"].sum().reindex(comparator_order)
    colors = ["#34495e", "#d35400", "#16a085", "#8e44ad", "#2980b9", "#7f8c8d"]

    fig, axes = plt.subplots(1, 3, figsize=(16.8, 5.2), constrained_layout=True)
    add_panel_label(axes[0], "(a)")
    add_panel_label(axes[1], "(b)")
    add_panel_label(axes[2], "(c)")
    axes[0].bar(labels, summary_stats.values, color=colors)
    axes[0].set_ylabel("Endpoint summaries with CI excluding zero")
    axes[0].tick_params(axis="x", rotation=22)
    axes[1].bar(labels, noise_stats.values, color=colors)
    axes[1].set_ylim(0, 1)
    axes[1].set_ylabel("Fraction above 95% noise band")
    axes[1].tick_params(axis="x", rotation=22)
    axes[2].bar(labels, rank_stats.values, color=colors)
    axes[2].set_ylabel("Noise-qualified rank changes")
    axes[2].tick_params(axis="x", rotation=22)
    save_figure(fig, dst, dpi=dpi)


def build_assay_readouts(dst: Path, *, dpi: int) -> None:
    assay_df = pd.read_csv(RUNS / "phase33_phase32_topas_cohort" / "phase34_bio_cohort" / "phase34_assay_proxy_table.csv")
    mean_assay = assay_df.groupby("mode")[
        [
            "mean_gammah2ax_peak",
            "mean_gammah2ax_valley",
            "mean_tunel_peak",
            "mean_tunel_valley",
            "cytokine_global_auc",
            "cytokine_valley_roi_auc",
        ]
    ].mean().reindex(["physical_only", "bystander_no_sink", "bystander_with_sink"])
    fig = plt.figure(figsize=(11.0, 7.4), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, wspace=0.24, hspace=0.22)
    x = np.arange(3, dtype=float)
    mode_labels = ["Physical", "No sink", "Sink"]
    peak_color = "#d35400"
    valley_color = "#1f77b4"
    single_color = "#6c5ce7"

    def line_panel(
        ax: plt.Axes,
        *,
        title: str,
        peak_key: str | None = None,
        valley_key: str | None = None,
        single_key: str | None = None,
        ylabel: str,
        panel_label: str,
    ) -> None:
        add_panel_label(ax, panel_label)
        ax.set_title(title, fontsize=11)
        ax.set_xticks(x)
        ax.set_xticklabels(mode_labels, rotation=18)
        ax.set_ylabel(ylabel)
        ax.grid(True, axis="y", alpha=0.25)
        ax.grid(False, axis="x")
        if single_key is not None:
            vals = mean_assay[single_key].to_numpy(dtype=float)
            ax.plot(x, vals, color=single_color, linewidth=2.3, marker="o", markersize=6)
            ax.text(x[-1] + 0.08, vals[-1], "AUC", color=single_color, fontsize=9, va="center")
            pad = 0.12 * max(vals.max() - vals.min(), 1.0)
            ax.set_ylim(max(0.0, vals.min() - pad), vals.max() + pad)
        else:
            peak_vals = mean_assay[peak_key].to_numpy(dtype=float)
            valley_vals = mean_assay[valley_key].to_numpy(dtype=float)
            ax.plot(x, peak_vals, color=peak_color, linewidth=2.3, marker="o", markersize=6)
            ax.plot(x, valley_vals, color=valley_color, linewidth=2.3, marker="o", markersize=6)
            span = max(float(max(peak_vals.max(), valley_vals.max()) - min(peak_vals.min(), valley_vals.min())), 0.08)
            label_offset = 0.06 * span
            ax.text(x[-1] + 0.08, peak_vals[-1] + label_offset, "Peak", color=peak_color, fontsize=9, va="center")
            ax.text(x[-1] + 0.08, valley_vals[-1] - label_offset, "Valley", color=valley_color, fontsize=9, va="center")
            ax.set_ylim(
                max(0.0, float(min(peak_vals.min(), valley_vals.min()) - 0.15 * span)),
                float(max(peak_vals.max(), valley_vals.max()) + 0.15 * span),
            )
        ax.set_xlim(-0.2, 2.45)

    ax1 = fig.add_subplot(gs[0, 0])
    line_panel(
        ax1,
        title="gammaH2AX",
        peak_key="mean_gammah2ax_peak",
        valley_key="mean_gammah2ax_valley",
        ylabel="Mean proxy signal",
        panel_label="(a)",
    )

    ax2 = fig.add_subplot(gs[0, 1])
    line_panel(
        ax2,
        title="TUNEL",
        peak_key="mean_tunel_peak",
        valley_key="mean_tunel_valley",
        ylabel="Mean positive fraction",
        panel_label="(b)",
    )

    ax3 = fig.add_subplot(gs[1, 0])
    line_panel(
        ax3,
        title="Cytokine Global Burden",
        single_key="cytokine_global_auc",
        ylabel="Global AUC (a.u.)",
        panel_label="(c)",
    )

    ax4 = fig.add_subplot(gs[1, 1])
    line_panel(
        ax4,
        title="Cytokine Valley Burden",
        single_key="cytokine_valley_roi_auc",
        ylabel="Valley ROI AUC (a.u.)",
        panel_label="(d)",
    )

    save_figure(fig, dst, dpi=dpi)


def build_complex_surrogate(dst: Path, *, dpi: int) -> None:
    with open(
        RUNS
        / "linac_6mv_polyenergetic_clinical_sfrt"
        / "analysis_phase10_complex_lattice_plan"
        / "phase10_complex_lattice_summary.json",
        encoding="utf-8",
    ) as f:
        summary = json.load(f)
    vols = np.load(
        RUNS
        / "linac_6mv_polyenergetic_clinical_sfrt"
        / "analysis_phase10_complex_lattice_plan"
        / "phase10_complex_lattice_volumes.npz"
    )
    helper_path = ROOT / "scripts" / "generate_phase10_extended_outputs.py"
    helper_dir = str(helper_path.parent)
    if helper_dir not in sys.path:
        sys.path.insert(0, helper_dir)
    spec = importlib.util.spec_from_file_location("generate_phase10_extended_outputs", helper_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load helper module from {helper_path}")
    helper_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper_mod)
    load_geometry_tensors = helper_mod.load_geometry_tensors

    geom = load_geometry_tensors(summary)
    uptake = geom["uptake_tensor"]
    vessel_mask = uptake[1] > 0.0
    x_cm = vols["x_cm"]
    y_cm = vols["y_cm"]
    z_cm = vols["z_cm"]
    target_idx = int(np.argmin(np.abs(z_cm - np.float32(5.03))))
    center_y = len(y_cm) // 2

    fig = plt.figure(figsize=(15.0, 5.2), constrained_layout=True)
    gs = fig.add_gridspec(1, 3, wspace=0.18)
    axes = [fig.add_subplot(gs[0, i]) for i in range(3)]
    add_panel_label(axes[0], "(a)")
    add_panel_label(axes[1], "(b)")
    add_panel_label(axes[2], "(c)")

    xy_dose = vols["dose_grid"][:, :, target_idx].T
    vessel_xy = vessel_mask[:, :, target_idx].T.astype(float)
    img0 = axes[0].imshow(
        xy_dose,
        origin="lower",
        extent=[float(x_cm[0]), float(x_cm[-1]), float(y_cm[0]), float(y_cm[-1])],
        aspect="equal",
        cmap="magma",
    )
    axes[0].contour(x_cm, y_cm, vessel_xy, levels=[0.5], colors=["#4db6ff"], linewidths=0.9)
    axes[0].set_xlabel("x (cm)")
    axes[0].set_ylabel("y (cm)")
    cb0 = plt.colorbar(img0, ax=axes[0], fraction=0.046, pad=0.02)
    cb0.set_label("Dose (Gy)")

    xy_surv = vols["final_survival"][:, :, target_idx].T
    img1 = axes[1].imshow(
        xy_surv,
        origin="lower",
        extent=[float(x_cm[0]), float(x_cm[-1]), float(y_cm[0]), float(y_cm[-1])],
        aspect="equal",
        cmap="viridis",
        vmin=0.0,
        vmax=1.0,
    )
    axes[1].set_xlabel("x (cm)")
    axes[1].set_ylabel("y (cm)")
    cb1 = plt.colorbar(img1, ax=axes[1], fraction=0.046, pad=0.02)
    cb1.set_label("Survival fraction")

    delta = (vols["deff_grid"] - vols["dose_grid"])[:, :, target_idx].T
    vmax = float(np.percentile(delta, 99.0))
    img2 = axes[2].imshow(
        delta,
        origin="lower",
        extent=[float(x_cm[0]), float(x_cm[-1]), float(y_cm[0]), float(y_cm[-1])],
        aspect="equal",
        cmap="inferno",
        vmin=0.0,
        vmax=vmax,
    )
    axes[2].set_xlabel("x (cm)")
    axes[2].set_ylabel("y (cm)")
    cb2 = plt.colorbar(img2, ax=axes[2], fraction=0.046, pad=0.02)
    cb2.set_label(r"$D_{\mathrm{eff}} - D_{\mathrm{phys}}$ (Gy)")

    save_figure(fig, dst, dpi=dpi)


def build_mc_stochasticity(dst: Path, *, dpi: int) -> None:
    vols = np.load(
        RUNS
        / "linac_6mv_polyenergetic_clinical_sfrt_phase12_mc"
        / "analysis_phase12_mc_coupled_plan"
        / "phase12_mc_coupled_volumes.npz"
    )
    x_cm = vols["x_cm"]
    y_cm = vols["y_cm"]
    z_cm = vols["z_cm"]
    target_idx = int(np.argmin(np.abs(z_cm - np.float32(5.03))))
    center_y = len(y_cm) // 2

    fig = plt.figure(figsize=(12.5, 7.0), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, wspace=0.16, hspace=0.18)
    ax_a1 = fig.add_subplot(gs[0, 0])
    ax_a2 = fig.add_subplot(gs[0, 1])
    ax_b1 = fig.add_subplot(gs[1, 0])
    ax_b2 = fig.add_subplot(gs[1, 1])
    add_panel_label(ax_a1, "(a)")
    add_panel_label(ax_b1, "(b)")

    sigma_xy = vols["sigma_norm_grid"][:, :, target_idx].T
    sigma_xz = vols["sigma_norm_grid"][:, center_y, :].T
    img0 = ax_a1.imshow(
        sigma_xy,
        origin="lower",
        extent=[float(x_cm[0]), float(x_cm[-1]), float(y_cm[0]), float(y_cm[-1])],
        aspect="equal",
        cmap="plasma",
        vmin=0.0,
        vmax=1.0,
    )
    ax_a1.set_xlabel("x (cm)")
    ax_a1.set_ylabel("y (cm)")
    ax_a2.imshow(
        sigma_xz,
        origin="lower",
        extent=[float(x_cm[0]), float(x_cm[-1]), float(z_cm[0]), float(z_cm[-1])],
        aspect="auto",
        cmap="plasma",
        vmin=0.0,
        vmax=1.0,
    )
    ax_a2.axhline(float(z_cm[target_idx]), color="white", linestyle="--", linewidth=0.8)
    ax_a2.set_xlabel("x (cm)")
    ax_a2.set_ylabel("z (cm)")
    cb0 = plt.colorbar(img0, ax=[ax_a1, ax_a2], fraction=0.03, pad=0.02)
    cb0.set_label("Normalized sigma")

    delta_xy = vols["delta_deff"][:, :, target_idx].T
    delta_xz = vols["delta_deff"][:, center_y, :].T
    vmax = float(np.percentile(vols["delta_deff"], 99.0))
    img1 = ax_b1.imshow(
        delta_xy,
        origin="lower",
        extent=[float(x_cm[0]), float(x_cm[-1]), float(y_cm[0]), float(y_cm[-1])],
        aspect="equal",
        cmap="magma",
        vmin=0.0,
        vmax=vmax,
    )
    ax_b1.set_xlabel("x (cm)")
    ax_b1.set_ylabel("y (cm)")
    ax_b2.imshow(
        delta_xz,
        origin="lower",
        extent=[float(x_cm[0]), float(x_cm[-1]), float(z_cm[0]), float(z_cm[-1])],
        aspect="auto",
        cmap="magma",
        vmin=0.0,
        vmax=vmax,
    )
    ax_b2.axhline(float(z_cm[target_idx]), color="white", linestyle="--", linewidth=0.8)
    ax_b2.set_xlabel("x (cm)")
    ax_b2.set_ylabel("z (cm)")
    cb1 = plt.colorbar(img1, ax=[ax_b1, ax_b2], fraction=0.03, pad=0.02)
    cb1.set_label(r"$\Delta D_{\mathrm{eff}}$ (Gy)")

    save_figure(fig, dst, dpi=dpi)


def main() -> int:
    args = parse_args()
    out_root = args.out_root.resolve()
    fig_dir = out_root / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    order = [
        fig_dir / "fig01_workflow.png",
        fig_dir / "fig02_calibration_transfer.png",
        fig_dir / "fig03_synthetic_cohort.png",
        fig_dir / "fig04_cohort_reinterpretation.png",
        fig_dir / "fig05a_uncertainty_sensitivity.png",
        fig_dir / "fig05b_sink_falsification.png",
        fig_dir / "fig06_assay_readouts.png",
        fig_dir / "supp_complex_surrogate.png",
        fig_dir / "supp_mc_stochasticity.png",
    ]

    build_workflow(order[0], dpi=args.dpi)
    build_calibration_transfer(order[1], dpi=args.dpi)
    build_synthetic_cohort(order[2], dpi=args.dpi)
    build_cohort_reinterpretation(order[3], dpi=args.dpi)
    build_uncertainty_sensitivity(order[4], dpi=args.dpi)
    build_sink_falsification(order[5], dpi=args.dpi)
    build_assay_readouts(order[6], dpi=args.dpi)
    build_complex_surrogate(order[7], dpi=args.dpi)
    build_mc_stochasticity(order[8], dpi=args.dpi)

    export_pdf(order, out_root / "PMB_SFRT_publishable_figures.pdf")
    print(f"Output root: {out_root}")
    print(f"Figures: {fig_dir}")
    print(f"Combined PDF: {out_root / 'PMB_SFRT_publishable_figures.pdf'}")
    return 0


if __name__ == "__main__":
    exit_code = int(main())
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(exit_code)
