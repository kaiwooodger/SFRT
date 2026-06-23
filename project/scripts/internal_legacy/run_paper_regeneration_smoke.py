#!/usr/bin/env python3
"""Portable paper-regeneration smoke test for the PMB revision freeze.

This script is intentionally lighter than a full transport rerun. It verifies
that a clean or pseudo-clean checkout can regenerate the reviewer-facing paper
artifacts from the checked-in/public-results package without depending on local
hidden run trees.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from build_revision_guardrail_tables import main as build_guardrails_main
from bystander_multispecies_pde_solver import calculate_effective_dose, calculate_phase7_survival
from run_phase26_vascular_sink_ablation import PRIMARY_ENDPOINTS, endpoint_z_scores


REPO_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = REPO_ROOT / "project"
MANUSCRIPT_ROOT = REPO_ROOT / "manuscript"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-root",
        type=Path,
        default=PROJECT_ROOT / "public_results" / "paper_regeneration_smoke",
    )
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def render_figure_r2(dst: Path, *, dpi: int) -> None:
    endpoint_csv = (
        PROJECT_ROOT
        / "public_results"
        / "phase33_34_cohort"
        / "phase34_bio_cohort_primary_oxygen_neutral"
        / "phase34_endpoint_table.csv"
    )
    rank_csv = (
        PROJECT_ROOT
        / "public_results"
        / "phase33_34_cohort"
        / "phase34_bio_cohort_primary_oxygen_neutral"
        / "phase34_rank_shift_table.csv"
    )
    endpoint_df = pd.read_csv(endpoint_csv)
    rank_df = pd.read_csv(rank_csv)

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

    dst.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(dst, dpi=dpi, bbox_inches="tight", pad_inches=0.03, facecolor="white")
    plt.close(fig)


def run_rank_audit() -> dict[str, float]:
    rows = load_csv(
        PROJECT_ROOT
        / "public_results"
        / "phase33_34_cohort"
        / "phase34_bio_cohort_primary_oxygen_neutral"
        / "phase34_endpoint_table.csv"
    )
    by_mode: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        by_mode.setdefault(row["mode"], []).append(row)

    max_score_diff = 0.0
    max_rank_diff = 0
    for mode_rows in by_mode.values():
        risk_scores = np.zeros(len(mode_rows), dtype=float)
        for ep in PRIMARY_ENDPOINTS:
            values = [float(r[ep.key]) for r in mode_rows]
            z = endpoint_z_scores(values, higher_is_better=ep.higher_is_better)
            risk_scores += z
        ranks = np.argsort(np.argsort(risk_scores)) + 1
        for i, row in enumerate(mode_rows):
            max_score_diff = max(max_score_diff, abs(float(row["risk_score"]) - float(risk_scores[i])))
            max_rank_diff = max(max_rank_diff, abs(int(row["rank"]) - int(ranks[i])))
    return {"max_score_diff": max_score_diff, "max_rank_diff": max_rank_diff}


def run_effective_dose_sanity() -> dict[str, float | bool]:
    alpha = 0.03
    beta = 0.003
    scaling = 0.0029365813
    dose = np.linspace(0.0, 20.0, 201, dtype=np.float32)
    lq = np.exp(-alpha * dose - beta * dose**2).astype(np.float32)

    haz0 = np.zeros_like(dose)
    surv0 = calculate_phase7_survival(lq, haz0, dose, (2.0, 2.0, 2.0), scaling, weight_immune=0.0, verbose=False)
    deff0 = calculate_effective_dose(surv0, alpha=alpha, beta=beta)

    haz_levels = [0.0, 1.0, 2.0, 4.0]
    monotonic = True
    for d in [1.0, 5.0, 10.0]:
        idx = int(round(d / 0.1))
        vals = []
        for h in haz_levels:
            surv = calculate_phase7_survival(
                lq,
                np.full_like(dose, h),
                dose,
                (2.0, 2.0, 2.0),
                scaling,
                weight_immune=0.0,
                verbose=False,
            )
            vals.append(float(calculate_effective_dose(surv, alpha=alpha, beta=beta)[idx]))
        monotonic &= all(vals[i] <= vals[i + 1] for i in range(len(vals) - 1))

    return {
        "max_abs_deff_minus_dphys_when_H0": float(np.max(np.abs(deff0 - dose))),
        "monotonic_deff_with_increasing_H": bool(monotonic),
    }


def assert_headline_numbers() -> dict[str, str]:
    primary = {
        row["metric"]: row["full_cohort_result"]
        for row in load_csv(
            PROJECT_ROOT / "public_results" / "phase38_bio_parameter_robustness_primary_oxygen_neutral" / "phase38_cohort_overview.csv"
        )
    }
    sink_noise = load_csv(
        PROJECT_ROOT / "public_results" / "revision_checks_20260616" / "step02_phase35_fullcohort" / "phase35_sink_delta_noise_table.csv"
    )
    rank_noise = load_csv(
        PROJECT_ROOT / "public_results" / "revision_checks_20260616" / "step02_phase35_fullcohort" / "phase35_sink_rank_noise_assessment.csv"
    )
    sigma_rows = {
        float(row["sigma_mm"]): row
        for row in load_csv(
            PROJECT_ROOT / "public_results" / "revision_checks_20260617" / "step10_smoothing_kernel_sensitivity" / "phase40_sigma_summary.csv"
        )
    }

    hits = sum(row["exceeds_95pct_noise_band"] == "True" for row in sink_noise)
    rank_hits = sum(row["noise_qualified_rank_change"] == "True" for row in rank_noise)

    checks = {
        "primary_with_sink_vs_physical": primary["with_sink vs physical rank shifts"],
        "primary_no_sink_vs_physical": primary["no_sink vs physical rank shifts"],
        "with_sink_vs_no_sink": primary["with_sink vs no-sink rank shifts"],
        "primary_brainstem_flags": primary["biology-added brainstem flags"],
        "endpoint_deltas_above_band": f"{hits}/{len(sink_noise)}",
        "noise_qualified_rank_changes": f"{rank_hits}/{len(rank_noise)}",
        "stable_smoothing_range": ",".join(str(int(v)) for v, row in sigma_rows.items() if row["primary_conclusion_survives"] == "True"),
        "unstable_smoothing_range": ",".join(str(int(v)) for v, row in sigma_rows.items() if row["primary_conclusion_survives"] != "True"),
    }
    expected = {
        "primary_with_sink_vs_physical": "6/10",
        "primary_no_sink_vs_physical": "5/10",
        "with_sink_vs_no_sink": "2/10",
        "primary_brainstem_flags": "2/10",
        "endpoint_deltas_above_band": "65/70",
        "noise_qualified_rank_changes": "0/10",
        "stable_smoothing_range": "4,6",
        "unstable_smoothing_range": "2",
    }
    mismatches = {key: (checks[key], expected[key]) for key in expected if checks[key] != expected[key]}
    if mismatches:
        raise RuntimeError(f"Headline number mismatch: {mismatches}")
    return checks


def main() -> int:
    args = parse_args()
    out_root = args.out_root.resolve()
    fig_out = out_root / "figures"
    out_root.mkdir(parents=True, exist_ok=True)
    fig_out.mkdir(parents=True, exist_ok=True)

    build_guardrails_main()

    r2_generated = fig_out / "figureR2_biological_reinterpretation.png"
    render_figure_r2(r2_generated, dpi=int(args.dpi))
    r2_bundle = PROJECT_ROOT / "public_results" / "phase37_results_overleaf_bundle" / "figures" / "figureR2_biological_reinterpretation.png"
    r2_match = sha256(r2_generated) == sha256(r2_bundle)
    if not r2_match:
        raise RuntimeError("Regenerated figureR2 did not match the staged manuscript bundle figure.")

    rs1_source = PROJECT_ROOT / "public_results" / "phase38_bio_parameter_robustness_primary_oxygen_neutral" / "figure_phase38_bio_parameter_robustness.png"
    rs1_generated = fig_out / "figureRS1_bio_parameter_robustness.png"
    shutil.copy2(rs1_source, rs1_generated)
    rs1_bundle = PROJECT_ROOT / "public_results" / "phase37_results_overleaf_bundle" / "figures" / "figureRS1_bio_parameter_robustness.png"
    rs1_match = sha256(rs1_generated) == sha256(rs1_bundle)
    if not rs1_match:
        raise RuntimeError("Primary robustness figure did not match the staged manuscript bundle figureRS1.")

    rank_audit = run_rank_audit()
    if rank_audit["max_score_diff"] != 0.0 or rank_audit["max_rank_diff"] != 0:
        raise RuntimeError(f"Rank audit failed: {rank_audit}")

    deff = run_effective_dose_sanity()
    if deff["max_abs_deff_minus_dphys_when_H0"] > 1e-4 or not deff["monotonic_deff_with_increasing_H"]:
        raise RuntimeError(f"Effective-dose sanity failed: {deff}")

    headline_checks = assert_headline_numbers()

    summary = {
        "status": "paper-regeneration passed; full TOPAS rerun not included",
        "figureR2_match": r2_match,
        "figureRS1_match": rs1_match,
        "rank_audit": rank_audit,
        "effective_dose_sanity": deff,
        "headline_checks": headline_checks,
    }
    (out_root / "paper_regeneration_smoke_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (out_root / "paper_regeneration_smoke_summary.md").write_text(
        "# Paper-regeneration smoke test\n\n"
        + "- Status: `paper-regeneration passed; full TOPAS rerun not included`\n"
        + f"- Regenerated `figureR2` matches staged bundle: `{r2_match}`\n"
        + f"- Verified `figureRS1` matches staged bundle: `{rs1_match}`\n"
        + f"- Rank audit: max score diff `{rank_audit['max_score_diff']}`, max rank diff `{rank_audit['max_rank_diff']}`\n"
        + f"- Effective-dose sanity: max |Deff - Dphys| at H=0 is `{deff['max_abs_deff_minus_dphys_when_H0']:.2e}`, monotonic=`{deff['monotonic_deff_with_increasing_H']}`\n"
        + "- Headline checks:\n"
        + "".join(f"  - `{key}` = `{value}`\n" for key, value in headline_checks.items()),
        encoding="utf-8",
    )
    print("=== PAPER-REGENERATION SMOKE TEST PASSED ===")
    print("Output root:", out_root)
    print("Summary:", out_root / "paper_regeneration_smoke_summary.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
