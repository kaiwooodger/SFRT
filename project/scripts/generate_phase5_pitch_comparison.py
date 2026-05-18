#!/usr/bin/env python
"""Generate a dedicated Phase 5 pitch-comparison figure.

This figure compares valley survival at 5 cm depth against the systemic
immune scalar across lattice pitch for the latest Phase 5 branch.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception as exc:  # pragma: no cover
    raise RuntimeError("matplotlib is required for figure generation") from exc


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description=(
            "Load a Phase 5 Phase-2 summary and generate a dedicated pitch-wise "
            "comparison of valley survival and systemic immune penalty."
        )
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=root
        / "runs"
        / "linac_6mv_polyenergetic_clinical_sfrt"
        / "analysis_phase2_multispecies_true_valley_phase5_rightpeak_depth50"
        / "phase2_multispecies_summary.json",
        help="Phase 5 summary JSON from the heavy run.",
    )
    parser.add_argument(
        "--depth-label",
        type=str,
        default="d=5cm",
        help="Depth label to extract from metrics_by_depth.",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=root
        / "runs"
        / "linac_6mv_polyenergetic_clinical_sfrt"
        / "analysis_phase5_pitch_comparison",
        help="Directory for the dedicated Phase 5 comparison outputs.",
    )
    parser.add_argument("--dpi", type=int, default=250, help="Figure DPI.")
    return parser.parse_args()


def write_csv(rows: List[Dict[str, object]], out_file: Path) -> None:
    if not rows:
        raise ValueError("No rows supplied for CSV output.")
    with out_file.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_phase5_pitch_comparison(
    rows: List[Dict[str, object]],
    *,
    phase5_model: Dict[str, object],
    emission_model: Dict[str, object],
    figure_file: Path,
    dpi: int,
) -> None:
    pitches = [float(row["pitch_mm"]) for row in rows]
    survival_lq = [float(row["valley_survival_lq"]) for row in rows]
    survival_phase5 = [float(row["valley_survival_phase5"]) for row in rows]
    immune = [float(row["systemic_immune_penalty"]) for row in rows]
    icd = [float(row["icd_volume_cm3"]) for row in rows]

    fig, ax1 = plt.subplots(figsize=(9.4, 6.0), constrained_layout=True)
    ax2 = ax1.twinx()

    ax1.plot(
        pitches,
        survival_lq,
        color="#111111",
        linestyle="--",
        marker="o",
        linewidth=2.4,
        markersize=7,
        label="Valley survival (LQ)",
    )
    ax1.plot(
        pitches,
        survival_phase5,
        color="#d62728",
        marker="o",
        linewidth=2.6,
        markersize=7,
        label="Valley survival (Phase 5)",
    )
    ax2.plot(
        pitches,
        immune,
        color="#1f77b4",
        marker="s",
        linewidth=2.4,
        markersize=6,
        label="Systemic immune penalty",
    )

    for x_value, immune_value, icd_value in zip(pitches, immune, icd):
        ax2.annotate(
            f"ICD={icd_value:.2f} cm^3",
            (x_value, immune_value),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
            fontsize=9,
            color="#1f77b4",
        )

    ax1.set_xlim(min(pitches) - 2.0, max(pitches) + 2.0)
    ax1.set_xticks(pitches)
    ax1.set_xlabel("Lattice pitch (mm)")
    ax1.set_ylabel("Valley survival at 5 cm")
    ax1.set_ylim(0.0, 1.02)
    ax2.set_ylabel("Systemic immune penalty, P_immune")
    ax2.set_ylim(0.0, max(0.6, max(immune) * 1.2))

    ax1.grid(alpha=0.25)
    ax1.set_title("Figure 10: Phase 5 pitch comparison at 5 cm depth")

    weight_box = phase5_model["channel_weights"]
    text = (
        f"Weights: ROS={float(weight_box['ros']):.2f}, "
        f"Cyto={float(weight_box['cytokine']):.2f}, "
        f"Immune={float(weight_box['immune']):.2f}\n"
        f"ICD threshold={float(phase5_model['icd_threshold_gy']):.1f} Gy, "
        f"V_half={float(phase5_model['immune_half_volume_cm3']):.1f} cm^3\n"
        f"Hypoxic depth={phase5_model.get('hypoxic_depth_note', 'see summary')}\n"
        f"Emission={emission_model['type']}"
    )
    ax1.text(
        0.98,
        0.02,
        text,
        transform=ax1.transAxes,
        fontsize=9,
        va="bottom",
        ha="right",
        bbox={"facecolor": "white", "alpha": 0.88, "edgecolor": "#b5b5b5"},
    )

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="lower left")

    fig.savefig(figure_file, dpi=dpi)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    summary = json.loads(args.summary_json.read_text(encoding="utf-8"))
    rows: List[Dict[str, object]] = []
    for pitch_key, pitch_summary in sorted(
        summary["generated_pitches"].items(),
        key=lambda item: float(item[1]["pitch_mm"]),
    ):
        depth_metrics = pitch_summary["metrics_by_depth"][str(args.depth_label)]
        rows.append(
            {
                "pitch_mm": float(pitch_summary["pitch_mm"]),
                "sampled_depth_cm": float(depth_metrics["sampled_depth_cm"]),
                "valley_survival_lq": float(depth_metrics["valley_survival_lq"]),
                "valley_survival_phase5": float(depth_metrics["valley_survival_total"]),
                "systemic_immune_penalty": float(pitch_summary["systemic_immune_penalty"]),
                "icd_volume_cm3": float(pitch_summary["icd_volume_cm3"]),
            }
        )

    phase5_model = dict(summary["phase5_model"])
    phase5_model["hypoxic_depth_note"] = (
        "50 mm from surface"
        if summary["emission_model"].get("hypoxic_depth_from_surface_mm") is not None
        else "center-offset mode"
    )

    csv_file = args.outdir / "phase5_pitch_comparison.csv"
    json_file = args.outdir / "phase5_pitch_comparison_summary.json"
    figure_file = args.outdir / "figure10_phase5_pitch_comparison.png"

    write_csv(rows, csv_file)
    plot_phase5_pitch_comparison(
        rows,
        phase5_model=phase5_model,
        emission_model=summary["emission_model"],
        figure_file=figure_file,
        dpi=int(args.dpi),
    )

    out_summary = {
        "input_summary": str(args.summary_json),
        "depth_label": str(args.depth_label),
        "figure": str(figure_file),
        "csv": str(csv_file),
        "phase5_model": phase5_model,
        "rows": rows,
    }
    json_file.write_text(json.dumps(out_summary, indent=2), encoding="utf-8")

    print(f"Phase 5 comparison figure: {figure_file}")
    print(f"Phase 5 comparison CSV: {csv_file}")
    print(f"Phase 5 comparison summary: {json_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
