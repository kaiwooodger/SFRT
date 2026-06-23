#!/usr/bin/env python
"""Translate Phase 7 survival outputs into clinically readable effective doses."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List

import numpy as np

from bystander_multispecies_pde_solver import calculate_effective_dose

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
            "Convert Phase 7 grand-unified survival metrics into effective dose outputs "
            "using an inverse LQ mapping."
        )
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=root
        / "runs"
        / "linac_6mv_polyenergetic_clinical_sfrt"
        / "analysis_phase7_grand_unified_sweep"
        / "phase7_grand_unified_summary.json",
        help="Phase 7 grand unified summary JSON.",
    )
    parser.add_argument(
        "--metrics-csv",
        type=Path,
        default=root
        / "runs"
        / "linac_6mv_polyenergetic_clinical_sfrt"
        / "analysis_phase7_grand_unified_sweep"
        / "phase7_grand_unified_metrics.csv",
        help="Phase 7 grand unified metrics CSV.",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=root
        / "runs"
        / "linac_6mv_polyenergetic_clinical_sfrt"
        / "analysis_phase8_effective_dose",
        help="Directory for Phase 8 effective-dose outputs.",
    )
    parser.add_argument("--alpha-eff", type=float, default=0.03, help="Effective-dose alpha parameter in Gy^-1.")
    parser.add_argument("--beta-eff", type=float, default=0.003, help="Effective-dose beta parameter in Gy^-2.")
    parser.add_argument(
        "--depth-label",
        type=str,
        default="d=5cm",
        help="Depth label used for the main pitch-wise clinic-facing comparison.",
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


def plot_valley_effective_dose_comparison(
    rows: List[Dict[str, object]],
    *,
    alpha_eff: float,
    beta_eff: float,
    figure_file: Path,
    dpi: int,
) -> None:
    pitches = [float(row["pitch_mm"]) for row in rows]
    physical = [float(row["valley_dose_gy"]) for row in rows]
    effective = [float(row["valley_effective_dose_gy"]) for row in rows]
    extra = [float(row["valley_effective_dose_minus_physical_gy"]) for row in rows]
    survival = [float(row["valley_survival_total"]) for row in rows]

    fig, ax = plt.subplots(figsize=(9.2, 5.8), constrained_layout=True)
    ax.plot(
        pitches,
        physical,
        color="#111111",
        linestyle="--",
        marker="o",
        linewidth=2.4,
        markersize=7,
        label="Physical valley dose",
    )
    ax.plot(
        pitches,
        effective,
        color="#d62728",
        marker="o",
        linewidth=2.6,
        markersize=7,
        label="Effective valley dose",
    )

    for pitch, deff, delta, sf in zip(pitches, effective, extra, survival):
        ax.annotate(
            f"SF={sf:.3f}\n+{delta:.2f} Gy",
            (pitch, deff),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
            fontsize=9,
            color="#d62728",
        )

    ax.set_xticks(pitches)
    ax.set_xlim(min(pitches) - 2.0, max(pitches) + 2.0)
    ax.set_xlabel("Lattice pitch (mm)")
    ax.set_ylabel("Dose-equivalent in Gy")
    ax.set_title("Figure 7: Phase 8 effective valley dose at 5 cm depth")
    ax.grid(alpha=0.25)
    ax.legend(loc="upper right")
    ax.text(
        0.98,
        0.02,
        f"LQ inversion: alpha={alpha_eff:.3f} Gy^-1, beta={beta_eff:.4f} Gy^-2",
        transform=ax.transAxes,
        fontsize=9,
        va="bottom",
        ha="right",
        bbox={"facecolor": "white", "alpha": 0.88, "edgecolor": "#b5b5b5"},
    )
    fig.savefig(figure_file, dpi=dpi)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    summary = json.loads(args.summary_json.read_text(encoding="utf-8"))
    with args.metrics_csv.open("r", encoding="utf-8", newline="") as handle:
        metrics = list(csv.DictReader(handle))

    detailed_rows: List[Dict[str, object]] = []
    for row in metrics:
        peak_eff = float(
            calculate_effective_dose(
                np.array([float(row["peak_survival_total"])], dtype=np.float32),
                alpha=float(args.alpha_eff),
                beta=float(args.beta_eff),
            )[0]
        )
        valley_eff = float(
            calculate_effective_dose(
                np.array([float(row["valley_survival_total"])], dtype=np.float32),
                alpha=float(args.alpha_eff),
                beta=float(args.beta_eff),
            )[0]
        )
        detailed_rows.append(
            {
                **row,
                "peak_effective_dose_gy": peak_eff,
                "valley_effective_dose_gy": valley_eff,
                "peak_effective_dose_minus_physical_gy": peak_eff - float(row["peak_dose_gy"]),
                "valley_effective_dose_minus_physical_gy": valley_eff - float(row["valley_dose_gy"]),
            }
        )

    valley_rows = [
        row for row in detailed_rows if str(row["depth_label"]) == str(args.depth_label)
    ]
    valley_rows = sorted(valley_rows, key=lambda row: float(row["pitch_mm"]))

    figure_file = args.outdir / "figure7_phase8_valley_effective_dose.png"
    detailed_csv = args.outdir / "phase8_effective_dose_metrics.csv"
    summary_json = args.outdir / "phase8_effective_dose_summary.json"
    summary_md = args.outdir / "phase8_effective_dose_summary.md"

    write_csv(detailed_rows, detailed_csv)
    plot_valley_effective_dose_comparison(
        valley_rows,
        alpha_eff=float(args.alpha_eff),
        beta_eff=float(args.beta_eff),
        figure_file=figure_file,
        dpi=int(args.dpi),
    )

    pitch_summary = {}
    for row in valley_rows:
        pitch_summary[str(int(float(row["pitch_mm"])))] = {
            "sampled_depth_cm": float(row["sampled_depth_cm"]),
            "valley_dose_gy": float(row["valley_dose_gy"]),
            "valley_survival_total": float(row["valley_survival_total"]),
            "valley_effective_dose_gy": float(row["valley_effective_dose_gy"]),
            "valley_effective_dose_minus_physical_gy": float(row["valley_effective_dose_minus_physical_gy"]),
            "systemic_immune_penalty": float(row["systemic_immune_penalty"]),
        }

    out_summary = {
        "input_summary": str(args.summary_json),
        "input_metrics_csv": str(args.metrics_csv),
        "outdir": str(args.outdir),
        "effective_dose_model": {
            "alpha_eff": float(args.alpha_eff),
            "beta_eff": float(args.beta_eff),
            "depth_label": str(args.depth_label),
        },
        "phase7_source": {
            "phase7_calibration_source": str(summary["phase7_calibration_source"]),
            "phase7_scaling_factor": float(summary["phase7_model"]["scaling_factor"]),
        },
        "pitch_summary": pitch_summary,
        "outputs": {
            "figure": str(figure_file),
            "metrics_csv": str(detailed_csv),
        },
    }
    summary_json.write_text(json.dumps(out_summary, indent=2), encoding="utf-8")

    lines = [
        "# Phase 8 Effective Dose Summary",
        "",
        (
            f"- Effective-dose inversion used `alpha={float(args.alpha_eff):.3f} Gy^-1` "
            f"and `beta={float(args.beta_eff):.4f} Gy^-2`."
        ),
        (
            f"- Source biology: Phase 7 scaling factor `"
            f"{float(summary['phase7_model']['scaling_factor']):.9f}` from "
            f"`{Path(str(summary['phase7_calibration_source'])).name}`."
        ),
        "",
        "| Pitch | Depth | Physical Valley Dose (Gy) | Phase 7 Survival | Effective Valley Dose (Gy) | Delta Deff-Physical (Gy) |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in valley_rows:
        lines.append(
            f"| {float(row['pitch_mm']):.0f} mm | {float(row['sampled_depth_cm']):.3f} cm | "
            f"{float(row['valley_dose_gy']):.3f} | "
            f"{float(row['valley_survival_total']):.3f} | "
            f"{float(row['valley_effective_dose_gy']):.3f} | "
            f"{float(row['valley_effective_dose_minus_physical_gy']):.3f} |"
        )
    summary_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Phase 8 effective-dose figure: {figure_file}")
    print(f"Phase 8 effective-dose metrics: {detailed_csv}")
    print(f"Phase 8 effective-dose summary: {summary_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
