#!/usr/bin/env python3
"""Print a compact summary of true-sink separation from selected comparators."""

from __future__ import annotations

import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TABLE = ROOT / "project" / "public_results" / "phase37a_vessel_falsification_cohort" / "phase37a_falsification_endpoint_summary.csv"
COMPARATORS = {
    "uniform_body_sink_mass_matched",
    "local_dropout_sink_20mm",
    "ap_flip_sink",
}


def main() -> None:
    with TABLE.open("r", encoding="utf-8", newline="") as handle:
        rows = [row for row in csv.DictReader(handle) if row["comparator_id"] in COMPARATORS]

    print("Minimal vascular-sink demo")
    print(f"Source table: {TABLE}")
    for row in rows:
        if row["metric"] not in {"pvdr", "brainstem_d2", "parotid_r_mean"}:
            continue
        print(
            f"{row['comparator_id']} | {row['label']} | "
            f"mean true-comparator={float(row['mean_true_minus_comparator']):.3f} {row['units']} | "
            f"CI excludes zero={row['ci_excludes_zero']}"
        )


if __name__ == "__main__":
    main()
