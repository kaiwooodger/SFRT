#!/usr/bin/env python3
"""Print a minimal physical-vs-biological comparison for one cohort case."""

from __future__ import annotations

import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TABLE = ROOT / "project" / "public_results" / "phase33_34_cohort" / "phase34_endpoint_table.csv"
CASE_ID = "case03"
KEYS = [
    ("pvdr", "PVDR"),
    ("spill_shell_0_5_mean", "Peri-GTV 0-5 mm mean"),
    ("brainstem_d2", "Brainstem D2"),
    ("parotid_r_mean", "Parotid R mean"),
]


def main() -> None:
    with TABLE.open("r", encoding="utf-8", newline="") as handle:
        rows = [row for row in csv.DictReader(handle) if row["plan_id"] == CASE_ID]

    by_model = {row["mode"]: row for row in rows}
    print(f"Minimal bioaware demo for {CASE_ID}")
    print(f"Source table: {TABLE}")
    for key, label in KEYS:
        physical = float(by_model["physical_only"][key])
        no_sink = float(by_model["bystander_no_sink"][key])
        with_sink = float(by_model["bystander_with_sink"][key])
        print(f"{label}: physical={physical:.3f}, no_sink={no_sink:.3f}, with_sink={with_sink:.3f}")


if __name__ == "__main__":
    main()
