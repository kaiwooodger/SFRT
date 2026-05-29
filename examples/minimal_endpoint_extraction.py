#!/usr/bin/env python3
"""Print the case-wise rank reinterpretation table used in the PMB results section."""

from __future__ import annotations

import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TABLE = ROOT / "project" / "public_results" / "phase33_34_cohort" / "phase34_rank_shift_table.csv"


def main() -> None:
    with TABLE.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    print("Minimal endpoint extraction demo")
    print(f"Source table: {TABLE}")
    print("plan_id, physical_rank, no_sink_rank, with_sink_rank, with_sink_vs_physical_shift")
    for row in rows:
        print(
            f"{row['plan_id']}, {row['physical_rank']}, {row['no_sink_rank']}, "
            f"{row['with_sink_rank']}, {row['with_sink_vs_physical_shift']}"
        )


if __name__ == "__main__":
    main()
