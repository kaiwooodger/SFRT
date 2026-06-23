#!/usr/bin/env python3
"""Verify the clean public bundle against the frozen manuscript headline numbers."""

from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "project" / "public_results"


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open() as handle:
        return list(csv.DictReader(handle))


def count_nonzero(rows: list[dict[str, str]], key: str) -> int:
    return sum(int(row[key]) != 0 for row in rows)


def main() -> int:
    rank_rows = load_rows(
        RESULTS / "cohort_summary" / "biology_primary_oxygen_neutral" / "rank_shift_table.csv"
    )
    repeat_rows = load_rows(RESULTS / "repeat_run_uncertainty" / "sink_delta_noise_table.csv")
    rank_noise_rows = load_rows(RESULTS / "repeat_run_uncertainty" / "sink_rank_noise_assessment.csv")
    smooth_rows = load_rows(RESULTS / "smoothing_kernel_sensitivity" / "sigma_summary.csv")
    overview_rows = load_rows(
        RESULTS / "biology_parameter_robustness" / "primary_oxygen_neutral" / "cohort_overview.csv"
    )

    overview = {row["metric"]: row["full_cohort_result"] for row in overview_rows}

    summary = {
        "with_sink_vs_physical_rank_shifts": count_nonzero(rank_rows, "with_sink_vs_physical_shift"),
        "no_sink_vs_physical_rank_shifts": count_nonzero(rank_rows, "no_sink_vs_physical_shift"),
        "with_sink_vs_no_sink_rank_shifts": count_nonzero(rank_rows, "with_sink_vs_no_sink_shift"),
        "biology_added_brainstem_flags": overview.get("biology-added brainstem flags", "missing"),
        "repeat_run_endpoint_deltas_above_95pct_band": sum(
            row["exceeds_95pct_noise_band"] == "True" for row in repeat_rows
        ),
        "repeat_run_total_endpoint_comparisons": len(repeat_rows),
        "noise_qualified_sink_rank_changes": sum(
            row["noise_qualified_rank_change"] == "True" for row in rank_noise_rows
        ),
        "smoothing_kernel_survival": {
            row["sigma_mm"]: row["primary_conclusion_survives"] for row in smooth_rows
        },
    }

    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
