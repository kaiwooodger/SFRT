#!/usr/bin/env python3
"""Phase 23: robust uncertainty-aware adaptive no-fly optimization."""

from __future__ import annotations

import sys

import run_phase17_fraction_aware_bio_optimization as phase17


DEFAULT_ARGS = [
    "--phase-number",
    "23",
    "--phase-description",
    "Robust uncertainty-aware adaptive SFRT optimization with partially non-relaxing no-fly zones, brainstem guard, and multi-seed multi-history candidate reranking.",
    "--course-strategy",
    "adaptive_nofly_weight_opt",
    "--objective-mode",
    "outside_gtv_spill",
    "--spot-radius-mm",
    "7.5",
    "--protocol-mix-per-family",
    "2",
    "--phase22-enable-weight-optimization",
    "--phase22-no-fly-distance-mm",
    "15.0",
    "--phase22-no-fly-physical-threshold-gy",
    "100.0",
    "--phase22-no-fly-effective-threshold-gy",
    "32.0",
    "--phase22-no-fly-oar-adjacent-threshold-gy",
    "18.0",
    "--phase22-no-fly-valley-threshold-gy",
    "32.0",
    "--phase22-brainstem-trigger-fraction",
    "0.65",
    "--phase22-brainstem-hard-avoidance-mm",
    "28.0",
    "--adaptive-veto-relax-mode",
    "partial",
    "--adaptive-veto-relax-fraction",
    "0.50",
    "--robust-top-k-candidates",
    "3",
    "--robust-seeds",
    "33,47,61",
    "--robust-histories",
    "100000,150000",
    "--robust-z-score",
    "1.0",
    "--robust-min-feasible-fraction",
    "0.67",
]


def main() -> int:
    argv = [sys.argv[0], *DEFAULT_ARGS, *sys.argv[1:]]
    original_argv = sys.argv
    sys.argv = argv
    try:
        return int(phase17.main())
    finally:
        sys.argv = original_argv


if __name__ == "__main__":
    raise SystemExit(main())
