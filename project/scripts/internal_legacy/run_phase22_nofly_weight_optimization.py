#!/usr/bin/env python3
"""Phase 22: hard no-fly + delivery-weight optimization + brainstem guard."""

from __future__ import annotations

import sys

import run_phase17_fraction_aware_bio_optimization as phase17


DEFAULT_ARGS = [
    "--phase-number",
    "22",
    "--phase-description",
    "Adaptive SFRT optimization with hard spill/hotspot no-fly zones, delivery-weight search, and brainstem-specific adaptive avoidance.",
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
