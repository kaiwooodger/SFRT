#!/usr/bin/env python3
"""Phase 24: joint-horizon, diversity-preserving adaptive SFRT optimization."""

from __future__ import annotations

import sys

import run_phase17_fraction_aware_bio_optimization as phase17


DEFAULT_ARGS = [
    "--phase-number",
    "24",
    "--phase-description",
    "Joint-horizon adaptive SFRT optimization with reserved future-space bias, mixed 2v/3v/4v templates, diversity reward, and softened body-hotspot hard gating.",
    "--course-strategy",
    "phase24_joint_horizon_diversity",
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
    "--phase24-min-vertices",
    "2",
    "--phase24-max-vertices",
    "4",
    "--phase24-template-pitch-scales",
    "0.85",
    "1.00",
    "1.15",
    "--phase24-soft-min-spacing-mm",
    "15.0",
    "--phase24-diversity-weight",
    "8.0",
    "--phase24-lookahead-weight",
    "18.0",
    "--phase24-future-center-count",
    "6",
    "--phase24-future-candidate-target",
    "4",
    "--disable-body-hotspot-hard-constraint",
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
