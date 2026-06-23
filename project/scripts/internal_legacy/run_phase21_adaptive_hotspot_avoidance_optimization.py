#!/usr/bin/env python3
"""Phase 21: adaptive hotspot-avoidance biology-guided course optimization.

This phase keeps the outside-GTV spill objective but converts cumulative hotspot
burden into an adaptive steering signal for the next fraction:
- repeated spot positions are discouraged using a spot-memory radius
- accumulated physical/effective burden regions become hotspot-memory regions
- the next-fraction lattice is rerouted away from those regions
- cumulative body Dmax is treated as a soft optimization penalty rather than a
  hard accept/reject gate
"""

from __future__ import annotations

import sys

import run_phase17_fraction_aware_bio_optimization as phase17


DEFAULT_ARGS = [
    "--phase-number",
    "21",
    "--phase-description",
    "Adaptive hotspot-avoidance biology-guided SFRT optimization using cumulative burden memory to steer the next fraction.",
    "--course-strategy",
    "adaptive_hotspot_avoidance",
    "--objective-mode",
    "outside_gtv_spill",
    "--spot-radius-mm",
    "7.5",
    "--protocol-mix-per-family",
    "2",
    "--disable-body-hotspot-hard-constraint",
    "--adaptive-spot-memory-radius-mm",
    "18.0",
    "--adaptive-hotspot-avoidance-distance-mm",
    "15.0",
    "--adaptive-hotspot-physical-threshold-gy",
    "120.0",
    "--adaptive-spill-effective-threshold-gy",
    "35.0",
    "--adaptive-oar-adjacent-effective-threshold-gy",
    "20.0",
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
