#!/usr/bin/env python3
"""Phase 18: clinical GTV-core lattice optimization with a safer delivery backbone.

This phase keeps the Phase 17 clinical GTV-core placement rules but changes the
source weighting so that:
- the broad AP base field carries more of the coverage burden
- lateral spot beams are de-emphasized to reduce hotspotting
- superior/posterior vertices receive extra lateral attenuation
"""

from __future__ import annotations

import sys

import run_phase17_fraction_aware_bio_optimization as phase17


DEFAULT_ARGS = [
    "--course-strategy",
    "clinical_gtv_core",
    "--spot-radius-mm",
    "7.5",
    "--base-history-fraction",
    "0.985",
    "--base-margin-mm",
    "12.0",
    "--base-min-ap-radius-mm",
    "70.0",
    "--base-min-lateral-radius-mm",
    "55.0",
    "--spot-ap-weight-scale",
    "1.0",
    "--spot-lateral-weight-scale",
    "0.35",
    "--superior-posterior-lateral-scale",
    "0.30",
    "--superior-threshold-mm",
    "6.0",
    "--posterior-threshold-mm",
    "6.0",
    "--lateral-radius-scale",
    "0.90",
    "--ap-spot-radius-scale",
    "1.05",
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
