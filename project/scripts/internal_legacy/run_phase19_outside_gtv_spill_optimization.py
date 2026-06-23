#!/usr/bin/env python3
"""Phase 19: optimize lattice courses by minimizing biology-aware outside-GTV spill.

This phase keeps the clinically constrained GTV-core lattice placement family but
changes the course objective so that:
- minimum target coverage is enforced as a floor rather than the main reward
- the primary optimization target is biology-aware burden outside the GTV
- peri-GTV shells, PTV valleys, and OAR-adjacent outside-GTV regions are penalized
"""

from __future__ import annotations

import sys

import run_phase17_fraction_aware_bio_optimization as phase17


DEFAULT_ARGS = [
    "--phase-number",
    "19",
    "--phase-description",
    "Outside-GTV spill-aware fraction-aware SFRT optimization using peri-GTV biological burden as the primary objective.",
    "--course-strategy",
    "clinical_gtv_core",
    "--objective-mode",
    "outside_gtv_spill",
    "--spot-radius-mm",
    "7.5",
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
