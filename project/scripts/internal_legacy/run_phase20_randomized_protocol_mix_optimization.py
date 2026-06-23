#!/usr/bin/env python3
"""Phase 20: protocol-constrained randomized search across multiple lattice families.

This phase keeps the clinical planning rules in place but assembles each fraction's
candidate set by stochastically mixing legal candidates from:
- adaptive mutation
- alternating-pattern complements
- larger-pitch sweeps
- reduced-vertex interlacing
- vascular sink hugging
- clinical GTV-core layouts
"""

from __future__ import annotations

import sys

import run_phase17_fraction_aware_bio_optimization as phase17


DEFAULT_ARGS = [
    "--phase-number",
    "20",
    "--phase-description",
    "Protocol-constrained randomized lattice-family mix using biology-aware outside-GTV spill scoring.",
    "--course-strategy",
    "randomized_protocol_mix",
    "--objective-mode",
    "outside_gtv_spill",
    "--spot-radius-mm",
    "7.5",
    "--protocol-mix-per-family",
    "2",
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
