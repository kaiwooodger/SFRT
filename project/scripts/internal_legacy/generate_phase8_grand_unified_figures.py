#!/usr/bin/env python
"""Launch the tuned grand-unified branch as an explicit Phase 8 run."""

from __future__ import annotations

import sys
from pathlib import Path

import generate_phase7_grand_unified_figures as shared_generator


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    default_args = [
        "--phase-label",
        "Phase 8",
        "--summary-stem",
        "phase8_grand_unified",
        "--outdir",
        str(
            root
            / "runs"
            / "linac_6mv_polyenergetic_clinical_sfrt"
            / "analysis_phase8_grand_unified_sweep_tuned_tail"
        ),
        "--cytokine-diffusion-coeff",
        "1.2",
        "--cytokine-decay-coeff",
        "0.001",
        "--pde-steps",
        "400",
        "--pde-dt",
        "0.12",
        "--phase7-scaling-factor",
        "0.0029365812996595296",
    ]
    sys.argv = [sys.argv[0], *default_args, *sys.argv[1:]]
    return shared_generator.main()


if __name__ == "__main__":
    raise SystemExit(main())
